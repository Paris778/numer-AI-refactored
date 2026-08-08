"""Round-aware Numerai dataset refresh policy — pure logic, no I/O, no numerapi.

``refresh_data.py`` performs all downloads and file I/O; this module only
decides *what* to do, given facts the script has already gathered (round
numbers, era ranges, available versions). Deterministic: same inputs, same
outputs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

__all__ = [
    "CURRENT_DATA_VERSION",
    "STATIC_FILES",
    "LIVE_FRESH_FILES",
    "EXPANDING_FILES",
    "detect_newer_version",
    "needs_live_refresh",
    "build_era_manifest",
    "classify_refresh_plan",
]

# The data version this repo's pipeline targets. Drift-guarded by a test
# asserting equality with configs/first_model.yaml's data.version.
CURRENT_DATA_VERSION = "v5.3"

# Files that change only when Numerai ships a new data version.
STATIC_FILES = ("features.json", "train.parquet", "train_benchmark_models.parquet")

# Files that change with every tournament round.
LIVE_FRESH_FILES = (
    "live.parquet",
    "live_benchmark_models.parquet",
    "live_example_preds.parquet",
    "live_example_preds.csv",
)

# Files that expand weekly as new validation eras are published.
EXPANDING_FILES = (
    "validation.parquet",
    "validation_benchmark_models.parquet",
    "validation_example_preds.parquet",
    "validation_example_preds.csv",
    "meta_model.parquet",
)

_VERSION_RE = re.compile(r"^v(\d+)\.(\d+)$")


def _parse_version(v: str) -> tuple[int, int]:
    """Parse ``v<major>.<minor>`` into integers, or raise ``ValueError``."""
    match = _VERSION_RE.match(v)
    if match is None:
        raise ValueError(
            f"Unrecognized dataset version {v!r}: expected 'v<major>.<minor>' "
            "(e.g. 'v5.3'); patch components are not supported"
        )
    return int(match.group(1)), int(match.group(2))


def detect_newer_version(available: Sequence[str], current: str) -> str | None:
    """Return the numerically-greatest version in ``available`` that strictly
    exceeds ``current``, or ``None``. Malformed entries raise (fail loudly —
    a strange filename in the API listing is a real signal)."""
    current_parsed = _parse_version(current)
    newest: tuple[tuple[int, int], str] | None = None
    for item in available:
        parsed = _parse_version(item)
        if parsed > current_parsed and (newest is None or parsed > newest[0]):
            newest = (parsed, item)
    return newest[1] if newest is not None else None


def needs_live_refresh(
    current_round: int, last_recorded: int | None, live_exists: bool
) -> bool:
    """True when live.parquet must be re-downloaded.

    Reconciles on any mismatch (stale *or* ahead-of-remote marker); the file
    must exist *and* match the current round.
    """
    return not live_exists or last_recorded is None or last_recorded != current_round


def build_era_manifest(
    era_ranges: Mapping[str, tuple[str | None, str | None]],
    round_id: int,
    today: str,
) -> list[dict[str, str | int | None]]:
    """Build refresh-ledger rows for ``numerai_era_data.csv``.

    ``era_ranges`` maps dataset name to ``(min_era, max_era)`` as read from the
    parquet ``era`` column. Live rounds are unlabeled: ``(None, None)`` is
    valid for ``live`` (the script serializes it to ``"X"``); any other dataset
    with an empty range raises — an empty parquet is a real error.
    """
    rows: list[dict[str, str | int | None]] = []
    for dataset in ("train", "validation", "live"):
        start, end = era_ranges[dataset]
        if start is None or end is None:
            if dataset != "live":
                raise ValueError(
                    f"{dataset} parquet has no era range (empty file): {(start, end)!r}"
                )
        rows.append(
            {
                "date": today,
                "dataset": dataset,
                "start_era": start,
                "end_era": end,
                "round_id": round_id if dataset == "live" else None,
            }
        )
    return rows


def classify_refresh_plan(
    round_advanced: bool,
    existing: set[str],
    live_only: bool = False,
) -> dict[str, str]:
    """Per-file refresh decision.

    Returns ``{filename: "refresh" | "ensure" | "skip"}``:
    - ``refresh`` — download now (round advanced, or the file is missing);
    - ``ensure``  — download only if missing (static files);
    - ``skip``    — already present and no trigger (or skipped by ``--live-only``).
    """
    plan: dict[str, str] = {}
    for name in STATIC_FILES:
        plan[name] = "ensure"
    for name in LIVE_FRESH_FILES:
        plan[name] = "refresh" if (round_advanced or name not in existing) else "skip"
    for name in EXPANDING_FILES:
        if live_only:
            plan[name] = "skip"
        else:
            plan[name] = "refresh" if (round_advanced or name not in existing) else "skip"
    return plan
