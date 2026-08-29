"""Cross-family run registry: global comparison + champion pointer only.

Runs live under ``experiments/<slug>/runs/<run_id>/run.json`` (persistence
lives in :mod:`nmr.experiment_store` — run recording is
``experiment_store.record_run_result``); this class iterates families for
comparison and owns the atomic ``champion.json`` pointer at the experiments
root. Champion writes are single-writer — the read-compare-write in
``promote_if_better`` is serialized by an inter-process advisory lock on
``<root>/champion.json.lock`` (design spec §9; the invariant is documented AND
enforced since the 2026-08-26 review).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nmr import paths
from nmr._atomicio import atomic_write_text
from nmr._filelock import file_lock

logger = logging.getLogger("nmr.registry")

__all__ = ["RunRegistry"]

_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
_SCORECARD_METRIC_FIELDS = (
    "corr_sharpe_ac",
    "rank_scalar",
    "corr",
    "mmc",
    "fnc",
    "deflated_sharpe",
    "std_corr",
    "max_drawdown",
)
# True when a larger value is better for that metric (parity-tested against
# nmr.meta._VERDICT_DIRECTIONS in tests/test_meta.py).
_SCORECARD_METRIC_DIRECTION = {
    "corr_sharpe_ac": True,
    "rank_scalar": True,
    "corr": True,
    "mmc": True,
    "fnc": True,
    "deflated_sharpe": True,
    "std_corr": False,
    "max_drawdown": False,
}


def _iter_run_records(root: Path):
    """Yield ``(slug, run_id, payload)`` for every ``run.json`` under ``root``.

    New layout: ``root/<slug>/runs/<run_id>/run.json`` (slug = family dir).
    """
    if not root.is_dir():
        return
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        runs_dir = entry / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            run_json = run_dir / "run.json"
            if run_json.is_file():
                yield entry.name, run_dir.name, json.loads(
                    run_json.read_text(encoding="utf-8")
                )


def _resolve_run_slug(root: Path, run_id: str) -> str:
    """Resolve ``run_id``'s family slug under the experiments layout root.

    Glob across families: 64-hex run_ids are unique (``run_id`` names the
    record directory ``root/<slug>/runs/<run_id>/``). Raises ``ValueError``
    when the record is absent or ambiguous. Single source of the scan —
    shared by :meth:`RunRegistry._resolve_slug` and
    ``nmr.meta.campaign_evidence`` (which holds a run_id but no slug).
    """
    matches: list[str] = []
    for slug, rid, _ in _iter_run_records(root):
        if rid == run_id:
            matches.append(slug)
    if not matches:
        raise ValueError(f"run {run_id} not found under {root}")
    unique = sorted(set(matches))
    if len(unique) > 1:
        raise ValueError(
            f"run {run_id} is ambiguous: found under families {unique}; "
            "pass slug explicitly"
        )
    return unique[0]


class RunRegistry:
    """Cross-family run registry: global comparison + champion pointer only.

    Runs live under ``experiments/<slug>/runs/<run_id>/run.json``; this class
    iterates families for comparison and owns ``champion.json``. Every read
    and write is rooted at ``self._root`` — run records at
    ``<root>/<slug>/runs/<run_id>/run.json``, the champion pointer at
    ``<root>/champion.json``, and the advisory lock file beside it — never the
    module-global ``paths.*`` helpers, so a registry over an isolated root
    cannot leak into the repo tree. Champion writes are single-writer,
    enforced by the ``champion.json.lock`` file lock around the
    read-compare-write (design spec §9).
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _iter_run_records(self):
        yield from _iter_run_records(self._root)

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(
                f"run_id={run_id!r} is not a 64-char lowercase hex string"
            )

    @staticmethod
    def _validate_identity(slug: str, run_id: str, payload: dict[str, Any]) -> None:
        """Embedded identity must match the record's path (2026-08-29 re-review,
        BLOCKING 1): ``payload.run_id`` must equal the path run_id, and
        ``manifest.config.run.name`` (when present) must equal the family slug.
        A misidentified record must never reach the champion pointer — fail loud."""
        embedded = payload.get("run_id")
        if embedded != run_id:
            raise ValueError(
                f"run record at {slug}/runs/{run_id} has embedded run_id={embedded!r}; "
                "record identity does not match its path"
            )
        run_name = (
            ((payload.get("manifest") or {}).get("config") or {}).get("run") or {}
        ).get("name")
        if run_name is not None and run_name != slug:
            raise ValueError(
                f"run record at {slug}/runs/{run_id} has manifest run.name={run_name!r}; "
                "does not match the family slug"
            )

    def _champion_path(self) -> Path:
        return self._root / "champion.json"

    def _champion_lock_path(self) -> Path:
        return self._root / "champion.json.lock"

    def _read_run(self, run_id: str, slug: str) -> dict[str, Any]:
        """Read a run record under ``self._root`` (fail loud, rooted).

        Identity-bound (2026-08-29 re-review, BLOCKING 1): the payload's
        embedded identity must match the path — ``payload.run_id`` == path
        run_id and ``manifest.config.run.name`` (when present) == family slug;
        a mismatched record raises ``ValueError``.
        """
        paths.validate_slug(slug)
        self._validate_run_id(run_id)
        path = self._root / slug / "runs" / run_id / "run.json"
        if not path.is_file():
            raise FileNotFoundError(f"no run record at {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._validate_identity(slug, run_id, payload)
        return payload

    def _resolve_slug(self, run_id: str) -> str:
        return _resolve_run_slug(self._root, run_id)

    def list(self) -> list[str]:
        found = []
        for slug, run_id, payload in self._iter_run_records():
            self._validate_identity(slug, run_id, payload)
            found.append(run_id)
        return sorted(found)

    def best(self, metric: str = "corr_sharpe_ac") -> tuple[str, str] | None:
        best: tuple[float, str, str] | None = None
        for slug, run_id, payload in self._iter_run_records():
            self._validate_identity(slug, run_id, payload)
            value = (payload.get("scorecard") or {}).get(metric)
            if value is None:
                continue
            if best is None or float(value) > best[0]:
                best = (float(value), run_id, slug)
        return (best[1], best[2]) if best else None

    def promote(self, run_id: str, slug: str | None = None) -> Path:
        """Promote ``run_id`` to champion; ``slug=None`` resolves it by scanning.

        Serialized with ``promote_if_better`` by the champion lock — a
        concurrent compare-and-promote cannot interleave a write.
        """
        with file_lock(self._champion_lock_path()):
            return self._promote_locked(run_id, slug)

    def _promote_locked(self, run_id: str, slug: str | None) -> Path:
        """Promote without acquiring the lock (caller holds it)."""
        self._validate_run_id(run_id)
        slug = self._resolve_slug(run_id) if slug is None else paths.validate_slug(slug)
        self._read_run(run_id, slug)  # existence check, fail loud
        payload = {
            "run_id": run_id,
            "experiment_slug": slug,
            "promoted_at": datetime.now(UTC).isoformat(),
        }
        logger.info("[promote] promoting %s/%s to champion", slug, run_id)
        return self._atomic_json_write(self._champion_path(), payload)

    def promote_if_better(
        self, run_id: str, slug: str | None = None, metric: str = "corr_sharpe_ac"
    ) -> tuple[Path, bool]:
        """Promote ``run_id`` only if its scorecard metric beats the champion's.

        Direction-aware (``max_drawdown``/``std_corr`` are lower-is-better).
        A scorecard-bearing candidate may displace a scorecard-less champion;
        a candidate lacking the metric is refused. A missing/dangling/corrupt
        champion pointer fails loud (never silently treated as no champion).

        The read-compare-write is wrapped in the inter-process advisory lock
        (``<root>/champion.json.lock``, 30 s timeout, clear error on expiry) —
        N concurrent writers serialize, so the final champion is the best
        value, never a stale last-write-wins race.
        """
        self._validate_run_id(run_id)
        if metric not in _SCORECARD_METRIC_FIELDS:
            raise ValueError(f"metric={metric!r} not in {sorted(_SCORECARD_METRIC_FIELDS)}")
        slug = self._resolve_slug(run_id) if slug is None else paths.validate_slug(slug)
        record = self._read_run(run_id, slug)
        candidate = (record.get("scorecard") or {}).get(metric)
        if candidate is None:
            raise ValueError(f"run {run_id} has no scorecard metric {metric}")
        with file_lock(self._champion_lock_path()):
            champion = self.resolve_champion()
            if champion is None:
                return self._promote_locked(run_id, slug), True
            champion_payload = json.loads(
                self._champion_path().read_text(encoding="utf-8")
            )
            champion_record = self._read_run(
                champion_payload["run_id"], champion_payload["experiment_slug"]
            )
            champion_value = (champion_record.get("scorecard") or {}).get(metric)
            if champion_value is None:
                return self._promote_locked(run_id, slug), True
            candidate_value = float(candidate)
            champion_value = float(champion_value)
            if _SCORECARD_METRIC_DIRECTION[metric]:
                better = candidate_value > champion_value
            else:
                better = candidate_value < champion_value
            if not better:
                logger.info(
                    "[promote_if_better] %s (%.6f) not better than champion %s (%.6f) "
                    "on %s; keeping champion",
                    run_id, candidate_value, champion_payload["run_id"], champion_value, metric,
                )
                return self._champion_path(), False
            return self._promote_locked(run_id, slug), True

    def resolve_champion(self) -> tuple[str, str] | None:
        pointer = self._champion_path()
        if not pointer.is_file():
            return None
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        run_id, slug = payload.get("run_id"), payload.get("experiment_slug")
        if not (isinstance(run_id, str) and isinstance(slug, str)):
            raise ValueError(f"corrupt champion pointer: {payload!r}")
        try:
            self._read_run(run_id, slug)
        except FileNotFoundError as exc:
            raise ValueError(f"champion pointer dangles: {slug}/{run_id}") from exc
        return run_id, slug

    def _atomic_json_write(self, path: Path, payload: dict[str, Any]) -> Path:
        atomic_write_text(path, json.dumps(payload, sort_keys=True, indent=2))
        return path
