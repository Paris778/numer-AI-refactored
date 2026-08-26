"""Cross-family run registry: global comparison + champion pointer only.

Runs live under ``experiments/<slug>/runs/<run_id>/run.json`` (persistence
lives in :mod:`nmr.experiment_store`); this class iterates families for
comparison and owns the atomic ``champion.json`` pointer at the experiments
root. Champion writes are single-writer (CLI/runner entry points only —
design spec §9).

Interim compat shims (removed in Task 11): ``record(RunResult)`` keeps the
legacy single-pool layout (``root/<run_id>/`` — ``tests/test_campaign.py``
pins it and its stub manifests carry no ``config.run.name``); iteration and
``promote``/``promote_if_better`` additionally accept legacy rows (slug
derived from ``manifest.config.run.name`` when a champion pointer needs one),
and ``promote``/``promote_if_better`` accept ``slug=None``, resolving the
family by scanning the registry root (fail loud on not-found/ambiguous). The
plan's callers are retargeted in Task 11.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nmr import experiment_store, paths
from nmr._atomicio import atomic_write_text
from nmr.runner import RunResult

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
    Legacy compat layout: ``root/<run_id>/run.json`` (64-hex run dir directly
    under the root, slug ``None``) — written by the compat ``record()`` until
    Task 11.
    """
    if not root.is_dir():
        return
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        runs_dir = entry / "runs"
        if runs_dir.is_dir():
            for run_dir in sorted(runs_dir.iterdir()):
                run_json = run_dir / "run.json"
                if run_json.is_file():
                    yield entry.name, run_dir.name, json.loads(
                        run_json.read_text(encoding="utf-8")
                    )
        elif _RUN_ID_PATTERN.fullmatch(entry.name):
            run_json = entry / "run.json"
            if run_json.is_file():
                yield None, entry.name, json.loads(
                    run_json.read_text(encoding="utf-8")
                )


def _write_parquet_atomic(path: Path, frame: pl.DataFrame) -> None:
    """Write a parquet frame via temp file + os.replace (no fsync)."""
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        frame.write_parquet(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class RunRegistry:
    """Cross-family run registry: global comparison + champion pointer only.

    Runs live under ``experiments/<slug>/runs/<run_id>/run.json``; this class
    iterates families for comparison and owns ``champion.json``. Champion
    writes are single-writer (CLI/runner entry points only — spec §9).
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

    def _read_run(self, run_id: str, slug: str) -> dict[str, Any]:
        """Read a run record from the experiments layout or the legacy root."""
        try:
            return experiment_store.read_run(slug, run_id)
        except FileNotFoundError:
            legacy = self._root / run_id / "run.json"
            if legacy.is_file():
                return json.loads(legacy.read_text(encoding="utf-8"))
            raise

    def _resolve_slug(self, run_id: str) -> str:
        matches: list[str] = []
        for slug, rid, payload in self._iter_run_records():
            if rid != run_id:
                continue
            if slug is None:
                name = ((payload.get("manifest") or {}).get("config") or {}).get(
                    "run"
                ) or {}
                candidate = name.get("name")
                if not isinstance(candidate, str) or not candidate:
                    raise ValueError(
                        f"run {run_id} is a legacy row without manifest "
                        "config.run.name; pass slug explicitly"
                    )
                matches.append(candidate)
            else:
                matches.append(slug)
        if not matches:
            raise ValueError(f"run {run_id} not found under {self._root}")
        unique = sorted(set(matches))
        if len(unique) > 1:
            raise ValueError(
                f"run {run_id} is ambiguous: found under families {unique}; "
                "pass slug explicitly"
            )
        return unique[0]

    def record(self, result: RunResult) -> Path:
        """Interim compat (removed in Task 11): legacy-layout record.

        Writes ``self._root/<run_id>/{run.json, oof.parquet,
        validation_preds.parquet?}`` exactly like the pre-refactor registry —
        ``tests/test_campaign.py`` pins this layout and its stub manifests
        carry no ``config.run.name`` (no slug available at record time). Task 7
        moves run persistence to :func:`nmr.experiment_store.record_run`.
        """
        run_dir = self._root / result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_parquet_atomic(run_dir / "oof.parquet", result.oof)
        if result.validation_predictions is not None:
            _write_parquet_atomic(
                run_dir / "validation_preds.parquet", result.validation_predictions
            )
        self._atomic_json_write(run_dir / "run.json", self._result_payload(result))
        logger.info("[record] run %s recorded -> %s", result.run_id, run_dir)
        return run_dir

    @staticmethod
    def _result_payload(result: RunResult) -> dict[str, Any]:
        scorecard_block = None
        if result.scorecard is not None:
            row = result.scorecard.to_frame().to_dicts()[0]
            scorecard_block = {
                key: value
                for key, value in row.items()
                if not key.startswith(("timing_", "quality_metric"))
            }
        return {
            "run_id": result.run_id,
            "metrics": dataclasses.asdict(result.metrics),
            "manifest": result.manifest,
            "scorecard": scorecard_block,
            "oof_path": "oof.parquet",
            "artifact_path": str(result.artifact.path) if result.artifact else None,
            "artifact_manifest": result.artifact.manifest if result.artifact else None,
        }

    def list(self) -> list[str]:
        return sorted(run_id for _, run_id, _ in self._iter_run_records())

    def best(self, metric: str = "corr_sharpe_ac") -> tuple[str, str] | None:
        best: tuple[float, str, str] | None = None
        for slug, run_id, payload in self._iter_run_records():
            value = (payload.get("scorecard") or {}).get(metric)
            if value is None:
                continue
            if best is None or float(value) > best[0]:
                best = (float(value), run_id, slug)
        return (best[1], best[2]) if best else None

    def promote(self, run_id: str, slug: str | None = None) -> Path:
        """Promote ``run_id`` to champion; ``slug=None`` resolves it by scanning."""
        self._validate_run_id(run_id)
        slug = self._resolve_slug(run_id) if slug is None else paths.validate_slug(slug)
        self._read_run(run_id, slug)  # existence check, fail loud
        payload = {
            "run_id": run_id,
            "experiment_slug": slug,
            "promoted_at": datetime.now(UTC).isoformat(),
        }
        logger.info("[promote] promoting %s/%s to champion", slug, run_id)
        return self._atomic_json_write(paths.champion_path(), payload)

    def promote_if_better(
        self, run_id: str, slug: str | None = None, metric: str = "corr_sharpe_ac"
    ) -> tuple[Path, bool]:
        """Promote ``run_id`` only if its scorecard metric beats the champion's.

        Direction-aware (``max_drawdown``/``std_corr`` are lower-is-better).
        A scorecard-bearing candidate may displace a scorecard-less champion;
        a candidate lacking the metric is refused. A missing/dangling/corrupt
        champion pointer fails loud (never silently treated as no champion).
        """
        self._validate_run_id(run_id)
        if metric not in _SCORECARD_METRIC_FIELDS:
            raise ValueError(f"metric={metric!r} not in {sorted(_SCORECARD_METRIC_FIELDS)}")
        slug = self._resolve_slug(run_id) if slug is None else paths.validate_slug(slug)
        record = self._read_run(run_id, slug)
        candidate = (record.get("scorecard") or {}).get(metric)
        if candidate is None:
            raise ValueError(f"run {run_id} has no scorecard metric {metric}")
        champion = self.resolve_champion()
        if champion is None:
            return self.promote(run_id, slug), True
        champion_payload = json.loads(paths.champion_path().read_text(encoding="utf-8"))
        champion_record = self._read_run(
            champion_payload["run_id"], champion_payload["experiment_slug"]
        )
        champion_value = (champion_record.get("scorecard") or {}).get(metric)
        if champion_value is None:
            return self.promote(run_id, slug), True
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
            return paths.champion_path(), False
        return self.promote(run_id, slug), True

    def resolve_champion(self) -> tuple[str, str] | None:
        pointer = paths.champion_path()
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
