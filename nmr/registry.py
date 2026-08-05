"""Run registry with atomic metadata writes and champion pointer management."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import polars as pl

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
# True when a larger value is better for that metric.
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
_METRIC_SUMMARY_FIELDS = ("mean", "std", "sharpe", "max_drawdown")


class RunRegistry:
    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def record(self, result: RunResult) -> Path:
        logger.info("[record] recording run %s", result.run_id)
        run_dir = self._root / result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        oof_path = run_dir / "oof.parquet"
        tmp_oof = run_dir / f"{oof_path.name}.tmp.{os.getpid()}"
        try:
            result.oof.write_parquet(tmp_oof)
            os.replace(tmp_oof, oof_path)
        finally:
            if tmp_oof.exists():
                tmp_oof.unlink()
        logger.info("[record] OOF written to %s", oof_path)

        scorecard_block = None
        if result.scorecard is not None:
            row = result.scorecard.to_frame().to_dicts()[0]
            scorecard_block = {
                key: value
                for key, value in row.items()
                if not key.startswith(("timing_", "quality_metric"))
            }

        run_payload = {
            "run_id": result.run_id,
            "metrics": dataclasses.asdict(result.metrics),
            "manifest": result.manifest,
            "scorecard": scorecard_block,
            "oof_path": oof_path.name,
            "artifact_path": str(result.artifact.path) if result.artifact else None,
            "artifact_manifest": result.artifact.manifest if result.artifact else None,
        }
        self._atomic_json_write(run_dir / "run.json", run_payload)
        logger.info("[record] run metadata written to %s/run.json", run_dir)
        return run_dir

    def list(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for run_file in self._root.glob("*/run.json"):
            entries.append(json.loads(run_file.read_text(encoding="utf-8")))
        entries.sort(
            key=lambda entry: (
                (self._root / entry["run_id"] / "run.json").stat().st_mtime,
                entry["run_id"],
            ),
            reverse=True,
        )
        return entries

    def best(self, metric: str = "sharpe") -> dict[str, Any] | None:
        if metric not in _METRIC_SUMMARY_FIELDS:
            raise ValueError(
                f"metric={metric!r} not in {sorted(_METRIC_SUMMARY_FIELDS)}"
            )
        runs = self.list()
        if not runs:
            return None
        return max(
            runs,
            key=lambda run: (float(run["metrics"][metric]), run["run_id"]),
        )

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(
                f"run_id={run_id!r} is not a 64-char lowercase hex string"
            )

    def promote(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        logger.info("[promote] promoting run %s to champion", run_id)
        run_json = self._root / run_id / "run.json"
        if not run_json.exists():
            raise FileNotFoundError(f"Run {run_id!r} does not exist in registry")

        champion_path = self._root / "champion.json"
        self._atomic_json_write(champion_path, {"run_id": run_id})
        logger.info("[promote] champion pointer written to %s", champion_path)
        return champion_path

    def promote_if_better(self, run_id: str, metric: str = "corr_sharpe_ac") -> tuple[Path, bool]:
        """Promote ``run_id`` only if its scorecard metric beats the champion's.

        Direction-aware: ``max_drawdown``/``std_corr`` are lower-is-better.
        A scorecard-bearing candidate may displace a scorecard-less champion
        (legacy OOF metrics are in-sample-biased). Legacy candidates (no
        scorecard) are refused — use :meth:`promote` for explicit overrides.
        """
        self._validate_run_id(run_id)
        if metric not in _SCORECARD_METRIC_FIELDS:
            raise ValueError(
                f"metric={metric!r} not in {sorted(_SCORECARD_METRIC_FIELDS)}"
            )
        run_json = self._root / run_id / "run.json"
        if not run_json.exists():
            raise FileNotFoundError(f"Run {run_id!r} does not exist in registry")
        candidate = json.loads(run_json.read_text(encoding="utf-8"))
        candidate_scorecard = candidate.get("scorecard")
        if not candidate_scorecard or metric not in candidate_scorecard:
            raise ValueError(
                f"Run {run_id!r} has no scorecard metric {metric!r}; "
                "legacy runs require manual promote()"
            )

        champion_path = self._root / "champion.json"
        if not champion_path.exists():
            logger.info("[promote_if_better] no champion; promoting %s", run_id)
            self._atomic_json_write(champion_path, {"run_id": run_id})
            return champion_path, True

        champion_id = json.loads(champion_path.read_text(encoding="utf-8")).get("run_id")
        if not champion_id:
            logger.warning(
                "[promote_if_better] champion pointer corrupt (missing run_id); "
                "treating as no champion"
            )
            self._atomic_json_write(champion_path, {"run_id": run_id})
            return champion_path, True
        champion_json = self._root / champion_id / "run.json"
        if not champion_json.exists():
            logger.warning(
                "[promote_if_better] champion %s missing; treating as no champion",
                champion_id,
            )
            self._atomic_json_write(champion_path, {"run_id": run_id})
            return champion_path, True

        champion = json.loads(champion_json.read_text(encoding="utf-8"))
        champion_scorecard = champion.get("scorecard")
        if not champion_scorecard or metric not in champion_scorecard:
            logger.info(
                "[promote_if_better] champion %s has no scorecard; promoting on presence",
                champion_id,
            )
            self._atomic_json_write(champion_path, {"run_id": run_id})
            return champion_path, True

        higher_is_better = _SCORECARD_METRIC_DIRECTION[metric]
        candidate_value = float(candidate_scorecard[metric])
        champion_value = float(champion_scorecard[metric])
        if higher_is_better:
            better = candidate_value > champion_value
        else:
            better = candidate_value < champion_value
        if not better:
            logger.info(
                "[promote_if_better] %s (%.6f) not better than champion %s (%.6f) on %s; "
                "keeping champion",
                run_id, candidate_value, champion_id, champion_value, metric,
            )
            return champion_path, False

        logger.info("[promote_if_better] promoting %s over %s on %s", run_id, champion_id, metric)
        self._atomic_json_write(champion_path, {"run_id": run_id})
        return champion_path, True

    def _atomic_json_write(self, path: Path, payload: dict[str, Any]) -> None:
        atomic_write_text(path, json.dumps(payload, sort_keys=True, indent=2))
