"""Run registry with atomic metadata writes and champion pointer management."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

from nmr.runner import RunResult

__all__ = ["RunRegistry"]


class RunRegistry:
    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def record(self, result: RunResult) -> Path:
        run_dir = self._root / result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        oof_path = run_dir / "oof.parquet"
        result.oof.write_parquet(oof_path)

        run_payload = {
            "run_id": result.run_id,
            "metrics": dataclasses.asdict(result.metrics),
            "manifest": result.manifest,
            "oof_path": oof_path.name,
            "artifact_path": str(result.artifact.path) if result.artifact else None,
            "artifact_manifest": result.artifact.manifest if result.artifact else None,
        }
        self._atomic_json_write(run_dir / "run.json", run_payload)
        return run_dir

    def list(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for run_file in self._root.glob("*/run.json"):
            entries.append(json.loads(run_file.read_text(encoding="utf-8")))
        entries.sort(
            key=lambda entry: (self._root / entry["run_id"] / "run.json")
            .stat()
            .st_mtime,
            reverse=True,
        )
        return entries

    def best(self, metric: str = "sharpe") -> dict[str, Any] | None:
        runs = self.list()
        if not runs:
            return None
        return max(runs, key=lambda run: float(run["metrics"][metric]))

    def promote(self, run_id: str) -> Path:
        run_json = self._root / run_id / "run.json"
        if not run_json.exists():
            raise FileNotFoundError(f"Run {run_id!r} does not exist in registry")

        champion_path = self._root / "champion.json"
        self._atomic_json_write(champion_path, {"run_id": run_id})
        return champion_path

    def _atomic_json_write(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=path.parent,
            prefix=f"{path.name}.tmp.",
            suffix=".json",
        ) as tmp:
            json.dump(payload, tmp, sort_keys=True, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_name = tmp.name

        os.replace(temp_name, path)
