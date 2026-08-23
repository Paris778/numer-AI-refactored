"""Train the first competitive model and produce a deployable artifact."""

# ruff: noqa: E402 — apply_thread_limits() must run before the imports below:
# polars/OpenMP/BLAS read their pool sizes at first use, not at import.
from __future__ import annotations

from nmr.hardware import apply_thread_limits

apply_thread_limits()

import json
import logging

from nmr import ExperimentRunner, load_config
from nmr.registry import RunRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> int:
    cfg = load_config("configs/first_model.yaml")
    runner = ExperimentRunner(cfg)
    result = runner.run(deploy=True)

    registry = RunRegistry(cfg.run.artifacts_dir / "registry")
    run_dir = registry.record(result)
    champion_path, promoted = registry.promote_if_better(result.run_id)
    print(f"promoted:    {promoted} (champion: {champion_path})")

    print("=" * 60)
    print("FIRST COMPETITIVE MODEL COMPLETE")
    print("=" * 60)
    print(f"run_id:      {result.run_id}")
    print(f"run_dir:     {run_dir}")
    print(f"oof_rows:    {result.oof.height}")
    print(f"metrics:     {result.metrics}")
    print(f"artifact:    {result.artifact.path if result.artifact else None}")
    print(f"weights:     {result.manifest['weights']}")
    print(f"pred_cols:   {result.manifest['pred_cols']}")

    summary_path = run_dir / "summary.json"
    summary = {
        "run_id": result.run_id,
        "metrics": {
            "mean": result.metrics.mean,
            "std": result.metrics.std,
            "sharpe": result.metrics.sharpe,
            "max_drawdown": result.metrics.max_drawdown,
        },
        "weights": result.manifest["weights"],
        "pred_cols": result.manifest["pred_cols"],
        "feature_cols": result.manifest["feature_cols"],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary:     {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
