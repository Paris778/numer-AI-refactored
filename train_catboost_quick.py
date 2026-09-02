"""Train a quick single-target CatBoost model (ender60) and record the run."""

# ruff: noqa: E402 — apply_thread_limits() must run before the imports below:
# polars/OpenMP/BLAS read their pool sizes at first use, not at import.
from __future__ import annotations

from nmr.hardware import apply_thread_limits

apply_thread_limits()

import logging

from nmr import ExperimentRunner, experiment_store, load_config, paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> int:
    cfg = load_config("configs/catboost-quick-ender60.yaml")
    runner = ExperimentRunner(cfg)
    result = runner.run(deploy=False)

    slug = paths.validate_slug(cfg.run.name)
    run_dir = experiment_store.record_run_result(slug, result)

    print("=" * 60)
    print("CATBOOST QUICK (ENDER60) COMPLETE")
    print("=" * 60)
    print(f"run_id:  {result.run_id}")
    print(f"run_dir: {run_dir}")
    print(f"oof_rows: {result.oof.height}")
    print(f"oof_corr_mean:  {result.metrics.mean:.5f}")
    print(f"oof_corr_sharpe: {result.metrics.sharpe:.5f}")
    if result.scorecard is not None:
        sc = result.scorecard
        print(f"scorecard_corr:           {sc.corr.value:.5f}")
        print(f"scorecard_mmc:            {sc.mmc.value:.5f}")
        print(f"scorecard_fnc:            {sc.fnc:.5f}")
        print(f"scorecard_corr_sharpe_ac: {sc.corr_sharpe_ac.value:.5f}")
        print(f"scorecard_n_eras:         {sc.n_eras}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
