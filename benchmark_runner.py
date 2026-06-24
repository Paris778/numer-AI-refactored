from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl
from nmr.benchmark import BenchmarkSuite, scorecards_to_frame
from tqdm.auto import tqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-data benchmark scorecards with progress bars and timing."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "v5.2",
        help="Directory containing validation.parquet, meta_model.parquet, and validation_benchmark_models.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "benchmark_scores.csv",
        help="CSV output path for benchmark scorecards plus runtime metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=77,
        help="Base RNG seed for deterministic benchmark generation.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=300,
        help="Bootstrap replicates for scorecard confidence intervals.",
    )
    parser.add_argument(
        "--min-overlap-eras",
        type=int,
        default=20,
        help="Minimum overlap eras required for overlap-sensitive metrics.",
    )
    parser.add_argument(
        "--horizon",
        choices=("20D", "60D"),
        default="20D",
        help="Primary scorecard horizon.",
    )
    parser.add_argument(
        "--min-train-eras",
        type=int,
        default=10,
        help="Minimum train eras for walk-forward classical baselines.",
    )
    return parser.parse_args()


def _load_inputs(
    data_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    validation_path = data_dir / "validation.parquet"
    meta_path = data_dir / "meta_model.parquet"
    benchmark_path = data_dir / "validation_benchmark_models.parquet"

    validation = pl.read_parquet(validation_path)
    meta_model = pl.read_parquet(meta_path).select(["era", "id", "numerai_meta_model"])
    benchmarks = pl.read_parquet(benchmark_path)

    feature_cols = [col for col in validation.columns if col.startswith("feature_")]
    target_cols = [
        col
        for col in validation.columns
        if col == "target" or col.startswith("target_")
    ]

    features = validation.select(["era", "id", *feature_cols])
    targets = validation.select(["era", "id", *target_cols])
    return meta_model, benchmarks, features, targets


def _candidate_strategies(
    suite: BenchmarkSuite,
    benchmarks: pl.DataFrame,
    *,
    seed: int,
    min_train_eras: int,
) -> list[tuple[str, str, pl.DataFrame, int]]:
    strategies: list[tuple[str, str, pl.DataFrame, int]] = []

    for idx, baseline in enumerate(
        ("constant-0.5", "uniform-random", "gaussian-random")
    ):
        strategies.append(
            (
                baseline,
                "null",
                suite.null_prediction_frame(baseline, seed=seed + idx),
                seed + idx,
            )
        )

    strategies.append(
        (
            "trivial",
            "classical",
            suite._trivial_prediction_frame(),
            seed,
        )
    )
    strategies.append(
        (
            "linear",
            "classical",
            suite._walk_forward_model_predictions(
                model_name="linear",
                min_train_eras=min_train_eras,
            ),
            seed,
        )
    )
    strategies.append(
        (
            "tree",
            "classical",
            suite._walk_forward_model_predictions(
                model_name="tree",
                min_train_eras=min_train_eras,
            ),
            seed,
        )
    )

    benchmark_cols = [col for col in benchmarks.columns if col not in {"era", "id"}]
    for col in sorted(benchmark_cols):
        predictions = benchmarks.select(["era", "id", pl.col(col).alias("prediction")])
        strategies.append((col, "benchmark_model", predictions, seed))

    return strategies


def main() -> int:
    args = _parse_args()
    data_dir = args.data_dir
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] data_dir={data_dir}")
    load_start = time.perf_counter()
    meta_model, benchmarks, features, targets = _load_inputs(data_dir)
    load_elapsed = time.perf_counter() - load_start
    print(
        "[load] "
        f"rows(targets)={targets.height} rows(benchmarks)={benchmarks.height} "
        f"feature_cols={len(features.columns) - 2} benchmark_cols={len(benchmarks.columns) - 2} "
        f"elapsed={load_elapsed:.2f}s"
    )

    benchmark_cols = [col for col in benchmarks.columns if col not in {"era", "id"}]
    benchmark_col = benchmark_cols[0] if benchmark_cols else None
    suite = BenchmarkSuite(
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=args.seed,
        horizon=args.horizon,
        benchmark_col=benchmark_col,
        n_boot=args.n_boot,
        min_overlap_eras=args.min_overlap_eras,
    )

    strategies = _candidate_strategies(
        suite,
        benchmarks,
        seed=args.seed,
        min_train_eras=args.min_train_eras,
    )

    scorecards = {}
    runtime_rows: list[dict[str, object]] = []
    overall_start = time.perf_counter()
    progress = tqdm(strategies, desc="Benchmark strategies", unit="strategy")
    for model_id, strategy_group, predictions, run_seed in progress:
        progress.set_postfix_str(model_id)
        start = time.perf_counter()
        score = suite.evaluate_predictions(
            predictions, model_id=model_id, seed=run_seed
        )
        elapsed = time.perf_counter() - start
        scorecards[model_id] = score
        runtime_rows.append(
            {
                "model_id": model_id,
                "strategy_group": strategy_group,
                "runtime_seconds": round(elapsed, 6),
                "prediction_rows": int(predictions.height),
                "seed": int(run_seed),
            }
        )
        print(
            f"[done] {model_id} group={strategy_group} "
            f"rank_scalar={score.rank_scalar:.8f} dsr={score.deflated_sharpe:.6f} "
            f"n_eras={score.n_eras} elapsed={elapsed:.2f}s"
        )

    total_elapsed = time.perf_counter() - overall_start
    scorecard_frame = scorecards_to_frame(scorecards)
    runtime_frame = pl.DataFrame(runtime_rows)
    out = scorecard_frame.join(runtime_frame, on="model_id", how="left").sort(
        "model_id"
    )
    out.write_csv(output_path)

    print(
        f"[write] path={output_path} rows={out.height} cols={len(out.columns)} total_elapsed={total_elapsed:.2f}s"
    )
    print(
        out.select(
            [
                "model_id",
                "strategy_group",
                "rank_scalar",
                "deflated_sharpe",
                "corr",
                "mmc",
                "runtime_seconds",
                "n_eras",
            ]
        ).sort("rank_scalar", descending=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
