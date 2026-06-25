from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import polars as pl

from nmr.benchmark import BenchmarkSuite, scorecards_to_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-data benchmark scorecards sequentially with step-wise logging and timing."
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
        "--labels-output",
        type=Path,
        default=Path("artifacts") / "benchmark_test_era_labels.csv",
        help="CSV output path for per-era label counts used in phase-1 profiling.",
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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console log verbosity.",
    )
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


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
) -> Iterator[tuple[str, str, pl.DataFrame, int]]:

    for idx, baseline in enumerate(
        ("constant-0.5", "uniform-random", "gaussian-random")
    ):
        yield (
            baseline,
            "null",
            suite.null_prediction_frame(baseline, seed=seed + idx),
            seed + idx,
        )

    yield (
        (
            "trivial",
            "classical",
            suite._trivial_prediction_frame(),
            seed,
        )
    )
    yield (
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
    yield (
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
        yield (col, "benchmark_model", predictions, seed)


def main() -> int:
    args = _parse_args()
    _configure_logging(args.log_level)
    log = logging.getLogger("benchmark_runner")

    data_dir = args.data_dir
    output_path = args.output
    labels_output_path = args.labels_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Step 1/5: loading inputs from %s", data_dir)
    load_start = time.perf_counter()
    meta_model, benchmarks, features, targets = _load_inputs(data_dir)
    load_elapsed = time.perf_counter() - load_start
    log.info(
        "Loaded inputs: rows(targets)=%d rows(benchmarks)=%d feature_cols=%d benchmark_cols=%d elapsed=%.2fs",
        targets.height,
        benchmarks.height,
        len(features.columns) - 2,
        len(benchmarks.columns) - 2,
        load_elapsed,
    )

    benchmark_cols = [col for col in benchmarks.columns if col not in {"era", "id"}]
    benchmark_col = benchmark_cols[0] if benchmark_cols else None
    log.info("Step 2/5: initializing benchmark suite")
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

    total_strategies = 6 + len(benchmark_cols)
    log.info(
        "Step 3/6: sequential evaluation will run %d strategies (6 built-ins + %d benchmark columns)",
        total_strategies,
        len(benchmark_cols),
    )

    candidates = list(
        _candidate_strategies(
            suite,
            benchmarks,
            seed=args.seed,
            min_train_eras=args.min_train_eras,
        )
    )

    log.info("Step 4/6: precomputing normalized test-era labels for all strategies")
    prepared: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    label_prep_start = time.perf_counter()
    for idx, (model_id, strategy_group, predictions, run_seed) in enumerate(
        candidates,
        start=1,
    ):
        strategy_start = time.perf_counter()
        normalized = suite.normalize_predictions(predictions)
        labels = normalized.group_by("era").len().rename({"len": "n_rows"}).sort("era")
        label_elapsed = time.perf_counter() - strategy_start

        prepared.append(
            {
                "model_id": model_id,
                "strategy_group": strategy_group,
                "seed": int(run_seed),
                "normalized": normalized,
                "label_seconds": round(label_elapsed, 6),
            }
        )

        era_values = labels.get_column("era").to_list()
        row_counts = labels.get_column("n_rows").to_list()
        for era, n_rows in zip(era_values, row_counts, strict=True):
            label_rows.append(
                {
                    "model_id": model_id,
                    "strategy_group": strategy_group,
                    "era": str(era),
                    "n_rows": int(n_rows),
                }
            )

        log.info(
            "Label prep [%d/%d] %s: eras=%d rows=%d elapsed=%.3fs",
            idx,
            total_strategies,
            model_id,
            len(era_values),
            normalized.height,
            label_elapsed,
        )

    label_total_elapsed = time.perf_counter() - label_prep_start
    label_frame = pl.DataFrame(label_rows).sort(["model_id", "era"])
    label_frame.write_csv(labels_output_path)
    log.info(
        "Label CSV written: path=%s rows=%d elapsed=%.2fs",
        labels_output_path,
        label_frame.height,
        label_total_elapsed,
    )

    scorecards = {}
    runtime_rows: list[dict[str, object]] = []
    overall_start = time.perf_counter()
    log.info(
        "Step 5/6: running scorecard metrics from precomputed normalized predictions"
    )
    for idx, payload in enumerate(
        prepared,
        start=1,
    ):
        model_id = str(payload["model_id"])
        strategy_group = str(payload["strategy_group"])
        run_seed = int(payload["seed"])
        normalized = payload["normalized"]
        label_seconds = float(payload["label_seconds"])

        log.info(
            "Metrics [%d/%d] %s (group=%s, rows=%d, seed=%d)",
            idx,
            total_strategies,
            model_id,
            strategy_group,
            normalized.height,
            run_seed,
        )
        start = time.perf_counter()
        score = suite.evaluate_normalized_predictions(
            normalized,
            model_id=model_id,
            seed=run_seed,
        )
        metric_elapsed = time.perf_counter() - start
        scorecards[model_id] = score
        runtime_rows.append(
            {
                "model_id": model_id,
                "strategy_group": strategy_group,
                "label_seconds": round(label_seconds, 6),
                "metric_seconds": round(metric_elapsed, 6),
                "runtime_seconds": round(label_seconds + metric_elapsed, 6),
                "prediction_rows": int(normalized.height),
                "seed": int(run_seed),
            }
        )
        log.info(
            "Completed %s: rank_scalar=%.8f dsr=%.6f n_eras=%d label=%.3fs metrics=%.3fs total=%.3fs",
            model_id,
            score.rank_scalar,
            score.deflated_sharpe,
            score.n_eras,
            label_seconds,
            metric_elapsed,
            label_seconds + metric_elapsed,
        )

    total_elapsed = time.perf_counter() - overall_start
    log.info("Step 6/6: writing output CSV to %s", output_path)
    scorecard_frame = scorecards_to_frame(scorecards)
    runtime_frame = pl.DataFrame(runtime_rows)
    out = scorecard_frame.join(runtime_frame, on="model_id", how="left").sort(
        "model_id"
    )
    out.write_csv(output_path)

    log.info(
        "CSV written: path=%s rows=%d cols=%d total_elapsed=%.2fs",
        output_path,
        out.height,
        len(out.columns),
        total_elapsed,
    )
    log.info(
        "Top rows by rank_scalar:\n%s",
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
        )
        .sort("rank_scalar", descending=True)
        .head(15),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
