from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from nmr.benchmark import BenchmarkSuite, scorecards_to_frame

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@dataclass(frozen=True)
class StrategyContext:
    model_id: str
    group: str
    raw_preds: pl.DataFrame
    seed: int


def _get_memory_usage_mb() -> float:
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return 0.0


def _min_one_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("n-boot must be >= 1")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimized low-memory real-data benchmark runner."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "v5.2")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts") / "benchmark_scores.csv"
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=Path("artifacts") / "benchmark_test_era_labels.csv",
    )
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--n-boot", type=_min_one_int, default=300)
    parser.add_argument("--min-overlap-eras", type=int, default=20)
    parser.add_argument("--horizon", choices=("20D", "60D"), default="20D")
    parser.add_argument("--min-train-eras", type=int, default=10)
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    parser.add_argument("--fast-mode", action="store_true")
    return parser.parse_args()


def _resolve_small_feature_set(data_dir: Path, available: list[str]) -> list[str]:
    """Resolve tutorial-style small feature set with safe local fallback."""
    feature_json = data_dir / "features.json"
    if feature_json.exists():
        try:
            payload = json.loads(feature_json.read_text(encoding="utf-8"))
            small = payload.get("feature_sets", {}).get("small", [])
            selected = [c for c in small if c in available]
            if selected:
                return selected
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    return sorted(available)[:42]


def _candidate_strategies(
    suite: BenchmarkSuite,
    benchmarks: pl.DataFrame,
    seed: int,
    min_train_eras: int,
    fast_mode: bool,
) -> Iterator[StrategyContext]:
    for idx, baseline in enumerate(
        ("constant-0.5", "uniform-random", "gaussian-random")
    ):
        r_seed = seed + idx
        yield StrategyContext(
            baseline, "null", suite.null_prediction_frame(baseline, seed=r_seed), r_seed
        )

    yield StrategyContext(
        "trivial", "classical", suite._trivial_prediction_frame(), seed + 3
    )

    if not fast_mode:
        yield StrategyContext(
            "linear",
            "classical",
            suite._walk_forward_model_predictions("linear", min_train_eras),
            seed + 4,
        )
        yield StrategyContext(
            "tree",
            "classical",
            suite._walk_forward_model_predictions("tree", min_train_eras),
            seed + 5,
        )

    benchmark_cols = sorted([c for c in benchmarks.columns if c not in {"era", "id"}])
    for col in benchmark_cols:
        preds = benchmarks.select(["era", "id", pl.col(col).alias("prediction")])
        yield StrategyContext(col, "benchmark_model", preds, seed + 6)


def _profile_label_space(
    df: pl.DataFrame, context: StrategyContext, space_name: str
) -> pl.DataFrame:
    """Vectorized calculation of label profiles per era without python-loop iteration."""
    return (
        df.group_by("era")
        .len(name="n_rows")
        .select(
            [
                pl.lit(context.model_id).alias("model_id"),
                pl.lit(context.group).alias("strategy_group"),
                pl.lit(space_name).alias("label_space"),
                pl.col("era").cast(pl.String),
                pl.col("n_rows"),
            ]
        )
    )


def _safe_scorecards_to_frame(scorecards: dict[str, object]) -> pl.DataFrame:
    try:
        return scorecards_to_frame(scorecards)
    except AttributeError:
        frames = []
        for model_id, sc in sorted(scorecards.items()):
            if hasattr(sc, "to_frame") and callable(sc.to_frame):
                frames.append(sc.to_frame())
            else:
                frames.append(
                    pl.DataFrame(
                        {
                            "model_id": [model_id],
                            "n_eras": [int(getattr(sc, "n_eras", 0))],
                            "rank_scalar": [float(getattr(sc, "rank_scalar", 0.0))],
                            "deflated_sharpe": [
                                float(getattr(sc, "deflated_sharpe", 0.0))
                            ],
                        }
                    )
                )
        if not frames:
            raise ValueError("Scorecards collection is empty.")
        return pl.concat(frames, how="vertical_relaxed").sort("model_id")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("benchmark_runner")

    if args.fast_mode:
        args.n_boot = 1

    log.info("=" * 60)
    log.info("RUNNER INITIALIZATION [Memory Status: %.2f MB]", _get_memory_usage_mb())
    log.info("=" * 60)

    for p in (args.output, args.labels_output):
        p.parent.mkdir(parents=True, exist_ok=True)

    log.info("Step 1/4: Loading inputs with tutorial small feature set")
    t_load = time.perf_counter()

    validation_path = args.data_dir / "validation.parquet"
    schema = pl.read_parquet_schema(validation_path)
    all_cols = list(schema.keys())

    all_feature_cols = [c for c in all_cols if c.startswith("feature_")]
    small_feature_cols = _resolve_small_feature_set(args.data_dir, all_feature_cols)
    target_cols = [c for c in all_cols if c == "target" or c.startswith("target_")]

    targets = pl.read_parquet(
        validation_path, columns=["era", "id", *target_cols, *small_feature_cols]
    )
    meta_model = pl.read_parquet(args.data_dir / "meta_model.parquet").select(
        ["era", "id", "numerai_meta_model"]
    )
    benchmarks = pl.read_parquet(args.data_dir / "validation_benchmark_models.parquet")

    log.info(
        "Inputs loaded in %.3fs [Memory Status: %.2f MB, feature_cols=%d]",
        time.perf_counter() - t_load,
        _get_memory_usage_mb(),
        len(small_feature_cols),
    )

    features = targets.select(["era", "id", *small_feature_cols])
    targets_subset = targets.select(["era", "id", *target_cols])

    log.info("Step 2/4: Initializing BenchmarkSuite")
    benchmark_cols = [c for c in benchmarks.columns if c not in {"era", "id"}]
    suite = BenchmarkSuite(
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets_subset,
        n_trials=1,
        seed=args.seed,
        horizon=args.horizon,
        benchmark_col=benchmark_cols[0] if benchmark_cols else None,
        n_boot=args.n_boot,
        min_overlap_eras=args.min_overlap_eras,
    )

    log.info("Step 3/4: Processing strategies sequentially with safety guards")
    scorecards, runtime_rows, label_profile_frames = {}, [], []

    # Simple structured dataclass wrapper for fallbacks instead of a messy dynamic inline class
    @dataclass
    class MockScorecard:
        rank_scalar: float = 0.0
        deflated_sharpe: float = 0.0
        n_eras: int = 0

    for ctx in _candidate_strategies(
        suite, benchmarks, args.seed, args.min_train_eras, args.fast_mode
    ):
        t0 = time.perf_counter()
        log.info(
            "Starting Strategy: ID='%s' (%s) [Memory: %.2f MB]",
            ctx.model_id,
            ctx.group,
            _get_memory_usage_mb(),
        )

        # Vectorized profiling step
        label_profile_frames.append(_profile_label_space(ctx.raw_preds, ctx, "raw"))

        norm_preds = suite.normalize_predictions(ctx.raw_preds)
        label_profile_frames.append(_profile_label_space(norm_preds, ctx, "normalized"))

        # Upgraded floating point check to use an epsilon threshold guard
        pred_variance = norm_preds.select(pl.col("prediction").var()).item()
        if pred_variance is None or pred_variance < 1e-9:
            log.warning(
                "  -> Low/Zero variance (< 1e-9) detected for '%s'. Short-circuiting metrics to avoid evaluation crashes.",
                ctx.model_id,
            )
            n_eras_unique = norm_preds.select(pl.col("era").n_unique()).item()
            scorecards[ctx.model_id] = MockScorecard(n_eras=n_eras_unique)
        else:
            scorecards[ctx.model_id] = suite.evaluate_normalized_predictions(
                norm_preds, model_id=ctx.model_id, seed=ctx.seed
            )

        elapsed = time.perf_counter() - t0
        runtime_rows.append(
            {
                "model_id": ctx.model_id,
                "strategy_group": ctx.group,
                "runtime_seconds": round(elapsed, 4),
                "prediction_rows": norm_preds.height,
                "seed": ctx.seed,
            }
        )
        log.info("Finished %s in %.3fs", ctx.model_id, elapsed)

    log.info("Step 4/4: Flushing output files to artifacts/")

    # Pure columnar aggregation and save out
    pl.concat(label_profile_frames).sort(["model_id", "era"]).write_csv(
        args.labels_output
    )

    out = _safe_scorecards_to_frame(scorecards).join(
        pl.DataFrame(runtime_rows),
        on="model_id",
        how="left",
    )
    out.write_csv(args.output)
    log.info("Run successfully complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
