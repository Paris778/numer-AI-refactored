"""Deterministic dataset analysis -> machine-readable dumps.

Thin control plane: wires ``nmr.analysis`` functions over train+validation
and writes JSON/parquet dumps under ``artifacts/reports/dataset_analysis/``
for the report renderer. See docs/superpowers/specs/2026-08-08-dataset-analysis-design.md §4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from nmr import analysis
from nmr._atomicio import atomic_write_text
from nmr.config import DataConfig
from nmr.data import IngestionAgent
from nmr.features import resolve_feature_sets
from nmr.refresh import CURRENT_DATA_VERSION

_ERA_BATCH = 40  # eras scanned per lazy pass (bounded transient memory)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics and write analysis dumps."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--version", default=CURRENT_DATA_VERSION)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "reports" / "dataset_analysis",
    )
    parser.add_argument("--features", choices=("small", "medium", "all"), default="all")
    parser.add_argument("--max-eras", type=int, default=None)
    parser.add_argument("--targets", action="append", default=None)
    parser.add_argument("--all-targets", action="store_true")
    parser.add_argument("--full-all-matrix", action="store_true")
    return parser.parse_args(argv)


def _atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".tmp.", suffix=".part"
    )
    os.close(fd)  # Windows: release the handle so os.replace can work
    tmp = Path(tmp_name)
    try:
        df.write_parquet(tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_write_json(payload: object, path: Path) -> None:
    """Atomic JSON write via the shared _atomicio text helper."""
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _era_chunks(
    agent: IngestionAgent,
    splits: Sequence[str],
    columns: Sequence[str],
    max_eras: int | None,
) -> list[pl.DataFrame]:
    """Collect era-partitioned chunks from the requested splits.

    For small column sets (meta/era/targets). Feature-heavy analyses use
    ``_iter_era_chunks`` instead — never materialize the full feature frame.
    """
    frames = [agent.scan(split, columns=columns).collect() for split in splits]
    if max_eras is not None:
        frames = [
            f.filter(pl.col("era").cast(pl.Int64) <= max_eras) for f in frames
        ]
    if not frames:
        raise ValueError("no split frames to analyze")
    combined = pl.concat(frames)
    return combined.partition_by("era", maintain_order=True)


def _iter_era_chunks(
    agent: IngestionAgent,
    splits: Sequence[str],
    columns: Sequence[str],
    max_eras: int | None,
) -> Iterator[pl.DataFrame]:
    """Yield one era at a time from batched lazy scans (bounded memory).

    The full feature universe (v5.3 ``all`` = 3555 columns) cannot be collected
    as one frame (~100 GB); this streams era batches via predicate pushdown,
    so each analysis holds at most one era plus accumulators. Deterministic:
    eras are yielded in ascending integer order.
    """
    era_sets: list[set[str]] = []
    for split in splits:
        labels = agent.scan(split, columns=["era"]).collect().get_column("era").to_list()
        era_sets.append(set(labels))
    all_eras = sorted(set().union(*era_sets), key=int)
    if max_eras is not None:
        all_eras = [e for e in all_eras if int(e) <= max_eras]

    for start in range(0, len(all_eras), _ERA_BATCH):
        batch = all_eras[start : start + _ERA_BATCH]
        parts = []
        for split in splits:
            part = (
                agent.scan(split, columns=columns)
                .filter(pl.col("era").is_in(batch))
                .collect()
            )
            if part.height:
                parts.append(part)
        if not parts:
            continue
        combined = pl.concat(parts)
        yield from combined.partition_by("era", maintain_order=True)


def _resolve_reference_targets(
    all_targets: list[str], explicit: list[str] | None, all_targets_flag: bool
) -> list[str]:
    if explicit:
        return explicit
    if all_targets_flag:
        return all_targets
    primary_20 = "target" if "target" in all_targets else all_targets[0]
    primary_60 = next(
        (t for t in all_targets if t.endswith("_60") and t != primary_20), primary_20
    )
    return [primary_20, primary_60]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    version_dir = args.data_dir / args.version
    features_path = version_dir / "features.json"
    if not features_path.exists():
        print(f"ERROR: {features_path} missing — run refresh_data.py first", file=sys.stderr)
        return 1

    feature_sets = resolve_feature_sets(features_path)
    all_targets = json.loads(features_path.read_text(encoding="utf-8"))["targets"]
    feature_cols = feature_sets[args.features]
    medium_cols = feature_sets["medium"]
    targets = _resolve_reference_targets(all_targets, args.targets, args.all_targets)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    config = DataConfig(version=args.version, feature_set=args.features, data_dir=args.data_dir)
    agent = IngestionAgent(config)
    splits = ("train", "validation")
    target_columns = [c for c in targets if c in agent.schema("train").names()]

    # overview + era structure
    overview_frames = {s: agent.scan(s, columns=["era", "id"]).collect() for s in splits}
    split_stats = analysis.describe_splits(overview_frames)
    _atomic_write_json(
        {
            "splits": {s: split_stats[s].__dict__ for s in splits},
            "feature_set": args.features,
            "n_features": len(feature_cols),
            "targets": all_targets,
            "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        },
        out / "overview.json",
    )
    _atomic_write_parquet(
        analysis.era_structure(pl.concat(list(overview_frames.values()))),
        out / "era_structure.parquet",
    )

    # target analysis on a small collected frame (era + targets only)
    target_frame = pl.concat(
        _era_chunks(agent, splits, ["era", *target_columns], args.max_eras)
    )
    _atomic_write_json(
        {
            t: analysis.target_profile(target_frame, [t]).row(0, named=True)
            for t in target_columns
        },
        out / "targets.json",
    )
    _atomic_write_parquet(
        analysis.target_correlation_matrix(target_frame, target_columns),
        out / "target_corr.parquet",
    )

    # feature analyses stream eras (bounded memory; never collect all features)
    feature_columns = ["era", *feature_cols, *target_columns]
    ic_by_era = analysis.feature_ic_by_era(
        _iter_era_chunks(agent, splits, feature_columns, args.max_eras),
        feature_cols,
        target_columns[0],
    )
    _atomic_write_parquet(ic_by_era, out / "feature_ic_by_era.parquet")
    screens = [
        analysis.feature_ic_screen(
            _iter_era_chunks(agent, splits, feature_columns, args.max_eras),
            feature_cols,
            [t],
        )
        for t in target_columns
    ]
    _atomic_write_parquet(
        pl.concat(screens), out / "feature_ic_screen.parquet"
    )
    _atomic_write_parquet(
        analysis.feature_summary(
            _iter_era_chunks(agent, splits, feature_columns, args.max_eras),
            feature_cols,
        ),
        out / "feature_summary.parquet",
    )

    # correlation structure: medium full matrix + selected-set summary
    medium_result = analysis.feature_correlation_structure(
        _iter_era_chunks(agent, splits, ["era", *medium_cols], args.max_eras),
        medium_cols,
    )
    _atomic_write_parquet(medium_result.top_pairs, out / "feature_corr_medium.parquet")
    selected_result = analysis.feature_correlation_structure(
        _iter_era_chunks(agent, splits, feature_columns, args.max_eras),
        feature_cols,
    )
    selected_summary = dict(selected_result.summary)
    selected_summary["top_pairs"] = selected_result.top_pairs.to_dicts()
    _atomic_write_json(selected_summary, out / "feature_corr_all_summary.json")
    if args.full_all_matrix:
        _atomic_write_parquet(
            pl.DataFrame(selected_result.matrix),
            out / "feature_corr_all_matrix.parquet",
        )

    _atomic_write_json(
        {
            "sets": {k: {"n_features": len(v)} for k, v in feature_sets.items()},
            "subset_relations": analysis.cross_set_membership(feature_sets)[
                "subset_relations"
            ].to_dicts(),
        },
        out / "set_membership.json",
    )

    # regimes (reuses the ic_by_era computed above)
    regimes = analysis.regime_analysis(ic_by_era)
    _atomic_write_json(
        {
            "regime_thresholds": regimes["regime_thresholds"],
            "crash_eras": regimes["crash_eras"],
            "hot_eras": regimes["hot_eras"],
            "ic_persistence": regimes["ic_persistence"],
        },
        out / "regimes.json",
    )
    _atomic_write_parquet(regimes["era_signal"], out / "era_signal.parquet")

    # benchmarks + meta model (validation coverage; empty list when absent)
    bench_rows: list[dict] = []
    sources: list[pl.LazyFrame] = []
    bench_path = version_dir / "validation_benchmark_models.parquet"
    meta_path = version_dir / "meta_model.parquet"
    if bench_path.exists():
        sources.append(pl.scan_parquet(bench_path))
    if meta_path.exists():
        sources.append(pl.scan_parquet(meta_path))
    if sources:
        target_side = agent.scan(
            "validation", columns=["era", "id", *target_columns]
        ).collect()
        bench_frame = (
            pl.concat([s.collect() for s in sources], how="align")
            .join(target_side, on=["era", "id"], how="inner")
        )
        excluded = {"era", "id", "data_type", *target_columns}
        bench_cols = [c for c in bench_frame.columns if c not in excluded]
        if bench_cols:
            bench_rows = analysis.benchmark_era_corr(
                bench_frame, bench_cols, target_columns[0]
            )["benchmarks"].to_dicts()
    _atomic_write_json({"benchmarks": bench_rows}, out / "benchmarks.json")

    # manifest (generated_at informational only — never hashed)
    era_csv = args.data_dir / "numerai_era_data.csv"
    refresh_date = None
    if era_csv.exists():
        try:
            df = pl.read_csv(era_csv, try_parse_dates=False)
            refresh_date = str(df["date"].max())
        except Exception:
            refresh_date = None
    _atomic_write_json(
        {
            "data_version": args.version,
            "feature_set": args.features,
            "feature_count": len(feature_cols),
            "target_count": len(target_columns),
            "era_ranges": {
                s: overview_frames[s]["era"].min() + ".." + overview_frames[s]["era"].max()
                for s in splits
            },
            "refresh_date": refresh_date,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        out / "manifest.json",
    )
    print(f"wrote analysis dumps to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
