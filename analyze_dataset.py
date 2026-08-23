"""Deterministic dataset analysis -> machine-readable dumps.

Thin control plane: wires ``nmr.analysis`` functions over train+validation
and writes JSON/parquet dumps under ``artifacts/reports/dataset_analysis/``
for the report renderer. Modular by stage: ``--only`` / ``--skip`` run a
subset (dependencies are auto-included), so a single metric can be computed
without re-running the pipeline, and new metrics can be added as new stages
later. Stage boundaries and per-era ticks print progress to stdout/stderr;
artifacts never contain wall-clock or progress state.
"""

# ruff: noqa: E402 — apply_thread_limits() must run before the imports below:
# polars/OpenMP/BLAS read their pool sizes at first use, not at import.
from __future__ import annotations

from nmr.hardware import apply_thread_limits

apply_thread_limits()

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from nmr import analysis
from nmr._atomicio import atomic_write_text
from nmr.config import DataConfig
from nmr.data import IngestionAgent
from nmr.features import resolve_feature_sets
from nmr.hardware import discover_hardware
from nmr.refresh import CURRENT_DATA_VERSION

_ERA_BATCH = 40  # eras scanned per lazy pass (bounded transient memory)

# Continuous FNE grid: neutralization proportions profiled for every signal.
_FNE_GRID = tuple(round(i / 10, 1) for i in range(11))

# Stage registry: name -> (description, dependencies). Dependencies are
# auto-included when a stage is selected, so `--only regimes` also computes
# `ic_by_era`. Order is the canonical execution order.
_STAGE_DEPS: dict[str, frozenset[str]] = {
    "overview": frozenset(),
    "targets": frozenset(),
    "ic_by_era": frozenset(),
    "screens": frozenset(),
    "screens_train": frozenset(),
    "summary": frozenset(),
    "psi": frozenset(),
    "drift": frozenset(),
    "derived_sets": frozenset(),
    "corr_medium": frozenset(),
    "corr_all": frozenset(),
    "set_membership": frozenset(),
    "ic_by_split": frozenset({"ic_by_era", "overview"}),
    "regimes": frozenset({"ic_by_era"}),
    "benchmarks": frozenset(),
    "meta_ortho": frozenset(),
    "manifest": frozenset({"overview"}),
}
_STAGE_ORDER = tuple(_STAGE_DEPS)


@dataclass
class _Ctx:
    """Shared per-run state handed to every stage."""

    args: argparse.Namespace
    out: Path
    feature_sets: dict[str, list[str]]
    all_targets: list[str]
    feature_cols: list[str]
    medium_cols: list[str]
    targets: list[str]
    target_columns: list[str]
    agent: IngestionAgent
    splits: tuple[str, ...]
    split_stats: dict[str, object]
    ic_by_era: pl.DataFrame | None = field(default=None)


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
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated stages to run (dependencies auto-included); "
        "e.g. --only targets,regimes",
    )
    parser.add_argument(
        "--skip",
        default=None,
        help="comma-separated stages to skip (a skipped stage is re-included "
        "if another selected stage depends on it); e.g. --skip corr_all",
    )
    return parser.parse_args(argv)


def _resolve_stages(only: str | None, skip: str | None) -> list[str]:
    """Resolve the requested stage list in canonical order.

    ``--only`` wins over ``--skip``; unknown names raise; dependencies of any
    selected stage are auto-included; ``manifest`` always runs last.
    """
    if only is not None and skip is not None:
        raise ValueError("use either --only or --skip, not both")
    if only is not None:
        names = [s.strip() for s in only.split(",") if s.strip()]
    elif skip is not None:
        skipped = {s.strip() for s in skip.split(",") if s.strip()}
        names = [s for s in _STAGE_ORDER if s not in skipped]
    else:
        names = list(_STAGE_ORDER)
    unknown = set(names) - set(_STAGE_ORDER)
    if unknown:
        raise ValueError(
            f"unknown stage(s): {sorted(unknown)}; valid stages: {list(_STAGE_ORDER)}"
        )
    selected = set(names)
    selected.add("manifest")  # manifest always runs; its deps are resolved below
    changed = True
    while changed:
        changed = False
        for stage in list(selected):
            for dep in _STAGE_DEPS[stage]:
                if dep not in selected:
                    selected.add(dep)
                    changed = True
    return [s for s in _STAGE_ORDER if s in selected]


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
    label: str | None = None,
) -> Iterator[pl.DataFrame]:
    """Yield one era at a time from batched lazy scans (bounded memory).

    The full feature universe (v5.3 ``all`` = 3555 columns) cannot be collected
    as one frame (~100 GB); this streams era batches via predicate pushdown,
    so each analysis holds at most one era plus accumulators. Deterministic:
    eras are yielded in ascending integer order. With ``label``, a progress
    tick prints to stderr every 100 eras (console only, never artifacts).
    """
    era_sets: list[set[str]] = []
    for split in splits:
        labels = agent.scan(split, columns=["era"]).collect().get_column("era").to_list()
        era_sets.append(set(labels))
    all_eras = sorted(set().union(*era_sets), key=int)
    if max_eras is not None:
        all_eras = [e for e in all_eras if int(e) <= max_eras]
    total = len(all_eras)

    processed = 0
    for start in range(0, total, _ERA_BATCH):
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
        for chunk in combined.partition_by("era", maintain_order=True):
            processed += 1
            if label is not None and (
                processed == 1 or processed % 100 == 0 or processed == total
            ):
                print(
                    f"  [{label}] era {processed}/{total}",
                    file=sys.stderr,
                    flush=True,
                )
            yield chunk


def _resolve_reference_targets(
    all_targets: list[str], explicit: list[str] | None, all_targets_flag: bool
) -> list[str]:
    if explicit:
        return explicit
    if all_targets_flag:
        return all_targets
    if "target" not in all_targets:
        raise ValueError(
            "primary target 'target' not in features.json targets; "
            "pass --targets explicitly"
        )
    primary_60 = next(
        (t for t in all_targets if t.endswith("_60") and t != "target"), None
    )
    if primary_60 is None:
        raise ValueError(
            "no distinct 60-day target (name ending '_60') in features.json "
            "targets; pass --targets explicitly"
        )
    return ["target", primary_60]


def _stage_overview(ctx: _Ctx) -> None:
    overview_frames = {
        s: ctx.agent.scan(s, columns=["era", "id"]).collect() for s in ctx.splits
    }
    ctx.split_stats = analysis.describe_splits(overview_frames)
    _atomic_write_json(
        {
            "splits": {s: ctx.split_stats[s].__dict__ for s in ctx.splits},
            "feature_set": ctx.args.features,
            "n_features": len(ctx.feature_cols),
            "targets": ctx.all_targets,
            "feature_sets": {k: len(v) for k, v in ctx.feature_sets.items()},
        },
        ctx.out / "overview.json",
    )
    _atomic_write_parquet(
        analysis.era_structure(pl.concat(list(overview_frames.values()))),
        ctx.out / "era_structure.parquet",
    )


def _stage_targets(ctx: _Ctx) -> None:
    target_frame = pl.concat(
        _era_chunks(
            ctx.agent, ctx.splits, ["era", *ctx.target_columns], ctx.args.max_eras
        )
    )
    _atomic_write_json(
        {
            t: analysis.target_profile(target_frame, [t]).row(0, named=True)
            for t in ctx.target_columns
        },
        ctx.out / "targets.json",
    )
    _atomic_write_parquet(
        analysis.target_correlation_matrix(target_frame, ctx.target_columns),
        ctx.out / "target_corr.parquet",
    )


def _stage_ic_by_era(ctx: _Ctx) -> None:
    feature_columns = ["era", *ctx.feature_cols, *ctx.target_columns]
    ctx.ic_by_era = analysis.feature_ic_by_era(
        _iter_era_chunks(
            ctx.agent, ctx.splits, feature_columns, ctx.args.max_eras, label="ic_by_era"
        ),
        ctx.feature_cols,
        ctx.target_columns[0],
    )
    _atomic_write_parquet(ctx.ic_by_era, ctx.out / "feature_ic_by_era.parquet")


def _stage_screens(ctx: _Ctx) -> None:
    feature_columns = ["era", *ctx.feature_cols, *ctx.target_columns]
    screens = [
        analysis.feature_ic_screen(
            _iter_era_chunks(
                ctx.agent,
                ctx.splits,
                feature_columns,
                ctx.args.max_eras,
                label=f"screen:{t}",
            ),
            ctx.feature_cols,
            [t],
        )
        for t in ctx.target_columns
    ]
    _atomic_write_parquet(
        pl.concat(screens), ctx.out / "feature_ic_screen.parquet"
    )


def _stage_screens_train(ctx: _Ctx) -> None:
    """Train-only stability screen — the sole input to subset derivation.

    Descriptive full-span characterization lives in ``screens``
    (``feature_ic_screen.parquet``, eras 0001..1231); this stage screens the
    train split only (eras 0001..0574) so feature-subset selection can never
    see validation-era labels (look-ahead leakage, committee Red Flag 1).
    """
    feature_columns = ["era", *ctx.feature_cols, *ctx.target_columns]
    screens = [
        analysis.feature_ic_screen(
            _iter_era_chunks(
                ctx.agent,
                ["train"],
                feature_columns,
                ctx.args.max_eras,
                label=f"screen_train:{t}",
            ),
            ctx.feature_cols,
            [t],
        )
        for t in ctx.target_columns
    ]
    _atomic_write_parquet(
        pl.concat(screens), ctx.out / "feature_ic_screen_train.parquet"
    )


def _stage_summary(ctx: _Ctx) -> None:
    feature_columns = ["era", *ctx.feature_cols, *ctx.target_columns]
    _atomic_write_parquet(
        analysis.feature_summary(
            _iter_era_chunks(
                ctx.agent, ctx.splits, feature_columns, ctx.args.max_eras, label="summary"
            ),
            ctx.feature_cols,
        ),
        ctx.out / "feature_summary.parquet",
    )


def _stage_psi(ctx: _Ctx) -> None:
    _atomic_write_parquet(
        analysis.feature_drift_psi(
            _iter_era_chunks(
                ctx.agent, ["train"], ["era", *ctx.medium_cols], ctx.args.max_eras,
                label="psi:train",
            ),
            _iter_era_chunks(
                ctx.agent, ["validation"], ["era", *ctx.medium_cols], ctx.args.max_eras,
                label="psi:val",
            ),
            ctx.medium_cols,
        ),
        ctx.out / "feature_drift_psi.parquet",
    )


def _stage_drift(ctx: _Ctx) -> None:
    """PSI + Wasserstein W1 + adversarial AUC in one sample pass."""
    _atomic_write_parquet(
        analysis.feature_drift_profile(
            _iter_era_chunks(
                ctx.agent, ["train"], ["era", *ctx.medium_cols], ctx.args.max_eras,
                label="drift:train",
            ),
            _iter_era_chunks(
                ctx.agent, ["validation"], ["era", *ctx.medium_cols], ctx.args.max_eras,
                label="drift:val",
            ),
            ctx.medium_cols,
        ),
        ctx.out / "feature_drift_profile.parquet",
    )


def _stage_derived_sets(ctx: _Ctx) -> None:
    """Write screen-derived feature sets for campaign configs.

    Pure function of the stage outputs: reads
    ``feature_ic_screen_train.parquet`` (the train-only screen — subset
    derivation must never see validation-era labels) and
    ``feature_drift_profile.parquet`` from the output dir and writes
    ``derived_feature_sets.json`` (shape ``{"feature_sets": {...}}``, sorted
    lists, always all four keys):
      - ``screen_stable``: stable features (primary reference target)
      - ``screen_nonlinear``: unstable but |Spearman| > threshold
      - ``screen_linear_or_nonlinear``: stable OR nonlinear
      - ``screen_drift_filtered``: linear-or-nonlinear minus drift-flagged
    Drift flags only exist for medium-universe features; features without a
    drift row are kept (no evidence of drift). Primary reference target is
    ``target`` when present, else the first distinct target in the screen.
    Empty sets are valid scientific results (an empty ``screen_stable`` means
    the full-gate screen found nothing); training on them fails loudly at
    ingestion (``IngestionAgent.features``).
    """
    screen_path = ctx.out / "feature_ic_screen_train.parquet"
    drift_path = ctx.out / "feature_drift_profile.parquet"
    if not screen_path.exists():
        raise RuntimeError(
            f"{screen_path} not found — run the 'screens_train' stage first "
            "(e.g. --only screens_train,drift,derived_sets)"
        )
    if not drift_path.exists():
        raise RuntimeError(
            f"{drift_path} not found — run the 'drift' stage first "
            "(e.g. --only screens_train,drift,derived_sets)"
        )
    screen = pl.read_parquet(screen_path)
    if "target" not in screen.columns:
        raise RuntimeError(f"{screen_path}: missing 'target' column")
    distinct_targets = screen["target"].unique().to_list()
    primary = "target" if "target" in distinct_targets else distinct_targets[0]
    rows = screen.filter(pl.col("target") == primary)
    stable = sorted(rows.filter(pl.col("stable")).get_column("feature").to_list())
    nonlinear = sorted(
        rows.filter((~pl.col("stable")) & pl.col("nonlinear"))
        .get_column("feature")
        .to_list()
    )
    lin_or_non = sorted(set(stable) | set(nonlinear))
    drifted = set(
        pl.read_parquet(drift_path)
        .filter(pl.col("drifted"))
        .get_column("feature")
        .to_list()
    )
    _atomic_write_json(
        {
            "feature_sets": {
                "screen_stable": stable,
                "screen_nonlinear": nonlinear,
                "screen_linear_or_nonlinear": lin_or_non,
                "screen_drift_filtered": [f for f in lin_or_non if f not in drifted],
            }
        },
        ctx.out / "derived_feature_sets.json",
    )


def _stage_corr_medium(ctx: _Ctx) -> None:
    medium_result = analysis.feature_correlation_structure(
        _iter_era_chunks(
            ctx.agent, ctx.splits, ["era", *ctx.medium_cols], ctx.args.max_eras,
            label="corr_medium",
        ),
        ctx.medium_cols,
    )
    _atomic_write_parquet(medium_result.top_pairs, ctx.out / "feature_corr_medium.parquet")
    # full medium matrix (780 x 780 float32, ~2.4 MB) for downstream
    # covariance work; top-100 pairs stay in feature_corr_medium.parquet
    _atomic_write_parquet(
        pl.DataFrame(
            medium_result.matrix,
            schema={f: pl.Float32 for f in medium_result.feature_order},
        ),
        ctx.out / "feature_corr_medium_matrix.parquet",
    )
    # per-feature-set redundancy, indexed from the medium correlation matrix
    _atomic_write_json(
        analysis.within_set_redundancy(medium_result, ctx.feature_sets).to_dicts(),
        ctx.out / "feature_set_redundancy.json",
    )


def _stage_corr_all(ctx: _Ctx) -> None:
    feature_columns = ["era", *ctx.feature_cols, *ctx.target_columns]
    selected_result = analysis.feature_correlation_structure(
        _iter_era_chunks(
            ctx.agent, ctx.splits, feature_columns, ctx.args.max_eras, label="corr_all"
        ),
        ctx.feature_cols,
    )
    selected_summary = dict(selected_result.summary)
    selected_summary["top_pairs"] = selected_result.top_pairs.to_dicts()
    _atomic_write_json(selected_summary, ctx.out / "feature_corr_all_summary.json")
    if ctx.args.full_all_matrix:
        _atomic_write_parquet(
            pl.DataFrame(selected_result.matrix),
            ctx.out / "feature_corr_all_matrix.parquet",
        )


def _stage_set_membership(ctx: _Ctx) -> None:
    _atomic_write_json(
        {
            "sets": {k: {"n_features": len(v)} for k, v in ctx.feature_sets.items()},
            "subset_relations": analysis.cross_set_membership(ctx.feature_sets)[
                "subset_relations"
            ].to_dicts(),
        },
        ctx.out / "set_membership.json",
    )


def _stage_ic_by_split(ctx: _Ctx) -> None:
    if ctx.ic_by_era is None:
        raise RuntimeError("internal error: ic_by_era stage must run first")
    _atomic_write_parquet(
        analysis.feature_ic_by_split(
            ctx.ic_by_era,
            train_max_era=int(ctx.split_stats["train"].max_era),
            val_min_era=int(ctx.split_stats["validation"].min_era),
        ),
        ctx.out / "feature_ic_by_split.parquet",
    )


def _stage_regimes(ctx: _Ctx) -> None:
    if ctx.ic_by_era is None:
        raise RuntimeError("internal error: ic_by_era stage must run first")
    regimes = analysis.regime_analysis(ctx.ic_by_era)
    _atomic_write_json(
        {
            "regime_thresholds": regimes["regime_thresholds"],
            "crash_eras": regimes["crash_eras"],
            "hot_eras": regimes["hot_eras"],
            "ic_persistence": regimes["ic_persistence"],
        },
        ctx.out / "regimes.json",
    )
    _atomic_write_parquet(regimes["era_signal"], ctx.out / "era_signal.parquet")


def _stage_benchmarks(ctx: _Ctx) -> None:
    bench_rows: list[dict] = []
    sources: list[pl.LazyFrame] = []
    version_dir = ctx.args.data_dir / ctx.args.version
    bench_path = version_dir / "validation_benchmark_models.parquet"
    meta_path = version_dir / "meta_model.parquet"
    if bench_path.exists():
        sources.append(pl.scan_parquet(bench_path))
    if meta_path.exists():
        sources.append(pl.scan_parquet(meta_path))
    if sources:
        target_side = ctx.agent.scan(
            "validation", columns=["era", "id", *ctx.target_columns]
        ).collect()
        bench_frame = (
            pl.concat([s.collect() for s in sources], how="align")
            .join(target_side, on=["era", "id"], how="inner")
        )
        excluded = {"era", "id", "data_type", *ctx.target_columns}
        bench_cols = [c for c in bench_frame.columns if c not in excluded]
        if bench_cols:
            bench_rows = analysis.benchmark_era_corr(
                bench_frame, bench_cols, ctx.target_columns[0]
            )["benchmarks"].to_dicts()
            # FNE profile: join validation medium features, then neutralize
            # each benchmark signal against the medium universe per era.
            bench_era_labels = list(bench_frame["era"].unique())
            bench_feats = bench_frame.join(
                ctx.agent.scan(
                    "validation", columns=["era", "id", *ctx.medium_cols]
                )
                .filter(pl.col("era").is_in(bench_era_labels))
                .collect(),
                on=["era", "id"],
                how="inner",
            )
            fne = analysis.neutralized_ic_profile(
                bench_feats.partition_by("era", maintain_order=True),
                bench_cols,
                ctx.medium_cols,
                ctx.target_columns[0],
                proportions=_FNE_GRID,
            )
            _atomic_write_json(
                {"profile": fne.to_dicts()}, ctx.out / "neutralized_ic.json"
            )
    _atomic_write_json({"benchmarks": bench_rows}, ctx.out / "benchmarks.json")


def _stage_meta_ortho(ctx: _Ctx) -> None:
    """Per-feature correlation vs the meta model and the target (the eras
    where the meta model exists), flagging consensus-orthogonal signal."""
    meta_path = ctx.args.data_dir / ctx.args.version / "meta_model.parquet"
    blocks: list[pl.DataFrame] = []
    if meta_path.exists():
        meta_frame = pl.scan_parquet(meta_path).collect()
        meta_cols = [
            c for c in meta_frame.columns if c not in {"era", "id", "data_type"}
        ]
        if meta_cols:
            target_side = ctx.agent.scan(
                "validation", columns=["era", "id", *ctx.target_columns]
            ).collect()
            joined = meta_frame.join(target_side, on=["era", "id"], how="inner")
            meta_eras = list(joined["era"].unique())
            feats = (
                ctx.agent.scan(
                    "validation", columns=["era", "id", *ctx.medium_cols]
                )
                .filter(pl.col("era").is_in(meta_eras))
                .collect()
            )
            joined = joined.join(feats, on=["era", "id"], how="inner")
            for meta_col in meta_cols:
                block = analysis.meta_orthogonality(
                    joined.partition_by("era", maintain_order=True),
                    ctx.medium_cols,
                    meta_col,
                    ctx.target_columns[0],
                )
                blocks.append(block.with_columns(pl.lit(meta_col).alias("meta")))
    if blocks:
        frame = pl.concat(blocks)
    else:
        frame = pl.DataFrame(
            schema={
                "feature": pl.Utf8,
                "meta": pl.Utf8,
                "corr_meta": pl.Float64,
                "corr_target": pl.Float64,
                "n_eras": pl.Int64,
                "orthogonal": pl.Boolean,
            }
        )
    _atomic_write_parquet(frame, ctx.out / "meta_ortho.parquet")


def _stage_manifest(ctx: _Ctx, stages_run: list[str]) -> None:
    # generated_at informational only — never hashed
    era_csv = ctx.args.data_dir / "numerai_era_data.csv"
    refresh_date = None
    if era_csv.exists():
        try:
            df = pl.read_csv(era_csv, try_parse_dates=False)
            refresh_date = str(df["date"].max())
        except Exception:
            refresh_date = None
    hardware = discover_hardware()
    _atomic_write_json(
        {
            "data_version": ctx.args.version,
            "feature_set": ctx.args.features,
            "feature_count": len(ctx.feature_cols),
            "target_count": len(ctx.target_columns),
            "era_ranges": {
                s: ctx.split_stats[s].min_era + ".." + ctx.split_stats[s].max_era
                for s in ctx.splits
            },
            "refresh_date": refresh_date,
            "stages_run": stages_run,
            "hardware": {
                "os": hardware.os_name,
                "cpu_logical_cores": hardware.cpu_logical_cores,
                "ram_total_gib": round(hardware.ram_total_gib, 1),
                "gpus": [
                    {"name": g.name, "memory_total_mib": g.memory_total_mib}
                    for g in hardware.gpus
                ],
            },
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        ctx.out / "manifest.json",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        stages = _resolve_stages(args.only, args.skip)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

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

    ctx = _Ctx(
        args=args,
        out=out,
        feature_sets=feature_sets,
        all_targets=all_targets,
        feature_cols=feature_cols,
        medium_cols=medium_cols,
        targets=targets,
        target_columns=target_columns,
        agent=agent,
        splits=splits,
        split_stats={},
    )

    stage_funcs = {
        "overview": _stage_overview,
        "targets": _stage_targets,
        "ic_by_era": _stage_ic_by_era,
        "screens": _stage_screens,
        "screens_train": _stage_screens_train,
        "summary": _stage_summary,
        "psi": _stage_psi,
        "drift": _stage_drift,
        "derived_sets": _stage_derived_sets,
        "corr_medium": _stage_corr_medium,
        "corr_all": _stage_corr_all,
        "set_membership": _stage_set_membership,
        "ic_by_split": _stage_ic_by_split,
        "regimes": _stage_regimes,
        "benchmarks": _stage_benchmarks,
        "meta_ortho": _stage_meta_ortho,
    }
    n_stages = len(stages)
    for idx, name in enumerate(stages, start=1):
        started = time.monotonic()
        print(f"[stage {idx}/{n_stages}] {name} ...", flush=True)
        if name == "manifest":
            _stage_manifest(ctx, stages)
        else:
            stage_funcs[name](ctx)
        elapsed = time.monotonic() - started
        print(f"[stage {idx}/{n_stages}] {name} done ({elapsed:.1f}s)", flush=True)
    print(f"wrote analysis dumps to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
