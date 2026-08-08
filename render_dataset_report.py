"""LLM-optimized dataset analysis report renderer.

Reads the dumps from analyze_dataset.py and renders a dense, schema-annotated
Markdown report under docs/04-research/. Pure formatting: every number comes
from the dumps; the same dumps produce byte-identical Markdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from nmr.refresh import CURRENT_DATA_VERSION


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(columns: list[str], rows: list[dict]) -> str:
    """Dense pipe table; pipes inside cell values are escaped."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for c in columns:
            value = str(_fmt(row.get(c))).replace("|", "\\|")
            cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _schema_block(text: str) -> str:
    return f"**Schema:** {text}"


def render_report(
    manifest: dict,
    overview: dict,
    era_structure_rows: list[dict],
    targets: dict,
    target_corr_rows: list[dict],
    feature_summary_rows: list[dict],
    ic_screen_rows: list[dict],
    regime: dict,
    era_signal_rows: list[dict],
    benchmark_rows: list[dict],
    corr_summary: dict,
    set_membership: dict,
) -> str:
    """Render the full report. Deterministic given identical inputs."""
    out: list[str] = []
    out.append("# Dataset Analysis — Numerai " + manifest["data_version"])
    out.append("")
    out.append("> Generated from `artifacts/reports/dataset_analysis/` dumps. "
               "All numbers have full precision in the dumps; tables are display-rounded. "
               "Schema lines precede every table.")
    out.append("")
    out.append(f"- Data version: `{manifest['data_version']}`")
    out.append(f"- Feature set: `{manifest['feature_set']}` ({manifest['feature_count']} features)")
    out.append(f"- Refresh date: `{manifest.get('refresh_date')}`")
    out.append(f"- Era ranges: train `{manifest['era_ranges'].get('train')}`, "
               f"validation `{manifest['era_ranges'].get('validation')}`")
    out.append("")

    out.append("## 1. Dataset Overview")
    out.append("")
    out.append(_schema_block("split | n_rows | n_eras | min_era | max_era"))
    out.append("")
    out.append(_table(
        ["split", "n_rows", "n_eras", "min_era", "max_era"],
        [{"split": k, **v} for k, v in overview["splits"].items()],
    ))
    out.append("")
    out.append(_schema_block("feature_set | n_features"))
    out.append("")
    out.append(_table(
        ["feature_set", "n_features"],
        [{"feature_set": k, "n_features": v} for k, v in overview.get("feature_sets", {}).items()],
    ))
    out.append("")
    out.append("- **Key takeaways:** the tournament is a per-era cross-section of obfuscated "
               "equities; eras are the unit of evaluation. Never pool rows across eras for "
               "metrics.")
    out.append("")

    out.append("## 2. Era Structure")
    out.append("")
    out.append(_schema_block("era | era_index | n_rows | n_ids | gap (non-consecutive era label)"))
    out.append("")
    out.append(_table(["era", "era_index", "n_rows", "n_ids", "gap"], era_structure_rows))
    out.append("")
    out.append("- **Key takeaways:** gaps in the era index are data anomalies; the "
               "train→validation boundary is a distribution-shift checkpoint.")
    out.append("")

    out.append("## 3. Targets")
    out.append("")
    out.append(_schema_block("target | n_eras_present | missing_rate | pooled_mean | pooled_std"))
    out.append("")
    out.append(_table(
        ["target", "n_eras_present", "missing_rate", "pooled_mean", "pooled_std"],
        [{"target": k, **v} for k, v in targets.items()],
    ))
    out.append("")
    out.append(_schema_block("target_a | target_b | mean_corr | n_eras (per-era Spearman, equal-era-weighted)"))
    out.append("")
    out.append(_table(["target_a", "target_b", "mean_corr", "n_eras"], target_corr_rows))
    out.append("")
    out.append("- **Key takeaways:** targets are integer ranks 0..5; auxiliary targets have "
               "staggered era availability — check `n_eras_present` before training on them.")
    out.append("")

    out.append("## 4. Features")
    out.append("")
    out.append(_schema_block("feature | pooled_mean | pooled_std | missing_rate"))
    out.append("")
    out.append(_table(
        ["feature", "pooled_mean", "pooled_std", "missing_rate"], feature_summary_rows
    ))
    out.append("")
    out.append(_schema_block("feature | target | mean_corr | n_eras | stable (per-era Pearson IC)"))
    out.append("")
    out.append(_table(
        ["feature", "target", "mean_corr", "n_eras", "stable"], ic_screen_rows
    ))
    out.append("")
    out.append(f"- Mean |pairwise corr|: `{_fmt(corr_summary.get('mean_abs_corr'))}`; "
               "top pairs in `feature_corr_medium.parquet` / `feature_corr_all_summary.json`.")
    out.append("- **Key takeaways:** prefer features passing the stability screen "
               "(`stable=True`); avoid highly redundant families.")
    out.append("")

    out.append("## 5. Regimes & Signal Dynamics")
    out.append("")
    out.append(_schema_block("era | mean_ic | regime | crash | hot"))
    out.append("")
    out.append(_table(["era", "mean_ic", "regime", "crash", "hot"], era_signal_rows))
    out.append("")
    out.append(f"- Crash eras (bottom decile): `{regime.get('crash_eras')}`")
    out.append(f"- Hot eras (top decile): `{regime.get('hot_eras')}`")
    out.append(f"- Adjacent-era IC persistence: mean "
               f"`{_fmt(regime.get('ic_persistence', {}).get('mean'))}`, "
               f"n `{regime.get('ic_persistence', {}).get('n_adjacent')}`")
    out.append("- **Key takeaways:** signal is regime-dependent; expect IC mean-reversion — "
               "never tune on crash eras in-sample.")
    out.append("")

    out.append("## 6. Benchmarks & Meta-Model")
    out.append("")
    out.append(_schema_block("benchmark | mean_corr | n_eras"))
    out.append("")
    out.append(_table(["benchmark", "mean_corr", "n_eras"], benchmark_rows))
    out.append("")
    out.append("- **Key takeaways:** benchmark models define the achievable floor; the meta "
               "model is the upper reference. Stay above the best benchmark after neutralization.")
    out.append("")

    out.append("## 7. Modeling Implications")
    out.append("")
    out.append("- Validate **era-grouped with purge** (8 eras for 20D, 16 for 60D); "
               "random row-level CV is leakage.")
    out.append("- Rank-gaussianize per era before ensembling; never blend raw outputs.")
    out.append("- Select features from the stability screen, not pooled correlation.")
    out.append("- Watch auxiliary-target era coverage before including them.")
    out.append("")
    return "\n".join(out)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the dataset analysis report.")
    parser.add_argument(
        "--dumps-dir",
        type=Path,
        default=Path("artifacts") / "reports" / "dataset_analysis",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    d = args.dumps_dir

    manifest = _load_json(d / "manifest.json")
    if manifest["data_version"] != CURRENT_DATA_VERSION:
        print(
            f"ERROR: dumps data_version {manifest['data_version']} != "
            f"CURRENT_DATA_VERSION {CURRENT_DATA_VERSION}",
            file=sys.stderr,
        )
        return 1
    required_dumps = (
        "overview.json", "era_structure.parquet", "targets.json",
        "target_corr.parquet", "feature_summary.parquet",
        "feature_ic_screen.parquet", "regimes.json", "era_signal.parquet",
        "benchmarks.json", "feature_corr_all_summary.json", "set_membership.json",
    )
    for name in required_dumps:
        if not (d / name).exists():
            print(f"ERROR: missing dump {d / name}", file=sys.stderr)
            return 1

    md = render_report(
        manifest=manifest,
        overview=_load_json(d / "overview.json"),
        era_structure_rows=pl.read_parquet(d / "era_structure.parquet").to_dicts(),
        targets=_load_json(d / "targets.json"),
        target_corr_rows=pl.read_parquet(d / "target_corr.parquet").to_dicts(),
        feature_summary_rows=pl.read_parquet(d / "feature_summary.parquet").to_dicts(),
        ic_screen_rows=pl.read_parquet(d / "feature_ic_screen.parquet").to_dicts(),
        regime=_load_json(d / "regimes.json"),
        era_signal_rows=pl.read_parquet(d / "era_signal.parquet").to_dicts(),
        benchmark_rows=_load_json(d / "benchmarks.json").get("benchmarks", []),
        corr_summary=_load_json(d / "feature_corr_all_summary.json"),
        set_membership=_load_json(d / "set_membership.json"),
    )
    refresh_date = manifest.get("refresh_date") or "0000-00"
    output = args.output or (
        Path("docs") / "04-research" / f"dataset-analysis-{refresh_date[:7]}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
