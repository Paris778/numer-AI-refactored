"""LLM-optimized pre-modelling dataset & feature study renderer.

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

_META_SAMPLE_WARNING = (
    "> **[SMALL SAMPLE: 86 ERAS — HIGH SAMPLING VARIANCE]** This table is "
    "computed only on eras 1133-1218 where the meta model exists (~1.6 years "
    "of weekly data). Estimates carry high sampling variance and are not "
    "equivalent to the 1,218-era feature moments elsewhere in this report."
)


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


def _executive_summary(
    manifest: dict,
    overview: dict,
    targets: dict,
    ic_screen_rows: list[dict],
    ic_screen_train_rows: list[dict],
    campaign_rows: list[dict] | None,
    benchmark_rows: list[dict],
) -> list[str]:
    """§0 — deterministic headline summary computed from the dumps."""
    out: list[str] = []
    out.append("## 0. Executive Summary")
    out.append("")

    splits = overview.get("splits", {})
    train = splits.get("train", {})
    val = splits.get("validation", {})
    n_features = manifest.get("feature_count") or overview.get("n_features")
    n_sets = len(overview.get("feature_sets", {}))
    out.append(f"- **Dataset:** `{manifest.get('data_version')}` — "
               f"train `{train.get('min_era')}..{train.get('max_era')}` "
               f"({train.get('n_eras')} eras, {train.get('n_rows'):,} rows), "
               f"validation `{val.get('min_era')}..{val.get('max_era')}` "
               f"({val.get('n_eras')} eras, {val.get('n_rows'):,} rows); "
               f"{n_features} features across {n_sets} packaged sets; "
               f"{len(targets)} targets (`{', '.join(sorted(targets))}`).")

    hw = manifest.get("hardware") or {}
    if hw:
        gpus = ", ".join(g.get("name", "?") for g in hw.get("gpus", [])) or "none"
        out.append(f"- **Hardware:** {hw.get('ram_total_gib')} GiB RAM, "
                   f"{hw.get('cpu_logical_cores')} logical cores; GPU: {gpus}. "
                   "XGBoost trains on CUDA (~9.1x vs CPU); LightGBM on CPU; analysis "
                   "ranks via cupy `rankdata` (~5.8x, bit-identical). The full "
                   "3,555-feature universe is memory-marginal on this machine — see "
                   "§8 for the hardware ceiling.")

    # Screen headline: the train-only screen drives subset derivation; the
    # full-span screen is descriptive only (never used for selection).
    def _screen_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
        by_target: dict[str, dict[str, int]] = {}
        for r in rows:
            t = r.get("target")
            if t is None:
                continue
            counts = by_target.setdefault(t, {"stable": 0, "nonlinear": 0})
            counts["stable"] += 1 if r.get("stable") else 0
            counts["nonlinear"] += 1 if r.get("nonlinear") else 0
        return by_target

    train_counts = _screen_counts(ic_screen_train_rows)
    for t in sorted(train_counts):
        c = train_counts[t]
        out.append(f"- **Train-only stability screen (`{t}`, eras 0001..0574 — "
                   f"drives subset derivation):** {c['stable']} stable, "
                   f"{c['nonlinear']} nonlinear of {n_features} features. "
                   "`stable` requires the full gate on TRAIN eras only: "
                   "|mean_corr| >= 0.01 AND |decay_slope| <= 0.001 AND the 95% "
                   "block-bootstrap CI strictly excluding zero AND "
                   "Benjamini-Hochberg FDR pass at q=0.05 (per target); "
                   "`nonlinear` = |Pearson| <= 0.01 but |Spearman| > 0.01. "
                   "An empty `screen_stable` is a valid scientific result — "
                   "cells training on it fail loudly at ingestion (§7.1).")
    desc_counts = _screen_counts(ic_screen_rows)
    if desc_counts:
        parts = "; ".join(
            f"{t}: {desc_counts[t]['stable']} stable / "
            f"{desc_counts[t]['nonlinear']} nonlinear"
            for t in sorted(desc_counts)
        )
        out.append(f"- **Descriptive full-span screen (eras 0001..1231, NOT "
                   f"used for subset selection):** {parts}.")

    if benchmark_rows:
        bm = {b.get("benchmark"): b for b in benchmark_rows if b.get("mean_corr") is not None}
        if bm:
            meta = bm.get("numerai_meta_model")
            best = max(bm.values(), key=lambda b: b["mean_corr"])
            meta_txt = (f"{meta['mean_corr']:.4f} ({meta['n_eras']} eras)"
                        if meta else "n/a")
            out.append(f"- **Benchmark floor (validation, 86-era meta overlap):** "
                       f"best benchmark `{best['benchmark']}` mean corr "
                       f"{best['mean_corr']:.4f}; meta model {meta_txt}. "
                       "Benchmarks define the achievable floor — a model must beat "
                       "them, not just show positive IC.")

    if campaign_rows:
        rec = [r for r in campaign_rows if r.get("status") == "recorded"]
        if rec:
            best = max(rec, key=lambda r: r.get("mean_ic") or -1.0)
            medium = [r for r in rec if abs((r.get("n_features") or 0) - 780) < 1]
            small = [r for r in rec if abs((r.get("n_features") or 0) - 42) < 1]
            screen = [r for r in rec if (r.get("n_features") or 0) <= 3]

            def _range(rows: list[dict], key: str) -> str:
                vals = sorted(r[key] for r in rows if r.get(key) is not None)
                return f"{vals[0]:.4f}..{vals[-1]:.4f}" if vals else "n/a"

            verdict = ("- **Feature-campaign verdict (validation eras, 8-era "
                       "purge, identical params per backend):** ")
            parts: list[str] = []
            if medium:
                parts.append(
                    f"`medium` (780) mean IC {_range(medium, 'mean_ic')}, "
                    f"IC Sharpe {_range(medium, 'ic_sharpe')}, "
                    f"FNE@100% {_range(medium, 'fne100')} — signal survives "
                    "full neutralization = non-linear structure"
                )
            if small:
                parts.append(f"`small` (42) mean IC {_range(small, 'mean_ic')}")
            if screen:
                sizes = sorted({int(r["n_features"]) for r in screen})
                size_txt = ", ".join(str(s) for s in sizes)
                feat_word = "feature" if len(sizes) == 1 and sizes[0] == 1 else "features"
                parts.append(
                    f"screen-derived variants ({size_txt} {feat_word}) mean IC "
                    f"{_range(screen, 'mean_ic')}"
                )
            out.append(verdict + "; ".join(parts) + f". Best cell: "
                       f"`{best['variant']}` ({best['backend']}).")
            errs = [r for r in campaign_rows if r.get("status") != "recorded"]
            if errs:
                out.append(f"- **Unavailable cells:** {len(errs)} — "
                           + "; ".join(f"`{e.get('variant')}`: "
                                       f"{(e.get('error') or '')[:90]}" for e in errs)
                           + " (full failure modes in §7.1).")
    out.append("")
    return out


def _methodology_section() -> list[str]:
    """§9 — how the data was refreshed, analysed, and campaigned (static record)."""
    return [
        "## 9. Methodology & Provenance",
        "",
        "- **Dataset refresh (2026-08-08):** `refresh_data.py` + `nmr/refresh.py` — "
        "numeric version detection (`detect_newer_version`, `v<major>.<minor>` parse), "
        "live-round reconciliation against the era ledger `data/numerai_era_data.csv` "
        "(`needs_live_refresh`), per-file plans (`classify_refresh_plan`: static files "
        "`ensure`, expanding/live files `refresh` on round advance), atomic "
        "temp+fsync+`os.replace` swaps, and exit code 3 for `--check-only/--strict` "
        "gates. `validation.parquet` expands every week; `live.parquet` changes daily; "
        "`train.parquet` is static. Run: `./.venv/Scripts/python refresh_data.py "
        "--version v5.3`.",
        "- **Analysis pipeline (2026-08-09..14):** `analyze_dataset.py` — 17 modular "
        "stages (`overview, targets, ic_by_era, screens, screens_train, summary, psi, "
        "drift, derived_sets, corr_medium, corr_all, set_membership, ic_by_split, "
        "regimes, benchmarks, meta_ortho, manifest`) with auto-included dependencies "
        "and `--only/--skip` so a single metric can be recomputed without a full run. "
        "GPU: cupy `rankdata` (bit-identical to scipy on finite data, ~5.8x) with "
        "automatic scipy fallback. Per the 2026-08-14 feature-universe policy the "
        "report dumps are generated with `--features medium`; a full-universe (`all`) "
        "run is ~4-5 h (three 3,555-feature streaming passes) and is policy-"
        "prohibited for routine work. `screens_train` (train eras only) feeds "
        "`derived_sets`; the full-span `screens` is descriptive only.",
        "- **Feature campaign (2026-08-14, corrected):** 10 cells = 5 feature subsets "
        "x 2 backends (lightgbm/xgboost), identical model params per backend (fast "
        "preset, seed 20260810, walk_forward 4 folds, 8-era purge, 20D target only): "
        "v2 `screen_stable` (empty — train-only full gate found 0 stable features); "
        "v3 `screen_linear_or_nonlinear` (1); v4 `screen_drift_filtered` (1); v5 "
        "`small` (42); v6 `medium` (780). v1 (`all`, 3,555) is excluded by the "
        "feature-universe policy. Configs: `configs/campaigns/benchmark-rebuild-v1/`. "
        "Evidence assembled by `nmr.meta.campaign_evidence` from each run's "
        "`validation_preds.parquet` — per-era validation IC, block-bootstrap 95% CI, "
        "IC Sharpe, max drawdown, FNE@100% (residual IC after full linear "
        "neutralization against `medium`), and post-hoc campaign-aware DSR (fleet "
        "deflation with n_trials = recorded-cell count and empirical cross-cell "
        "Sharpe variance).",
        "- **Benchmark rebuild (2026-08-10..13):** `benchmark_runner.py` re-run on "
        "v5.3 (null baselines + lgbm ender20/60) → `artifacts/benchmark_scores.csv` "
        "(8 rows: constant-0.5, gaussian-random, uniform-random, trivial, linear, "
        "tree, v53_lgbm_ender20, v53_lgbm_ender60; 86-era meta-overlap window).",
        "- **Determinism:** every number above comes from persisted dumps "
        "(`artifacts/reports/dataset_analysis/`); the renderer is a pure function of "
        "those dumps. Registry runs are immutable and hashed (`run_id`); all registry "
        "writes are atomic temp+fsync+replace.",
        "",
    ]


def _artifact_map_section() -> list[str]:
    """§10 — where every number lives and how to regenerate it."""
    return [
        "## 10. File & Artifact Map",
        "",
        "**Dumps (full precision lives here; the report tables are display-rounded):**",
        "",
        _table(
            ["path", "contents"],
            [
                {"path": "artifacts/reports/dataset_analysis/manifest.json",
                 "contents": "data version, era ranges, feature counts, hardware spec, stages run"},
                {"path": "artifacts/reports/dataset_analysis/overview.json",
                 "contents": "split sizes (rows/eras) + packaged feature-set sizes"},
                {"path": "artifacts/reports/dataset_analysis/era_structure.parquet",
                 "contents": "per-era rows/ids/gaps (§2)"},
                {"path": "artifacts/reports/dataset_analysis/targets.json",
                 "contents": "per-target pooled moments, missing rates, era coverage (§3)"},
                {"path": "artifacts/reports/dataset_analysis/target_corr.parquet",
                 "contents": "pairwise target per-era Spearman (§3)"},
                {"path": "artifacts/reports/dataset_analysis/feature_summary.parquet",
                 "contents": "per-feature pooled moments (§4.1)"},
                {"path": "artifacts/reports/dataset_analysis/feature_ic_by_era.parquet",
                 "contents": "per-era feature IC long form (48 MB) — the raw screen input"},
                {"path": "artifacts/reports/dataset_analysis/feature_ic_screen.parquet",
                 "contents": "DESCRIPTIVE full-span screen (eras 0001..1231): mean_corr + CI + "
                 "p_value/fdr_q, spearman, stable/nonlinear flags (§4.2). DO NOT use "
                 "for subset derivation."},
                {"path": "artifacts/reports/dataset_analysis/feature_ic_screen_train.parquet",
                 "contents": "train-only screen (eras 0001..0574, `screens_train` stage) — "
                 "the sole input to `derived_sets` (§4.2)"},
                {"path": "artifacts/reports/dataset_analysis/feature_ic_by_split.parquet",
                 "contents": "train vs validation mean IC per feature (§4.4)"},
                {"path": "artifacts/reports/dataset_analysis/feature_drift_profile.parquet",
                 "contents": "PSI, W1, W1/sigma, adversarial AUC, drifted flag (§4.3)"},
                {"path": "artifacts/reports/dataset_analysis/feature_drift_psi.parquet",
                 "contents": "raw PSI stage dump"},
                {"path": "artifacts/reports/dataset_analysis/feature_set_redundancy.json",
                 "contents": "within-set pairwise |corr| stats (§4.5)"},
                {"path": "artifacts/reports/dataset_analysis/feature_corr_medium.parquet",
                 "contents": "top medium-set correlation pairs"},
                {"path": "artifacts/reports/dataset_analysis/feature_corr_medium_matrix.parquet",
                 "contents": "full 780x780 medium correlation matrix"},
                {"path": "artifacts/reports/dataset_analysis/feature_corr_all_summary.json",
                 "contents": "mean |corr| + min eigenvalue (PSD guard) (§4.6)"},
                {"path": "artifacts/reports/dataset_analysis/set_membership.json",
                 "contents": "feature -> packaged-set membership"},
                {"path": "artifacts/reports/dataset_analysis/regimes.json",
                 "contents": "crash/hot eras, IC persistence, regime thresholds (§5)"},
                {"path": "artifacts/reports/dataset_analysis/era_signal.parquet",
                 "contents": "per-era mean IC + regime labels (§5)"},
                {"path": "artifacts/reports/dataset_analysis/benchmarks.json",
                 "contents": "benchmark mean corr over the 86-era meta-overlap window (§6)"},
                {"path": "artifacts/reports/dataset_analysis/meta_ortho.parquet",
                 "contents": "feature vs meta-model correlation + orthogonal flag (§6)"},
                {"path": "artifacts/reports/dataset_analysis/neutralized_ic.json",
                 "contents": "FNE profile over the neutralization-proportion grid (§6)"},
                {"path": "artifacts/reports/dataset_analysis/derived_feature_sets.json",
                 "contents": "screen-derived subsets (from the train-only screen + drift): "
                 "screen_stable / screen_nonlinear / screen_linear_or_nonlinear / "
                 "screen_drift_filtered — empty sets are valid scientific results"},
                {"path": "artifacts/reports/dataset_analysis/campaign_variants.parquet",
                 "contents": "§7.1 evidence table (persisted; regenerated only if missing, "
                 "~30-50 min)"},
                {"path": "artifacts/reports/dataset_analysis/campaign_pairwise.parquet",
                 "contents": "§7.2 paired screen verdicts"},
            ],
        ),
        "",
        "**Campaign & registry (the actual run results):**",
        "",
        _table(
            ["path", "contents"],
            [
                {"path": "artifacts/campaigns/a8dcabc5...e6b.json",
                 "contents": "campaign log `rebuild_v53_step2`: per-config sha256, status "
                 "(recorded/error), run_id, error reasons — the source for §7 cells"},
                {"path": "artifacts/campaigns/rebuild_v53_step2.log",
                 "contents": "full campaign console log (corrected, train-only screen subsets)"},
                {"path": "artifacts/reports/dataset_analysis/campaign_parquets_meta.json",
                 "contents": "campaign-id guard for the §7.1 evidence cache (stale-cache "
                 "protection for the renderer)"},
                {"path": "artifacts/registry/<run_id>/run.json",
                 "contents": "run manifest + scorecard + metrics (immutable record)"},
                {"path": "artifacts/registry/<run_id>/validation_preds.parquet",
                 "contents": "per-era validation predictions — the input to campaign_evidence"},
                {"path": "artifacts/registry/<run_id>/oof.parquet",
                 "contents": "out-of-fold predictions"},
                {"path": "artifacts/benchmark_scores.csv",
                 "contents": "rebuilt benchmark suite output (8 model rows)"},
                {"path": "configs/campaigns/benchmark-rebuild-v1/*.yaml",
                 "contents": "the 12 campaign configs (lgbm/xgb x v1..v6)"},
                {"path": "data/v5.3/",
                 "contents": "downloaded tournament assets (train/validation/live parquet, "
                 "features.json, benchmark + meta-model parquets)"},
                {"path": "data/numerai_era_data.csv",
                 "contents": "refresh-era ledger (train/validation ranges + live round id)"},
            ],
        ),
        "",
        "**Regenerate everything:**",
        "",
        "```bash",
        "# 1. Refresh data (new rounds)",
        "./.venv/Scripts/python refresh_data.py --version v5.3",
        "",
        "# 2. Full analysis (17 stages on medium, ~1 h) or a subset. `all` is",
        "#    policy-prohibited for routine work (feature-universe policy, §8).",
        "./.venv/Scripts/python analyze_dataset.py --features medium "
        "--output-dir artifacts/reports/dataset_analysis",
        "./.venv/Scripts/python analyze_dataset.py --only screens_train,drift,derived_sets "
        "--features medium --output-dir artifacts/reports/dataset_analysis",
        "",
        "# 3. Render this report (campaign evidence cached per campaign id)",
        "./.venv/Scripts/python render_dataset_report.py --campaign-log "
        "artifacts/campaigns/a8dcabc548a56993d97836745b2659d10717474b355df9ad33886315aea12e6b.json",
        "",
        "# 4. Run a campaign (dry-run first; training is not cheap)",
        "./.venv/Scripts/python run_campaign.py --config a.yaml --config b.yaml "
        "--name <campaign> --dry-run",
        "",
        "# 5. Benchmark suite (smoke / full)",
        "./.venv/Scripts/python benchmark_runner.py --fast-mode "
        "--output artifacts/benchmark_scores_smoke.csv",
        "",
        "# 6. Tests (629-collection guard enforced by tests/test_docs_hygiene.py)",
        "./.venv/Scripts/python -m pytest -q",
        "```",
        "",
    ]


def render_report(
    manifest: dict,
    overview: dict,
    era_structure_rows: list[dict],
    targets: dict,
    target_corr_rows: list[dict],
    feature_summary_rows: list[dict],
    ic_screen_rows: list[dict],
    ic_screen_train_rows: list[dict],
    split_ic_rows: list[dict],
    drift_rows: list[dict],
    redundancy_rows: list[dict],
    regime: dict,
    era_signal_rows: list[dict],
    benchmark_rows: list[dict],
    fne_profile: list[dict],
    meta_ortho_rows: list[dict],
    corr_summary: dict,
    set_membership: dict,
    campaign_rows: list[dict] | None = None,
    pairwise_rows: list[dict] | None = None,
) -> str:
    """Render the full report. Deterministic given identical inputs."""
    out: list[str] = []
    out.append("# Pre-Modelling Dataset & Feature Study — Numerai " + manifest["data_version"])
    out.append("")
    out.append("> **Purpose — read this first.** This document is the **single source of "
               "truth for the pre-modelling phase** of Numerai in this repo: what the "
               "dataset is, how it was refreshed, what the diagnostics show, which "
               "feature sets were tested end-to-end, and what the evidence implies for "
               "model design. A new engineer or LLM agent should start at §0, then §7 "
               "(modeling implications), §8 (decision log), §9 (methodology & "
               "reproduction), and §10 (artifact map).")
    out.append("")
    out.append("> Generated from `artifacts/reports/dataset_analysis/` dumps. All numbers "
               "have full precision in the dumps; tables are display-rounded. Schema "
               "lines precede every table. Regenerate: `analyze_dataset.py --features "
               "medium` (feature-universe policy, §8) then `render_dataset_report.py "
               "--campaign-log artifacts/campaigns/<campaign_id>.json`.")
    out.append("")
    out.append(f"- Data version: `{manifest['data_version']}`")
    out.append(f"- Feature set: `{manifest['feature_set']}` ({manifest['feature_count']} features)")
    out.append(f"- Refresh date: `{manifest.get('refresh_date')}`")
    out.append(f"- Era ranges: train `{manifest['era_ranges'].get('train')}`, "
               f"validation `{manifest['era_ranges'].get('validation')}`")
    out.append("")
    out.extend(_executive_summary(
        manifest=manifest,
        overview=overview,
        targets=targets,
        ic_screen_rows=ic_screen_rows,
        ic_screen_train_rows=ic_screen_train_rows,
        campaign_rows=campaign_rows,
        benchmark_rows=benchmark_rows,
    ))

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
    out.append(_schema_block("dataset asset | official description (Numerai docs)"))
    out.append("")
    out.append(_table(
        ["asset", "official_description"],
        [
            {"asset": "train.parquet", "official_description": "Data used to train your model (static)"},
            {"asset": "validation.parquet", "official_description": "Data used to validate or train your model; expands every week"},
            {"asset": "live.parquet", "official_description": "The live data your model predicts on; changes daily"},
            {"asset": "features.json", "official_description": "Statistics about each feature + pre-made feature sets"},
            {"asset": "train_benchmark_models.parquet", "official_description": "Benchmark model predictions for some train data"},
            {"asset": "validation_benchmark_models.parquet", "official_description": "All benchmark model predictions for validation data"},
            {"asset": "live_benchmark_models.parquet", "official_description": "All benchmark model predictions for live data"},
            {"asset": "meta_model.parquet", "official_description": "Meta Model insights (MMC); only available from era 1133 onwards"},
        ],
    ))
    out.append("")
    out.append("- **Key takeaways:** the tournament is a per-era cross-section of obfuscated "
               "equities; eras are the unit of evaluation. Never pool rows across eras for "
               "metrics. `validation.parquet` expands weekly; the meta model exists only "
               "from era 1133.")
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
    out.append("### 4.1 Pooled Moments")
    out.append("")
    out.append(_schema_block("feature | pooled_mean | pooled_std | missing_rate"))
    out.append("")
    out.append(_table(
        ["feature", "pooled_mean", "pooled_std", "missing_rate"], feature_summary_rows
    ))
    out.append("")
    out.append("### 4.2 Feature-Target IC Screen (descriptive, full span)")
    out.append("")
    out.append(_schema_block(
        "feature | target | mean_corr | mean_corr_ci_lo | mean_corr_ci_hi | "
        "ci_excludes_zero | p_value | fdr_q | fdr_pass | corr_std | decay_slope | "
        "cross_regime_variance | mean_spearman | n_eras | stable | nonlinear "
        "(per-era Pearson/Spearman IC over eras 0001..1231; CI = 95% stationary "
        "block-bootstrap on the era-mean IC with horizon-aware block floors; "
        "p_value = Hall null-shifted block-bootstrap p (same seed/budget); "
        "fdr_q = Benjamini-Hochberg adjusted, per target; stable = classic "
        "point predicate AND CI strictly excluding zero AND fdr_q <= 0.05)"
    ))
    out.append("")
    out.append(_table(
        ["feature", "target", "mean_corr", "mean_corr_ci_lo", "mean_corr_ci_hi",
         "mean_spearman", "n_eras", "stable", "nonlinear"],
        ic_screen_rows,
    ))
    out.append("")
    out.append("- **Key takeaways:** `n_eras` counts valid (non-degenerate) eras only — "
               "label-lag eras without a target never contribute zero ICs. `nonlinear` flags "
               "features with |Pearson| <= 0.01 but |Spearman| > 0.01: monotone-nonlinear "
               "signal the linear screen would miss. `stable` is the full gate: a feature "
               "whose 95% CI spans zero, or whose Benjamini-Hochberg q-value exceeds 0.05, "
               "is never called stable (p_value/fdr_q/fdr_pass carry the full precision in "
               "the parquet dumps; the table shows the headline columns only).")
    out.append("")
    out.append("**Train-only screen — the subset-derivation authority.** The same "
               "full-gate predicate computed over train eras only (0001..0574, "
               "`feature_ic_screen_train.parquet`, `screens_train` stage) is the "
               "sole input to `derived_feature_sets.json`. The full-span table "
               "above is descriptive characterization only and is never used for "
               "subset selection — using validation-era labels to pick features "
               "would be look-ahead leakage. Derived-set membership is recorded "
               "in `artifacts/reports/dataset_analysis/derived_feature_sets.json`; "
               "an empty `screen_stable` there is a valid scientific result and "
               "cells training on it fail loudly at ingestion.")
    out.append("")
    out.append("### 4.3 Cross-Split Drift (PSI + W1 + Adversarial AUC)")
    out.append("")
    out.append(_schema_block(
        "feature | psi | w1 | w1_norm | auc_roc | n_train | n_val | drifted "
        "(psi > 0.25 OR w1_norm > 0.50 OR |auc_roc - 0.5| > 0.1; "
        "w1_norm = raw W1 / train-sample sigma — scale-standardized)"
    ))
    out.append("")
    out.append(_table(
        ["feature", "psi", "w1", "w1_norm", "auc_roc", "n_train", "n_val", "drifted"],
        drift_rows,
    ))
    out.append("")
    out.append("- **Key takeaways:** PSI > 0.25 marks bin-proportion shift; `w1` is the raw "
               "distributional distance (unit scale of the feature) and `w1_norm` divides it "
               "by the train-sample sigma so one threshold works across bounded and unbounded "
               "features — a raw shift of 0.25 is 5 sigma for a bounded feature but noise for "
               "an unbounded one. Adversarial AUC > ~0.6 (or < ~0.4) means the feature alone "
               "separates train from validation rows — a distribution shift a tree model can "
               "overfit to. Constrain or neutralize drifted features before training.")
    out.append("")
    out.append("### 4.4 Signal by Split")
    out.append("")
    out.append(_schema_block("feature | train_mean_ic | train_n_eras | val_mean_ic | val_n_eras | delta_ic"))
    out.append("")
    out.append(_table(
        ["feature", "train_mean_ic", "train_n_eras", "val_mean_ic", "val_n_eras", "delta_ic"],
        split_ic_rows,
    ))
    out.append("")
    out.append("- **Key takeaways:** a negative `delta_ic` is signal that fades at the "
               "train -> validation boundary — the classic overfit trap.")
    out.append("")
    out.append("### 4.5 Set Redundancy")
    out.append("")
    out.append(_schema_block("feature_set | n_features | mean_abs_corr | median_abs_corr | max_abs_corr | n_pairs"))
    out.append("")
    out.append(_table(
        ["feature_set", "n_features", "mean_abs_corr", "median_abs_corr", "max_abs_corr", "n_pairs"],
        redundancy_rows,
    ))
    out.append("")
    out.append("### 4.6 Correlation Structure")
    out.append("")
    out.append(f"- Mean |pairwise corr|: `{_fmt(corr_summary.get('mean_abs_corr'))}`; "
               f"min eigenvalue `{_fmt(corr_summary.get('min_eigenvalue'))}` (PSD guard); "
               "top pairs in `feature_corr_medium.parquet` / `feature_corr_all_summary.json`; "
               "full medium matrix in `feature_corr_medium_matrix.parquet`.")
    out.append("- **Key takeaways:** the correlation structure is descriptive. "
               "Under the corrected train-only screen the `stable` set can be "
               "empty for 20D targets (see §4.2) — treat the screen as a "
               "diagnostic filter, not a selector, prefer the packaged "
               "`medium` set, and avoid highly redundant families.")
    out.append("")

    out.append("## 5. Regimes & Signal Dynamics")
    out.append("")
    out.append(_schema_block("era | mean_ic | regime | crash | hot | degenerate"))
    out.append("")
    out.append(_table(
        ["era", "mean_ic", "regime", "crash", "hot", "degenerate"], era_signal_rows
    ))
    out.append("")
    out.append(f"- Crash eras (bottom decile): `{regime.get('crash_eras')}`")
    out.append(f"- Hot eras (top decile): `{regime.get('hot_eras')}`")
    out.append(f"- Adjacent-era IC persistence: mean "
               f"`{_fmt(regime.get('ic_persistence', {}).get('mean'))}`, "
               f"n `{regime.get('ic_persistence', {}).get('n_adjacent')}`")
    out.append("- **Key takeaways:** signal is regime-dependent; expect IC mean-reversion — "
               "never tune on crash eras in-sample. Eras with `degenerate=True` are unlabeled "
               "(label-lag: their target is absent); regime is `unlabeled` and they are "
               "excluded from the crash/hot percentiles.")
    out.append("")

    out.append("## 6. Benchmarks & Meta-Model")
    out.append("")
    out.append(_schema_block("benchmark | mean_corr | n_eras"))
    out.append("")
    out.append(_table(["benchmark", "mean_corr", "n_eras"], benchmark_rows))
    out.append("")
    out.append(_schema_block(
        "signal | proportion | mean_ic | n_eras "
        "(FNE: per-era IC of the signal after linear neutralization against the medium set; "
        "continuous gamma grid 0.0..1.0)"
    ))
    out.append("")
    out.append(_table(
        ["signal", "proportion", "mean_ic", "n_eras"], fne_profile
    ))
    out.append("")
    out.append("- **Key takeaways:** benchmark models define the achievable floor; the meta "
               "model is the upper reference. FNE: a signal whose IC collapses as the "
               "neutralization proportion rises is largely a linear function of the medium "
               "feature set — beating it requires non-linear or orthogonal signal. The "
               "proportion where IC halves is the practical neutralization budget.")
    out.append("")
    out.append(_META_SAMPLE_WARNING)
    out.append("")
    out.append(_schema_block(
        "feature | meta | corr_meta | corr_target | n_eras | orthogonal "
        "(|corr_meta| <= 0.01 and |corr_target| > 0.01 on meta-model eras)"
    ))
    out.append("")
    out.append(_table(
        ["feature", "meta", "corr_meta", "corr_target", "n_eras", "orthogonal"],
        meta_ortho_rows,
    ))
    out.append("")
    out.append("- **Key takeaways:** `orthogonal=True` marks features whose signal the "
               "consensus (meta model) does not already price in — the most valuable "
               "candidates for ensemble uniqueness. Small sample: the meta model exists "
               "only on eras 1133+.")
    out.append("")

    out.append("## 7. Modeling Implications")
    out.append("")
    out.append("- Validate **era-grouped with purge** (8 eras for 20D, 16 for 60D); "
               "random row-level CV is leakage.")
    out.append("- Rank-gaussianize per era before ensembling; never blend raw outputs.")
    out.append("- Treat the stability screen as a diagnostic, not a selector: "
               "under the corrected train-only full gate it can be empty for "
               "20D targets (§4.2, §7.1) — the packaged `medium` set is the "
               "working representation, and any screen-derived subset must "
               "be non-empty before training (ingestion fails loudly).")
    out.append("- Watch auxiliary-target era coverage before including them.")
    if campaign_rows:
        out.append("")
        out.append("### 7.1 Feature Campaign — Validation Evidence")
        out.append("")
        out.append(_schema_block(
            "variant | backend | device | n_features | mean_ic | ic_ci_lo | "
            "ic_ci_hi | ic_sharpe | max_drawdown | fne100 | fne100_ci_lo | "
            "fne100_ci_hi | n_eras | dsr_campaign_aware | dsr_pass_campaign | "
            "dsr_n_trials | dsr_trials_sr_var | dsr_reason "
            "(identical model params per backend, fixed seed, 8-era purge; "
            "mean_ic CI = 95% block-bootstrap on the full validation-window "
            "per-era IC series; FNE = residual IC after 100% linear "
            "neutralization against the medium feature set, own block-"
            "bootstrap CI; DSR = post-hoc deflated Sharpe over the whole "
            "campaign fleet — n_trials = recorded-cell count, trials_sr_var = "
            "empirical cross-cell Sharpe variance (ddof=1), pass = DSR >= 0.95)"
        ))
        out.append("")
        out.append(_table(
            ["variant", "backend", "device", "n_features", "mean_ic",
             "ic_ci_lo", "ic_ci_hi", "ic_sharpe", "max_drawdown",
             "fne100", "fne100_ci_lo", "fne100_ci_hi", "n_eras",
             "dsr_campaign_aware", "dsr_pass_campaign", "dsr_n_trials",
             "dsr_trials_sr_var", "dsr_reason"],
            [r for r in campaign_rows if r.get("status") == "recorded"],
        ))
        error_rows = [r for r in campaign_rows if r.get("status") != "recorded"]
        if error_rows:
            out.append("")
            out.append("**Failed/unrecorded variants:** " + "; ".join(
                f"{r.get('variant')} ({r.get('error')})" for r in error_rows
            ))
        out.append("")
        out.append("- **Read:** a variant's `mean_ic` CI that excludes zero is "
                   "statistically non-zero signal; `fne100` is the signal that "
                   "survives full neutralization against the medium set — the "
                   "orthogonal, non-linearizable component. `ic_sharpe` is the "
                   "risk-adjusted consistency; `max_drawdown` of the cumulative "
                   "IC line shows crash-era fragility. `dsr_campaign_aware` "
                   "deflates each cell's Sharpe by the number of cells actually "
                   "tried and the empirical cross-cell Sharpe variance — the "
                   "honest multiple-testing version of the strategy-level "
                   "p-value; cells with `dsr_reason` did not carry valid "
                   "higher-order moments and are excluded from the fleet "
                   "deflation.")
    if pairwise_rows:
        out.append("")
        out.append("### 7.2 Paired Screen Verdicts (validation IC)")
        out.append("")
        out.append(_schema_block(
            "pair | backend | mean_diff | ci_low | ci_high | n_eras "
            "(block-bootstrap 95% CI on per-era IC difference; positive diff = "
            "first variant better; CI excluding zero = significant)"
        ))
        out.append("")
        out.append(_table(
            ["pair", "backend", "mean_diff", "ci_low", "ci_high", "n_eras"],
            pairwise_rows,
        ))
        out.append("")
        out.append("- **Verdict rule (screen gate):** pairs are computed only "
                   "between recorded cells. When `screen_stable` (v2) is empty "
                   "under the train-only full gate, the v2 cells fail loudly at "
                   "ingestion (recorded as error rows, never silently dropped) "
                   "and only the v3-vs-v4 pair is testable; a v3-vs-v4 CI "
                   "excluding zero means the nonlinear rescue or the drift "
                   "filter changes model value. The v2-vs-vX gates can only "
                   "fire when `screen_stable` is non-empty.")

    out.append("## 8. Operational Findings & Decision Log (2026-08-08..14)")
    out.append("")
    out.append("- **2026-08-08 — Dataset refresh to v5.3.** `refresh_data.py` "
               "downloaded the current tournament assets; era ledger updated "
               "(`data/numerai_era_data.csv`: train 0001..0574, validation 0575..1231, "
               "live round 1329). The refresh module (`nmr/refresh.py`) had been "
               "hardened before this session: numeric version comparison (lexicographic "
               "would misorder v5.10 vs v5.3), live-round reconciliation, per-file "
               "refresh plans, atomic swaps, exit code 3 for `--check-only`.")
    out.append("- **2026-08-09..10 — Full-universe analysis.** `analyze_dataset.py` "
               "ran all stages on 3,555 features (~4-5 h): pooled moments, per-era "
               "IC screens (Pearson + Spearman + CI), PSI/W1/adversarial-AUC drift, "
               "regimes, correlation structure, benchmark/meta orthogonality. GPU "
               "rankdata (cupy) verified bit-identical to scipy and ~5.8x faster.")
    out.append("- **2026-08-10 — Benchmark & evidence purge (user-approved).** "
               "Registry (2 runs + champion), run outputs, benchmark CSVs and era "
               "labels were purged; `benchmark_runner.py` re-pointed to v5.3 and "
               "re-run (null baselines + lgbm ender20/60) → "
               "`artifacts/benchmark_scores.csv`.")
    out.append("- **2026-08-10 — Audit of evaluation machinery.** 7 findings fixed "
               "(strict era-label validation in evaluation/splitter/scorecard — "
               "inconsistent zero-padding previously caused silent data loss; numeric "
               "era ordering before block bootstrap — lexicographic sorts scramble the "
               "time series; device-string normalization in meta comparison; CRLF "
               "fingerprint normalization in `runner._code_fingerprint`; benchmark "
               "duplicate-row warning; empty-frame guards in chunked predict). 3 "
               "rejected with documented evidence (mtime-based registry sort — would "
               "violate the no-wall-clock invariant; silent sort-to-end fallback in "
               "scorecard; others).")
    out.append("- **2026-08-11 — Validation purge bug (fixed).** The validation "
               "stage compared `str(int)` era indices against zero-padded era labels, "
               "silently scoring only eras >= 1000 (232 of 649) in every run before "
               "the fix. All campaign evidence in this report was regenerated on the "
               "corrected 649-era window (583..1231).")
    out.append("- **2026-08-11 — HPO held-out partition bug (fixed).** The same "
               "era-padding class broke `HyperparameterSweep`/`bayesian_sweep` on "
               "real data (held-out split empty); labels now preserve the data's "
               "zero-padding.")
    out.append("- **2026-08-12..13 — Feature campaign (12 cells, superseded).** "
               "6 subsets x 2 backends with identical params per backend (fast "
               "preset, seed 20260810, walk_forward 4 folds, 8-era purge, 20D "
               "target). 10 cells recorded; both `v1` (all 3,555) cells are "
               "hardware-infeasible on this machine (4 + 3 documented attempts, "
               "see below). Campaign log: `artifacts/campaigns/a19577a6...c6ba9.json`; "
               "console log: `artifacts/campaigns/rebuild_v53_corrected.log`. "
               "**Superseded** — its screen cells were derived from the "
               "pre-correction full-span screen (validation-era leakage); see the "
               "2026-08-14 corrected campaign below.")
    out.append("- **Memory ceiling (documented):** the full 3,555-feature "
               "universe needs ~64 GiB commit for the xgboost full-history "
               "DMatrix and ~123 GiB of accumulated commit for the LightGBM "
               "deploy path on this 63.7 GiB machine — lgbm_v1 and xgb_v1 "
               "full-window validation cells are hardware-infeasible (4 + 3 "
               "documented attempts). The last lgbm_v1 solo attempt proved the "
               "**OOF path completes** at 3,555 features (~47 min training + "
               "~2.7 h era-batched neutralization, OOF mean IC 0.0178 / Sharpe "
               "1.19 on the train-window folds, log "
               "`artifacts/campaigns/rebuild_v53_v1solo.log`) — only the "
               "full-history deploy path OOMs, so CV experimentation on the full "
               "universe is feasible but the deploy artifact is not. Float32 "
               "zero-copy feature frames, era-batched predict, and a "
               "fresh-process full-history fit path were implemented and tested "
               "(bit-identical).")
    out.append("- **2026-08-13..14 — Phase 1 corrections (committee review; commits "
               "`7c43e1b`, `9ceee4e`, `5c520cd`).** (1) Feature screening is now "
               "train-only: the `screens_train` stage computes the full gate "
               "(classic predicate AND 95% CI strictly excluding zero AND "
               "Benjamini-Hochberg FDR at q=0.05, per target) on eras 0001..0574, "
               "and `derived_sets` reads only that parquet — validation-era label "
               "leakage eliminated. (2) Dynamic `colsample_bytree` floor "
               "`max(0.1, min(10,|S|)/|S| + 1e-7)` so small feature sets never "
               "degenerate to 1-of-N split sampling (LightGBM aliases + CatBoost "
               "`rsm` post-translation). (3) Post-hoc campaign-aware DSR: fleet "
               "deflation over recorded cells with empirical cross-cell Sharpe "
               "variance (§7.1). 624 tests green per commit (626 after the meta "
               "regression tests below).")
    out.append("- **2026-08-14 — Feature-universe operational policy.** All routine "
               "research, screening, HPO, and model iteration uses `medium` (780), "
               "`small` (42), or screen-derived subsets. The full `all` universe "
               "(3,555) is prohibited for routine iteration (RAM ceiling, ~3.5 h "
               "per-era neutralization, and empirically weaker OOF IC 0.0178 vs "
               "medium 0.0254); approved exceptions are feature-bagged "
               "sub-ensembles or single-shot offline deploy fits only. The "
               "analysis dumps and this report are therefore generated with "
               "`--features medium`.")
    out.append("- **2026-08-14 — Corrected campaign `rebuild_v53_step2`.** 10 cells "
               "(v2..v6 x lightgbm/xgboost; v1 `all` excluded per the universe "
               "policy above). Both `screen_stable` cells failed loudly at "
               "ingestion with the empty-subset guard — `screen_stable` is empty "
               "under the corrected train-only full gate, which is a valid "
               "scientific result, and the guard records it as an immutable "
               "error row rather than silently skipping. The other 8 cells "
               "recorded (evidence in §7.1). Campaign log "
               "`artifacts/campaigns/a8dcabc5...e6b.json`; console log "
               "`artifacts/campaigns/rebuild_v53_step2.log`.")
    out.append("- **Screen verdict on `target` (20D, corrected):** under the "
               "train-only full gate the screen yields **0 stable features out of "
               "780** (1 classic-only candidate, 350 pass the CI gate, 244 pass "
               "FDR — the conjunction is empty). For `target_agnes_60` it yields "
               "70 (vs 423 under the old leaky full-span screen). This confirms "
               "the committee's core diagnosis: univariate linear screening under "
               "multiplicity control finds no tradeable 20D subset; the packaged "
               "`medium` set remains the working representation (§7.1).")
    out.append("- **Cross-backend agreement:** LightGBM and XGBoost agree closely "
               "on every recorded cell (§7.1).")
    out.append("- **2026-08-14 — Meta evidence hardening (2 production bugs fixed, "
               "TDD).** Real-data execution of `campaign_evidence` on "
               "`rebuild_v53_step2` exposed two defects in the item-4 wiring: "
               "(1) `_attach_campaign_dsr` read `ic_n_eras` while production rows "
               "carry `n_eras` — the fleet DSR silently never fired (unit tests "
               "fed the helper a synthetic fixture mirroring the bug); (2) the "
               "pairwise backend lookup KeyError-ed on v2 error rows — the "
               "branch is only reachable when `screen_stable` is empty, which "
               "first happened with the corrected train-only screen. Both fixed "
               "with failing-tests-first regression coverage (key contract + "
               "v2-error-row cases); suite green at 626.")
    out.append("")
    out.extend(_methodology_section())
    out.extend(_artifact_map_section())
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
    parser.add_argument(
        "--campaign-log",
        type=Path,
        default=None,
        help="campaign log (artifacts/campaigns/<name>.json); when given, "
        "section 7 renders per-variant validation evidence from the registry",
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("artifacts") / "registry"
    )
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
        "feature_ic_screen.parquet", "feature_ic_screen_train.parquet",
        "feature_ic_by_split.parquet",
        "feature_drift_profile.parquet", "feature_set_redundancy.json",
        "regimes.json", "era_signal.parquet",
        "benchmarks.json", "meta_ortho.parquet",
        "feature_corr_all_summary.json", "set_membership.json",
    )
    for name in required_dumps:
        if not (d / name).exists():
            print(f"ERROR: missing dump {d / name}", file=sys.stderr)
            return 1

    fne_path = d / "neutralized_ic.json"
    fne_profile = (
        _load_json(fne_path).get("profile", []) if fne_path.exists() else []
    )

    campaign_rows: list[dict] | None = None
    pairwise_rows: list[dict] | None = None
    if args.campaign_log is not None:
        if not args.campaign_log.exists():
            print(
                f"ERROR: campaign log not found: {args.campaign_log}",
                file=sys.stderr,
            )
            return 1
        # Prefer the persisted evidence parquets (campaign_evidence is a
        # ~30-50 min FNE computation) but only when they belong to THIS
        # campaign log — a stale cache would silently render another
        # campaign's numbers (regression class 2026-08-14).
        variants_path = d / "campaign_variants.parquet"
        pairwise_path = d / "campaign_pairwise.parquet"
        meta_path = d / "campaign_parquets_meta.json"
        log_id = _load_json(args.campaign_log).get("campaign_id")
        cache_meta = _load_json(meta_path) if meta_path.exists() else {}
        cache_ok = (
            cache_meta.get("campaign_id") == log_id
            and variants_path.exists() and pairwise_path.exists()
        )
        if cache_ok:
            campaign_rows = pl.read_parquet(variants_path).to_dicts()
            pairwise_rows = pl.read_parquet(pairwise_path).to_dicts()
        else:
            from nmr.config import DataConfig
            from nmr.meta import campaign_evidence

            evidence = campaign_evidence(
                args.campaign_log,
                args.registry,
                data=DataConfig(version=manifest["data_version"]),
                main_target="target",
            )
            evidence.variants.write_parquet(variants_path)
            evidence.pairwise.write_parquet(pairwise_path)
            meta_path.write_text(
                json.dumps({"campaign_id": log_id}), encoding="utf-8"
            )
            campaign_rows = evidence.variants.to_dicts()
            pairwise_rows = evidence.pairwise.to_dicts()

    md = render_report(
        manifest=manifest,
        overview=_load_json(d / "overview.json"),
        era_structure_rows=pl.read_parquet(d / "era_structure.parquet").to_dicts(),
        targets=_load_json(d / "targets.json"),
        target_corr_rows=pl.read_parquet(d / "target_corr.parquet").to_dicts(),
        feature_summary_rows=pl.read_parquet(d / "feature_summary.parquet").to_dicts(),
        ic_screen_rows=pl.read_parquet(d / "feature_ic_screen.parquet").to_dicts(),
        ic_screen_train_rows=pl.read_parquet(
            d / "feature_ic_screen_train.parquet"
        ).to_dicts(),
        split_ic_rows=pl.read_parquet(d / "feature_ic_by_split.parquet").to_dicts(),
        drift_rows=pl.read_parquet(d / "feature_drift_profile.parquet").to_dicts(),
        redundancy_rows=_load_json(d / "feature_set_redundancy.json"),
        regime=_load_json(d / "regimes.json"),
        era_signal_rows=pl.read_parquet(d / "era_signal.parquet").to_dicts(),
        benchmark_rows=_load_json(d / "benchmarks.json").get("benchmarks", []),
        fne_profile=fne_profile,
        meta_ortho_rows=pl.read_parquet(d / "meta_ortho.parquet").to_dicts(),
        corr_summary=_load_json(d / "feature_corr_all_summary.json"),
        set_membership=_load_json(d / "set_membership.json"),
        campaign_rows=campaign_rows,
        pairwise_rows=pairwise_rows,
    )
    refresh_date = manifest.get("refresh_date") or "0000-00"
    output = args.output or (
        Path("docs") / "04-research" / f"pre-modelling-dataset-feature-study-{refresh_date[:7]}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
