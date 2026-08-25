# Benchmark "Line in the Sand" — The 5-Tier Hierarchy

> **Purpose of this file:** a standing memory aid for the tiered benchmark floor every model must clear before capital deployment. The authoritative spec is the evaluation bible (`evaluation-suite-bible.md`, §11 E6 gate) and the design spec `docs/superpowers/specs/2026-08-15-benchmark-hierarchy-design.md`. If a tier, gate, or threshold changes, change it in the bible first, then here.

## 1) The one idea

Tiers 0–3 exist so a candidate's scorecard can be read as a rung on a ladder; Tier 4 is the production gate. Every rung emits a complete scorecard through `evaluate_model()`, so the ladder is directly comparable row-for-row with a real candidate.

| Tier | Rungs | Role |
| --- | --- | --- |
| 0 | constant-0.5, uniform-random, gaussian-random (clipped), feature-mean (small) | statistical zero-floor; a candidate indistinguishable from tier 0 is defective |
| 1 | Ridge small / medium / 4-target blend (purged, standardized) | linear factor frontier; non-linear models must beat it |
| 2 | shallow LightGBM/XGBoost + canonical fast preset | depth/interaction hurdle |
| 3 | hello-numerai, neutralized-50, sunshine 4×20D ensemble (in-process re-fits) | canonical community references |
| 4 | `v53_lgbm_ender60` (capital gate) + `v53_lgbm_ender20` (informational) official Numerai benchmark columns | the line in the sand for capital |

## 2) Hard gates (enforced by `nmr/benchmark.py`)

- **G — Tier-0 null floor:** |CORR| ≤ 0.005 and |AC-Sharpe| ≤ 0.15 for the three structural nulls (constant-0.5, uniform-random, gaussian-random). `null_feature_mean` is scored but excluded from the gate — it is not structurally null on v5.3 (corr 0.0029, sharpe 0.257). There is **no DSR check**: null DSRs span 0.11–1.0, so deflated Sharpe has no constant null value. A structural null scoring above its floor means a broken metric.
- **G — Tier-4 production gate:** measured on `v53_lgbm_ender60` (the gated capital line; `v53_lgbm_ender20` is scored alongside as an informational tier-4 row) over the shared meta-model overlap window — CORR ≥ 0.0286, AC-Sharpe ≥ 0.78, FNC@medium ≥ 0.020, DSR ≥ 0.95, GPR ≥ 1.50, CAGR > 0, turnover ≤ 0.35. Thresholds live in `configs/benchmarks/tier4_gate.yaml` and are re-pinned to measured v5.3 values with evidence when they deviate. Turnover is structurally unavailable on v5.3 (consecutive validation eras share zero ids): it is reported as measured=None/pass=None in the gate report, **excluded from hard failure**, and logged loudly.
- **G — Monotonicity:** per-tier max of mean CORR orders Tier0 < Tier1 < Tier2 < Tier3 ≤ Tier4 (atol 1e-5); `rank_scalar` is selectable via `metric="rank_scalar"` but its noise spread swamps the null-vs-ridge rung on real data (evidence in the design-spec amendments). Enforced in full runs; logged-only in `--fast-mode` (fast tree params degrade tiers 2–3 by design).
- **G — Determinism:** same data-version + seed + configs ⇒ identical scorecard hashes across processes (`scorecards_sha256`).

## 3) Fit topology (leakage rules)

Tiers 1–3 fit on `train.parquet` eras minus the final 8 (purge buffer) and predict `validation.parquet` eras. The split is asserted by `train_validation_purged_split()`: strict chronological ordering, exact 8-era buffer, numeric era labels only. Features are float32 end-to-end. Ridge features are standardized with trimmed-train statistics (zero-variance features → 0.0). Multi-target blends are equal-weight in the per-era rank-Gaussian domain (`Ensembler`), then re-gaussianized.

## 4) Tier anchors (report-only reference lines)

Tier 1–3 `anchors` in the YAMLs are sanity reference lines logged against measured values — they are **not** enforced gates. Measured tier-1..3 corrs on v5.3 are far below the spec's aspirational anchors; that is expected and documented. The only enforced absolute numbers are the tier-0 floor and the tier-4 thresholds.

## 5) Notes & deviations

- **FNE is FNC@medium (780),** not the full 3,555 universe: full-universe validation FNC is memory-prohibited by the feature-universe policy. The tier-4 gate field `fnc_min` is measured against medium.
- Tier-4 point estimates are identical between fast and full modes (the reference is a data column); only tier 1–3 rungs degrade in fast mode.
- Run: `python benchmark_runner.py --data-dir data/v5.3 --seed 42 --n-boot 1000` (full) or `--fast-mode` (smoke). Outputs: `artifacts/reports/benchmark_hierarchy_scorecard.csv` + `benchmark_gate_report.csv`; the smoke convention writes `benchmark_hierarchy_scorecard_smoke.csv` + `benchmark_gate_report_smoke.csv`.

## Untiered Benchmark Fleet

A fourth config layer, `configs/benchmarks/fleet/`, holds benchmark models
with **no tier assignment**: silly heuristics, tutorial small/deep variants,
community example scripts (shallow/deep), and the Finance Arena v0.2-v1.5.1
recreations — 19 cells. They are scored through the identical
`evaluate_model` pipeline and reported in
`artifacts/reports/benchmark_fleet_scorecard.csv` with a `placement` column
(measured CORR vs the per-tier max-corr rungs), informational tier-4 gate
verdicts, and a `selection_bias` flag (true only for the v1.5.1 search cell,
whose candidate selection uses validation — never compare it naively).

Fleet results never participate in the hard gates (null floor, tier-4 gate,
monotonicity). Anchors are report-only and re-pinned after measurement.
Full design: `docs/superpowers/specs/2026-08-19-benchmark-fleet-design.md`.
