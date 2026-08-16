# Design Spec: Executive Model Performance Report (HTML)

> Status: APPROVED (director disposition 2026-08-16). Implementation authorized across all sections.
> Scope: new pure data engine `nmr/dashboard.py` + top-level chart layer `dashboard_charts.py`; rewrite of `generate_dashboard.py` (HTML primary); rewiring of `dashboard_app.py` (Streamlit) onto the shared engine; new `tests/test_dashboard.py`.

## 1. Mission

Give the portfolio owner / Director of Investing a single, double-clickable HTML report that answers four capital-allocation questions against the standardized validation suite (same v5.3 validation set, same 86-era meta-overlap scoring window, same purge/embargo regime — enforced upstream by `PurgedEraSplitter`):

1. Which model should we allocate capital to?
2. Does it beat our benchmark hurdles?
3. What is our worst-case drawdown and downside risk?
4. Is the performance real or just luck?

Output: `artifacts/dashboard.html` — self-contained (single embedded Plotly engine; file size unbounded — director ruling 2026-08-16: the full engine is embedded inline, ~4.9 MB), offline, no CDN script tags, deterministic given identical artifacts.

## 2. Approved Decisions Log

| # | Question | Decision |
|---|---|---|
| 1 | Delivery model | **Shared engine + HTML primary.** One pure engine in `nmr/`; `generate_dashboard.py` compiles the HTML; `dashboard_app.py` (Streamlit) consumes the same engine/charts but is not rebuilt. |
| 2 | Module layout | **`nmr/dashboard.py` (pure, tested, plotly-free) + top-level `dashboard_charts.py` (plotly-only figure builders).** AGENTS.md pins Streamlit/Plotly imports outside `nmr/`; both renderers consume both layers. |
| 3 | Gate status semantics | **Absolute hurdles only.** `CHAMPION` = `champion.json` pointer (currently absent → "None Designated"); `CAPITAL READY` = clears all hard tier-4 hurdles from `configs/benchmarks/tier4_gate.yaml`; `RESEARCH` = otherwise (all 29 current runs). Benchmark rows are exempt from these statuses (see #13). |
| 4 | Missing capital cells (`cagr_1y`, `gain_to_pain_ratio`, `kelly_fraction`, `mmc_down` — absent in all 29 run.json scorecards) | **Recompute at report time with stored-first fallback.** Registry stays immutable; recompute uses `nmr/evaluation.py` per-era metrics + `nmr/payout.py` (oracle-parity code). |
| 5 | Graph scope | Core owner set: Sharpe leaderboard bar + cumulative wealth curve + drawdown curve + market-dislocation shading. Regime/perturb/horizon surfaces excluded (cells null in 29/29 runs). |
| 6 | Benchmark reference | Tier-4 per-era curve computed directly from `validation_benchmark_models.parquet` column `v53_lgbm_ender60` (independent of which scorecard CSV exists). Benchmark CSV fallback chain: full → smoke → legacy `benchmark_scores.csv`. |
| 7 | Table grouping | Executive table groups rows: **Active Champion / Proprietary Research Fleet / Benchmark Reference Floor** (tier-0 nulls through tier-4), so hurdles are never confused with proprietary models. |
| 8 | Determinism | No wall-clock timestamp in the HTML; footer carries data version + registry stats. Stable sort orders (metric desc, run_id tiebreak). |
| 9 | Gate engine / agent harness from the parent plan | **Out of scope.** Gate assertions already exist (`assert_tier0_null_floor`, `assert_tier4_gate`, `assert_hierarchy_monotone`, `promotion_verdict`); the dashboard projects them read-only, never enforces. |
| 10 | Visual companion / new deps | None. No new third-party dependencies (Plotly is already a user-granted pinned dep). |
| 11 | Plotly embedding | **Single engine injection.** `generate_dashboard.py` injects `plotly.offline.get_plotlyjs()` exactly once in the document `<head>`; each figure renders via `plotly.io.to_html(fig, include_plotlyjs=False, full_html=False)`. Never `include_plotlyjs=True` per figure (triple ~3.5 MB embed + DOM collisions). |
| 12 | Era ordering | **Numeric chronological sort everywhere** via `nmr.evaluation.sorted_era_labels()` before cumulative products and drawdown watermarks. Lexicographic ordering is a documented regression class in this repo — never sort era strings as plain strings. |
| 13 | Gate status by source | `evaluate_gate_status` conditions on `source`: benchmark rows can never receive CHAMPION / CAPITAL READY / RESEARCH. The tier-4 reference column gets `GATE HURDLE`; other tiers get `BENCHMARK`. |
| 14 | Degradation | Missing/unreadable `validation_preds.parquet` (legacy run) → the four capital cells become `None`, status stays `RESEARCH`, an informational log line is emitted — dashboard compilation must never abort. |
| 15 | Recompute sentinel | Stored-first trigger = presence of **all three scalar cells** `cagr_1y`, `gain_to_pain_ratio`, `kelly_fraction`. Per-cell `is not None` checks are ambiguous: `mmc_down` is legitimately `None` under the 5-downside-era minimum gate. |
| 16 | Kelly input rigor | `kelly_fraction` receives the **raw (unclipped)** payout series; `annual_compounded_return` and `gain_to_pain_ratio` receive the **clipped** series. Both enforced by tests. |
| 17 | Benchmark CSV mapping | `load_unified_leaderboard` maps **all** standardized scorecard columns (incl. `fnc`, `deflated_sharpe`, `cagr_1y`, `gain_to_pain_ratio`, `kelly_fraction`, CIs) from the benchmark CSV into the unified schema — the current `dashboard_app` benchmark loader silently drops several. |

## 3. Architecture & Module Topology

```
Storage (immutable reads only)
  artifacts/registry/{run_id}/run.json              ── registry metadata + scorecard
  artifacts/registry/{run_id}/validation_preds.parquet ── [era, id, prediction]
  artifacts/registry/champion.json                  ── {"run_id": ...} (absent today)
  configs/benchmarks/tier4_gate.yaml                ── 7 gate thresholds + reference column
  artifacts/reports/benchmark_hierarchy_scorecard*.csv ── tier scorecards
  data/v5.3/{validation.parquet, meta_model.parquet}   ── targets + meta-model
      │
      ▼
nmr/dashboard.py (pure engine — polars/json/numpy + nmr.{evaluation,payout,benchmark,meta,config} only)
  resolve_benchmark_path()    → benchmark CSV via full → smoke → legacy chain
  load_unified_leaderboard()  → unified polars frame (registry runs + benchmark tiers)
  reconcile_capital_metrics() → stored-first, recompute-fallback capital cells (single shared scan)
  extract_payout_timeseries() → per-era payout/wealth/drawdown series + meta drawdown mask
  evaluate_gate_status()      → CHAMPION / CAPITAL READY / RESEARCH / GATE HURDLE / BENCHMARK + per-field pass/fail
      │
      ▼
dashboard_charts.py (top-level, plotly-only figure builders, no metric math)
  build_leaderboard_bar_chart(), build_cumulative_wealth_chart(), build_drawdown_chart()
      │
      ├── generate_dashboard.py → artifacts/dashboard.html (single offline file, one inline plotly engine)
      └── dashboard_app.py     → Streamlit app reusing the same engine + figures
```

## 4. Module Contracts — `nmr/dashboard.py`

Exported in `nmr/__init__.py` imports **and** `__all__` (AGENTS.md §6 rule).

```python
def resolve_benchmark_path(
    benchmark_path: Path | None = None,
    reports_dir: Path | None = None,
) -> Path | None: ...

def load_unified_leaderboard(
    registry_dir: Path,
    benchmark_path: Path | None = None,
) -> pl.DataFrame: ...

def reconcile_capital_metrics(
    leaderboard: pl.DataFrame,
    registry_dir: Path,
    data_dir: Path,
) -> pl.DataFrame: ...

def extract_payout_timeseries(
    registry_dir: Path,
    data_dir: Path,
    run_ids: Sequence[str],
    include_tier4_ref: bool = True,
    tier4_column: str = "v53_lgbm_ender60",
) -> dict[str, Any]: ...

def evaluate_gate_status(
    leaderboard: pl.DataFrame,
    gate_config_path: Path,
    champion_path: Path,
) -> pl.DataFrame: ...
```

Semantics:

- **`resolve_benchmark_path`** — fallback chain: given path → `benchmark_hierarchy_scorecard.csv` → `benchmark_hierarchy_scorecard_smoke.csv` → legacy `artifacts/benchmark_scores.csv` → `None`.
- **`load_unified_leaderboard`** — mirrors the current `dashboard_app.load_registry_frame` / `load_benchmarks` column semantics but owns them as the single ingestion path. Explicit-`None` discipline: a legitimate scorecard `0.0` never falls back to legacy train-OOF `metrics`. Corrupt `run.json` degrades to skip (same precedent as today). Benchmark rows carry `source="benchmark"`, tier label as `run_name`, and the **complete** scorecard column mapping (decision #17).
- **`reconcile_capital_metrics`** — stored-first per decision #15: if `cagr_1y`, `gain_to_pain_ratio`, and `kelly_fraction` are all present in `payload["scorecard"]`, the block is trusted (including a stored `mmc_down` that may be `None` with its `mmc_down_reason`). Otherwise recompute:
  1. Scan the 86-era validation overlap once: `validation.parquet` columns `["era", "id", "target"]` and `meta_model.parquet` columns `["era", "id", "numerai_meta_model"]`, held as in-memory polars lookups (no per-run re-scan of data assets).
  2. Per run: join `validation_preds.parquet` against the shared lookups; `EvaluationEngine().per_era_corr` (pred vs target) and `per_era_mmc` (pred vs meta vs target) → `{era: float}` dicts.
  3. Derive via `nmr/payout.py`: `payout_series(corr, mmc)` → `annual_compounded_return(series.clipped)` → `cagr_1y`; `gain_to_pain_ratio(series.clipped)`; `kelly_fraction(series.raw)` (decision #16); `mmc_down` = mean per-era MMC over `nmr.evaluation.downside_era_indices(meta_corr_by_era)` (strictly `CORR_meta < 0`), with the scorecard's `_MMC_DOWN_MIN_ERAS = 5` minimum-era gate.
  4. Missing/unreadable `validation_preds.parquet` → capital cells `None`, log warning, continue (decision #14).
  - Acceptance: recomputed mean per-era CORR must equal the stored scorecard `corr` for a real fixture run within float tolerance; recomputed `cagr_1y` for the tier-4 reference must match the benchmark CSV cell exactly.
- **`extract_payout_timeseries`** — per-era `r_t = clip(0.75·CORR_t + 2.25·MMC_t, ±0.05)` via `payout_series`; cumulative wealth `Π(1+r_t)` from 1.0; underwater drawdown; meta drawdown mask `{era: meta_corr < 0}` (meta corr via `per_era_corr` on meta predictions vs targets). All era iteration sorted numerically via `nmr.evaluation.sorted_era_labels` (decision #12). Tier-4 reference curve from `validation_benchmark_models.parquet`. Returns:
  ```python
  {
      "eras": list[str],                    # numerically sorted
      "meta_downside_mask": list[bool],     # aligned with eras
      "series": {
          "<model_id>": {
              "label": str,
              "cumulative_wealth": list[float],
              "drawdown": list[float],
              "cagr": float,
              "mdd": float,
          },
          ...
      },
  }
  ```
- **`evaluate_gate_status`** — reads the gate via the existing `load_benchmark_file(tier4_gate.yaml)` so thresholds can never drift from the benchmark config. Projects each run against the 7 hurdles (`corr`, `corr_sharpe_ac`, `fnc`, `deflated_sharpe`, `gain_to_pain_ratio`, `cagr_1y`, turnover — turnover `None` → not a hard failure, matching `assert_tier4_gate`). Status assignment (decision #13):
  ```python
  # reference_column is read from the tier-4 gate YAML (load_benchmark_file)
  if row["source"] == "benchmark":
      status = "GATE HURDLE" if row["model_id"] == reference_column else "BENCHMARK"
  elif is_champion:
      status = "CHAMPION"
  elif all_hard_hurdles_passed:
      status = "CAPITAL READY"
  else:
      status = "RESEARCH"
  ```
  Returns one row per model with status + per-field pass/fail receipts.

## 5. Chart Layer — `dashboard_charts.py`

Thin plotly figure builders; inputs are clean polars frames from the engine; no formulas.

```python
def build_leaderboard_bar_chart(df, *, hurdle_sharpe: float) -> go.Figure: ...
def build_cumulative_wealth_chart(timeseries_payload) -> go.Figure: ...
def build_drawdown_chart(timeseries_payload) -> go.Figure: ...
```

- **Leaderboard:** horizontal bars, top 10 by `corr_sharpe_ac`, ascending y-order (highest on top); asymmetric CI error bars (`corr_sharpe_ac_ci_low/high`); vertical dashed red reference line at the tier-4 Sharpe hurdle (0.78 today); champion hatching (`pattern_shape="/"`); hover template with Annualized Return, Max Drawdown, Deflated Sharpe.
- **Wealth:** cumulative wealth from 1.0 for top-3 contenders by Sharpe + tier-4 reference; `vrect` shaded spans over eras where `meta_downside_mask` is `True`; high-contrast styling between contenders and the tier-4 baseline; dark theme.
- **Drawdown:** underwater curves `(CumWealth_t / max_{s≤t} CumWealth_s) − 1`, filled to zero with red shading.

## 6. HTML Report Layout (`artifacts/dashboard.html`)

Single scrollable page, dark theme. Plotly engine injected **once** in `<head>` via `plotly.offline.get_plotlyjs()`; figures via `plotly.io.to_html(fig, include_plotlyjs=False, full_html=False)` (decision #11).

```
🏆 NUMERAI INVESTMENT PERFORMANCE REPORT
Evaluation Window: 86 Overlap Eras (v5.3) | Data Version: v5.3

KPI SUMMARY CARDS
[Active Champion]        [Top Research Contender]   [Benchmark Hurdle]
None Designated          <run_name · short id>      v53_lgbm_ender60
(Unallocated)            SR vs 0.78 hurdle gap      Hurdle SR from tier4_gate.yaml
[Fleet Best Return]      [Worst Fleet Drawdown]     [Capital Readiness]
Best single-run CAGR     Worst payout-series MDD    n / N Models Qualified

1. CUMULATIVE WEALTH GENERATION & DOWNSIDE PROTECTION   [wealth chart]
2. RISK-ADJUSTED RETURN LEADERBOARD (95% bootstrap CIs) [bar chart + hurdle line]
3. EXECUTIVE ALLOCATION & RISK DECISION TABLE           [grouped table]
   Groups: Active Champion | Research Fleet | Benchmark Floor
   Columns: Status | Model | Ann. Return | Sharpe (AC) + CI | Max DD | GtP |
            Downside Protection | Statistical Confidence (DSR)
   Failing gate fields tinted.
▸ Technical & Audit Metadata (collapsible accordion, closed by default)
   per run: backend, preset, feature set/subset, neutralization, seed, device,
   targets, full scorecard JSON.
```

KPI semantics: Fleet Best Return = highest recomputed `cagr_1y` across research-fleet rows; Worst Fleet Drawdown = most negative payout-series drawdown; Capital Readiness = count of `CAPITAL READY` rows over research-fleet total. Footer: data version + registry stats; no generation timestamp.

## 7. Verification Plan

- **`tests/test_dashboard.py`** (unit, synthetic fixtures):
  - Schema integrity: `load_unified_leaderboard` produces identical schemas across empty registries, legacy-OOF runs, and full-scorecard runs.
  - Explicit-`None` fallthrough: scorecard `corr_sharpe_ac == 0.0` must not fall back to `metrics["sharpe"]`.
  - Recomputation parity: `reconcile_capital_metrics` matches direct `nmr.payout.payout_report()` / `annual_compounded_return` / `gain_to_pain_ratio` / `kelly_fraction` values within `1e-6` on synthetic fixtures; Kelly verified to receive the raw series.
  - Gate projection: `evaluate_gate_status` returns exact expected statuses for synthetic runs crossing/failing each hurdle, plus `GATE HURDLE` / `BENCHMARK` for benchmark rows (never `CAPITAL READY`).
  - Sorting determinism: `extract_payout_timeseries` produces identical payload hashes across repeated executions with arbitrary dict insertion orders; era ordering equals `sorted_era_labels`.
  - Degradation: missing `validation_preds.parquet` → `None` capital cells, no exception.
  - Sentinel: stored block trusted only when all three scalar cells present; stored `mmc_down=None` with reason preserved.
- **Real-data acceptance** (skip-marked like existing real-data tests): recomputed per-era CORR mean ≈ stored scorecard `corr`; recomputed tier-4 `cagr_1y` == benchmark CSV cell; all 29 runs yield complete capital columns.
- **Pre-sign-off gates (mandatory, AGENTS.md §7):** `ruff check .` + `pytest -q` (full suite) + `generate_dashboard.py` run producing `artifacts/dashboard.html`; inspect the HTML: exactly one inline Plotly engine (no `<script src>` tags — no CDN dependencies), and all 29 current runs display non-null `cagr_1y`, `gain_to_pain_ratio`, `kelly_fraction`.

## 8. Scope Exclusions & Risks

- **Excluded:** Streamlit rebuild (thin rewiring only); `ProductionGateEngine` module; autonomous-agent workflow harness; regime/perturb/horizon surfaces (cells null fleet-wide); pairwise model-difference distributions; registry backfill; any new third-party dependency; daemons/databases.
- **Risks:** (a) full (non-smoke) benchmark hierarchy CSV does not exist — only the smoke one; the tier-4 reference curve is unaffected (computed from the raw benchmark parquet), but tier-1–3 rows in the table come from smoke params until a full run regenerates the CSV (label the source). (b) Recomputed cells are float-consistent, not bit-identical, with a hypothetical future stored cell — stored-first ordering guarantees no conflict within one report. (c) `champion.json` may appear later; the engine must keep treating it as an opaque pointer and never derive it.
