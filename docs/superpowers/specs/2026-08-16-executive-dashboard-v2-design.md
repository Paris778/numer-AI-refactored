# Design Spec: Executive Dashboard v2 — Multi-Metric Trajectory & Signal Diversification

> Status: APPROVED (director disposition 2026-08-16). Implementation authorized across all sections.
> Scope: in-place evolution of the shipped v1.0 executive dashboard (`nmr/dashboard.py`, `dashboard_charts.py`, `generate_dashboard.py`, `tests/test_dashboard.py`). Supersedes the v1 spec (`2026-08-16-executive-dashboard-design.md`) for everything it covers; the v1 file stays as the historical record.

## 1. Mission

Add two capabilities to the executive report (`artifacts/dashboard.html`):

1. **Multi-metric performance trajectory** — one interactive Plotly chart toggling across 7 per-era metric dimensions (Net Payout, CORR 20D, MMC 20D, CORR 60D, MMC 60D, BMC, CWMM) with a Standard vs Cumulative view knob and market-stress shading.
2. **Signal diversification** — a pairwise similarity matrix across the top-5 research models + the tier-4 baseline, so the portfolio owner can judge whether staking several candidates actually diversifies risk.

Everything else (KPI cards, Sharpe leaderboard, executive table, drawdown chart, accordion, boundaries, determinism, degradation) is preserved from v1.

## 2. Approved Decisions Log

| # | Question | Decision |
|---|---|---|
| 1 | Metric set (§1.2 says 7, contract said 5) | **7 dimensions:** `payout`, `corr20`, `mmc20`, `corr60`, `mmc60`, `bmc`, `cwmm`. |
| 2 | Similarity definition | **Per-era rank-gaussianized predictions, pooled Pearson** over the shared 86-era meta window (canonical `Ensembler.rank_normalize` for the per-era gaussianization). |
| 3 | Timeseries payload | **Replace** `extract_payout_timeseries` with `extract_multimetric_timeseries` — no second parallel function; callers/tests migrate. |
| 4 | Chart mechanics | **Visibility groups**: one trace per (metric × model × view) ≈ 84 traces; dropdown buttons + a Standard/Cumulative mode knob switch `visible` arrays. No data duplicated in the HTML (server-side data-swap buttons rejected). |
| 5 | Similarity scope | Top-5 research fleet by `corr_sharpe_ac` + tier-4 baseline column (6 × 6 matrix). |
| 6 | Spec drift corrections (v2 input document) | Size cap dropped (director ruling: unbounded); `window.Plotly`/`cdn.plot.ly` bundle-literal checks replaced by shipped intent-level assertions (embed marker, no `<script src>`, render-call count); "legacy CSV" chain entry removed; header shows the truth (no false "Medium universe" claim). |
| 7 | `dashboard_app.py` | **No changes** — its views do not consume the new payloads (v2 input's Task 4 was completed in v1). |
| 8 | Drawdown chart | Adapted to the new payload shape (`drawdowns` + `eras` + `meta_downside_mask`), semantics unchanged (payout underwater series). |
| 9 | Cumulative semantics | `payout.cumulative` = `cumprod(1 + r_t)` (wealth); all correlation-family metrics (`corr*/mmc*/bmc/cwmm`) use `cumsum` of the per-era values. |

## 3. Architecture (delta from v1)

```
Storage (unchanged, read-only)
  artifacts/registry/{run_id}/{run.json, validation_preds.parquet}
  artifacts/registry/champion.json
  configs/benchmarks/tier4_gate.yaml
  artifacts/reports/benchmark_hierarchy_scorecard*.csv
  data/v5.3/{validation.parquet, meta_model.parquet, validation_benchmark_models.parquet}
      │
      ▼
nmr/dashboard.py (pure engine — v1 surface + two new functions)
  extract_multimetric_timeseries()   ← replaces extract_payout_timeseries()
  extract_pairwise_similarity_matrix()  ← new
      │
      ▼
dashboard_charts.py (plotly-only)
  build_multimetric_timeseries_chart()  ← new (replaces build_cumulative_wealth_chart)
  build_similarity_matrix_chart()       ← new
  build_drawdown_chart()                ← adapted payload
  build_leaderboard_bar_chart()         ← unchanged
      │
      ▼
generate_dashboard.py → artifacts/dashboard.html (layout v2, four figures)
dashboard_app.py      → unchanged
```

Boundary invariants (unchanged from v1): `nmr/` never imports plotly/streamlit; registry strictly read-only; offline single engine embed; no wall-clock in output; numeric era ordering via `sorted_era_labels`; missing data assets degrade to empty payloads (never raise).

## 4. Module Contracts — `nmr/dashboard.py`

```python
def extract_multimetric_timeseries(
    registry_dir: Path,
    data_dir: Path,
    run_ids: Sequence[str],
    include_tier4_ref: bool = True,
    tier4_column: str = "v53_lgbm_ender60",
) -> dict[str, Any]: ...

def extract_pairwise_similarity_matrix(
    registry_dir: Path,
    data_dir: Path,
    run_ids: Sequence[str],
    include_tier4_ref: bool = True,
    tier4_column: str = "v53_lgbm_ender60",
) -> tuple[list[str], list[str], list[list[float]]]: ...
```

### `extract_multimetric_timeseries`

Payload shape (exact):

```python
{
    "eras": list[str],                  # sorted_era_labels(meta eras) — the 86-era window
    "meta_downside_mask": list[bool],   # strictly CORR_meta < 0 per axis era
    "metrics": {
        metric_name: {
            "<model_id>": {
                "standard": list[float],     # per-era values, aligned to eras
                "cumulative": list[float],   # payout: cumprod(1+r); others: cumsum
                "label": str,                # "run_name · id8" / tier4_column
            },
            ...  # sorted model ids
        }
        for metric_name in ("payout", "corr20", "mmc20", "corr60", "mmc60", "bmc", "cwmm")
    },
    "drawdowns": {"<model_id>": list[float]},   # payout underwater series, aligned to eras
}
```

Computation:

- Shared lookups loaded once and extended: validation targets now include `target`, `target_ender_20`, `target_ender_60`; benchmarks include `tier4_column` (for BMC); meta unchanged. All filtered to the meta-era axis.
- Per model (and per tier-4 reference): `EvaluationEngine().per_era_corr` for `payout`-adjacent CORR is **not** used — payout = `payout_series(corr60_per_era, mmc60_per_era).clipped` (the canonical 0.75/2.25 ±5% series, 60D = v5.3 default). `corr20`/`corr60` = `per_era_corr` vs `target_ender_20` / `target_ender_60`. `mmc20`/`mmc60` = `per_era_mmc` with the matching target column and `meta_col="numerai_meta_model"`. `bmc` = `per_era_bmc` vs `tier4_column`. `cwmm` = `per_era_cwmm` (pred vs meta).
- Missing data assets → `{"eras": [], "meta_downside_mask": [], "metrics": {}, "drawdowns": {}}` + warning (never raise). Missing per-run preds → warning + skip that model across all metrics.
- Determinism: numeric era ordering, sorted model ids, labels from run.json manifest (as v1).

### `extract_pairwise_similarity_matrix`

- Returns `(labels, run_ids, matrix)`: `labels` = display labels ("run_name · id8" / tier-4 column name); `run_ids` = the corresponding ids; `matrix[i][j]` = pooled Pearson of the per-era rank-gaussianized prediction vectors of models `i` and `j` over the shared meta-era window (inner join on `[era, id]`).
- Rank-gaussianization: `Ensembler.rank_normalize` per era (unit-variance form), the repo's canonical rank-gaussianizer.
- Diagonal = 1.0; symmetric; values in `[-1, 1]`. Deterministic: sorted model ids. Missing preds → warning + that model dropped from the matrix. Missing data assets → return `([], [], [])` + warning (never raise).

## 5. Chart Layer — `dashboard_charts.py`

```python
def build_multimetric_timeseries_chart(payload: dict[str, Any]) -> go.Figure: ...
def build_similarity_matrix_chart(labels: list[str], matrix: list[list[float]]) -> go.Figure: ...
def build_drawdown_chart(payload: dict[str, Any]) -> go.Figure: ...
def build_leaderboard_bar_chart(df: pl.DataFrame, *, hurdle_sharpe: float) -> go.Figure: ...  # unchanged
```

- **Multimetric chart**: trace per (metric, model, view); `updatemenus[0]` = dropdown with the 7 metric names (buttons set `visible` for that metric's standard+cumulative traces, hide all others); `updatemenus[1]` = mode knob with "Cumulative View" / "Standard View" (buttons toggle the cumulative vs standard groups of the active metric). Stress-era `vrect` spans from `meta_downside_mask`. Dark theme (`#0d1117`/`#161b22`), legend hidden, hovertemplates use `html.escape`-ed labels, empty payload → annotation "Timeseries data unavailable without local v5.3 assets".
- **Similarity chart**: `go.Heatmap(z=matrix, x=labels, y=labels)`, `colorscale="RdBu_r"`, `zmid=0.5`, per-cell value annotations, labels escaped; empty matrix (`[]`) → annotation "Similarity matrix unavailable without local v5.3 assets" (no traces).
- **Drawdown chart**: traces from `payload["drawdowns"]` over `payload["eras"]`, `fill="tozeroy"` red fill; empty payload → the same unavailable annotation as v1.

## 6. HTML Layout (`artifacts/dashboard.html`)

Header: `Evaluation window: 86 overlap eras (1133–1218) · data v5.3` (no universe claim — the fleet mixes feature universes). KPI cards unchanged from v1. Sections:

1. ALPHA GENERATION & MULTI-METRIC PERFORMANCE TRAJECTORY (multimetric chart)
2. RISK-ADJUSTED RETURN LEADERBOARD (unchanged)
3. SIGNAL DIVERSIFICATION & PAIRWISE SIMILARITY MATRIX (top-5 fleet + tier-4 baseline heatmap)
4. EXECUTIVE ALLOCATION & RISK DECISION TABLE (unchanged)
5. CAPITAL DRAWDOWN (adapted chart)
6. Technical & audit accordion (unchanged)

Four figures now (multimetric, leaderboard, similarity, drawdown): stable `div_id`s, single engine embed via the `<!-- plotly-engine-embed -->` marker, render-call count assertion 3 → 4, no `<script src>` tags, no wall-clock. Size unbounded (director ruling).

## 7. Verification Plan

- **Unit tests (`tests/test_dashboard.py`)**, replacing the migrated v1 payout/drawdown tests:
  - Multimetric payload: exact key set, all series aligned to `eras`, numeric era order, determinism (repeat + insertion-order hash), cumulative semantics (payout `1.05^3` wealth on the perfect-corr fixture; corr-family cumsum), degradation to empty payload on missing assets, missing-run skip.
  - Similarity matrix: identity diagonal, symmetry, two identical-signal models → 1.0, rank-gaussian robustness (scale-shifted copy of a model still → 1.0), labels/ids order determinism, empty-data degradation.
  - Charts: dropdown has 7 metric buttons + 2 mode buttons; visibility array lengths match trace count; heatmap `z` equals the matrix; drawdown chart consumes the new payload; empty-payload annotations.
  - HTML: 4 render calls after the engine marker; "SIGNAL DIVERSIFICATION" section present; embed marker once; no `<script src>`.
- **Real-data acceptance** (skip-marked, same convention): all 7 metric dicts populated for the real top run with 86-era arrays; similarity matrix on the real top-5 + tier-4 baseline within `[-1, 1]`, diagonal 1.
- **Pre-sign-off gates (AGENTS.md §7):** `ruff check .` + `pytest -q` (full suite) + `generate_dashboard.py` → inspect the HTML (marker, no external scripts, four render calls, all sections present).

## 8. Scope Exclusions & Risks

- **Excluded:** Streamlit changes; new dependencies; server-side data-swap button mechanics (duplicates data); CORR60/MMC60 dropdown renaming; any change to gate semantics, KPI cards, or the executive table.
- **Risks:** (a) ≈84 traces make the figure object large but static HTML stays within one engine embed (size unbounded by ruling); (b) `bmc` per era requires the benchmark column on the whole axis — `validation_benchmark_models.parquet` covers all validation eras, verified; (c) the tier-4 baseline participates in both the multimetric chart and the similarity matrix via the same `tier4_column` from `tier4_gate.yaml`; (d) v1's `build_cumulative_wealth_chart` is deleted — nothing outside the report consumed it.
