# Design Spec: Executive Dashboard v2 — Multi-Metric Trajectory & Signal Diversification

> Status: APPROVED (director disposition 2026-08-16). Implementation authorized across all sections.
> Scope: in-place evolution of the shipped v1.0 executive dashboard (`nmr/dashboard.py`, `dashboard_charts.py`, `generate_dashboard.py`, `tests/test_dashboard.py`). Supersedes the v1 spec (`2026-08-16-executive-dashboard-design.md`) for everything it covers; the v1 file stays as the historical record.

## 1. Mission

Add two capabilities to the executive report (`artifacts/dashboard.html`):

1. **Multi-metric performance trajectory** — one interactive chart toggling across 7 per-era metric dimensions (Net Payout, CORR 20D, MMC 20D, CORR 60D, MMC 60D, BMC, CWMM) with a Standard vs Cumulative view switch and market-stress shading.
2. **Signal diversification** — a pairwise similarity matrix across the top-5 research models + the tier-4 baseline, with a diversification-quality badge and an equal-weighted ensemble-Sharpe card.

Everything else (KPI cards, Sharpe leaderboard, executive table, drawdown chart, accordion, boundaries, determinism, degradation) is preserved from v1.

## 2. Approved Decisions Log

| # | Question | Decision |
|---|---|---|
| 1 | Metric set (§1.2 says 7, contract said 5) | **7 dimensions:** `payout`, `corr20`, `mmc20`, `corr60`, `mmc60`, `bmc`, `cwmm`. |
| 2 | Similarity definition | **Per-era rank-gaussianized predictions, pooled Pearson** over the shared 86-era meta window (canonical `Ensembler.rank_normalize` for the per-era gaussianization). |
| 3 | Timeseries payload | **Replace** `extract_payout_timeseries` with `extract_multimetric_timeseries` — no second parallel function; callers/tests migrate. |
| 4 | Chart mechanics (amended — director review) | **No plotly `updatemenus`.** Dual static visibility menus cannot maintain independent (metric × view) state. The chart is rendered by an embedded **vanilla-JS controller** (payload embedded as JSON; two JS state variables; `Plotly.react` on change). Single-file offline delivery retained; no Node/npm toolchain. |
| 5 | Similarity scope | Top-5 research fleet by `corr_sharpe_ac` + tier-4 baseline column (6 × 6 matrix). |
| 6 | Spec drift corrections (v2 input document) | Size cap dropped (director ruling: unbounded); `window.Plotly`/`cdn.plot.ly` bundle-literal checks replaced by shipped intent-level assertions (embed marker, no `<script src>`, render-call count); "legacy CSV" chain entry removed; header shows the truth (no false "Medium universe" claim). |
| 7 | `dashboard_app.py` | **No changes** — its views do not consume the new payloads (v2 input's Task 4 was completed in v1). |
| 8 | Drawdown chart | Adapted to the new payload shape (`drawdowns` + `eras` + `meta_downside_mask`), semantics unchanged (payout underwater series). |
| 9 | Cumulative semantics | `payout.cumulative` = `cumprod(1 + r_t)` (wealth); all correlation-family metrics (`corr*/mmc*/bmc/cwmm`) use `cumsum` of the per-era values. |
| 10 | Target columns (P0 fix) | **Dynamic resolution** via `_resolve_horizon_targets(schema_cols)` with fallback chains (`target_ender_20` → `target_cyrusd_20` → `target_20` → `target`; 60D analogous) — never hardcoded column crashes. |
| 11 | BMC self-comparison (P0 fix) | When a model **is** `tier4_column`, short-circuit `bmc = {era: 0.0 for era in axis}` (zero value-add over itself) — no degenerate residual computation. |
| 12 | Similarity computation (P1 fix) | **Single multi-way inner join** on `[era, id]` across all K preds → aligned `(N, K)` matrix → per-era rank-gaussianize per column → one `np.corrcoef`. Global inner-join intersection (not pairwise-complete). |
| 13 | `payout.standard` | Per-era clipped payout returns `r_t ∈ [-0.05, +0.05]` (the `payout_series(…).clipped` array). |
| 14 | Cumulative origin | Cumulative lists are aligned 1:1 with `eras` (no synthetic origin point; index `t` = cumulative through era `t`). |
| 15 | Degenerate predictions | Per era: a zero-variance rank-gaussianized column becomes zeros (no NaN propagation); a globally zero-variance column yields 0.0 correlations (not NaN) in the matrix. |
| 16 | Augmentation 1 — diversification badge | Max-pairwise-overlap + mean-fleet-overlap indicators above the heatmap; thresholds: `< 0.65` EXCELLENT, `≤ 0.85` MODERATE, `> 0.85` HIGH REDUNDANCY. |
| 17 | Augmentation 2 — ensemble Sharpe card | Equal-weighted blended Sharpe `mean(μ) / sqrt(wᵀΣw)` from the per-era payout series of the top-3 fleet models (w = 1/3), shown as a card beneath the matrix (heuristic, non-interactive). |

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
nmr/dashboard.py (pure engine — v1 surface + new functions)
  _resolve_horizon_targets()            ← new helper (P0)
  extract_multimetric_timeseries()      ← replaces extract_payout_timeseries()
  extract_pairwise_similarity_matrix()  ← new (multi-way join + np.corrcoef)
      │
      ▼
dashboard_charts.py (plotly/JS presentation)
  multimetric_chart_html(payload)       ← new: div + embedded JSON payload + vanilla-JS controller
  build_similarity_matrix_chart()       ← new (static go.Figure)
  build_drawdown_chart()                ← adapted payload
  build_leaderboard_bar_chart()         ← unchanged
      │
      ▼
generate_dashboard.py → artifacts/dashboard.html (layout v2; 3 plotly.py figures + 1 JS chart = 4 render calls)
dashboard_app.py      → unchanged
```

Boundary invariants (unchanged from v1): `nmr/` never imports plotly/streamlit; registry strictly read-only; offline single engine embed; no wall-clock in output; numeric era ordering via `sorted_era_labels`; missing data assets degrade to empty payloads (never raise).

## 4. Module Contracts — `nmr/dashboard.py`

```python
def _resolve_horizon_targets(schema_cols: Sequence[str]) -> tuple[str, str]: ...

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

### `_resolve_horizon_targets`

```python
target_20 = next((c for c in ("target_ender_20", "target_cyrusd_20", "target_20", "target") if c in schema_cols), "target")
target_60 = next((c for c in ("target_ender_60", "target_cyrusd_60", "target_60", "target") if c in schema_cols), target_20)
```

Resolved once per call from the validation parquet schema; missing both → `("target", "target")` and the downstream `per_era_corr` fails loudly with its own message (no silent columns).

### `extract_multimetric_timeseries`

Payload shape (exact):

```python
{
    "eras": list[str],                  # sorted_era_labels(meta eras) — the 86-era window
    "meta_downside_mask": list[bool],   # strictly CORR_meta < 0 per axis era
    "metrics": {
        metric_name: {
            "<model_id>": {
                "standard": list[float],     # per-era values, aligned 1:1 to eras
                "cumulative": list[float],   # payout: cumprod(1+r); others: cumsum; no origin point
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

- Shared lookups loaded once and extended: validation targets (`target`, `target_ender_20`, `target_ender_60` resolved dynamically), benchmarks (`tier4_column`), meta. All filtered to the meta-era axis.
- Per model (and per tier-4 reference): `payout` = `payout_series(corr60, mmc60).clipped` (canonical 0.75/2.25 ±5%); `corr20`/`corr60` = `per_era_corr` vs the resolved 20D/60D target columns; `mmc20`/`mmc60` = `per_era_mmc` with the matching target and `meta_col="numerai_meta_model"`; `bmc` = `per_era_bmc` vs `tier4_column` **except when the model is the reference itself → `{era: 0.0}` short-circuit**; `cwmm` = `per_era_cwmm`.
- Missing data assets → `{"eras": [], "meta_downside_mask": [], "metrics": {}, "drawdowns": {}}` + warning (never raise). Missing per-run preds → warning + skip that model across all metrics.

### `extract_pairwise_similarity_matrix`

- Single multi-way inner join of all K `validation_preds.parquet` files on `[era, id]` (global intersection over the shared meta window) → aligned `(N_rows, K)` matrix.
- Per-era rank-gaussianization per column via `Ensembler.rank_normalize`; per era, a zero-variance column becomes zeros (guard against degenerate predictions); globally zero-variance columns yield 0.0 correlation rows (no NaN).
- `matrix = np.corrcoef(gaussianized.T).tolist()`; returns `(labels, run_ids, matrix)` with diagonal 1.0, symmetry, values in `[-1, 1]`; deterministic (sorted ids). Missing preds → warning + model dropped. Missing data assets → `([], [], [])` + warning.

## 5. Presentation Layer — `dashboard_charts.py`

```python
def multimetric_chart_html(payload: dict[str, Any]) -> str: ...
def build_similarity_matrix_chart(labels: list[str], matrix: list[list[float]]) -> go.Figure: ...
def build_drawdown_chart(payload: dict[str, Any]) -> go.Figure: ...
def build_leaderboard_bar_chart(df: pl.DataFrame, *, hurdle_sharpe: float) -> go.Figure: ...  # unchanged
```

- **`multimetric_chart_html`** returns the full `<div id="multimetric-chart">` + `<script>` block: the payload embedded as `const MM_PAYLOAD = {...}` (`json.dumps(..., sort_keys=True)` — deterministic), a fixed vanilla-JS controller with two state variables (`currentMetric`, `currentView`), an HTML `<select>` (7 metric options) and two view `<button>`s wired via `addEventListener`, and `applyState()` rebuilding traces from `MM_PAYLOAD` via `Plotly.react`. Stress-era `vrect` spans are fixed layout shapes from `meta_downside_mask`. No plotly `updatemenus` anywhere. Labels rendered into hovertemplates via a small JS escape helper. Empty payload → the controller renders the "Timeseries data unavailable without local v5.3 assets" annotation.
- **Similarity chart**: static `go.Heatmap(z=matrix, x=labels, y=labels)`, `colorscale="RdBu_r"`, `zmid=0.5`, per-cell value annotations, labels escaped; empty matrix → annotation (no traces).
- **Drawdown chart**: traces from `payload["drawdowns"]` over `payload["eras"]`, `fill="tozeroy"` red fill; empty payload → the same unavailable annotation as v1.

## 6. HTML Layout (`artifacts/dashboard.html`)

Header: `Evaluation window: 86 overlap eras (1133–1218) · data v5.3` (no universe claim — the fleet mixes feature universes). KPI cards unchanged from v1. Sections:

1. ALPHA GENERATION & MULTI-METRIC PERFORMANCE TRAJECTORY (JS-controller chart; models = top-3 by Sharpe + tier-4 baseline)
2. RISK-ADJUSTED RETURN LEADERBOARD (unchanged)
3. SIGNAL DIVERSIFICATION & PAIRWISE SIMILARITY MATRIX — heatmap (top-5 fleet + tier-4 baseline) preceded by the **Diversification Quality badge** (`Mean Overlap X.XX · Max Overlap X.XX` with EXCELLENT / MODERATE / HIGH REDUNDANCY coloring per decision #16) and followed by the **Equal-Weight Ensemble Sharpe card** (top-3 blend, decision #17)
4. EXECUTIVE ALLOCATION & RISK DECISION TABLE (unchanged)
5. CAPITAL DRAWDOWN (adapted chart)
6. Technical & audit accordion (unchanged)

Render-call accounting: 3 static plotly.py figures + 1 JS `Plotly.newPlot` = 4 calls after the single engine embed (`<!-- plotly-engine-embed -->` marker). Stable `div_id`s, no `<script src>` tags, no wall-clock. Size unbounded (director ruling).

## 7. Verification Plan

- **Unit tests (`tests/test_dashboard.py`)**, replacing the migrated v1 payout/drawdown tests:
  - Multimetric payload: exact key set; alignment 1:1 with `eras` (no origin point); numeric era order; determinism (repeat + insertion-order hash); cumulative semantics (payout `1.05^3` wealth on the perfect-corr fixture; corr-family cumsum); `_resolve_horizon_targets` fallback chain on synthetic schemas; BMC short-circuit (`bmc[ref] == {era: 0.0}`); degradation to empty payload; missing-run skip.
  - Similarity matrix: identity diagonal; symmetry; identical-signal (incl. scale-shifted copy) → 1.0; zero-variance column guard (constant predictions → 0.0 row, no NaN); global inner-join alignment (a run missing an era is dropped from the intersection, not padded); labels/ids order determinism; empty-data degradation.
  - Presentation: `multimetric_chart_html` output contains the exact serialized payload, the `multimetric-chart` div, `<select>` with 7 options, both view buttons, and no `updatemenus`/external script tags; similarity heatmap `z` equals the matrix; drawdown consumes the new payload; empty-payload annotations.
  - HTML: 4 render calls after the engine marker; "SIGNAL DIVERSIFICATION" section + badge + ensemble card present; embed marker once; no `<script src>`.
- **Real-data acceptance** (skip-marked, same convention): all 7 metric dicts populated for the real top run with 86-era arrays; similarity matrix on the real top-5 + tier-4 baseline within `[-1, 1]`, diagonal 1, badge thresholds render.
- **Pre-sign-off gates (AGENTS.md §7):** `ruff check .` + `pytest -q` (full suite) + `generate_dashboard.py` → inspect the HTML (marker, no external scripts, four render calls, all sections present); open the file once in a browser to confirm the metric dropdown and view toggle behave (manual check — JS cannot be pytest-tested without a browser).

## 8. Scope Exclusions & Risks

- **Excluded:** Streamlit changes; Node/npm toolchain; server-side data-swap buttons; new dependencies; changes to gate semantics, KPI cards, or the executive table; any optimizer in the ensemble card (equal-weight only).
- **Risks:** (a) the JS controller is new surface — covered by structural tests plus the mandatory manual browser check at sign-off; (b) the tier-4 baseline appears in both the multimetric chart and the matrix via the same `tier4_column` from `tier4_gate.yaml`; (c) v1's `build_cumulative_wealth_chart` is deleted — nothing outside the report consumed it; (d) the ensemble-Sharpe card is a heuristic (simultaneous-staking assumption) and is labeled as such in the HTML.
