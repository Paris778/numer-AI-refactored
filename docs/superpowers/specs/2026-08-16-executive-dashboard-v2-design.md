# Design Spec: Executive Dashboard v2 — Multi-Metric Trajectory & Signal Diversification

> **Historical record:** Superseded by `2026-08-18-vanilla-dashboard-design.md`; the Plotly architecture below is no longer active.

> Status: Historical; superseded. Implementation details below are retained for audit history.
> Scope: in-place evolution of the shipped v1.0 executive dashboard (`nmr/dashboard.py`, `dashboard_charts.py`, `generate_dashboard.py`, `tests/test_dashboard.py`). Supersedes the v1 spec (`2026-08-16-executive-dashboard-design.md`) for everything it covers; the v1 file stays as the historical record.

## 1. Mission

Add two capabilities to the executive report (`artifacts/dashboard.html`):

1. **Multi-metric performance trajectory** — one interactive chart toggling across 7 per-era metric dimensions (Net Payout, CORR 20D, MMC 20D, CORR 60D, MMC 60D, BMC, CWMM) with a Standard vs Cumulative view switch and market-stress shading.
2. **Signal diversification** — a pairwise similarity matrix across the top-5 research models + the tier-4 baseline, with a diversification-quality badge (max/mean overlap + stress-regime correlation delta) and an equal-weighted ensemble-Sharpe card.

Everything else (KPI cards, Sharpe leaderboard, executive table, drawdown chart, accordion, boundaries, determinism, degradation) is preserved from v1.

## 2. Approved Decisions Log

| # | Question | Decision |
|---|---|---|
| 1 | Metric set (§1.2 says 7, contract said 5) | **7 dimensions:** `payout`, `corr20`, `mmc20`, `corr60`, `mmc60`, `bmc`, `cwmm`. |
| 2 | Similarity definition | **Per-era rank-gaussianized predictions, pooled Pearson** over the shared 86-era meta window (canonical `Ensembler.rank_normalize` for the per-era gaussianization). |
| 3 | Timeseries payload | **Replace** `extract_payout_timeseries` with `extract_multimetric_timeseries` — no second parallel function; callers/tests migrate. |
| 4 | Chart mechanics (review round 1) | **No plotly `updatemenus`.** Embedded **vanilla-JS controller** (payload as JSON; two JS state variables; `Plotly.react` on change). Single-file offline delivery retained; no Node/npm toolchain. |
| 5 | Similarity scope | Top-5 research fleet by `corr_sharpe_ac` + tier-4 baseline; **K = min(5, N_fleet) + (1 if ref)** — fewer than 5 fleet runs shrinks the matrix gracefully (review round 2). |
| 6 | Spec drift corrections (v2 input document) | Size cap dropped (director ruling: unbounded); `window.Plotly`/`cdn.plot.ly` bundle-literal checks replaced by shipped intent-level assertions (embed marker, no `<script src>`, render-call count); "legacy CSV" chain entry removed; header shows the truth. |
| 7 | `dashboard_app.py` | **No changes** — its views do not consume the new payloads. |
| 8 | Drawdown chart | Adapted to the new payload shape (`drawdowns` + `eras` + `meta_downside_mask`), semantics unchanged (payout underwater series). |
| 9 | Cumulative semantics | `payout.cumulative` = `cumprod(1 + r_t)` (wealth); all correlation-family metrics (`corr*/mmc*/bmc/cwmm`) use `cumsum` of the per-era values. |
| 10 | Target columns (P0, round 1) | **Dynamic resolution** via `_resolve_horizon_targets(schema_cols)` with fallback chains (`target_ender_20` → `target_cyrusd_20` → `target_20` → `target`; 60D analogous). |
| 11 | BMC self-comparison (P0, round 1) | When a model **is** `tier4_column`, short-circuit `bmc = {era: 0.0 for era in axis}` — no degenerate residual computation. |
| 12 | Similarity computation (P1, round 1) | **Single multi-way join** on `[era, id]` across all K candidates → aligned `(N, K)` matrix → per-era rank-gaussianize per column → one `np.corrcoef`. Global inner-join intersection. |
| 13 | `payout.standard` | Per-era clipped payout returns `r_t ∈ [-0.05, +0.05]` (the `payout_series(…).clipped` array). |
| 14 | Cumulative origin | Cumulative lists aligned 1:1 with `eras` (no synthetic origin point; index `t` = cumulative through era `t`). |
| 15 | Degenerate predictions | Per era: zero-variance rank-gaussianized column → zeros (no NaN); globally zero-variance column → 0.0 correlation rows. |
| 16 | Augmentation 1 (round 1) | Diversification badge: max-pairwise-overlap + mean-fleet-overlap; thresholds `< 0.65` EXCELLENT, `≤ 0.85` MODERATE, `> 0.85` HIGH REDUNDANCY. |
| 17 | Augmentation 2 (round 1) | Equal-weighted blended Sharpe card (top-3 fleet payout series, w = 1/3, heuristic, non-interactive). |
| 18 | Lookup column collision (P0, round 2) | Target column list passed to `pl.read_parquet` must be **deduplicated** (`list(dict.fromkeys(["era", "id", "target", target_20, target_60]))`) — Polars raises on duplicate select columns when the fallback resolves both horizons to `"target"`. |
| 19 | Payout target parity (P0, round 2) | `payout` in the timeseries is anchored to **`main_target="target"`** (same anchor as `reconcile_capital_metrics`/`scorecard.py`) so the chart's cumulative return and the executive table's `cagr_1y` agree by construction. `corr60`/`mmc60` remain standalone metrics vs the explicit 60D column. |
| 20 | Tier-4 in similarity (P0, round 2) | The Kth similarity candidate is read from **`data/v5.3/validation_benchmark_models.parquet` column `tier4_column`** — it has no registry directory; never attempt `artifacts/registry/{tier4_column}/`. |
| 21 | JS axis config (P1, round 2) | The controller's `applyState()` applies a **`METRIC_CONFIG` dict** (per metric × view: `yaxis.title` + `yaxis.tickformat`, e.g. payout cumulative = "Cumulative Wealth (1.0 Stake)"/`.3f`, payout standard = "Per-Era Net Return"/`.2%`, corr-family `'.4f'`) via `Plotly.react` layout — metric switches without axis updates render broken scales. |
| 22 | Floating-point clamp (P1, round 2) | `matrix = np.clip(np.corrcoef(...), -1.0, 1.0)` — `1.0000000000000002`-style values break assertions and heatmap bounds. |
| 23 | Missing horizon target (round 2) | In the **dashboard extraction path**, a missing horizon target column populates that metric slice with **zeros + a warning** — never aborts report generation. (Standalone `per_era_corr` callers keep fail-loud behavior.) |
| 24 | JS scoping (round 2) | All controller event listeners are scoped to the rendered root div (single render per page; no `document.getElementById` collisions). |
| 25 | Augmentation 3 (round 2) | Heatmap highlights the **top-ranked model's row/column** (visual indicator on row/col 0). |
| 26 | Augmentation 4 (round 2) | Badge also shows **stress-regime correlation delta**: mean over off-diagonal pairs of `(ρ_stress − ρ_normal)`, where `ρ_stress`/`ρ_normal` are pooled correlations restricted to meta-downside eras (`meta_downside_mask`) vs the rest. |
| 27 | Ensemble card bounds (P2, round 2) | N_fleet < 3 → the ensemble-Sharpe card renders "—" (insufficient models), never a division error. |

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
  extract_pairwise_similarity_matrix()  ← new (multi-way join incl. benchmark column + np.corrcoef)
      │
      ▼
dashboard_charts.py (plotly/JS presentation)
  multimetric_chart_html(payload)       ← div + embedded JSON payload + vanilla-JS controller
  build_similarity_matrix_chart()       ← static go.Figure (top-row/col highlight)
  build_drawdown_chart()                ← adapted payload
  build_leaderboard_bar_chart()         ← unchanged
      │
      ▼
generate_dashboard.py → artifacts/dashboard.html (layout v2; 3 plotly.py figures + 1 JS chart = 4 render calls)
dashboard_app.py      → unchanged
```

Boundary invariants (unchanged from v1): `nmr/` never imports plotly/streamlit; registry strictly read-only; offline single engine embed; no wall-clock in output; numeric era ordering via `sorted_era_labels`; missing data assets degrade to empty payloads (never raise); report generation never aborts on missing optional columns (decision #23).

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
) -> tuple[list[str], list[str], list[list[float]], dict[str, Any]]: ...
```

(plan-level refinement: the stress-regime delta rides along as a fourth element — `{"mean_delta": float | None, "n_pairs": int}` — rather than a separate function.)

### `_resolve_horizon_targets`

```python
target_20 = next((c for c in ("target_ender_20", "target_cyrusd_20", "target_20", "target") if c in schema_cols), "target")
target_60 = next((c for c in ("target_ender_60", "target_cyrusd_60", "target_60", "target") if c in schema_cols), target_20)
```

Resolved once per call from the validation parquet schema.

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

- Shared lookups loaded once and extended: validation target columns resolved dynamically and **deduplicated** (`list(dict.fromkeys(["era", "id", "target", target_20, target_60]))`, decision #18), benchmarks (`tier4_column`), meta. All filtered to the meta-era axis.
- Per model (and per tier-4 reference): `payout` = `payout_series(per_era_corr vs "target", per_era_mmc vs "target")` — **anchored to the primary target** (decision #19); `corr20`/`corr60` = `per_era_corr` vs the resolved 20D/60D columns; `mmc20`/`mmc60` = `per_era_mmc` with the matching target and `meta_col="numerai_meta_model"`; `bmc` = `per_era_bmc` vs `tier4_column`, short-circuited to `{era: 0.0}` when the model is the reference itself (decision #11); `cwmm` = `per_era_cwmm`.
- Missing horizon target column → that metric slice populated with zeros per era + warning; report generation never aborts (decision #23).
- Missing data assets → `{"eras": [], "meta_downside_mask": [], "metrics": {}, "drawdowns": {}}` + warning (never raise). Missing per-run preds → warning + skip that model across all metrics.

### `extract_pairwise_similarity_matrix`

- Candidates: registry preds for `run_ids` **plus** the benchmark column from `data/v5.3/validation_benchmark_models.parquet` (`tier4_column`) when `include_tier4_ref` — never a registry directory (decision #20). K = number of usable candidates (bounded by the caller to min(5, fleet) + ref, decision #5).
- Single multi-way inner join of all candidate frames on `[era, id]` (global intersection over the shared meta window) → aligned `(N_rows, K)` matrix.
- Per-era rank-gaussianization per column via `Ensembler.rank_normalize`; per era, zero-variance column → zeros; globally zero-variance column → 0.0 correlation rows.
- `matrix = np.clip(np.corrcoef(gaussianized.T), -1.0, 1.0).tolist()` (decision #22); returns `(labels, run_ids, matrix)`; deterministic (sorted ids); missing preds → warning + model dropped; missing data assets → `([], [], [])` + warning.

## 5. Presentation Layer — `dashboard_charts.py`

```python
def multimetric_chart_html(payload: dict[str, Any]) -> str: ...
def build_similarity_matrix_chart(labels: list[str], matrix: list[list[float]]) -> go.Figure: ...
def build_drawdown_chart(payload: dict[str, Any]) -> go.Figure: ...
def build_leaderboard_bar_chart(df: pl.DataFrame, *, hurdle_sharpe: float) -> go.Figure: ...  # unchanged
```

- **`multimetric_chart_html`** returns the full `<div id="multimetric-chart">` + `<script>` block: the payload embedded as `var payload = {...}` (`json.dumps(..., sort_keys=True)`), a fixed vanilla-JS controller scoped to the rendered root (decision #24) with state variables `currentMetric`/`currentView`, an HTML `<select>` (7 options) and two view `<button>`s, and `applyState()` rebuilding traces via `Plotly.react(root, traces, layout)` **including the `METRIC_CONFIG` y-axis title/tickformat per metric × view** (decision #21 — exact titles per review round 2). Stress-era `vrect` spans are fixed layout shapes from `meta_downside_mask`. Hovertemplate labels via a small JS escape helper. Empty payload → the "Timeseries data unavailable without local v5.3 assets" annotation.
- **Similarity chart**: static `go.Heatmap(z=matrix, x=labels, y=labels)`, `colorscale="RdBu_r"`, `zmid=0.5`, per-cell value annotations, labels escaped, **visual highlight on row/column 0** (top-ranked model, decision #25); empty matrix → annotation (no traces).
- **Drawdown chart**: traces from `payload["drawdowns"]` over `payload["eras"]`, `fill="tozeroy"` red fill; empty payload → the same unavailable annotation as v1.

## 6. HTML Layout (`artifacts/dashboard.html`)

Header: `Evaluation window: 86 overlap eras (1133–1218) · data v5.3`. KPI cards unchanged from v1. Sections:

1. ALPHA GENERATION & MULTI-METRIC PERFORMANCE TRAJECTORY (JS-controller chart; models = top-3 by Sharpe + tier-4 baseline)
2. RISK-ADJUSTED RETURN LEADERBOARD (unchanged)
3. SIGNAL DIVERSIFICATION & PAIRWISE SIMILARITY MATRIX — heatmap (top-5 fleet + tier-4 baseline, K per decision #5) preceded by the **Diversification Quality badge**: `Mean Overlap · Max Overlap` with EXCELLENT / MODERATE / HIGH REDUNDANCY coloring (decision #16) **plus the stress-regime correlation delta** (decision #26), and followed by the **Equal-Weight Ensemble Sharpe card** (top-3 blend; N_fleet < 3 → "—", decision #27)
4. EXECUTIVE ALLOCATION & RISK DECISION TABLE (unchanged)
5. CAPITAL DRAWDOWN (adapted chart)
6. Technical & audit accordion (unchanged)

Render-call accounting: 3 static plotly.py figures + 1 JS `Plotly.newPlot` = 4 calls after the single engine embed (`<!-- plotly-engine-embed -->` marker). Stable `div_id`s, no `<script src>` tags, no wall-clock. Size unbounded (director ruling).

## 7. Verification Plan

- **Unit tests (`tests/test_dashboard.py`)**, replacing the migrated v1 payout/drawdown tests:
  - Multimetric payload: exact key set; alignment 1:1 with `eras`; numeric era order; determinism; cumulative semantics (payout `1.05^3` wealth on the perfect-corr fixture; corr-family cumsum); **payout parity** (payout series equals the one `reconcile_capital_metrics` derives — same target anchor); `_resolve_horizon_targets` fallback chain; **dedup** (a schema where both horizons fall back to `"target"` loads without Polars duplicate-column errors); BMC short-circuit; **missing-horizon zeros + warning**; degradation to empty payload; missing-run skip.
  - Similarity matrix: identity diagonal; symmetry; identical-signal (incl. scale-shifted copy) → 1.0; zero-variance column guard; **clamp** (no `|ρ| > 1`); global inner-join alignment; **tier-4 column read from the benchmark parquet** (synthetic fixture without a registry dir for the ref); K-bound behavior (fewer than 5 fleet runs → smaller matrix); labels/ids order determinism; empty-data degradation.
  - Presentation: `multimetric_chart_html` output contains the exact serialized payload, the `multimetric-chart` div, `<select>` with 7 options, both view buttons, the `METRIC_CONFIG` titles, and no `updatemenus`/external script tags; similarity heatmap `z` equals the matrix with row-0 highlight; drawdown consumes the new payload; empty-payload annotations.
  - HTML: 4 render calls after the engine marker; "SIGNAL DIVERSIFICATION" section + badge (+ stress delta) + ensemble card present; ensemble card renders "—" for N_fleet < 3; embed marker once; no `<script src>`.
- **Real-data acceptance** (skip-marked, same convention): all 7 metric dicts populated for the real top run with 86-era arrays; payout cumulative terminal value equals the table's `cagr_1y` compounding within float tolerance; similarity matrix on the real top-5 + tier-4 baseline within `[-1, 1]`, diagonal 1, badge thresholds render.
- **Pre-sign-off gates (AGENTS.md §7):** `ruff check .` + `pytest -q` (full suite) + `generate_dashboard.py` → inspect the HTML (marker, no external scripts, four render calls, all sections present); open the file once in a browser to confirm the metric dropdown, view toggle, and axis titles behave (manual check — JS cannot be pytest-tested without a browser).

## 8. Scope Exclusions & Risks

- **Excluded:** Streamlit changes; Node/npm toolchain; server-side data-swap buttons; new dependencies; changes to gate semantics, KPI cards, or the executive table; any optimizer in the ensemble card (equal-weight only).
- **Risks:** (a) the JS controller is new surface — covered by structural tests plus the mandatory manual browser check at sign-off; (b) the tier-4 baseline appears in the multimetric chart, the matrix, and the leaderboard via the same `tier4_column` from `tier4_gate.yaml`; (c) v1's `build_cumulative_wealth_chart` is deleted — nothing outside the report consumed it; (d) the ensemble-Sharpe card is a heuristic (simultaneous-staking assumption) and is labeled as such in the HTML; (e) the stress-delta statistic is descriptive (mean pair shift), not a risk model.
