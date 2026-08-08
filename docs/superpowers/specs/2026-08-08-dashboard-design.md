# Streamlit + Plotly Dashboard — Design

> Status: drafted 2026-08-08 (user delegated: "plan your own work and proceed"). Sub-project 3 of the external-library grant (BO ✓ → CatBoost → dashboard).

**Goal:** An interactive Streamlit+Plotly dashboard over the registry scorecards, benchmark CSV, and campaign logs — read-only, thin control plane, reusing `nmr/` public APIs for all data shaping.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Dependencies | `streamlit==1.61.1`, `plotly==6.6.0` (already installed as transitive; pinned as direct) | User-granted; latest streamlit, plotly pinned to the installed version to avoid upgrade churn |
| Entry point | `dashboard_app.py` at repo root | Thin control plane; matches `generate_dashboard.py` placement |
| Data source | `RunRegistry.list()` entries + `artifacts/benchmark_scores*.csv` + `artifacts/campaigns/*.json` | All read-only; no data-layer joins in V1 |
| Business logic | Reuse `nmr` public APIs: `meta.fleet_summary`, `meta.promotion_verdict`, registry entries, scorecard cells | No logic in the script (AGENTS §2.1); shaping helpers are pure and tested (generate_dashboard precedent) |
| Era-series charts | **Deferred to V2** | Requires data-layer joins (oof + meta + targets); the request targets "scorecards and robustness reports", which V1 covers fully |

## Architecture

`dashboard_app.py` = pure data-shaping helpers (unit-tested via `tests/test_scripts.py`, mirroring `generate_dashboard._build_html`) + a thin Streamlit render layer (not unit-tested; smoke via import).

Views (V1):

1. **Leaderboard** — registry runs + benchmark CSV merged, Sharpe-ranked, Plotly bar chart with `corr_sharpe_ac` CI error bars (`_ci_low`/`_ci_high` from scorecard cells), filterable by `backend`/`preset`/`feature_set`/`source`. Champion (`champion.json`) highlighted.
2. **Run detail** — all scorecard cells (value + CI + n_eras for corr/mmc/corr_sharpe_ac/mean_payout), robustness flags (bmc/horizon/perturb/regime presence + values), manifest summary (weights, `oof_device`, feature cols, config, `resolved_device`), run_dir path.
3. **Fleet analysis** — `meta.fleet_summary(registry.list(), n_trials=..., dsr_confidence=...)` table: per-run metric cells, DSR pass/fail, `oof_device`, grouping attrs (preset/feature_set/feature_subset/neutralization_proportion), robustness flags; scatter of neutralization_proportion vs `corr_sharpe_ac`.
4. **Campaign browser** — `artifacts/campaigns/*.json`: hypothesis → config paths → run_ids → statuses (recorded/skipped/error); links into the leaderboard.
5. **Robustness matrix** — Plotly heatmap of per-run robustness indicators (`has_bmc`, `has_horizon`, `has_perturb`, `has_regime`, `max_feature_exposure`, `std_corr`, `max_drawdown`) — the "outliers: good headline CORR, poor robustness" view the meta-analysis skill targets.

Shaping helpers (pure, tested): `load_registry_frame(registry_dir) -> pl.DataFrame`, `load_benchmarks(path) -> pl.DataFrame`, `merge_leaderboard(registry, benchmarks) -> pl.DataFrame` (shared columns; `source` = trained/benchmark), `load_campaigns(campaigns_dir) -> pl.DataFrame`, `robustness_matrix(registry) -> pl.DataFrame`. All column-selection/renaming only — no metrics computed (scorecard cells already carry values; `fleet_summary` is the only computation and it lives in `nmr/meta.py`).

## Testing

- `tests/test_scripts.py` (extend): pure-helper tests — registry-frame columns + source tagging (trained vs trained_legacy vs benchmark), benchmark normalization, leaderboard merge (benchmark rows join with `source="benchmark"`), campaign load schema, robustness-matrix columns; empty-input handling (empty registry dir, missing benchmark CSV). Import-time smoke for `dashboard_app` (module imports without launching the server).
- No Streamlit server launch in tests (headless CI); the render layer is thin by construction.
- Test count: +8–10 → count sync (established precedent).

## Run contract

`streamlit run dashboard_app.py` — documented in README quickstart (one line) and ARCHITECTURE §O. The app reads only `artifacts/` (registry, campaigns, benchmark CSVs) — never writes.

## Docs (same change set)

- `README.md`: tree entry for `dashboard_app.py` + one-line quickstart (`streamlit run dashboard_app.py`).
- `ARCHITECTURE.md`: §O table row (dashboard_app.py: reads registry + benchmark + campaigns; `streamlit run`; read-only).
- `AGENTS.md`: toolkit row "inspect runs / campaigns interactively | `dashboard_app.py`" — optional, low value; include only if it fits the table's "If you need to…" pattern (it does: "inspect runs interactively").
- `requirements.txt`: pins.
- Doc-SSOT count sync.

## Out of scope

- Era-series / per-era CORR charts (V2: needs oof+meta+targets joins through the runner/evaluation path).
- Write paths (no promotions, no registry mutations from the app — read-only by design).
- Authentication/remote deployment (local research tool).
