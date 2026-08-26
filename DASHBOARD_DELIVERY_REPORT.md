# 🏆 Numerai Top Performers Dashboard — DELIVERY REPORT

> **Historical record:** This report describes the retired four-tab dashboard and
> is superseded by [DASHBOARD_DELIVERY_SUMMARY.md](DASHBOARD_DELIVERY_SUMMARY.md),
> which documents the current offline Model Tournament renderer and verification state.

**Date:** 2026-08-25 | **Status:** Historical; superseded

---

## 1. HOW TO VIEW YOUR NEW DASHBOARD

The dashboard is **running right now** on this machine.

**Option A — It's already open:**
Open your browser to **http://localhost:8502**

**Option B — Restart it later (e.g., after reboot):**
```powershell
cd c:\dev\numer-AI-refactored
.\.venv\Scripts\streamlit run dashboard_app.py --server.port 8502
```
Then open http://localhost:8502

> The first time you open the **📈 Era-Level** tab it takes ~12 seconds to compute
> the per-era analytics (it's cached afterwards, so it's instant on every later load).

---

## 2. WHAT CHANGED — AND WHY

You told me clearly: **"I'm not gonna hedge on multiple models. I just want to see
my top performers and I will decide manually how much I want to invest in each model."**

So I **threw out the hedging/portfolio-optimization/scenario-builder framing** and
rebuilt the dashboard around **your top performers and the full evidence you need
to make a manual capital call on each one.** No black-box "recommended allocation" —
just ranked performance and the raw data behind it.

The dashboard has **4 tabs**:

### 🏆 Tab 1 — Top Performers (the main view)
- **KPI cards** for your #1 model: Sharpe, CORR, Max Drawdown, Robustness score.
- **A sortable, filterable leaderboard** ranking all 30 evaluable models by
  CORR Sharpe (switchable to raw CORR, Deflated Sharpe, or Era Stability).
- Each row shows: **Sharpe (+95% CI), CORR, Deflated Sharpe, Stability (σ),
  Max Drawdown, Robustness (x/4)** — every number you need to judge a model.
- **Expand any model** to read a human explainer: what it is, why it was built,
  its risk profile, diversification, and research intent. No more opaque hash IDs.

### 📈 Tab 2 — Era-Level (is it consistent or lucky?)
This is the honest-answer tab. Using your **real validation predictions**, it plots
per-era trajectories (86 eras, 1133–1218) for any models you pick:
- **Cumulative Payout** (compounded), **CORR 20D / 60D**, **MMC 20D / 60D**
- **Drawdown chart** from the real cumulative payout path
- **Red bands = meta-model downside eras** — so you can see exactly how each model
  behaves when the market meta is weak (the #1 thing to check before staking capital).
- An **era summary table**: mean, std, min/max, % positive eras, longest win streak.

### 📊 Tab 3 — Head-to-Head
Pick 2–4 models and compare:
- **Risk vs Return scatter** (bubble size = drawdown) — the ideal models sit top-left.
- **Side-by-side metric table** (Sharpe, CORR, CI, Deflated, Stability, Drawdown).
- **Cumulative payout overlay** — watch their wealth paths diverge.

### 📋 Tab 4 — Model Directory
Every model in the catalog with architecture + risk badges and an expandable
full explainer. Filter by architecture or tag.

---

## 3. FILES CREATED / CHANGED

**New (the engine of the dashboard):**
- `dashboard_ui/service.py` — added `compute_top_performers()` (typed, ranked result
  with the full decision surface) and `load_timeseries()` (memoized real per-era data).
  New Pydantic models `TopPerformerRowModel` / `TopPerformersResult`.
- `nmr/explainers.py` — model explainer system. Every registry model now gets a
  human-readable profile (static curated catalog + dynamic generation from `run.json`).
- `nmr/scenarios.py` — kept (tested) but **not** surfaced in the UI, per your direction.
- `dashboard_ui/tables.py` — sortable table components (fixed `sort_column` support,
  removed the buggy ProgressColumn that caused Vega warnings).
- `dashboard_ui/charts_components.py`, `dashboard_ui/components.py` — reusable
  component layer from the earlier phase (chart primitives, design tokens).

**Rewritten:**
- `dashboard_app.py` — the 4-tab Top Performers dashboard above.

**Tests added:**
- `tests/test_dashboard_service.py` (+10: top performers ranking, min-sharpe floor,
  typed rows, JSON roundtrip, timeseries memoization)
- `tests/test_explainers.py` (+1: dynamic profile generation from real registry)
- `tests/test_scenarios.py`, `tests/test_dashboard_tables.py`,
  `tests/test_dashboard_charts_components.py`

---

## 4. TEST RESULTS (honest)

```
91 passed  — all dashboard service, components, tables, explainers, scenarios,
             and original dashboard_ui tests
```

**2 pre-existing failures** in `tests/test_dashboard.py` (NOT caused by this work):
- `test_real_recompute_matches_stored_corr`
- `test_real_multimetric_payload_and_payout_parity`

These are real-data drift tests that recompute metrics from current validation
preds and compare them to **stored scorecard cells from pre-rebuild registry rows**.
That mismatch is the **documented hazard in `AGENTS.md`** ("stale era-range manifest
fields in pre-rebuild registry rows"). I did not touch `nmr/dashboard.py` — the
recompute logic is unchanged. Flagging so you can decide whether to refresh those
registry rows (a deliberate data-refresh act, not something I'd do unilaterally).

---

## 5. WHAT THE NUMBERS MEAN (quick reference)

| Metric | Meaning |
|---|---|
| **Sharpe (CORR Sharpe)** | Risk-adjusted return; higher = better |
| **CORR** | Raw era correlation; the raw signal strength |
| **Deflated Sharpe** | Probability the Sharpe is genuine, not luck (≈1.0 = strong) |
| **Stability (σ)** | Std of per-era CORR; **lower = more consistent** |
| **Max DD** | Max drawdown **magnitude** (16% = 16% peak-to-trough) |
| **Robustness (x/4)** | How many stress checks exist (BMC, horizon, perturbation, regime) |
| **% > 0 eras** | Share of eras where the model made money |

---

## 6. WHAT I'D LOOK AT FIRST (based on the real data)

Top of your leaderboard right now:

| Rank | Model | Sharpe | CORR | Max DD | Robustness |
|---|---|---|---|---|---|
| 1 | mt-std-v1 · 2c5e5f39 | 0.588 | 0.0142 | 15.8% | 2/4 |
| 2-4 | brb1-lgbm-v6 (3 runs) | 0.480 | 0.0132 | 18.8% | 1/4 |
| 5-7 | brb1-xgb-v6 (3 runs) | 0.464 | 0.0143 | 27.9% | 1/4 |
| 8 | brb1-lgbm-v1 · 93a69643 | 0.461 | 0.0084 | 19.7% | 1/4 |

**Before investing, open 📈 Era-Level → Cumulative Payout** and check whether the
top model's edge is smooth or one lucky streak, and how it behaves in the red
(downside) bands. That's the whole point of this dashboard: **you see the evidence,
you make the call.**

---

*Prepared by GitHub Copilot. All claims verified by tests + live browser verification
of every tab.*
