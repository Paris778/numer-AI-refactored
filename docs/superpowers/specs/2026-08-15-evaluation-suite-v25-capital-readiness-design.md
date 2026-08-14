# Quantitative Evaluation Suite v2.5 (Capital-Readiness Engine) — Design Spec

- **Date:** 2026-08-15
- **Status:** Design approved (Director dispositions: strict-MMC-down locked, raw-series
  Kelly locked, Approach A locked, Part 2 approved). Implementation pending user review of
  this spec.
- **Scope:** Add six capital-readiness metrics to the evaluation suite: annualized geometric
  stake return (CAGR 1Y), gain-to-pain ratio (GPR), bounded empirical Kelly fraction,
  downside meta-model contribution (MMC_down), cross-era prediction turnover, and a
  multi-round overlapping capital-velocity simulator.
- **Target systems:** `nmr/payout.py`, `nmr/evaluation.py`, `nmr/scorecard.py`.
  `nmr/inference.py`, `nmr/runner.py`, `nmr/benchmark.py`, `nmr/campaign.py` are unchanged.

## 1. Context & Problem

The current evaluation engine (`Evaluation Suite v2`) computes statistical metrics (CORR,
MMC, FNC, BMC, CWMM) and downside diagnostics (CVaR, max drawdown, Sortino, Calmar, burn
streak) over the arithmetic mean round payout. It does not answer capital questions:
geometric volatility drag, multi-round capital lockup and cash drag (up to 20 overlapping
20D rounds), regime-conditioned hedge contribution, or prediction turnover friction.

All six new metrics are derivable **inside** `evaluate_model` from the already-joined `base`
frame (era/id/pred/meta/target/features). No new inputs, no changes to runner, benchmark,
campaign, or their callers.

Empirical grounding (v5.3 validation, measured 2026-08-15):

- Meta-model coverage in the scorecard join spans 86 eras; the meta model's per-era CORR is
  negative in 6 of those 86 (~7%). MMC_down with the strict `CORR_meta < 0` definition and
  `M_min = 5` is barely non-vacuous today and will legitimately return `None` on shorter
  windows — this is the fail-loud convention, not a defect.
- On the ±5%-clipped payout series, μ/σ² is ~40–60 for any competitive model, so a bounded
  Kelly fraction computed on clipped returns saturates at 1.0 and carries no information.
  Kelly is therefore computed on the **raw unclipped** series (director-locked).

## 2. Locked Decisions (director dispositions)

1. **MMC_down — strict:** `E_down = {t : CORR_meta(t) < 0.0}`; `M_min = 5`; `None` +
   reason code `"insufficient_downside_eras"` when `|E_down| < 5`. No tercile or drawdown
   variants (they duplicate `regime_conditioned_corr` and are path-dependent).
2. **Kelly — raw series:** `kelly_fraction = min(1.0, max(0.0, μ_raw / σ²_raw))` with
   `ddof=0`; 0.0 when `σ² = 0` or `μ ≤ 0`. Clipped-series Kelly was rejected
   (Popoviciu-bounded variance → degenerate step function). Consistent with the existing
   precedent that `payout_report` routes `series.raw` (not clipped) to `deflated_sharpe`.
3. **Architecture — Approach A:** extend `PayoutResult`, `MetricScorecard`, and
   `evaluate_model` in place; pure helpers live in `payout.py` / `evaluation.py`.
4. **Canonical bytes — include new fields:** the new metrics are deterministic pure
   functions of the ordered era series (no wall-clock, no RNG, no paths), so they belong in
   the canonical payload exactly like `sortino`/`calmar`. Stripping remains reserved for
   `timing_*` and `quality_metric_*` columns. This is a deliberate deviation from the RFC's
   "volatile simulation keys" wording, which mischaracterizes the simulator as non-
   deterministic — it is not.

## 3. Mathematical Contracts

Let `E = {e_1 … e_N}` be the chronologically sorted evaluation eras. `r_t =
clip(pf·(0.75·CORR_t + 2.25·MMC_t), −clip, +clip)` is the clipped round return (existing
`payout_series`); `raw_t = pf·(0.75·CORR_t + 2.25·MMC_t)` is the unclipped return.

### 3.1 `annual_compounded_return(clipped, *, eras_per_year=52.0) -> float`

```
CAGR = (∏(1 + r_t))^(52/N) − 1   if ∏(1 + r_t) > 0
     = −1.0                      if ∏(1 + r_t) ≤ 0
     = 0.0                       if N < 2
```

All arithmetic in float64. Input validated via the existing `_as_finite_1d` (1-D, non-empty,
finite-only; raises otherwise).

### 3.2 `gain_to_pain_ratio(clipped) -> float`

```
GPR = Σ max(0, r_t) / Σ |min(0, r_t)|
    = +inf   if Σ|r⁻| = 0 and Σr⁺ > 0
    = 0.0    if Σ|r⁻| = 0 and Σr⁺ = 0
```

`+inf` is precedented (`calmar` returns it when mdd = 0 with positive mean).
`_sanitize_json_payload` maps non-finite floats to `"Infinity"` strings in canonical bytes;
parquet/CSV carry `inf` natively. No crash path.

### 3.3 `kelly_fraction(raw, *, ...) -> float` (raw series, director-locked)

```
μ = mean(raw), σ² = var(raw, ddof=0)
kelly = min(1.0, max(0.0, μ / σ²))   if σ² > 0 and μ > 0
      = 0.0                          otherwise
```

The helper validates its input like all payout helpers; `payout_report` passes `series.raw`.
Bounded strictly in [0, 1].

### 3.4 MMC_down (strict, director-locked)

```
E_down = { t : CORR_meta(t) < 0.0 }            # per-era meta-model CORR vs main target
mmc_down        = mean(MMC_t for t ∈ E_down)   if |E_down| ≥ 5
                = None                          otherwise
mmc_down_n_eras = |E_down|                      always
mmc_down_reason = None                          if |E_down| ≥ 5
                = "insufficient_downside_eras"  otherwise
```

`CORR_meta(t)` uses the same custom per-era CORR path as the model's own CORR
(`power_1_5(rank_gaussianize(meta))` vs centered target) — the engine's existing
`per_era_corr` with `pred_col=meta_col`.

### 3.5 `per_era_turnover(df, *, pred_col, era_col="era", id_col="id") -> dict[str, float]`

For consecutive chronological eras `e_{k−1}, e_k` with shared-ID set `I_k`:

```
ρ_k        = Spearman(pred_{k−1}|I_k, pred_k|I_k)
Turnover_k = 1.0 − ρ_k                    # bounded [0, 2]
ρ_k        = 0.0 if not finite            # degenerate transition → Turnover = 1.0
```

Transitions with `|I_k| < 10` are skipped. Missing `era_col`/`id_col`/`pred_col` raise
`ValueError`. The scorecard aggregates `turnover_mean` and `turnover_std` (ddof=0) over the
valid transitions; fewer than 2 valid transitions → both `None`.

### 3.6 `simulate_overlapping_portfolio(clipped, *, horizon_eras=20, initial_capital=1.0, eras_per_year=52.0) -> OverlappingSimulationResult`

Multi-round lockup simulator (RFC algorithm verbatim): at each era, tranches maturing at `t`
pay `principal·(1 + r_{t−K})` (the initiating era's return — correct round semantics); then
`min(cash, total_equity/K)` is deployed as a new tranche maturing at `t + K`. Outputs:
`portfolio_cagr` (geometric growth of final equity, annualized), `portfolio_max_drawdown`
(peak-to-trough of the equity curve), `avg_capital_utilization` (mean locked/total-equity),
`final_equity`.

Documented definitional properties:

- Equity and utilization are recorded **before** the era's deployment.
- Returns on tranches initiated in the final K eras remain unrealized in `final_equity`
  (still locked at end of series).
- `n < horizon_eras` → zeroed result (`cagr=0`, `mdd=0`, `util=0`, `final=initial_capital`).

Horizon mapping via module-level constant `_HORIZON_ERAS = {"20D": 20, "60D": 60}` — no
magic values; `payout_report` derives `horizon_eras` from its existing `horizon` argument.

## 4. Module Boundaries & File Plan

### 4.1 `nmr/payout.py` (pure NumPy only)

- New frozen dataclass `OverlappingSimulationResult`.
- New pure helpers: `annual_compounded_return`, `gain_to_pain_ratio`, `kelly_fraction`,
  `simulate_overlapping_portfolio`.
- `PayoutResult` gains `cagr_1y: float`, `gain_to_pain_ratio: float`, `kelly_fraction:
  float`, `overlapping_sim: OverlappingSimulationResult | None = None` (default preserves
  existing direct constructions).
- `payout_report` computes the three scalars (CAGR/GPR on `series.clipped`, Kelly on
  `series.raw`) and the simulator; returns the extended `PayoutResult`.
- Module docstring must explicitly state the terminal-tranche at-cost accounting
  convention: tranches still locked at the end of the series are carried at par
  principal (no mark-to-market, no unrealized payoff) in `final_equity` and the equity
  curve (panel-required documentation fix for §3.6).
- `__all__` updated.

### 4.2 `nmr/evaluation.py`

- New pure helper `downside_era_indices(meta_corr: Mapping[str, float], *, threshold:
  float = 0.0) -> list[str]` — chronological sort via existing `sorted_era_labels`
  (fail-loud on non-numeric labels).
- New `per_era_turnover` (polars + scipy `spearmanr`, both already dependencies).
- `__all__` updated.

### 4.3 `nmr/scorecard.py`

- `evaluate_model` computes, all timed via the existing `_mark` pattern:
  1. `meta_corr_by_era = evaluator.per_era_corr(base, pred_col=meta_col,
     target_col=main_target)` → `downside_era_indices` → `mmc_down` / `mmc_down_n_eras` /
     `mmc_down_reason`.
  2. `per_era_turnover(base, ...)` → `turnover_mean` / `turnover_std`, or `None` +
     `turnover_reason`:
     - `"id column unavailable"` when `base` lacks `id_col`;
     - `"insufficient_transitions"` when fewer than 2 valid transitions.
  3. Payout fields → `cagr_1y`, `gain_to_pain_ratio`, `kelly_fraction`,
     `sim_portfolio_cagr`, `sim_portfolio_mdd`, `sim_capital_utilization`.
- `MetricScorecard` gains 12 required fields (no defaults — fail-loud convention):

| Field | Type | Origin |
|---|---|---|
| `cagr_1y` | `float` | `payout.cagr_1y` |
| `gain_to_pain_ratio` | `float` | `payout.gain_to_pain_ratio` |
| `kelly_fraction` | `float` | `payout.kelly_fraction` |
| `mmc_down` | `float \| None` | mean MMC over `E_down` when ≥ 5 eras |
| `mmc_down_n_eras` | `int \| None` | `\|E_down\|` |
| `mmc_down_reason` | `str \| None` | `"insufficient_downside_eras"` or `None` |
| `turnover_mean` | `float \| None` | mean of `1 − ρ_k` |
| `turnover_std` | `float \| None` | population std (ddof=0) of transitions |
| `turnover_reason` | `str \| None` | see 4.3.2 |
| `sim_portfolio_cagr` | `float` | `overlapping_sim.portfolio_cagr` |
| `sim_portfolio_mdd` | `float` | `overlapping_sim.portfolio_max_drawdown` |
| `sim_capital_utilization` | `float` | `overlapping_sim.avg_capital_utilization` |

- `to_frame()` flattens all 12 columns.

### 4.4 `nmr/__init__.py`

New public symbols added to imports **and** `__all__`: `OverlappingSimulationResult`,
`annual_compounded_return`, `gain_to_pain_ratio`, `kelly_fraction`,
`simulate_overlapping_portfolio`, `downside_era_indices`, `per_era_turnover`.

## 5. Determinism & Canonical Hash Policy

- All six metrics are deterministic: float64 pure math (`np.prod`, `np.mean`, `np.var`),
  rank-based Spearman (invariant to row order within eras), no wall-clock, no RNG, no
  absolute paths. They participate in `canonical_scorecards_bytes` (default behavior —
  only `timing_*` / `quality_metric_*` columns are stripped; no new stripping logic).
- `gain_to_pain_ratio = +inf` is sanitized to the string `"Infinity"` in canonical JSON by
  the existing `_sanitize_json_payload` (same path `calmar` already uses).
- Cross-process determinism tests (`tests/test_benchmark_slice1.py`,
  `tests/test_scorecard.py::test_scorecard_real_v52_determinism_cross_process`) compare two
  independent runs rather than pinned literals, so including new deterministic columns is
  safe. Any pinned-literal hash will surface as RED in TDD and be regenerated deliberately.

## 6. Error & Degenerate Handling

- All payout helpers validate via `_as_finite_1d` (1-D, non-empty, finite) — raises
  `ValueError` otherwise. No silent coercion.
- `kelly_fraction` with `σ² = 0` or `μ ≤ 0` → `0.0` (defined, not degenerate).
- `payout_report` continues to require ≥ 2 overlapping eras (unchanged); therefore
  `evaluate_model` always produces sim fields (never `None` there).
- `mmc_down` / `turnover` legitimately absent → `None` + reason code, matching the existing
  `horizon_reason` / `bmc_reason` / `regime_reason` convention.
- Turnover with constant predictions in one era → non-finite ρ → ρ = 0.0 → Turnover = 1.0
  (bounded, spec'd behavior).
- `per_era_turnover` on frames missing required columns raises `ValueError`.

## 7. Test Plan (TDD — RED first)

`tests/test_payout.py`:

1. `test_annual_cagr_math` — +1% over 52 eras → (1.01)^52 − 1.
2. `test_annual_cagr_ruin` — one era at −100% → −1.0 exactly; `n < 2` → 0.0.
3. `test_gain_to_pain_ratio` — (0.03, 0.03, 0.03, −0.01) → 9.0.
4. `test_gain_to_pain_zero_burn` — all positive → `inf`; all zero → 0.0.
5. `test_kelly_fraction_bounds` — bounded [0, 1]; σ²=0 or μ≤0 → 0.0.
6. `test_kelly_uses_raw_not_clipped` — contract: a series whose clipped variant saturates
   (returns 1.0) must yield a strictly smaller raw-series fraction (director-locked input).
7. `test_overlapping_sim_drag` — volatile 20D lockup series: portfolio CAGR below the serial
   geometric product (volatility drag).
8. `test_overlapping_sim_short_series` — `n < horizon_eras` → zeroed result.
9. `test_overlapping_sim_lockup_math` — constant positive returns: tranche accounting
   verified against hand-computed equity path.

`tests/test_evaluation.py` (or existing home):

10. `test_per_era_turnover_identical` — identical preds across eras → 0.0.
11. `test_per_era_turnover_inverse` — inverted preds → 2.0.
12. `test_per_era_turnover_skips_small_intersection` — `|I_k| < 10` transitions skipped;
    < 2 valid transitions → mean/std `None`.
13. `test_per_era_turnover_missing_columns` — raises `ValueError`.
14. `test_downside_era_indices_strict` — slices exactly the `CORR < 0` eras, chronological;
    non-numeric labels raise.

`tests/test_scorecard.py`:

15. `test_mmc_down_filtering` — meta CORR < 0 in 10 of 30 eras → MMC_down = mean over the
    exact 10; `n_eras = 10`; reason `None`.
16. `test_mmc_down_insufficient` — 2 downside eras → `None` + `"insufficient_downside_eras"`.
17. `test_scorecard_to_frame_new_columns` — all 12 new columns present with expected types.
18. `test_scorecard_capital_metrics_flow_from_payout` — a synthetic evaluate_model run
    carries consistent CAGR/GPR/Kelly/sim values between `PayoutResult` and the scorecard
    row.
19. Determinism: existing two-run `scorecards_sha256` equality tests must stay green
    (new fields are deterministic).

SSOT & hygiene:

20. `tests/test_docs_hygiene.py` test-count claims updated (629 → new total) in `AGENTS.md`
    (two spots) and the renderer recipe comment.

## 8. SSOT Documentation Updates (same commit — AGENTS.md self-update directive)

- `ARCHITECTURE.md`: scorecard schema and payout sections (12 new fields, new helpers,
  module map rows for `downside_era_indices` / `per_era_turnover`).
- `docs/06-evaluation/evaluation-suite-bible.md`: definitions of the six metrics, the
  raw-series Kelly deviation with rationale, the strict MMC_down contract, the `+inf` GPR
  convention, simulator definitional properties, canonical-bytes inclusion rationale.
- `AGENTS.md`: test-count claims only (no structural change).
- `CONTRIBUTING.md` / `README.md`: no changes (no workflow or setup impact).

## 9. Out of Scope

- Changes to `nmr/inference.py`, `nmr/runner.py`, `nmr/benchmark.py`,
  `nmr/campaign.py`, `nmr/opt.py`, `nmr/meta.py` (consumers are unaffected).
- Regime-conditioned or quantile-based MMC variants (rejected — duplicate
  `regime_conditioned_corr`).
- Fractional/half-Kelly scaling, drawdown-based sim extensions, live-round tracking.
- Journal-writing or dashboard surfaces.
