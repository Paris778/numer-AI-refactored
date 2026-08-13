# Campaign-Aware DSR Trial-Tracking — Design Spec

- **Date:** 2026-08-14
- **Status:** Draft — pending user review (Director disposition: Option 1 post-hoc recompute; Option 1 IC-Sharpe full-window deflation)
- **Scope:** Phase 1, item 4: campaign/sweep-level Deflated Sharpe Ratio with `n_trials > 1` and
  empirical cross-trial Sharpe variance. Single-run scorecards and `nmr/runner.py` stay untouched
  (`n_trials = 1` remains the standalone contract — backward compatible).

## 1. Context & Problem

Single-run scorecards compute DSR with `n_trials = 1` (committee Red Flag 6): during feature
exploration across dozens of cells/trials, the multiple-testing bar is absent and p-values are
over-optimistic. The cumulative trial count and the cross-trial Sharpe variance are only knowable
**after** a campaign/sweep completes, so the correction belongs in the post-hoc aggregation layer —
never in per-run scorecards (which must stay deterministic and self-contained).

## 2. Mathematical Contract (locked)

For a campaign of `N_cells` recorded runs with full-window validation IC series `X_{i,t}`
(t = 1..N_obs, numeric-ordered eras; N_obs up to 649):

1. Per-run moments, all from the **same series** (the same-distribution invariant):
   - `SR_i = mean_i / std_i` (std ddof = 0, matching `MetricSummary`),
   - `skew_i`, `kurt_i` via `era_series_stats` (skew bias=False; kurtosis fisher=False, bias=False).
2. Empirical cross-trial Sharpe variance: `trials_sr_var = Var({SR_k}, ddof=1)`
   (≥ 2 recorded cells required; the bible's rule — no analytic fallback, ever).
3. Campaign-aware DSR per run:
   `DSR_i = deflated_sharpe(SR_i, n_trials=N_cells, n_obs=N_obs, skew=skew_i, kurt=kurt_i,
   trials_sr_var=trials_sr_var, sr0_benchmark=0.0)`.
4. `dsr_pass_campaign = DSR_i >= 0.95` (the fleet_summary `dsr_confidence` convention).

Deflation target: the **full-window validation IC Sharpe**. (The scorecard's 86-era
`corr_sharpe_ac` and the payout Sharpe are explicitly rejected as targets: the former mixes an
86-era Sharpe with full-window moments; the latter would silently shrink every campaign to the
meta-model's 86-era overlap — both violate the same-distribution rule.)

## 3. Components

### `nmr/meta.py` — `campaign_evidence`

The IC series machinery already exists (numeric-ordered `ic_frames` per recorded cell). Extend the
variants assembly:

- Per recorded cell with `n_eras >= 4` and `std(ics) > 0`: compute `SR`, `skew`, `kurt` via
  `era_series_stats`.
- Across recorded cells (≥ 2 with valid moments): `trials_sr_var = np.var(sharpe_vector, ddof=1)`.
- Per cell: `DSR_i` via `deflated_sharpe`. Any `ValueError` (radicand ≤ 0, non-finite moments) or
  degenerate series (zero variance, n_eras < 4) → `dsr_campaign_aware = None` +
  `dsr_reason` — evidence assembly **never crashes**.
- New `CampaignEvidence.variants` columns: `dsr_campaign_aware`, `dsr_pass_campaign`,
  `dsr_reason`, `dsr_n_trials`, `dsr_trials_sr_var` (the last two constant across the table).

### `nmr/research.py` — held-out moments (single training pass)

`_held_out_metric` currently discards the per-era series after reducing to the metric. Add
`_held_out_metric_full(config, *, metric_name) -> tuple[float, _HeldOutMoments]` where
`_HeldOutMoments` (frozen dataclass) carries `ic_sharpe`, `ic_skew`, `ic_kurt`, `ic_n_eras` from
the same per-era CORR dict. `_held_out_metric` delegates to it and returns only the float
(public contract unchanged). No extra training.

### `nmr/opt.py` — `SweepResult` post-hoc DSR

- `HyperparameterSweep.run` and `bayesian_sweep` capture `_HeldOutMoments` per trial and store
  them as `SweepResult.trials` columns (`ic_sharpe`, `ic_skew`, `ic_kurt`, `ic_n_eras`).
- New pure helper `sweep_dsr(trials: pl.DataFrame) -> pl.DataFrame` (in `nmr/opt.py`): over
  COMPLETE trials with valid moments, `n_trials = N`, `trials_sr_var = np.var(ic_sharpes, ddof=1)`,
  per-trial `deflated_sharpe` with the same degenerate/error handling → adds `dsr_sweep_aware`,
  `dsr_pass_sweep`, `dsr_reason`, `dsr_n_trials`, `dsr_trials_sr_var` columns. Sweeps with < 2
  valid trials get DSR columns = None (documented, not an error).

### `nmr/scorecard.py`, `nmr/runner.py`

**Untouched.** Standalone scorecards keep `n_trials = 1`.

## 4. Determinism & Read-Only Guarantees

All new computations are pure functions of recorded artifacts (parquets/registry JSONs) and trial
frames. No registry writes, no new stochastic ops, no canonical-hash or run_id changes.
`validation_preds.parquet` rows are already numeric-ordered before any DSR computation.

## 5. Testing Plan

- `nmr/meta.py`: synthetic two-run campaign (engineered IC series) → exact `trials_sr_var` value,
  DSR matches a hand-computed `deflated_sharpe` call; degenerate cell (constant IC) → None +
  reason; single-recorded-run campaign → DSR None (needs ≥ 2); radicand-failure cell → None +
  reason, assembly still succeeds; determinism (two calls equal).
- `nmr/research.py`: `_held_out_metric_full` equals `_held_out_metric`'s value and carries
  consistent moments (skew/kurt match `era_series_stats` on the recomputed series) on synthetic
  data.
- `nmr/opt.py`: `HyperparameterSweep` trials frame carries moment columns; `sweep_dsr` produces
  the hand-checked DSR for a 3-trial sweep and None-columns for a 1-trial sweep.
- Full suite + benchmark smoke per the verification gate.

## 6. Out of Scope

- `fleet_summary`/`promotion_verdict` campaign-aware DSR (registry-level; needs a declared trial
  grouping — revisit if the meta-analysis protocol demands it).
- Any change to per-run scorecards, `run_id` payloads, or the evaluation bible's DSR formula.
