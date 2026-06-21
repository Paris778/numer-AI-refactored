# Numerai Classic Strategy Bible

This document is the consolidated tactical guide for Numerai Classic.

It merges and refactors the prior bibles into a single operating reference, with emphasis on constraints that drive live utility: leakage control, rank-domain scoring, stability, and uniqueness.

## 1) Operating Principles

- Model the environment as noisy, non-stationary, low-signal tabular time series.
- Optimize for durable per-era performance, not one-shot global fit.
- Treat eras as the natural statistical unit.
- Seek both alpha (CORR) and orthogonality (MMC).
- Prefer robust, simple ensembles over single brittle models.

## 2) Data Reality

- Obfuscated features and IDs force statistical learning without ticker tracking.
- Main target is stock-specific residual return; auxiliary targets provide alternate residualizations and horizons.
- Feature values may be missing; target main is not.
- Feature predictive power is unstable over time, so exposure management matters.

## 3) Validation Doctrine

Never do random row-level CV.

Use era-grouped walk-forward with purge/embargo.

Recommended operational convention in this repository:

- 20D: 8-era purge (benchmark convention)
- 60D: 16-era purge

Minimum theoretical overlap-safe convention:

- 20D: 4-era embargo
- 60D: 16-era embargo

Evaluate every serious candidate with:

- Per-era CORR mean
- Per-era Sharpe
- Cumulative drawdown behavior
- MMC profile
- FNC and feature exposure diagnostics

## 4) Scoring Geometry You Must Respect

### CORR

- Rank-based, gaussianized, tail-accented (power 1.5).
- Absolute prediction magnitude is secondary to ordering.
- Tail ordering quality is disproportionately rewarded.

### MMC

- Measures contribution after removing projection on the meta model.
- High CORR with high crowd similarity often yields weak MMC.
- Distinctive but target-aligned signal is valuable.

### FNC

- Tests whether your signal survives feature neutralization.
- Helps detect fragile linear feature dependence.

## 5) Feature Exposure and Neutralization

Use neutralization as a risk lever, not an ideology.

- Full neutralization: strongest exposure control, possible alpha loss.
- Partial neutralization: often better practical trade-off.
- Community experience suggests later eras can prefer smaller neutralization proportions (for example 0.25-0.5).

Track:

- Mean max feature exposure
- Exposure stability across eras
- Sharpe and drawdown before and after neutralization

## 6) Target Ensembling Doctrine

Train target specialists, then combine in rank space.

Rules:

- Do not average raw regression outputs directly.
- Rank/percentile each component per era first.
- Blend ranked components, then optionally re-normalize.

Useful stack shape:

1. Level 1: seed ensemble within each target
2. Level 2: target ensemble across complementary targets
3. Optional post-process: partial neutralization

Candidate targets to mix typically include the main target and low-to-moderate correlation auxiliaries.

## 7) Model Family and Hyperparameter Baseline

Tree ensembles remain strong practical baselines for v5-style data.

Representative high-capacity baseline:

- n_estimators: 20000 to 30000
- learning_rate: 0.001
- max_depth: 6 to 10
- num_leaves: 64 to 1024
- colsample_bytree: 0.1
- min_data_in_leaf: large (for example 5000 to 10000)

Interpretation:

- Heavy regularization and low learning rate are used to survive noise and regime drift.
- Feature subsampling is central for robustness.

## 8) Deployment Contract

The hosted model-upload path expects a serialized function, not just a fitted estimator.

Exact contract:

- Expose a function with this signature:
  `predict(live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame = None) -> pd.DataFrame`
- Return a DataFrame with a single `prediction` column, indexed to the live IDs.
- `live_benchmark_models` is optional; use it only if you ensemble against benchmark predictions.
- Serialize the function with `cloudpickle` (not plain `pickle`), because it captures the
  local/global context and helper functions the function depends on at run time.

Pre-upload validation (assert before serializing):

- `prediction` column is present.
- Output index equals the live-features index (no dropped/extra IDs).
- No NaN values in the output.
- Row count matches the live universe.

Operational best practice:

- Keep preprocessing deterministic.
- Avoid hidden global state that will not serialize.
- Version data assumptions (dataset version, feature set, target) explicitly.

See `05-notebooks/example-model-sunshine.ipynb` for a working end-to-end `predict` + cloudpickle example.

## 9) Retraining and Regime Drift

There is no universal retraining frequency.

Observed community patterns:

- Weekly retraining is common when automation cost is near zero.
- Some long-lived static models remain competitive.

Correct procedure:

- Backtest step-forward freshness windows.
- Decide retraining cadence from measured trade-offs in Sharpe/drawdown/uniqueness.

## 10) Practical Failure Modes

- Random CV leakage produces false confidence.
- Optimizing only mean CORR creates brittle live behavior.
- Overly complex ensembles can add noise instead of diversification.
- Excessive zero-filling or tie-inducing can harm transformed-correlation geometry.

## 11) Minimal Production Checklist

- Era-aware validation with purge/embargo
- Rank-domain ensembling
- Per-era metric suite (CORR, Sharpe, MMC, drawdown)
- Exposure diagnostics and neutralization decision
- Submission format validation before upload
- Reproducible pipeline from train to live predict

This file is strategy-level guidance. Canonical protocol truth remains in 01-canon.
