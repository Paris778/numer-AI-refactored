---
name: numerai-scientific-modeling
description: "Use when performing NumerAI tournament research, scientific model development, validation, ensembling, neutralization, and productionizing predictions for strongest expected leaderboard and staking performance. Trigger phrases: NumerAI, RAPS, MMC, CORR, FNC, feature exposure, benchmark correlation, neutralization, walk-forward, era CV, model upload, predict pipeline."
---

# NumerAI Scientific Modeling Skill

## Mission

Design, test, and harden models for NumerAI tournament performance with a scientific workflow that prioritizes robust out-of-sample behavior and payout quality, not just raw in-sample fit.

## Operating Principles

1. Optimize for expected tournament utility, not a single metric.
2. Use era-aware validation and purge/embargo to reduce leakage from overlapping targets.
3. Prefer robust, diversified ensembles over brittle single models.
4. Control feature exposure and benchmark crowding to preserve MMC.
5. Every claim must be supported by a reproducible experiment and tracked metrics.

## Repository-Specific Ground Truth

1. Use `utils.metrics.calculate_metrics` as the canonical evaluation API.
2. `calculate_metrics` requires validation data with: `era`, `prediction`, target column, and the full feature list passed in `features`.
3. Use `submissions/model_metrics_history.csv` with `utils.model_benchmark.record_model_metrics(metrics, model_name, notebook_name, path=None, force=False)`.
4. Rank candidate models primarily by `1_RAPS`, then inspect `4_Mean_MMC`, `3_Mean_CORR`, `6_Mean_FNC`, `10_Benchmark_Corr`, and drawdown/tail metrics.
5. Submission predictions must be in `[0, 1]` and era-rank-normalized before final output.

## Scientific Workflow

### Phase 1: Problem Framing

1. Define objective target(s): main target plus auxiliary targets if useful.
2. Define success criteria before modeling:
- Primary: higher `1_RAPS` than baseline.
- Secondary: positive or improving MMC, stable CORR, controlled feature exposure, acceptable drawdown.
3. Define risk constraints:
- Avoid large degradation in `4_Mean_MMC` even when CORR improves.
- Keep crowding (`10_Benchmark_Corr`) and `12_Max_Feature_Exposure` under control.

### Phase 2: Data and Validation Design

1. Use current data version and feature metadata (`features.json`).
2. Build era-based splits with purge/embargo windows for 20D and 60D targets.
3. Avoid random row-level CV.
4. When auxiliary targets contain NaN, drop rows only where required for that target-specific training task.

### Phase 3: Baselines and Diagnostics

1. Build simple, reproducible baselines first (for example Ridge or benchmark blend).
2. Produce per-era diagnostics from `calculate_metrics` output and inspect regime behavior.
3. Track:
- Mean and Sharpe of payout proxy
- Drawdown and tail behavior
- Win rate
- Feature exposure
- Correlation with benchmark predictions

### Phase 4: Candidate Generation

Test diverse families and transformations:

1. Target specialization:
- Train specialists on multiple auxiliary targets.
- Blend or stack specialists for main target.
2. Model diversity:
- Linear (Ridge/ElasticNet)
- Tree-based (LightGBM/XGBoost)
- Different regularization and feature subsets
3. Post-processing:
- Era-wise rank normalization
- Neutralization strength sweeps (for example 0.25, 0.5, 0.75, 1.0)
- Optional benchmark-aware decorrelation

### Phase 5: Selection and Robustness

1. Evaluate all candidates on identical validation framework.
2. Select by primary objective (`1_RAPS`) only after confirming no hidden risk blowups.
3. Require robustness checks:
- No single-era dependence for total performance
- Acceptable behavior in recent eras
- Stable metrics under small hyperparameter perturbations

### Phase 6: Productionization

1. Retrain the chosen pipeline on full allowed historical data (respecting chosen methodology).
2. Provide a production `predict(live_features, live_benchmark_models) -> DataFrame` function.
3. Ensure final output dataframe has one column: `prediction`.
4. Apply final era-wise rank normalization before returning predictions.
5. Export pipeline with `cloudpickle`.

## Required Agent Behaviors

1. Always present an experiment plan before broad hyperparameter sweeps.
2. Keep an experiment log table for each run:
- run_name
- data_version
- target set
- model family
- neutralization strategy
- key hyperparameters
- `1_RAPS`, `4_Mean_MMC`, `3_Mean_CORR`, `6_Mean_FNC`, `10_Benchmark_Corr`, `7_Max_Drawdown_CORR`
- decision (promote/reject)
3. When claiming "better model", include exact delta vs baseline on primary and risk metrics.
4. If compute is limited, favor fewer high-quality experiments over brute-force search.
5. Never skip leakage checks, era-aware splitting, or final rank normalization.

## NumerAI Metric Interpretation Rules

1. `3_Mean_CORR` up with `4_Mean_MMC` down can reduce staking utility.
2. High `10_Benchmark_Corr` often implies crowding and weak uniqueness.
3. Weak `6_Mean_FNC` or extreme feature exposure often indicates brittle signal.
4. Prefer smoother payout profile (Sharpe/volatility/drawdown/tail) over noisy spikes.

## Deliverable Format

When asked to run this skill, return:

1. Research plan and hypotheses.
2. Exact code changes made (or notebook cell patches).
3. Evaluation table sorted by `1_RAPS`.
4. Final model choice with rationale and risk trade-offs.
5. Production export status (`predict` function + pickle path).
6. Next 3 experiments to run if improvement is still needed.

## Fast Checklist

1. Era-aware split with purge/embargo used.
2. Baseline established and logged.
3. Candidate diversity tested.
4. Neutralization sweep tested.
5. Metrics computed through `utils.metrics.calculate_metrics`.
6. Best candidate selected by `1_RAPS` with risk sanity checks.
7. Final `prediction` output rank-normalized per era.
8. Run recorded in model history.
