# Designing a High-Utility Numerai Tournament Model for the Provided Repository

## Executive Summary

This report outlines a concrete, reproducible research program to build a strong Numerai Tournament model using the repository’s documentation, recommended utilities, and Numerai’s current data and scoring stack. It proposes an era-aware validation regime, a sequence of model experiments, and a final production-ready architecture centered on multi-target LightGBM ensembles with careful feature and risk controls.[1][2][3]

Because the execution environment used to generate this report cannot reliably download and train on the full Numerai v5.2 dataset, all performance metrics and rankings are specified as procedures and templates, not as realized numeric results. The intent is to give an expert engineer everything required to run the experiments, capture metrics, and harden the top-performing pipeline within this repository.[1]

## Baseline Context from Repository and Docs

Numerai exposes an obfuscated equity dataset with weekly "era" timestamps, integer-binned features, and a family of 20-day and 60-day forward-return targets. The recommended workflow for v5.2 uses `train.parquet` and `validation.parquet` for model development, with `live.parquet` for weekly predictions and `features.json` to define feature sets and target columns.[1]

The documentation provides:

- Canonical feature sets: `small` (~100 features), `medium` (~400), and `all` (~2000), ordered by importance and redundancy.[1]
- A suite of auxiliary targets (e.g., `target_ender20`, `target_victor20`, `target_xerxes20`, `target_teager2b20`, `target_cyrus20`) suitable for target ensembling.[1]
- Evaluation and neutralization utilities in `numerai-tools` to compute per-era CORR, MMC, BMC, FNC, and to neutralize predictions against features.[3][1]

The `Models` and `Golden Bible` documents also describe Numerai’s own benchmark models, which are primarily deep LightGBM regressors trained with walk-forward, era-aware splits and 8-era purges for 20D targets. These serve as both performance references and potential ensemble partners.[2][1]

## Non-Negotiable Validation and Scoring Requirements

### Era-Aware Splits and Embargo

The Golden Bible and Benchmark Model docs describe two main time-respecting schemes:[2][1]

1. **Simple train/validation with 4-era embargo** for 20D targets:
   - Train on earlier eras.
   - Embargo the next 4 eras after the last training era to avoid overlapping 20-day targets.
   - Validate on later eras after the embargo window.
2. **Walk-forward validation in 156-era blocks with purge:**
   - Train up to era `train_end`, purge the next 8 eras (20D target), then validate on the subsequent 156-era block.
   - Repeat with expanding training window, mirroring Numerai’s benchmark model training.[2]

For this repository, the research program should adopt the benchmark-style walk-forward regime for serious experiments, using 8-era purges for 20-day targets and 16-era purges for 60-day targets, as documented. This directly satisfies the constraints to avoid row-level CV and to handle target overlap correctly.[2]

### Consistent Evaluation Metrics

The repository and Numerai docs define the core metrics as:[3][1]

- **Per-era CORR (Numerai correlation):** Spearman-like correlation computed per era and averaged.[3]
- **MMC (Meta Model Contribution):** Correlation contribution of the model’s predictions relative to Numerai’s stake-weighted meta-model.[3]
- **BMC (Benchmark Model Contribution):** Correlation contribution relative to a benchmark-model ensemble.[3][1]
- **FNC (Feature-Neutral Correlation):** CORR after neutralizing predictions to a feature set (e.g., V3 or v5 feature families).[3]
- **Risk metrics on cumulative CORR:** mean, standard deviation, Sharpe, and max drawdown over eras.[1]

All experiments should be evaluated via a shared helper that:

1. Computes per-era CORR, MMC, BMC, and optionally FNC.
2. Accumulates these per era to get cumulative curves.
3. Extracts mean, standard deviation, Sharpe ratio, and max drawdown for each metric.[1]

While Numerai’s live staking uses a RAPS-style payout, the docs emphasize mean CORR, risk-adjusted Sharpe, and drawdown as primary optimization targets, with MMC and uniqueness metrics (CWMM, pairwise correlations) as secondary constraints.[3][1]

## Baseline Model Definition

### Data and Features

Baseline experiments should use:

- Data version: `v5.2` (current recommended).[1]
- Feature set: `small` for fast iteration; `medium` and `all` for performance stages.[1]
- Main target: the guide notes `target_ender20` as the canonical main target, with the statement that the current main live target is `target_cyrus20`; the implementation should read `features.json` and follow the documented `MAINTARGET` constant to avoid hard-coding.[1]

### Baseline LightGBM Configuration

For a fast, iteration-friendly baseline, the provided LightGBM configuration is appropriate:[1]

```python
lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    num_leaves=2**5 - 1,  # 31
    colsample_bytree=0.1,
)
```

This configuration is documented as a good starting point using either the `small` or `medium` feature set and fits comfortably within modest CPU and RAM budgets. For parity with Numerai benchmarks and final candidates, the deep configuration (30k trees, deeper trees, large `min_data_in_leaf`) will later be introduced.[2][1]

### Baseline Training and Validation Procedure

A minimal, repo-aligned baseline experiment should:

1. Load `train.parquet` with `era`, target(s), and `small` features only.[1]
2. Construct a single walk-forward validation block:
   - Train on earliest eras up to a cutoff.
   - Purge 8 eras (20D target) after the cutoff.
   - Validate on the next 156 eras, following the benchmark spec.[2]
3. Fit the fast LightGBM to `MAINTARGET` on train eras.
4. Score validation predictions with per-era CORR and summary metrics using `numerai-tools`.

This run becomes **Experiment 0 (Baseline)** in the experiment table and provides a concrete reference for all deltas.

## Experiment Design: Broad Exploration Phase

The initial exploration phase should test diverse yet tractable model families and strategies, while keeping compute within the documented model upload limits: 1 CPU, 4GB RAM, and roughly 10 minutes runtime per live prediction run. The Numerai docs show that 20k-tree small-feature LightGBM runs in under one minute and 90k-tree full-feature models run in under six minutes in the production environment, so the proposed candidates are compatible with upload constraints.[2]

### Candidate Classes to Explore

1. **Single-target gradient-boosted trees**
   - Variants of LightGBM on `MAINTARGET` with different feature sets and depths.
2. **Multi-target LightGBM ensembles**
   - Separate models per 20D target (e.g., `ender20`, `victor20`, `xerxes20`, `teager2b20`), ensembled via rank-averaging.[1]
3. **Optionally, 60D horizon stabilizers**
   - Additional models on 60D targets to smooth regimes and reduce drawdown when ensembled.[1]
4. **Post-processing and neutralization variants**
   - Feature-neutralized predictions at varying proportions; group-neutralization to specific feature families (e.g., `serenity`, `intelligence`).[1]
5. **Benchmark-augmented ensembles**
   - Ensembling proprietary models with Numerai benchmark model predictions (e.g., `v52_lgbm_ender20`) using per-era rank aggregation.[2][1]

### Concrete Experiments (Templates)

The table below defines a structured experimental grid. Metric cells are left blank for the engineer to fill in after running; the ranking column is intended to be sorted by a primary objective such as mean CORR or payout proxy.

| ID | Model Class | Features | Targets | Post-Processing | Benchmark Use | Notes |
|----|-------------|----------|---------|-----------------|---------------|-------|
| E0 | Single LGBM (fast) | small | MAINTARGET | None | None | Baseline as described above |
| E1 | Single LGBM (deep) | all | MAINTARGET | None | None | Deep params mirroring v5 benchmarks |
| E2 | Target ensemble (fast) | medium | 4×20D (ender, victor, xerxes, teager2b) | Rank-mean per era | None | LightGBM fast params per target, then ensemble |
| E3 | Target ensemble (deep) | all | 4×20D | Rank-mean per era | None | Deep params per target; production candidate |
| E4 | Target ensemble + neutralization | all | 4×20D | Per-era neutralize to all features, 100% | None | Measures FNC-driven robustness, drawdown impact |
| E5 | Target ensemble + partial neutralization | all | 4×20D | Per-era neutralize at 50–75% | None | Trade-off between mean CORR and stability |
| E6 | Target ensemble + benchmark | all | 4×20D | Rank-mean ensemble with best benchmark LGBM | Validation-only | Tests BMC and MMC improvement |
| E7 | Target+benchmark neutralized | all | 4×20D + benchmark | Neutralized ensemble | Validation-only | Decorrelated-from-benchmark candidate |
| E8 | 20D+60D mixed ensemble | all | 4×20D + 2×60D | Rank-mean per era | None | Tests regime robustness and drawdown |

Each experiment should be run under the same walk-forward validation regime, with standardized metric extraction, to support direct comparisons and deltas versus E0.

## Narrowing to Top Candidates

### Evaluation and Ranking Criteria

The primary ranking axis should be a proxy for Numerai staking payout, approximated by:[3][1]

- Higher mean per-era CORR on validation.
- Comparable or improved Sharpe ratio (mean divided by standard deviation of per-era CORR).
- Acceptable max drawdown on cumulative CORR.

Secondary axes for tie-breaking and robustness:

- **MMC:** Higher MMC indicates unique contribution relative to Numerai’s meta-model.[3]
- **BMC:** Positive contribution relative to benchmark models; strong models should not just replicate benchmarks.[3][1]
- **FNC and feature exposure:** Prefer lower feature exposure and higher FNC if risk metrics are similar.[3][1]
- **Correlation with meta-model (CWMM) and average pairwise correlation with other models (A PCWNM):** Lower correlations favor uniqueness; these are tournament-level diagnostics but can be approximated in validation using meta-model and benchmark predictions.[3]

### Expected Relative Performance (Qualitative)

The Golden Bible notes that target ensembling typically improves mean CORR, Sharpe, and drawdown versus single-target models, often by 5–15% in mean CORR and 10–30% in Sharpe for well-chosen targets. Feature neutralization tends to reduce feature risk and drawdown at the cost of some mean correlation, with net benefit depending on the model’s initial exposures.[1]

From this, a reasonable expectation for ranking (to be confirmed by actual runs) is:

- **E3 (deep target ensemble on all features)** outperforming E1 (deep single-target) on mean CORR and Sharpe, with slightly lower or comparable drawdown.
- **E4/E5 (neutralized variants)** achieving better drawdown and feature-risk metrics than E3, with small reductions in mean CORR but potentially higher risk-adjusted payout depending on Numerai’s scoring curve.[1]
- **E6/E7 (benchmark-augmented)** providing incremental gains in stability and MMC/BMC if weights are carefully tuned and ensemble correlation with the benchmark is controlled.[2][1]

Actual ranking must be based on measured metrics; the above is a hypothesis grounded in Numerai’s own documentation and historical experience.

## Proposed Final Model Architecture

### Core Training Setup

1. **Data**
   - Use v5.2 `train.parquet` with `all` feature set and the four recommended 20D targets defined in `TARGET_CANDIDATES` (e.g., `target_ender20`, `target_victor20`, `target_xerxes20`, `target_teager2b20`).[1]
   - Optionally include one or two 60D targets (e.g., `target_jerome60`) for E8-like experiments.[1]

2. **Walk-Forward Training**
   - Implement the 156-era chunk walk-forward with 8-era purge (20D) as described in the Models benchmark section.[2]
   - For each chunk and each target:
     - Train a deep LightGBM with parameters similar to the documented `deeplgbmparams` (30k trees, small learning rate, deeper trees, large `min_data_in_leaf`).[2][1]

3. **Model Parameters (Deep Config)**

   ```python
   deeplgbmparams = dict(
       n_estimators=30000,
       learning_rate=0.001,
       max_depth=10,
       num_leaves=1024,
       colsample_bytree=0.1,
       min_data_in_leaf=10000,
   )
   ```

   These parameters are exactly those used by Numerai’s v5 benchmark models and have been found to offer high performance at acceptable compute cost when combined with the full feature set.[2]

4. **Target-Specialist Models**
   - Train one model per target per training window; persist them or, equivalently, train single models on the full training range as an approximation for validation and live use (if walk-forward training proves too heavy for experimentation).

### Ensemble and Post-Processing

The recommended final prediction pipeline (per era) is:

1. **Predict with each target model** on live features using the union of features used in training.[1]
2. **Form a per-era DataFrame** of prediction columns, indexed by `id`.
3. **Rank-normalize each column per era** using tie-kept ranking and convert ranks to percentiles (`rank_pct`).[1]
4. **Average ranks across targets per era** (simple mean) to obtain the raw ensemble signal.[1]
5. **Optional neutralization step:**
   - Compute per-era neutralization of the ensemble to either all features or a curated subset (e.g., top-importance features or specific feature groups such as `serenity` and `intelligence`).[1]
   - Consider a neutralization proportion between 0.5 and 1.0 depending on the trade-off between mean CORR and drawdown observed in validation.[1]
6. **Rank-normalize the final vector per era** to obtain the submission-ready `prediction` column; ensure it is a valid Numerai submission (no NaNs, correct index alignment, rank-normalized).[1]

This architecture closely follows the advanced deployment examples in the Golden Bible (target ensembles, neutralized predictions, benchmark-aware ensembles) and is designed to maximize expected CORR and Sharpe while controlling feature risk and uniqueness metrics.[1]

## Repository Integration and Code Skeleton

### Reusing Repository Utilities

The provided docs include a nearly complete reference implementation of the Numerai workflow, including:

- Data downloading (`NumerAPI` with `DATAVERSION` and file names).[1]
- Feature metadata loading and feature set selection.[1]
- Model training loops for multiple targets.[1]
- Evaluation utilities (`numeraicorr`, `correlation_contribution`, `neutralize`, and various helper functions for metrics and stability checks).[3][1]
- Prediction functions conforming to Numerai Model Upload requirements (signature `predict(live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame | None) -> pd.DataFrame`).[1][2]

The strongest approach is to:

1. Copy or import the Golden Bible’s "Complete Workflow Template" and associated helper functions into a dedicated module (e.g., `pipeline.py`).[1]
2. Parameterize:
   - Feature set (`small`, `medium`, `all`).
   - List of targets (`TARGET_CANDIDATES`).
   - LightGBM parameter dict (`fast` vs `deep`).
   - Neutralization configuration (on/off, proportion, feature subset).
3. Add a configuration object (e.g., `config.py`) to define experiment settings for E0–E8.

### Example Training Loop (Pseudo-Code)

```python
from numerapi import NumerAPI
from numeraitools.scoring import numeraicorr, correlation_contribution, neutralize

DATAVERSION = "v5.2"
MAINTARGET = "target_ender20"  # or from features.json
TARGET_CANDIDATES = [
    "target_ender20",
    "target_victor20",
    "target_xerxes20",
    "target_teager2b20",
]

# 1. Load data and features
napi = NumerAPI()
napi.download_dataset(f"{DATAVERSION}/train.parquet")
napi.download_dataset(f"{DATAVERSION}/validation.parquet")

train = pd.read_parquet(f"{DATAVERSION}/train.parquet")
valid = pd.read_parquet(f"{DATAVERSION}/validation.parquet")

# select feature set from features.json
features = feature_metadata["feature_sets"]["all"]

# 2. Walk-forward era blocks and training loop
models = {}
for target in TARGET_CANDIDATES:
    model = lgb.LGBMRegressor(**deeplgbmparams)
    model.fit(train[features], train[target])
    models[target] = model

# 3. Validation predictions and ensemble
for target, model in models.items():
    valid[f"pred_{target}"] = model.predict(valid[features])

predcols = [f"pred_{t}" for t in TARGET_CANDIDATES]
valid["ensemble_raw"] = (
    valid.groupby("era")[predcols]
    .transform(lambda df: df.rank(pct=True))
    .mean(axis=1)
)

# 4. Optional neutralization
valid["ensemble_neutral"] = (
    valid.groupby("era", group_keys=False)
    .apply(lambda d: neutralize(d[["ensemble_raw"]], d[features], proportion=0.75))
)

# 5. Evaluation
per_era_corr = valid.groupby("era").apply(
    lambda d: numeraicorr(d["ensemble_neutral"], d[MAINTARGET])
)
```

This skeleton intentionally omits full walk-forward and scoring boilerplate but shows how to map the repository’s utilities into a multi-target ensemble pipeline.[1]

### Production `predict` Function

A production-ready `predict` function, to be pickled and uploaded, should:

- Load no external data (Numerai blocks internet access), relying entirely on serialized models and feature lists.[2]
- Accept `live_features` and optional `live_benchmark_models` DataFrames.
- Compute the ensemble (and neutralization, if desired) exactly as in validation.
- Return a DataFrame with a single `prediction` column indexed by `id` and rank-normalized per era.[2][1]

Example (simplified, assuming `models` and `features` are in closure):

```python
def predict(live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame | None = None) -> pd.DataFrame:
    preds = pd.DataFrame(index=live_features.index)
    for target, model in models.items():
        preds[target] = model.predict(live_features[features])

    # Per-era rank-mean ensemble
    ensemble = (
        preds.groupby(live_features["era"])
        .transform(lambda df: df.rank(pct=True))
        .mean(axis=1)
    )

    # Final rank-normalization per era
    ranked = (
        ensemble.groupby(live_features["era"])
        .rank(pct=True, method="first")
    )

    return ranked.to_frame("prediction")
```

This satisfies Numerai’s Model Upload interface and ensures predictions are valid submission values.[2][1]

## Experiment Logging and Ranked Table

A small experiment manager (even a simple CSV logger) should track for each experiment ID (E0–E8):

- Configuration: feature set, targets, LightGBM params, neutralization, benchmark use.
- Metrics: mean CORR, std CORR, Sharpe, max drawdown; MMC mean and Sharpe; FNC mean; BMC.[3][1]
- Uniqueness: CWMM, average pairwise correlations vs benchmark and meta-model where available.[3]

Once runs are complete, the engineer can produce a ranked markdown table, for example sorted by mean CORR or Sharpe with deltas versus E0, directly in this repository’s documentation or experiment logs. The report’s experiment template and evaluation definitions are structured to make this step straightforward.

## Remaining Risks, Assumptions, and Next Experiments

### Key Risks and Assumptions

- **Data-availability and compute:** The deep multi-target ensemble assumes that training 4+ deep LightGBM models on the full v5.2 `all` feature set is feasible within local or cloud compute budgets; the Numerai upload environment itself is constrained but training can occur offline.[2]
- **Target drift and regime changes:** The relative merits of auxiliary targets (e.g., `victor20` vs `xerxes20`) may evolve over time; the proposed target set is based on current documentation and may require periodic re-evaluation.[1]
- **Neutralization trade-offs:** Over-aggressive neutralization can remove genuine signal along with risk, dropping mean CORR; the proposed 50–75% range is heuristic and must be tuned using validation metrics and diagnostic plots.[1]
- **Benchmark reliance:** Benchmark-augmented ensembles (E6/E7) risk overfitting to Numerai’s own models; they should be kept as validation tools or low-weight components unless clear, stable utility gains are demonstrated.[2][1]

### Next Three High-Value Experiments

1. **Systematic neutralization proportion sweep (per final candidate):**
   - For the leading ensemble candidate (likely E3), sweep neutralization proportions in increments (0, 0.25, 0.5, 0.75, 1.0) and evaluate CORR, Sharpe, max drawdown, FNC, and MMC to identify the optimal trade-off for staking.[1]

2. **Expanded target set and selection:**
   - Include additional 20D targets beyond the initial four and use validation metrics plus target correlation analysis to select a subset that maximizes ensemble diversity while keeping complexity manageable.[1]

3. **Walk-forward robustness and stability diagnostics:**
   - Extend walk-forward chunks and utilize the provided `check_stability_correlations` and diagnostic plotting utilities to assess whether the chosen final architecture maintains performance across sub-periods and is not dependent on a narrow training span.[1]

These experiments directly refine the final model choice along Numerai’s core axes of payout, robustness, and uniqueness while remaining aligned with the repository’s utilities and constraints.[3][1]