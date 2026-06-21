This comprehensive guide consolidates the architectural, statistical, and operational knowledge required to compete in the Numerai Data Science Tournament. It is designed to be used as a "system prompt" or "knowledge base" for LLMs and Copilots.

---

# The Numerai Strategy & Implementation Bible (v5.2)

## 1. Core Philosophy & Data Structure

Numerai is a global equity strategy competition using **obfuscated tabular data**. You are predicting relative stock returns without knowing the stock names or feature definitions.

### Key Data Characteristics

* **Eras**: Data is organized into "eras" (weekly intervals). Each row is a stock; each era is a snapshot of the market.


* **IDs**: Unique per stock per era. You cannot track a single stock’s performance over time due to encryption.


* 
**Features**: Quantitative attributes (P/E ratio, RSI, etc.) binned into integers 0-4.


* 
**Targets**: 20-day or 60-day stock-specific returns, neutralized for market, country, and sector beta.


* 
**Obfuscation**: The data is engineered so it has no utility outside the tournament, requiring models to rely on pure statistical signal rather than domain knowledge.



---

## 2. Validation & Training Framework

**Never use random cross-validation.** Because targets are forward-looking (20 or 60 days), overlapping eras contain leaked information.

### The Gold Standard: Era-Based Validation

* 
**Purged/Walk-Forward CV**: Train on a chunk of eras, purge the following 4 eras (for 20-day targets) or 16 eras (for 60-day targets) to prevent leakage, then validate on the next chunk. Always group splits strictly by `era`.


* 
**Embargo**: When combining training and validation data, always embargo the first 4 eras after your last training era.


* **Per-Era Metrics (Sharpe is King)**: Always calculate metrics (CORR, MMC, FNC) on a per-era basis first, then average them. Flattened metrics are extremely misleading in time-series finance. Measure the stability using the **Era-Wise Sharpe Ratio** (`mean(era_scores) / std(era_scores)`). High mean CORR is worthless if the Sharpe is below 1.0.



---

## 3. Scoring Metrics & Optimization

### CORR (Correlation)

The primary measure of "Alpha." It is a specialized Spearman correlation.

* 
**Calculation**: Rank-transform (tie-kept), Gaussianize, and raise to the power of 1.5 to accentuate the importance of the "tails" (extreme predictions).


* 
**Benchmark**: A Mean CORR of 0.01–0.03 is considered good performance.



### MMC (Meta Model Contribution)

Measures how much *unique* signal you provide to the overall ensemble (the Meta Model).

* **Optimization**: High CORR often leads to low MMC if your model is similar to the crowd. To gain MMC, your model must be orthogonal to the Meta Model while still correlating with the target.



### FNC (Feature Neutral Correlation)

The correlation of your model after its predictions have been neutralized to all 1,000+ features. High FNC indicates your model is finding non-linear signals that aren't just "feature exposure".

---

## 4. Advanced Modeling Techniques

### Feature Neutralization

Since features are inconsistent over time, models with high linear exposure to specific features are risky.

* 
**Mechanism**: Use least-squares regression to find the orthogonal component of your predictions with respect to a feature matrix.


* 
**Proportion**: You can apply partial neutralization (e.g., 0.5 or 0.75) to balance risk reduction with signal retention.



### Target Ensembling

Training on auxiliary targets (e.g., `target_nomi_20`, `target_victor_20`) often yields more robust results than training only on the main `target`. Each target represents a slightly different perspective on risk/reward.

* 
**Strategy**: Train independent models on diverse targets and ensemble them using **rank-averaging**.
* 
**Rank vs. Continuous Blending**: Never average the continuous raw outputs of regression models in Numerai. You must convert predictions to percentiles rank (`scipy.stats.rankdata` or `pd.Series.rank(pct=True)`) *before* averaging. The tournament scores on rank-order correlation (Spearman), making raw prediction magnitude irrelevant.
*
**Seed Diversification**: To counter row/column sub-sampling variance, consider training 3–5 different random seeds per target, then rank-averaging those seeds (Level 1) before rank-averaging the targets together (Level 2).



---

## 5. Deployment & Automation

### Model Uploads (The Recommended Path)

Numerai allows you to upload a serialized `.pkl` file. They provide the infrastructure to run your model daily against new live data.

* 
**Requirement**: The pickle must contain a `predict(live_features, live_benchmark_models)` function.


* 
**Library**: Use `cloudpickle` for serialization, as it captures the local context and global variables required for the function to run in the cloud.



---

## 6. Staking & NMR Economics

Staking is the "skin in the game" mechanism that allows Numerai to trust predictions.

* 
**Rewards/Burns**: Positive scores yield NMR payouts; negative scores cause a portion of the stake to be "burned" (destroyed).


* 
**Payout Factor**: Payouts are capped at ±5% per round and scale based on the total NMR staked in the tournament relative to a threshold (e.g., 72,000 NMR for Numerai Classic).


* 
**Withdrawals**: Releasing a stake takes 1 month (the duration of a scoring round). New accounts have a 30-day withdrawal lock unless the amount is >0.1 NMR.



---

## 7. Standard Hyperparameters (LGBM) & Model Selection

* **LightGBM vs XGBoost**: LightGBM is generally preferred due to its raw efficiency over colossal datasets with binned tabular features.
* **Loss Function Paradox**: While `regression` (RMSE) is standard, standard gradient boosting often overfits the "noise" in the middle of the feature distribution. Alpha in Numerai lives in the tails (ranks 0.0-0.1 and 0.9-1.0). Early stopping *must* be done on valid Era-wise Spearman or Era-wise Sharpe, not minimizing RMSE.

For a "Deep & Slow" configuration using v5.2 data:

```python
{
  "n_estimators": 30000,          # Massive capacity to learn weak signals
  "learning_rate": 0.001,         # Micro-stepping prevents overfitting to local era noise
  "max_depth": 6,                 # Keep shallow to prevent complex feature interactions
  "num_leaves": 64,               # Constrained splits
  "colsample_bytree": 0.1,        # CRITICAL: Forces tree to use different features, fighting transient drift
  "subsample": 0.8,
  "min_data_in_leaf": 5000        # Huge minimum samples per leaf for extreme regularization
}
```


## 8. Misc : 

currently, payouts and burns are capped at 5% per round. There are about 260 rounds per year.

The core MMC intuition: MMC rewards predictions that are uniquely correlated with the target beyond what the meta-model already knows. 