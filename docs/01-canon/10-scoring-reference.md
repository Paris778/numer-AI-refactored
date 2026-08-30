# Scoring Reference

> **Canonical scope:** Definitions, formulas, transformations, metric names, and scoring terminology.
>
> **Use with:** [Live scoring](09-scoring-live.md) for timelines and [API and MCP](11-api-and-mcp.md) for round-specific score configuration. The current round configuration overrides generic examples in this reference.

The definitions below describe the principal functions and statistical tools used to publish scores. The open-source implementation is in `numerai-tools/scoring`. Install it with:

```bash
python -m pip install numerai-tools
```

## Statistical operations

### Tie-broken rank

A percentile rank of a series that breaks ties using the row ID or index.

### Tie-kept rank

A percentile rank that assigns every member of a tie the average of the tie-broken ranks for that tie group.

### Correlation

A correlation coefficient between two series.

- **Pearson correlation:** Correlation of the input values without ranking.
- **Spearman correlation:** Rank-order correlation; ties receive mean rank.
- **Tie-broken-rank correlation:** Pearson correlation between the target and tie-broken ranked predictions, with a sorted index and no missing values. It cannot reach 1.0 when the target contains ties because predictions do not.

### Variance normalization

Normalize a vector so its standard deviation is 1.

### Power transform

Raise each value to a power while preserving sign:

```text
power(x, p) = sign(x) * abs(x) ** p
```

### Gaussianization

Transform ranked values with the standard normal inverse CDF. The scoring pipeline uses this to standardize prediction distributions before correlation or ensemble operations.

### Neutralization

Given a vector `s` and neutralizer matrix `N`, subtract the component of `s` explained by `N`:

```text
neutralized(s, N) = s - N @ pinv(N) @ s
```

### Orthogonalization

For centered vectors `u` and `v`, remove the component of `v` in the direction of `u`:

```text
orthogonalize(v, u) = v - u * ((v.T @ u) / (u.T @ u))
```

Orthogonalization is the two-vector form of neutralization.

## Core metrics

### Numerai Corr (CORR)

Given predictions `s` and target `t`:

1. Apply tie-kept rank to `s`.
2. Gaussianize the ranked predictions.
3. Center `t` around zero.
4. Apply the signed power-1.5 transform to both vectors.
5. Calculate Pearson correlation.

```python
import numpy as np
from scipy import stats


def numerai_corr(preds, target):
    ranked_preds = (preds.rank(method="average") - 0.5) / preds.count()
    gaussianized_preds = stats.norm.ppf(ranked_preds)
    centered_target = target - target.mean()

    preds_p15 = np.sign(gaussianized_preds) * np.abs(gaussianized_preds) ** 1.5
    target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5
    return np.corrcoef(preds_p15, target_p15)[0, 1]
```

Only prediction rank affects CORR; absolute prediction scale does not. The power transform gives greater influence to the tails because the hedge fund tends to trade stocks with the highest or lowest predicted returns.

Website score names can identify different horizons and targets:

- `CORR20V2`: Numerai Corr against a 20-day target, including historical Ender-20 rounds.
- `CORR60`: Numerai Corr against a 60-day target, including v5.3 Ender-60 diagnostics and rounds configured for Ender-60 scoring.
- `CORJ60`: Correlation against the 60-day auxiliary target named Jerome.
- `CORT20`: Correlation against the 20-day auxiliary target named Teager.

The score name and the round's score configuration identify the scoring horizon. The generic target alias in a downloadable dataset does not.

### Meta Model Contribution (MMC)

MMC is the covariance of a prediction with the target after the prediction is neutralized to the Meta Model. BMC applies the same idea against Benchmark Models.

For prediction vector `s`, Meta Model vector `m`, and target `t`:

1. Tie-kept rank and Gaussianize `s` and `m`.
2. Orthogonalize `s` with respect to `m`.
3. Center `t` around zero.
4. Take the covariance of the orthogonalized prediction and centered target.

```python
import pandas as pd


def contribution(predictions, meta_model, live_targets):
    prediction_values = gaussian(tie_kept_rank(predictions)).values
    meta_values = gaussian(tie_kept_rank(meta_model.to_frame()))[
        meta_model.name
    ].values
    neutral_predictions = orthogonalize(prediction_values, meta_values)
    centered_targets = live_targets - live_targets.mean()
    values = (centered_targets @ neutral_predictions) / len(live_targets)
    return pd.Series(values, index=predictions.columns)
```

MMC rewards target-aligned signal that is distinct from the consensus signal.

### Benchmark Model Contribution (BMC)

BMC uses the MMC calculation against Benchmark Models rather than the Stake-Weighted Meta Model. Live leaderboard BMC uses the stake-weighted Benchmark Models available at the time. Diagnostics use a single benchmark model trained on the payout target, because current validation modeling should be compared with a current benchmark rather than early example predictions.

Live BMC uses the 60-day Ender target and resolves on the 60D2L timeline. Benchmark joins require sufficient overlapping eras; early `train_benchmark_models.parquet` data can be absent.

### Feature Neutral Correlation (FNC)

FNC is correlation with the target after removing linear exposure to Numerai features:

1. Tie-rank and normalize the submission.
2. Neutralize it against the feature matrix.
3. Variance-normalize the residual.
4. Calculate correlation with the target.

```python
import numpy as np
import pandas as pd


def calculate_fnc(sub, targets, features):
    normalized = (sub.rank(method="first") - 0.5) / len(sub)
    feature_values = features.values
    residual = normalized - feature_values.dot(
        np.linalg.pinv(feature_values).dot(normalized)
    )
    residual = residual / residual.std()
    return np.corrcoef(
        pd.Series(residual).rank(pct=True, method="first"),
        targets,
    )[0, 1]
```

The website's current FNC version is FNCv3, neutralized against the `medium` subset of v3 features.

### Correlation With Meta Model (CWMM)

CWMM is informational. It ranks and Gaussianizes a submission, applies the power-1.5 transform, and calculates Pearson correlation with the Stake-Weighted Meta Model. It has no oracle counterpart in the core payout metrics.

### Other informational scores

- **SEASON:** A round's payout-designated CORR and MMC scores multiplied by their default payout multipliers, with missing component scores treated as zero. The exact definitions and multipliers are round-specific.
- **MCWNM:** Maximum Pearson correlation of a submission with another Tournament submission from the same round.
- **APCWNM:** Average Pearson correlation of a submission with each other Tournament submission from the same round.

## Factors, features, and targets

### Factors

Factors are unencrypted data from Numerai's providers, possibly cleaned or formatted. They include signals well known in finance. Numerai neutralizes targets, portfolios, and the Meta Model to these factors.

### Features

Features are encrypted stock-market signals provided for machine-learning use. A dataset contains several variations of a smaller feature set. Numerai usually penalizes exposure to features but does not always remove all exposure.

### Target naming

- `target_[name]_20`: a 20D2L target with five bins, 10%/40%/50% uniformity, and target-specific factor or feature neutralizers.
- `target_[name]_60`: a 60D2L target with five bins, 10%/40%/50% uniformity, and target-specific factor or feature neutralizers.

The target name identifies the residualization and horizon. Use explicit target columns when horizon matters.

### Timeline notation

- `20D` means 20 weekdays of returns.
- `60D` means 60 weekdays of returns.
- `2L` means two weekdays skipped before return calculation.
- `XDYL` means `X` weekdays of returns with `Y` days of returns lag.
- Data lag is the time required for vendors to process return data; scores generally begin after returns lag plus data lag.

## Meta Models

- **Stake-Weighted Meta Model (SWMM):** Stake-weighted average of Numerai submissions. The Numerai Hedge Fund uses it for trading.
- **Benchmark Meta Model (BMM):** Stake-weighted average of Benchmark Model predictions.

## Score configuration terms

A round score configuration can include:

- score name and version;
- display name;
- number of score days;
- returns lag and data delay;
- scoring start and end;
- universe;
- whether it is a Canon Score;
- whether it is selected for payout; and
- its payout multipliers and thresholds.

The selected configurations for a round must be preserved by identity. Similar multipliers do not make different score definitions interchangeable. See [API and MCP](11-api-and-mcp.md#round-score-configuration).

## Neutralization facts

- `neutralize()` neutralizes the columns passed in `columns`.
- To neutralize predictions, pass the prediction column in `columns` and the feature columns in `neutralizers`.
- Neutralization output is not automatically scaled to `[0, 1]`; rank or scale it afterward when the submission contract requires it.
- Neutralization fits a linear model using the neutralizers and subtracts the fitted values.
- Full neutralization removes linear relationships between the prediction and neutralizers.
- Neutralization is applied to predictions, not the training target, and does not require the target for live prediction neutralization.
- A purely linear model using the same features as neutralizers can be reduced to zero by full neutralization.
- Neutralization can improve simple models and degrade complex models. Test it out of sample.
