# Models and Benchmark Models

> **Canonical scope:** Model slots, benchmark models, walk-forward validation, LightGBM presets, ensembles, and neutralization.

## Model slots

A model is a slot within a tournament. A model has submissions, scores, stake, payouts, and a model-level rank. An account can contain multiple models and has an account-level rank aggregated across its models.

Slots are counted independently for Numerai, Signals, and Crypto. The number of available slots depends on the account's Grandmasters tier for that tournament. The current limit is shown on the models page. Contact support if more slots are needed.

Running genuinely different ideas in separate slots lets participants compare live performance rather than relying only on in-sample research.

## Benchmark models

Numerai Benchmark Models are standard models built by the Numerai team. Their predictions are published each round so participants can submit and stake on them if desired. They provide a reference point for model performance.

The benchmark model list and recent performance are available at [numer.ai/~benchmark_models](https://numer.ai/~benchmark_models).

Download benchmark predictions with NumerAPI:

```python
from numerapi import NumerAPI

napi = NumerAPI()
VERSION = "v5.3"
napi.download_dataset(
    f"{VERSION}/train_benchmark_models.parquet",
    "train_benchmark_models.parquet",
)
napi.download_dataset(
    f"{VERSION}/validation_benchmark_models.parquet",
    "validation_benchmark_models.parquet",
)
napi.download_dataset(
    f"{VERSION}/live_benchmark_models.parquet",
    "live_benchmark_models.parquet",
)
```

## Walk-forward validation

Benchmark predictions use walk-forward cross-validation. A prediction for a validation era is generated only by a model trained on data available before that prediction date.

The benchmark data is divided into chunks of 156 eras. For each chunk, the model is trained through the era immediately before the purge buffer:

| Window | Train start | Train end | Validation start | Validation end |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 148 | 157 | 312 |
| 2 | 1 | 304 | 313 | 468 |
| 3 | 1 | 460 | 469 | 624 |
| 4 | 1 | 616 | 625 | 780 |
| ... | ... | ... | ... | ... |

The purge is always:

- `8` eras for 20-day targets;
- `16` eras for 60-day targets.

This buffer is required because forward-looking target windows overlap. Random row-level cross-validation is not a valid substitute.

## Standard Large LightGBM

Most benchmark models use parameters equivalent to:

```python
standard_large_lgbm_params = {
    "n_estimators": 20000,
    "learning_rate": 0.001,
    "max_depth": 6,
    "num_leaves": 2**6,
    "colsample_bytree": 0.1,
}
```

## Deep LightGBM

The deep benchmark preset uses more trees and deeper trees:

```python
deep_lgbm_params = {
    "n_estimators": 30000,
    "learning_rate": 0.001,
    "max_depth": 10,
    "num_leaves": 1024,
    "colsample_bytree": 0.1,
    "min_data_in_leaf": 10000,
}
```

## Ensembles

Numerai ensembles operate in rank-Gaussianized space. The standard sequence is:

1. Gaussianize each prediction on a per-era basis.
2. Standardize each prediction to standard deviation 1.
3. Take the dot product with a weight vector.
4. Gaussianize the resulting vector.
5. Neutralize the result when the ensemble definition requires it.

A simplified single-era implementation is:

```python
import pandas as pd
from scipy import stats


def rank_gauss_pow1(series: pd.Series) -> pd.Series:
    ranked = (series.rank(method="average") - 0.5) / series.count()
    gaussianized = pd.Series(stats.norm.ppf(ranked), index=series.index)
    return gaussianized / gaussianized.std()


ensemble_cols = ["V4_LGBM_NOMI20", "V42_RAIN_ENSEMBLE"]
weight_vector = [0.1, 0.9]
for column in ensemble_cols:
    predictions[column] = predictions.groupby(
        "era", group_keys=False
    )[column].transform(rank_gauss_pow1)
blended = predictions[ensemble_cols].dot(weight_vector)
```

The prediction distribution does not carry the same meaning as its rank order. Preserve ties according to the score or ensemble definition being implemented.

## Neutralization

Neutralization regresses predictions against a set of neutralizer columns and subtracts the fitted linear exposure. It is normally applied per era. The result is orthogonal to the neutralizers under the linear model used for the projection.

A reference implementation is:

```python
import numpy as np
import pandas as pd
from scipy import stats


def neutralize(df, columns, neutralizers=None, proportion=1.0, era_col="era"):
    neutralizers = [] if neutralizers is None else neutralizers
    computed = []
    for era in df[era_col].unique():
        era_data = df[df[era_col] == era]
        scores = era_data[columns].to_numpy()
        ranked_scores = []
        for score in scores.T:
            rank = (pd.Series(score).rank(method="first") - 0.5) / len(score)
            ranked_scores.append(stats.norm.ppf(rank))
        scores = np.array(ranked_scores).T
        exposures = (
            era_data[neutralizers]
            .fillna(era_data[neutralizers].median())
            .fillna(0.5)
            .to_numpy()
        )
        projection = exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32), rcond=1e-6).dot(
                scores.astype(np.float32)
            )
        )
        scores -= proportion * projection
        scores /= pd.DataFrame(scores).std(ddof=0, axis=0).values
        computed.append(scores)
    return pd.DataFrame(
        np.concatenate(computed), columns=columns, index=df.index
    )
```

Neutralization does not require the training target when applied to live predictions. It can improve a simple model and degrade a complex model, so measure its effect out of sample. A purely linear model using the same features as neutralizers can be reduced to zero by full neutralization.

## Community models

The Numerai community operates NumerBay, an independent marketplace where participants may buy and sell predictions. Participants submit and stake any purchased predictions under their own model. Numerai does not endorse, review, verify, or guarantee models listed on NumerBay or their reported performance.
