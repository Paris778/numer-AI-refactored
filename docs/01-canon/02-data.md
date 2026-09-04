# Numerai Dataset

> **Canonical scope:** Dataset identity, row structure, eras, features, targets, versions, files, and download patterns.

## Dataset model

The Numerai dataset is a tabular description of the global stock market over time.

- Each row represents a stock at a specific point in time.
- `id` identifies that stock observation.
- `era` identifies the time bucket.
- Features describe attributes known at that time.
- Targets measure future returns relative to that time.

## IDs

The `id` is unique per stock per era. It cannot be used to track the same stock across eras. Treat it as a unique identifier for one stock observation in one era.

## Eras

An era represents a point in time. Historical train and validation eras are Fridays, because Friday is the latest market close available for generating predictions over the weekend. Historical eras are one week apart.

For modeling and evaluation, treat an era as a statistical unit rather than treating every row as independent. Numerai metrics are commonly calculated per era and then aggregated.

Targets can look forward 20 or 60 market days while historical eras arrive weekly. The resulting target windows overlap, so random row-level cross-validation is unsafe. Use era-grouped validation with the required purge buffer; see [Models](03-models.md).

In the live tournament, a new live era is released each day. Live eras are one day apart.

### Comparable validation overlap

For comparable model payout and MMC reporting, use only the eligible intersection of `validation.parquet` and `meta_model.parquet`. The overlap is derived from the current data snapshot, not hard-coded: rows with an unavailable selected target are excluded, and the resulting `n_eras` is part of the scorecard/dashboard evidence. In v5.3 the current Meta Model coverage begins at era 1133, so the present overlap is 86 eras; a data refresh can move or expand that window.

Models are fit on `train.parquet` and scored on the held-out validation/meta overlap. Do not extend an Atomic payout proxy to the full validation period when Meta Model coverage is absent, and do not annualize the weekly-era mean payout. The payout policy and the distinction between this per-era proxy and account-level Atomic settlement returns are defined in the evaluation specification ([`evaluation-suite-bible.md`](../06-evaluation/evaluation-suite-bible.md) §5.0 and §16).

## Features

Features include fundamentals, technical signals, market data, analyst information, and other engineered signals. Feature values for an era represent attributes as of that era. Numerai designs features to be point-in-time to avoid leakage.

Individual feature power can be inconsistent over time. Avoid relying too heavily on a small number of features or on features with high exposure unless out-of-sample evidence supports that choice.

Some feature values are `NaN` because the source value was unavailable. Do not treat a missing value as a measured value without an explicit missing-data policy.

## Targets

Targets represent future stock-specific performance relative to the era. Numerai neutralizes targets against selected forms of beta, such as markets, countries, sectors, or common factors, to focus on alpha-like returns.

Targets differ by:

- the factors or features that are residualized;
- the return horizon, such as 20 or 60 market days; and
- the target's name and version-specific definition.

### v5.3 target identity

In v5.3:

- `target_ender_20` is the explicit Ender-20 target.
- `target_ender_60` is the explicit Ender-60 target.
- `target` is an alias for `target_ender_60`.

Code whose horizon matters must select an explicit target column. The generic `target` alias means the default target for the current dataset version; it is not a stable target identity.

Models trained on other targets can outperform or ensemble well with a model trained on the default target. Target selection is separate from live scoring: changing a dataset alias does not rewrite historical scores or select a tournament round's target. The round's score configuration controls that round.

Recent validation eras can be published before their 60-day returns mature. For those eras, `target_ender_60` and its alias are null. Filter the exact target selected:

```python
TARGET_COL = "target_ender_60"
validation = validation.dropna(subset=[TARGET_COL])
```

## Downloading data

The data API is the preferred access path. List available datasets before downloading:

```python
from numerapi import NumerAPI

napi = NumerAPI()
datasets = napi.list_datasets()
```

Use the latest compatible version for new work. Minor versions usually preserve file compatibility. Major versions can change dataset structure or contents and generally require retraining.

## v5.3 (Quantum)

v5.3, released in July 2026, introduced 807 new features. Its default target is Ender-60. The dataset files use `int8` feature formats.

Download the core v5.3 files:

```python
from numerapi import NumerAPI

api = NumerAPI()
for filename in (
    "train.parquet",
    "validation.parquet",
    "validation_example_preds.parquet",
    "live.parquet",
    "live_example_preds.parquet",
    "features.json",
    "train_benchmark_models.parquet",
    "validation_benchmark_models.parquet",
    "live_benchmark_models.parquet",
    "meta_model.parquet",
):
    api.download_dataset(f"v5.3/{filename}", filename)
```

## Dataset files

| File | Purpose |
| --- | --- |
| `train.parquet` | Historical data used to train models. |
| `validation.parquet` | Data used to validate or train models; recent 60-day targets can be null. |
| `validation_example_preds.parquet` | Ender-60 example predictions from `v53_lgbm_ender60` on eligible validation eras. |
| `live.parquet` | Current live features used to generate submissions; changes daily. |
| `live_example_preds.parquet` | Ender-60 example predictions on the live data. |
| `features.json` | Feature statistics and predefined feature sets. |
| `train_benchmark_models.parquet` | Benchmark predictions for some training data. |
| `validation_benchmark_models.parquet` | Benchmark predictions for validation data. |
| `live_benchmark_models.parquet` | Benchmark predictions for live data. |
| `meta_model.parquet` | Meta Model information; v5.3 data is available from era 1133 onward. |

## Feature sets and column selection

Parquet is the primary data format and is suited to large columnar datasets. Select only the columns needed by the model:

```python
import json
import pandas as pd
from numerapi import NumerAPI

VERSION = "v5.3"
napi = NumerAPI()
napi.download_dataset(f"{VERSION}/features.json")
with open(f"{VERSION}/features.json", encoding="utf-8") as handle:
    feature_metadata = json.load(handle)

small_features = feature_metadata["feature_sets"]["small"]
TARGET_COL = "target_ender_60"
columns = ["era", TARGET_COL, *small_features]

napi.download_dataset(f"{VERSION}/train.parquet")
training_data = pd.read_parquet(f"{VERSION}/train.parquet", columns=columns)
```

Common feature-set names include `small`, `medium`, and `all`, plus version-specific or obfuscated family sets. Always resolve feature sets from the matching `features.json` rather than assuming the contents are unchanged across versions.
