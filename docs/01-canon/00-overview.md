# Numerai Tournament Overview

> **Canonical scope:** What Numerai is, how the tournament works, and where to find the detailed rules.

## What Numerai is

Numerai is a data science tournament. Numerai provides a free, obfuscated, hedge-fund-grade dataset. Participants build machine learning models, submit predictions, and receive scores over time. Participants may optionally stake NMR on their predictions and earn or burn NMR according to the configured score.

Numerai also operates a quant global equity market-neutral hedge fund. Staked predictions are combined into the Stake-Weighted Meta Model (SWMM), which the hedge fund uses to make trades. Numerai makes money from these predictions when they are correct, so staking aligns participant incentives with the quality of the signal.

## Tournament lifecycle

1. Numerai publishes historical training and validation data, plus live features as they become available.
2. A participant trains a model using the features and a selected target.
3. The participant generates one prediction for every live `id`.
4. The participant submits the prediction file during a round's submission window.
5. Numerai scores the submission against the round's configured target and score definitions.
6. Staked submissions contribute to the Stake-Weighted Meta Model and receive a payout or burn after settlement.

A **round** is the unit that connects a live submission, its scoring schedule, and its staking outcome. A round's API score configuration is authoritative when it differs from a generic dataset or documentation example.

## Data

Each row describes a stock at a point in time. The point in time is represented by an **era**. Historical eras are weekly; live eras are daily. The `id` is unique within an era, so it cannot be used to track the same stock across eras.

Features are obfuscated quantitative attributes known at the era. Targets measure future stock-specific performance relative to that era. See [Data](02-data.md) for IDs, eras, features, targets, versions, and files.

Example download and read flow:

```python
from numerapi import NumerAPI
import pandas as pd

VERSION = "v5.3"

napi = NumerAPI()
napi.download_dataset(f"{VERSION}/train.parquet")
training_data = pd.read_parquet(f"{VERSION}/train.parquet")
```

## Modeling

The modeling objective is to predict a selected target from the features. Use era-aware validation because target windows are forward-looking and overlap across historical eras. See [Models](03-models.md).

## Submissions

A submission is a vector of floating-point predictions with one value per live `id`:

- `0` means the lowest predicted return.
- `0.5` means the average predicted return.
- `1` means the highest predicted return.

Submit through NumerAPI, Numerai CLI, a local server, or Model Uploads. See [Submissions](04-submissions.md) and [Model Uploads](05-model-uploads.md).

## Scoring and staking

The primary payout scores are CORR and MMC. Informational diagnostics include FNC, CWMM, and BMC. Score names, horizons, targets, multipliers, and scoring windows are configured per round. See [Live scoring](09-scoring-live.md) and [Scoring reference](10-scoring-reference.md).

Staking is optional. Unstaked submissions are scored, but only staked submissions carry weight in the SWMM. Legacy continuous staking and Atomic Blockchain Staking are separate systems; see [Legacy staking](06-staking-legacy.md) and [Atomic staking](07-staking-atomic.md).

## Next pages

- [FAQ](01-faq.md): ownership, investment boundaries, NMR, and the hedge fund.
- [Data](02-data.md): dataset structure and download recipes.
- [Models](03-models.md): benchmarks, validation, ensembles, and neutralization.
- [Submissions](04-submissions.md): round lifecycle and submission handling.
- [API and MCP](11-api-and-mcp.md): credentials and agent integrations.
