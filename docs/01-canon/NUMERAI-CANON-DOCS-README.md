# Numerai Canon

> **Status:** Official Numerai documentation for this repository.
>
> **Scope:** Tournament data, modeling, submissions, scoring, staking, account systems, and API access.
>
> **Authority rule:** When a rule depends on a specific round, score configuration, tournament, or dataset version, use that specific configuration as authoritative. This directory documents the rules and concepts; it does not replace the current round configuration returned by the API.

## Read first

1. [Overview](00-overview.md) for the tournament lifecycle.
2. [FAQ](01-faq.md) for common misconceptions and boundaries.
3. [Data](02-data.md) for dataset structure, targets, versions, and files.
4. [Models](03-models.md) for model slots, benchmarks, validation, ensembles, and neutralization.
5. [Submissions](04-submissions.md) for rounds and submission handling.
6. [Model Uploads](05-model-uploads.md) for hosted prediction automation.

## Domain map

| Topic | Canonical page | Use it for |
| --- | --- | --- |
| Tournament orientation | [00-overview.md](00-overview.md) | What Numerai is and the end-to-end lifecycle |
| Common questions | [01-faq.md](01-faq.md) | Ownership, investment boundaries, NMR, and the hedge fund |
| Dataset | [02-data.md](02-data.md) | IDs, eras, features, targets, versions, and downloads |
| Models | [03-models.md](03-models.md) | Model slots, benchmark models, walk-forward validation, ensembles |
| Submissions | [04-submissions.md](04-submissions.md) | Rounds, schedules, late submissions, queueing, and automation choices |
| Hosted automation | [05-model-uploads.md](05-model-uploads.md) | Model Uploads, `predict`, cloudpickle, runtime limits, and CLI |
| Legacy staking | [06-staking-legacy.md](06-staking-legacy.md) | Continuous staking and payouts before the Atomic cutover |
| Atomic staking | [07-staking-atomic.md](07-staking-atomic.md) | Staking v3, allocation strategies, claims, migration, and contracts |
| Grandmasters | [08-grandmasters.md](08-grandmasters.md) | Seasons, Canon Scores, qualification, tiers, and titles |
| Live scoring | [09-scoring-live.md](09-scoring-live.md) | Score timelines, leaderboards, diagnostics, and permanent history |
| Scoring reference | [10-scoring-reference.md](10-scoring-reference.md) | Definitions, formulas, metrics, and score names |
| API and agents | [11-api-and-mcp.md](11-api-and-mcp.md) | API keys, NumerAPI, GraphQL, and Numerai MCP |

## Reading conventions

- **Round-specific configuration wins.** Score names, horizons, multipliers, payout selection, and scoring windows can change by round. See [API and MCP](11-api-and-mcp.md) for the round score configuration query.
- **Dataset aliases are not historical identities.** A generic target alias describes the current dataset version; it does not rewrite historical scoring. See [Data](02-data.md) and [Live scoring](09-scoring-live.md).
- **Legacy and Atomic staking are separate systems.** Read [Legacy staking](06-staking-legacy.md) for pre-cutover positions and [Atomic staking](07-staking-atomic.md) for the blockchain-native system.
- **Examples are illustrative.** Code samples show the shape of an integration. Use the current API response, current dataset files, and current hosted-runtime requirements when operating a live system.

## Canonical vocabulary

- **Era:** A time bucket in the dataset. Historical train and validation eras are weekly; live eras are daily.
- **Round:** A tournament submission and scoring lifecycle.
- **Target:** A future-return label associated with a dataset version and horizon.
- **CORR:** Correlation of a prediction with the configured target using Numerai Corr.
- **MMC:** Meta Model Contribution.
- **FNC:** Feature Neutral Correlation.
- **BMC:** Benchmark Model Contribution.
- **SWMM:** Stake-Weighted Meta Model.
- **NMR:** Numeraire, the tournament utility token used for staking, payouts, and burns.
