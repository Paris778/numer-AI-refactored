# Live Scoring

> **Canonical scope:** Score types, daily score timelines, final resolution, leaderboards, diagnostics, and permanent score history.

For formulas and statistical definitions, see [Scoring reference](10-scoring-reference.md). For round score configuration fields, see [API and MCP](11-api-and-mcp.md).

## Score types

### Payout scores

The two primary scores currently used for payouts are:

- **CORR:** Correlation of the prediction with the configured target.
- **MMC:** Meta Model Contribution of the prediction.

### Informational scores

The following scores are informational and are not used directly for payouts:

- **FNC:** Correlation with the target after neutralizing the prediction against Numerai's features.
- **CWMM:** Correlation of the prediction with the Meta Model, which is a stake-weighted average of predictions.
- **BMC:** Correlation contribution against the stake-weighted Benchmark Models. In diagnostics, BMC uses a single benchmark trained on the payout target rather than the live stake-weighted benchmark set.

Score names, target identities, horizons, multipliers, and payout selection are properties of the round's score configuration. See the current round rather than assuming that a generic dataset target alias describes historical scoring.

## Daily scoring

Within a round, submissions receive daily score updates until the configured 20-day or 60-day score is final.

### 20D2L timeline

A 20D2L score uses 20 days of returns after 2 lag days. A historical 20-day weekend round illustrates the sequence:

1. The round opens on Saturday and closes on Monday.
2. The first scoring day is Friday, four days after close.
3. The first score uses a 1D2L target: one day of returns after two lag days.
4. The second score uses a 2D2L target and includes two days of returns.
5. The final score uses a 20D2L target and is released about four weeks after the first scoring day.

The four-day delay consists of two days of data processing plus the two-day return lag. Scores are normally updated Tuesday through Saturday.

### 60D2L timeline

A 60D2L score follows the same logic with 60 days of returns after two lag days. It resolves 40 business days later than the corresponding 20D2L process, roughly 12 weeks after the round opens. Rounds paying out on a 60D2L score lock stake for the longer scoring period.

### Ender-60 cutover

The round's configured target is authoritative. The v5.3 generic target alias does not change historical results. Historical Ender-20 rounds remain 20-day rounds.

Starting with the Ender-60 scoring cutover at round `1343`, CORR, MMC, and BMC use the 60-day Ender target. Live BMC uses the 60-day Ender target and resolves on the 60D2L timeline; its public name remains BMC.

## Leaderboards and reputation

Only final scores count toward live model performance. Numerai ranks accounts and models separately.

A model's reputation is its one-year average score and determines model-leaderboard rank. An account's reputation is a stake-weighted average of its models' scores: each model score is multiplied by that model's proportion of the account's stake and the products are summed.

If an account has one unstaked model, its account score is the model's score. An account with no staked model has no staked contribution to the SWMM.

## Diagnostics

Diagnostics compute and chart scores over the validation dataset. v5.3 diagnostics use:

- explicit `target_ender_60`;
- the `v53_lgbm_ender60` benchmark; and
- matching Ender-60 example predictions.

Validation eras whose 60-day target has not matured are not eligible for target-dependent diagnostics. Filter the exact target used by the diagnostic instead of assuming all validation rows are eligible.

Model Uploads automatically run validation diagnostics after a hosted model succeeds. Participants can also start diagnostics manually from the Scores page.

> **Warning:** Past performance does not guarantee future performance, especially in financial machine learning. Validation metrics can be overfit. Training on validation data makes those metrics in-sample and weakens their out-of-sample meaning.

## Score history is permanent

A model's round history cannot be reset, cleared, or zeroed. The following actions do not change scores already earned:

- retraining or changing the modeling approach;
- re-uploading a `.pkl` through Model Uploads;
- switching automation methods or disabling Model Uploads;
- archiving and later unarchiving a model; or
- skipping rounds and later resuming submissions.

A changed model or pipeline affects only future predictions. Past scores remain in the model record.

Reputation still changes as a one-year trailing average. Scores age out of the reputation window even though they remain permanently recorded. This is a rolling window, not a history reset.

To start a new track record, create a new model and submit to it.
