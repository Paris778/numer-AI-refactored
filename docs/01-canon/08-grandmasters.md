# Grandmasters and Seasons

> **Canonical scope:** Grandmasters seasons, Canon Scores, stake-weighted averages, qualification, tiers, and titles.

## Grandmasters

Grandmasters is a prestige system for Numerai participants. Users receive titles for stake-weighted model performance over a calendar year.

A new season starts at the beginning of each calendar year. After 20 qualifying submissions, an account receives a provisional title based on its average year-to-date stake-weighted account score ranking. The provisional title can change as the season continues.

After the final round of the season resolves and rankings are finalized, titles are awarded.

## Canon Scores

Canon Scores are official score values for a round that settle different scoring versions into a consistent metric.

For example, the payout score `CORR20` changed over time from simple Spearman correlation to the modern `numerai_corr` implementation. Canon CORR accounts for those version changes and provides one continuous score that reflects the official score for each point in time.

## Stake-Weighted Average scores

Grandmasters rankings measure both model performance and the confidence represented by stake. A stake-weighted average (SWA) balances a model's score with its stake.

For the 2022 and 2023 seasons, rankings used stake-weighted average scores derived from Canon CORR and Canon TC metrics.

A season score is the average of the account's stake-weighted average scores for all rounds included in the season. A round in which the participant neither submits a model nor places a stake contributes zero to the seasonal average.

### Example

Suppose an account has two staked models and one unstaked model in a round:

| Model | Score | Stake |
| --- | ---: | ---: |
| A | 0.01 | 10 NMR |
| B | 0.02 | 5 NMR |
| C | 0.03 | 0 NMR |

The round's stake-weighted average is:

```text
SWA = ((0.01 * 10) + (0.02 * 5)) / (10 + 5) = 0.0167
```

Model C is excluded because it is unstaked.

## Ranking tiers

At the end of a season, participants receive titles based on their ranking in each independent scoring category:

1. **Grandmasters:** First place.
2. **Masters:** Top 10.
3. **Experts:** Top 100.
4. **Researchers:** Top third.
5. **Contributors:** Middle third.
6. **Apprentices:** Bottom third.
7. **Novices:** Fewer than 20 qualified submissions.

An account can receive different titles in different scoring categories. For example, an account may be an Expert for one score and a Researcher for another.

## Qualification

A season requires 20 qualified rounds. A round qualifies when:

- the account has at least one on-time submission; and
- the account has at least 1 NMR total at risk across its models' on-time submissions.

For Numerai Crypto V3 staking rounds, the threshold is 0.1 NMR because V3 stake is split across overlapping rounds.

The 20 qualified rounds must be distinct rounds. Multiple submissions in one round do not increase the qualification count. A participant who maintains 1 NMR on one model and submits on time for 20 separate rounds meets the qualification criteria.

## Titles and Discord

Earned titles appear in several places:

- the highest title can appear as a colored profile border on leaderboards;
- titles appear on the account profile;
- hovering over a title shows the titles earned across categories; and
- after Discord integration, the highest title appears as a Discord role.

To connect Discord, open the account profile, select **Edit account profile**, select **Connect Discord account**, and authorize Numerai's Discord bot, **The Craibinator**. Titles then appear as roles and the connected Discord profile appears on the Numerai profile page.
