# Why The Numerai Framework Works

This document explains the system-level objective behind Classic tournament modeling.

## 1) The Three-Layer System

### Hedge fund execution layer

- Numerai operates a market-neutral hedge fund.
- The fund uses the aggregated meta-model signal for capital allocation.

### Crowdsourced model layer

- Users submit ranked predictions from obfuscated data.
- Numerai combines staked submissions into the stake-weighted meta model.

### Crypto incentive layer

- NMR staking creates skin in the game.
- Positive contribution can earn payouts; negative contribution can burn stake.

Together, these layers align model quality with capital weighting.

## 2) Why Orthogonality Matters

If all participants submit near-identical signals:

- Capital concentrates on one consensus vector.
- Regime shifts can cause synchronized failure and deep drawdowns.

MMC exists to reward unique target-aligned signal, not just consensus correlation.

## 3) Why Sharpe Dominates Raw Mean

Institutional deployment prioritizes risk-adjusted return.

- A volatile high-mean signal has low safe capacity.
- A steadier medium-mean signal can be levered safely and often has greater portfolio utility.

Therefore:

- Stability, drawdown, and covariance behavior determine practical deployability.

## 4) Why The Standard Pipeline Looks The Way It Does

- Era-aware purged validation: prevents target-overlap leakage.
- Rank-domain transformations: match production scoring geometry.
- Neutralization: controls unstable linear exposures.
- Target/seed ensembling: improves robustness and uniqueness.

This is not ceremony; it is adaptation to adversarial, non-stationary market data.

## 5) Practical Objective For Researchers

Produce a repeatable factory of signals that are:

- positively aligned to target,
- stable across eras,
- partially orthogonal to crowd consensus,
- economically viable under staking mechanics.

That combination is what the capital allocator can use at scale.
