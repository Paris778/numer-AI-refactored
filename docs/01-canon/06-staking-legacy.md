# Legacy Continuous Staking

> **Canonical scope:** The legacy continuous staking and payout system used before the Atomic Blockchain Staking cutover.
>
> **Status:** Historical and transitional. Atomic Blockchain Staking is the active route for tournaments after their cutover. See [Atomic staking](07-staking-atomic.md).

## What legacy staking is

Staking means locking NMR on your own submissions during their scoring period. After a staked submission finishes scoring, a positive score earns NMR and a negative score burns part of the staked NMR.

Staking is optional. Unstaked submissions are scored like staked submissions, but only staked submissions influence the Stake-Weighted Meta Model. See [FAQ](01-faq.md) for the ownership and investment boundaries.

The staking opportunity is not an offer to participate in an investment contract, a security, a swap based on the return of financial assets, an interest in Numerai's hedge fund, Numerai itself, or its fees. Participants with different expectations should not stake. Read the Terms of Service.

## Why staking exists

Numerai trades capital based on the Meta Model and needs a way to distinguish good-faith predictions from noise. Putting a participant's own NMR at risk signals confidence that the prediction is a genuine attempt to be accurate.

Burned NMR is destroyed. It does not go to Numerai, another participant, or any other party. Staked NMR is not a loan to Numerai and is not spent or traded while locked.

## NMR

NMR is the utility token used for staking, payouts, and burns. NMR can be acquired through services such as Coinbase or Uniswap, subject to regional availability and the participant's own wallet and transaction decisions.

For Atomic Blockchain Staking, NMR is sent to the Privy embedded wallet address shown on the V3 Stakes page and then deposited into the allocation strategy. See [Atomic staking](07-staking-atomic.md).

## Legacy payout formula

The legacy system's payout or burn was capped at 5% per round:

```text
score = corr20 * corr_multiplier + mmc20 * mmc_multiplier
payout = stake * clip(payout_factor * score, -0.05, 0.05)
```

Where:

- `corr20` and `mmc20` are the 20-day CORR and MMC scores.
- `stake` is the model's stake at the round close and is the at-risk stake for that round.
- At-risk stake is `0` when there is no valid submission for the round.
- The multipliers are configured per round and can change; inspect the current round rather than hard-coding them.
- Idle NMR is not at risk and is not included in the round stake.

The legacy payout factor is:

```text
payout_factor = min(1, stake_threshold / total_at_risk)
```

| Tournament | Legacy stake threshold |
| --- | ---: |
| Numerai | 72,000 |
| Signals | 36,000 |
| Crypto | 10,000 |

## Legacy withdrawal

The legacy stake release process does not apply to Atomic Blockchain Staking. For Atomic staking, reduce future exposure by changing the per-round stake and withdraw idle strategy NMR; active round stake remains locked until settlement. See [Atomic staking](07-staking-atomic.md#reducing-future-exposure).

## Cutover

The legacy system and Atomic Blockchain Staking are not interchangeable. A tournament's cutover determines which system handles new positions and payouts. For the Numerai Classic transition, Atomic staking and Ender-60 scoring start at round 1343, opening August 28, 2026. Rounds before the cutover were staked on the legacy continuous system and settle there.

Check the tournament's current round and staking configuration before taking action. See [Atomic staking](07-staking-atomic.md) for the blockchain-native lifecycle, migration, claims, and current contract inventory.

## Tax reports

Tax reports for the previous year are usually released in mid-January. They are available from the account menu under **Settings** and **TAXES & REPORTS**.
