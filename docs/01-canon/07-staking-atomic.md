# Atomic Blockchain Staking

> **Canonical scope:** Numerai's blockchain-native staking system, also called staking v3 in API and contract names.
>
> **Numerai Classic status:** Atomic staking becomes the active route at the Ender-60 cutover, round `1343`, opening August 28, 2026. Earlier rounds settle through the legacy continuous staking system; see [Legacy staking](06-staking-legacy.md).

## System model

Atomic Blockchain Staking (ABS) defines one stake position for each `(round, staker, model)` tuple. Each position is backed by NMR locked in the staking contract instead of sharing one continuous stake across several unresolved rounds.

When a round resolves, the position can be claimed for its original stake plus a payout or minus a burn. A participant can also claim and restake atomically in one transaction. Either both operations succeed or the transaction reverts.

Participants can interact with contracts and APIs directly or automate staking with an allocation strategy. Most participants should use **Dashboard > V3 Stakes**, which manages the Privy embedded wallet and allocation-strategy flow.

> **Important:** The opportunity to stake is not an offer by Numerai to participate in an investment contract, a security, a swap based on the return of financial assets, an interest in Numerai's hedge fund, Numerai itself, or its fees. Participants with different expectations should not stake. Read the Terms of Service.

## Getting started

1. Open **Dashboard > V3 Stakes**.
2. Authenticate the Privy embedded wallet with an email code. The wallet is assigned to the account and shared across tournaments. It owns the participant's stakes and allocation strategy.
3. Enable the account's allocation strategy. The strategy is a smart contract owned by the wallet and invocable by Numerai. There is one strategy per account per tournament.
4. Deposit NMR into the allocation strategy. This is idle NMR until it is deployed into a staking position.
5. Configure each model's per-round stake and payout mode, then save the model settings.
6. Continue submitting normally. If sufficient idle NMR is available, Numerai invokes the strategy and stakes automatically.

When a round resolves, its position becomes claimable. If the participant continues submitting, Numerai automatically claims the resolved stake and restakes it on a later round. Otherwise, the position remains claimable until it is claimed or automatically finalized on the first scoring-resolution run after the seven-day grace period.

The V3 Stakes page exposes Stake Activity and Wallet Activity, including transfers and automatic stakes. A submitted staking transaction remains visibly pending through on-chain confirmation and backend indexing. If indexing is delayed, use **Retry refresh** or the always-available **Refresh** control beside Stake Activity. The last confirmed values remain visible until refresh succeeds.

## Allocation strategies

An account has one allocation strategy per tournament. Each model configuration contains:

- a per-round stake amount;
- a staking mode, either `constant` or `compound`; and
- an automation flag.

### Per-round stake

The per-round stake is the NMR locked for one position. For a target total value locked (TVL) and `R` concurrently unresolved rounds:

```text
per-round stake S = TVL / R
```

| Tournament | Concurrent rounds R | Per-round stake S |
| --- | ---: | --- |
| Numerai | 64 | TVL / 64 |
| Signals | 64 | TVL / 64 |
| Crypto | 24 | TVL / 24 |

For example, allocating 64 NMR to Numerai means staking 1 NMR per round once all 64 positions are active.

### Staking modes

ABS supports two automated modes:

- **Constant:** Attempts to stake exactly the configured per-round amount. If insufficient NMR is available, it stakes whatever is available.
- **Compound:** Attempts to stake exactly the configured per-round amount until payouts begin, then compounds payouts into future stakes. If insufficient NMR is available, it stakes whatever is available.

### Automation flag

Disabling automation effectively sets the model's per-round stake to `0` and tells Numerai to stop invoking the model. Existing positions continue through settlement; resolved payouts and burns still settle automatically.

### Idle strategy balance

The allocation strategy can hold NMR that is not currently locked. It must contain idle NMR before it can stake future rounds.

Under **Constant** mode, settlement value above the configured per-round stake returns to the idle balance and can top up a later position after a burn. A settlement below the configured per-round stake funds the next stake instead of returning value to idle balance.

Under **Compound** mode, payouts roll into future stakes rather than the idle balance. Each invocation compounds only the newest settled claim. If claims accumulate, for example while a model is paused, older claims are finalized into the idle balance after the seven-day grace period.

## Round lifecycle

1. **Create and open:** Numerai creates the round in its database and on-chain. On-chain details include the tournament ID and timing derived from the off-chain round. The on-chain staking window closes after the off-chain round so Numerai can invoke enabled strategies.
2. **Authorize and stake:** For each eligible selected submission, Numerai signs an EIP-712 authorization binding the tournament, round, staker, model, submission hash, maximum amount, nonce, and deadline. The staker or allocation strategy submits the authorization and amount. The contract verifies the signature and nonce, records one position for `(round, staker, model)`, and permits only one staker per `(round, model)`.
3. **Score and prepare settlement:** Numerai computes scores daily. After the final scoring day, it calculates each position's payout or burn and creates a claim containing the round, staker, model, payout amount, and burn amount. Claims form a Merkle tree.
4. **Resolve on-chain:** Numerai posts the Merkle root and aggregate payout and burn amounts and funds the contract for payouts. The contract marks the round resolved and verifies that payout reserves are sufficient. Positions become claimable.
5. **Claim or claim and restake:** A stake claim supplies the payout, burn, and Merkle proof. The contract verifies the proof and settles the original stake net of payout or burn. A participant can claim manually or allow the allocation strategy to claim and restake automatically later.

During automatic processing, Numerai attempts to claim payouts in descending order of model stake and then burns in descending order of model stake. This allows payouts, when available, to cover some or all burns.

Resolved positions do not remain unclaimed indefinitely. On the first scoring-resolution run after a position has been unclaimed for more than seven days, Numerai finalizes it and returns the remaining NMR to the staker after applying the payout or burn.

### Absorbed accounts and orphaned strategies

If a model moves because one account is absorbed into another, its on-chain strategy remains associated with the old account until settlement cleanup. After finalizing a position, Numerai transfers released NMR from the orphaned strategy to the current account's strategy, or to its wallet if no strategy exists.

Numerai also drains idle NMR from an orphaned strategy that has position or funding history but no resolved claim. For an otherwise unused strategy discovered only from its factory, NMR moves only after Numerai recovers a safe model-linked current account. Until then, it remains in the strategy rather than being sent to the disabled account's wallet.

## Contract upgrades

Allocation strategies are tied to the staking contract generation where they were created. When a tournament upgrades to a new generation, an old strategy cannot restake directly into the replacement contract.

After each old round resolves, Numerai finalizes every remaining claim on the old contract, including positions staked directly from a wallet, and sweeps the old strategy's full NMR balance into the owner's Numerai Wallet. If the account has a replacement strategy, Numerai moves the swept NMR into it automatically. Otherwise, it remains in the legacy strategy owner's wallet and can fund a new strategy.

If old models now belong to another account, NMR moves to that account's replacement strategy or wallet. New automated stakes use the replacement strategy. Participants do not need to claim or move migrating positions manually; old strategies are re-checked each round.

## Payouts, burns, and claims

For an Atomic staking round:

```text
weighted_score = sum(score * multiplier for each payout score)
round_return = clip(payout_factor * weighted_score, -1, 1)
settlement = round_stake * round_return

payout = max(settlement, 0)
burn = max(-settlement, 0)
claimable = round_stake + payout - burn
```

For ABS, `payout_factor = 1` and the clip is `+/-1`. The legacy 5% per-round clip is removed, but a position cannot gain or lose more than 100% of the NMR staked in that round.

Score types and multipliers are tournament and round configuration, not constants in the staking contract. Check the current round before modeling returns.

### Merkle claims

A resolved round has one model-scoped claim for each position. The contract stores the Merkle root and aggregate liabilities rather than every claim. `v3StakeClaim` returns the individual payout, burn, and proof needed to verify a leaf.

Claims cannot be redirected between models or reused. The contract marks `(round, staker, model)` as claimed and decrements remaining payout and burn amounts as claims are collected.

### Burn netting during restaking

A direct claim returns:

```text
releasable NMR = round stake + payout - burn
```

When a resolved claim is atomically restaked, its burn can become deferred burn debt for the same model. A later payout for that model offsets the debt before becoming a net payout.

Deferred burn debt:

- cannot be withdrawn or used as liquid NMR;
- cannot fund a stake or count toward Compound mode;
- cannot be netted against another model; and
- remains backed by NMR in the staking contract until offset or realized as a burn.

If a participant stops restaking and claims directly, the outstanding burn is finalized rather than carried forward.

## Numerai Classic transition

Numerai Classic transitions from legacy Ender-20 staking and payouts to Atomic staking and Ender-60 payouts at round `1343`, beginning August 28, 2026. The transition uses `3 * CORR60 + 15 * MMC60`.

A historical simulation across rounds 1172-1260 compared legacy Ender-20 and Atomic Ender-60 participant returns after accounting for the 1/64 per-round stake, payout factor of 1, and longer compounding horizon. This is informational only, is not an advertisement to stake, and does not predict future returns.

## Reducing future exposure

Lower the model's per-round stake to reduce future exposure. Setting it to `0` is the Atomic equivalent of requesting a full release: no new positions open, while existing positions become available as they resolve and are claimed.

NMR already staked in an active round cannot be removed early. Idle strategy NMR can be withdrawn to the Privy embedded wallet at any time. The legacy v2 stake-release process does not apply to ABS.

### Withdrawing NMR

1. Withdraw idle NMR from the allocation strategy to the Privy embedded wallet on the V3 Stakes page.
2. Use the embedded wallet controls to send NMR to an external Ethereum address.

Active round stake remains locked until settlement.

> **Transaction warning:** If a wallet or strategy transaction is pending, do not submit it again. Use **Check status** in Stake Activity to query its receipt. Leaving or reloading the V3 Stakes page does not cancel a submitted transaction, but monitoring and displayed balances may remain stale until the page is refreshed.

## V2-to-V3 migration

When a tournament starts migration, its legacy off-chain staking controls are disabled. The staking UI changes to a Web3-native page that creates and controls an allocation strategy through a non-custodial Privy wallet.

The legacy Numerai Wallet remains accessible at [numer.ai/wallet](https://numer.ai/wallet) for withdrawals and wallet history, but it is no longer the place to manage new stakes after migration.

Legacy staked NMR was held in the wallet with off-chain accounting. ABS moves that NMR into blockchain staking contracts. At each legacy round resolution, a portion transfers to an Atomic position: generally 1/24 for a 24-round tournament and 1/64 for Numerai Classic's 64-round overlap. A model that skipped legacy rounds can release a larger share because fewer legacy positions hold stake at risk.

Participants do not need to perform migration transactions. Numerai provisions the Privy wallet and allocation strategy and moves the legacy stake as rounds resolve.

### Migration schedule

| Tournament | Migration starts | Expected migration length |
| --- | --- | --- |
| Crypto | June 16, 2026 | 24 business days |
| Signals | TBD | 64 business days |
| Numerai Classic | August 28, 2026 | 24 business days |

At cutover, pending v2 changes are retired before migration. Pending increases or decreases are reflected in initial allocation-strategy settings. A pending decrease is not released to the Numerai Wallet; the legacy request is cancelled. The resulting stake moves to the strategy over the migration schedule.

## Return accounting

Return percentages on model and account pages represent profit over the NMR committed. Profit is the payouts and burns for rounds settled in the reporting window.

Committed NMR is NMR actually locked in positions: what was locked when the window opened plus, for each round, any stake exceeding what earlier positions had already released. Idle NMR in an allocation strategy is not committed because it is not staked and earns nothing. Restaking settled NMR is not new capital in either mode.

Consequences:

- Loading a stake takes a full overlap period. Staking 1 NMR per round deploys 64 NMR on Numerai or Signals and 24 NMR on Crypto once every round is live.
- Compound mode restakes settlements and can earn more on the same committed NMR than Constant mode, which leaves released value idle.
- Reusing settled NMR does not raise the capital basis. Pausing still produces no return for skipped rounds and stops compounding.
- Lowering the per-round stake in Compound mode does not necessarily reduce exposure while settlements exceed the new value, because Compound can stake the released value. To reduce exposure, use Constant mode, which stakes exactly the configured per-round amount and leaves the rest idle.

## API and automation

API tokens that automate ABS require the **Manage web3 staking** (`web3_staking`) scope. See [API and MCP](11-api-and-mcp.md) for authentication and scope handling.

Important GraphQL fields:

- `v3StakeConfig`: staking contract metadata for a tournament.
- `v3StakeRound`: on-chain round status.
- `v3StakeAuth`: a staking authorization for an eligible selected submission.
- `v3StakeClaim`: a claim proof for an authenticated model and staker after resolution.
- `v3StakeClaims`: resolved, unclaimed proofs and each `claimableAmountWei` for an authenticated account's models at a staker address, scoped to one tournament.

Important `v3StakeRound` fields:

| Field | Meaning |
| --- | --- |
| `state` | Derived state such as open, closed, resolving, or resolved. |
| `openTime`, `closeTime`, `resolveTime` | On-chain round time boundaries. |
| `totalStaked` | NMR principal locked across positions. |
| `totalPayout` | Aggregate positive payout in the posted settlement tree. |
| `remainingPayout` | Payout reserve not consumed by claims. |
| `remainingBurn` | Burn amount not consumed by claims. |
| `merkleRoot` | Root used to verify model-scoped claims. |
| `payoutFactor` | Deprecated legacy compatibility field; null for upgraded contracts. Read payout policy from tournament round configuration. |
| `stakeThreshold`, `stakeCap` | Deprecated aliases for a legacy contract value; null for upgraded contracts. Read stake policy from tournament round configuration. |

Once ABS is enabled for a tournament, legacy V2 staking mutations for that tournament are disabled. Use the Atomic flow and v3 API fields.

> **Authorization warning:** Authorizations are short-lived, nonce-protected, and bound to a selected submission. Do not cache or replay an authorization after changing a submission or after another transaction consumes the staker's nonce.

## Wallet support and contract inventory

The website currently uses a Privy embedded wallet. Contract-aware participants can interact with deployed contracts directly, but they must obtain valid stake authorizations and claim proofs from the API and are responsible for gas and contract integration.

Each tournament publishes its own contract inventory. Tournament-specific pages are the source for active routes, retained settlement contracts, dormant networks, and verified explorer links.

- Numerai (tournament `8`): Ethereum mainnet; active from the Ender-60 Atomic cutover at round `1343`.
- Numerai Crypto (tournament `12`): Ethereum mainnet; active Crypto staking contract.

USDC staking is not currently implemented. It is planned as a future v3 update with a separate USDC pool.

## FAQ

### Does lower leverage mean lower payouts?

Yes and no. Legacy continuous staking generated leverage from overlapping rounds and payout clips. ABS removes that overlap leverage: a position can gain or lose at most 100% of the amount staked for that round.

### Can I bring my own wallet?

The contracts are public and contract-aware participants can interact with them directly or instantiate an allocation strategy. The website rollout uses the non-custodial Privy wallet. Support for other wallet solutions depends on the current website implementation.

### What happens to a pending stake release?

At migration, pending changes are reflected in the initial allocation strategy settings. The legacy request is retired, and the resulting funds move through the migration schedule. They can then be reallocated or withdrawn through ABS.

### Will NMR be burned immediately or netted?

In the claim-and-restake flow, burns can accumulate as per-model deferred debt and be offset by future payouts. If a model stops staking or claims without restaking, the outstanding burn is finalized.

### Does an inactive staked slot still hold stake?

A slot that no longer submits has no new eligible position. In ABS, released value becomes idle in the allocation strategy after it is claimed, and can be withdrawn or used for later staking.
