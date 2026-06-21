# Staking

## Introduction

When you are ready and confident in your model's performance, you may optionally stake it with [NMR](https://www.coinbase.com/price/numeraire) - Numerai's cryptocurrency. "Staking" on a model or submission means locking up NMR during the scoring period of submissions.

By staking on Numerai, for the duration of staked submissions, you give Numerai permission to payout or burn the NMR at stake. After a staked submission finishes [scoring](https://docs.numer.ai/numerai-tournament/scoring), positive scores are rewarded with more NMR, while negative scores cause staked NMR to be *burned*. This serves two functions:

1. "Skin in the game" allows Numerai to trust the quality of staked predictions.   
2. Payouts and burns continuously improve the weights of the [*SWMM*](https://docs.numer.ai/scoring/definitions#meta-models). 

It is important to note that the opportunity to stake is not an offer by Numerai to participate in an investment contract, a security, a swap based on the return of any financial assets, an interest in Numerai’s hedge fund, or in Numerai itself or any fees we earn. Users with different expectations should not stake.

**Please read our** [**Terms of Service**](https://numer.ai/terms) **for further information.**

## Numeraire (NMR)

NMR is the cryptocurrency that powers staking and payouts on Numerai.

* You can read more about NMR on Github: <https://github.com/erasureprotocol/NMR>
* You can see token statistics here: <https://numer.ai/nmr>

To stake NMR on your model you must first acquire NMR, then deposit it into your [Numerai wallet](https://numer.ai/wallet). When you are ready, you can withdraw it to a given eth address.

## Depositing NMR

We recommend two places for buying NMR&#x20;

* [Coinbase](https://www.coinbase.com/price/numeraire) - a safe and easy place to buy NMR with USD, GBP, EUR or Bitcoin. A good place to start if you are new to crypto. Availability depends on your region.
* [Uniswap](https://app.uniswap.org/#/swap?outputCurrency=0x1776e1f26f98b1a5df9cd347953a26dd3cb46671) - a decentralized exchange you can use to swap Ethereum based tokens like ETH or USDC for NMR. Requires you to bring your own wallet like MetaMask. Great for DeFi enthusiasts.

Once you own NMR, use the [Wallet](https://numer.ai/wallet) address to send NMR to your Numerai Wallet.

### Restrictions

To prevent fraud and money laundering, we place withdrawal restrictions on new accounts. In order to withdraw from your Numerai Wallet, your account must either be 30 days old or you must withdraw more than 0.1 NMR. This means if you just created your account and deposit 0.1 NMR, you cannot withdraw this amount until either your account reaches 30 days of age or you deposit more NMR first.

## Increasing Stake

Head over to [numer.ai/staking](https://numer.ai/staking), and click on the "manage stake" button next to the model on which you would like to stake.

This will open the Stake Modal. You can use this to stake NMR from your wallet on your model.

## Releasing Stake

Staked NMR will remain locked until you release it back to your wallet, which takes 1 month. Specifically, the NMR will be released at the resolution of your last active round at the time of the request. While pending release, the unstaked amount may still be subject to burns but will not count towards upcoming payouts.

## Payouts

Your payout is a primarily a function of your scores. If you have a positive score you will get a payout. If you have a negative score a portion of your stake will burn.

The maximum payout or burn per round is capped at ±5% and uses the following formula:

```
payout = stake * clip(payout_factor * (corr * 0.75 + mmc * 2.25), -0.05, 0.05) 
```
Updated formula : 0.75corr + 2.25mmc

`stake` is your model's stake value at the `close` of the round. This is also referred to as the stake value `at-risk` for a round. Your stake value `at-risk` for a round does not include any unstaked amounts that are pending release, and is set to 0 if you have no valid submission for a round.

#### The Payout Factor

The `payout_factor` is a dynamic value that scales inversely with total NMR staked based on the `staking_threshold`.&#x20;

```python
payout_factor = min(1, stake_threshold / total_at_risk) 
```

| Tournament | Stake Threshold |
| ---------- | --------------- |
| Numerai    | 72000           |
| Signals    | 36000           |
| Crypto     | 10000           |

## Withdrawing NMR

When you are ready to withdraw NMR from your Numerai Wallet visit the [Wallet Page](https://numer.ai/wallet), specify an amount, and copy your destination address into the relevant text box.

Finally, click "Next" and wait for the page to show a confirmation. It may take a few moments to complete the transaction, so please be patient.

## Tax Reporting

We usually release tax reports for the previous year in mid-January. You can find these by hovering over the account icon in the top right > click "Settings" > "TAXES & REPORTS" section.&#x20;
