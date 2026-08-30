# Frequently Asked Questions

> **Canonical scope:** Common questions about submissions, ownership, NMR, the tournament, and the hedge fund.

For a step-by-step introduction, start with [Overview](00-overview.md).

## What you submit

### Do I have to share my model or code with Numerai?

No. You submit a prediction file: one floating-point value for each `id` in the live data. Numerai receives those values. Your code, data, features, and trained model stay on your machine.

There is one opt-in exception. Model Uploads require you to upload a pickled model that Numerai can unpickle and run. If you do not want Numerai to access the model, use Numerai CLI or your own infrastructure and submit prediction files instead. See [Model Uploads](05-model-uploads.md).

### Why submit predictions instead of keeping my model proprietary?

Participating does not require giving up your model. You provide a signal, not the intellectual property behind the model. Numerai aggregates independent signals into a Meta Model, and participants can earn NMR for predictions that perform well without running a fund, raising capital, or executing trades themselves.

### Does Numerai see my data sources?

No. In Numerai Signals and Numerai Crypto, where participants bring their own data, Numerai receives the submitted signal values rather than the underlying sources.

## What you can and cannot invest in

### Can I invest in the Meta Model?

No. The Meta Model is not an investment product, and there is no way to buy exposure to it. In Numerai and Signals, the Meta Model is not published; it trades the hedge fund's own capital. The Numerai Crypto Meta Model is published to participants, but Numerai's hedge fund does not trade crypto or use the Crypto Meta Model.

### Can I stake on, invest in, or back another user's model?

No. Staking works only on your own submissions. This is what makes a stake a credible signal of the participant's confidence.

You can download Numerai Benchmark Model predictions. You can also buy predictions that participants choose to list on NumerBay, an independent community marketplace. Numerai does not endorse, review, or guarantee anything sold there. In both cases, you submit and stake under your own model and at your own risk.

### Can I invest in the hedge fund?

Only accredited investors can invest in the hedge fund. Contact `investing@numer.ai` for information. This is separate from the tournament and from NMR.

### Is fund performance disclosed?

No.

## NMR

### What is NMR?

NMR (Numeraire) is the utility token used for staking, payouts, and burns. In the tournament, it is what a participant puts at risk on their own predictions: positive scores can pay NMR, while negative scores can burn part of the stake.

See [Legacy staking](06-staking-legacy.md) and [Atomic staking](07-staking-atomic.md) for more detail.

### Does holding NMR give me ownership or rights over Numerai?

No. NMR is not equity and provides no ownership, shareholding, voting or governance rights, dividends, revenue or profit share, or claim on Numerai, its hedge fund, its assets under management, or its fees. Holding or staking NMR does not make a participant an investor in the fund.

> **Important:** The opportunity to stake is not an offer by Numerai to participate in an investment contract, a security, a swap based on the return of financial assets, an interest in Numerai's hedge fund, or in Numerai itself or its fees. Participants with different expectations should not stake. Read the Terms of Service.

### What is NMR's relationship to the hedge fund?

NMR is the tournament's settlement asset, not a claim on the fund. The tournament and the fund are connected by incentives. A participant's stake is not lent, spent, or traded by Numerai. Burned NMR is destroyed rather than transferred to Numerai or another participant.

### Why are payouts in NMR instead of USD?

USD cannot be burned. NMR can be sent to a null address, making it unusable by anyone. When NMR is burned because of poor performance, it disappears rather than being transferred to another party.

## The tournament and the fund

### What kind of hedge fund does Numerai operate?

Numerai operates a quant global equity market-neutral hedge fund.

### Does the fund trade crypto?

No. Numerai hosts the Crypto tournament, but neither Numerai nor its hedge fund trades cryptocurrencies, and Numerai Crypto does not feed the hedge fund.

### Do I have to stake?

No. Unstaked submissions are scored like staked submissions, so participants can build a live track record without risking NMR. Only staked submissions influence the Stake-Weighted Meta Model used by the hedge fund.

### Why cannot I trade the predictions myself?

The Numerai dataset is obfuscated, so a participant cannot map a prediction back to a stock. Tournament predictions are therefore not directly tradable outside Numerai. Numerai Signals is the relevant tournament for a signal that participants can also trade themselves.

### Can I lose my score history or start over?

Score history is permanent. Creating a new model is the only way to start from zero. See [Live scoring](09-scoring-live.md) for the history rule.
