# Feature Neutral Correlation (FNC)

## What is FNC?

Feature neutral correlation (FNC) is the correlation of a model with the target, after its predictions have been neutralized to Numerai's features.

Since features are known to be inconsistent on their own, models with too much linear exposure to features are expected to perform poorly. By neutralizing this linear exposure to features, FNC isolates the predictive performance of the model that isn't just from the feature exposure.

## Calculation

To calculate a user's FNC for a given round we

* Normalize the predictions in their submission
* Neutralize their submission to Numerai's features for that round
* Calculate the Spearman rank-order correlation of their neutralized submission to the target

The current (FNCv3) calculation follows the canonical chain in
[`00-definitions.md`](00-definitions.md) and is implemented in
`nmr/evaluation.py` (`_custom_fnc`), parity-tested against
`numerai_tools.scoring`:

```python
def calculate_fnc(sub, targets, features):
    """FNCv3 — canonical chain: tie-kept rank -> gaussianize ->
    neutralize vs [F | intercept] -> variance normalize -> numerai corr."""
    from scipy.stats import rankdata
    from scipy.stats import norm as gaussian_ppf

    n = len(sub)

    # 1. tie-kept rank, then gaussianize
    ranked = (rankdata(sub.values, method="average") - 0.5) / n
    s = gaussian_ppf(ranked.clip(1e-12, 1 - 1e-12))

    # 2. neutralize to features WITH an intercept column, then variance
    #    normalize (regression against [F | 1] via least squares)
    f = np.column_stack([features.values, np.ones(n)])
    neutral = s - f @ np.linalg.lstsq(f, s, rcond=1e-6)[0]
    neutral = neutral / neutral.std()

    # 3. numerai corr: tie-kept rank -> gaussianize -> pow 1.5 on the
    #    prediction; target is centered then pow 1.5; Pearson correlation
    r = (rankdata(neutral, method="average") - 0.5) / n
    sp = gaussian_ppf(r.clip(1e-12, 1 - 1e-12)) ** 1.5
    tp = (targets.values - targets.values.mean()) ** 1.5
    fnc = float(np.corrcoef(sp, tp)[0, 1])

    return fnc
```

Historical note: older samples in the wild used `rank(method="first")`,
no gaussianize, a pinv without intercept, and a plain Spearman correlation.
That chain does **not** match the current oracle and will fail the repo's
parity tests — do not copy it.


## FNC on the website

The current version of FNC shown on the website is called `FNCv3` which is neutral to the "medium" subset of features in the V3 data.

## Discussion

Read more about feature neutralization and feature exposure [here](https://forum.numer.ai/t/model-diagnostics-feature-exposure/899).
