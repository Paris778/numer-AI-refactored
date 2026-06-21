# Community Wisdom (Curated)

This file condenses high-signal community observations into practical heuristics.

## 1) Data and Regime Reality

- The dataset is noisy and low signal.
- Live performance commonly occurs in streaks by market regime.
- A model can be sound and still underperform for extended windows.

Implication:

- Judge models with longer validation windows and regime-aware diagnostics.
- Avoid overreacting to short live drawdowns.

## 2) Ensemble Size and Complexity

- Small, focused ensembles are often more stable than very large blends.

Implication:

- Add components only when they contribute measurable diversification or robustness.

## 3) Neutralization Proportion Is Contextual

- Earlier tutorials often showcase 0.5 neutralization.
- Later-era behavior may benefit from lower values (for example 0.25).

Implication:

- Sweep neutralization proportion as a hyperparameter.
- Choose by Sharpe and drawdown, not only mean CORR.

## 4) Retraining Frequency Has No Universal Rule

Observed patterns:

- Weekly retraining is common in low-cost automated setups.
- Some static models remain competitive for many months.

Implication:

- Use step-forward retraining experiments to identify your optimal cadence.
- Consider blending a stable long-horizon model with a frequently retrained model.

## 5) Data Refresh Facts

- Train data changes on version updates.
- Validation grows over time.

Implication:

- Version your experiments by dataset version and validation cut date.

## 6) Staking Lifecycle Reminder

- Stakes are applied per round and resolve after the round completes.
- Economic outcomes are delayed by the round timeline (roughly monthly for Classic/Crypto).

Implication:

- Expect delayed feedback loops for capital changes.

## 7) CORR Geometry Intuition

Community math intuition (Andralienware worked example):

- CORR is effectively a dot product between transformed predictions and the target, divided by the product of their standard deviations. You cannot change the target, so you only control the prediction vector.
- This makes CORR-maximization equivalent to maximizing dot product subject to an L2-norm constraint on predictions.
- Extreme predictions dominate the prediction's variance; middling predictions contribute little variance.
- Therefore, filling uncertain predictions with the neutral value (e.g. 0.5 / zero after centering) usually HURTS expected CORR. Even a slightly-better-than-random guess on uncertain stocks adds more expected dot product than the variance it costs.

Implication:

- Do not zero-fill or flatten predictions you are merely uncertain about; emit your best ranked guess.
- Preserve meaningful rank dispersion wherever the model has any signal.
- The full numeric derivation is in `99-archive/raw-source/community-notes.md`.

This file is heuristic guidance, not protocol law.
