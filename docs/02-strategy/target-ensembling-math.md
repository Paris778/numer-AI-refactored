# Target Ensembling Math Notes

This note summarizes practical stacking approaches when combining many auxiliary-target specialists into a stronger main-target predictor.

## 1) Problem Statement

Given target-specialist prediction vectors p1..pk, build an aggregate predictor p* with better robustness and utility than any single component.

## 2) Linear Stacking Baseline

Model:

p* = sum(wi * pi)

Start with constrained linear stacking:

- Ridge or Lasso to handle multicollinearity across auxiliary predictors
- Optional non-negative constraints when directionality assumptions are desired

Why it works:

- Auxiliary targets often share latent structure with the main target
- Regularization stabilizes unstable weights

## 3) Feature-Weighted Stacking

Static weights can be too rigid. Let weights vary with context:

wi = f(X)

Where X is a compact feature context vector.

Practical implementation:

- Meta-learner takes [p1..pk plus selected context features]
- Learns when to trust each specialist

## 4) Dimensionality Reduction Route

If component predictions are highly correlated:

- Apply PCA to the prediction matrix
- Regress on top principal components

Benefit:

- Keep shared signal, suppress redundant component noise

## 5) Uncertainty-Aware Route

Multi-task probabilistic methods can model joint uncertainty between main and auxiliary tasks.

Benefit:

- Confidence-aware blending when auxiliary signals disagree

Cost:

- Higher complexity and tuning burden

## 6) Numerai-Specific Constraints

- Blend in rank domain per era, not raw regression domain
- Evaluate with era-aware metrics and leakage-safe validation
- Include uniqueness diagnostics (MMC/BMC-style behavior)

## 7) Recommended Escalation Order

1. Regularized linear stacking baseline
2. Weighted rank blend with small curated target set
3. Context-aware meta-learner
4. Probabilistic or advanced nonlinear stackers

Use each stage only if it earns measurable gains in Sharpe, drawdown, and uniqueness without harming reliability.
