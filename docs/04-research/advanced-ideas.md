You’re already doing the “obvious good stuff” (multi-target LGBM ensemble, SNNR‑based target selection, embargoed validation), so real gains now probably require either: (a) smarter boosting / objective design, or (b) using recent tabular DL as a meta‑layer or representation learner on top of your trees.

Below are concrete research‑driven directions, each tied to recent work and to how you’d actually plug it into your current Numerai stack.

***

## 1. LightGBM–level upgrades beyond “deeper + more trees”

### 1.1. Treat HPO as a serious research problem, not tuning

Recent benchmarking of GBDT variants (XGBoost, LightGBM, CatBoost) shows that with a fixed time budget, *Bayesian* hyperparameter optimization (TPE/SMBO) can move you a non‑trivial margin beyond “reasonable” hand‑tuned settings, and that the relative ranking of libraries depends heavily on the optimized region of the space. For Numerai, that suggests: [arxiv](https://arxiv.org/pdf/2305.17094.pdf)

- Run a proper TPE/Bayesian search over:
  - `max_depth`, `num_leaves`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `lambda_l1/l2`, `min_gain_to_split`.
  - Learning‑rate schedules (e.g., start higher then cosine/linear decay) and n_estimators jointly instead of fixing 30k + 0.001 everywhere.
- Optimize directly on *era‑mean CORR or a Sharpe proxy* from your `calculate_metrics` util, not on RMSE/logloss.

You can structure search per target (Ender, Victor, etc.) and then reuse the same “shape” of params for similar targets; this is precisely the kind of thing the “Why do trees beat DL on tabular?” paper points to: trees are very sensitive to how you regularize depth/leaves vs feature subsampling. [arxiv](http://arxiv.org/pdf/2207.08815.pdf)

**How to plug in:**  
Wrap your existing training block for a single target in an objective callable that:

1. Trains a model with given params on a fixed training era window.
2. Scores on your current validation set with your era‑aware metric.
3. Returns negative Sharpe / negative mean CORR as the loss.

Then run Optuna / hyperopt around that. This is low‑risk and could easily give you a few percent in mean CORR if you’ve mostly used “sensible” but not aggressively optimized params so far.

***

### 1.2. Gradient boosting variants: dropout & cyclic boosting

Recent surveys of “enhanced GBDT” point out that dropout‑style boosting (like XGBoost’s DART) and cyclic GBM variants can improve generalization on noisy, over‑parameterized tabular problems. [arxiv](http://arxiv.org/pdf/2412.14916.pdf)

You can’t directly turn LGBM into DART, but you can **mimic dropout boosting**:

- During training, for each iteration or small block of iterations:
  - Temporarily subsample existing trees when computing predictions (randomly drop a fraction).
  - Fit the next tree on the residuals defined by this noisy ensemble.
- At inference, use the full ensemble (no dropout).

This pushes the ensemble toward more robust, less co‑adapted trees – very similar spirit to DART, but implemented on top of LightGBM’s Python API.

**How to plug in:**  
Implement a custom boosting loop using `lgb.train` with `init_score` rather than using `LGBMRegressor` end‑to‑end:

- Maintain your own `current_pred` vector and a list of trees.
- Before fitting tree \(k\), randomly mask some previous trees when computing `current_pred`.
- Train tree \(k\) on residuals; store it and update `current_pred` with the full ensemble.

You could do this only for the most important 2–3 targets (Ender, Victor, Teager) to keep training time sane.

***

### 1.3. Probabilistic / distributional boosting and tail‑aware objectives

Recent work on probabilistic GBDT (PGBM, XGBoostLSS, NGBoost variants) shows that modelling *distributions* instead of point predictions often improves risk‑sensitive metrics in finance and insurance. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10611362/)

For Numerai, you don’t need explicit predictive distributions, but you *do* care about:

- Getting the ranking right in the tails (extreme under/over‑performance).
- Stability of sign and magnitude across regimes.

Ideas:

- Use quantile objectives (or approximate them) in LightGBM for a subset of targets; then *derive* your final prediction as a function of several quantiles (e.g. median + skew proxy), and rank that.  
- Implement a custom loss that **overweights errors in outer target bins** (0 and 1 for Numerai). This approximates a payout that rewards calling the big winners/losers right.
- Combine with your current ensemble: for each target, have one standard CORR‑optimized LGBM and one tail‑focused LGBM, then meta‑learn how to combine them per era.

***

## 2. Deep tabular models as *meta‑models* over your LGBMs

The last 2 years produced a credible line of transformer/tabular architectures that can match or beat GBDTs under some regimes: ExcelFormer, tabular transformers with stochastic competition, TabDPT (tabular foundation model), and RuleNet. At the same time, large benchmarks still show GBDTs winning “raw” on many tabular tasks – but combinations of DL + trees can outperform trees alone. [arxiv](http://arxiv.org/pdf/2407.13238.pdf)

Given you already have strong tree ensembles, the sweet spot for Numerai is:

> Use deep tabular models as **stackers / representation learners** on top of (features, tree predictions) rather than trying to replace the trees outright.

### 2.1. ExcelFormer‑style meta‑learner

ExcelFormer is a Transformer variant that outperforms prior tabular DL and often beats tuned GBDTs on CC18/CTR23 benchmarks. Its design focuses on: [arxiv](https://arxiv.org/html/2301.02819v5)

- rotational robustness of features,
- reduced data demand,
- avoiding over‑smoothing.

**How to adapt it:**

- Inputs to meta‑model:
  - Numerai features (small or medium set).
  - Your per‑target LGBM predictions (Ender, Victor, Teager, etc.).
  - Optionally, benchmark model predictions and simple era‑level stats (vol, mean pred).
- Target:
  - Either the main Numerai target (Ender/Cyrus20) or your *current* ensemble prediction if you want a pure distillation first, then fine‑tune on real target.

You don’t need full ExcelFormer re‑implementation: you can approximate the idea with a tabular transformer (e.g. TabTransformer) plus cross‑feature attention and some feature masking; but if you’re up for it, porting ExcelFormer from the paper codebase and swapping in Numerai is a good PhD‑level side‑project.

Training details:

- Keep strict era‑based train/val splits like now.
- Use small batches but large number of eras; heavy regularization (dropout, feature masking, stochastic depth) as recommended in tabular DL survey. [arxiv](https://arxiv.org/pdf/2110.01889.pdf)
- Use early stopping on era‑mean CORR on your validation split.

Deployment:

- Upload the meta‑model as your Numerai `predict` (it ingests raw features and maybe baked‑in tree predictions, so you have to serialize base models too or pre‑compute their logic inside the NN).  
- Or, simpler: generate ExcelFormer predictions offline, then treat them as an *additional base model* in your current LGBM ensemble.

***

### 2.2. TabPFN/TabDPT‑style “prior‑fitted” meta‑learner

TabPFN and TabDPT style models are transformers pre‑trained across many tabular datasets and then used in a “few‑shot” / in‑context fashion where the whole dataset is fed as context. You likely can’t replicate their pretraining, but you *can borrow the idea*: [arxiv](http://arxiv.org/pdf/2410.18164.pdf)

- Sample sub‑tables (e.g. 5–10k rows × 100 features) from Numerai across eras, treat each as a “dataset instance”.
- Train a small Transformer that ingests:
  - A block of context rows (features + tree predictions + era id).
  - A query row.
  - Learns to predict the target of the query given the context.
- This encourages the model to learn **meta‑patterns across eras and feature configurations**, rather than memorizing individual ids.

Once trained, you can:

- Run it in a simpler “row-wise” mode: drop the context mechanism and just use the encoder as a feature extractor on top of (features, tree preds), then stack a small head for Numerai target.
- Or use a distilled version as in 2.1.

This is more work, but if you want a truly novel model that plausibly has different inductive biases than trees (thus better MMC/uniqueness), this is one of the few research‑backed paths.

***

## 3. Architecture ideas directly inspired by “why trees beat DL”

The Grinsztajn et al. and Shwartz‑Ziv papers isolate **three properties** that give trees an edge: robustness to uninformative features, rotational non‑invariance, and ability to represent irregular, piecewise functions cheaply. That suggests concrete hybrid tricks: [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S1566253521002360)

### 3.1. Tree‑gated feature selection into deep models

- Use your best LGBM (per target) to compute:
  - Global feature importance (split gain).
  - Era‑local importance (per‑era SHAP or gain).
- Build a small NN/transformer that:
  - Sees raw features *and* a mask/score vector from trees, and
  - Applies **learnable feature gates** (per‑feature scalars in ) that are initialized from tree importance and regularized toward sparsity. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/55857162/fde02c25-f90d-478d-a693-e385b08f1566/Scoring-Meta-Model-Contribution.txt)
- The NN then only has to reason about a compressed, tree‑filtered feature space, which addresses the “robustness to uninformative features” critique in the DL‑vs‑trees paper. [arxiv](http://arxiv.org/pdf/2207.08815.pdf)

Effectively, this is: “let trees choose what to care about, let deep model learn richer interactions among those.”

### 3.2. Neural Oblivious Decision Ensembles (NODE) as a numerai‑tailored tree/NN hybrid

NODE is an architecture that *neuralizes* oblivious decision trees and ensembles them inside a differentiable network, designed specifically for tabular data. It gives you: [arxiv](https://arxiv.org/pdf/1909.06312.pdf)

- Tree‑like behavior (oblivious, axis‑aligned splits) with full differentiability.
- Better compatibility with custom losses.

Given Numerai’s piecewise target and feature binning, NODE‑like modules are a natural fit:

- Train a NODE model either on raw features or on raw features + your tree ensemble predictions.
- Use a custom loss approximating numerai‑corr (e.g. rank‑based loss or pairwise loss) and/or Sharpe.

This is more niche than mainstream TabTransformer, but aligned with what you want: cutting‑edge yet still tree‑ish.

***

## 4. Numerai‑specific: objective shaping and meta‑learning

Regardless of architecture, two high‑leverage ideas that are *very* Numerai‑specific:

### 4.1. Custom objectives approximating Numerai CORR / payout

Almost everyone optimizes MSE on target bins and then measures Numerai corr ex‑post. You can push closer to “optimize what you care about”:

- Use **pairwise ranking loss** (LambdaRank/RankNet‑like) on pairs of stocks within an era, which is more aligned with Spearman/numeraicorr than MSE.
- Approximate numerai‑corr gradient: within an era, your objective is corr(\(s\), \(t\)), whose derivative wrt \(s\) can be written in closed form if you fix ranking transformation; you can approximate it in a custom LightGBM objective.
- Add a penalty term for high feature exposure or high correlation with benchmark predictions, turning FNC/MMC into soft constraints.

This makes both LGBM and DL more aligned with the live scoring function.

### 4.2. Era‑adaptive stacking / meta‑weights

Meta‑learning findings on tabular data suggest that *how much you optimize hyperparameters and ensemble weights* matters as much as the base model family. You can: [cair.org](https://www.cair.org.za/sites/default/files/2024-04/55-sacair23_1.pdf)

- Train a **meta‑weighting model** \(w(e)\) that, given simple stats for each base model and the current era, outputs weights for combining them:
  - Inputs: rolling CORR, rolling Sharpe per model over past N eras, recent feature exposure summaries, era cluster ID (from clustering).
  - Output: simplex over base models.
- Enforce walk‑forward: weights for era \(E\) are trained only using data up to \(E-1\).

This should be relatively cheap to add on top of your current ensemble and may give you regime‑aware performance bumps without touching base models.

***

## 5. Where to start (concrete suggestion for you)

Given your current notebook (small feature set, strong multi‑target LGBM ensemble, Numerai metrics utils), a realistic high‑ROI path would be:

1. **Serious Bayesian HPO for LGBM** on your top 3–5 targets with an era‑Sharpe objective (this is “easy wins”). [arxiv](https://arxiv.org/abs/2305.17094)
2. **Build a simple tabular transformer meta‑model** that ingests (features + your per‑target LGBM predictions) and predicts the main target, trained under your existing era‑aware CV; treat it as an extra model in your ensemble first. [arxiv](https://arxiv.org/pdf/2401.15238.pdf)
3. **Experiment with a custom tree objective**: pairwise ranking inside eras, or a loss that upweights extreme bins and approximates numerai‑corr. Use this on just one or two target models to gauge uplift.