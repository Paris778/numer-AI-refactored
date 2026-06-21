You’re right to treat the docs as baseline; below are ideas that explicitly go *beyond* tree-based LGBM ensembles, but still respect Numerai’s constraints.

## Non-tree multi-target “super-models”

Instead of more trees, treat your current LGBM ensemble as a *feature generator* for a higher‑capacity model:

- Train a multi-output neural net (MLP or tabular transformer) that predicts all 20D and 60D targets jointly from features *and* from your existing LGBM predictions as inputs; this gives you shared representations across targets plus a learned non-linear combiner of your trees. [forum.numer](https://forum.numer.ai/t/neural-nets-are-all-you-need-really/1064)
- Distill your best tree ensemble into this NN (teacher–student): first fit NN to mimic tree predictions, then fine‑tune it on targets with a loss that’s a weighted sum of “match trees” and “improve CORR/MMC vs target”. [forum.numer](https://forum.numer.ai/t/neural-nets-are-all-you-need-really/1064)
- In production, the NN becomes the single uploaded model; trees stay offline as part of the pretraining regime.

## Direct MMC / uniqueness-driven residual models

Use Numerai’s meta/benchmark predictions as *inputs* and explicitly train for uniqueness:

- Download meta-model and benchmark predictions and build a *residual model*: predict \(r = t - \alpha \hat{m}\), where \(\hat{m}\) is meta or benchmark, and \(\alpha\) is fit per-era (or globally) so that the residual is orthogonal on average; then submit \( \hat{r} + \alpha \hat{m} \) after re-ranking. [docs.numer](https://docs.numer.ai/numerai-tournament/scoring/definitions)
- Train a model on the residual target with a loss that combines standard numerai-corr with a penalty on correlation to \(\hat{m}\) (or to benchmark ensemble), effectively turning MMC/BMC into a differentiable regularizer.  
- You can push this further with *adversarial orthogonalization*: a small discriminator tries to reconstruct meta-model predictions from your predictions, and your main model is trained to make that discriminator fail, increasing MMC while preserving CORR.

## Era-conditional mixture-of-experts

Instead of one global model (even an ensemble), build era-conditional experts:

- Cluster eras using unsupervised learning on era-level statistics (e.g., mean/variance of each feature, or a low-dimensional embedding from an autoencoder trained on feature distributions per era) to identify latent regimes. [forum.numer](https://forum.numer.ai/t/numerai-self-supervised-learning-data-augmentation-projects/5003)
- Train separate models (tree or NN) per regime cluster plus one global model; at prediction time, a lightweight “gating” function maps each live era into a mixture over regime clusters and blends the experts’ predictions before ranking.  
- You can enforce robustness by requiring that each expert is trained only on eras from *earlier* time than the eras it predicts (i.e., cluster then apply walk-forward inside each cluster).

## Self-supervised representation learning on features

Leverage the massive unlabeled structure with self-supervised objectives, then plug into your existing LGBMs:

- Train a denoising autoencoder or contrastive model on the feature matrix only (no targets), per era or globally, to learn a low-dimensional latent; use those latents as additional features to your tree ensemble or NN. [forum.numer](https://forum.numer.ai/t/numerai-self-supervised-learning-data-augmentation-projects/5003)
- Design augmentations that respect bin structure: random masking/dropout of features, small jitter within bins, or shuffling features within a feature-group (intelligence/serenity/etc.) while keeping era and target aligned.  
- A more radical option is to train a conditional generative model \(p(\text{features} \mid \text{target bin})\) and use its learned internal representation or its log-likelihood / “surprisal” as meta-features feeding your downstream model. [forum.numer](https://forum.numer.ai/t/numerai-self-supervised-learning-data-augmentation-projects/5003)

## Era-invariance and adversarial robustness

Try to *remove* the model’s ability to overfit specific eras while keeping signal on targets:

- Add an adversarial head that predicts era from your model’s penultimate representation; train the main model to predict target, while the adversary is trained to predict era, with a gradient reversal layer so the representation becomes era-invariant. [forum.numer](https://forum.numer.ai/t/numerai-self-supervised-learning-data-augmentation-projects/5003)
- Combine this with group-wise neutralization: periodically neutralize your predictions (or intermediate representation) to high-variance features or specific feature groups, and include a penalty on post-neutralization drop in CORR in your loss.  
- The goal is a model whose raw predictions are already “low-exposure / era-stable”, so your final neutralization step is gentle and doesn’t nuke most of the edge.

## Novel stacking and meta-learning over your LGBMs

Given you already have several LGBM ensembles, treat them as diverse base learners and get more sophisticated with the meta layer:

- Rather than static rank-averaging, train a *meta-learner* (small NN or ridge/elastic net) on validation that takes as input: base-model predictions, era-level diagnostics (volatility of each base model, feature-exposure summaries, recent rolling CORR), and outputs meta-weights per era.  
- To avoid overfitting validation, learn meta-weights via nested walk-forward: for each validation era, fit meta-weights only on prior eras, then score out-of-sample. This preserves time structure while still letting you adapt weights across regimes.  
- Add constraints/penalties on meta-weights such that the resulting meta-prediction has bounded correlation with benchmark/meta-model, explicitly trading off raw CORR vs uniqueness.

## Multi-objective loss approximating payout

Move away from “maximize CORR only” and bake a RAPS-like proxy into the objective:

- On each mini-batch of eras, compute approximate per-era numerai-corr, then compute batch Sharpe and a differentiable drawdown proxy (e.g., squared negative changes in cumulative CORR) and use a loss like \(-\text{Sharpe} + \lambda \cdot \text{drawdown}\).  
- Optionally, add a term approximating MMC/BMC by correlating your batch predictions with precomputed meta/benchmark predictions and penalizing high correlation.  
- This is more natural with differentiable models (NN/transformer), but you can approximate some of it in tree ensembles by using custom objectives that overweight tails or specific target bins (e.g., over-reward getting extreme bins correct), which tends to help payout under Numerai’s nonlinear scoring.

***

If you tell me what you currently have (e.g., “3× LGBM target specialists + simple rank-mean on v5.2 all features, with 100% feature-neutralization”), I can pick 1–2 of these and sketch a concrete training + validation + deployment plan tailored to your stack, including how to slot it into a model-upload `predict` without blowing the Numerai compute limits.