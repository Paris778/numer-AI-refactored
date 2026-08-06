# State-of-the-Art Deep Learning for Obfuscated, Non-Stationary Tabular Regression

## Executive Summary

Recent progress in tabular foundation models (TabFMs), in-context learning (ICL), and retrieval-augmented tabular deep learning offers realistic, empirically validated successors to tuned gradient-boosted tree ensembles for difficult tabular problems. TabPFN-3, TabICLv2, and Google’s TabFM form the current SOTA frontier, consistently outperforming or matching strong GBDT baselines on broad benchmarks such as TabArena and TALENT under minimal or zero tuning.[^1][^2][^3][^4][^5]
For retrieval-augmented methods, TabR demonstrates that integrating a k-NN-style retrieval module into neural tabular models is both efficient and competitive with, and sometimes superior to, tree ensembles on large-scale datasets.[^6]

For an obfuscated, high-dimensional, non-stationary financial-like setting with temporal eras, no entity tracking, and tail-focused rank-correlation metrics, these models can be adapted but still need careful engineering: relying on synthetic priors alone is risky, and cross-era validation plus explicit feature-neutralization and meta-model contribution objectives are required to target FNC/MMC-like metrics rather than plain MSE.[^7][^8]
A hybrid benchmark blueprint combining TabPFN-3, TabICLv2, Google TabFM, and a customized TabR—with era-purged combinatorial cross-validation and orthogonalization-aware losses—provides a realistic experimental path to test deep-learning successors to tree-based boosters on the described dataset.[^9][^10]

***

## 1. Problem Setting and Constraints

### 1.1 Data characteristics

The target application resembles Numerai-style obfuscated equity data: features and targets are heavily transformed to remove semantic meaning while preserving rank and ordinal structure; rows are grouped into “eras” representing temporal cross-sectional snapshots, and entity IDs are randomized each era, preventing tracking of specific entities through time.[^11]
Features are numerous (over 2,000 dense, mostly continuous or binned columns), and the objective is to predict residualized continuous targets that are later evaluated via a tail-heavy rank correlation metric computed on Gaussianized or power-transformed predictions.[^12][^13]

Because evaluation involves correlation of neutralized predictions (FNC) and orthogonal contribution relative to a meta-model (MMC/BMC), standard regression losses like MSE or MAE are misaligned with the production objective. Models must therefore be robust to non-stationarity across eras, avoid over-reliance on linear feature exposure, and produce orthogonal residual signals that survive feature and meta-model neutralization.[^8][^7]

### 1.2 Consequences for model design

Several structural implications arise:

- **Obfuscation and feature semantics**: Foundation models that rely on textual column names or cross-modal semantics bring little advantage; architectures that treat columns as anonymous numerical or categorical features (TabPFN, TabICL, TabFM, TabR, FT-Transformer-style models) are better aligned.[^14][^5]
- **Era-based non-stationarity**: Training and validation must respect temporal structure through purged and embargoed cross-validation schemes to avoid leakage from overlapping forward-return labels and autocorrelation.[^15][^10]
- **Tail and rank-based metrics**: Loss functions must emphasize rank correlation and tail dependence, for example by using Gaussianization of predictions during training, differentiable approximations to Spearman correlation, or asymmetric weighting of large residuals.[^13]
- **Neutralization constraints**: Models should be trained with explicit orthogonalization objectives to features and/or a baseline meta-model, aligning training with FNC/MMC-style evaluation and discouraging simple feature exposure.[^7][^8]

These constraints strongly favor models that can: (1) leverage synthetic priors to learn generic tabular inductive biases, (2) scale in-context computation (TabFMs and ICL), and (3) incorporate retrieval of similar historical patterns without re-training (TabR-style architectures).

A key structural argument for relevance: GBDTs have no memory of the feature space across eras, whereas in-context and retrieval-augmented learners can dynamically adapt to the spatial distribution of each era — precisely the property that makes tabular foundation models and TabR-style retrieval a natural fit for era-structured, ID-anonymized data such as Numerai's.

***

## 2. Literature Taxonomy: Tabular ML 2023–2026

### 2.1 High-level surveys and benchmarks

Two recent surveys provide a comprehensive overview of deep learning for tabular data and tabular representation learning, concluding that until recently GBDTs still dominated supervised tabular tasks, but foundation models and retrieval-augmented approaches are starting to change that picture. The deep tabular learning survey categorizes methods into data transformations, specialized architectures (TabNet, SAINT, FT-Transformer, NODE, etc.), and regularization-based models; empirically, tuned tree ensembles retain an advantage on many classic tabular benchmarks, though techniques like SAINT and TabNet close the gap in some regimes.[^3][^4]
The representation-learning survey emphasizes the emerging category of **general models** or tabular foundation models that can be applied zero-shot or with minimal tuning across heterogeneous datasets, highlighting TabPFN, TabICL, and related models as key exemplars.[^4]

Recent benchmark initiatives—TabArena, TALENT, and various NeurIPS/ICLR tabular benchmark papers—report that TabPFN-style and TabICL-style foundation models, and more recently TabICLv2 and TabPFN-3, can outperform heavily tuned XGBoost/LightGBM/CatBoost in average Elo score across dozens of datasets, both classification and regression. This marks a notable shift from earlier “why do tree-based models still win?” studies where deep models lagged behind.[^1][^3][^4]

### 2.2 Taxonomy of relevant model families

For the specific problem, the relevant models fall into four main families:

1. **Gradient-boosted trees and ensembles (baseline)** – XGBoost, LightGBM, CatBoost, tuned with standard tricks (cross-validation, feature subsampling, monotonic constraints, etc.). These remain strong baselines.[^3]
2. **Tabular foundation models & ICL** – TabPFN and TabPFN-3, TabICL and TabICLv2, Google’s TabFM, and related PFN-style or ICL-style models that treat a dataset as a context for an in-context learner.[^2][^5][^1]
3. **Retrieval-augmented tabular models** – TabR and follow-up work where neural tabular encoders are augmented with k-NN-style retrieval from training data, including scalable ICL with retrieval.[^16][^6]
4. **General deep tabular architectures** – FT-Transformer, SAINT, TabNet, NODE, MLP-Mixer variants, and MambaTab; these are less tailored to foundation or retrieval paradigms but provide practical building blocks and baselines.[^17][^3]

The remainder of the report focuses on vectors A–D, analyzing TabFMs and ICL, retrieval augmentation, feature-neutralization-aware objectives, and cross-era validation.

***

## 3. Vector A: Tabular Foundation Models and In-Context Learning

### 3.1 TabPFN and TabPFN-3

**Architecture and prior**. TabPFN models tabular prediction as Bayesian inference approximated by a transformer trained on synthetic datasets generated from structural causal models or Bayesian neural networks. During pre-training, the model learns to predict masked target values given training examples and labels, effectively learning a generic learning algorithm that can be executed via a forward pass at inference. TabPFN alternates attention across rows and columns, enabling it to model feature interactions and sample-wise relations in a permutation-invariant manner.[^14]

TabPFN-2.5 scales this approach to approximately 50,000 rows and 2,000 features per table using optimized architecture variants and priors; TabPFN-Wide and related extensions increase feature limits further. TabPFN-3 introduces a redesigned architecture with row compression, improved decoder modules, and inference optimizations such as row chunking and reduced KV caches, enabling support for up to 1 million training rows and about 200 features or 100,000 rows with 2,000 features in a single model configuration.[^1][^14]

**Performance and scaling**. On the TabArena benchmark and internal datasets, TabPFN-3 outperforms tuned and ensembled GBDTs, achieving a 93% win rate over classic ML baselines and Pareto-dominating the speed/accuracy frontier in the reported ranges. It supports new capabilities such as many-class classification and time-series forecasting (TabPFN-TS-3), and its “Thinking Mode” or test-time compute scaling allows multiple stochastic forward passes or extended internal computation to improve accuracy with additional inference-time cost.[^18][^1]

**Inference-time scaling (“Thinking Mode” and Scaling Mode)**. Prior Labs introduced “Scaling Mode” for TabPFN-2.5 and folded related ideas into TabPFN-3, enabling inference on datasets with millions of rows by combining row chunking, KV cache reuse, and memory-efficient attention; inference remains a single forward pass, but the effective context is partitioned. “Thinking Mode” (TabPFN-3-Plus) further increases test-time compute by performing multiple internal reasoning steps or ensembles, reportedly yielding +420 Elo on large TabArena subsets versus standard baselines while remaining significantly faster than AutoML methods.[^14][^1]

Mathematically, TabPFN’s complexity is roughly quadratic in the number of rows within a chunk and linear in the number of chunks, with row-compression and attention tricks reducing the constant factors. This implies that for very large cross-sections, one must choose between reduced per-step context (smaller chunks) and increased compute (more chunks and passes), making row-chunking and KV caching essential for scalability.[^1][^14]

### 3.2 TabICL and TabICLv2

**TabICL** is an open-source tabular foundation model designed to perform ICL on large tabular datasets, combining column-level and row-level transformers with a dataset-level ICL transformer; it treats the entire dataset (train + test rows) as a single prompt, enabling zero-shot prediction without dataset-specific training.[^19][^20]

**TabICLv2** significantly advances this design through three pillars: a more diverse synthetic data generation engine, architectural innovations such as repeated feature grouping and query-aware scalable softmax (QASSMax), and improved pre-training protocols using the Muon optimizer. The model alternates column-wise and row-wise attention, compresses rows into embeddings, and then applies an ICL transformer at the dataset level, similar in spirit to TabFM and TabPFN but with open weights and modest pre-training cost.[^5][^21]

On TabArena and TALENT, TabICLv2 without any tuning surpasses RealTabPFN-2.5, which itself is an ensembled, real-data-fine-tuned variant of TabPFN v2.5, and is markedly faster, especially on CPUs. TabICLv2 generalizes effectively to million-scale datasets under 50 GB GPU memory through disk offloading and attention optimizations, making it attractive for local experimentation.[^22][^23][^5]

### 3.3 Google TabFM (TimesFM-style foundation for tables)

TabFM extends the TimesFM logic to tabular data, providing a zero-shot foundation model for tabular classification and regression. It uses alternating row and column attention for deep contextualization of the table, followed by row compression and a Transformer that performs ICL across the compressed row embeddings, synthesizing mechanisms from TabPFN and TabICL into a hybrid architecture.[^2]

TabFM is trained entirely on large-scale synthetic tables generated via structural causal models with varied random functions, similar in philosophy to TabPFN and TabICL priors. Benchmarks on TabArena (38 classification and 13 regression datasets) show that TabFM and an ensemble variant (TabFM-Ensemble) achieve top-10 Elo rankings and outperform heavily tuned classical baselines without dataset-specific training.[^2]

Because TabFM is deeply integrated into Google’s BigQuery (via AI.PREDICT), it provides an operationally convenient, managed TabFM that can be invoked on arbitrary tabular data with minimal setup, albeit without the low-level control available with TabPFN and TabICLv2.[^2]

### 3.4 Extrapolation to residualized financial manifolds

A core question is how TabFMs pre-trained on synthetic priors extrapolate to highly residualized, obfuscated financial data where the targets resemble noisy residuals rather than primary structural signals.  

Empirical evidence from TabPFN’s applications to time series forecasting, healthcare, and small-sample domains suggests that synthetic SCM-based priors can generalize surprisingly well to heterogeneous real-world tasks, especially when datasets share generic tabular structures like noisy nonlinear functions of many variables. However, recent surveys emphasize that transfer learning from upstream tabular data is most effective when the pre-training distributions partially match downstream tasks; fully synthetic training can lead to under- or over-regularization in specialized domains if prior diversity is insufficient.[^17][^4][^14][^1]

TabICLv2’s synthetic engine intentionally increases prior diversity using random Cauchy graphs, multiple random function families (MLPs, tree ensembles, Gaussian processes, etc.), and data filtering via ExtraTrees-based improvability checks; this is aimed at improving robustness to complex real-world manifolds such as finance. Google TabFM similarly relies on large-scale SCM-based synthetic tables and reports strong performance on diverse benchmarks, suggesting decent extrapolative generalization.[^24][^5][^2]

For residualized financial targets like Numerai’s, there is limited public evidence directly comparing TabPFN-3/TabICLv2/TabFM to bespoke GBDTs on obfuscated hedge-fund-style datasets, but Numerai competitors and blog posts report using earlier TabPFN versions as feature generators or meta-model components with competitive results. Overall, extrapolation is plausible but not guaranteed; careful validation with purged/embargoed era splits is required.[^25][^11]

### 3.5 Inference scaling and row chunking strategies

TabPFN-3 and TabICLv2 both rely on row chunking plus KV cache optimizations to handle large tables. The core idea is to process subsets of rows at a time, reusing cached key/value tensors when possible and compressing information to maintain a form of global context.[^5][^1]
For TabPFN-3, Prior Labs reports that a reduced KV cache and row-chunking scale to 1M rows on a single H100 with fast inference, and SHAP computation can be accelerated up to 120× using KV caching. TabICLv2 uses disk offloading and selective Q/K/V projection computation to fit million-scale datasets within 50 GB, at the cost of increased I/O.[^5][^1]

In practice, for large cross-sectional eras (e.g., tens of thousands of rows), a pragmatic strategy is:

- Chunk rows per era into sub-blocks of size compatible with GPU memory (e.g., 5–20k rows),
- Use foundation models in ICL mode where each chunk includes a subset of training rows plus current test rows, and
- Aggregate predictions across multiple chunks or stochastic passes (Thinking Mode) to approximate a larger-context Bayes predictor.

Theoretical bounds: complexity is dominated by attention operations; for row-attention with sequence length \(n\), standard transformers incur \(O(n^2 d)\) compute and \(O(n^2)\) memory, whereas TabICLv2’s QASSMax aims to maintain expressivity with better scaling in \(n\) via log-\(n\)-aware query rescaling. Exact asymptotic bounds for “Thinking Mode” are not provided, but test-time scaling behaves similarly to ensembling: roughly linear in the number of additional passes or internal reasoning steps for a given chunk.[^5][^1]

***

## 4. Vector B: Retrieval-Augmented Tabular Regression (TabR)

### 4.1 TabR architecture and performance

TabR is a retrieval-augmented deep model for tabular data that integrates a k-NN-style module into a feed-forward network to retrieve relevant training examples for each query object. Given an input example, TabR uses an encoder to produce a representation, then retrieves nearest neighbors from a memory of training data using an attention-like mechanism that operates over embeddings and stored labels; the retrieved context is combined with the query embedding to make the final prediction.[^26][^6]

On public benchmarks with datasets up to several million objects, TabR achieves the best average performance among tabular deep learning models and sets new SOTA on several datasets, even outperforming GBDT models on a benchmark designed to favor tree methods. Importantly, TabR is reported to be simple and significantly more efficient than prior retrieval-based models, making retrieval augmentation practically viable.[^6]

### 4.2 Scaling retrieval to large cross-sectional datasets

Retrieval augmentation incurs additional memory and compute overhead proportional to the size of the retrieval index and the number of neighbors per query. TabR’s design focuses on efficient approximate k-NN retrieval in learned embedding space, using an attention-like mechanism that scales sub-quadratically with dataset size.[^6]
In practice:

- A pre-built nearest-neighbor index over training embeddings (e.g., FAISS or Annoy) allows approximate k-NN search with \(O(\log N)\) or sub-linear time per query, where \(N\) is the number of cached training examples.
- Retrieval over millions of rows is feasible with GPU-accelerated similarity search and small k (e.g., 10–100 neighbors per query).

Recent work on scalable ICL over tabular data via retrieval suggests combining foundation models with retrieval modules to select a subset of training rows to include in the ICL context, effectively turning TabR-like retrieval into a pre-filter for TabFMs rather than a full model. This is particularly relevant for obfuscated, era-based financial data, where retrieving neighbors from similar eras or similar residual target behavior may improve robustness to non-stationarity.[^16]

### 4.3 Retrieval as implicit target neutralization

Retrieving similar target vectors across historical eras can act as a form of implicit target neutralization, depending on how retrieval is structured:

- If retrieval uses only feature embeddings, the retrieved context mainly reflects feature similarity and may reinforce feature exposure rather than neutralize it.
- If retrieval also considers target history or meta-model residuals, then neighbors with similar meta-model errors can provide a localized residual manifold, helping the model focus on orthogonal deviations rather than raw target levels.

TabR as originally proposed focuses on supervised retrieval of nearest neighbors in feature space; the paper does not target feature-neutral correlation or MMC-like objectives directly. However, the architectural pattern—retrieving labeled neighbors and using attention to blend their signals—can be adapted to: (1) retrieving neighbors with similar neutralized residuals, and (2) imposing orthogonalization constraints between retrieved signals and baseline predictions in the loss (see section 5).[^6]

### 4.4 Combining TabR with TabFMs and ICL

Recent work on scalable ICL via retrieval shows that retrieval can be used to select training examples for inclusion in an ICL context, thereby extending the effective context length of TabFMs without quadratic attention over all rows. A hybrid system can operate as follows:[^16]

- A retrieval index is built over row embeddings produced by a base model (e.g., a shallow MLP or early layers of a TabFM/TabICLv2 encoder).
- For each test row, a small set of relevant training rows is retrieved from the index.
- The TabFM or TabPFN-3 is invoked in ICL mode on this subset plus the test row, acting as a powerful local learner.

This approach effectively marries TabR’s retrieval module with TabPFN/TabICLv2’s in-context learning, enabling a model to adapt at inference time to specific regions of the feature manifold without retraining. The combination is particularly promising in non-stationary settings where different eras correspond to different local regimes; retrieval within relevant eras may approximate regime-specific local models without leaking information across time if retrieval is restricted to past eras only.

***

## 5. Vector C: Feature-Neutralization and Orthogonal Contribution

### 5.1 FNC and MMC in Numerai-style settings

Feature Neutral Correlation (FNC) measures the correlation of a model’s predictions with the target after neutralizing predictions with respect to the feature matrix; it is computed by first normalizing the predictions, then regressing them onto features in the current round, and finally correlating the residuals with the target. Meta-Model Contribution (MMC) measures the covariance of a model’s neutralized predictions with the target after neutralizing against a meta-model’s predictions rather than raw features.[^27][^8][^7]

In both cases, neutralization can be formalized through linear projection: given predictions \(p\) and a design matrix \(X\) (features or meta-model predictions), neutralized predictions are \(p_{\perp X} = p - X(X^\top X)^{-1} X^\top p\), where \((X^\top X)^{-1} X^\top\) is the Moore–Penrose pseudoinverse-based projection operator; FNC/MMC then evaluate correlation or covariance between \(p_{\perp X}\) and the target.[^8][^7]

### 5.2 Loss formulations targeting post-orthogonalized covariance

Directly optimizing for FNC/MMC-like metrics requires differentiating through the neutralization step. Conceptually, a loss for FNC-style optimization could be:

\[ \mathcal{L}_{\mathrm{FNC}} = -\mathrm{corr}(p_{\perp X}, y), \quad p_{\perp X} = p - X(X^\top X + \lambda I)^{-1} X^\top p, \]

together with standard regularizers on predictions and weights. While there is limited published work explicitly targeting FNC/MMC in deep models, Numerai-related blogs and implementations commonly approximate FNC-like behavior by:

- Computing batch-wise neutralized predictions using ridge regression (for numerical stability), and
- Backpropagating through the neutralization step using automatic differentiation frameworks.[^12][^25]

This is feasible for mini-batches where \(X\) has manageable dimension (e.g., 50–200 neutralizing features) and the matrix \(X^\top X\) is well-conditioned. For thousands of features, low-rank approximations or feature sub-sampling are required.

Rank-based objectives for MMC-like metrics can be expressed using differentiable approximations to Spearman correlation (e.g., soft ranking) or by computing correlation on Gaussianized predictions, as in FNCv4 where predictions are ranked, Gaussianized, neutralized, and then correlated with a factor-neutral target. Combining differentiable neutralization and differentiable rank correlation yields a fully end-to-end loss that matches production scoring.[^13]

### 5.3 Differentiable pseudo-inverse layers

The Moore–Penrose pseudoinverse is differentiable almost everywhere with respect to its inputs when singular values are non-zero, and deep learning frameworks support automatic differentiation through matrix inverse and solve operations. Although there is no widely deployed standard “pseudo-inverse layer” in deep tabular architectures, several building blocks exist:[^28]

- Implementing neutralization as a differentiable linear layer using `torch.linalg.lstsq` or `torch.linalg.solve` with Tikhonov regularization for \(X^\top X + \lambda I\) inversion.
- Using custom autograd functions that compute \(X^+\) via SVD and propagate gradients analytically, though this is more expensive.

Research on differentiable least-squares and implicit layers (e.g., deep equilibrium models) shows that solving linear systems inside the forward pass is tractable at moderate sizes and can be used as a differentiable layer. For FNC/MMC, a practical approach is to implement a “neutralization layer” that accepts a batch of predictions and a set of neutralizer features (or meta-model predictions), computes the ridge-regression projection, and outputs residuals; this layer is then used both during training (for loss computation) and optionally at inference.[^3]

Given the high dimensionality (2,000+ features), direct batch-wise pseudo-inverse is expensive. Therefore, in practice:

- Neutralization is performed against a subset of features (e.g., top principal components, known “risk factors”, or a compressed representation produced by a fixed random projection or autoencoder).
- Alternatively, neutralization is done against a fixed baseline model (e.g., a GBDT meta-model), which has low-dimensional outputs and hence a cheap pseudo-inverse.

### 5.4 Orthogonalization with respect to a meta-model

Orthogonal contribution metrics like MMC/BMC neutralize predictions relative to a meta-model’s predictions rather than features. This is more tractable as the neutralizer matrix has dimensions \(N \times K\) where \(K\) is the number of baseline models, often in the tens.[^11][^8]

In a deep-learning context, an MMC-style loss can be written as:

\[ \mathcal{L}_{\mathrm{MMC}} = -\mathrm{cov}(p_{\perp m}, y), \quad p_{\perp m} = p - m(m^\top m + \lambda I)^{-1} m^\top p, \]

with optional penalties on raw correlation and feature exposure. Implementationally, this requires computing neutralized predictions for each batch and backpropagating through the small linear solve for \(m\). There is no explicit literature on MMC-optimized deep tabular models, but the building blocks (covariance-based losses, neutralization layers) are standard.

***

## 6. Vector D: Cross-Era Generalization and Temporal Leakage

### 6.1 Purged and embargoed cross-validation

Standard k-fold CV fails for time-series or overlapping-label financial data because training and validation sets share overlapping label windows, allowing information leakage. Marcos López de Prado’s “purged k-fold CV” removes any training observations whose label windows overlap with the test set and adds an embargo period after the test set to further avoid leakage from autocorrelation.[^10][^15]

Combinatorial Purged Cross-Validation (CPCV) generalizes this idea by forming multiple train/test splits from ordered blocks and applying purging and embargoing to each split; this yields more robust performance estimates in presence of regime shifts and overlapping labels. Several open-source implementations (e.g., Mizar Labs, GitHub repos) provide CPCV routines for financial ML pipelines.[^28][^9][^10]

For era-based tabular data analogous to Numerai’s, eras already provide block structure; a CPCV scheme can treat each era (or chunk of eras) as a block and create train/test folds that ensure all test eras post-date train eras, with purging of eras whose forward-looking targets overlap the test windows. This is particularly important for foundation and retrieval models whose high capacity amplifies leakage if splits are naive.

### 6.2 Deep tabular architectures with era-aware or adversarial losses

There is limited direct literature on deep tabular models with explicit adversarial era-alignment for financial-style obfuscated eras. However, related work in domain adaptation and distribution shift for tabular data proposes:

- **Adversarial domain confusion** where a discriminator is trained to predict era or domain labels from intermediate representations, and the main encoder is trained to minimize task loss and maximize domain confusion (via gradient reversal), encouraging era-invariant representations.[^4][^3]
- **Conditional domain adaptation** where representations are aligned conditional on labels or predicted targets, controlling for label shift.

In the tabular representation-learning survey, domain adaptation and open-environment tabular ML are highlighted as emerging topics, with methods using domain adversarial nets and importance weighting across domains. These techniques can be applied to era labels: each era is treated as a domain, and the model is trained to produce representations from which era is hard to predict while still predicting targets well.[^4]

For obfuscated financial eras, practical strategies include:

- Adding an adversarial head on top of shared feature encoders (including TabR encoders or pre-TabFM embeddings) to predict era indices; use gradient reversal to encourage era-invariance.
- Using contrastive losses that group examples by similar residual behavior across eras, encouraging representation similarity for cross-era patterns.
- Explicitly penalizing performance gaps between eras in validation, e.g., using stability metrics that measure variability of correlation across eras.

### 6.3 Avoiding regime-specific overfitting

Modern deep tabular architectures are powerful enough to memorize regime-specific feature–target mappings. Without careful CV and regularization, they can overfit particular eras, especially when labels are residualized but still exhibit non-stationary regimes.

Key mitigation measures include:

- **Era-based CPCV**: ensuring validation eras are disjoint and strictly later than training eras, minimizing temporal leakage.[^15][^10]
- **Regularization via priors**: using TabPFN/TabICL priors that have seen broad synthetic distributions and thus are biased towards generic functional forms rather than idiosyncratic patterns.[^1][^5]
- **Adversarial era alignment**: as discussed above, forcing embeddings to be era-agnostic while still predictive.
- **Retrieval restricted to past eras**: TabR-style models must only retrieve neighbors from historical eras to avoid peeking into the future; retrieving era-local neighbors from past eras approximates regime-specific local modeling without leakage.

***

## 7. Theoretical Analysis: Rank-Based Tail Correlation with Feature Constraints

### 7.1 Rank correlation and Gaussianization

The evaluation metric described is a tail-heavy rank correlation computed on Gaussianized, power-transformed predictions. Numerai’s later FNC and FNCv4 definitions involve ranking predictions, applying Gaussianization (mapping ranks to quantiles of a standard normal), neutralizing to features, and then computing rank correlation with a factor-neutral target.[^7][^13]
This process creates a distributionally robust metric that is insensitive to monotonic transformations of predictions but sensitive to correct ordering, especially in the tails due to Gaussianization or power transforms emphasizing extreme values.

From a learning perspective, optimizing Spearman correlation or related rank metrics directly is challenging because ranking is non-differentiable. Common approximations include:

- Using soft-rank operators to approximate ranks with differentiable functions.
- Using pairwise ranking losses like ListNet or LambdaRank, which approximate rank-based objectives with pairwise comparisons.
- Gaussianizing predictions during training and computing differentiable approximations of covariance or correlation with the target.

When combined with neutralization, the objective becomes:

\[ \mathcal{L} = -\mathrm{corr}(g(p)_{\perp X}, g(y)), \]

where \(g\) denotes Gaussianization or a power transform, and \(p_{\perp X}\) is neutralized predictions. The pseudo-inverse and Gaussianization steps are both differentiable in practice (using smoothed approximations), enabling gradient-based optimization.

### 7.2 Tail-heavy objectives and risk control

Tail-heavy correlation metrics focus on getting the ordering of extreme predictions correct, akin to optimizing a risk-adjusted Sharpe-like metric rather than raw MSE. In finance, similar objectives appear in maximizing the correlation between strategy returns and target returns while penalizing variance or drawdowns.[^25]
Using Gaussianized predictions effectively downweights central regions and upweights tails, which encourages models to produce confident predictions only where they have strong evidence, aligning with risk management goals.

In the described setting, a composite loss could blend:

- A standard regression loss (MSE/Huber) to stabilize training,
- A rank-based, neutralized correlation loss to target the evaluation metric, and
- Regularizers on exposure to features or meta-model predictions to encourage orthogonality.

### 7.3 Orthogonality constraints and pseudo-inverse layers

Orthogonality to features or meta-model predictions is naturally expressed through pseudo-inverse-based projections (as discussed in section 5). Implementing this as a differentiable layer ensures that the model’s gradient pushes predictions into subspaces orthogonal to specified vectors, effectively learning residual signals.  
The combination of rank-based tail correlation and orthogonality yields an objective where the model is rewarded for capturing idiosyncratic, non-linear, regime-robust signals that are not explained by raw features or existing models.

***

## 8. Benchmark Blueprint: Experimental Design

### 8.1 Datasets, splits, and baselines

**Datasets**. The primary dataset is the obfuscated, era-based, high-dimensional table described in the problem; if multiple vintages exist (e.g., changing feature sets over time), experiments should segment by feature schema to maintain stationarity.

**Splits**. Use an era-based CPCV scheme with purging and embargoing:

- Treat each era as a block; define label horizon and embargo length according to target construction (e.g., about 4–5 weeks forward returns).[^10][^15]
- Generate multiple train/test folds where train eras always precede test eras and any eras with overlapping label windows are removed from train.
- Reserve the latest contiguous block of eras as an out-of-sample test set, untouched during model development.

**Baselines**:

- Tuned XGBoost, LightGBM, CatBoost regressors optimized for correlation-based objectives and with post-hoc neutralization.
- Simple deep baselines: FT-Transformer, SAINT, MLP-Mixer-based models, tuned with MSE plus rank-based auxiliary losses.

Evaluation metrics: raw correlation, FNC-style feature-neutral correlation, MMC/BMC-style meta-model contribution against a chosen baseline model, and stability metrics (variance of per-era correlation and FNC).[^8][^11][^7]

### 8.2 TabPFN-3 and TabICLv2 setup

**TabPFN-3**:

- Use the official TabPFN-3 PyTorch implementation with regression support; configure row chunking and KV cache settings compatible with GPU limits for 2,000+ features.[^29][^1]
- For each CPCV fold, treat training eras as context and generate predictions for validation eras using ICL; consider using TabPFN-3’s time-series or relational variants if era information can be encoded as an additional feature.
- Experiment with Thinking Mode (TabPFN-3-Plus via API, if accessible) by increasing test-time compute for validation and test sets; measure marginal Elo or correlation gains versus compute cost.[^30][^1]
- Because the objective is rank-based and neutralized, explore post-hoc transformation of TabPFN predictions (Gaussianization, tail weighting) during validation to gauge their impact before attempting to incorporate such transformations into training.

**TabICLv2**:

- Deploy TabICLv2 from its open-source implementation; leverage disk offloading if GPU memory is limited.[^20][^5]
- For each CPCV fold, feed the combined train+validation eras to TabICLv2 in ICL mode, ensuring that test rows attend only to past rows or using masking to prevent future leakage.
- Evaluate zero-shot performance (no fine-tuning) as well as light fine-tuning on the training eras, if the implementation permits, with combined MSE and rank-correlation losses.

### 8.3 Google TabFM setup

- Use the TabFM model published by Google Research (Hugging Face/BigQuery integration) for regression tasks on the local dataset.[^2]
- Preprocess the obfuscated table into the format expected by TabFM, ensuring consistent handling of numeric and categorical features.
- For local experiments, run TabFM in its zero-shot mode for each CPCV fold; optionally, experiment with TabFM-Ensemble if the code or BigQuery integration exposes the SVD/cross-feature ensemble variant.[^2]
- Compare TabFM’s performance to TabPFN-3/TabICLv2 under identical splits and metrics, focusing on FNC/MMC-style post-processing rather than raw correlation alone.

### 8.4 Customized TabR networks

- Implement TabR using the open-source Yandex repository; adapt the architecture for regression and high-dimensional inputs.[^31][^6]
- Build a retrieval index over training rows per fold using embeddings from the TabR encoder; ensure retrieval is restricted to past eras only for each validation/test era.
- Experiment with three retrieval schemes:
  - Feature-based retrieval (as in original TabR).
  - Meta-model-residual retrieval, where the index is built over residuals of a baseline model (GBDT or TabPFN), encouraging retrieval of similar residual behavior.
  - Era-local retrieval, where retrieval is constrained to a small window of past eras to approximate regime-local modeling.

Loss functions should combine standard regression loss with differentiable approximations to neutralized rank correlation, using a neutralization layer that projects predictions orthogonally to a subset of features or a meta-model.[^7][^8]

### 8.5 Integrating neutralization-aware objectives

For all deep models (TabR, FT-Transformer, SAINT, and optionally fine-tuned TabPFN/TabICLv2), incorporate neutralization-aware objectives:

- Implement a differentiable neutralization layer that solves a ridge regression of predictions on neutralizer features (e.g., PCA-compressed features or baseline model predictions) and returns residuals; ensure gradients flow through the solve.[^28]
- Define a composite loss combining:
  - MSE/Huber on raw predictions,
  - Negative correlation between neutralized, Gaussianized predictions and the target, and
  - Regularization on feature or meta-model exposure.

This aligns training with FNC/MMC-style metrics and should improve performance under the production evaluation.

### 8.6 Cross-era generalization diagnostics

To assess cross-era robustness, track:

- Per-era correlation, FNC, and MMC distributions for each model.
- Stability metrics such as the standard deviation of per-era correlation and the frequency of negative FNC/MMC eras.
- Performance under synthetic regime shifts, e.g., by training on early eras and testing on late eras with known distribution changes.

Use domain-adversarial training on era labels for TabR and other deep models to encourage era-invariant representations; compare results with and without adversarial loss.[^4]

***

## 9. Practical Recommendations

1. **Baseline first with CPCV**: Implement purged and embargoed CPCV based on López de Prado’s methodology and evaluate GBDTs under FNC/MMC-like metrics to establish a robust baseline.[^15][^10]
2. **Deploy TabICLv2 and TabPFN-3 as primary TabFMs**: Use TabICLv2 locally (open, efficient), and TabPFN-3 for both local and API-based “Thinking Mode” experiments; benchmark them under the same CPCV scheme and tail-heavy metrics.[^5][^1]
3. **Add TabFM as a managed baseline**: Where access to Google’s TabFM is available, include it as an additional zero-shot TabFM baseline, especially to explore operational convenience and integration with SQL-based pipelines.[^2]
4. **Prototype TabR and retrieval-enhanced TabFMs**: Implement TabR-based architectures with retrieval constrained to past eras and experiment with retrieval guided by residuals rather than raw features.[^16][^6]
5. **Introduce differentiable neutralization layers early**: For any deep model, incorporate neutralization-aware losses targeting FNC/MMC-like metrics instead of relying solely on post-hoc neutralization; use compressed features or meta-model outputs as neutralizers to keep the pseudo-inverse tractable.[^8][^7]
6. **Apply era-aware domain adaptation**: Use adversarial era-alignment on intermediate representations and measure stability across eras, reducing regime-specific overfitting in non-stationary environments.[^4]

This blueprint should enable a systematic exploration of whether TabFMs, ICL-based Tabular models, and retrieval-augmented deep networks can deliver truly orthogonal, tail-focused alpha signals beyond tuned GBDTs on obfuscated, era-based financial data.

---

## References

1. [TabPFN-3: Technical Report](https://priorlabs.ai/technical-reports/tabpfn-3) - TabPFN-3 dramatically pushes the frontier of tabular prediction and brings substantial gains on time...

2. [A zero-shot foundation model for tabular data](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/) - TabFM brings the out-of-the-box convenience of modern foundation models directly to tabular ML workf...

3. [Deep Neural Networks and Tabular Data: A Survey](https://pubmed.ncbi.nlm.nih.gov/37015381/) - by V Borisov · 2024 · Cited by 1870 — This work provides an overview of state-of-the-art deep learni...

4. [Representation Learning for Tabular Data: A ...](https://github.com/LAMDA-Tabular/Tabular-Survey) - In this survey, we systematically introduce the field of tabular representation learning, covering t...

5. [A better, faster, scalable, and open tabular foundation model](https://arxiv.org/abs/2602.11139) - by J Qu · 2026 · Cited by 57 — With only moderate pretraining compute, TabICLv2 generalizes effectiv...

6. [Tabular Deep Learning Meets Nearest Neighbors in 2023](https://arxiv.org/abs/2307.14338) - by Y Gorishniy · 2023 · Cited by 151 — In addition to the much higher performance, TabR is simple an...

7. [Feature Neutral Correlation (FNC)](https://docs.numer.ai/numerai-tournament/scoring/feature-neutral-correlation) - Feature neutral correlation (FNC) is the correlation of a model with the target, after its predictio...

8. [Meta Model Contribution (MMC)](https://docs.numer.ai/numerai-tournament/scoring/meta-model-contribution-mmc) - Meta Model Contribution (MMC) is the covariance of a model with the target, after its predictions ha...

9. [Combinatorial Purged Cross-Validation Insights | PDF](https://www.scribd.com/document/725401650/SSRN-id4778909) - Purged K-Fold Cross- an efficient full price process S but inefficiencies in both Validation, as out...

10. [Combinatorial Purged Cross Validation - Mizar](https://docs.mizar.com/mizar/mizarlabs/model/combinatorial-purged-cross-validation)

11. [How Numerai Works: Tournament, Staking, and the Meta-Model](https://nmrdash.com/articles/how-numerai-works) - How the Numerai tournament works: obfuscated data, MMC scoring, NMR staking, and the stake-weighted ...

12. [The Numerai Metrics Cheatsheet: CORR, MMC, BMC ... - nmrdash](https://nmrdash.com/articles/numerai-metrics-cheatsheet) - Numerai has shipped six MMC versions, four FNC variants, three CORR families, and quietly renamed CO...

13. [Definitions | Numerai Docs](https://docs.numer.ai/numerai-signals/scoring/definitions)

14. [TabPFN](https://en.wikipedia.org/wiki/TabPFN) - TabPFN (Tabular Prior-data Fitted Network) is a machine learning model for tabular datasets proposed...

15. [Purged Cross-Validation — Quant Signal Glossary](https://microalphas.com/glossary/purged-cross-validation/) - Cross-validation for financial time series that removes observations adjacent to the test set from t...

16. [Scalable In-Context Learning on Tabular Data via Retrieval ...](https://arxiv.org/html/2502.03147v1) - A recent success in this domain is TabR (Gorishniy et al., 2024) , which enhances representations fo...

17. [[PDF] A Survey on Deep Tabular Learning | Semantic Scholar](https://www.semanticscholar.org/paper/A-Survey-on-Deep-Tabular-Learning-Somvanshi-Das/bcb47761db8f8dfa012c49c2b3405da515bcc0e9) - This survey reviews the evolution of deep learning models for tabular data, from early fully connect...

18. [TabPFN](https://priorlabs.ai/tabpfn) - Meet TabPFN-3. The tabular foundation model for state-of-the-art predictions on structured data. try...

19. [TabICLv2: A state-of-the-art tabular foundation model](https://github.com/soda-inria/tabicl) - TabICL is a tabular foundation model (like TabPFN). It uses in-context learning (ICL) to learn from ...

20. [An Open Tabular Foundation Model — TabICL](https://tabicl.readthedocs.io/en/latest/) - State-of-the-art accuracy — zero tuning required. TabICLv2 is competitive with heavily tuned XGBoost...

21. [A better, faster, scalable, and open tabular foundation model](https://hal.science/hal-05538427v1/file/TabICLv2_%20A%20better,%20faster,%20scalable,%20and%20open%20tabular%20foundation%20model.pdf) - by J Qu · 2026 · Cited by 57 — TabICLv2 generalizes effectively to million-scale datasets under 50GB...

22. [Excited to announce TabICLv2 — our new state-of-the-art tabular ...](https://www.linkedin.com/posts/jingang-qu-0a80a5138_excited-to-announce-tabiclv2-our-new-activity-7432019020855394304-Z-Nk) - 🚀 Excited to announce TabICLv2 — our new state-of-the-art tabular foundation model! This is the resu...

23. [A better, faster, scalable, and open tabular foundation model](https://www.semanticscholar.org/paper/TabICLv2:-A-better,-faster,-scalable,-and-open-Qu-Holzm%C3%BCller/24444704448b5f3462bea940b2d1606a4de15ea1) - TabICLv2, a new state-of-the-art foundation model for regression and classification built on three p...

24. [[论文评述] TabICLv2: A better, faster, scalable, and open tabular ...](https://www.themoonlight.io/zh/review/tabiclv2-a-better-faster-scalable-and-open-tabular-foundation-model) - TabICLv2 是一种用于回归和分类的最新表格基础模型 (Tabular Foundation Model, TFM)，其性能超越了以往的梯度提升树和现有最先进的 TabPFNv2 及其变体。该模型...

25. [Applying Sound Financial Data Processing Techniques to ...](https://fenix.tecnico.ulisboa.pt/downloadFile/844820067127785/Francisco_Venancio_Extended_Abstract.pdf)

26. [TABR: TABULAR DEEP LEARNING MEETS NEAREST NEIGHBORS IN 2023 阅读笔记](https://blog.csdn.net/nbdnbb/article/details/136229040) - 文章浏览阅读1.2k次，点赞22次，收藏25次。本文是TABR论文的阅读笔记。针对表格数据问题（例如分类、回归）的深度学习（DL）模型目前正受到研究人员越来越多的关注。TabR旨在设计检索增强模型提高...

27. [Scoring - Numerai Docs](https://docs.numer.ai/numerai-tournament/scoring)

28. [Source code for mizarlabs.model.model_selection](https://mizarlabs.readthedocs.io/en/latest/_modules/mizarlabs/model/model_selection.html)

29. [PriorLabs/TabPFN - Foundation Model for Tabular Data](https://github.com/PriorLabs/tabpfn) - TabPFN: Foundation Model for Tabular Data ⚡. TabPFN supports Python 3.10+. TabPFN-3 model: Core impl...

30. [tabpfn-v3 - Microsoft Marketplace](https://marketplace.microsoft.com/et-ee/product/saas/priorlabs.tabpfn-v3?tab=overview) - State-of-the-art Tabular Foundation Model for fast, accurate predictions on structured data.

31. [tabular-dl-tabr/README.md at main · yandex-research/tabular-dl-tabr](https://github.com/yandex-research/tabular-dl-tabr/blob/main/README.md) - The implementation of "TabR: Unlocking the Power of Retrieval-Augmented Tabular Deep Learning" - yan...

