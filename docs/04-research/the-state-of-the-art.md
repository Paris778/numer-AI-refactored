Part 1: The Bleeding Edge of Tabular Regression & The Successors to GBDTs
For nearly a decade, Gradient Boosted Decision Trees (GBDTs)—specifically XGBoost, LightGBM, and CatBoost—were the undisputed kings of structured, tabular machine learning. Deep learning models (like TabNet) repeatedly failed to beat them in terms of generalization, compute efficiency, and robust out-of-the-box performance.

This paradigm has shifted. The absolute bleeding edge of regression and tabular ML research consists of Tabular Foundation Models (TabFMs) designed for In-Context Learning (ICL), and Retrieval-Augmented Tabular Architectures.

1. The Successors to XGBoost and LightGBM
Instead of training a tree ensemble from scratch on your data, the successors are transformer-based foundation models trained on millions of synthetic datasets. At inference time, you pass your training data and your unlabelled test targets as a single sequence, and the model predicts in a single forward pass—zero-shot, with no hyperparameter tuning:

TabPFN-3 (Prior Labs, May 2026): * The latest evolution of Tabular Prior-data Fitted Networks was published as a conference paper in 2023. It introduces the TabPFN model, which is a PFN specifically designed for tabular data. • Accurate Predictions on Small Data With a Tabular Foundation Model [Hol+25] was published in Nature in 2025. It improved the TabPFN model; the improved model is often referred to as TabPFN v2. The first paper [Mül+21] mainly provided a proof of concept that PFNs can be used to per- form Bayesian inference.]. It handles datasets up to 1 million rows and 2,000 features and ranks first across standard benchmarks (like TabArena), outperforming GBDTs tuned for 8 hours in a fraction of the time.  

It introduces "Thinking Mode" (TabPFN-3-Plus), applying test-time compute scaling to tabular data (solving/searching priors dynamically at inference) to widen the gap over ensembled frameworks like AutoGluon.  

TabFM (Google Research, June 30, 2026):  

An off-the-shelf tabular foundation model that bypasses per-dataset training completely.  

It uses a hybrid attention architecture: alternating row and column attention to automatically model cross-feature dependencies, followed by row compression to keep in-context learning computation scaling linearly rather than quadratically. On the TabArena benchmark, Google's TabFM-Ensemble dominates traditional ensembled GBDTs.  

TabICLv2 (Inria SODA, February 2026):

An open-source, highly scalable alternative to TabPFN. It introduces Query-Aware Scalable Softmax and repeated feature grouping to allow fast, low-memory zero-shot regression on massive datasets.

TabR (Tabular Retrieval-Augmented Generation):

Rather than standard parametric modeling, TabR queries "neighboring" or historically similar rows during forward inference, acting as a differentiable k-NN hybrid that significantly reduces target uncertainty in complex spaces.

2. Why is this Relevant to Numerai?
The core data structure of Numerai (obfuscated, weekly tabular eras where IDs cannot be tracked across time, and the goal is strictly alpha/residual prediction) aligns perfectly with these developments. Because GBDTs have zero structural memory of feature spaces, in-context meta-learning and retrieval-augmented deep learning models can learn the spatial distribution of the eras dynamically.