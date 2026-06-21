For noisy regression, there is no single “best” model; in practice you combine noise‑robust preprocessing, loss functions, regularization, and typically tree ensembles or support‑vector methods, tuned to your noise type and data size. [digibug.ugr](https://digibug.ugr.es/bitstream/handle/10481/71715/On_the_Regressand_Noise_Problem_Model.pdf;jsessionid=8D2C284A433111C5DF4ED08868DA22D6?sequence=1)

## Key strategies (independent of model)

- Use **robust loss**: Huber, Tukey/biweight, or quantile loss reduce the influence of large residuals compared with squared loss, which is very sensitive to outliers. [jmlr](https://jmlr.org/papers/volume19/17-295/17-295.pdf)
- Regularize strongly: L2 (ridge) or elastic‑net shrink coefficients and prevent overfitting spurious fluctuations in noisy features; L1 (lasso) also performs feature selection when some features are mostly noise. [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/regularization-techniques-in-machine-learning/)
- Clean or down‑weight noisy samples: automated outlier detection, sample re‑weighting, or label‑smoothing / noise‑aware training can materially improve regression performance when target noise is high. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC8243133/)
- Use ensembling: bagging and boosting average over many base models and typically improve robustness vs a single high‑variance regressor. [arxiv](https://arxiv.org/abs/2408.10942)

## Models that empirically handle noise well

- Tree ensembles: Random Forests and Gradient Boosting / XGBoost routinely perform strongly on noisy tabular regression; studies of “regressand noise” find XGBoost among the more robust nonlinear methods, just behind SVMs and certain ELMs on many datasets. [digibug.ugr](https://digibug.ugr.es/bitstream/handle/10481/71715/On_the_Regressand_Noise_Problem_Model.pdf;jsessionid=8D2C284A433111C5DF4ED08868DA22D6?sequence=1)
- Support Vector Regression (SVR): SVR with an appropriate kernel and ε‑insensitive loss is consistently among the most robust models to noise and outliers; further variants that relax constraints via fuzzy inequalities improve robustness even more. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0960077921000916)
- Robust linear models: Ridge/elastic‑net linear regression with a robust loss (e.g., Huber) plus good feature engineering often matches or beats more complex models when signal‑to‑noise is low. [jmlr](https://jmlr.org/papers/volume19/17-295/17-295.pdf)

### Empirical robustness ranking (typical tabular settings)

| Rank (approx) | Model family                           | Notes on noise robustness |
| --- | --- | --- |
| 1   | SVR / SVM regressors                            | Very robust to label noise across noise levels. [digibug.ugr](https://digibug.ugr.es/bitstream/handle/10481/71715/On_the_Regressand_Noise_Problem_Model.pdf;jsessionid=8D2C284A433111C5DF4ED08868DA22D6?sequence=1) |
| 2   | Gradient Boosting / XGBoost                     | Strong on many noisy datasets, slightly below SVMs overall. [digibug.ugr](https://digibug.ugr.es/bitstream/handle/10481/71715/On_the_Regressand_Noise_Problem_Model.pdf;jsessionid=8D2C284A433111C5DF4ED08868DA22D6?sequence=1) |
| 3   | Random Forests and bagging ensembles            | Averaging reduces variance from noisy samples. [arxiv](https://arxiv.org/abs/2408.10942) |
| 4   | Regularized linear models (ridge, elastic net)  | Excellent when relationship is close to linear and features are well‑chosen. [jmlr](https://jmlr.org/papers/volume19/17-295/17-295.pdf) |
| 5   | Unregularized linear / basic neural nets        | Tend to overfit noise unless heavily regularized and early‑stopped. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC8243133/) |

## When noise is in features vs targets

- Noisy targets (y): Upper bounds on achievable \(R^2\) drop purely due to target variance; cleaning very noisy samples helps most once you already use relevant features, while model choice matters less beyond a point. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC8243133/)
- Noisy features (X): Regularization and feature selection are crucial; methods like MoG‑LASSO explicitly model feature noise to improve robustness in high‑dimensional settings. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9420311/)

## Practical default recipes

- Small–medium tabular dataset, moderate noise: start with gradient boosting (XGBoost/LightGBM) using Huber or quantile loss, plus early stopping and moderate L2 regularization. [digibug.ugr](https://digibug.ugr.es/bitstream/handle/10481/71715/On_the_Regressand_Noise_Problem_Model.pdf;jsessionid=8D2C284A433111C5DF4ED08868DA22D6?sequence=1)
- High‑dimensional, potentially noisy features: use elastic‑net or MoG‑LASSO‑style sparse regression, possibly followed by a tree ensemble on the selected features. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9420311/)
- Strong outliers in y: try SVR with robust kernel and tune ε and C, or a robust linear model with Huber/Tukey loss and outlier down‑weighting. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0960077921000916)

To give something concrete, what data regime are you mostly in (tabular vs images/time series, and roughly how many samples vs features)?