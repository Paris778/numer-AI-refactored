**Short answer:**  
For *low‑signal*, *large*, *tabular* regression problems (like Numerai), the methods that remain genuinely competitive beyond XGBoost/LightGBM fall into **five families**: boosted trees, linear‑regularized ensembles, kernel approximations, monotonic neural nets, and hybrid/meta‑learners. Below is the distilled, practitioner‑grade list used by serious quant groups.

---

## ⭐ 1. **CatBoost** (Yandex Research)  
**Why it’s competitive:**  
- Superior handling of categorical interactions  
- Very stable on noisy, low‑signal data  
- Often beats LGBM/XGB when signal is weak and feature interactions are subtle  
- Less sensitive to hyperparameters → good for short feedback loops  

**When to use:**  
- You want a “plug‑and‑play” model that behaves well even when signal is tiny  
- You want fewer tuning degrees of freedom

---

## ⭐ 2. **HistGradientBoosting (HGB)** — scikit-learn  
**Why it’s competitive:**  
- Extremely stable, conservative tree growth  
- Often matches XGBoost accuracy with simpler tuning  
- Strong performance on *very large* tabular datasets  
- Excellent for low-signal because it avoids overfitting aggressively  

**When to use:**  
- You want deterministic, reproducible behavior  
- You want a model that rarely overfits even with weak signal

---

## ⭐ 3. **Regularized Linear Ensembles (ElasticNet + stacking)**  
**Why they matter:**  
Low-signal regimes often reward **simplicity**.  
A well-tuned ElasticNet or ridge regression can outperform complex models when the signal-to-noise ratio is extremely low.

**Competitive variants:**  
- ElasticNet with cross-validated α and l1_ratio  
- Ridge regression with target encoding + interaction features  
- Linear stacking on top of tree models  

**When to use:**  
- You want maximum stability and interpretability  
- You want a baseline that rarely collapses in noisy eras

---

## ⭐ 4. **Kernel Approximation Models (Random Fourier Features + Linear)**  
**Why they’re competitive:**  
- Approximate RBF kernels at scale  
- Capture nonlinear structure without tree-based instability  
- Surprisingly strong in low-signal settings  

**Competitive variants:**  
- RFF + Ridge  
- Nystrom + Linear  
- Fastfood transforms  

**When to use:**  
- You want smooth nonlinear modeling without tree variance  
- You want GPU-friendly linear algebra

---

## ⭐ 5. **TabNet / NODE / FT-Transformer (Deep Tabular Models)**  
Deep learning is *not* generally superior for tabular data, but **some architectures** are competitive when signal is weak but structure is complex.

### **NODE (Neural Oblivious Decision Ensembles)**  
- Gradient boosting–inspired neural architecture  
- Strong on large tabular datasets  
- More stable than MLPs

### **FT-Transformer**  
- Transformer adapted for tabular data  
- Good at capturing cross-feature interactions  
- Needs heavy regularization in low-signal regimes

### **TabNet**  
- Sparse attention  
- Works well when features have hierarchical importance  
- Less competitive than NODE/FTT in pure low-signal tasks

**When to use:**  
- You have huge data and want to capture subtle interactions  
- You can afford GPU training + heavy regularization

---

## ⭐ 6. **Probabilistic Gradient Boosting (PGBM, NGBoost, XGBoostLSS)**  
These are **state-of-the-art** for low-signal regression where uncertainty matters.

### **PGBM (Oxford + Imperial)**  
- Probabilistic LightGBM  
- Produces full predictive distributions  
- Often more stable than LGBM in noisy regimes

### **NGBoost (Stanford + Google)**  
- Natural gradient boosting  
- Very strong calibration  
- Good for low-signal because it optimizes distribution parameters directly

### **XGBoostLSS (Tübingen)**  
- Probabilistic XGBoost  
- Best-in-class for distributional regression  
- Excellent for risk-sensitive tasks (finance, insurance)

**When to use:**  
- You want calibrated predictions  
- You want robustness in noisy eras  
- You want distributional outputs for meta-modeling

---

## ⭐ 7. **Monotonic Neural Networks (Deep Lattice Networks)**  
**Why they matter:**  
- Enforce monotonicity constraints  
- Reduce variance in low-signal settings  
- Used in high-stakes ML at Google

**When to use:**  
- You know some features should have monotonic relationships  
- You want strong regularization without losing flexibility

---

## ⭐ 8. **Hybrid / Meta-Learning Approaches**  
These often outperform any single model in low-signal regimes.

### **Stacking**  
- Combine LGBM + XGB + CatBoost + linear models  
- Meta-learner often improves stability

### **Blending**  
- Weighted averages of multiple GBMs  
- Extremely stable for Numerai-like tasks

### **Feature bagging / model bagging**  
- Reduces variance  
- Helps when signal is weak and features are noisy

---

## ⭐ 9. **Sparse Models (Lasso + Feature Selection)**  
When signal is extremely weak, sparse models can outperform everything else.

**Why:**  
- They eliminate noise aggressively  
- They produce stable predictions  
- They avoid overfitting subtle patterns that aren’t real

---

## ⭐ Summary Table (practical)**

| Method | Strength | Weakness | Best Use Case |
|-------|----------|----------|---------------|
| CatBoost | Stability, low-signal performance | Slower | Noisy tabular data |
| HGB | Deterministic, stable | Less flexible | Large datasets |
| ElasticNet | Very stable | Limited nonlinear power | Extremely low signal |
| RFF + Linear | Nonlinear + stable | Needs tuning | Smooth interactions |
| NODE | Strong deep tabular | Heavy compute | Large datasets |
| FT-Transformer | Captures interactions | Needs regularization | High-dimensional |
| PGBM / NGBoost | Probabilistic SOTA | Slower | Risk-sensitive regression |
| Deep Lattice | Monotonic constraints | Hard to tune | Structured tabular |
| Stacking | Best overall stability | More complexity | Competitions, finance |