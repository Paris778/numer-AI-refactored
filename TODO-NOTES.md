# TODO 
---------------------------------------

Add numerAI MCP for agents 

Do new benchmark models (simple and trivial) 
---- Also add all the old ones (try to recreate them) - OneNote

Update libraries (lgbm etc) 

-----------------------------
| Family / method            | Where it wins                          | Caveat in low-signal tabular regression            |
|----------------------------|----------------------------------------|----------------------------------------------------|
| **Stacked GBM ensembles**  | Best overall on many tabular benchmarks| More complex, harder to maintain                   |
| **TabPFN / foundation models** | Top single models on TabArena        | Need GPU, still young ecosystem                    |
| **CatBoost (single booster)** | Often top among tree ensembles        | Slightly slower, fewer knobs than LGBM/XGB         |
| **LightGBM (single booster)** | Near‑top, best speed/scale            | Can overfit if not carefully regularized           |
| **Random forest / bagging**   | Very stable baseline                  | Usually slightly worse raw accuracy than tuned GBM |

### Direct answer

There isn’t a single universal “winner”, but if you force a ranking for **low‑signal, large tabular regression**:

1. **Stacked/ensembled boosters (AutoGluon‑style, or custom stacking of CatBoost + LightGBM + XGBoost + linear models)**  
   - Consistently give the **best performance** when you care about out‑of‑sample stability, not just one lucky split.  
   - This is what most serious competition/quant setups converge to: multiple strong but different GBMs, plus a simple meta‑learner.

2. **TabPFN‑style foundation models (on benchmarks like TabArena)**  
   - As *single* models, they now beat or match tuned GBMs across many datasets, including low‑signal ones.  
   - They’re very strong, but operationally heavier (GPU, new tooling) than classic boosters.

3. **Among classic single GBMs, CatBoost very slightly edges out LightGBM/XGBoost on average**, especially when signal is weak and categorical/interaction structure matters, with **LightGBM** usually next and **XGBoost** the most stable but rarely the absolute top.

So: for a Numerai‑like regime, the best-performing *practical* choice is usually **a stacked ensemble of diverse GBMs (CatBoost + LGBM + XGB) plus a regularized linear meta‑model**, not any single booster or bagger on its own.