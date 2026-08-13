# Train-Only, FDR-Controlled Feature Screen — Design Spec

- **Date:** 2026-08-13
- **Status:** Approved (Director disposition: APPROVED with 3 mandatory amendments — incorporated verbatim below)
- **Scope:** Phase 1, items 1–2 of the committee-corrected work plan: train-only subset derivation + native Benjamini–Hochberg FDR + CI-aware `stable` predicate.

## 1. Context & Problem

The `derived_sets` analysis stage built campaign feature subsets (`screen_stable`, …) from a
stability screen computed over **both** splits (train `0001..0574` + validation `0575..1231`,
1,218 eras — `analyze_dataset.py` `_stage_screens`). Subset selection therefore saw
validation-era labels: selection-stage look-ahead leakage (committee Red Flag 1). Separately,
the `stable` predicate tested only point estimates (|mean_corr| ≥ 0.01, |decay_slope| ≤ 0.001)
and ignored the already-computed horizon-aware block-bootstrap CIs and any multiple-testing
control (committee Weaknesses 3, 5).

## 2. Decisions (locked)

1. **Full gate.** `stable = |mean_corr| ≥ 0.01 ∧ |decay_slope| ≤ 0.001 ∧ CI excludes zero ∧ fdr_q ≤ 0.05`.
   An empty `screen_stable` is a valid scientific result, not a pipeline failure.
2. **Bootstrap p-value.** Hall's two-sided null-shifted block bootstrap, studentized with a
   **global** bootstrap SE denominator: T₀ = x̄ / se₀, T_b* = (x̄*_b − x̄) / se₀. Since se₀ is a
   positive scalar constant, it cancels in the comparison |T_b*| ≥ |T₀| ⇔ |x̄*_b − x̄| ≥ |x̄|, so
   p = (1 + Σ_b I(|x̄*_b − x̄| ≥ |x̄|)) / (B + 1). Cost is O(B), not O(B²); the comparison is exact.
   **Degenerate-series convention (final green-light patch):** a constant series
   (std < 1e-12) or fewer than 2 observations returns p = 1.0. At zero variance the
   studentized statistic is undefined (the cancellation requires se₀ > 0); the repo's
   fail-safe doctrine never claims signal from degenerate inputs. Documented consequence:
   a constant non-zero IC series gets p = 1.0 while its CI is [c, c] — formally non-dual,
   deliberately conservative; such features can never be `stable`.
3. **Separate train-only stage.** New `screens_train` stage (train eras only) writes
   `feature_ic_screen_train.parquet`; `derived_sets` reads **only** that file. The existing
   full-span `screens` stage remains as descriptive characterization and is labeled as such.
4. **Zero new dependencies.** BH implemented in pure NumPy (no statsmodels); bootstrap p-value
   reuses `nmr/inference.py`'s seeded circular-block machinery and horizon floors (5 eras for
   20D targets, 13 for 60D), with `n_boot = ci_boot`, `seed = ci_seed` — the same budget and
   block-index convention as the existing CI, so p-value and CI are a consistent pair.

## 3. Mandatory Amendments (incorporated)

### A. Non-finite p-value mapping in Benjamini–Hochberg

- Any NaN/Inf/non-finite p-value is **coerced to 1.0** before sorting and ranking.
- Ranking length m = total input length N (including non-finites).
- Adjusted q-values enforce backward monotonicity: q_(i) = min(1.0, min_{k≥i} (m/k) p_(k)).
- Non-finite inputs evaluate to `fdr_q = 1.0` and `fdr_pass = False`.

### B. Strict boundary condition for `ci_excludes_zero`

`ci_excludes_zero = (ci_lo > 0 AND ci_hi > 0) OR (ci_lo < 0 AND ci_hi < 0)`.
If either bound touches zero exactly (or is null), the flag is `False`. Exposed as a column.

### C. Ingestion exception contract for empty subsets

`IngestionAgent.features(subset_name)` resolving to 0 feature columns raises `ValueError`
immediately, with the exact message:

```
Resolved feature set '<name>' is empty (0 features). Cannot train pipeline on an empty subset. Verify 'screens_train' output.
```

### D. Constant / zero-variance IC fallback (final green-light patch)

`block_bootstrap_pvalue`: if `std(series) < 1e-12` or fewer than 2 observations → `p = 1.0`.
A zero-variance series cannot support a bootstrap significance claim; this is the fail-safe
branch of the repo's degenerate-data doctrine (see §2, decision 2).

### E. Explicit 1-based BH ranks, per-target grouping, and schema contract (final green-light patch)

1. BH ranking uses explicit 1-based ranks `np.arange(1, m + 1)` — no zero-division, m counts
   the full input length including non-finites.
2. BH is computed **strictly per target** inside `feature_ic_screen`'s target loop — p-values
   are never pooled across 20D/60D horizons.
3. The 16-column Polars dtype contract (`SCREEN_PARQUET_SCHEMA`, §5) is enforced with an
   explicit schema cast at the `feature_ic_screen` boundary, so parquet aggregation downstream
   never sees type drift.

## 4. Components

| File | Change |
|---|---|
| `nmr/inference.py` | Add `block_bootstrap_pvalue(series, *, block_len, n_boot, seed) -> float` (finite 1-D only; validated; deterministic per seed) and `benjamini_hochberg(p_values, *, q=0.05) -> np.ndarray` (adjusted q-values aligned to input order; Amendment A). Export in `__all__`. |
| `nmr/analysis.py` | `feature_ic_screen`: per-feature bootstrap p-values + BH per target; new columns `ci_excludes_zero`, `p_value`, `fdr_q`, `fdr_pass`; `stable` overridden to the full gate; extract `_screen_block_len(n_eras, horizon)` shared by the CI and p-value paths. |
| `nmr/data.py` | `features()` raises on empty subsets (Amendment C). |
| `analyze_dataset.py` | New `screens_train` stage (train split only → `feature_ic_screen_train.parquet`); `derived_sets` reads the train file only; `RuntimeError` naming `screens_train` if missing. Full-span `screens` unchanged. |
| `nmr/__init__.py` | Export `block_bootstrap_pvalue`, `benjamini_hochberg`. |

## 5. Screen parquet column order

`feature, target, mean_corr, mean_corr_ci_lo, mean_corr_ci_hi, ci_excludes_zero, p_value,
fdr_q, fdr_pass, corr_std, decay_slope, cross_regime_variance, mean_spearman, n_eras, stable,
nonlinear`

## 6. Data flow

```
train.parquet (0001..0574) ──> [screens_train] ──> feature_ic_screen_train.parquet
                                                          │
                                                          ▼
                                                  [derived_sets] <── [drift]
                                                          │
                                                          ▼
                                              derived_feature_sets.json
                                                          │
                                                          ▼
                                        IngestionAgent (empty-set guard)

train+validation (0001..1231) ──> [screens] ──> feature_ic_screen.parquet   (descriptive only)
```

The "descriptive full-span, DO NOT use for subset derivation" disclosure lives in the
renderer's schema line, the stage docstrings, and ARCHITECTURE.md §O — polars cannot write
custom key-value parquet metadata. The structural guarantee is the separate artifact file:
`derived_sets` cannot read the full-span file.

## 7. Determinism & performance

All stochastic operations are seeded (`ci_seed`) and use the same block-index convention as
`block_bootstrap_ci`; output is cross-process deterministic (tested). The p-value pass
doubles the screen's bootstrap cost — seconds for `medium`, minutes at the full 3,555-feature
universe (same order as the existing CI pass).

## 8. Error handling

- Features with < 2 valid eras: p-value/fdr columns null, `stable = False`. Both the CI
  and p-value paths slice each feature to its own finite eras and resolve a per-feature
  block length — partial era coverage can never exceed `block_len <= n` (review patch).
- `block_bootstrap_pvalue`: empty / non-finite / `block_len` / `n_boot` violations raise `ValueError`.
- `benjamini_hochberg`: invalid `q` or 2-D input raises `ValueError`; non-finite p-values coerced per Amendment A.
- `derived_sets`: missing `feature_ic_screen_train.parquet` raises `RuntimeError` naming the `screens_train` stage.
- Empty derived sets are valid outputs; training on an empty subset fails loudly at ingestion (Amendment C).

## 9. Testing plan

- `tests/test_inference.py`: BH known example (n=6, first five rejected at q=0.05), order
  invariance, empty/all-uniform arrays, validation (q range, 2-D), Amendment A coercion
  (NaN/Inf → fdr_q = 1.0); bootstrap p-value — tiny p for consistent signal, p = 1 for exact
  zero mean, determinism + bounds, validation errors.
- `tests/test_analysis.py`: engineered-IC screen gate test (strong feature passes all gates;
  marginal feature passes the classic point predicate but CI spans zero → `stable = False`);
  new-column contract (exact list incl. `ci_excludes_zero`, `fdr_pass`); p/fdr/stable
  determinism across calls.
- `tests/test_analyze_dataset.py`: `screens_train` restricted to train eras (n_eras ≤ train-era
  count); `derived_sets` reads the train file (deleting the full-span parquet is harmless,
  deleting the train parquet raises); dump lists include `feature_ic_screen_train.parquet`.
- `tests/test_data.py`: empty supplemental subset raises with the Amendment C message.

## 10. Out of scope (separate specs)

- `colsample_bytree` dynamic floor (Phase 1, item 3).
- DSR cumulative-trial tracking (Phase 1, item 4).
- Corrected campaign re-run and golden-doc regeneration (require the corrected analysis run).
- Renderer table gains for p/fdr columns (dumps carry full precision; renderer prose updated).
