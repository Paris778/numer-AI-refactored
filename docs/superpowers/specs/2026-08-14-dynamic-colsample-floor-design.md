# Dynamic `colsample_bytree` Floor — Design Spec

- **Date:** 2026-08-14
- **Status:** Approved (Director disposition: APPROVED, Approach A)
- **Scope:** Phase 1, item 3 of the committee-corrected work plan: prevent small feature
  subsets (|S| ≤ 10) from being crippled by sampling a sub-fraction that evaluates to
  ~1 feature per tree split.

## 1. Context & Problem

The benchmark-rebuild campaign's screen cells (v2–v4, 3 features) ran `preset: fast` with no
override, so `colsample_bytree = 0.1` sampled 1 of 3 features per tree — no split choice
(committee Red Flag 8). The fix is a **raise-only dynamic floor** on the feature-sampling
fraction, sized to the resolved feature count.

## 2. Mathematical Contract

For a feature set of size |S| ≥ 1 and a resolved parameter value c_resolved ∈ (0, 1]:

```
colsample_floor(|S|) = min(1.0, max(0.1, min(10.0, float(|S|)) / float(|S|) + 1e-7))
c_effective            = min(1.0, max(c_resolved, colsample_floor(|S|)))
```

The ε = 1e-7 expansion guards the float32 truncation hazard: C++ backends cast
`static_cast<int>(n_features * fraction)` in single precision, and `float32(10/42) × 42`
evaluates to 9.9999999… → 9 sampled features. ε lands ~7 ulps above the boundary (well above
rounding, far below the next feature), so `float32(floor(42)) × 42 ≥ 10.000004` → 10. The
expansion applies **inside** the `max(0.1, …)` bound, so |S| ≥ 100 keeps floor = 0.1 exactly
(the ≥100-feature bit-identical regression is preserved) and |S| ≤ 10 keeps floor = 1.0
exactly. A floor is a lower bound, never a ceiling: an explicitly configured value above the
floor is preserved. Behavior table (locked):

| |S| | floor | config | effective | sampled |
|---|---|---|---|---|---|
| 3 | 1.0 | 0.1 (preset) | **1.0** | 3 |
| 10 | 1.0 | 0.1 (preset) | **1.0** | 10 |
| 42 | 0.2380953358 | 0.1 (preset) | **0.2380953358** | 10 |
| 42 | 0.2380953358 | 0.5 (explicit) | **0.5** | 21 |
| 780 | 0.1 | 0.1 (preset) | **0.1** | 78 |

## 3. Components (`nmr/models.py` only)

- `_colsample_floor(n_features: int) -> float` — pure helper; `ValueError("n_features must be >= 1")`
  for `n_features < 1` (defense-in-depth; ingestion already blocks empty subsets); returns
  `min(1.0, max(0.1, min(10.0, float(n_features)) / float(n_features) + 1e-7))`.
- `_resolved_params(self, *, use_gpu: bool, n_features: int) -> dict` — applies the raise-only
  floor on the backend-final sampling key(s):
  - LightGBM: floor **every present member** of the alias group
    `{colsample_bytree, feature_fraction, sub_feature}` (verified against lightgbm 4.6.0's
    `_ConfigAliases`: all three are one alias group, and unknown kwargs flow through the
    sklearn wrapper's `**kwargs` into the native engine — flooring only `colsample_bytree`
    would let a user-supplied `feature_fraction`/`sub_feature` bypass the floor).
    Flooring all present members is precedence-proof regardless of which alias the wrapper
    forwards to C++.
  - XGBoost: `colsample_bytree` (canonical native name).
  - CatBoost: `rsm` *after* `_translate_catboost` (so a user-supplied CatBoost-native `rsm`
    is bounded identically).
- Thread `n_features` through `_fit_model(features, …)` (source of truth: `features.shape[1]`)
  → `_device_candidate_params(use_gpu, n_features)` → `_resolved_params`. Both the GPU and CPU
  candidates receive the identical floored value (cross-device parameter parity).
- Full-history subprocess path needs no spec change: the child derives the count from
  `len(spec["feature_cols"])` at fit time.

## 4. Known Divergence (accepted)

`resolve_model_params` and the Optuna baseline anchor remain untouched: an enqueued anchor
trial may show the raw `colsample_bytree` value while the actual trial-0 fit uses the floored
one. Visible only when the search space includes `colsample_bytree`; consistent with how
runtime device constraints already operate. Documented in ARCHITECTURE §G/§S.

## 5. Determinism & Run-Identity

Pure deterministic function; no stochastic ops, no hashed-payload changes. Consequence:
existing configs with < 100 features get changed effective params → new `run_id`s (the code
fingerprint changes regardless). Configs with ≥ 100 features are bit-identical in behavior.

## 6. Testing Plan

- `_colsample_floor` unit tests: |S| ∈ {1, 3, 10, 42, 780}, the `n_features < 1` guard, and the
  raise-only override rule.
- **Float32 truncation guard:** `int(np.float32(_colsample_floor(42)) * 42) == 10` — the ε
  expansion must survive the single-precision cast + multiply that the C++ backends perform.
- `_resolved_params` integration tests: floor applied to `colsample_bytree` (LightGBM/XGBoost)
  and to `rsm` (CatBoost, incl. user-native `rsm`), on both CPU and GPU candidate specs; an
  explicit override above the floor is preserved; |S| ≥ 100 leaves the value untouched.
- **LightGBM alias tests:** user-supplied `feature_fraction` and `sub_feature` each get floored;
  a config carrying multiple members of the alias group floors all of them.
- Regression: preset resolution for ≥ 100 features unchanged (bit-identical params).

## 7. Audit Log

- **Refinement 1 (float32 truncation, audit 2026-08-14): accepted** — ε = 1e-7 inside the
  `max(0.1, …)` bound. Note: the audit's behavior-table row for |S| = 780 (0.1000001)
  contradicts its own formula; the formula as written yields exactly 0.1, which is what this
  spec implements (preserves the ≥100-feature bit-identical regression).
- **Refinement 2 (LightGBM alias normalization): accepted with a stronger fix** — verified
  against the installed oracle (lightgbm 4.6.0 `_ConfigAliases`; `**kwargs` → native engine).
  This spec floors **every present member** of the alias group rather than the audit's
  first-match, which is precedence-proof under the wrapper's alias resolution.

## 8. Out of Scope

- Phase 1, item 4 (DSR cumulative-trial tracking) — separate spec.
- Corrected campaign re-run and golden-doc regeneration.
