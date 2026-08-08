# CatBoost Backend — Design

> Status: drafted 2026-08-08 (user delegated: "plan your own work and proceed"). Sub-project 2 of the external-library grant (BO ✓ → CatBoost → dashboard).

**Goal:** Add `catboost` as a third model backend for `ModelOrchestrator` (alongside LightGBM/XGBoost), with the same determinism, leakage-safety, GPU-fallback, deployment, and tested-boundary guarantees.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Dependency | `catboost==1.2.10` (user-granted exception) | Latest; verified available via pip index |
| Config | `VALID_MODEL_BACKENDS = ("lightgbm", "xgboost", "catboost")` | Third closed-enum value; invalid values still raise at load |
| Param model | Preset dicts stay backend-agnostic; `_resolved_params` gains a **catboost translation branch** (mirrors the existing xgboost translation) | One source of truth for presets (`_CANONICAL_PRESETS` + `resolve_model_params`); per-backend translation is the established pattern |
| Determinism | `random_seed=seed`, `thread_count=1`, CPU single-thread, verified empirically | Matches the `n_jobs=1`/seeded invariant of the other backends |
| Scope note | No categorical-feature support wiring | Numerai v5 features are all numeric (obfuscated); CatBoost's value here is algorithmic diversity for ensembling, not categorical handling. Documented, not built |

## Architecture

- `nmr/config.py`: `VALID_MODEL_BACKENDS` += `"catboost"`.
- `nmr/models.py`:
  - `_translate_catboost(resolved: dict[str, Any], *, use_gpu: bool) -> dict[str, Any]` — module-level helper mapping the shared preset/override knobs to CatBoost's API:
    - `n_estimators` → `iterations`; `learning_rate` → `learning_rate`; `max_depth` → `depth`; `min_data_in_leaf` → `min_data_in_leaf`; `colsample_bytree` → `rsm`.
    - `num_leaves` is **dropped** for catboost (symmetric depth-limited trees; `depth` already bounds capacity) — documented, not an error.
    - Fixed contract params: `loss_function="RMSE"`, `random_seed=<seed>`, `thread_count=1`, `verbose=False`, `allow_writing_files=False` (CatBoost writes files by default — must be disabled for repo hygiene), `task_type="GPU"|"CPU"`, `devices="0"` when GPU.
  - `_resolved_params(use_gpu=...)` gains a catboost branch calling the translation (after `resolve_model_params`).
  - `_build_model` → `catboost.CatBoostRegressor(**params)`.
  - `_fit_model`: `backend_errors` tuple += `catboost.CatBoostError`; `resolved_device` derived from `task_type == "GPU"`.
  - `_device_candidate_params` unchanged (GPU-first with CPU fallback; the catboost branch produces distinct GPU/CPU param dicts).
- `requirements.txt`: pin `catboost==1.2.10`.
- **No changes** to evaluation, splitter, ensembling, risk, scorecard, registry — metrics are backend-agnostic; parity tests untouched.

## Determinism & deployment

- **Determinism:** CatBoost CPU + `thread_count=1` + fixed `random_seed` is deterministic (same config+data+code+device ⇒ identical OOF). Verified by a new test (two `train_cross_validation` runs, same seed → identical OOF frames) mirroring the runner's determinism test. GPU determinism is NOT guaranteed — the per-device caveat already in AGENTS.md covers it.
- **Deployment:** the deploy closure embeds the trained `CatBoostRegressor` via cloudpickle; `load_predict` roundtrip is tested locally. **Caveat (documented in AGENTS.md hazards + ARCHITECTURE §G):** CatBoost availability in Numerai's hosted predict runtime is unverified — a catboost-backed artifact must be validated against the hosted runtime before it is staked. Local `load_predict` fidelity is covered by the existing F-019-style test pattern.
- Nothing new enters `canonical_scorecards_bytes`; the run_id environment fingerprint is unchanged. **Decision: do NOT add `catboost` to `_environment_fingerprint`** — adding it would invalidate every existing run_id for a marginal benefit; per-backend version drift is governed by the `requirements.txt` pin (same policy as `optuna`). The fingerprint stays a coarse cross-backend stability marker over `{numpy, polars, pandas, lightgbm, xgboost}`; catboost-backend reproducibility rests on the pin + CI, documented in ARCHITECTURE §G.

## Testing (`tests/test_models.py` + `tests/test_runner.py` extensions)

- Config: `ModelConfig(backend="catboost")` valid; `backend="bogus"` still raises (existing test).
- Translation: `_translate_catboost` maps the knobs (iterations/rsm/depth/min_data_in_leaf), drops `num_leaves`, and sets the fixed contract params (`loss_function`, `random_seed`, `thread_count`, `verbose`, `allow_writing_files`, `task_type`).
- Determinism: `train_cross_validation` with catboost, same seed → identical OOF (vtest fixture, `iterations=10` override).
- GPU-fallback: monkeypatch the catboost GPU fit to raise → `resolved_device == "cpu"` and the model still trains (mirrors the lightgbm fallback test).
- Runner end-to-end: `ExperimentRunner` with `backend="catboost"` on the vtest fixture (deploy=False + validation scorecard off for speed; then one deploy=True `load_predict` roundtrip with non-constant predictions).
- Resolved device recorded in manifest (`oof_device`).
- Test count: +8–10 → count sync in AGENTS/README/CONTRIBUTING (established precedent).

## Docs (same change set)

- `AGENTS.md`: mission line "multi-target LightGBM/XGBoost training" → "LightGBM/XGBoost/CatBoost"; §6 toolkit row for models unchanged (points to `nmr/models.py`); §8 hazards += the hosted-runtime caveat line.
- `ARCHITECTURE.md`: §G gains the catboost branch (translation table, fixed contract params, determinism note, hosted-runtime caveat); §5 tool registry backends list += catboost.
- `README.md`: stack line += CatBoost; tree comment for `models.py` mentions the three backends.
- `requirements.txt`: pin.
- Doc-SSOT count sync.

## Out of scope

- Categorical-feature plumbing (no categorical columns in Numerai data).
- GPU determinism guarantees (per-device caveat already documented).
- Adding catboost to the run_id environment fingerprint (pin-based reproducibility instead; documented).
- No metric, splitter, ensembling, risk, or scorecard changes.
