# CatBoost Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved CatBoost-backend design: `catboost` as a third `ModelOrchestrator` backend with the same determinism, GPU-fallback, deployment, and tested-boundary guarantees as LightGBM/XGBoost.

**Architecture:** `_CANONICAL_PRESETS` stays backend-agnostic; `models._resolved_params` gains a catboost translation branch (fixed contract params win over renamed user params; `num_leaves` dropped). Determinism via `random_seed` + `thread_count=1` + CPU, verified empirically. `VALID_MODEL_BACKENDS` gains `"catboost"`. Deploy works locally (`load_predict` roundtrip tested); hosted-runtime compatibility documented as a caveat, not assumed.

**Tech Stack:** Python 3.11+ (venv 3.12), Polars, LightGBM/XGBoost, **CatBoost 1.2.10** (user-granted dependency exception).

## Global Constraints

- `nmr/` is the only tested boundary; scripts contain zero business logic.
- TDD: no production code without a failing test first.
- **Dependency exception (user-granted 2026-08-08):** `catboost==1.2.10` pinned, imported ONLY in `nmr/models.py`.
- Determinism: `random_seed=seed`, `thread_count=1`, CPU single-thread; verified by a same-seed identical-OOF test. GPU determinism NOT guaranteed (per-device caveat).
- `allow_writing_files=False` is mandatory (CatBoost writes files by default — repo hygiene).
- Contract params (`loss_function`, `random_seed`, `thread_count`, `verbose`, `allow_writing_files`, `task_type`) are NOT overridable by user params.
- `num_leaves` is dropped for catboost (symmetric depth-limited trees); `depth` bounds capacity.
- `catboost` NOT added to the run_id environment fingerprint (pin-based reproducibility; same policy as `optuna`).
- No metric, splitter, ensembling, risk, or scorecard changes; nothing enters `canonical_scorecards_bytes`.
- Doc SSOT same-change-set; AGENTS ≤ 32 KB; test-count claims synced numeric-only (precedent).
- Git flow (user-authorized pattern): work on branch `catboost-backend` (created), commits per task, `main` untouched until the user chooses integration. No push.
- Verification honesty: run commands, read output, report truthfully.

## File Structure

| File | Responsibility |
|---|---|
| `nmr/models.py` (modify) | `_translate_catboost`, `_resolved_params` branch, `_build_model`, `_fit_model` errors + device |
| `nmr/config.py` (modify) | `VALID_MODEL_BACKENDS` += `"catboost"` |
| `nmr/__init__.py` (modify) | No new exports (backend is config-driven); verify nothing needed |
| `requirements.txt` (modify) | Pin `catboost==1.2.10` |
| `configs/example.yaml` (modify) | `backend: lightgbm # lightgbm \| xgboost \| catboost` |
| `tests/test_models.py`, `tests/test_runner.py` (modify) | CatBoost translation/determinism/fallback + runner integration |
| `ARCHITECTURE.md`, `AGENTS.md`, `README.md` (modify) | §G, §5, mission/hazards, stack/tree |

---

### Task 1: Pin + install `catboost==1.2.10` + API verification

**Files:**
- Modify: `requirements.txt` (append `catboost==1.2.10`)

**Interfaces:**
- Produces: CatBoost available in `.venv` for Tasks 3–5. Configuration change (no test required per the TDD config-file exception).

- [ ] **Step 1: Pin** — append `catboost==1.2.10` to `requirements.txt`.
- [ ] **Step 2: Install** — `.venv/Scripts/python -m pip install catboost==1.2.10`.
- [ ] **Step 3: Verify the API (do NOT guess)** — `.venv/Scripts/python -c` checks that `catboost.CatBoostRegressor` accepts these kwargs: `iterations`, `depth`, `rsm`, `min_data_in_leaf`, `loss_function`, `random_seed`, `thread_count`, `verbose`, `allow_writing_files`, `task_type`, `devices`; and `catboost.CatBoostError` exists. Record the exact kwarg list observed — the translation (Task 3) must only use verified kwargs.
- [ ] **Step 4: Commit** — `build(deps): pin catboost==1.2.10 (user-granted exception)` on `catboost-backend`.

---

### Task 2: Config — `VALID_MODEL_BACKENDS` += `"catboost"`

**Files:**
- Modify: `nmr/config.py:23`, `configs/example.yaml`
- Test: `tests/test_config.py` (append)

**Interfaces:**
- Produces: `ModelConfig(backend="catboost")` valid; `backend="bogus"` still raises. Consumed by Tasks 3–6.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_config.py
def test_catboost_backend_is_valid():
    from nmr.config import ModelConfig

    assert ModelConfig(backend="catboost").backend == "catboost"


def test_invalid_backend_still_raises():
    from nmr.config import ModelConfig

    import pytest as _pytest

    with _pytest.raises(ValueError, match="backend"):
        ModelConfig(backend="bogus")
```

- [ ] **Step 2: Run test to verify it fails** — `.venv/Scripts/python -m pytest tests/test_config.py -q` — FAIL: `ValueError` for `catboost` (not in the tuple).
- [ ] **Step 3: Implement** — `VALID_MODEL_BACKENDS = ("lightgbm", "xgboost", "catboost")` in `nmr/config.py`; update `configs/example.yaml` line: `backend: lightgbm         # lightgbm | xgboost | catboost`.
- [ ] **Step 4: Run test to verify it passes** — all config tests green.
- [ ] **Step 5: Commit** — `feat(config): add catboost backend to valid set` on `catboost-backend`.

---

### Task 3: `nmr/models.py` — catboost translation + integration

**Files:**
- Modify: `nmr/models.py` (add `_translate_catboost` after `resolve_model_params`; `_resolved_params` catboost branch; `_build_model`; `_fit_model` backend_errors + device)
- Test: `tests/test_models.py` (append)

**Interfaces:**
- Consumes: `resolve_model_params` (existing), `catboost` (Task 1).
- Produces:
  - `_translate_catboost(resolved: dict[str, Any], *, seed: int, use_gpu: bool) -> dict[str, Any]` — module-level. Rename map: `n_estimators → iterations`, `colsample_bytree → rsm`, `max_depth → depth`; drop `num_leaves`; pass all other resolved keys through unchanged; then overlay the fixed contract (contract WINS): `loss_function="RMSE"`, `random_seed=seed`, `thread_count=1`, `verbose=False`, `allow_writing_files=False`, `task_type="GPU"|"CPU"`, plus `devices="0"` when GPU.
  - `_resolved_params` catboost branch: `base = resolve_model_params(...)` → `return _translate_catboost(base, seed=self._seed, use_gpu=use_gpu)`.
  - `_build_model`: catboost → `catboost.CatBoostRegressor(**params)`.
  - `_fit_model`: `backend_errors` tuple += `catboost.CatBoostError` when backend is catboost; `resolved_device` derived from `task_type == "GPU"`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_models.py
from nmr.models import _translate_catboost


def test_translate_catboost_maps_preset_knobs() -> None:
    resolved = {
        "n_estimators": 2000, "learning_rate": 0.01, "max_depth": 5,
        "num_leaves": 31, "colsample_bytree": 0.1, "min_data_in_leaf": 100,
    }
    params = _translate_catboost(resolved, seed=42, use_gpu=False)
    assert params["iterations"] == 2000
    assert params["learning_rate"] == 0.01
    assert params["depth"] == 5
    assert params["rsm"] == 0.1
    assert params["min_data_in_leaf"] == 100
    assert "num_leaves" not in params          # dropped: symmetric depth-limited trees


def test_translate_catboost_contract_params_are_fixed_and_win() -> None:
    resolved = {"random_seed": 1, "thread_count": 8, "n_estimators": 100}
    params = _translate_catboost(resolved, seed=42, use_gpu=False)
    assert params["loss_function"] == "RMSE"
    assert params["random_seed"] == 42         # contract wins over user params
    assert params["thread_count"] == 1         # contract wins over user params
    assert params["verbose"] is False
    assert params["allow_writing_files"] is False
    assert params["task_type"] == "CPU"
    assert params["iterations"] == 100         # non-contract keys still map


def test_translate_catboost_gpu_sets_task_type_and_devices() -> None:
    params = _translate_catboost({"n_estimators": 100}, seed=42, use_gpu=True)
    assert params["task_type"] == "GPU"
    assert params["devices"] == "0"


def test_catboost_cv_oof_is_deterministic_under_seed(tmp_path) -> None:
    # Reuse the existing _write_data fixture in tests/test_models.py; run
    # train_cross_validation twice with backend="catboost", same seed, and
    # assert the OOF frames are identical. Use iterations=10 override for speed.
    cfg = ModelConfig(backend="catboost", preset="fast", params={"n_estimators": 10})
    orch = ModelOrchestrator(cfg, seed=17)
    first = orch.train_cross_validation(df, feature_cols=..., target_col="target", splitter=..., era_col="era")
    second = ModelOrchestrator(cfg, seed=17).train_cross_validation(df, ...)
    assert first.oof.equals(second.oof)
    assert orch.resolved_device == "cpu"


def test_catboost_gpu_fallback_records_cpu(tmp_path, monkeypatch) -> None:
    # Mirror the existing lightgbm GPU-fallback test: monkeypatch the catboost
    # GPU fit (task_type="GPU") to raise catboost.CatBoostError, run
    # train_cross_validation, assert resolved_device == "cpu" and the OOF is
    # non-empty (the CPU candidate succeeded).
```

- [ ] **Step 2: Run test to verify it fails** — `.venv/Scripts/python -m pytest tests/test_models.py -q` — FAIL: `_translate_catboost` not defined.
- [ ] **Step 3: Implement** per the Interfaces contract (read the existing xgboost branch of `_resolved_params` and the lightgbm GPU-fallback test first — mirror their structure).
- [ ] **Step 4: Run test to verify it passes** — all `tests/test_models.py` green (existing + new).
- [ ] **Step 5: Commit** — `feat(models): add catboost backend with translation and GPU fallback` on `catboost-backend`.

---

### Task 4: Runner integration — catboost end-to-end

**Files:**
- Test: `tests/test_runner.py` (append)

**Interfaces:**
- Consumes: Task 3 (models support).
- Produces: proof that `ExperimentRunner` works end-to-end with `backend="catboost"` (deterministic run, deploy + `load_predict` roundtrip, `oof_device` recorded).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_runner.py
def test_runner_catboost_end_to_end(tmp_path) -> None:
    cfg = _config(tmp_path)
    catboost_cfg = ExperimentConfig(
        data=cfg.data, split=cfg.split,
        model=ModelConfig(backend="catboost", preset="fast",
                          params={"n_estimators": 10, "learning_rate": 0.05}),
        evaluation=cfg.evaluation, run=cfg.run,
    )
    runner = ExperimentRunner(catboost_cfg)
    first = runner.run(deploy=True)
    second = ExperimentRunner(catboost_cfg).run(deploy=False)
    assert first.run_id == second.run_id
    assert first.oof.equals(second.oof)
    assert first.manifest["oof_device"] == "cpu"
    assert first.artifact is not None
    loaded = load_predict(first.artifact.path)
    live_features = pd.DataFrame(
        {"f1": [0.1, 0.2, 0.3], "f2": [0.3, 0.4, 0.5]}, index=["a", "b", "c"]
    )
    pred = loaded(live_features)
    assert pred["prediction"].notna().all()
    assert pred["prediction"].nunique() > 1  # non-constant pipeline output
```

- [ ] **Step 2: Run test to verify it fails** — `.venv/Scripts/python -m pytest tests/test_runner.py -q` — FAIL: catboost config or model errors (or `ModuleNotFoundError` if `_build_model` is unimplemented).
- [ ] **Step 3: Implement** — nothing new is expected in the runner; fix any integration gaps surfaced by the test (e.g., `_build_deploy_pipeline` closure — catboost's `predict` returns raw values, matching the existing contract; if the closure needs a catboost-specific branch, add it minimally and test).
- [ ] **Step 4: Run test to verify it passes** — `tests/test_runner.py` green.
- [ ] **Step 5: Commit** — `feat(runner): catboost end-to-end support` on `catboost-backend`.

---

### Task 5: Docs + count sync

**Files:**
- Modify: `AGENTS.md`, `ARCHITECTURE.md`, `README.md`, AGENTS/README/CONTRIBUTING count claims

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: SSOT docs per the design.

- [ ] **Step 1: `AGENTS.md`** — mission line: "multi-target LightGBM/XGBoost training" → "multi-target LightGBM/XGBoost/CatBoost training"; §8 hazards += the hosted-runtime caveat: "CatBoost-backed artifacts: local `load_predict` fidelity is tested, but CatBoost availability in Numerai's hosted predict runtime is UNVERIFIED — validate a catboost deploy against the hosted runtime before staking on it."
- [ ] **Step 2: `ARCHITECTURE.md` §G** — add the catboost branch: translation table (rename map, `num_leaves` dropped, contract params win), determinism note (`random_seed` + `thread_count=1` + CPU; per-device caveat), `allow_writing_files=False`, fingerprint-exclusion note (pin-based reproducibility). §5 tool registry: backend list += `catboost`.
- [ ] **Step 3: `README.md`** — stack line += CatBoost; `models.py` tree comment → "ModelOrchestrator — LightGBM/XGBoost/CatBoost, CV OOF + anchor".
- [ ] **Step 4: Count sync** — compute the new total (393 + tests added across Tasks 2–4) and update the numeric claims in AGENTS.md/README.md/CONTRIBUTING.md (numeric-only, precedent).
- [ ] **Step 5: Run** — focused suites + full `pytest -q` + `tests/test_docs_hygiene.py -q`; all green.
- [ ] **Step 6: Commit** — `docs: document catboost backend (ARCH §G, AGENTS mission + hazard, README)` on `catboost-backend`.

---

### Task 6: Full verification gate

- [ ] **Step 1: Full suite** — `.venv/Scripts/python -m pytest -q` — PASS, exact count reported.
- [ ] **Step 2: Benchmark smoke** — `.venv/Scripts/python benchmark_runner.py --fast-mode --output artifacts/benchmark_scores_smoke.csv --labels-output artifacts/benchmark_test_era_labels_smoke.csv` — exits 0 (training-path change).
- [ ] **Step 3: Doc-SSOT scan** — AGENTS ≤ 32 KB; §G matches `models.py`; no duplication.
- [ ] **Step 4: Record** — report truthfully.

---

## Self-Review

**Spec coverage** (design → task): dependency pin + API verification → T1; config enum → T2; translation + integration + determinism + fallback → T3; runner e2e + deploy roundtrip → T4; docs + count → T5; gate → T6. All design decisions (num_leaves dropped, contract-wins, fingerprint exclusion, hosted-runtime caveat) mapped. **Placeholder scan:** the only adapts are "mirror the existing lightgbm GPU-fallback test" and "read the xgboost branch first" — deliberate integration instructions, not placeholders. **Type consistency:** `_translate_catboost(resolved, *, seed, use_gpu)` consistent across T3 tests/impl; `ModelConfig(backend="catboost")` across T2/T3/T4.
