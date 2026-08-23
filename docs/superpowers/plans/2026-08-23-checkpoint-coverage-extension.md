# Checkpoint Coverage Extension — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the OOF checkpoint/resume system to the deploy fits (per-target pickled models) and the validation stage (per-batch prediction frames), reusing the existing code/device identity manifest discipline.

**Spec:** `docs/superpowers/specs/2026-08-23-checkpoint-coverage-extension-design.md` (authority).

**Architecture:** Extract the manifest/identity/atomic-write helpers from the OOF checkpoint implementation into reusable functions in `nmr/_oof.py`; `_build_deploy_pipeline` and `_run_validation_stage` gain `*_checkpoint_dir: Path | None = None` keywords (default None = today's behavior); `ExperimentRunner.run()` passes `artifacts/runs/<run_id>/deploy_checkpoints` and `.../validation_checkpoints`.

## Global Constraints

- Determinism invariant: same run_id + matching manifest ⇒ resumed stage byte-identical to uninterrupted run; proven by mixed-resume tests (delete one unit → refit → equality + load/fit log mix).
- Identity: `manifest.json` at each checkpoint root with `code_sha256` (SHA-256 of `nmr/models.py` + `nmr/splitter.py` + `nmr/runner.py`) and `device` (post-fit `resolved_device`); written atomically at the FIRST completed unit; mismatch/torn tree → `ValueError` with delete-to-refit guidance.
- Atomic writes via `nmr/_atomicio.py::atomic_write_bytes` only (cloudpickle → bytes; frames → BytesIO → bytes).
- Public API safety: existing `tests/test_runner.py` + `tests/test_checkpointing.py` stay green unchanged; `None` dirs reproduce today's behavior exactly.
- **CPU discipline (fleet smoke running on this machine)**: run ONLY the targeted test files per task. Full suite + CI are the merge gate.
- `cloudpickle` loads restricted to artifacts this repo produced (existing hazard; the sha256 manifest detects corruption, not tampering).

---

### Task A: Shared checkpoint helpers (behavior-neutral refactor)

**Files:**
- Modify: `nmr/_oof.py` (extract + OOF path refactor)
- Modify: `nmr/models.py` (only if the OOF path's helper imports move — prefer keeping models.py untouched by importing the shared helpers from nmr._oof as it already does)
- Test: `tests/test_checkpointing.py` (existing tests are the regression net — add nothing new except a direct unit test for each extracted helper)

**Interfaces:**
- Produces (in `nmr/_oof.py`):
  - `fitting_code_sha256() -> str` (extend the existing file set with `nmr/runner.py`)
  - `checkpoint_manifest(device: str) -> dict[str, str]` — `{"code_sha256": ..., "device": device}`
  - `verify_checkpoint_manifest(manifest_path: Path, current_device: str | None) -> None` — code exact-compare; device exact-compare when `current_device` known; schema reject (device must be in `_KNOWN_RESOLVED_DEVICES`) when unknown; raises the existing message texts (code mismatch / device mismatch)
  - `ensure_no_torn_tree(manifest_path: Path) -> None` — parts-without-manifest raise
  - `write_frame_atomic(frame, path)` (existing `_write_frame_atomic` renamed/exported) and `write_bytes_atomic(data, path)`
- Refactor: `ModelOrchestrator._cv_fold_parts` switches to these helpers with **no behavior change** (the existing checkpoint tests are the proof).

- [ ] **Step 1**: Extract the helpers into `nmr/_oof.py`; refactor `_cv_fold_parts` in `nmr/models.py` to call them (it already local-imports from `nmr._oof`). Extend the code-identity file set with `nmr/runner.py` — NOTE: this changes `code_sha256` values, so any OOF checkpoint written before this commit becomes invalid on resume (correct behavior: old checkpoints raise; the error message tells the operator to delete — acceptable, and the campaign is done so nothing is at risk).
- [ ] **Step 2**: Add direct unit tests for each helper in `tests/test_checkpointing.py` (manifest roundtrip, verify pass/mismatch paths, torn-tree raise, atomic frame/bytes write leaves no temp files).
- [ ] **Step 3**: Run `./.venv/Scripts/python -m pytest tests/test_checkpointing.py tests/test_models.py -q` → all green (existing tests unchanged prove the refactor is behavior-neutral modulo the code-set change, which the tests don't pin).
- [ ] **Step 4**: Lint + commit: `refactor(oof): extract shared checkpoint helpers (manifest/identity/atomic-write)`.

---

### Task B: Deploy-fit checkpoints

**Files:** `nmr/runner.py`, `tests/test_runner.py`

- [ ] **Step 1: Failing test** (in `tests/test_runner.py`, synthetic experiment helpers — read the file first): run the synthetic experiment with deploy enabled and a tmp artifacts dir → assert `deploy_checkpoints/<target>.pkl` files exist for all 4 targets + `manifest.json`; delete one `.pkl` → run again (same config+data → same run_id) → the deploy closure's validation predictions equal the first run's (compare the scorecard or the predictions frame the run exposes); caplog shows one `train_full_history` and one `loaded deploy checkpoint`.
- [ ] **Step 2: Implement** in `_build_deploy_pipeline(..., deploy_checkpoint_dir: Path | None = None)`: per target, `pkl_path = deploy_checkpoint_dir / f"{target}.pkl"`; manifest verify/init at the first fitted target (device = orchestrator's post-fit `resolved_device`); if `pkl_path` exists → `cloudpickle.loads(pkl_path.read_bytes())` (wrap load errors in the corrupt-checkpoint ValueError) + log `loaded deploy checkpoint`; else `train_full_history` + `cloudpickle.dumps` → `atomic_write_bytes` + log. Wire `run()`: `deploy_checkpoint_dir = artifacts_dir / "runs" / run_id / "deploy_checkpoints"` passed only when `deploy or validation_scorecard` (the pipeline's callers).
- [ ] **Step 3: Identity tests**: tamper the deploy manifest's `code_sha256` / `device` → ValueError; delete manifest but keep `.pkl` → torn-tree ValueError.
- [ ] **Step 4**: `./.venv/Scripts/python -m pytest tests/test_runner.py tests/test_checkpointing.py -q` → green; lint; commit: `feat(runner): per-target deploy-fit checkpoints with identity manifest`.

---

### Task C: Validation predict-batch checkpoints + docs

**Files:** `nmr/runner.py`, `tests/test_runner.py`, `ARCHITECTURE.md`, `AGENTS.md`, `docs/superpowers/2026-08-23-endeavour-report.md`

- [ ] **Step 1: Failing test**: synthetic validation stage with a checkpoint dir → `validation_checkpoints/preds_batch_*.parquet` files + manifest; delete one batch → rerun → the stage's predictions frame equals the uninterrupted run's; caplog shows load+compute mix.
- [ ] **Step 2: Implement** in `_run_validation_stage(..., validation_checkpoint_dir: Path | None = None)`: replace the single `_predict_in_era_batches` call with a checkpoint-aware batched loop (same batch boundaries `_VAL_PREDICT_ERA_BATCH`; per batch: load-if-exists / predict+atomic-write; manifest verify/init at the first computed batch — device: the predict closure has no device; use `str(orchestrator.resolved_device)` — for the validation stage the predicting models were already fitted, so pass the orchestrator's resolved device in via `run()`; if unknown at that point, fall back to `"cpu"`? NO — see note: the deploy models already carry the device in the deploy manifest; the validation manifest should record the SAME device value the deploy manifest recorded. Implementation: `run()` passes `device=str(model_orchestrator.resolved_device)` down; if it is None (shouldn't be — fits ran), raise loudly). Keep the final `evaluate_model` call uncheckpointed.
- [ ] **Step 3: Identity tests** mirroring Task B (tamper manifest → ValueError; torn tree → ValueError).
- [ ] **Step 4: Docs** — `ARCHITECTURE.md` checkpoint section extended (deploy + validation layouts, same identity rules, evaluate_model out of scope); `AGENTS.md` §8: extend the existing OOF checkpoint hazard with one sentence covering deploy/validation checkpoints (file at ~31.5 KB; budget 32,768 B — verify size after edit); `docs/superpowers/2026-08-23-endeavour-report.md` open item 3 marked done with commit refs.
- [ ] **Step 5**: `./.venv/Scripts/python -m pytest tests/test_runner.py tests/test_checkpointing.py -q` → green; lint; commit: `feat(runner): validation predict-batch checkpoints + docs`.
