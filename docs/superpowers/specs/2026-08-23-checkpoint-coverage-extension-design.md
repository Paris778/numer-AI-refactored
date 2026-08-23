# Design Spec: Checkpoint Coverage Extension — Deploy Fits & Validation Predicts

> Status: APPROVED (director disposition 2026-08-23). Extends the OOF checkpoint/resume system (spec `2026-08-20-oof-checkpoint-resume-design.md`) to the two remaining uninsured stages of `ExperimentRunner.run()`: the per-target full-history deploy fits and the era-batched validation predicts. Same identity rules, same atomicity, same determinism invariant.

## 1. Mission

The `mt-std-v1` campaign spent ~66.5 h in one process: 16 CV folds (now insured), then **4 full-history deploy fits** (hours each), then the **validation scorecard stage** (~12 h of quiet compute). A crash in either uninsured stage still loses everything done in that stage and forces a full restart. This spec adds fold-style checkpoints to both stages so a resumed run skips completed work, with the existing code/device identity manifest making stale reuse impossible.

## 2. Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Deploy checkpoint unit | **Per target.** `_build_deploy_pipeline` fits one full-history model per target; persist each fitted model with `cloudpickle` to `artifacts/runs/<run_id>/deploy_checkpoints/<target>.pkl` immediately after its fit. On resume, a present + identity-valid checkpoint is loaded instead of refit. |
| 2 | Deploy checkpoint payload | The fitted model object only (the predict closure is rebuilt from the loaded model — the closure construction is cheap and deterministic given the model). |
| 3 | Validation checkpoint unit | **Per era batch.** `_run_validation_stage` already predicts via `_predict_in_era_batches`; persist each batch's prediction frame to `artifacts/runs/<run_id>/validation_checkpoints/preds_batch_<NN>.parquet`. On resume, computed batches are loaded; missing ones are predicted and written. The final `evaluate_model` scorecard call is NOT checkpointed (single call, no clean granularity) — documented as out of scope. |
| 4 | Identity | Reuse the exact OOF manifest discipline: a `manifest.json` at each checkpoint root with `code_sha256` (SHA-256 of `nmr/models.py` + `nmr/splitter.py` + `nmr/runner.py` — the runner now contributes to the staged behavior) and `device`. Written atomically at the FIRST completed unit (deploy: first fitted target; validation: first predicted batch). Mismatch / torn tree → `ValueError` with delete-to-refit guidance. |
| 5 | Shared helpers | Extract the manifest/identity/atomic-write helpers used by the OOF checkpoints into reusable functions in `nmr/_oof.py` (or a new `nmr/_checkpoint.py`): `checkpoint_manifest(root) -> dict` (code+device), `verify_or_init_manifest(root, manifest) -> None`, `write_frame_atomic`, `write_bytes_atomic`. The OOF path refactors onto them (no behavior change — its tests are the regression net). |
| 6 | Determinism invariant | Same run_id + matching manifest ⇒ a resumed deploy/validation stage produces byte-identical outputs to an uninterrupted run (pickled model predicts == fresh-fit predicts; parquet batch roundtrip is lossless). Proven by mixed resume tests (one unit deleted → refit → equality + load/fit log mix). |
| 7 | API shape | `_build_deploy_pipeline(..., deploy_checkpoint_dir: Path | None = None)` and `_run_validation_stage(..., validation_checkpoint_dir: Path | None = None)`; `ExperimentRunner.run()` passes both dirs under `artifacts/runs/<run_id>/`. Default `None` = today's behavior; research/promotion callers unchanged (promotion calls `_build_deploy_pipeline` via its own path — check call sites; it may stay checkpoint-free until adopted). |
| 8 | RAM safety | Deploy checkpoints add one pickled LightGBM (~50–200 MB per target on medium) — trivial. Validation batch frames are small per batch (already bounded by `_VAL_PREDICT_ERA_BATCH`). No new RAM ceilings. |

## 3. Tests (synthetic fixtures; `tests/test_checkpointing.py` + `tests/test_runner.py`)

1. **Deploy mixed resume**: fit the pipeline with a deploy checkpoint dir → delete one target's `.pkl` → resume → the deploy closure's validation predictions equal the uninterrupted run's, bit-for-bit; log shows one `train_full_history` and one `loaded deploy checkpoint`.
2. **Deploy identity guards**: tampered `code_sha256` / `device` in the deploy manifest → `ValueError`; `.pkl` present without manifest → torn-tree `ValueError`.
3. **Validation mixed resume**: run the validation stage with a checkpoint dir → delete one batch parquet → resume → final predictions frame equals the uninterrupted run's (the scorecard too, since inputs are identical); log shows load+compute mix.
4. **Validation identity guards**: same manifest tamper tests as OOF/deploy.
5. **Legacy paths untouched**: existing `tests/test_runner.py` and `tests/test_checkpointing.py` stay green unchanged; `None` dirs reproduce today's behavior exactly.
6. **Shared-helper refactor is behavior-neutral**: the existing OOF checkpoint tests pass unchanged after the helper extraction.

## 4. Files

- Modify: `nmr/_oof.py` (extract shared checkpoint helpers; OOF path refactored onto them), `nmr/runner.py` (`_build_deploy_pipeline` + `_run_validation_stage` + `run()` wiring), possibly `nmr/promote.py` (only if its call sites need the new keyword — check first)
- Modify: `tests/test_checkpointing.py`, `tests/test_runner.py`
- Docs (same commit): `ARCHITECTURE.md` (extend the checkpoint section), `AGENTS.md` §8 (extend the existing OOF checkpoint hazard — the file is at ~31.5 KB, budget 32,768 B), `docs/superpowers/2026-08-23-endeavour-report.md` (item 3 marked done)

## 5. Out of Scope

- Checkpointing `evaluate_model`'s scorecard computation (single call; rerun on resume).
- Checkpointing the meta/benchmark parquet loads (minutes, not hours).
- Promotion-path deploy fits (adopt later if the promotion path becomes long-running).
- `cloudpickle` trust changes: loading stays restricted to artifacts this repo produced (existing hazard).
