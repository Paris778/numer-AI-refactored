# Design Spec: OOF Fold-Checkpointing & Resume (Campaign Crash Insurance)

> Status: APPROVED (director disposition 2026-08-20). Scope: fold-granularity incremental persistence of cross-validation OOF predictions with skip-on-resume, inside `ExperimentRunner` runs. Nothing else in the run lifecycle changes.

## 1. Mission

A campaign run is one `ExperimentRunner.run()` call containing 4 targets × 4 folds = 16 fits (measured: the running `mt_std_v1` campaign at 2026-08-20 is ~21+ hours in with zero artifacts persisted — the registry row is written only when the whole run finishes). A crash at fit 15 loses all 16. This change persists each fold's OOF predictions to parquet **as the fold completes**, and on restart skips folds whose checkpoint already exists. Determinism is preserved by construction: fits are seeded, and the checkpoint roundtrip is lossless float32 parquet — a resumed run must reproduce the uninterrupted run bit-for-bit, which makes the resume logic self-testing.

## 2. Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Granularity | **Per fold.** `ModelOrchestrator.train_cross_validation` already collects per-fold `oof_parts`; each part is persisted immediately after its fold completes. Max lost work on a crash = the fold in flight. |
| 2 | Storage layout | `artifacts/runs/<run_id>/oof_checkpoints/<target_col>/fold_<NN>.parquet` (zero-padded 2-digit fold index). Keyed by `run_id` (which already binds config + data fingerprint) → stale-reuse is impossible by construction; different inputs never share a checkpoint dir. |
| 3 | Skip semantics | A fold part whose parquet exists is **loaded instead of fitted**. Fold-disjointness validation (`seen_val_eras`) still runs over every fold, loaded or fitted. A corrupt/unreadable checkpoint raises (fail loud, no silent fallback to refit). |
| 4 | What is NOT checkpointed | Fold models (the CV `models` tuple is unused downstream in the runner — the deploy pipeline re-fits full history independently), the deploy fit, blending/neutralization/evaluation (cheap, re-runnable). |
| 5 | API shape | `train_cross_validation(..., checkpoint_dir: Path \| None = None)` and `train_multi_target_oof(..., checkpoint_dir: Path \| None = None)`. Default `None` = exactly today's behavior; research/HPO callers unchanged. `ExperimentRunner` passes `self._config.run.artifacts_dir / "runs" / self._run_id / "oof_checkpoints"`. |
| 6 | Atomicity | Every checkpoint write goes through the existing `nmr/_atomicio.py` temp-file + fsync + `os.replace` pattern (same discipline as registry writes). |
| 7 | Determinism invariant (hard) | For the same run_id, OOF assembled from checkpoints must equal the freshly fitted OOF **bit-for-bit** (`pl.DataFrame.equals`). Enforced by tests; a violation is a correctness bug, not a tuning detail. |

## 3. Behavior Contract

- First run: fits every fold, writing `fold_NN.parquet` after each. OOF = concat of fitted parts (unchanged output).
- Resume run (same run_id): loads existing parts, fits only missing ones, writes them. OOF byte-identical to the first run.
- Idempotency: running the same run_id repeatedly is safe — all parts exist → zero fits.
- Campaign interplay: unchanged (`campaign.py` still skips configs whose run_id is recorded; a crashed campaign config now resumes its remaining folds instead of refitting from zero).
- Logging: `[train_cross_validation] <target>: fold N/M loaded from checkpoint <path>` vs `... trained in X.Xs`.

## 4. Tests (synthetic fixtures only — no real-data dependency)

New `tests/test_checkpointing.py`:

1. **Resume equals fresh**: fit with `checkpoint_dir` set → capture OOF A. New orchestrator instance, same dir → all folds load → OOF B. Assert `A.equals(B)` (bit-for-bit) and that zero fits happened (caplog: no "trained in" lines, all "loaded from checkpoint").
2. **Partial resume**: delete one target's fold parts → that target refits, the other loads → assembled OOF still equals A; log shows the mix.
3. **Disjointness still enforced**: a synthetic splitter with overlapping val eras raises even when parts are loaded (validation runs before the load/skip decision).
4. **Atomic write**: checkpoint files appear only after each fold completes (no partial file); a second concurrent writer cannot corrupt (reuse the `_atomicio` helper's own tests as evidence; assert the helper is the only write path).
5. **Corrupt checkpoint raises**: write garbage to a fold parquet → resume raises `ValueError` (no silent refit).
6. **Default None = legacy**: `train_cross_validation` without `checkpoint_dir` behaves exactly as before (existing `tests/test_models.py` CV tests stay green, unchanged).
7. **Runner wiring**: `tests/test_runner.py` synthetic run with `artifacts_dir` → checkpoint dir created under `runs/<run_id>/oof_checkpoints/`; a second synthetic run with the same config+data (same run_id) loads instead of fits (log assertion).

## 5. Files

- Modify: `nmr/models.py` (`train_cross_validation` + `train_multi_target_oof` wrapper signature — `_oof.py` passes it through), `nmr/_oof.py` (thread `checkpoint_dir`), `nmr/runner.py` (pass the run-scoped checkpoint dir)
- Create: `tests/test_checkpointing.py`
- Modify: `tests/test_runner.py` (runner wiring test)
- Docs (same commit): `ARCHITECTURE.md` (OOF path + checkpoint layout + resume semantics). No AGENTS.md changes (49 B headroom — none available).

## 6. Out of Scope (explicit)

- Checkpointing the deploy/full-history fit, ensemble weights, neutralization, or evaluation stages.
- Campaign-level restarts (already covered by run-id skipping).
- Cross-process locking for two concurrent identical run_ids (an operational anti-pattern already; documented in the spec that concurrent duplicate runs are unsupported).
- Thread caps (separate work item).
