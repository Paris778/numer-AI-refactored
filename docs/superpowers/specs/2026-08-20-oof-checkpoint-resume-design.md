# Design Spec: OOF Fold-Checkpointing & Resume (Campaign Crash Insurance) — v2

> Status: APPROVED (director disposition 2026-08-20). v2 supersedes v1 after review: the public `train_cross_validation` API stays untouched (no `models` corruption possible), checkpoints carry a code+device identity manifest, and the determinism guarantee is tested in the only case that can break it (mixed load+fit within one target).

## 1. Mission

A campaign run is one `ExperimentRunner.run()` call containing 4 targets × 4 folds = 16 fits (measured: the `mt_std_v1` campaign was 21+ hours in with zero artifacts persisted — the registry row is written only when the whole run finishes). A crash at fit 15 loses all 16. This change persists each fold's OOF predictions to parquet **as the fold completes**, and on restart loads folds whose checkpoint exists instead of refitting. Determinism is preserved by construction — fits are seeded and the parquet roundtrip is lossless — and is **proven** by tests covering the mixed load+fit resume case.

## 2. Decisions (v2)

| # | Question | Decision |
|---|---|---|
| 1 | Granularity | **Per fold.** Max lost work on a crash = the fold in flight. |
| 2 | Storage layout | `artifacts/runs/<run_id>/oof_checkpoints/<target_col>/fold_<NN>.parquet` + one `manifest.json` at the checkpoint root. Keyed by `run_id` (config + data fingerprint). |
| 3 | API shape | **New method** `ModelOrchestrator.train_oof_with_checkpoints(df, *, feature_cols, target_col, splitter, era_col="era", checkpoint_dir: Path) -> pl.DataFrame` (OOF only, checkpoint-aware). The public `train_cross_validation` is **unchanged** — no new parameter, `CVResult.models` stays "one model per fold" and can never be silently truncated by resume. `nmr/_oof.py::train_multi_target_oof(..., checkpoint_dir: Path \| None = None)` routes to the new method when set, to `train_cross_validation` otherwise. `ExperimentRunner` passes `artifacts/runs/<run_id>/oof_checkpoints`. |
| 4 | Skip semantics | A fold part whose parquet exists is **loaded instead of fitted**. Fold-disjointness validation runs over every fold, loaded or fitted. Corrupt checkpoint → `ValueError` (fail loud, no silent refit). |
| 5 | Code identity (review blocker #3) | `manifest.json` records `code_sha256` = SHA-256 over the concatenated source bytes of `nmr/models.py` + `nmr/splitter.py` (the two modules that define fold geometry and fit behavior). On resume, a mismatch raises `ValueError` telling the operator to delete the checkpoint dir to force a full refit. Rationale: `run_id` binds config+data, not code — checkpoints must never silently survive a fitting-code change. Git SHA is rejected (meaningless on dirty trees). |
| 6 | Device identity (review blocker #7) | `manifest.json` also records `device` = `ModelOrchestrator.resolved_device` at fit time. Cross-device resume raises (determinism holds per-device only). **Manifest is written atomically at the FIRST fitted fold, never before**: `resolved_device` is `None` until a fit completes, so an earlier write would record `"None"` and the device guard would pass vacuously on resume. A checkpoint tree containing fold parts but no manifest is treated as inconsistent and raises. |
| 7 | Atomicity | Every checkpoint write serializes the frame to bytes and goes through the existing `nmr/_atomicio.py::atomic_write_bytes` (temp-file + fsync + `os.replace`). No hand-rolled writes. |
| 8 | Determinism invariant (hard) | Same run_id + same code_sha256 + same device ⇒ OOF assembled from checkpoints equals the freshly fitted OOF **bit-for-bit** (`pl.DataFrame.equals`). Tested in the mixed case (one fold loaded, one refit within the same target). |
| 9 | Retention | Checkpoints live inside the run's own artifact dir and are deleted with it. Clearing `artifacts/runs/` remains the existing ask-first operation; no separate retention machinery. |
| 10 | Docs law | `ARCHITECTURE.md` gets the module-level detail; `AGENTS.md` gets a short §8 hazard (trim an existing verbose bullet to stay within the 32 KB budget — "trim before you grow", not "skip the update"). |
| 11 | CI gate | Local runs during the campaign are targeted-only (CPU-bound box). The FULL suite + CI must be green before merge; stated explicitly in the plan. |

## 3. Behavior Contract

- First run: fits every fold, writing `fold_NN.parquet` after each; `manifest.json` is written with the current `code_sha256` + `device` before the first fold write.
- Resume run (same run_id): manifest must match current code + device, else `ValueError`. Matching manifest → load existing parts, fit missing ones only. OOF byte-identical to the first run.
- Idempotency: repeated runs with all parts present perform zero fits.
- Concurrent duplicate run_ids are unsupported (operational anti-pattern; unchanged).

## 4. Tests (synthetic fixtures only)

New `tests/test_checkpointing.py`:

1. **Mixed resume is bit-for-bit** (the v1 gap): fit all → delete exactly ONE fold parquet within one target → resume. Assert `fresh.equals(resumed)` AND the log shows both `"loaded from checkpoint"` and `"trained in"` for that target. This is the only scenario that can break determinism; it must pass.
2. **All-loaded resume equals fresh** (roundtrip proof) + zero fits on second pass (caplog).
3. **Partial target resume**: delete one target's directory via `shutil.rmtree` → refit that target only → equals fresh.
4. **Code mismatch raises**: tamper `manifest.json`'s `code_sha256` → resume raises `ValueError` (message names the manifest path and says to delete the dir to refit).
5. **Device mismatch raises**: tamper `device` → same.
6. **Corrupt checkpoint raises**: overwrite a fold parquet with garbage → `ValueError`, no silent refit.
7. **Atomic writes**: after a run, the checkpoint tree contains only `manifest.json` + `fold_NN.parquet` files (no temp files).
8. **Legacy path untouched**: `train_cross_validation` without checkpoints behaves exactly as before; `CVResult.models` still contains one model per fold (existing `tests/test_models.py` covers this — must stay green unchanged).

Runner wiring test in `tests/test_runner.py`: synthetic run creates `runs/<run_id>/oof_checkpoints/`; a second synthetic run with the same config+data (same run_id) loads instead of fits (caplog).

## 5. Files

- Modify: `nmr/models.py` (new `train_oof_with_checkpoints` + shared private fold-loop helper; `train_cross_validation` refactored to delegate internally with checkpoint_dir=None so the OOF path stays single-sourced)
- Modify: `nmr/_oof.py` (thread `checkpoint_dir`)
- Modify: `nmr/runner.py` (pass the run-scoped checkpoint dir)
- Create: `tests/test_checkpointing.py`; Modify: `tests/test_runner.py`
- Docs (same commit): `ARCHITECTURE.md` + `AGENTS.md` (§8 hazard, with a compensating trim) + handoff doc §8 item 1 marked done

## 6. Out of Scope (explicit)

- Deploy/full-history fit, ensemble weights, neutralization, evaluation stages.
- Cross-process locking; concurrent duplicate run_ids.
- Thread caps (separate work item).
