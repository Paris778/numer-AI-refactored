# OOF Checkpointing & Resume — Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> v2 fixes the v1 review: `CVResult.models` can no longer be corrupted (public API untouched), checkpoints carry a code+device identity manifest, the mixed load+fit determinism case is tested, the defective tests are corrected, AGENTS.md is updated (with a trim, not skipped), and CI is named as the merge gate.

**Goal:** Fold-granularity OOF checkpointing + skip-on-resume inside `ExperimentRunner` runs, with a bit-for-bit resumed-equals-fresh guarantee proven in the mixed load+fit case, and code/device identity so checkpoints can never silently survive a code or device change.

**Architecture:** New `ModelOrchestrator.train_oof_with_checkpoints(...) -> pl.DataFrame` (OOF-only, checkpoint-aware) shares a private fold loop with the untouched `train_cross_validation`. `nmr/_oof.py::train_multi_target_oof(..., checkpoint_dir=None)` routes to it when set. Each fold part is written atomically (existing `atomic_write_bytes`) under `artifacts/runs/<run_id>/oof_checkpoints/<target>/fold_NN.parquet`; a root `manifest.json` records `code_sha256` (SHA-256 of `nmr/models.py` + `nmr/splitter.py` source bytes) and `device` (the orchestrator's resolved device); any mismatch on resume raises.

**Spec:** `docs/superpowers/specs/2026-08-20-oof-checkpoint-resume-design.md` (v2 — authority).

**Tech Stack:** Python 3.11+, Polars, pytest + ruff (E/F/I/UP @120).

## Global Constraints

- All business logic in `nmr/`; tests in `tests/`.
- Determinism: same run_id + code_sha256 + device ⇒ resumed OOF equals fresh OOF bit-for-bit; the MIXED case (one fold loaded + one refit in the same target) is the required proof.
- Atomic writes: `nmr/_atomicio.py::atomic_write_bytes` ONLY (serialize the frame to bytes first — see Task A). Never `write_parquet` directly to the final path.
- Fail loudly: corrupt checkpoint, code mismatch, or device mismatch ⇒ `ValueError`; no silent refit.
- Public API safety: `train_cross_validation` signature and `CVResult` semantics unchanged; `CVResult.models` always one model per fold.
- Test commands: `./.venv/Scripts/python -m pytest ...`; lint `./.venv/Scripts/python -m ruff check ...`.
- **CPU discipline (campaign running)**: run ONLY the targeted files per task. The FULL suite and CI are the merge gate — this work must not be considered done until CI is green; state this honestly in every report.
- **AGENTS.md**: update required by §2.8 — add the hazard AND trim an existing verbose bullet to stay under 32 KB (see Task C).

---

### Task A: Checkpoint core — `train_oof_with_checkpoints` + manifest + tests

**Files:**
- Modify: `nmr/models.py`
- Modify: `nmr/_oof.py`
- Test: `tests/test_checkpointing.py` (new)

**Interfaces:**
- Consumes: `nmr/_atomicio.py` (`atomic_write_bytes` — read first), `CVResult`, `PurgedEraSplitter`, `ModelOrchestrator.resolved_device` (read `nmr/models.py` for the exact attribute)
- Produces:
  - `ModelOrchestrator._cv_fold_parts(df, *, feature_cols, target_col, splitter, era_col="era", checkpoint_dir: Path | None = None) -> tuple[list[object | None], list[pl.DataFrame]]` — private shared fold loop (see code). `None` models for loaded folds.
  - `ModelOrchestrator.train_cross_validation(df, *, feature_cols, target_col, splitter, era_col="era") -> CVResult` — refactored to delegate to `_cv_fold_parts(checkpoint_dir=None)`; asserts no `None` in models (defensive; unreachable today); behavior byte-identical to today.
  - `ModelOrchestrator.train_oof_with_checkpoints(df, *, feature_cols, target_col, splitter, era_col="era", checkpoint_dir: Path) -> pl.DataFrame` — checkpoint-aware OOF-only path; returns the concatenated OOF; never exposes models.
  - `nmr/_oof.py::train_multi_target_oof(modeler, df, *, feature_cols, splitter, targets, checkpoint_dir: Path | None = None) -> pl.DataFrame` — routes per target to `train_oof_with_checkpoints` when `checkpoint_dir` is set, else `train_cross_validation` (legacy path untouched).

- [ ] **Step 1: Write failing tests** — create `tests/test_checkpointing.py`. READ `tests/test_models.py` FIRST and reuse its existing synthetic config/splitter/modeler helpers verbatim (do NOT invent APIs). Adapt the helper names below to whatever that file actually provides.

```python
"""OOF fold-checkpointing & resume contracts (spec 2026-08-20-oof-checkpoint-resume v2)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from nmr._oof import train_multi_target_oof


# _synthetic_train / _modeler / _splitter: reuse the construction helpers from
# tests/test_models.py verbatim (read that file; adapt names below).


def _run(ckpt: Path | None, train: pl.DataFrame) -> pl.DataFrame:
    return train_multi_target_oof(
        _modeler(), train, feature_cols=["f1", "f2"],
        splitter=_splitter(), targets=["target", "target_ender_20"],
        checkpoint_dir=ckpt,
    )


def test_all_loaded_resume_equals_fresh_and_fits_nothing(tmp_path, caplog):
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    fresh = _run(ckpt, train)
    caplog.clear()
    with caplog.at_level("INFO", logger="nmr.models"):
        resumed = _run(ckpt, train)
    assert fresh.equals(resumed)
    assert "loaded from checkpoint" in caplog.text
    assert "trained in" not in caplog.text


def test_mixed_resume_within_target_is_bit_for_bit(tmp_path, caplog):
    """The only case that can break determinism: fold loaded + fold refit."""
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    fresh = _run(ckpt, train)
    parts = sorted((ckpt / "target").glob("fold_*.parquet"))
    assert len(parts) >= 2
    parts[0].unlink()  # delete exactly ONE fold within one target
    caplog.clear()
    with caplog.at_level("INFO", logger="nmr.models"):
        resumed = _run(ckpt, train)
    assert fresh.equals(resumed)
    assert "loaded from checkpoint" in caplog.text
    assert "trained in" in caplog.text  # the refit actually happened


def test_partial_target_resume_refits_only_missing_target(tmp_path):
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    fresh = _run(ckpt, train)
    shutil.rmtree(ckpt / "target_ender_20")  # rmtree, NOT unlink (directory)
    resumed = _run(ckpt, train)
    assert fresh.equals(resumed)


def test_code_mismatch_raises(tmp_path):
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    _run(ckpt, train)
    manifest_path = ckpt / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["code_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="code_sha256"):
        _run(ckpt, train)


def test_device_mismatch_raises(tmp_path):
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    _run(ckpt, train)
    manifest_path = ckpt / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["device"] = "totally_different_device"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="device"):
        _run(ckpt, train)


def test_corrupt_checkpoint_raises(tmp_path):
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    _run(ckpt, train)
    parts = sorted((ckpt / "target").glob("fold_*.parquet"))
    parts[0].write_bytes(b"garbage")
    with pytest.raises(ValueError, match="corrupt OOF checkpoint"):
        _run(ckpt, train)


def test_checkpoint_tree_contains_no_temp_files(tmp_path):
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    _run(ckpt, train)
    all_files = sorted(p.name for p in ckpt.rglob("*") if p.is_file())
    for name in all_files:
        assert name == "manifest.json" or (
            name.startswith("fold_") and name.endswith(".parquet")
        ), f"unexpected file in checkpoint tree: {name}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_checkpointing.py -q`
Expected: FAIL — `TypeError: train_multi_target_oof() got an unexpected keyword argument 'checkpoint_dir'`.

- [ ] **Step 3: Implement** — in `nmr/models.py`:

```python
    def _cv_fold_parts(
        self,
        df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        target_col: str,
        splitter: PurgedEraSplitter,
        era_col: str = "era",
        checkpoint_dir: Path | None = None,
    ) -> tuple[list[object | None], list[pl.DataFrame]]:
        """Shared fold loop: fit or load each fold; (models, oof_parts).

        ``checkpoint_dir=None`` = the legacy fit-everything path (models has no
        None entries). With a checkpoint dir, existing fold parquets are loaded
        (models entry None — the OOF-only caller discards models) and new folds
        are fitted then atomically persisted. The checkpoint root carries a
        manifest.json with code+device identity; a mismatch raises.
        """
        folds = splitter.split(df.get_column(era_col).to_list())
        logger.info("[train_cross_validation] %s: %d folds", target_col, len(folds))
        models: list[object | None] = []
        oof_parts: list[pl.DataFrame] = []
        seen_val_eras: set[str] = set()

        manifest_path = checkpoint_dir / "manifest.json" if checkpoint_dir else None
        if manifest_path is not None:
            expected_manifest = _checkpoint_manifest()
            if manifest_path.exists():
                stored = json.loads(manifest_path.read_text(encoding="utf-8"))
                if stored.get("code_sha256") != expected_manifest["code_sha256"]:
                    raise ValueError(
                        "OOF checkpoint code mismatch: fitting code changed since "
                        f"the checkpoints were written ({manifest_path}). "
                        "Delete the oof_checkpoints directory to force a full refit."
                    )
                if stored.get("device") != expected_manifest["device"]:
                    raise ValueError(
                        "OOF checkpoint device mismatch: checkpoints were fitted "
                        f"on device {stored.get('device')!r}, current device is "
                        f"{expected_manifest['device']!r}. Delete the "
                        "oof_checkpoints directory to force a full refit."
                    )
            else:
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(
                    manifest_path,
                    json.dumps(expected_manifest, sort_keys=True).encode("utf-8"),
                )

        for fold in folds:
            overlap = seen_val_eras & set(fold.val_eras)
            if overlap:
                raise ValueError(
                    f"Validation eras must be disjoint across folds, got {sorted(overlap)}"
                )
            part_path = (
                checkpoint_dir / target_col / f"fold_{fold.index + 1:02d}.parquet"
                if checkpoint_dir is not None else None
            )
            if part_path is not None and part_path.exists():
                try:
                    fold_predictions = pl.read_parquet(part_path)
                except Exception as exc:
                    raise ValueError(
                        f"corrupt OOF checkpoint {part_path}: {exc}"
                    ) from exc
                models.append(None)
                logger.info(
                    "[train_cross_validation] %s: fold %d/%d loaded from checkpoint %s",
                    target_col, fold.index + 1, len(folds), part_path,
                )
            else:
                logger.info(
                    "[train_cross_validation] %s: fold %d/%d train_eras=%d val_eras=%d",
                    target_col, fold.index + 1, len(folds),
                    len(fold.train_eras), len(fold.val_eras),
                )
                t0 = time.time()
                model, fold_predictions = self._fit_predict_fold(
                    df, fold=fold, feature_cols=feature_cols,
                    target_col=target_col, era_col=era_col,
                    purge_eras=splitter.purge_eras,
                )
                logger.info(
                    "[train_cross_validation] %s: fold %d/%d trained in %.1fs",
                    target_col, fold.index + 1, len(folds), time.time() - t0,
                )
                models.append(model)
                if part_path is not None:
                    part_path.parent.mkdir(parents=True, exist_ok=True)
                    _write_frame_atomic(fold_predictions, part_path)
            oof_parts.append(fold_predictions)
            seen_val_eras.update(fold.val_eras)

        if not oof_parts:
            raise ValueError("No folds produced OOF predictions")
        return models, oof_parts
```

With two module-level helpers in `nmr/models.py` (or better: the manifest + frame-write helpers live in `nmr/_oof.py` — the OOF module owns OOF persistence; `nmr/models.py` imports them. Choose `nmr/_oof.py` to keep models.py lean):

In `nmr/_oof.py`:

```python
import hashlib
import io
import json
from pathlib import Path

from nmr._atomicio import atomic_write_bytes

_CODE_IDENTITY_FILES = ("nmr/models.py", "nmr/splitter.py")


def checkpoint_manifest() -> dict[str, str]:
    """Current code+device identity for OOF checkpoint manifests."""
    digest = hashlib.sha256()
    for relative in _CODE_IDENTITY_FILES:
        path = Path(__file__).resolve().parents[1] / relative
        digest.update(path.read_bytes())
    return {
        "code_sha256": digest.hexdigest(),
        "device": _current_device(),
    }


def _current_device() -> str:
    from nmr.models import ModelOrchestrator  # local import: avoid cycles
    return str(getattr(ModelOrchestrator, "resolved_device", "cpu"))
```

Hmm — device must be the ACTUAL orchestrator instance's resolved device, not a class attribute. Correct wiring: the fold loop needs the device from `self`. `ModelOrchestrator` has `resolved_device` as an instance property (runner.py:300 reads `model_orchestrator.resolved_device`). So `_cv_fold_parts` builds the manifest itself using `self.resolved_device`:

```python
        if manifest_path is not None:
            expected_manifest = {
                "code_sha256": _fitting_code_sha256(),
                "device": str(self.resolved_device),
            }
```

with `_fitting_code_sha256()` in `nmr/_oof.py` hashing the two module files. The implementer adapts the exact attribute (`resolved_device` — verify by reading `nmr/models.py`; if it is computed lazily after a fit, the manifest must be written only when the first fold is fitted — the manifest write block above runs before the loop, so if the device is only known after the first fit, move manifest creation to the first fitted-fold write; if `resolved_device` is available up front (runner reads it right after orchestrator construction), keep it as written).

Then:

```python
    def train_cross_validation(
        self,
        df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        target_col: str,
        splitter: PurgedEraSplitter,
        era_col: str = "era",
    ) -> CVResult:
        models, oof_parts = self._cv_fold_parts(
            df, feature_cols=feature_cols, target_col=target_col,
            splitter=splitter, era_col=era_col, checkpoint_dir=None,
        )
        if any(m is None for m in models):  # unreachable defensive guard
            raise ValueError("checkpoint-less CV produced a None model entry")
        oof = pl.concat(oof_parts, how="vertical")
        logger.info(
            "[train_cross_validation] %s: OOF complete rows=%d", target_col, oof.height
        )
        return CVResult(oof=oof, models=tuple(models))

    def train_oof_with_checkpoints(
        self,
        df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        target_col: str,
        splitter: PurgedEraSplitter,
        era_col: str = "era",
        checkpoint_dir: Path,
    ) -> pl.DataFrame:
        """Checkpoint-aware OOF training; returns OOF only (models discarded).

        See the checkpoint spec (2026-08-20-oof-checkpoint-resume) for the
        resume contract and code/device identity rules.
        """
        _, oof_parts = self._cv_fold_parts(
            df, feature_cols=feature_cols, target_col=target_col,
            splitter=splitter, era_col=era_col, checkpoint_dir=checkpoint_dir,
        )
        oof = pl.concat(oof_parts, how="vertical")
        logger.info(
            "[train_oof_with_checkpoints] %s: OOF complete rows=%d", target_col, oof.height
        )
        return oof
```

In `nmr/_oof.py::train_multi_target_oof` — new parameter + routing:

```python
def train_multi_target_oof(
    modeler: ModelOrchestrator,
    df: pl.DataFrame,
    *,
    feature_cols: Sequence[str],
    splitter: PurgedEraSplitter,
    targets: Sequence[str],
    checkpoint_dir: Path | None = None,
) -> pl.DataFrame:
    """... existing docstring ... + checkpoint_dir routes to the
    checkpoint-aware OOF-only path (spec 2026-08-20-oof-checkpoint-resume)."""
    stacked: pl.DataFrame | None = None
    for target in targets:
        if checkpoint_dir is not None:
            part = modeler.train_oof_with_checkpoints(
                df, feature_cols=feature_cols, target_col=target,
                splitter=splitter, era_col="era", checkpoint_dir=checkpoint_dir,
            )
        else:
            result = modeler.train_cross_validation(
                df, feature_cols=feature_cols, target_col=target,
                splitter=splitter, era_col="era",
            )
            part = result.oof
        part = part.rename({"prediction": f"pred_{target}"})
        if stacked is None:
            stacked = part
        else:
            stacked = stacked.join(part, on=["id", "era"], how="inner")
    assert stacked is not None
    return stacked
```

`_write_frame_atomic(frame, path)`: serialize to bytes then reuse the existing helper:

```python
def _write_frame_atomic(frame: pl.DataFrame, path: Path) -> None:
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    atomic_write_bytes(path, buffer.getvalue())
```

(The implementer places this in `nmr/_oof.py` and imports it in models.py, or inlines the BytesIO pattern — either way `atomic_write_bytes` is the only writer.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_checkpointing.py tests/test_models.py -q`
Expected: PASS — new contract tests AND the untouched legacy model tests (proves the public API refactor is behavior-identical).

- [ ] **Step 5: Lint + commit**

Run: `./.venv/Scripts/python -m ruff check nmr/models.py nmr/_oof.py tests/test_checkpointing.py` → clean, then:

```bash
git add nmr/models.py nmr/_oof.py tests/test_checkpointing.py
git commit -m "feat(runner): fold-granularity OOF checkpointing with code/device identity and skip-on-resume"
```

---

### Task B: Runner wiring

**Files:**
- Modify: `nmr/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: Task A's `train_multi_target_oof(..., checkpoint_dir=...)`; `self._run_id` and `self._config.run.artifacts_dir` (both already exist).

- [ ] **Step 1: Write failing test** — append to `tests/test_runner.py` (read the file first; reuse its synthetic config/run helpers):

```python
def test_runner_writes_and_reuses_oof_checkpoints(tmp_path, caplog):
    result1 = run_synthetic_experiment(tmp_path)  # existing helper, adapted for caplog
    ckpt_root = (
        tmp_path / "artifacts" / "runs" / result1.run_id / "oof_checkpoints"
    )
    assert (ckpt_root / "manifest.json").exists()
    assert sorted(p.name for p in (ckpt_root / "target").glob("fold_*.parquet"))
    caplog.clear()
    with caplog.at_level("INFO", logger="nmr.models"):
        result2 = run_synthetic_experiment(tmp_path)  # same config+data -> same run_id
    assert result2.oof.equals(result1.oof)
    assert "loaded from checkpoint" in caplog.text
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_runner.py -k oof_checkpoints -q`
Expected: FAIL — no checkpoint dir created.

- [ ] **Step 3: Implement** — in `nmr/runner.py::run()`:

```python
        cv_oof = self._train_multi_target_oof(
            train_df,
            feature_cols=feature_cols,
            splitter=splitter,
            model_orchestrator=model_orchestrator,
            checkpoint_dir=(
                self._config.run.artifacts_dir
                / "runs"
                / self._run_id
                / "oof_checkpoints"
            ),
        )
```

And in `ExperimentRunner._train_multi_target_oof`, add `checkpoint_dir: Path | None = None` to the signature and pass it to `train_multi_target_oof(...)`.

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_runner.py -q`
Expected: PASS (new test + all pre-existing runner tests).

- [ ] **Step 5: Lint + commit**

Run: `./.venv/Scripts/python -m ruff check nmr/runner.py tests/test_runner.py` → clean, then:

```bash
git add nmr/runner.py tests/test_runner.py
git commit -m "feat(runner): wire run-scoped OOF checkpoint directory"
```

---

### Task C: Docs + merge gate

**Files:**
- Modify: `ARCHITECTURE.md` (OOF path section: checkpoint layout, manifest identity rules, resume semantics, atomic-write rule, retention = deleted with the run dir, "concurrent duplicate run_ids unsupported")
- Modify: `AGENTS.md` — §8 hazard addition (required by §2.8), with a compensating trim to stay under the 32,768 B budget. Suggested trim: condense the verbose "Coverage specs must be package-level" bullet to one sentence. Hazard text:

```markdown
### OOF fold checkpoints (2026-08-20)
`ExperimentRunner` persists per-fold OOF parts under `artifacts/runs/<run_id>/oof_checkpoints/<target>/fold_NN.parquet` + a `manifest.json` recording code identity (SHA-256 of `nmr/models.py` + `nmr/splitter.py`) and fit device. Resume loads existing folds; any code/device mismatch raises — delete the directory to force a full refit (never silently reuse stale OOF). Checkpoints are deleted with their run dir; clearing `artifacts/runs/` remains ask-first.
```

- Modify: `docs/superpowers/2026-08-19-benchmark-fleet-handoff.md` (§8 queued-work item 1: mark DONE with the commit refs)

- [ ] **Step 1: Docs edits as above.** Verify AGENTS.md size: `stat -c %s AGENTS.md` must stay ≤ 32,768.

- [ ] **Step 2: Targeted verification**

Run: `./.venv/Scripts/python -m pytest tests/test_checkpointing.py tests/test_models.py tests/test_runner.py tests/test_campaign.py -q`
Expected: PASS.

- [ ] **Step 3: Merge gate — the full suite and CI**

The campaign owns the CPU, so the FULL suite run is deferred to CI. This work is NOT done until `ruff check .` and the full `pytest -q` are green in CI (`.github/workflows/ci.yml`). State this explicitly in the final report. If the campaign finishes first, run the full suite locally before merging.

- [ ] **Step 4: Commit**

```bash
git add ARCHITECTURE.md AGENTS.md docs/superpowers/2026-08-19-benchmark-fleet-handoff.md
git commit -m "docs: OOF checkpoint/resume SSOT updates (architecture, agents hazard, handoff)"
```
