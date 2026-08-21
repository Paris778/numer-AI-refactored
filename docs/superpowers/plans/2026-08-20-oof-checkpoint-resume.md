# OOF Checkpointing & Resume — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold-granularity OOF checkpointing + skip-on-resume inside `ExperimentRunner` runs, with a bit-for-bit resumed-equals-fresh determinism guarantee.

**Architecture:** Thread a keyword-only `checkpoint_dir: Path | None = None` through `ModelOrchestrator.train_cross_validation` → `nmr/_oof.py::train_multi_target_oof` → `ExperimentRunner._train_multi_target_oof` (runner passes `artifacts/runs/<run_id>/oof_checkpoints`). Each fold part is atomically written (existing `nmr/_atomicio.py` pattern) immediately after its fit; on entry a fold whose parquet exists is loaded instead of fitted. Default `None` preserves today's behavior for research/HPO callers.

**Spec:** `docs/superpowers/specs/2026-08-20-oof-checkpoint-resume-design.md`

**Tech Stack:** Python 3.11+, Polars, pytest + ruff (E/F/I/UP @120).

## Global Constraints

- All business logic in `nmr/`; tests in `tests/`; no logic in scripts.
- Determinism is sacred: same run_id + data + seeds ⇒ same OOF bit-for-bit, resumed or fresh. Checkpoint keying = run_id only (config + data fingerprint already inside it).
- Atomic writes: temp-file + fsync + `os.replace` via `nmr/_atomicio.py` (read it first — reuse its exact API, do not hand-roll).
- Fail loudly: corrupt checkpoint → ValueError; no silent refit.
- Default-off: `checkpoint_dir=None` = today's exact behavior; existing tests must stay green unchanged.
- Test commands: `./.venv/Scripts/python -m pytest ...`; lint `./.venv/Scripts/python -m ruff check ...`.
- **CPU discipline (campaign is running)**: run ONLY the targeted test files listed per task. Do NOT run the full suite; do NOT run real-data tests. Report this honestly.
- No AGENTS.md edits (49 B headroom).

---

### Task A: Checkpoint core in `train_cross_validation` + `train_multi_target_oof`

**Files:**
- Modify: `nmr/models.py` (`train_cross_validation`, ~line 232)
- Modify: `nmr/_oof.py`
- Test: `tests/test_checkpointing.py` (new)

**Interfaces:**
- Consumes: `nmr/_atomicio.py` (read it first — use its write helper verbatim), `PurgedEraSplitter`, existing `CVResult`
- Produces:
  - `ModelOrchestrator.train_cross_validation(df, *, feature_cols, target_col, splitter, era_col="era", checkpoint_dir: Path | None = None) -> CVResult` — inside the fold loop: `part_path = checkpoint_dir / target_col / f"fold_{fold.index + 1:02d}.parquet"`; if exists → `pl.read_parquet(part_path)` + log `"fold %d/%d loaded from checkpoint %s"`; else fit via `_fit_predict_fold`, then atomic-write the fold parquet (create parent dirs first) and log `"fold %d/%d trained in %.1fs"` (keep the existing log line). Fold-disjointness check runs for every fold before the load/skip decision. Corrupt/unreadable checkpoint: let the read error surface, or wrap in `ValueError(f"corrupt OOF checkpoint {part_path}: {exc}")` — fail loud.
  - `nmr/_oof.py::train_multi_target_oof(modeler, df, *, feature_cols, splitter, targets, checkpoint_dir: Path | None = None)` — pass `checkpoint_dir` through to `train_cross_validation` (per-target subdirs come from inside).

- [ ] **Step 1: Write failing tests** — create `tests/test_checkpointing.py` (synthetic fixtures; reuse the synthetic train-frame pattern from `tests/test_models.py` if one exists — read that file first):

```python
"""OOF fold-checkpointing & resume contracts (spec 2026-08-20-oof-checkpoint-resume)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nmr._oof import train_multi_target_oof
from nmr.config import ExperimentConfig, load_config
from nmr.models import ModelOrchestrator
from nmr.splitter import PurgedEraSplitter


def _synthetic_train(n_eras: int = 30, rows: int = 6, seed: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows_out = []
    for era in range(1, n_eras + 1):
        signal = float(era % 5)
        for i in range(rows):
            rows_out.append({
                "era": f"{era:04d}", "id": f"t{era}_{i}",
                "f1": signal + rng.normal(0, 0.5),
                "f2": rng.normal(0, 1),
                "target": signal + rng.normal(0, 0.5),
                "target_ender_20": signal * 0.5 + rng.normal(0, 0.5),
            })
    return pl.DataFrame(rows_out)


def _modeler() -> ModelOrchestrator:
    # small fast LGBM config; adapt to the repo's ModelConfig construction
    # pattern used by tests/test_models.py (read that file first).
    ...  # implementer: build ModelOrchestrator with preset="fast",
    ...  # params={"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2, "num_leaves": 4}
    ...  # via the same helper the existing model tests use.


def _splitter() -> PurgedEraSplitter:
    # mirrors the repo's synthetic splitter helper from tests/test_models.py
    ...


def test_resume_equals_fresh_bit_for_bit(tmp_path):
    df = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    fresh = train_multi_target_oof(
        _modeler(), df, feature_cols=["f1", "f2"],
        splitter=_splitter(), targets=["target", "target_ender_20"],
        checkpoint_dir=ckpt,
    )
    resumed = train_multi_target_oof(
        _modeler(), df, feature_cols=["f1", "f2"],
        splitter=_splitter(), targets=["target", "target_ender_20"],
        checkpoint_dir=ckpt,
    )
    assert fresh.equals(resumed)


def test_partial_resume_refits_only_missing(tmp_path):
    df = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    fresh = train_multi_target_oof(
        _modeler(), df, feature_cols=["f1", "f2"],
        splitter=_splitter(), targets=["target", "target_ender_20"],
        checkpoint_dir=ckpt,
    )
    (ckpt / "target_ender_20").unlink(missing_ok=True)  # force refit of one target
    resumed = train_multi_target_oof(
        _modeler(), df, feature_cols=["f1", "f2"],
        splitter=_splitter(), targets=["target", "target_ender_20"],
        checkpoint_dir=ckpt,
    )
    assert fresh.equals(resumed)


def test_resume_loads_without_fitting(tmp_path, caplog):
    df = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    train_multi_target_oof(
        _modeler(), df, feature_cols=["f1", "f2"],
        splitter=_splitter(), targets=["target"], checkpoint_dir=ckpt,
    )
    caplog.clear()
    with caplog.at_level("INFO", logger="nmr.models"):
        train_multi_target_oof(
            _modeler(), df, feature_cols=["f1", "f2"],
            splitter=_splitter(), targets=["target"], checkpoint_dir=ckpt,
        )
    text = caplog.text
    assert "loaded from checkpoint" in text
    assert "trained in" not in text


def test_corrupt_checkpoint_raises(tmp_path):
    df = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    train_multi_target_oof(
        _modeler(), df, feature_cols=["f1", "f2"],
        splitter=_splitter(), targets=["target"], checkpoint_dir=ckpt,
    )
    parts = list((ckpt / "target").glob("fold_*.parquet"))
    assert parts
    parts[0].write_bytes(b"garbage")
    with pytest.raises(ValueError, match="corrupt OOF checkpoint"):
        train_multi_target_oof(
            _modeler(), df, feature_cols=["f1", "f2"],
            splitter=_splitter(), targets=["target"], checkpoint_dir=ckpt,
        )


def test_default_none_preserves_legacy(tmp_path):
    df = _synthetic_train()
    a = train_multi_target_oof(
        _modeler(), df, feature_cols=["f1", "f2"],
        splitter=_splitter(), targets=["target"],
    )
    b = train_multi_target_oof(
        _modeler(), df, feature_cols=["f1", "f2"],
        splitter=_splitter(), targets=["target"],
    )
    assert a.equals(b)
    assert not (tmp_path / "ckpt").exists()
```

Note: the `_modeler`/`_splitter` helpers marked `...` must be filled by reading `tests/test_models.py` and reusing the exact construction helpers it already has (no invented config APIs).

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_checkpointing.py -q`
Expected: FAIL — `TypeError: train_multi_target_oof() got an unexpected keyword argument 'checkpoint_dir'`.

- [ ] **Step 3: Implement** — in `nmr/models.py`, change the signature and fold loop of `train_cross_validation`:

```python
    def train_cross_validation(
        self,
        df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        target_col: str,
        splitter: PurgedEraSplitter,
        era_col: str = "era",
        checkpoint_dir: Path | None = None,
    ) -> CVResult:
        folds = splitter.split(df.get_column(era_col).to_list())
        logger.info("[train_cross_validation] %s: %d folds", target_col, len(folds))
        models: list[object] = []
        oof_parts: list[pl.DataFrame] = []
        seen_val_eras: set[str] = set()

        for fold in folds:
            overlap = seen_val_eras & set(fold.val_eras)
            if overlap:
                raise ValueError(
                    f"Validation eras must be disjoint across folds, got {sorted(overlap)}"
                )

            part_path = None
            if checkpoint_dir is not None:
                part_path = checkpoint_dir / target_col / f"fold_{fold.index + 1:02d}.parquet"
            if part_path is not None and part_path.exists():
                try:
                    fold_predictions = pl.read_parquet(part_path)
                except Exception as exc:  # fail loud, no silent refit
                    raise ValueError(
                        f"corrupt OOF checkpoint {part_path}: {exc}"
                    ) from exc
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
                    atomic_write_parquet(fold_predictions, part_path)
            oof_parts.append(fold_predictions)
            seen_val_eras.update(fold.val_eras)
        ...
```

`atomic_write_parquet` is the `nmr/_atomicio.py` helper (read that file — if it exposes a generic atomic-write function, use it; if it exposes only JSON helpers, add a sibling `atomic_write_parquet(frame, path)` in `nmr/_atomicio.py` using the same temp+fsync+replace pattern, with a test in `tests/test_checkpointing.py`).

In `nmr/_oof.py`: add `checkpoint_dir: Path | None = None` and pass it through to `modeler.train_cross_validation(...)` (import `Path` from pathlib).

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_checkpointing.py tests/test_models.py -q`
Expected: PASS — new contract tests AND the untouched legacy model tests.

- [ ] **Step 5: Lint + commit**

Run: `./.venv/Scripts/python -m ruff check nmr/models.py nmr/_oof.py nmr/_atomicio.py tests/test_checkpointing.py` → clean, then:

```bash
git add nmr/models.py nmr/_oof.py nmr/_atomicio.py tests/test_checkpointing.py
git commit -m "feat(runner): fold-granularity OOF checkpointing with skip-on-resume"
```

---

### Task B: Runner wiring

**Files:**
- Modify: `nmr/runner.py` (`run()` — pass the run-scoped checkpoint dir into `_train_multi_target_oof`)
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: Task A's `checkpoint_dir` parameters; `self._run_id`, `self._config.run.artifacts_dir` (both already exist in `ExperimentRunner`).

- [ ] **Step 1: Write failing test** — append to `tests/test_runner.py` (read the file first; reuse its synthetic config/run helpers):

```python
def test_runner_writes_and_reuses_oof_checkpoints(tmp_path):
    # reuse the file's existing minimal synthetic ExperimentConfig helper;
    # point artifacts_dir at tmp_path / "artifacts"
    result1 = run_synthetic_experiment(tmp_path)   # existing helper, adapted
    ckpt_root = tmp_path / "artifacts" / "runs" / result1.run_id / "oof_checkpoints"
    assert ckpt_root.exists()
    assert (ckpt_root / "target").glob("fold_*.parquet")  # non-empty
    result2 = run_synthetic_experiment(tmp_path)   # same config+data -> same run_id
    assert result2.oof.equals(result1.oof)
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_runner.py -k oof_checkpoints -q`
Expected: FAIL — no checkpoint dir created.

- [ ] **Step 3: Implement** — in `nmr/runner.py::run()`, change the OOF call:

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

### Task C: Docs + targeted verification

**Files:**
- Modify: `ARCHITECTURE.md` (OOF path section: checkpoint layout `artifacts/runs/<run_id>/oof_checkpoints/<target>/fold_NN.parquet`, resume semantics, determinism invariant, atomic-write rule, "concurrent duplicate run_ids unsupported")
- Modify: `docs/superpowers/2026-08-19-benchmark-fleet-handoff.md` (§8 queued-work item 1: mark DONE with commit refs)

- [ ] **Step 1: Docs edits as above** (no AGENTS.md changes — no headroom).

- [ ] **Step 2: Targeted verification**

Run: `./.venv/Scripts/python -m pytest tests/test_checkpointing.py tests/test_models.py tests/test_runner.py tests/test_campaign.py -q`
Expected: PASS. (Full suite deliberately deferred — the `mt_std_v1` campaign owns the CPU; note this in the commit message.)

- [ ] **Step 3: Commit**

```bash
git add ARCHITECTURE.md docs/superpowers/2026-08-19-benchmark-fleet-handoff.md
git commit -m "docs: OOF checkpoint/resume spec + plan references; handoff update (full suite deferred: campaign CPU-bound)"
```
