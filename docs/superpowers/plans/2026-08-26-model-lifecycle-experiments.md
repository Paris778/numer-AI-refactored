# Model Lifecycle & Self-Contained Experiment Layout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three global per-model pools (`artifacts/runs/`, `artifacts/registry/`, `artifacts/models/`) with self-contained per-family directories under `experiments/`, add a display-name naming layer, a six-state lifecycle, and a train-only (`partial`) promotion scope with local cross-check scoring.

**Architecture:** New pure modules `nmr/paths.py` (path derivation), `nmr/lifecycle.py` (validity/stage derivation), `nmr/experiment_store.py` (persistence + atomic publication) are built first and fully tested. `nmr/scorecard.py` gains a non-breaking `evaluate_cross_check`. Then the pipeline (`runner`, `promote`, `registry`) and discovery/dashboard layers are retargeted onto them. Docs and `.gitignore` update in the same final commit set.

**Tech Stack:** Python 3.11+, Polars, cloudpickle, `numerai_tools` (scoring oracle), pytest, ruff (E/F/I/UP @120). Working tree is `main` (no worktree).

**Spec:** `docs/superpowers/specs/2026-08-26-model-lifecycle-experiments-design.md` (v4 — authoritative contract).

## Global Constraints

- All logic lives in `nmr/`; scripts/dashboards are thin control planes. Every business rule below is a `nmr/` function covered by `tests/`.
- **Vocabulary:** promotion request scope is `"train_only"`; persisted `training_scope` is `"partial"` or `"full"` — never `"train_only"` in a manifest.
- **Slug rule:** family slug matches `^[a-z0-9_-]+$` (lowercase only — case-collision safety on NTFS).
- **Determinism:** run-id payloads must exclude absolute paths and wall-clock fields. Editing `nmr/*.py` changes run-ids (code identity) — that is expected, not a bug. `generated_at` is excluded from canonical serializations.
- **Fit-phase isolation:** a `train_only` fit must never open `validation.parquet`; the post-fit cross-check phase may.
- **Immutability:** exports are immutable; `force=True` repoints `current.json` only, never replaces a slot. Promoting an existing slot raises `ValueError` before any write.
- **Trusted-source:** `load_predict()` is only ever called on artifacts under `experiments/` (or the legacy `artifacts/` tree during migration of tests) — never on outside paths.
- **Verification gates:** run `ruff check .` + targeted `pytest` after every task; full suite at the end. Real-data tests are skipped without `data/v5.3/` — report skips, never fake green.
- **Commit discipline:** one commit per task, small and focused.

---

### Task 1: `nmr/paths.py` — path derivation

**Files:**
- Create: `nmr/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Consumes: `nmr/config.REPO_ROOT`
- Produces (used by every later task):
  - `EXPERIMENTS_ROOT: Path`
  - `SLUG_RE: re.Pattern`
  - `validate_slug(slug: str) -> str`
  - `experiment_dir(slug) -> Path`
  - `run_dir(slug, run_id) -> Path`
  - `run_json_path(slug, run_id) -> Path`
  - `export_dir(slug, scope, run_id) -> Path`
  - `export_json_path(slug, scope, run_id) -> Path`
  - `current_pointer_path(slug) -> Path`
  - `champion_path() -> Path`
  - `shared_cache_dir(artifacts_dir: Path | None = None) -> Path`
  - `shared_reports_dir(artifacts_dir: Path | None = None) -> Path`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paths.py
import pytest
from nmr import paths
from nmr.config import REPO_ROOT

def test_experiment_layout():
    assert paths.EXPERIMENTS_ROOT == REPO_ROOT / "experiments"
    assert paths.experiment_dir("ender-xgb-v1") == REPO_ROOT / "experiments" / "ender-xgb-v1"
    assert paths.run_dir("ender-xgb-v1", "ab" * 32) == (
        REPO_ROOT / "experiments" / "ender-xgb-v1" / "runs" / ("ab" * 32)
    )
    assert paths.run_json_path("f", "a" * 64).name == "run.json"
    assert paths.export_dir("f", "partial", "a" * 64) == (
        REPO_ROOT / "experiments" / "f" / "exports" / "partial" / ("a" * 64)
    )
    assert paths.export_json_path("f", "full", "a" * 64).name == "export.json"
    assert paths.current_pointer_path("f").name == "current.json"
    assert paths.champion_path() == REPO_ROOT / "experiments" / "champion.json"
    assert paths.shared_cache_dir() == REPO_ROOT / "artifacts" / "cache"
    assert paths.shared_reports_dir() == REPO_ROOT / "artifacts" / "reports"

def test_validate_slug_rejects_bad():
    for bad in ("UPPER", "has space", "semi;colon", ""):
        with pytest.raises(ValueError):
            paths.validate_slug(bad)
    assert paths.validate_slug("ender-xgb_v1") == "ender-xgb_v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_paths.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'nmr.paths'`

- [ ] **Step 3: Write the implementation**

```python
# nmr/paths.py
"""Pure path derivation for the experiment layout.

Single place that knows where anything lives under ``experiments/`` or the
shared machine cache. No module hardcodes ``experiments`` or ``artifacts``
strings. Nothing here reads/writes the filesystem or enters a canonical hash
except as config-provided values (shared helpers take ``artifacts_dir``).
"""
from __future__ import annotations

import re
from pathlib import Path

from nmr.config import REPO_ROOT

EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# Lowercase-only: prevents case-collision overwrites on case-insensitive
# filesystems and matches the run.name / family convention.
SLUG_RE = re.compile(r"^[a-z0-9_-]+$")


def validate_slug(slug: str) -> str:
    """Validate a family slug; raises ValueError."""
    if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
        raise ValueError(
            f"invalid family slug {slug!r}: must match {SLUG_RE.pattern}"
        )
    return slug


def experiment_dir(slug: str) -> Path:
    return EXPERIMENTS_ROOT / validate_slug(slug)


def run_dir(slug: str, run_id: str) -> Path:
    return experiment_dir(slug) / "runs" / run_id


def run_json_path(slug: str, run_id: str) -> Path:
    return run_dir(slug, run_id) / "run.json"


def export_dir(slug: str, scope: str, run_id: str) -> Path:
    if scope not in ("partial", "full"):
        raise ValueError(f"scope must be 'partial' or 'full', got {scope!r}")
    return experiment_dir(slug) / "exports" / scope / run_id


def export_json_path(slug: str, scope: str, run_id: str) -> Path:
    return export_dir(slug, scope, run_id) / "export.json"


def current_pointer_path(slug: str) -> Path:
    return export_dir(slug, "full", "") / "current.json"


def champion_path() -> Path:
    return EXPERIMENTS_ROOT / "champion.json"


def shared_cache_dir(artifacts_dir: Path | None = None) -> Path:
    return (artifacts_dir or DEFAULT_ARTIFACTS_DIR) / "cache"


def shared_reports_dir(artifacts_dir: Path | None = None) -> Path:
    return (artifacts_dir or DEFAULT_ARTIFACTS_DIR) / "reports"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_paths.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add nmr/paths.py tests/test_paths.py
git commit -m "feat(paths): pure path derivation for the experiment layout"
```

---

### Task 2: `nmr/lifecycle.py` — validity, stage derivation, ordering

**Files:**
- Create: `nmr/lifecycle.py`
- Test: `tests/test_lifecycle.py`
- Modify: `nmr/__init__.py` (add `paths`, `lifecycle` to imports + `__all__`)

**Interfaces:**
- Consumes: `nmr.paths` (Task 1), `nmr.deployment.load_predict` (existing), `nmr.families` regex-free validation moved here via `paths.validate_slug`
- Produces:
  - `SCOPES = ("partial", "full")`
  - `LIFECYCLE_STAGES = ("uninitialized", "research", "partial", "degraded", "full", "staked")`
  - `StakedRecord` (frozen dataclass: `run_id, scope, numerai_model_id, staked_at, status`)
  - `ExportVersion` (frozen dataclass: `family, scope, run_id, slot_dir, training_scope, promoted_at, training_rows, training_era_range, config, tier4_gate_passed, rehearsal`)
  - `load_staked_record(meta_path: Path) -> StakedRecord | None`
  - `valid_export(family, scope, run_id) -> ExportVersion | None`
  - `scan_valid_exports(family, scope) -> list[ExportVersion]`
  - `current_full_status(family) -> Literal["full", "degraded", "none"]`
  - `derive_stage(family, staked) -> tuple[str, str]` — `(lifecycle_stage, current_full_status)`
  - `sort_exports(exports) -> list[ExportVersion]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lifecycle.py
import json
import pytest
from nmr import lifecycle, paths
from nmr.config import REPO_ROOT
from nmr.lifecycle import ExportVersion, StakedRecord

def _write_export(slug, scope, run_id, *, training_scope=None, sha="0" * 64, meta_json=None):
    slot = paths.export_dir(slug, scope, run_id)
    slot.mkdir(parents=True, exist_ok=True)
    (slot / "predict.pkl").write_bytes(b"x")
    (slot / "predict.pkl.manifest.json").write_text(
        json.dumps({"sha256": sha})
    )
    ts = training_scope or scope
    (slot / "export.json").write_text(
        json.dumps({"family": slug, "training_scope": ts,
                    "promoted_from_run_id": run_id, "promoted_at": "2026-08-26T10:00:00+00:00",
                    "config": {}})
    )
    if scope == "partial":
        (slot / "scorecard.json").write_text(json.dumps({"schema_version": 3}))
    if meta_json is not None:
        paths.experiment_dir(slug).joinpath("meta.json").write_text(json.dumps(meta_json))
    return slot

def test_stage_derivation_total():
    # uninitialized: dir exists, no run.json
    paths.experiment_dir("fam1").mkdir(parents=True, exist_ok=True)
    assert lifecycle.derive_stage("fam1", None) == ("uninitialized", "none")
    # research: run.json present, no exports
    paths.run_json_path("fam1", "a" * 64).parent.mkdir(parents=True, exist_ok=True)
    paths.run_json_path("fam1", "a" * 64).write_text("{}")
    assert lifecycle.derive_stage("fam1", None) == ("research", "none")
    # partial: valid partial export, no full
    _write_export("fam1", "partial", "b" * 64)
    assert lifecycle.derive_stage("fam1", None) == ("partial", "none")
    # degraded: valid full export, dangling pointer
    _write_export("fam1", "full", "c" * 64)
    assert lifecycle.derive_stage("fam1", None) == ("degraded", "degraded")
    # full: pointer at valid slot
    paths.current_pointer_path("fam1").write_text(json.dumps({"run_id": "c" * 64}))
    assert lifecycle.derive_stage("fam1", None) == ("full", "full")
    # staked: active stake on valid full
    staked = StakedRecord(run_id="c" * 64, scope="full", numerai_model_id="m1",
                          staked_at="2026-08-26T11:00:00+00:00", status="active")
    assert lifecycle.derive_stage("fam1", staked) == ("staked", "full")

def test_staked_stale_when_export_invalid():
    staked = StakedRecord(run_id="dead" * 16, scope="full", numerai_model_id="m1",
                          staked_at="2026-08-26T11:00:00+00:00", status="active")
    assert lifecycle.derive_stage("fam1", staked)[0] != "staked"

def test_export_identity_binding():
    # manifest promoted_from_run_id != slot dir run_id -> invalid
    slot = _write_export("fam2", "full", "a" * 64)
    (slot / "export.json").write_text(
        json.dumps({"family": "fam2", "training_scope": "full",
                    "promoted_from_run_id": "b" * 64, "promoted_at": "x", "config": {}})
    )
    assert lifecycle.valid_export("fam2", "full", "a" * 64) is None

def test_sort_exports_deterministic():
    e1 = ExportVersion(family="f", scope="partial", run_id="b" * 64, slot_dir=__import__("pathlib").Path("."),
                       training_scope="partial", promoted_at="2026-08-26T10:00:00+00:00",
                       training_rows=None, training_era_range=None, config={},
                       tier4_gate_passed=None, rehearsal=False)
    e2 = ExportVersion(family="f", scope="partial", run_id="a" * 64, slot_dir=__import__("pathlib").Path("."),
                       training_scope="partial", promoted_at="2026-08-26T10:00:00+00:00",
                       training_rows=None, training_era_range=None, config={},
                       tier4_gate_passed=None, rehearsal=False)
    e3 = ExportVersion(family="f", scope="partial", run_id="c" * 64, slot_dir=__import__("pathlib").Path("."),
                       training_scope="partial", promoted_at="2026-08-25T10:00:00+00:00",
                       training_rows=None, training_era_range=None, config={},
                       tier4_gate_passed=None, rehearsal=False)
    got = lifecycle.sort_exports([e1, e2, e3])
    assert [e.run_id for e in got] == ["a" * 64, "b" * 64, "c" * 64]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_lifecycle.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'nmr.lifecycle'`

- [ ] **Step 3: Write the implementation**

```python
# nmr/lifecycle.py
"""Lifecycle: export validity, stage derivation (a total function), ordering.

Derives the six lifecycle stages from filesystem state. Importing this module
must not trigger heavy loads; ``valid_export`` calls ``load_predict`` (hash-
verified) per the spec's validity predicate — trusted-source rule applies.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nmr import paths

logger = logging.getLogger("nmr.lifecycle")

SCOPES = ("partial", "full")
LIFECYCLE_STAGES = ("uninitialized", "research", "partial", "degraded", "full", "staked")

# Badge precedence: staked > full > degraded > partial > research > uninitialized.
_PRECEDENCE = {stage: i for i, stage in enumerate(LIFECYCLE_STAGES)}

_STAGE_ORDER_SCORES: dict[str, int] = {
    "uninitialized": 0, "research": 1, "partial": 2, "degraded": 3, "full": 4, "staked": 5,
}


@dataclass(frozen=True)
class StakedRecord:
    run_id: str
    scope: str
    numerai_model_id: str | None
    staked_at: str | None
    status: str


@dataclass(frozen=True)
class ExportVersion:
    family: str
    scope: str
    run_id: str
    slot_dir: Path
    training_scope: str
    promoted_at: str | None
    training_rows: int | None
    training_era_range: tuple[int, int] | None
    config: dict[str, Any]
    tier4_gate_passed: bool | None
    rehearsal: bool


def load_staked_record(meta_path: Path) -> StakedRecord | None:
    """Read ``meta.json``'s staked record; None when absent or malformed."""
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    staked = payload.get("staked") if isinstance(payload, dict) else None
    if not isinstance(staked, dict) or not isinstance(staked.get("run_id"), str):
        return None
    return StakedRecord(
        run_id=staked["run_id"],
        scope=staked.get("scope", "full"),
        numerai_model_id=staked.get("numerai_model_id"),
        staked_at=staked.get("staked_at"),
        status=str(staked.get("status", "")),
    )


def _read_export_json(slot_dir: Path) -> dict[str, Any] | None:
    p = slot_dir / "export.json"
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def valid_export(family: str, scope: str, run_id: str) -> ExportVersion | None:
    """Validate one export slot; None on any violation. Identity binding: slot
    dir run_id == manifest.promoted_from_run_id == family slug match."""
    if scope not in SCOPES:
        return None
    slot_dir = paths.export_dir(family, scope, run_id)
    payload = _read_export_json(slot_dir)
    if payload is None:
        return None
    if payload.get("family") != family:
        return None
    if payload.get("promoted_from_run_id") != run_id:
        return None
    training_scope = payload.get("training_scope")
    if training_scope != scope:
        return None
    pkl = slot_dir / "predict.pkl"
    sibling = slot_dir / "predict.pkl.manifest.json"
    if not pkl.is_file() or not sibling.is_file():
        return None
    try:
        manifest = json.loads(sibling.read_text(encoding="utf-8"))
        expected = manifest.get("sha256")
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(expected, str) or expected != _sha256_of(pkl):
        return None
    # Verified loadability (trusted-source: only repo artifacts reach here).
    from nmr.deployment import load_predict
    try:
        load_predict(slot_dir)
    except Exception:  # noqa: BLE001 - any failure => invalid
        return None
    if scope == "partial" and not (slot_dir / "scorecard.json").is_file():
        return None
    era_range = payload.get("training_era_range")
    return ExportVersion(
        family=family, scope=scope, run_id=run_id, slot_dir=slot_dir,
        training_scope=str(training_scope),
        promoted_at=payload.get("promoted_at"),
        training_rows=int(payload["training_rows"]) if isinstance(payload.get("training_rows"), (int, float)) else None,
        training_era_range=(int(era_range[0]), int(era_range[1])) if isinstance(era_range, list) and len(era_range) == 2 else None,
        config=payload.get("config") if isinstance(payload.get("config"), dict) else {},
        tier4_gate_passed=payload.get("tier4_gate_passed"),
        rehearsal=bool(payload.get("rehearsal", False)),
    )


def _sha256_of(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_valid_exports(family: str, scope: str) -> list[ExportVersion]:
    """All VALID exports of one scope for a family, sorted (deterministic)."""
    base = paths.export_dir(family, scope, "x").parent  # .../exports/<scope>
    if not base.is_dir():
        return []
    found = []
    for entry in sorted(base.iterdir()):
        if entry.is_dir() and not entry.name.startswith(".tmp-"):
            version = valid_export(family, scope, entry.name)
            if version is not None:
                found.append(version)
    return sort_exports(found)


def sort_exports(exports: list[ExportVersion]) -> list[ExportVersion]:
    """ISO-8601 ``promoted_at`` descending; unparseable sorts last; tie-break
    ``run_id`` ascending."""
    def key(e: ExportVersion):
        try:
            if e.promoted_at is None:
                raise ValueError("missing")
            from datetime import datetime
            ts = datetime.fromisoformat(e.promoted_at)
            return (0, -ts.timestamp(), e.run_id)
        except (ValueError, TypeError):
            return (1, 0.0, e.run_id)
    return sorted(exports, key=key)


def current_full_status(family: str) -> Literal["full", "degraded", "none"]:
    """'full' when the pointer resolves to a valid full slot; 'degraded' when
    valid full slots exist but the pointer is missing/dangling; else 'none'."""
    pointer = paths.current_pointer_path(family)
    has_valid_full = any(scan_valid_exports(family, "full"))
    if not has_valid_full:
        return "none"
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        run_id = payload.get("run_id") if isinstance(payload, dict) else None
    except (json.JSONDecodeError, OSError):
        run_id = None
    if isinstance(run_id, str) and valid_export(family, "full", run_id) is not None:
        return "full"
    return "degraded"


def _has_runs(family: str) -> bool:
    runs_dir = paths.experiment_dir(family) / "runs"
    return runs_dir.is_dir() and any(runs_dir.iterdir())


def derive_stage(family: str, staked: StakedRecord | None) -> tuple[str, str]:
    """Total function over filesystem state -> (lifecycle_stage, current_full_status)."""
    status = current_full_status(family)
    stage_score = 0
    if _has_runs(family):
        stage_score = _STAGE_ORDER_SCORES["research"]
    if any(scan_valid_exports(family, "partial")):
        stage_score = max(stage_score, _STAGE_ORDER_SCORES["partial"])
    if status in ("full", "degraded"):
        stage_score = max(stage_score, _STAGE_ORDER_SCORES[status])
    if staked is not None and staked.status == "active" \
            and staked.scope == "full" and valid_export(family, "full", staked.run_id) is not None:
        stage_score = _STAGE_ORDER_SCORES["staked"]
    stage = next(s for s, v in _STAGE_ORDER_SCORES.items() if v == stage_score)
    return stage, status
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_lifecycle.py -q`
Expected: PASS

- [ ] **Step 5: Register in `nmr/__init__.py`**

Add to the imports block and `__all__`: `"paths"`, `"lifecycle"` (module imports — see the existing `__init__.py` style; the spec's public-API rule requires exports in `__all__`).

- [ ] **Step 6: Commit**

```bash
git add nmr/lifecycle.py nmr/__init__.py tests/test_lifecycle.py
git commit -m "feat(lifecycle): export validity, total stage derivation, deterministic ordering"
```

---

### Task 3: `.gitignore` + `experiments/` scaffold

**Files:**
- Modify: `.gitignore`
- Create: `experiments/.gitkeep`
- Test: `tests/test_gitignore.py`

**Interfaces:**
- Consumes: nothing (filesystem only)
- Produces: the git-tracked/ignored classification the whole feature relies on

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gitignore.py
import subprocess
from pathlib import Path
import pytest
from nmr.config import REPO_ROOT

TRACKED = [
    "experiments/demo/README.md",
    "experiments/demo/base_config.yaml",
    "experiments/demo/meta.json",
    "experiments/demo/runs/abc/run.json",
    "experiments/demo/exports/full/abc/export.json",
    "experiments/demo/exports/partial/abc/scorecard.json",
    "experiments/demo/exports/full/current.json",
    "experiments/champion.json",
]
IGNORED = [
    "experiments/demo/runs/abc/oof.parquet",
    "experiments/demo/runs/abc/validation_preds.parquet",
    "experiments/demo/runs/abc/predict.pkl",
    "experiments/demo/exports/full/abc/predict.pkl",
]

def _check_ignore(path: str) -> bool:
    out = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=REPO_ROOT, capture_output=True
    )
    return out.returncode == 0

def test_gitignore_classification():
    for p in TRACKED:
        assert not _check_ignore(p), f"{p} must be trackable"
    for p in IGNORED:
        assert _check_ignore(p), f"{p} must be ignored"
    assert not _check_ignore("experiments/.gitkeep")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_gitignore.py -q`
Expected: FAIL — paths not yet matched (no `experiments/` rules).

- [ ] **Step 3: Add the rules to `.gitignore` (append at the end)**

```gitignore
# self-contained experiment layout: versioned metadata, ignored heavy files
experiments/**
!experiments/
!experiments/**/
!experiments/*/README.md
!experiments/*/base_config.yaml
!experiments/*/meta.json
!experiments/*/runs/*/run.json
!experiments/*/exports/**/export.json
!experiments/*/exports/**/scorecard.json
!experiments/*/exports/full/current.json
!experiments/champion.json
!experiments/**/.gitkeep
```

Create `experiments/.gitkeep` (empty file).

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_gitignore.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add .gitignore experiments/.gitkeep tests/test_gitignore.py
git commit -m "chore(gitignore): version experiment metadata, ignore heavy artifacts"
```

---

### Task 4: `evaluate_cross_check` in `nmr/scorecard.py`

**Files:**
- Modify: `nmr/scorecard.py`
- Test: `tests/test_scorecard.py` (extend), `tests/test_parity.py` untouched

**Interfaces:**
- Consumes: existing `evaluate_model` internals (must NOT change its public behavior)
- Produces:
  - `@dataclass(frozen=True) CrossCheckResult: scorecard: MetricScorecard; per_era: dict[str, list[dict[str, float | str]]]; raw_sharpe: float`
  - `evaluate_cross_check(predictions, *, meta_model, features, targets, horizon="20D", main_target="target", seed=42) -> CrossCheckResult`

- [ ] **Step 1: Refactor `evaluate_model` internals (no behavior change), write failing tests**

The internal per-era computation in `evaluate_model` computes `corr_by_era`, `mmc_by_era`, `fnc_by_era` as locals then builds `MetricScorecard`. Extract those into an internal helper `_compute_era_series(evaluator, ...) -> tuple[dict, dict, dict]` consumed by both `evaluate_model` and the new function. The existing `evaluate_model` tests (`tests/test_scorecard.py`) are the regression net — they must pass unchanged.

```python
# tests/test_scorecard.py (append)
def test_cross_check_result_shape(synthetic_frames):
    predictions, meta_model, features, targets = synthetic_frames
    result = scorecard.evaluate_cross_check(
        predictions, meta_model=meta_model, features=features, targets=targets,
        horizon="20D", main_target="target", seed=42,
    )
    assert isinstance(result.scorecard, scorecard.MetricScorecard)
    assert set(result.per_era) == {"corr", "mmc", "fnc"}
    for era_entry in result.per_era["corr"]:
        assert set(era_entry) == {"era", "value"}
    assert isinstance(result.raw_sharpe, float)
```

> The `synthetic_frames` fixture already exists in `tests/test_scorecard.py` — reuse it.

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/Scripts/python -m pytest tests/test_scorecard.py -q`
Expected: FAIL — `AttributeError: module has no attribute 'evaluate_cross_check'`

- [ ] **Step 3: Implement**

```python
# nmr/scorecard.py (add near evaluate_model)
@dataclass(frozen=True)
class CrossCheckResult:
    """Cross-check evaluation: scorecard + labeled per-era series + raw Sharpe."""
    scorecard: MetricScorecard
    per_era: dict[str, list[dict[str, float | str]]]
    raw_sharpe: float


# Fixed replay constants (spec §7) — the cross-check never varies these.
CROSSCHECK_N_TRIALS = 1


def evaluate_cross_check(
    predictions: pl.DataFrame,
    *,
    meta_model: pl.DataFrame,
    features: pl.DataFrame,
    targets: pl.DataFrame,
    horizon: Horizon = "20D",
    main_target: str = "target",
    seed: int = 42,
) -> CrossCheckResult:
    """Era-grouped official-backend cross-check of a partial export.

    Reuses the exact per-era computation path of ``evaluate_model`` (shared
    internal helper — no duplicated math) but returns the per-era series and
    raw Sharpe the scorecard itself does not retain. Replay parameters are the
    fixed constants above + ``evaluate_model`` defaults.
    """
    from nmr.config import ExperimentConfig  # noqa: F401 (type only)
    evaluator = _build_evaluator(backend="official", horizon=horizon, main_target=main_target)
    corr_by_era, mmc_by_era, fnc_by_era = _compute_era_series(
        evaluator, predictions, meta_model=meta_model, features=features, targets=targets,
        main_target=main_target,
    )
    scorecard_result = _build_scorecard_from_series(
        predictions=predictions,
        corr_by_era=corr_by_era, mmc_by_era=mmc_by_era, fnc_by_era=fnc_by_era,
        meta_model=meta_model, features=features, targets=targets,
        n_trials=CROSSCHECK_N_TRIALS, seed=seed, horizon=horizon,
        main_target=main_target, backend="official",
    )
    raw_sharpe = _plain_sharpe(list(corr_by_era.values()))
    per_era = {
        "corr": [{"era": str(e), "value": float(v)} for e, v in corr_by_era.items()],
        "mmc": [{"era": str(e), "value": float(v)} for e, v in mmc_by_era.items()],
        "fnc": [{"era": str(e), "value": float(v)} for e, v in fnc_by_era.items()],
    }
    return CrossCheckResult(scorecard=scorecard_result, per_era=per_era, raw_sharpe=raw_sharpe)
```

> The implementer must map this onto the ACTUAL internals of `evaluate_model` (its evaluator construction, `_build_scorecard_from_series` equivalent, and `era_series_stats`/`ac_adjusted_sharpe` usage). The invariants: (1) `evaluate_model`'s public behavior is byte-identical (existing tests prove it), (2) the per-era series come from the shared internal path, (3) `backend="official"` is forced. If a shared internal helper already exists, reuse it; extract one only if `evaluate_model` currently inlines the computation.

- [ ] **Step 4: Run tests**

Run: `./.venv/Scripts/python -m pytest tests/test_scorecard.py tests/test_parity.py -q`
Expected: PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add nmr/scorecard.py tests/test_scorecard.py
git commit -m "feat(scorecard): evaluate_cross_check with per-era series and raw Sharpe"
```

---

### Task 5: `nmr/experiment_store.py` — persistence + atomic publication

**Files:**
- Create: `nmr/experiment_store.py`
- Test: `tests/test_experiment_store.py`

**Interfaces:**
- Consumes: `nmr.paths`, `nmr._atomicio.atomic_write_text`, `nmr.lifecycle`
- Produces:
  - `create_scaffold(slug, *, display_name, base_config) -> Path`
  - `ensure_family(slug, *, display_name, base_config) -> Path`
  - `record_run(slug, run_id, payload) -> Path`
  - `read_run(slug, run_id) -> dict[str, Any]`
  - `stage_export(slug, scope, run_id) -> Path`
  - `discard_staged_export(slug, scope, run_id) -> None`
  - `publish_staged_export(slug, scope, run_id) -> Path`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_experiment_store.py
import json
import pytest
from nmr import experiment_store, paths

def test_record_run_creates_scaffold_and_run(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    run_id = "a" * 64
    p = experiment_store.record_run("ender-xgb-v1", run_id, {"scorecard": {}})
    assert p.name == "run.json"
    assert paths.experiment_dir("ender-xgb-v1").joinpath("meta.json").is_file()
    assert paths.experiment_dir("ender-xgb-v1").joinpath("base_config.yaml").is_file()
    assert paths.experiment_dir("ender-xgb-v1").joinpath("README.md").is_file()
    assert json.loads(p.read_text())["scorecard"] == {}

def test_stage_publish_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    run_id = "a" * 64
    staging = experiment_store.stage_export("f", "partial", run_id)
    assert staging.name == f".tmp-{run_id}"
    (staging / "predict.pkl").write_bytes(b"x")
    (staging / "export.json").write_text("{}")
    final = experiment_store.publish_staged_export("f", "partial", run_id)
    assert final == paths.export_dir("f", "partial", run_id)
    assert not staging.exists()
    assert (final / "predict.pkl").read_bytes() == b"x"

def test_discard_staged(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    staging = experiment_store.stage_export("f", "full", "a" * 64)
    (staging / "x").write_text("x")
    experiment_store.discard_staged_export("f", "full", "a" * 64)
    assert not staging.exists()

def test_republish_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    run_id = "a" * 64
    experiment_store.stage_export("f", "full", run_id)
    experiment_store.publish_staged_export("f", "full", run_id)
    staging = experiment_store.stage_export("f", "full", run_id)
    (staging / "x").write_text("x")
    with pytest.raises(ValueError):
        experiment_store.publish_staged_export("f", "full", run_id)  # slot exists
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/Scripts/python -m pytest tests/test_experiment_store.py -q`
Expected: FAIL — module missing

- [ ] **Step 3: Implement**

```python
# nmr/experiment_store.py
"""Run/export persistence and atomic publication for the experiment layout.

``record_run`` creates the family scaffold atomically with the first run.json
(spec §2 family-creation rule). Export slots are staged under ``.tmp-<run_id>``
and published by a single directory rename; discovery ignores ``.tmp-`` names.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nmr import paths
from nmr._atomicio import atomic_write_text

logger = logging.getLogger("nmr.experiment_store")

_RID_RE = __import__("re").compile(r"^[0-9a-f]{64}$")


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RID_RE.fullmatch(run_id):
        raise ValueError(f"run_id must be 64-char lowercase hex, got {run_id!r}")
    return run_id


def _write_scaffold(slug: str, *, display_name: str, base_config: dict[str, Any]) -> Path:
    exp = paths.experiment_dir(slug)
    exp.mkdir(parents=True, exist_ok=True)
    meta_path = exp / "meta.json"
    if not meta_path.exists():
        atomic_write_text(meta_path, json.dumps({"display_name": display_name, "staked": None}, sort_keys=True))
    base_cfg = exp / "base_config.yaml"
    if not base_cfg.exists():
        import yaml
        atomic_write_text(base_cfg, yaml.safe_dump(base_config, sort_keys=False))
    readme = exp / "README.md"
    if not readme.exists():
        atomic_write_text(readme, f"# {slug}\n\n<!-- human record: what was done, decisions, results -->\n")
    return exp


def ensure_family(slug: str, *, display_name: str, base_config: dict[str, Any]) -> Path:
    return _write_scaffold(slug, display_name=display_name, base_config=base_config)


def record_run(slug: str, run_id: str, payload: dict[str, Any]) -> Path:
    """Persist run.json (atomic) — creating the family scaffold on first run."""
    _validate_run_id(run_id)
    display_name = str((payload.get("manifest") or {}).get("display_name") or slug)
    base_config = (payload.get("manifest") or {}).get("config") or {}
    _write_scaffold(slug, display_name=display_name, base_config=base_config)
    run_dir = paths.run_dir(slug, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = paths.run_json_path(slug, run_id)
    atomic_write_text(target, json.dumps(payload, sort_keys=True))
    return target


def read_run(slug: str, run_id: str) -> dict[str, Any]:
    path = paths.run_json_path(slug, run_id)
    if not path.is_file():
        raise FileNotFoundError(f"no run record at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def stage_export(slug: str, scope: str, run_id: str) -> Path:
    _validate_run_id(run_id)
    parent = paths.export_dir(slug, scope, run_id).parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".tmp-{run_id}"
    if staging.exists():
        import shutil
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def discard_staged_export(slug: str, scope: str, run_id: str) -> None:
    _validate_run_id(run_id)
    staging = paths.export_dir(slug, scope, run_id).parent / f".tmp-{run_id}"
    if staging.exists():
        import shutil
        shutil.rmtree(staging)


def publish_staged_export(slug: str, scope: str, run_id: str) -> Path:
    """Atomically rename the staged dir into the final slot. Raises ValueError
    when the slot already exists (exports are immutable — spec §6)."""
    _validate_run_id(run_id)
    final = paths.export_dir(slug, scope, run_id)
    staging = final.parent / f".tmp-{run_id}"
    if not staging.is_dir():
        raise FileNotFoundError(f"no staged export at {staging}")
    if final.exists():
        discard_staged_export(slug, scope, run_id)
        raise ValueError(f"export slot {final} already exists; exports are immutable")
    import os
    os.replace(staging, final)  # single directory rename; target is absent
    return final
```

> Note: `os.replace` on directories works on both POSIX and Windows when the destination does not exist. `atomic_write_text` already implements temp+fsync+replace.

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/Scripts/python -m pytest tests/test_experiment_store.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nmr/experiment_store.py tests/test_experiment_store.py
git commit -m "feat(store): run persistence, family scaffold, atomic export publication"
```

---

### Task 6: `nmr/registry.py` — cross-family champion

**Files:**
- Modify: `nmr/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: `nmr.experiment_store.read_run`, `nmr.paths`
- Produces (retargeted class):
  - `RunRegistry(root: Path)` — root = experiments root
  - `list() -> list[str]` — all run_ids across families
  - `best(metric="corr_sharpe_ac") -> tuple[str, str] | None` — `(run_id, slug)`
  - `promote(run_id: str, slug: str) -> Path`
  - `promote_if_better(run_id: str, slug: str, metric="corr_sharpe_ac") -> tuple[Path, bool]`
  - `resolve_champion() -> tuple[str, str]` — `(run_id, slug)`

- [ ] **Step 1: Write the failing tests (extend `tests/test_registry.py`)**

```python
def test_cross_family_list_and_best(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    reg = RunRegistry(tmp_path / "experiments")
    payload_a = {"manifest": {"config": {"run": {"name": "fam-a"}}}, "scorecard": {"corr_sharpe_ac": 0.5}}
    payload_b = {"manifest": {"config": {"run": {"name": "fam-b"}}}, "scorecard": {"corr_sharpe_ac": 0.9}}
    experiment_store.record_run("fam-a", "a" * 64, payload_a)
    experiment_store.record_run("fam-b", "b" * 64, payload_b)
    assert set(reg.list()) == {"a" * 64, "b" * 64}
    assert reg.best() == ("b" * 64, "fam-b")

def test_champion_pointer_has_slug(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    reg = RunRegistry(tmp_path / "experiments")
    experiment_store.record_run("fam-a", "a" * 64, {"scorecard": {}})
    path = reg.promote("a" * 64, "fam-a")
    payload = json.loads(path.read_text())
    assert payload == {"run_id": "a" * 64, "experiment_slug": "fam-a", "promoted_at": payload["promoted_at"]}
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/Scripts/python -m pytest tests/test_registry.py -q`
Expected: FAIL on the new tests

- [ ] **Step 3: Reimplement `RunRegistry`**

```python
# nmr/registry.py (reimplement; keep the class name and docstring spirit)
class RunRegistry:
    """Cross-family run registry: global comparison + champion pointer only.

    Runs live under ``experiments/<slug>/runs/<run_id>/run.json``; this class
    iterates families for comparison and owns ``champion.json``. Champion
    writes are single-writer (CLI/runner entry points only — spec §9).
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _iter_run_records(self):
        if not self._root.is_dir():
            return
        for slug_dir in sorted(self._root.iterdir()):
            runs_dir = slug_dir / "runs"
            if not runs_dir.is_dir():
                continue
            for run_dir in sorted(runs_dir.iterdir()):
                run_json = run_dir / "run.json"
                if run_json.is_file():
                    yield slug_dir.name, run_dir.name, json.loads(run_json.read_text(encoding="utf-8"))

    def list(self) -> list[str]:
        return sorted(run_id for _, run_id, _ in self._iter_run_records())

    def best(self, metric: str = "corr_sharpe_ac") -> tuple[str, str] | None:
        best = None
        for slug, run_id, payload in self._iter_run_records():
            value = ((payload.get("scorecard") or {}).get(metric))
            if value is None:
                continue
            if best is None or float(value) > best[0]:
                best = (float(value), run_id, slug)
        return (best[1], best[2]) if best else None

    def promote(self, run_id: str, slug: str) -> Path:
        experiment_store.read_run(slug, run_id)  # existence check, fail loud
        payload = {"run_id": run_id, "experiment_slug": slug,
                   "promoted_at": datetime.now(UTC).isoformat()}
        return self._atomic_json_write(paths.champion_path(), payload)

    def promote_if_better(self, run_id: str, slug: str, metric: str = "corr_sharpe_ac") -> tuple[Path, bool]:
        record = experiment_store.read_run(slug, run_id)
        candidate = ((record.get("scorecard") or {}).get(metric))
        if candidate is None:
            raise ValueError(f"run {run_id} has no scorecard metric {metric}")
        champion = self.resolve_champion()
        if champion is None:
            return self.promote(run_id, slug), True
        champion_payload = json.loads(paths.champion_path().read_text(encoding="utf-8"))
        champion_record = experiment_store.read_run(champion_payload["experiment_slug"], champion_payload["run_id"])
        champion_value = ((champion_record.get("scorecard") or {}).get(metric))
        if champion_value is None or float(candidate) > float(champion_value):
            return self.promote(run_id, slug), True
        return paths.champion_path(), False

    def resolve_champion(self) -> tuple[str, str] | None:
        pointer = paths.champion_path()
        if not pointer.is_file():
            return None
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        run_id, slug = payload.get("run_id"), payload.get("experiment_slug")
        if not (isinstance(run_id, str) and isinstance(slug, str)):
            raise ValueError(f"corrupt champion pointer: {payload!r}")
        try:
            experiment_store.read_run(slug, run_id)
        except FileNotFoundError:
            raise ValueError(f"champion pointer dangles: {slug}/{run_id}")
        return run_id, slug
```

> The existing `_atomic_json_write` helper (temp + fsync + os.replace) is retained. Callers of `promote(run_id)` / `promote_if_better(run_id, ...)` / `resolve_champion_run_id()` elsewhere (CLIs, dashboard) are updated in Task 11 — until then, keep `resolve_champion_run_id(registry_dir)` as a thin wrapper returning `resolve_champion()[0]` if you need to avoid a mid-plan break, and remove it in Task 11.

- [ ] **Step 4: Run to verify pass + no regressions**

Run: `./.venv/Scripts/python -m pytest tests/test_registry.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nmr/registry.py tests/test_registry.py
git commit -m "feat(registry): cross-family iteration + champion pointer with experiment_slug"
```

---

### Task 7: `nmr/runner.py` retarget + rebuild identity

**Files:**
- Modify: `nmr/runner.py`
- Test: `tests/test_runner.py`, `tests/test_checkpointing.py` (retarget paths), `tests/test_experiment_layout.py` (add runner round-trip)

**Interfaces:**
- Consumes: `nmr.paths`, `nmr.experiment_store.record_run`, `nmr.registry` (indirect)
- Produces:
  - `ExperimentRunner` writes all outputs under `experiments/<slug>/runs/<run_id>/`
  - run.json manifest gains `data_fingerprint`, `code_fingerprint`, `environment`, `pipeline_device`, `oof_device` (§3.1)
  - `_compute_code_fingerprint()` helper (portable, no absolute paths) — check whether `nmr/_oof.py` already has code-identity hashing to reuse

- [ ] **Step 1: Write failing tests (extend `tests/test_experiment_layout.py`)**

```python
def test_runner_outputs_under_experiment(tmp_path, monkeypatch, synthetic_config):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    runner = ExperimentRunner(synthetic_config)
    result = runner.run()
    slug = synthetic_config.run.name
    run_id = result.run_id
    assert paths.run_json_path(slug, run_id).is_file()
    assert paths.run_dir(slug, run_id).joinpath("oof.parquet").is_file()
    assert paths.run_dir(slug, run_id).joinpath("validation_preds.parquet").is_file()

def test_run_manifest_persists_rebuild_identity(synthetic_run_json):
    payload = json.loads(synthetic_run_json.read_text())
    manifest = payload["manifest"]
    for field in ("data_fingerprint", "code_fingerprint", "environment",
                  "pipeline_device", "oof_device"):
        assert isinstance(manifest.get(field), str)
```

> Reuse the existing synthetic-run fixtures from `tests/test_runner.py` where they exist; otherwise build `synthetic_config` from the existing fixture pattern.

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/Scripts/python -m pytest tests/test_experiment_layout.py -q`
Expected: FAIL — outputs still under `artifacts/runs/<run_id>` and manifest lacks the identity fields.

- [ ] **Step 3: Retarget the runner**

Mechanical path swaps in `nmr/runner.py` (each is a one-line change):
- `run_dir` resolution: replace `self._config.run.artifacts_dir / "runs" / self._run_id` with `paths.run_dir(slug, self._run_id)` where `slug = self._config.run.name` (validate with `paths.validate_slug`).
- Checkpoint dirs (`oof_checkpoints`, `deploy_checkpoints`, `validation_checkpoints`): derive from `paths.run_dir(...)` (currently `runner.py:301`, `:411`, `:427`).
- Research `predict.pkl` (currently `runner.py:450`): write to `paths.run_dir(slug, run_id) / "predict.pkl"` + sibling manifest beside it.
- Validation predictions (`validation_preds.parquet`) and `oof.parquet`: write to `paths.run_dir(...)`.
- Registry record: replace `registry.record(result)` with `experiment_store.record_run(slug, result.run_id, payload)` where `payload` is the existing assembled dict (run_id, manifest, scorecard, oof_path, artifact_path, artifact_manifest, metrics).

Manifest assembly — add the rebuild identity block (§3.1) to the manifest dict built around `runner.py:456`:

```python
# in manifest assembly
"data_fingerprint": self._data_fingerprint,          # computed value, now persisted
"code_fingerprint": _compute_code_fingerprint(),     # portable sha256 over nmr/*.py
"environment": _portable_environment(),              # normalized dep versions, no paths
"pipeline_device": str(config.model.device),
"oof_device": self._last_fit_device or str(config.model.device),
```

```python
# nmr/runner.py (module-level helpers)
def _compute_code_fingerprint() -> str:
    """Portable SHA-256 over the nmr package sources (no paths, no timestamps)."""
    import hashlib
    h = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _portable_environment() -> str:
    """Normalized dependency identity — package names + versions only."""
    import importlib.metadata as md
    names = ("numpy", "polars", "lightgbm", "xgboost", "catboost", "scipy", "numerai-tools", "cloudpickle")
    parts = [f"{n}=={md.version(n)}" for n in names]
    return ",".join(sorted(parts))
```

- [ ] **Step 4: Retarget the path-dependent tests**

In `tests/test_runner.py`, `tests/test_checkpointing.py`, `tests/test_promote.py`, `tests/test_registry.py`, `tests/test_dashboard.py`: replace every fixture/tmp path under `artifacts/runs/<run_id>` / `artifacts/registry/<run_id>` with the `paths.run_dir(...)` / `experiments` equivalents. The tests' *assertions* stay identical — only the expected paths change.

- [ ] **Step 5: Run the full targeted subset**

Run: `./.venv/Scripts/python -m pytest tests/test_runner.py tests/test_checkpointing.py tests/test_experiment_layout.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add nmr/runner.py tests/test_runner.py tests/test_checkpointing.py tests/test_experiment_layout.py
git commit -m "feat(runner): outputs under experiments/, rebuild identity persisted"
```

---

### Task 8: `nmr/promote.py` — scope, fit isolation, cross-check, atomic staging

**Files:**
- Modify: `nmr/promote.py`
- Test: `tests/test_promote.py`, `tests/test_experiment_layout.py`

**Interfaces:**
- Consumes: `nmr.experiment_store` (staging), `nmr.lifecycle`, `nmr.scorecard.evaluate_cross_check`, `nmr.paths`
- Produces:
  - `promote_full_version(run_id, family, *, models_dir=None, registry_dir=None, override_gate=False, force=False, data_dir=None, rehearsal=False, scope: Literal["train_only", "full"] = "full") -> PromotionResult`
  - `PromotionResult` gains `scope: str` and `cross_check_path: Path | None`
  - export slots written as `export.json` (not `manifest.json`); persisted `training_scope` is `"partial"`/`"full"`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_promote.py (append) — synthetic fixtures exist in this module
def test_train_only_scope_fits_train_only(synthetic_run, tmp_path, monkeypatch, data_dir_with_split):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    opened = []
    original_load = ...  # spy on the data loader used by the fit phase
    result = promote_full_version(
        synthetic_run.run_id, "fam1", scope="train_only",
        override_gate=True, data_dir=data_dir_with_split,
    )
    assert result.scope == "train_only"
    export = lifecycle.valid_export("fam1", "partial", synthetic_run.run_id)
    assert export is not None
    assert export.training_scope == "partial"
    # fit-phase isolation: validation never opened during the FIT phase
    assert opened == ["train"]  # the fit-phase spy saw only train.parquet

def test_train_only_writes_cross_check_scorecard(...):
    result = promote_full_version(..., scope="train_only", override_gate=True, data_dir=...)
    slot = paths.export_dir("fam1", "partial", run_id)
    sc = json.loads((slot / "scorecard.json").read_text())
    assert sc["schema_version"] == 3 and sc["replay"]["backend"] == "official"

def test_full_scope_requires_current_pointer(...):
    result = promote_full_version(..., scope="full", override_gate=True, data_dir=...)
    pointer = json.loads(paths.current_pointer_path("fam1").read_text())
    assert pointer["run_id"] == run_id

def test_repromotion_rejected(...):
    promote_full_version(..., scope="full", force=True, ...)
    with pytest.raises(ValueError):
        promote_full_version(..., scope="full", force=True, ...)  # slot exists
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/Scripts/python -m pytest tests/test_promote.py -q`
Expected: FAIL — no `scope` parameter

- [ ] **Step 3: Implement**

```python
# nmr/promote.py — signature change
def promote_full_version(
    run_id: str,
    family: str,
    *,
    models_dir: Path | None = None,
    registry_dir: Path | None = None,
    override_gate: bool = False,
    force: bool = False,
    data_dir: Path | None = None,
    rehearsal: bool = False,
    scope: Literal["train_only", "full"] = "full",
) -> PromotionResult:
```

Scope plumbing (the load-bearing changes):

1. **Fit data — `_full_history_frame(config, feature_cols, target_cols, orchestrator, *, scope)`:** for `scope="train_only"`, load `train.parquet` only and raise `FileNotFoundError` if `validation.parquet` is referenced (fit-phase isolation). The spawned-worker spec must pass `include_validation=(scope == "full")`.

2. **RAM guard — `_ram_guard(config, models_dir, *, scope)`:** `current_rows` = rows of the scope's file(s) only (`train_only` → train only). Shared report paths switch from `models_dir.parent / "reports"` to `paths.shared_reports_dir(config.run.artifacts_dir)`.

3. **Target dir + record naming:** staging via `experiment_store.stage_export(family, persisted_scope, run_id)`; write `predict.pkl` + `predict.pkl.manifest.json` + `export.json` into staging; then `experiment_store.publish_staged_export(...)` (immutability rejection built in). Persisted `training_scope` = `"partial"` for `train_only`, `"full"` otherwise. `export.json` carries `family`, `promoted_from_run_id`, `promoted_at`, `training_scope`, `training_rows`, `training_era_range`, `config`, `tier4_gate_passed`, `tier4_receipts`, `override_used`, `rehearsal`, `config_normalizations`.

4. **`current.json`:** only for `scope="full"` (atomic write, `force` gates repointing to a *different* slot — never overwriting a slot).

5. **Cross-check (partial only):** after staging (before publish), run the staged `predict.pkl` on validation eras and call `evaluate_cross_check(...)`; write `scorecard.json` (schema_version 3, §7 — including `window.eras`, `replay`, `per_era`, `raw_sharpe`, `generated_at`). Any failure → `experiment_store.discard_staged_export(...)` + re-raise.

6. **`PromotionResult`:** add `scope: str` and `cross_check_path: Path | None` (the `scorecard.json` path for partials, `None` for full).

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/Scripts/python -m pytest tests/test_promote.py tests/test_experiment_layout.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nmr/promote.py tests/test_promote.py
git commit -m "feat(promote): train_only scope, fit isolation, cross-check scoring, atomic staging"
```

---

### Task 9: `nmr/families.py` wrapper + `nmr/meta.py` / `nmr/submission.py` retarget

**Files:**
- Modify: `nmr/families.py`, `nmr/meta.py`, `nmr/submission.py`
- Test: `tests/test_families.py`, `tests/test_meta.py`, `tests/test_submission.py`

**Interfaces:**
- Consumes: `nmr.lifecycle`, `nmr.paths`, `nmr.experiment_store`
- Produces (public names preserved — compatibility wrapper):
  - `FullVersion` → re-exported alias of `lifecycle.ExportVersion` (full scope)
  - `scan_full_versions(models_dir) -> dict[str, FullVersion]` — over `lifecycle.scan_valid_exports(family, "full")`
  - `load_full_version(models_dir, family) -> FullVersion | None` — the current-pointer export, or None
  - `family_has_full_version(models_dir, family) -> bool`
  - `available_slots(models_dir, family) -> list[str]` — valid + all slot dirs
  - `full_manifest_path` / `FULL_MANIFEST_NAME` — deprecated compat shims (logged), removed in Task 11
  - `validate_family_name(family)` — delegates to `paths.validate_slug`

- [ ] **Step 1: Write failing tests (extend `tests/test_families.py`)**

```python
def test_scan_full_versions_over_experiments(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    _write_export("fam1", "full", "a" * 64)  # helper from test_lifecycle or local
    versions = families.scan_full_versions(tmp_path / "experiments")
    assert set(versions) == {"fam1"}
    assert versions["fam1"].run_id == "a" * 64

def test_full_version_requires_pointer():
    _write_export("fam2", "full", "b" * 64)  # no current.json
    assert families.load_full_version(tmp_path / "experiments", "fam2") is None
    paths.current_pointer_path("fam2").write_text(json.dumps({"run_id": "b" * 64}))
    assert families.load_full_version(tmp_path / "experiments", "fam2") is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/Scripts/python -m pytest tests/test_families.py -q`
Expected: FAIL — old implementation reads `artifacts/models/`

- [ ] **Step 3: Reimplement as a thin wrapper delegating to `nmr/lifecycle.py`**

```python
# nmr/families.py (reimplement bodies; keep public names)
def scan_full_versions(models_dir: Path) -> dict[str, FullVersion]:
    # models_dir is accepted for call-site compat; the layout is EXPERIMENTS_ROOT.
    found = {}
    for slug_dir in sorted(paths.EXPERIMENTS_ROOT.iterdir()):
        if not slug_dir.is_dir() or not paths.SLUG_RE.fullmatch(slug_dir.name):
            continue
        versions = lifecycle.scan_valid_exports(slug_dir.name, "full")
        for v in versions:
            found[v.family] = v
    return found


def load_full_version(models_dir: Path, family: str) -> FullVersion | None:
    status = lifecycle.current_full_status(family)
    if status != "full":
        return None
    pointer = json.loads(paths.current_pointer_path(family).read_text(encoding="utf-8"))
    return lifecycle.valid_export(family, "full", pointer["run_id"])


def family_has_full_version(models_dir: Path, family: str) -> bool:
    return lifecycle.current_full_status(family) == "full"


def available_slots(models_dir: Path, family: str) -> list[str]:
    base = paths.export_dir(family, "full", "x").parent
    if not base.is_dir():
        return []
    return sorted(e.name for e in base.iterdir()
                  if e.is_dir() and not e.name.startswith(".tmp-"))
```

`nmr/meta.py` and `nmr/submission.py`: replace path resolution with `paths.run_dir(...)` / `paths.export_dir(...)` (mechanical — every `artifacts/registry`/`artifacts/models` reference becomes the `paths.*` equivalent).

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/Scripts/python -m pytest tests/test_families.py tests/test_meta.py tests/test_submission.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nmr/families.py nmr/meta.py nmr/submission.py tests/test_families.py tests/test_meta.py tests/test_submission.py
git commit -m "refactor(families): wrapper over lifecycle; retarget meta/submission paths"
```

---

### Task 10: Dashboard — lifecycle fields, partial rows, row IDs

**Files:**
- Modify: `nmr/dashboard.py`, `dashboard_ui/charts.py`, `dashboard_ui/report.py`, `dashboard_ui/static/app.js`
- Test: `tests/test_dashboard.py`, `tests/test_dashboard_ui.py`

**Interfaces:**
- Consumes: `nmr.lifecycle.derive_stage` / `scan_valid_exports` / `load_staked_record`, `nmr.paths`
- Produces:
  - Unified schema gains `display_name: str`, `lifecycle_stage: str`, `current_full_status: str`
  - `EVALUABLE_ROWS` becomes `pl.col("source").is_in(["trained", "trained_legacy"])` (partial + full excluded from ranking)
  - Row IDs: `family::full::<run_id>` and `family::partial::<run_id>` (no bare `family::full`)

- [ ] **Step 1: Write failing tests (extend `tests/test_dashboard.py`)**

```python
def test_payload_carries_lifecycle(tmp_path, monkeypatch, synthetic_registry_with_export):
    payload = dash.build_tournament_payload(tmp_path / "experiments")
    family_row = next(r for r in payload["rows"] if r["family"] == "fam1")
    assert family_row["lifecycle_stage"] in dash_lifecycle_stages()
    assert family_row["display_name"]  # from meta.json
    partial_rows = [r for r in payload["rows"] if "::partial::" in r["model_id"]]
    assert partial_rows and all(r["source"] == "partial" for r in partial_rows)
    assert all(r["model_id"].startswith("fam1::") for r in partial_rows)

def test_partial_rows_not_evaluable():
    import nmr.dashboard as dash
    frame = ...  # unified leaderboard with a partial row
    ranked = frame.filter(dash.EVALUABLE_ROWS)
    assert not any("::partial::" in mid for mid in ranked["model_id"])
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -q`
Expected: FAIL — schema lacks the fields

- [ ] **Step 3: Implement**

- `UNIFIED_SCHEMA` (dashboard.py:67): add `display_name`, `lifecycle_stage`, `current_full_status` columns.
- Row assembly (per family): read `meta.json` via `lifecycle.load_staked_record` + display_name; call `derive_stage(family, staked)`; fill the new columns; render one `family::full::<run_id>` row per valid full export and one `family::partial::<run_id>` per valid partial export.
- `EVALUABLE_ROWS` (dashboard.py:121): `pl.col("source").is_in(["trained", "trained_legacy"])`.
- Scan source: replace `artifacts/registry` + `artifacts/models` scanning with `experiments/` iteration (`paths.EXPERIMENTS_ROOT`), reading `run.json` per run and exports via `lifecycle`.
- `dashboard_ui/report.py` + `charts.py` + `static/app.js`: render the lifecycle badge from `lifecycle_stage` (+ `current_full_status` when it's `degraded`, + `stale` flag), display_name in labels, and escape display_name (the renderer already escapes — verify in tests).

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py tests/test_dashboard_ui.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nmr/dashboard.py dashboard_ui tests/test_dashboard.py tests/test_dashboard_ui.py
git commit -m "feat(dashboard): lifecycle stage, display_name, diagnostic-only partial rows"
```

---

### Task 11: CLIs + config semantics + legacy-shim removal

**Files:**
- Modify: `train_first_model.py`, `run_campaign.py`, `promote_model.py`, `rehearse_promotion.py`, `generate_dashboard.py`, `dashboard_ui/app.py`, `nmr/config.py` (artifacts_dir comment), `configs/example.yaml`
- Test: `tests/test_scripts.py` (retarget), `tests/test_config.py`

**Interfaces:**
- Consumes: all prior tasks
- Produces: CLI defaults resolve through `nmr/paths`; legacy shims (`full_manifest_path`, `FULL_MANIFEST_NAME`, `resolve_champion_run_id`) removed and their call sites updated

- [ ] **Step 1: Retarget CLI defaults (test-first)**

For each CLI, update the default `registry_dir` / `models_dir` resolution to the new layout and add/update a test in `tests/test_scripts.py` that asserts the resolved path:

```python
# train_first_model.py / run_campaign.py — registry construction
registry = RunRegistry(paths.EXPERIMENTS_ROOT)   # instead of DEFAULT_MODELS_DIR.parent / "registry"
# promote_model.py / rehearse_promotion.py — family/exports
slot_dir = paths.export_dir(family, "full" if not args.train_only else "partial", run_id)
# generate_dashboard.py / dashboard_ui/app.py
data_root = paths.EXPERIMENTS_ROOT
```

Update `run_campaign.py` per-run routing: each constituent run resolves its own `experiment_dir(config.run.name)` (campaign evidence stays under `artifacts/campaigns/`).

Remove the compat shims (`full_manifest_path`, `FULL_MANIFEST_NAME`, `resolve_champion_run_id`) and update their call sites (dashboard, meta, scripts) to the new APIs.

`nmr/config.py:283`: update the `artifacts_dir` docstring/comment to "shared machine cache root (cache/reports/campaigns); run/export outputs derive from EXPERIMENTS_ROOT". Update `configs/example.yaml`'s annotated `run.artifacts_dir` description.

- [ ] **Step 2: Run targeted tests**

Run: `./.venv/Scripts/python -m pytest tests/test_scripts.py tests/test_config.py -q`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add train_first_model.py run_campaign.py promote_model.py rehearse_promotion.py generate_dashboard.py dashboard_ui/app.py nmr/config.py configs/example.yaml tests/test_scripts.py tests/test_config.py
git commit -m "refactor(cli): resolve defaults via nmr/paths; drop legacy shims"
```

---

### Task 12: Docs (SSOT) + lifecycle workflow doc

**Files:**
- Create: `docs/02-strategy/model-lifecycle.md`
- Modify: `AGENTS.md`, `ARCHITECTURE.md` (incl. the stale artifact-layout section at `ARCHITECTURE.md:292`), `CONTRIBUTING.md`, `README.md`

**Interfaces:**
- Consumes: the final behavior of Tasks 1–11
- Produces: the SSOT documentation per the spec's §13

- [ ] **Step 1: Write `docs/02-strategy/model-lifecycle.md`**

Sections (each must match the implemented behavior exactly):
1. The lifecycle — the six states, how they are derived, badge precedence, `stale`/`degraded` surfacing.
2. The workflow — research → partial (train-only export + local cross-check) → upload the partial to Numerai → compare platform diagnostics vs `scorecard.json` → full (train+validation) → stake (record in `meta.json`).
3. Naming — slug template `<theme>-<backend>-<vN>`, `display_name` in `meta.json`, hash-only-in-tooltips rule.
4. Layout — the §3 tree, what is git-tracked vs ignored, rebuild-identity fields and the fingerprint-refusal rule.
5. How we operate — family creation, immutability, re-promotion rejection, single-writer champion, upload/stake being manual acts.

- [ ] **Step 2: Update the four SSOT files**

- `AGENTS.md`: toolkit table (registry → `experiments/` + `nmr/paths` + `nmr/lifecycle`); operational hazards — add the **junction/worktree deletion hazard** ("never junction `artifacts/` or `experiments/` into scratch worktrees; `git worktree remove` recurses through junctions") and the **single-writer champion invariant**; update path references (`artifacts/registry/` → `experiments/`).
- `ARCHITECTURE.md`: artifact/registry schema sections + the stale layout section at `ARCHITECTURE.md:292`; the §N registry/artifact layout and §W dashboard window definitions.
- `CONTRIBUTING.md`: any commands/expected paths that changed.
- `README.md`: annotated project tree — replace the `artifacts/{runs,registry,models}` description with `experiments/`.

- [ ] **Step 3: Full verification gate**

Run: `./.venv/Scripts/python -m ruff check . && ./.venv/Scripts/python -m pytest -q`
Expected: PASS (report skips for real-data tests without `data/v5.3/`)

- [ ] **Step 4: Commit**

```bash
git add docs/02-strategy/model-lifecycle.md AGENTS.md ARCHITECTURE.md CONTRIBUTING.md README.md
git commit -m "docs: model lifecycle workflow + experiment layout (SSOT)"
```

---

## Self-Review Notes (filled during plan authoring)

- **Spec coverage:** §2 domain model/vocabulary → Tasks 2, 5; §3 layout/inventory → Tasks 1, 3, 5; §3.1 rebuild identity → Task 7; §4 naming → Tasks 2, 10, 12; §5 lifecycle → Tasks 2, 10; §6 promotion → Task 8; §7 cross-check → Tasks 4, 8; §8 dashboard → Task 10; §9 storage/retarget → Tasks 1, 5, 6, 7, 11; §10 champion → Task 6; §11 determinism → Task 7 (fingerprint persistence) + Task 1 (no path in hashes); §12 tests → each task's tests; §13 files → all tasks; §14 out-of-scope → no task (correct).
- **Ordering rationale:** pure modules first (1–4), storage (5–6), pipeline (7–8), discovery/dashboard (9–10), CLIs (11), docs (12). Each task ends green with a commit.
- **Known implementer judgment calls (flagged, not hidden):** the exact extraction of `evaluate_model` internals (Task 4 — guarded by the existing scorecard tests), the runner's manifest assembly location, and the dashboard row-assembly refactor (Task 10 — guarded by existing dashboard tests).
