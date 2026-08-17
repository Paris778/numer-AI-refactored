# Model Family Full-Version Marker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the dashboard a discoverable research↔full (train+validation) version marker: `nmr/families.py` read-only discovery of `artifacts/models/<family>/full/manifest.json`, unified-schema columns, HTML FULL badges + Promoted Full Versions group, Streamlit columns, and SSOT docs sync.

**Architecture:** A model "family" is identified by its research runs' `run.name` (e.g. `brb1-xgb-v6`). Promotion to a full version is marked by a valid manifest at `artifacts/models/<family>/full/manifest.json` (family == dir name, lowercase, artifact file must exist). `nmr/families.py` owns scan/validate (read-only). `nmr/dashboard.py` scans once, stamps trained rows with `has_full_version` via set membership, appends `source="full"` rows with null metrics, stamps them `FULL` in `evaluate_gate_status`, and exposes the single `EVALUABLE_ROWS` chart predicate. `generate_dashboard.py` renders the FULL chip + pinned full group; `dashboard_app.py` broadens the Streamlit source filter and pins full rows. Spec: `docs/superpowers/specs/2026-08-17-model-families-full-version-marker-design.md`.

**Tech Stack:** Python 3.11+, Polars 1.41+, pytest, ruff (E/F/I/UP @120). No new dependencies.

## Global Constraints

- **Tested boundary:** all business logic lives in `nmr/`; `generate_dashboard.py` / `dashboard_app.py` are thin control planes (rendering/wiring only).
- **Registry immutability:** never write to `artifacts/registry/`. `nmr/families.py` is strictly read-only.
- **Canonical hashes untouched:** no timing/absolute-path fields enter `run_id`, `canonical_scorecards_bytes()`, or cache keys. `nmr/families.py` is display metadata only.
- **`nmr/` UI-free:** `nmr/` must never import plotly or streamlit.
- **Lint:** `./.venv/Scripts/python -m ruff check .` must pass (E/F/I/UP, line-length 120).
- **AGENTS.md test-count claim (CRITICAL):** `tests/test_docs_hygiene.py::test_docs_test_count_matches_suite` runs `pytest --collect-only` on every full-suite run and fails if the "772 tests" claim in AGENTS.md (line 33) doesn't equal the collected count. **Any task that adds tests MUST bump that claim in the same commit** (step "Bump the test-count claim" below).
- **Package-API test (CRITICAL):** `tests/test_package_api.py` auto-discovers every non-underscore `nmr/*.py` module and requires `nmr/__init__.py` to re-export every name in each module's `__all__` (imports AND `__all__`). Adding `nmr/families.py` and new `nmr/dashboard.py` exports requires `nmr/__init__.py` updates in the same commit.
- **Test isolation:** `load_unified_leaderboard(..., benchmark_path=False)` must not read live repo CSVs. `models_dir` defaults must not change behavior when `artifacts/models/` is absent (empty scan).
- **Windows venv:** always `./.venv/Scripts/python -m ...` — never the `Scripts/pip` shim.
- **Git:** commit per task. Do NOT `git push`. The LF→CRLF warning on commit is benign; ignore it.
- **Verification gate per task:** after the targeted tests pass and the count is bumped, run the FULL suite (`./.venv/Scripts/python -m pytest -q`) before committing — never claim green without executing it.
- **Every commit green (review item):** each task's commit must pass the full suite (ruff + pytest). No red intermediate commits — the `nmr/__init__.py` re-exports for `nmr.families` are therefore added in Task 1 itself, not deferred to Task 2.

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `nmr/families.py` | Read-only family/full-version discovery + manifest validation | Create (Task 1) |
| `tests/test_families.py` | Unit tests for `nmr/families.py` | Create (Task 1) |
| `nmr/dashboard.py` | UNIFIED_SCHEMA +3 cols, `load_unified_leaderboard(models_dir, scan-once, full rows)`, `evaluate_gate_status` FULL branch, `EVALUABLE_ROWS` | Modify (Task 2) |
| `nmr/__init__.py` | Re-export families + new dashboard symbols | Modify (Task 2) |
| `tests/test_dashboard.py` | Leaderboard/gate/predicate tests + rendering tests | Modify (Tasks 2–4) |
| `generate_dashboard.py` | FULL chip, Promoted Full Versions group, `_bar_input` filter | Modify (Task 3) |
| `dashboard_app.py` | `_LEADERBOARD_SCHEMA` +3 cols, source filter, pinned sort, chart filter | Modify (Task 4) |
| `AGENTS.md` | Test-count claim (+ toolkit row) | Modify (Tasks 1–4 count; Task 5 row) |
| `ARCHITECTURE.md` | families module contract + manifest schema + schema columns | Modify (Task 5) |
| `README.md` | Annotated-tree line for `artifacts/models/` | Modify (Task 5) |

---

### Task 1: `nmr/families.py` — read-only family layer

**Files:**
- Create: `nmr/families.py`
- Test: `tests/test_families.py`
- Modify: `nmr/__init__.py` (families re-exports — review item: keeps `test_package_api.py` green on this commit)
- Modify: `AGENTS.md` (test-count claim only)

**Interfaces:**
- Consumes: `REPO_ROOT` from `nmr.config` (already exists).
- Produces (used by Task 2): `DEFAULT_MODELS_DIR: Path`, `FAMILY_DIR_NAME = "models"`, `FULL_DIR_NAME = "full"`, `FULL_MANIFEST_NAME = "manifest.json"`, `FullVersion` (frozen dataclass with fields `family: str`, `manifest_path: Path`, `artifact_path: str | None`, `promoted_from_run_id: str | None`, `promoted_at: str | None`, `config: dict[str, Any]`), `full_manifest_path(models_dir: Path, family: str) -> Path`, `load_full_version(models_dir: Path, family: str) -> FullVersion | None`, `scan_full_versions(models_dir: Path) -> dict[str, FullVersion]`, `family_has_full_version(models_dir: Path, family: str) -> bool`.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_families.py` with exactly:

```python
"""Unit tests for nmr.families — the read-only model-family / full-version layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import nmr.families as fam


def _write_full_manifest(
    models_dir: Path,
    family: str,
    *,
    family_name: str | None = None,
    training_scope: str = "full",
    promoted_from_run_id: str = "a" * 64,
    artifact_path: str | None = "predict.pkl",
    body: str | None = None,
) -> Path:
    full_dir = models_dir / family / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    # Only write artifacts for safe relative paths — never write outside the
    # tmp models dir (absolute / ../ / empty artifact_paths are validation
    # fixtures and must not touch the real filesystem).
    if artifact_path is not None and artifact_path:
        candidate = Path(artifact_path)
        if not candidate.is_absolute() and ".." not in candidate.parts:
            (full_dir / candidate).write_text("weights", encoding="utf-8")
    manifest = body
    if manifest is None:
        manifest = json.dumps(
            {
                "family": family_name if family_name is not None else family,
                "training_scope": training_scope,
                "promoted_from_run_id": promoted_from_run_id,
                "promoted_at": "2026-08-17T12:00:00Z",
                "artifact_path": artifact_path,
                "config": {"run": {"name": family}},
            }
        )
    path = full_dir / "manifest.json"
    path.write_text(manifest, encoding="utf-8")
    return path


def test_full_manifest_path_resolves(tmp_path: Path) -> None:
    assert fam.full_manifest_path(tmp_path, "brb1-xgb-v6") == (
        tmp_path / "brb1-xgb-v6" / "full" / "manifest.json"
    )


@pytest.mark.parametrize("bad", ["../evil", "a/b", "a b", "a:b", "", "ModelA", "A"])
def test_full_manifest_path_rejects_invalid_family(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        fam.full_manifest_path(tmp_path, bad)


@pytest.mark.parametrize("bad", ["../evil", "a/b", "a b", "a:b", "", "ModelA", "A"])
def test_load_full_version_rejects_invalid_family(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        fam.load_full_version(tmp_path, bad)


def test_load_full_version_happy_path(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6")
    v = fam.load_full_version(tmp_path, "brb1-xgb-v6")
    assert v is not None
    assert v.family == "brb1-xgb-v6"
    assert v.promoted_from_run_id == "a" * 64
    assert v.manifest_path == tmp_path / "brb1-xgb-v6" / "full" / "manifest.json"
    assert v.config == {"run": {"name": "brb1-xgb-v6"}}


def test_load_full_version_missing_manifest_returns_none(tmp_path: Path) -> None:
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_corrupt_json_returns_none(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6", body="{not json")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_family_mismatch_returns_none(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6", family_name="other-family")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_wrong_scope_returns_none(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6", training_scope="research")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_missing_run_id_returns_none(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6", promoted_from_run_id="")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


@pytest.mark.parametrize(
    "bad_artifact",
    ["", "../predict.pkl", "C:\\abs\\predict.pkl", "/abs/predict.pkl"],
)
def test_load_full_version_rejects_invalid_artifact_path(
    tmp_path: Path, bad_artifact: str
) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6", artifact_path=bad_artifact)
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_rejects_hollow_promotion(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6", artifact_path="predict.pkl")
    (tmp_path / "brb1-xgb-v6" / "full" / "predict.pkl").unlink()
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_scan_full_versions_only_valid(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6")
    _write_full_manifest(tmp_path, "brb1-lgbm-v6")
    # invalid: corrupt json, mixed-case dir, dir with no manifest
    _write_full_manifest(tmp_path, "brb1-xgb-v5", body="{not json")
    mixed = tmp_path / "Brb1-Xgb-V4" / "full"
    mixed.mkdir(parents=True)
    (mixed / "manifest.json").write_text(
        json.dumps(
            {
                "family": "Brb1-Xgb-V4",
                "training_scope": "full",
                "promoted_from_run_id": "b" * 64,
                "artifact_path": "predict.pkl",
            }
        ),
        encoding="utf-8",
    )
    (mixed / "predict.pkl").write_text("x", encoding="utf-8")
    (tmp_path / "lonely" / "full").mkdir(parents=True)
    found = fam.scan_full_versions(tmp_path)
    assert set(found) == {"brb1-xgb-v6", "brb1-lgbm-v6"}


def test_scan_full_versions_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert fam.scan_full_versions(tmp_path / "nope") == {}


def test_family_has_full_version(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6")
    assert fam.family_has_full_version(tmp_path, "brb1-xgb-v6") is True
    assert fam.family_has_full_version(tmp_path, "brb1-lgbm-v6") is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./.venv/Scripts/python -m pytest -q tests/test_families.py -v`
Expected: FAIL — collection error `ModuleNotFoundError: No module named 'nmr.families'`.

- [ ] **Step 3: Implement `nmr/families.py`**

Create `nmr/families.py` with exactly:

```python
"""Read-only model-family / full-version discovery layer.

A model "family" is identified by its research runs' ``run.name`` (e.g.
``brb1-xgb-v6``). Promotion to a full version (trained on train+validation)
is marked by a valid manifest at ``artifacts/models/<family>/full/manifest.json``.
This module is strictly read-only: it never writes to ``artifacts/models/``
or ``artifacts/registry/``; the promotion writer is a future workstream.
Nothing here enters a canonical hash — it is display metadata only.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nmr.config import REPO_ROOT

logger = logging.getLogger("nmr.families")

FAMILY_DIR_NAME = "models"
FULL_DIR_NAME = "full"
FULL_MANIFEST_NAME = "manifest.json"
DEFAULT_MODELS_DIR = REPO_ROOT / "artifacts" / FAMILY_DIR_NAME

# Lowercase-only: prevents case-collision overwrites on case-insensitive
# filesystems (Windows NTFS / macOS APFS) and matches the run.name convention.
_FAMILY_NAME_RE = re.compile(r"^[a-z0-9_-]+$")

__all__ = [
    "DEFAULT_MODELS_DIR",
    "FAMILY_DIR_NAME",
    "FULL_DIR_NAME",
    "FULL_MANIFEST_NAME",
    "FullVersion",
    "family_has_full_version",
    "full_manifest_path",
    "load_full_version",
    "scan_full_versions",
]


@dataclass(frozen=True)
class FullVersion:
    """A validated full-version promotion of a model family."""

    family: str
    manifest_path: Path
    artifact_path: str | None
    promoted_from_run_id: str | None
    promoted_at: str | None
    config: dict[str, Any]


def _require_valid_family(family: str) -> str:
    if not _FAMILY_NAME_RE.fullmatch(family):
        raise ValueError(
            f"invalid family name {family!r}: must match {_FAMILY_NAME_RE.pattern}"
        )
    return family


def full_manifest_path(models_dir: Path, family: str) -> Path:
    """Resolve ``models_dir/<family>/full/manifest.json`` (family validated)."""
    _require_valid_family(family)
    return Path(models_dir) / family / FULL_DIR_NAME / FULL_MANIFEST_NAME


def _validate_artifact(manifest_dir: Path, artifact_path: object) -> str | None:
    """Artifact must be a non-empty relative path (no /, drive, or ..) whose
    file exists beside the manifest. None on any violation."""
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        return None
    candidate = Path(artifact_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    if not (manifest_dir / candidate).is_file():
        return None
    return candidate.as_posix()


def load_full_version(models_dir: Path, family: str) -> FullVersion | None:
    """Load and validate the family's full-version manifest, or None.

    None when: manifest missing, JSON invalid, or any validation rule fails
    (family != dir name, scope != "full", missing run id, missing/invalid
    artifact). A warning is logged on structural violations.
    """
    _require_valid_family(family)
    path = full_manifest_path(models_dir, family)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("nmr.families: corrupt manifest at %s", path)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("family") != family:
        logger.warning(
            "nmr.families: manifest family %r != dir %r",
            payload.get("family"), family,
        )
        return None
    if payload.get("training_scope") != "full":
        return None
    run_id = payload.get("promoted_from_run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return None
    artifact = _validate_artifact(path.parent, payload.get("artifact_path"))
    if artifact is None:
        logger.warning(
            "nmr.families: hollow or invalid artifact for %s (artifact_path=%r)",
            family, payload.get("artifact_path"),
        )
        return None
    config = payload.get("config")
    return FullVersion(
        family=family,
        manifest_path=path,
        artifact_path=artifact,
        promoted_from_run_id=run_id,
        promoted_at=payload.get("promoted_at"),
        config=config if isinstance(config, dict) else {},
    )


def scan_full_versions(models_dir: Path) -> dict[str, FullVersion]:
    """Return ``{family: FullVersion}`` for every valid manifest under models_dir."""
    base = Path(models_dir)
    if not base.is_dir():
        return {}
    found: dict[str, FullVersion] = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or not _FAMILY_NAME_RE.fullmatch(entry.name):
            continue
        version = load_full_version(base, entry.name)
        if version is not None:
            found[version.family] = version
    return found


def family_has_full_version(models_dir: Path, family: str) -> bool:
    """True when the family has a valid full-version manifest."""
    return load_full_version(models_dir, family) is not None
```

- [ ] **Step 3b: Re-export `nmr.families` from `nmr/__init__.py`**

`tests/test_package_api.py` auto-discovers `nmr/families.py` and fails unless every name in its `__all__` is importable from the `nmr` package. Add the re-export NOW so this commit stays green.

(a) In `nmr/__init__.py`, insert the import block between the `.evaluation` and `.features` import blocks:

```python
from .families import (
    DEFAULT_MODELS_DIR,
    FAMILY_DIR_NAME,
    FULL_DIR_NAME,
    FULL_MANIFEST_NAME,
    FullVersion,
    family_has_full_version,
    full_manifest_path,
    load_full_version,
    scan_full_versions,
)
```

(b) Add the same names to `__all__` (insert alphabetically; exact position does not matter functionally, keep tidy): `"DEFAULT_MODELS_DIR"`, `"FAMILY_DIR_NAME"`, `"FULL_DIR_NAME"`, `"FULL_MANIFEST_NAME"`, `"FullVersion"`, `"family_has_full_version"`, `"full_manifest_path"`, `"load_full_version"`, `"scan_full_versions"`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `./.venv/Scripts/python -m pytest -q tests/test_families.py`
Expected: PASS (29 collected).

- [ ] **Step 5: Lint the new files**

Run: `./.venv/Scripts/python -m ruff check nmr/families.py tests/test_families.py`
Expected: no violations.

- [ ] **Step 6: Bump the AGENTS.md test-count claim**

Run: `./.venv/Scripts/python -m pytest --collect-only -q 2>&1 | tail -2`
Read the final line — it reports `N tests collected` (expect 772 + 29 = 801).
Edit `AGENTS.md` line 33: replace the current count (`772 tests`) with the reported `N tests` (keep the surrounding sentence intact).

- [ ] **Step 7: Run the full suite**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS, including `tests/test_docs_hygiene.py::test_docs_test_count_matches_suite` (count now matches) and `tests/test_package_api.py::test_nmr_package_reexports_all_module_public_symbols` (families re-exported in Step 3b).

- [ ] **Step 8: Commit**

```bash
git add nmr/families.py nmr/__init__.py tests/test_families.py AGENTS.md
git commit -m "feat(families): read-only full-version manifest discovery layer"
```

---

### Task 2: `nmr/dashboard.py` — schema columns, scan-once leaderboard, FULL gate status, EVALUABLE_ROWS; `nmr/__init__.py` re-exports

**Files:**
- Modify: `nmr/dashboard.py` (UNIFIED_SCHEMA ~line 52; header imports ~line 20; `__all__` ~line 33; `load_unified_leaderboard` ~line 178; `evaluate_gate_status` ~line 341)
- Modify: `nmr/__init__.py` (EVALUABLE_ROWS re-export only — families symbols already exported in Task 1)
- Test: `tests/test_dashboard.py` (append tests; reuse `_registry_entry`/`_write_registry` helpers already in the file)
- Modify: `AGENTS.md` (test-count claim)

**Interfaces:**
- Consumes: Task 1's `nmr.families` (`DEFAULT_MODELS_DIR`, `scan_full_versions`, `FullVersion`).
- Produces: `UNIFIED_SCHEMA` with new columns `family`/`training_scope`/`has_full_version`; `EVALUABLE_ROWS: pl.Expr`; `load_unified_leaderboard(registry_dir, benchmark_path=None, reports_dir=None, models_dir=None)` appending `source="full"` rows; `evaluate_gate_status` stamping `FULL`. Tasks 3–4 consume `EVALUABLE_ROWS` and the new row fields.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard.py`:

```python
def test_unified_schema_has_family_columns() -> None:
    for col in ("family", "training_scope", "has_full_version"):
        assert col in dash.UNIFIED_SCHEMA.names()


def _write_models_dir(tmp_path: Path, families: dict[str, dict]) -> Path:
    """families: {family: manifest-dict}; a predict.pkl artifact is auto-created."""
    models = tmp_path / "models"
    for family, manifest in families.items():
        full = models / family / "full"
        full.mkdir(parents=True)
        (full / "predict.pkl").write_text("weights", encoding="utf-8")
        (full / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return models


def _full_manifest_dict(family: str, run_id: str) -> dict:
    return {
        "family": family,
        "training_scope": "full",
        "promoted_from_run_id": run_id,
        "promoted_at": "2026-08-17T12:00:00Z",
        "artifact_path": "predict.pkl",
        "config": {
            "run": {"name": family},
            "data": {"feature_set": "all", "feature_subset": "medium", "targets": ["target"]},
            "model": {"backend": "xgboost", "preset": "fast"},
        },
    }


def test_load_unified_leaderboard_family_columns_and_full_rows(tmp_path: Path) -> None:
    entry = _registry_entry("a" * 64)
    entry["manifest"]["config"]["run"]["name"] = "brb1-xgb-v6"
    _write_registry(tmp_path, [entry])
    models = _write_models_dir(
        tmp_path, {"brb1-xgb-v6": _full_manifest_dict("brb1-xgb-v6", "a" * 64)}
    )
    frame = dash.load_unified_leaderboard(
        tmp_path, benchmark_path=False, models_dir=models
    )
    rows = {r["model_id"]: r for r in frame.to_dicts()}
    trained = rows["a" * 64]
    assert trained["family"] == "brb1-xgb-v6"
    assert trained["training_scope"] == "research"
    assert trained["has_full_version"] is True
    full = rows["brb1-xgb-v6::full"]
    assert full["source"] == "full"
    assert full["run_name"] == "brb1-xgb-v6"
    assert full["training_scope"] == "full"
    assert full["has_full_version"] is False
    assert full["corr"] is None
    assert full["corr_sharpe_ac"] is None
    assert full["backend"] == "xgboost"
    assert full["feature_subset"] == "medium"
    assert full["run_dir"] == str(models / "brb1-xgb-v6" / "full")


def test_load_unified_leaderboard_scan_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64), _registry_entry("b" * 64)])
    calls = {"n": 0}
    real_scan = dash.scan_full_versions

    def counting_scan(models_dir: Path) -> dict:
        calls["n"] += 1
        return real_scan(models_dir)

    monkeypatch.setattr(dash, "scan_full_versions", counting_scan)
    dash.load_unified_leaderboard(
        tmp_path, benchmark_path=False, models_dir=tmp_path / "models"
    )
    assert calls["n"] == 1


def test_load_unified_leaderboard_missing_models_dir(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64)])
    frame = dash.load_unified_leaderboard(
        tmp_path, benchmark_path=False, models_dir=tmp_path / "nope"
    )
    assert frame.height == 1
    assert frame.row(0, named=True)["has_full_version"] is False


def test_load_unified_leaderboard_dangling_lineage_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64)])
    models = _write_models_dir(
        tmp_path,
        {"orphan-family": _full_manifest_dict("orphan-family", "f" * 64)},
    )
    with caplog.at_level(logging.WARNING, logger="nmr.dashboard"):
        frame = dash.load_unified_leaderboard(
            tmp_path, benchmark_path=False, models_dir=models
        )
    assert "orphan-family" in caplog.text  # dangling lineage warned
    rows = {r["model_id"]: r for r in frame.to_dicts()}
    assert "orphan-family::full" in rows  # still rendered


def test_gate_status_full_rows_stamped_full(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        [{"model_id": "brb1-xgb-v6::full", "source": "full", "corr": None,
          "corr_sharpe_ac": None, "fnc": None, "deflated_sharpe": None,
          "gain_to_pain_ratio": None, "cagr_1y": None, "turnover_mean": None}],
        schema=dash.UNIFIED_SCHEMA,
        strict=False,
    )
    out = dash.evaluate_gate_status(frame, _GATE_YAML, tmp_path / "champion.json").row(0, named=True)
    assert out["status"] == "FULL"
    assert out["gate_corr"] is None
    assert out["gate_turnover_mean"] is None


def test_evaluable_rows_predicate() -> None:
    frame = pl.DataFrame(
        [{"model_id": "a", "source": "trained"},
         {"model_id": "b", "source": "benchmark"},
         {"model_id": "c::full", "source": "full"}],
        schema=dash.UNIFIED_SCHEMA,
        strict=False,
    )
    keep = frame.filter(dash.EVALUABLE_ROWS).get_column("model_id").to_list()
    assert keep == ["a", "b"]
```

(These tests reference `_GATE_YAML` and `logging`, which already exist in `tests/test_dashboard.py`.)

- [ ] **Step 2: Run the failing tests**

Run: `./.venv/Scripts/python -m pytest -q tests/test_dashboard.py -k "family or full or evaluable or scan_once" -v`
Expected: FAIL — `AttributeError: 'Schema' object has no attribute ...` / missing `EVALUABLE_ROWS` / missing `scan_full_versions` import.

- [ ] **Step 3: Implement the `nmr/dashboard.py` changes**

(a) Header imports — in the existing `from nmr.config import REPO_ROOT` area, add:

```python
from nmr.families import DEFAULT_MODELS_DIR, scan_full_versions
```

(b) `__all__` — add `"EVALUABLE_ROWS",` (alphabetically before `"UNIFIED_SCHEMA"`).

(c) `UNIFIED_SCHEMA` — in the schema dict, immediately after `"model_id": pl.String, "source": pl.String, "run_name": pl.String,` insert:

```python
        "family": pl.String, "training_scope": pl.String, "has_full_version": pl.Boolean,
```

(d) After the `UNIFIED_SCHEMA` definition, add the chart predicate:

```python
# Single predicate for every chart / candidate-selection path: rows that carry
# validation metrics. Source-based (never null) so benchmark rows (null
# training_scope) stay visible in charts; full rows (in-sample metrics) are
# excluded everywhere.
EVALUABLE_ROWS: pl.Expr = pl.col("source") != "full"
```

(e) `load_unified_leaderboard` — signature and body. Change the signature to:

```python
def load_unified_leaderboard(
    registry_dir: Path,
    benchmark_path: Path | None | bool = None,
    reports_dir: Path | None = None,
    models_dir: Path | None = None,
) -> pl.DataFrame:
```

Immediately after `rows: list[dict] = []`, add the single family scan (scan-once):

```python
    full_versions = scan_full_versions(
        Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    )
    promoted_families = set(full_versions)
```

In the trained-row dict (the `rows.append({...})` for registry runs), add:

```python
                "family": run_cfg.get("name", "unknown"),
                "training_scope": "research",
                "has_full_version": run_cfg.get("name", "unknown") in promoted_families,
```

Immediately AFTER the `for run_file in ...` loop and BEFORE `resolved = resolve_benchmark_path(...)`, append the full rows:

```python
    for family in sorted(full_versions):
        version = full_versions[family]
        if not (Path(registry_dir) / version.promoted_from_run_id).is_dir():
            logger.warning(
                "nmr.dashboard: full version %s lineage dangling "
                "(promoted_from_run_id %s not in registry)",
                family, version.promoted_from_run_id,
            )
        full_row = dict.fromkeys(UNIFIED_SCHEMA.names())  # all metric cells null
        cfg_data = (version.config.get("data") or {}) if version.config else {}
        cfg_model = (version.config.get("model") or {}) if version.config else {}
        targets = cfg_data.get("targets") or []
        full_row.update(
            {
                "model_id": f"{family}::full",
                "source": "full",
                "run_name": family,
                "family": family,
                "training_scope": "full",
                "has_full_version": False,
                "backend": cfg_model.get("backend"),
                "preset": cfg_model.get("preset"),
                "feature_set": cfg_data.get("feature_set"),
                "feature_subset": cfg_data.get("feature_subset"),
                "n_targets": len(targets) if targets else None,
                "targets": ", ".join(targets) if targets else None,
                "run_dir": str(version.manifest_path.parent),
            }
        )
        rows.append(full_row)
```

(f) `evaluate_gate_status` — add the FULL branch at the top of the row loop, before the `if row["source"] == "benchmark":` branch:

```python
        if row["source"] == "full":
            status = "FULL"
        elif row["source"] == "benchmark":
```

(The `gate_*` receipts for full rows are all `None` automatically — every metric cell is null.)

- [ ] **Step 4: Update `nmr/__init__.py` — `EVALUABLE_ROWS` re-export**

(The `nmr.families` symbols were already re-exported in Task 1 Step 3b.)

(a) Add `EVALUABLE_ROWS,` to the `from .dashboard import (...)` block (before `UNIFIED_SCHEMA,`).

(b) Add `"EVALUABLE_ROWS"` to `__all__` (insert alphabetically near the other `E`-prefixed entries; exact position does not matter functionally, keep tidy).

- [ ] **Step 5: Run the targeted tests**

Run: `./.venv/Scripts/python -m pytest -q tests/test_families.py tests/test_dashboard.py`
Expected: PASS.

- [ ] **Step 6: Lint**

Run: `./.venv/Scripts/python -m ruff check .`
Expected: no violations.

- [ ] **Step 7: Bump the AGENTS.md test-count claim**

Run: `./.venv/Scripts/python -m pytest --collect-only -q 2>&1 | tail -2`
Edit `AGENTS.md` line 33: replace the current count with the reported `N tests`.

- [ ] **Step 8: Run the full suite**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS — `test_package_api` now green (families re-exported), docs-hygiene count matches.

- [ ] **Step 9: Commit**

```bash
git add nmr/dashboard.py nmr/__init__.py tests/test_dashboard.py AGENTS.md
git commit -m "feat(dashboard): family columns, scan-once full rows, FULL gate status, EVALUABLE_ROWS"
```

---

### Task 3: `generate_dashboard.py` — FULL chip, Promoted Full Versions group, chart filter

**Files:**
- Modify: `generate_dashboard.py` (`_bar_input` ~line 61; `_table_rows` ~line 124; `_STATUS_BADGE` ~line 140; `_row_html` ~line 160; CSS in `_build_html` ~line 325)
- Test: `tests/test_dashboard.py` (append)
- Modify: `AGENTS.md` (test-count claim)

**Interfaces:**
- Consumes: `EVALUABLE_ROWS` from `nmr.dashboard`; `has_full_version` row field (Task 2).
- Produces: `_table_rows` returning the group order Champion → [header] → Full → Fleet → Benchmark; `_row_html` handling `_group_header` rows; `_bar_input` excluding full rows.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard.py`:

```python
def _lb_row(model_id: str, source: str, run_name: str, sharpe: float | None = None,
            has_full: bool = False) -> dict:
    return {"model_id": model_id, "source": source, "run_name": run_name,
            "corr_sharpe_ac": sharpe, "has_full_version": has_full}


def test_generate_dashboard_table_rows_grouping() -> None:
    rows = [
        _lb_row("ch" * 32, "trained", "champ-run", 0.9),
        _lb_row("a" * 64, "trained", "brb1-xgb-v6", 0.5, has_full=True),
        _lb_row("brb1-xgb-v6::full", "full", "brb1-xgb-v6"),
        _lb_row("bench_a", "benchmark", "ref", 0.78),
    ]
    frame = pl.DataFrame(rows, schema=dash.UNIFIED_SCHEMA, strict=False)
    ordered = generate_dashboard._table_rows(frame, champion="ch" * 32)
    kinds = [
        "header" if r.get("_group_header") else r["source"]
        for r in ordered
    ]
    assert kinds == ["trained", "header", "full", "trained", "benchmark"]


def test_generate_dashboard_row_html_full_chip() -> None:
    row = {
        **_lb_row("a" * 64, "trained", "brb1-xgb-v6", 0.5),
        "status": "RESEARCH",
        "cagr_1y": None, "corr_sharpe_ac_ci_low": None, "corr_sharpe_ac_ci_high": None,
        "max_drawdown": None, "gain_to_pain_ratio": None, "mmc_down": None,
        "deflated_sharpe": None, "gate_cagr_1y": None, "gate_corr_sharpe_ac": None,
        "gate_gain_to_pain_ratio": None, "gate_deflated_sharpe": None,
    }
    html_out = generate_dashboard._row_html(row)
    assert 'class="badge full">FULL</span>' in html_out


def test_generate_dashboard_bar_input_excludes_full_rows() -> None:
    rows = [
        _lb_row("a" * 64, "trained", "r1", 0.5),
        _lb_row("brb1-xgb-v6::full", "full", "brb1-xgb-v6"),
    ]
    frame = pl.DataFrame(rows, schema=dash.UNIFIED_SCHEMA, strict=False)
    out = generate_dashboard._bar_input(frame, champion=None)
    assert out.height == 1
    assert out.get_column("label").to_list() == ["r1 · " + "a" * 8]
```

- [ ] **Step 2: Run the failing tests**

Run: `./.venv/Scripts/python -m pytest -q tests/test_dashboard.py -k "table_rows_grouping or full_chip or bar_input_excludes" -v`
Expected: FAIL — no `_group_header` handling, no chip, full rows still in `_bar_input`.

- [ ] **Step 3: Implement `generate_dashboard.py` changes**

(a) Add `EVALUABLE_ROWS` to the existing `from nmr.dashboard import ...` block (find the import block near the top of the file — it currently imports `load_unified_leaderboard`, `reconcile_capital_metrics`, etc.; add `EVALUABLE_ROWS,`).

(b) `_bar_input` — filter before ranking:

```python
def _bar_input(leaderboard: pl.DataFrame, champion: str | None) -> pl.DataFrame:
    evaluable = leaderboard.filter(EVALUABLE_ROWS)
    top = evaluable.sort("corr_sharpe_ac", descending=True, nulls_last=True).head(10)
```

(c) `_STATUS_BADGE` — add the FULL entry:

```python
    "FULL": "full",
```

(d) `_table_rows` — insert the full group between champion and fleet:

```python
def _table_rows(leaderboard: pl.DataFrame, champion: str | None) -> list[dict]:
    rows = leaderboard.to_dicts()
    champion_rows = [r for r in rows if champion is not None and r["model_id"] == champion]
    full_rows = sorted(
        [r for r in rows if r["source"] == "full"],
        key=lambda r: (str(r["run_name"] or ""), str(r["model_id"])),
    )
    fleet_rows = sorted(
        [r for r in rows
         if r["source"] in ("trained", "trained_legacy") and r["model_id"] != champion],
        key=lambda r: (-(r["corr_sharpe_ac"] if r["corr_sharpe_ac"] is not None
                        else float("-inf")), r["model_id"]),
    )
    bench_rows = sorted(
        [r for r in rows if r["source"] == "benchmark"],
        key=lambda r: ((r["tier"] if r["tier"] is not None else 99), r["model_id"]),
    )
    if full_rows:
        return champion_rows + [{"_group_header": "Promoted Full Versions"}] + full_rows + fleet_rows + bench_rows
    return champion_rows + fleet_rows + bench_rows
```

(e) `_row_html` — handle the group header and render the chip:

```python
def _row_html(row: dict) -> str:
    if row.get("_group_header"):
        return (
            '<tr class="group-header"><td colspan="9">'
            f"{html.escape(row['_group_header'])}</td></tr>"
        )
    status = _status_badge(row.get("status", "RESEARCH"))
    sharpe = _fmt(row.get("corr_sharpe_ac"))
    ci = "—"
    if row.get("corr_sharpe_ac_ci_low") is not None and row.get("corr_sharpe_ac_ci_high") is not None:
        ci = f"[{_fmt(row['corr_sharpe_ac_ci_low'])}–{_fmt(row['corr_sharpe_ac_ci_high'])}]"
    model_label = html.escape(_bar_label(row))
    if row.get("has_full_version"):
        model_label += ' <span class="badge full">FULL</span>'
    return (
        "<tr>"
        f"<td>{status}</td>"
        f"<td>{model_label}</td>"
        f"{_td_gate(_fmt(row.get('cagr_1y'), pct=True), row.get('gate_cagr_1y'))}"
        f"{_td_gate(sharpe, row.get('gate_corr_sharpe_ac'))}"
        f"<td class=\"num\">{ci}</td>"
        f"<td class=\"num\">{_fmt(row.get('max_drawdown'), pct=True)}</td>"
        f"{_td_gate(_fmt(row.get('gain_to_pain_ratio')), row.get('gate_gain_to_pain_ratio'))}"
        f"<td class=\"num\">{_fmt(row.get('mmc_down'))}</td>"
        f"{_td_gate(_fmt(row.get('deflated_sharpe')), row.get('gate_deflated_sharpe'))}"
        "</tr>"
    )
```

(f) CSS — inside the `<style>` block in `_build_html`, immediately after the `.badge.benchmark` line, add (note the doubled braces — this is inside an f-string):

```
  .badge.full {{ background: rgba(210, 153, 34, 0.18); color: #d29922; border: 1px solid #9e6a03; }}
  .group-header td {{ background: #21262d; color: #e6edf3; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
```

- [ ] **Step 4: Run the targeted tests**

Run: `./.venv/Scripts/python -m pytest -q tests/test_dashboard.py -k "table_rows_grouping or full_chip or bar_input_excludes" -v`
Expected: PASS. Also run the pre-existing generate_dashboard tests to confirm no regression:

Run: `./.venv/Scripts/python -m pytest -q tests/test_dashboard.py -k "generate_dashboard or build_html" -v`
Expected: PASS (fixtures contain no full rows / no `has_full_version`, so output is unchanged).

- [ ] **Step 5: Lint**

Run: `./.venv/Scripts/python -m ruff check generate_dashboard.py tests/test_dashboard.py`
Expected: no violations.

- [ ] **Step 6: Bump the AGENTS.md test-count claim**

Run: `./.venv/Scripts/python -m pytest --collect-only -q 2>&1 | tail -2`
Edit `AGENTS.md` line 33: replace the current count with the reported `N tests`.

- [ ] **Step 7: Run the full suite**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add generate_dashboard.py tests/test_dashboard.py AGENTS.md
git commit -m "feat(dashboard): FULL chip + Promoted Full Versions group in executive table"
```

---

### Task 4: `dashboard_app.py` — Streamlit columns, source filter, pinned sort, chart filter

**Files:**
- Modify: `dashboard_app.py` (`_LEADERBOARD_SCHEMA` ~line 42; `load_registry_frame` ~line 97; `_shaped_leaderboard_pdf` ~line 265; `render_leaderboard` ~line 289; imports ~line 37)
- Test: `tests/test_dashboard.py` (append)
- Modify: `AGENTS.md` (test-count claim)

**Interfaces:**
- Consumes: `EVALUABLE_ROWS` from `nmr.dashboard`; Task 2's `load_unified_leaderboard(models_dir=...)`.
- Produces: `load_registry_frame(registry_dir, models_dir=None)` including `source == "full"` rows with pinned ordering; `_shaped_leaderboard_pdf` sorting full rows first.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard.py`:

```python
def test_dashboard_app_load_registry_frame_includes_full_sources(tmp_path: Path) -> None:
    import dashboard_app as app

    entry = _registry_entry("a" * 64)
    entry["manifest"]["config"]["run"]["name"] = "brb1-xgb-v6"
    _write_registry(tmp_path, [entry])
    models = _write_models_dir(
        tmp_path, {"brb1-xgb-v6": _full_manifest_dict("brb1-xgb-v6", "a" * 64)}
    )
    frame = app.load_registry_frame(tmp_path, models_dir=models)
    sources = frame.get_column("source").to_list()
    assert "full" in sources
    assert "trained" in sources
    full_row = frame.filter(pl.col("model_id") == "brb1-xgb-v6::full").row(0, named=True)
    assert full_row["backend"] == "xgboost"  # filled from manifest snapshot
    assert full_row["has_full_version"] is False


def test_dashboard_app_shaped_leaderboard_pins_full_rows_first() -> None:
    import dashboard_app as app

    rows = [
        {"model_id": "a" * 64, "source": "trained", "run_name": "r1",
         "corr_sharpe_ac": 0.5, "corr_sharpe_ac_ci_low": None, "corr_sharpe_ac_ci_high": None},
        {"model_id": "brb1-xgb-v6::full", "source": "full", "run_name": "brb1-xgb-v6",
         "corr_sharpe_ac": None, "corr_sharpe_ac_ci_low": None, "corr_sharpe_ac_ci_high": None},
    ]
    frame = pl.DataFrame(rows, schema=dash.UNIFIED_SCHEMA, strict=False)
    pdf = app._shaped_leaderboard_pdf(frame, champion=None)
    assert list(pdf["model_id"]) == ["brb1-xgb-v6::full", "a" * 64]
    assert "_is_full" not in pdf.columns


def test_dashboard_app_robustness_matrix_excludes_full_rows() -> None:
    import dashboard_app as app

    rows = [
        {"model_id": "a" * 64, "source": "trained", "run_name": "r1", "corr_sharpe_ac": 0.5,
         "has_bmc": True, "has_horizon": False, "has_perturb": True, "has_regime": False,
         "max_feature_exposure": 0.3, "std_corr": 0.2, "max_drawdown": 0.1},
        {"model_id": "brb1-xgb-v6::full", "source": "full", "run_name": "brb1-xgb-v6",
         "corr_sharpe_ac": None, "has_bmc": None, "has_horizon": None, "has_perturb": None,
         "has_regime": None, "max_feature_exposure": None, "std_corr": None,
         "max_drawdown": None},
    ]
    frame = pl.DataFrame(rows, schema=dash.UNIFIED_SCHEMA, strict=False)
    matrix = app.robustness_matrix(frame)
    assert matrix.get_column("model_id").to_list() == ["a" * 64]
```

- [ ] **Step 2: Run the failing tests**

Run: `./.venv/Scripts/python -m pytest -q tests/test_dashboard.py -k "dashboard_app" -v`
Expected: FAIL — `TypeError: load_registry_frame() got an unexpected keyword argument 'models_dir'` / `_shaped_leaderboard_pdf` does not pin full rows.

- [ ] **Step 3: Implement `dashboard_app.py` changes**

(a) Add `EVALUABLE_ROWS` to the `from nmr.dashboard import ...` block (near line 37; it currently imports `load_unified_leaderboard`, `load_benchmark_frame`, etc.).

(b) `_LEADERBOARD_SCHEMA` — add after the `"source": pl.String,` entry:

```python
        "family": pl.String,
        "training_scope": pl.String,
        "has_full_version": pl.Boolean,
```

(c) `load_registry_frame` — broaden the source filter, thread `models_dir`, fill null backend/preset:

```python
def load_registry_frame(
    registry_dir: Path, models_dir: Path | None = None
) -> pl.DataFrame:
    """Load registry runs into a leaderboard frame (engine delegation).

    Projects the engine's unified frame down to ``_LEADERBOARD_SCHEMA`` for
    the Streamlit views; parsing and None-discipline live in
    ``nmr.dashboard.load_unified_leaderboard``. Full-version rows
    (``source == "full"``) are included so the Source multiselect surfaces
    them. ``models_dir`` defaults to the engine's ``DEFAULT_MODELS_DIR``.
    """
    frame = load_unified_leaderboard(
        registry_dir, benchmark_path=False, models_dir=models_dir
    )
    selected = frame.filter(
        pl.col("source").is_in(["trained", "trained_legacy", "full"])
    )
    if selected.height == 0:
        return _EMPTY_LEADERBOARD
    projected = selected.select(_LEADERBOARD_SCHEMA.names())
    return projected.with_columns(
        pl.col("backend").fill_null("unknown"),
        pl.col("preset").fill_null("unknown"),
    )
```

(d) `_shaped_leaderboard_pdf` — pin full rows first, then Sharpe. Replace the first line of the body:

```python
    shaped = leaderboard.with_columns(pl.col("source").eq("full").alias("_is_full"))
    pdf = shaped.sort(
        ["_is_full", _BAR_METRIC], descending=[True, True], nulls_last=[False, True]
    ).to_pandas()
```

and before returning, drop the temporary column. Find the `return` statement of the function (it returns `pdf` with `champion`/`ci_plus`/`ci_minus`/`label` columns) and change it to:

```python
    return pdf.drop(columns=["_is_full"])
```

(e) `render_leaderboard` — chart on evaluable rows only; dataframe keeps the full frame (already pinned by `_shaped_leaderboard_pdf`):

```python
def render_leaderboard(leaderboard: pl.DataFrame, champion: str | None) -> None:
    """Bar chart of ``corr_sharpe_ac`` with CI error bars + sortable dataframe.

    The chart shows evaluable rows only (EVALUABLE_ROWS — full-version rows
    carry no validation metrics). The dataframe keeps all rows; full rows are
    pinned first by ``_shaped_leaderboard_pdf``.
    """
    if leaderboard.height == 0:
        st.info("No runs to display — train one with `train_first_model.py`.")
        return
    evaluable = leaderboard.filter(EVALUABLE_ROWS)
    if evaluable.height == 0:
        st.info("No evaluable runs to display (all rows are full versions).")
        return
    pdf = _shaped_leaderboard_pdf(evaluable, champion)
    fig = px.bar(
        pdf,
        x="label",
        y=_BAR_METRIC,
        color="source",
        error_y="ci_plus",
        error_y_minus="ci_minus",
        pattern_shape="champion",
        pattern_shape_map={True: "/", False: ""},
        hover_data=["run_name", "model_id", "backend", "preset", "corr", "max_drawdown"],
        title="CORR Sharpe (auto-correlated)",
    )
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig)
    table_pdf = _shaped_leaderboard_pdf(leaderboard, champion)
    st.dataframe(
        table_pdf.drop(columns=["champion", "ci_plus", "ci_minus", "label"]),
        column_config={
            "has_full_version": st.column_config.CheckboxColumn(
                label="Full",
                help="Has a promoted full (train+validation) version",
            ),
        },
    )
```

(f) `robustness_matrix` — exclude full rows (their robustness cells are all null; without this they leak NaN rows into the heatmap):

```python
def robustness_matrix(registry: pl.DataFrame) -> pl.DataFrame:
    """Project the robustness cells of evaluable rows (numeric casts for heatmap)."""
    columns = [
        "model_id",
        "has_bmc",
        "has_horizon",
        "has_perturb",
        "has_regime",
        "max_feature_exposure",
        "std_corr",
        "max_drawdown",
    ]
    casts = {
        "has_bmc": pl.Boolean,
        "has_horizon": pl.Boolean,
        "has_perturb": pl.Boolean,
        "has_regime": pl.Boolean,
        "max_feature_exposure": pl.Float64,
        "std_corr": pl.Float64,
        "max_drawdown": pl.Float64,
    }
    evaluable = (
        registry.filter(EVALUABLE_ROWS) if "source" in registry.columns else registry
    )
    frame = evaluable.select(columns)
    return frame.cast(casts)
```

(g) `render_robustness_matrix` — guard on the filtered matrix, not the raw input (an all-full registry would otherwise crash `px.imshow` on an empty frame):

```python
def render_robustness_matrix(registry: pl.DataFrame) -> None:
    """Plotly heatmap over ``robustness_matrix`` (booleans shown as 0/1)."""
    matrix = robustness_matrix(registry)
    if matrix.height == 0:
        st.info("No evaluable runs in the registry.")
        return
    numeric = matrix.with_columns(pl.col(flag).cast(pl.Int8) for flag in _ROBUSTNESS_CELLS)
    pdf = numeric.to_pandas().set_index("model_id").astype(float)
    fig = px.imshow(
        pdf,
        x=pdf.columns,
        y=pdf.index,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Robustness matrix",
    )
    st.plotly_chart(fig)
    st.caption(
        "Boolean cells (has_*) shown as 0/1; numeric cells "
        "(max_feature_exposure, std_corr, max_drawdown) shown raw."
    )
    st.dataframe(matrix)
```

(h) `render_run_detail` — full-version rows read `manifest.json` (their `run_dir` has no `run.json`). Add a pure helper next to `_read_run_payload` (around line 211; `json` is already imported):

```python
def _read_full_manifest(run_dir: Path) -> dict | None:
    """Read and parse a full-version manifest.json (None on missing/corrupt)."""
    path = Path(run_dir) / "manifest.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None
```

In `render_run_detail`, insert a full-row branch at the top of the row loop, immediately inside `with st.expander(label):` (before `payload = _read_run_payload(...)`):

```python
        with st.expander(label):
            if row["source"] == "full":
                manifest = _read_full_manifest(Path(row["run_dir"]))
                if manifest is None:
                    st.caption("Full-version row / missing manifest.json — leaderboard row only.")
                    st.dataframe(pl.DataFrame([row], strict=False))
                    continue
                st.subheader("Promoted Full Version Manifest")
                st.json(manifest)
                continue
            payload = _read_run_payload(Path(row["run_dir"]))
```

- [ ] **Step 4: Run the targeted tests**

Run: `./.venv/Scripts/python -m pytest -q tests/test_dashboard.py -k "dashboard_app" -v`
Expected: PASS.

- [ ] **Step 5: Lint**

Run: `./.venv/Scripts/python -m ruff check dashboard_app.py tests/test_dashboard.py`
Expected: no violations (note: `dashboard_app.py` is a top-level script, ruff still checks it).

- [ ] **Step 6: Bump the AGENTS.md test-count claim**

Run: `./.venv/Scripts/python -m pytest --collect-only -q 2>&1 | tail -2`
Edit `AGENTS.md` line 33: replace the current count with the reported `N tests`.

- [ ] **Step 7: Run the full suite**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add dashboard_app.py tests/test_dashboard.py AGENTS.md
git commit -m "feat(dashboard): Streamlit full-version columns, pinned sort, evaluable chart filter"
```

---

### Task 5: SSOT docs sync + final verification gate

**Files:**
- Modify: `ARCHITECTURE.md`
- Modify: `AGENTS.md` (toolkit row + artifact mention)
- Modify: `README.md` (annotated-tree line)

**Interfaces:**
- Consumes: the implemented surface from Tasks 1–4 (document it exactly as built; the manifest schema lives in the spec).

- [ ] **Step 1: `ARCHITECTURE.md` — add the families contract**

Find the module dependency graph section (search for `nmr/runner.py` or `dashboard.py` in the dependency list) and add `nmr/families.py` as a leaf module (read-only, consumed by `nmr/dashboard.py`; future promotion writer will be its sole writer).

Find where the artifact layout / registry is documented (search for `artifacts/registry`) and add `artifacts/models/<family>/full/manifest.json` to the layout with the one-line annotation "promoted full-version marker (family == run.name; read-only via nmr/families.py)".

Add a new section (place it near the dashboard module spec, search for `dashboard.py` or `UNIFIED_SCHEMA`) titled `## Model Families & Full Versions (nmr/families.py)` containing exactly this content (trim to the house style if the surrounding sections are terser):

```markdown
## Model Families & Full Versions (nmr/families.py)

A model **family** is the set of registry runs sharing `manifest.config.run.name`
(e.g. `brb1-xgb-v6`; duplicate reruns belong to one family). Promotion to a
**full version** (trained on train+validation, deployed) is marked by a valid
manifest at `artifacts/models/<family>/full/manifest.json` — manifest
presence + validity IS the marker. `nmr/families.py` is the read-only
discovery layer (no writes; the promotion writer is a future workstream).

Manifest schema (`manifest.json`):

| Field | Requirement |
|---|---|
| `family` | equals the directory name (lowercase `^[a-z0-9_-]+$`) |
| `training_scope` | `"full"` |
| `promoted_from_run_id` | non-empty registry run id (dangling lineage warns, never invalidates) |
| `promoted_at` | display metadata only — never in a canonical hash |
| `artifact_path` | non-empty relative path, no `/`, drive, or `..`; resolved against the manifest's own `full/` dir; file must exist (hollow promotions rejected) |
| `config` | snapshot of the promoted research config |

Public API: `full_manifest_path`, `load_full_version`, `scan_full_versions`,
`family_has_full_version`, `FullVersion`, constants `FAMILY_DIR_NAME` /
`FULL_DIR_NAME` / `FULL_MANIFEST_NAME` / `DEFAULT_MODELS_DIR`.

Leaderboard integration (`nmr/dashboard.py`): `UNIFIED_SCHEMA` carries
`family`, `training_scope` (`"research"` / `"full"`), `has_full_version`.
`load_unified_leaderboard` scans `artifacts/models/` ONCE, stamps trained rows
via set membership, and appends one `source="full"` row per valid manifest
(`model_id = "<family>::full"`, all metric cells null — in-sample metrics are
never shown as comparable OOF numbers). `evaluate_gate_status` stamps full
rows `FULL` (all gate receipts null). `EVALUABLE_ROWS = pl.col("source") != "full"`
is the single chart-inclusion predicate (source-based so benchmark rows with
null `training_scope` remain visible).
```

- [ ] **Step 2: `AGENTS.md` — toolkit row + artifact mention**

(a) In the Agent Toolkit table (the `| If you need to... | Look in... |` table), add a row:

```markdown
| Change model-family / full-version discovery | `nmr/families.py` — read-only scan of `artifacts/models/<family>/full/manifest.json` (spec: `ARCHITECTURE.md` Model Families section) |
```

(b) The `AGENTS.md` line 33 test-count claim is already current from Tasks 1–4 — verify it matches `./.venv/Scripts/python -m pytest --collect-only -q 2>&1 | tail -2`.

(c) Budget check: AGENTS.md must stay ≤ 32 KiB — the docs-hygiene T4 test enforces it; the toolkit row addition is small, but if T4 fails, trim prose elsewhere rather than growing the file.

- [ ] **Step 3: `README.md` — annotated-tree line**

Find the `artifacts/` block in the README's annotated project tree (search for `artifacts/registry`) and add one line:

```markdown
├── artifacts/models/      # promoted full-version markers (<family>/full/manifest.json)
```

(Adjust indentation to match the surrounding tree exactly.)

- [ ] **Step 4: Docs-hygiene + targeted verification**

Run: `./.venv/Scripts/python -m pytest -q tests/test_docs_hygiene.py tests/test_families.py tests/test_dashboard.py`
Expected: PASS.

- [ ] **Step 5: Full lint + functional gate**

Run: `./.venv/Scripts/python -m ruff check .`
Run: `./.venv/Scripts/python -m pytest -q`
Expected: both PASS. Report the final collected count (expect 772 + 42 = 814).

- [ ] **Step 6: Real-data smoke**

Run: `./.venv/Scripts/python generate_dashboard.py --help` (verify the CLI still loads), then:
Run: `./.venv/Scripts/python generate_dashboard.py`
Expected: `artifacts/dashboard.html` regenerates successfully. With no `artifacts/models/` yet, no FULL chips or full rows appear — expected. Leave the regenerated `artifacts/dashboard.html` in the working tree (it is machine-generated; do not commit it).

- [ ] **Step 7: Commit**

```bash
git add ARCHITECTURE.md AGENTS.md README.md
git commit -m "docs: model families & full-version marker — nmr/families.py contract, artifact schema, schema columns"
```

- [ ] **Step 8: Report**

Summarize: files changed per task, final test count, lint result, smoke result, and the note that the JS-controller chart in the dashboard HTML still awaits the human's one-time browser check.
