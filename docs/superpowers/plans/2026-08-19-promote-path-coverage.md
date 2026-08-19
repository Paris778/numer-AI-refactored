# Promote Path Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise `nmr/promote.py` (72%, 92/334 missed) and `nmr/models.py` (83%, 59/345 missed) to ≥ 90% statement coverage with pure test additions — no production-code changes — so the money path and the spawned full-history worker are exercised, not just visited.

**Architecture:** Test-only plan. Each task adds focused unit tests to `tests/test_promote.py` (promotion writer) or `tests/test_models.py` (orchestrator) using the existing synthetic fixture patterns (`_make_data`/`_config`/`_stored_config_dict` in test_promote.py; `_model_frame`/`_tiny_model_params` in test_models.py). Unreachable-by-construction paths (RAM-guard raise branches) are reached by monkeypatching module constants or by writing synthetic `ram_curve.json` / `full_version_ram_estimate.json` files into `tmp_path`; the spawned worker is exercised **in-process** by calling `_full_history_fit_worker(spec, queue.Queue())` directly, and its parent-side failure path via a real spawn against a missing data dir.

**Tech Stack:** Python 3.12, pytest, polars/numpy (fixtures), lightgbm `fast` preset (tiny fits), coverage via pytest-cov.

## Global Constraints

- **Coverage measurement — working form only.** Run `./.venv/Scripts/python -m pytest -q --no-header -p no:cacheprovider --cov=nmr --cov-branch --cov-report=term-missing` and grep the report rows. **NEVER pass dotted submodule specs** (`--cov=nmr.promote`): coverage 7.13.5 resolves dotted sources lazily via `find_spec('nmr.promote')` from inside a trace callback; that imports the parent package `nmr`, re-entering numpy's in-flight extension init → `ImportError: cannot load module more than once per process` (root-caused 2026-08-19; documented in `AGENTS.md` §8 and `CONTRIBUTING.md` by the CI-coverage-floor plan). Package-level `--cov=nmr` is immune. `--cov-branch` records branch coverage in the same report — the gate target stays statement-based (90%), branch numbers are recorded for visibility.
- **No production-code changes.** If a test cannot be written without touching `nmr/`, stop and report — a coverage gap must be closed with a test, not by deleting or weakening code. The one allowed exception is adding a `# pragma: no cover` only if the line is genuinely unreachable and documented in the commit message.
- **Fast tests only.** Tiny synthetic frames (≤ 10 eras × ≤ 8 rows), `preset="fast"`, `params={"n_estimators": 1, ...}`. Any test fitting a full-preset model is a failure. Spawn tests use `NMR_FULL_HISTORY_SPAWN_MIN_BYTES=1`.
- **No v5.3 data.** Every test in this plan builds its own `tmp_path` fixtures; nothing is `skipif`-gated on real data.
- **Style.** `ruff check .` must stay clean (E/F/I/UP, line-length 120). `tests/test_promote.py` is LF; `tests/test_models.py` is CRLF — the Edit tool preserves the file's line endings automatically. Match the surrounding comment style (docstring per test, one blank line between functions).
- **Verify per task.** Each task ends with `ruff check tests/<file>.py` and the targeted `pytest tests/<file>.py::<test_name>` run; the final task runs the full fast gate.

---

### Task 1: `_evaluate_gate` — strict-threshold and missing-evidence branches

**Files:**
- Modify: `tests/test_promote.py` (append at end of file)
- Test: `tests/test_promote.py`

**Interfaces:**
- Consumes: `nmr.promote._evaluate_gate(scorecard: dict, gate: Tier4GateConfig) -> tuple[bool, dict]` (promote.py:147); `nmr.benchmark.Tier4GateConfig` fields: `corr_min, corr_sharpe_ac_min, fnc_min, deflated_sharpe_min, gain_to_pain_min, cagr_min, turnover_max` (benchmark.py:121); existing helper `_passing_scorecard()` (test_promote.py:109).
- Produces: nothing used by other tasks.

- [ ] **Step 1: Add the two failing tests**

Append to `tests/test_promote.py`:

```python
def _gate() -> Tier4GateConfig:
    return Tier4GateConfig(
        corr_min=0.0286,
        corr_sharpe_ac_min=0.5,
        fnc_min=0.01,
        deflated_sharpe_min=0.3,
        gain_to_pain_min=1.0,
        cagr_min=0.05,
        turnover_max=0.05,
    )


def test_evaluate_gate_missing_evidence_fails() -> None:
    """A hard field with no measured value is a failure — never promoted on faith."""
    from nmr.promote import _evaluate_gate

    scorecard = {k: v for k, v in _passing_scorecard().items() if k != "corr"}
    passed, receipts = _evaluate_gate(scorecard, _gate())
    assert passed is False
    assert receipts["corr"]["measured"] is None
    assert receipts["corr"]["passed"] is False
    assert receipts["cagr_1y"]["passed"] is True  # other fields still evaluated


def test_evaluate_gate_strict_cagr_fails_at_threshold() -> None:
    """cagr_1y uses strict `>`: equality at the threshold fails (promote.py:165)."""
    from nmr.promote import _evaluate_gate

    scorecard = _passing_scorecard()
    scorecard["cagr_1y"] = 0.05  # exactly cagr_min — strict needs strictly greater
    passed, receipts = _evaluate_gate(scorecard, _gate())
    assert passed is False
    assert receipts["cagr_1y"]["passed"] is False
```

Add the import to the existing `from nmr...` block at the top of the file: `from nmr.benchmark import Tier4GateConfig`.

- [ ] **Step 2: Run to verify they pass**

Run: `./.venv/Scripts/python -m pytest -q tests/test_promote.py::test_evaluate_gate_missing_evidence_fails tests/test_promote.py::test_evaluate_gate_strict_cagr_fails_at_threshold -p no:cacheprovider`
Expected: `2 passed`

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_promote.py
git add tests/test_promote.py
git commit -m "test: cover _evaluate_gate strict and missing-evidence branches"
```

---

### Task 2: `_load_registry_run` — corrupt JSON and non-mapping payloads

**Files:**
- Modify: `tests/test_promote.py`
- Test: `tests/test_promote.py`

**Interfaces:**
- Consumes: `nmr.promote._load_registry_run(registry_dir: Path, run_id: str) -> dict` (promote.py:190); existing `_RID = "a" * 64` (test_promote.py:34).

- [ ] **Step 1: Add the tests**

```python
def test_load_registry_run_corrupt_json(tmp_path: Path) -> None:
    from nmr.promote import _load_registry_run

    registry = tmp_path / "registry"
    run_dir = registry / _RID
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt run.json"):
        _load_registry_run(registry, _RID)


def test_load_registry_run_non_mapping(tmp_path: Path) -> None:
    from nmr.promote import _load_registry_run

    registry = tmp_path / "registry"
    run_dir = registry / _RID
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a mapping"):
        _load_registry_run(registry, _RID)
```

- [ ] **Step 2: Run to verify they pass**

Run: `./.venv/Scripts/python -m pytest -q tests/test_promote.py::test_load_registry_run_corrupt_json tests/test_promote.py::test_load_registry_run_non_mapping -p no:cacheprovider`
Expected: `2 passed`

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_promote.py
git add tests/test_promote.py
git commit -m "test: cover _load_registry_run corrupt and non-mapping paths"
```

---

### Task 3: `_ram_guard` — fitted-curve path (under-guard pass)

**Files:**
- Modify: `tests/test_promote.py`
- Test: `tests/test_promote.py`

**Interfaces:**
- Consumes: `nmr.promote._ram_guard(config: ExperimentConfig, models_dir: Path) -> None` (promote.py:233). It reads `<models_dir.parent>/reports/ram_curve.json` with schema `{"fit": {"intercept_gib", "slope_gib_per_row"}, "fit_ws": {...}, "points": [{"parent_commit_gib", "parent_ws_gib"}]}`, scans `config.data.path("train.parquet")` + `validation.parquet` row counts via `_scan_len`, and calls `_raise_if_over_guard`. Existing helpers: `_make_data(tmp_path / "data")`, `_config(data)`, `_stored_config_dict` (test_promote.py:37-106).
- Produces: nothing used by other tasks.

- [ ] **Step 1: Add the test**

```python
def test_ram_guard_curve_path_passes_when_under_guard(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Zero-intercept/zero-slope curve → extrapolated commit ≈ 0 → guard passes,
    exercising the fitted-curve branch end to end (promote.py:259-297)."""
    from nmr.promote import _ram_guard

    data = _make_data(tmp_path / "data")
    config = _config(data)
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    (reports / "ram_curve.json").write_text(
        json.dumps(
            {
                "fit": {"intercept_gib": 0.0, "slope_gib_per_row": 0.0},
                "fit_ws": {"intercept_gib": 0.0, "slope_gib_per_row": 0.0},
                "points": [{"parent_commit_gib": 0.0, "parent_ws_gib": 0.0}],
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.INFO, logger="nmr.promote"):
        _ram_guard(config, tmp_path / "models")  # must not raise
    assert "extrapolated full-version combined commit" in caplog.text
```

Add `import logging` to the top import block of `tests/test_promote.py`.

- [ ] **Step 2: Run to verify it passes**

Run: `./.venv/Scripts/python -m pytest -q tests/test_promote.py::test_ram_guard_curve_path_passes_when_under_guard -p no:cacheprovider`
Expected: `1 passed`

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_promote.py
git add tests/test_promote.py
git commit -m "test: cover _ram_guard fitted-curve path"
```

---

### Task 4: `_raise_if_over_guard` — all three refusal branches

**Files:**
- Modify: `tests/test_promote.py`
- Test: `tests/test_promote.py`

**Interfaces:**
- Consumes: `nmr.promote._ram_guard` (same as Task 3); `nmr.models._machine_memory_limits() -> tuple[int | None, int | None]` (models.py:860); module constants `nmr.promote._RAM_GUARD_BYTES` (promote.py:73) and `_RAM_WS_FRACTION` (promote.py:78). A curve with `fit.slope_gib_per_row = 1e9` produces combined commit/WS ≫ any machine limit.
- Produces: nothing used by other tasks.

- [ ] **Step 1: Add the three tests**

```python
def _huge_curve(*, commit_slope: float, ws_slope: float, ws_intercept: float = 0.0) -> str:
    return json.dumps(
        {
            "fit": {"intercept_gib": 0.0, "slope_gib_per_row": commit_slope},
            "fit_ws": {"intercept_gib": ws_intercept, "slope_gib_per_row": ws_slope},
            "points": [{"parent_commit_gib": 0.0, "parent_ws_gib": 0.0}],
        }
    )


def test_ram_guard_over_ceiling_raises(tmp_path: Path) -> None:
    """combined commit > 45 GiB ceiling → RuntimeError naming the guard."""
    from nmr.promote import _ram_guard

    data = _make_data(tmp_path / "data")
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    (reports / "ram_curve.json").write_text(_huge_curve(commit_slope=1e9, ws_slope=0.0), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exceeds the 45 GiB guard"):
        _ram_guard(_config(data), tmp_path / "models")


def test_ram_guard_over_commit_limit_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """combined commit under the ceiling but over the machine commit limit."""
    from nmr.models import _machine_memory_limits
    from nmr.promote import _ram_guard

    _, commit_limit = _machine_memory_limits()
    if commit_limit is None:
        pytest.skip("platform reports no commit limit (Unix)")
    monkeypatch.setattr("nmr.promote._RAM_GUARD_BYTES", 2**70)
    data = _make_data(tmp_path / "data")
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    (reports / "ram_curve.json").write_text(_huge_curve(commit_slope=1e9, ws_slope=0.0), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exceeds the machine commit limit"):
        _ram_guard(_config(data), tmp_path / "models")


def test_ram_guard_over_working_set_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """commit small but combined working set > 85% of physical RAM → thrash refusal."""
    from nmr.models import _machine_memory_limits
    from nmr.promote import _ram_guard

    physical, _ = _machine_memory_limits()
    if physical is None:
        pytest.skip("platform reports no physical RAM")
    monkeypatch.setattr("nmr.promote._RAM_GUARD_BYTES", 2**70)
    data = _make_data(tmp_path / "data")
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    (reports / "ram_curve.json").write_text(_huge_curve(commit_slope=0.0, ws_slope=1e9), encoding="utf-8")
    with pytest.raises(RuntimeError, match="would thrash"):
        _ram_guard(_config(data), tmp_path / "models")
```

- [ ] **Step 2: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_promote.py -k ram_guard -p no:cacheprovider`
Expected: `4 passed` (3 new + Task 3's test). Any `1 skipped` on Unix is expected; on this Windows box all three run.

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_promote.py
git add tests/test_promote.py
git commit -m "test: cover _raise_if_over_guard refusal branches"
```

---

### Task 5: `_ram_guard` — estimate path, fallbacks, and unreadable files

**Files:**
- Modify: `tests/test_promote.py`
- Test: `tests/test_promote.py`

**Interfaces:**
- Consumes: `nmr.promote._ram_guard`; estimate schema at `<models_dir.parent>/reports/full_version_ram_estimate.json`: `{"peak_commit_bytes", "peak_bytes", "parent_peak_commit_bytes", "parent_peak_bytes", "train_validation_rows"}` (promote.py:298-331). Existing helpers as Task 3.

- [ ] **Step 1: Add the four tests**

```python
def _write_estimate(reports: Path, payload: dict) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "full_version_ram_estimate.json").write_text(json.dumps(payload), encoding="utf-8")


def test_ram_guard_estimate_path_passes_when_under_guard(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No curve on disk → single-point estimate, through-origin extrapolation."""
    from nmr.promote import _ram_guard

    data = _make_data(tmp_path / "data")
    _write_estimate(
        tmp_path / "reports",
        {
            "peak_commit_bytes": 1,
            "peak_bytes": 1,
            "parent_peak_commit_bytes": 0,
            "parent_peak_bytes": 0,
            "train_validation_rows": 1,
        },
    )
    with caplog.at_level(logging.WARNING, logger="nmr.promote"):
        _ram_guard(_config(data), tmp_path / "models")  # must not raise
    assert "single-point estimate extrapolation" in caplog.text


def test_ram_guard_estimate_missing_dual_metric_skips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from nmr.promote import _ram_guard

    data = _make_data(tmp_path / "data")
    _write_estimate(tmp_path / "reports", {"peak_bytes": 1, "parent_peak_bytes": 0, "train_validation_rows": 1})
    with caplog.at_level(logging.WARNING, logger="nmr.promote"):
        _ram_guard(_config(data), tmp_path / "models")  # must not raise
    assert "lacks dual-metric data" in caplog.text


def test_ram_guard_corrupt_curve_falls_back_to_estimate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from nmr.promote import _ram_guard

    data = _make_data(tmp_path / "data")
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    (reports / "ram_curve.json").write_text("{corrupt", encoding="utf-8")
    _write_estimate(
        reports,
        {
            "peak_commit_bytes": 1,
            "peak_bytes": 1,
            "parent_peak_commit_bytes": 0,
            "parent_peak_bytes": 0,
            "train_validation_rows": 1,
        },
    )
    with caplog.at_level(logging.WARNING, logger="nmr.promote"):
        _ram_guard(_config(data), tmp_path / "models")  # must not raise
    assert "unreadable RAM curve" in caplog.text


def test_ram_guard_corrupt_estimate_skips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from nmr.promote import _ram_guard

    data = _make_data(tmp_path / "data")
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    (reports / "full_version_ram_estimate.json").write_text("{corrupt", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="nmr.promote"):
        _ram_guard(_config(data), tmp_path / "models")  # must not raise
    assert "unreadable RAM estimate" in caplog.text
```

- [ ] **Step 2: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_promote.py -k "ram_guard_estimate or ram_guard_corrupt" -p no:cacheprovider`
Expected: `4 passed`

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_promote.py
git add tests/test_promote.py
git commit -m "test: cover _ram_guard estimate path and fallback warnings"
```

---

### Task 6: `resolve_champion_run_id` — the atomic pointer reader

**Files:**
- Modify: `tests/test_promote.py`
- Test: `tests/test_promote.py`

**Interfaces:**
- Consumes: `nmr.promote.resolve_champion_run_id(registry_dir: Path) -> str` (promote.py:427).

- [ ] **Step 1: Add the test**

```python
def test_resolve_champion_run_id(tmp_path: Path) -> None:
    from nmr.promote import resolve_champion_run_id

    registry = tmp_path / "registry"
    registry.mkdir()
    with pytest.raises(FileNotFoundError, match="no champion"):
        resolve_champion_run_id(registry)
    champion = registry / "champion.json"
    champion.write_text("{corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt champion"):
        resolve_champion_run_id(registry)
    champion.write_text(json.dumps({"run_id": "not-hex"}), encoding="utf-8")
    with pytest.raises(ValueError, match="no valid run_id"):
        resolve_champion_run_id(registry)
    champion.write_text(json.dumps({"run_id": _RID}), encoding="utf-8")
    assert resolve_champion_run_id(registry) == _RID
```

- [ ] **Step 2: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_promote.py::test_resolve_champion_run_id -p no:cacheprovider`
Expected: `1 passed`

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_promote.py
git add tests/test_promote.py
git commit -m "test: cover resolve_champion_run_id pointer reader"
```

---

### Task 7: `promote_full_version` — remaining refusal branches

**Files:**
- Modify: `tests/test_promote.py`
- Test: `tests/test_promote.py`

**Interfaces:**
- Consumes: `nmr.promote.promote_full_version(run_id, family, *, models_dir, registry_dir, ...)` (promote.py:442); existing helpers `_make_data`, `_stored_config_dict`, `_write_registry` (test_promote.py:117), `_passing_scorecard`.

- [ ] **Step 1: Add the five tests**

```python
def test_promote_manifest_without_config_refused(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    run_dir = registry / _RID
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": _RID, "manifest": {}, "scorecard": _passing_scorecard()}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no config dict"):
        promote_full_version(
            _RID, "brb1-lgbm-v6", models_dir=tmp_path / "models", registry_dir=registry
        )


def test_promote_gate_missing_from_yaml_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    data = _make_data(tmp_path / "data")
    registry = tmp_path / "registry"
    _write_registry(registry, stored_config=_stored_config_dict(data))
    monkeypatch.setattr(
        "nmr.promote.load_benchmark_file", lambda path: types.SimpleNamespace(gate=None)
    )
    with pytest.raises(ValueError, match="no gate"):
        promote_full_version(
            _RID, "brb1-lgbm-v6", models_dir=tmp_path / "models", registry_dir=registry
        )


def test_promote_corrupt_current_pointer_requires_force(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    registry = tmp_path / "registry"
    _write_registry(registry, stored_config=_stored_config_dict(data))
    full_dir = tmp_path / "models" / "brb1-lgbm-v6" / "full"
    full_dir.mkdir(parents=True)
    (full_dir / CURRENT_POINTER_NAME).write_text("{corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="repointing requires force"):
        promote_full_version(
            _RID, "brb1-lgbm-v6", models_dir=tmp_path / "models", registry_dir=registry
        )


def test_promote_missing_feature_cols_refused(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    registry = tmp_path / "registry"
    _write_registry(registry, stored_config=_stored_config_dict(data), feature_cols=[])
    with pytest.raises(ValueError, match="no feature_cols"):
        promote_full_version(
            _RID, "brb1-lgbm-v6", models_dir=tmp_path / "models", registry_dir=registry
        )


def test_promote_weight_count_mismatch_refused(tmp_path: Path) -> None:
    data = _make_data(tmp_path / "data")
    registry = tmp_path / "registry"
    _write_registry(
        registry, stored_config=_stored_config_dict(data), weights=[1.0, 1.0]
    )
    with pytest.raises(ValueError, match="do not match targets"):
        promote_full_version(
            _RID, "brb1-lgbm-v6", models_dir=tmp_path / "models", registry_dir=registry
        )
```

- [ ] **Step 2: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_promote.py -k "manifest_without_config or gate_missing or corrupt_current_pointer or missing_feature_cols or weight_count_mismatch" -p no:cacheprovider`
Expected: `5 passed`

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_promote.py
git add tests/test_promote.py
git commit -m "test: cover promote_full_version refusal branches"
```

---

### Task 8: `_build_truncated_data` — missing asset and insufficient eras

**Files:**
- Modify: `tests/test_promote.py`
- Test: `tests/test_promote.py`

**Interfaces:**
- Consumes: `nmr.promote._build_truncated_data(stored_config: dict, rehearsal_root: Path, *, train_eras: int, validation_eras: int) -> tuple[Path, int]` (promote.py:625). The stored config's `data.data_dir`/`data.version` pick the source; `_make_data` produces 8 train eras / 8 validation eras in `vtest`.

- [ ] **Step 1: Add the two tests**

```python
def test_build_truncated_data_missing_asset(tmp_path: Path) -> None:
    from nmr.promote import _build_truncated_data

    stored = _stored_config_dict(tmp_path / "data")
    stored["data"]["data_dir"] = str(tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="data assets missing"):
        _build_truncated_data(
            stored, tmp_path / "rehearsal", train_eras=1, validation_eras=1
        )


def test_build_truncated_data_insufficient_eras(tmp_path: Path) -> None:
    from nmr.promote import _build_truncated_data

    data = _make_data(tmp_path / "data")  # 8 train eras
    stored = _stored_config_dict(data)
    with pytest.raises(ValueError, match="rehearsal needs 9/1"):
        _build_truncated_data(
            stored, tmp_path / "rehearsal", train_eras=9, validation_eras=1
        )
```

- [ ] **Step 2: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_promote.py -k build_truncated_data -p no:cacheprovider`
Expected: `2 passed`

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_promote.py
git add tests/test_promote.py
git commit -m "test: cover _build_truncated_data error branches"
```

---

### Task 9: `measure_full_history_peak` — the RAM-curve measurement entry point

**Files:**
- Modify: `tests/test_promote.py`
- Test: `tests/test_promote.py`

**Interfaces:**
- Consumes: `nmr.promote.measure_full_history_peak(stored_config, feature_cols, target_cols, weights, *, data_dir, seed=42) -> tuple[int | None, int | None, int | None, int | None, int]` (promote.py:702). Forces the spawn path via `NMR_FULL_HISTORY_SPAWN_MIN_BYTES=1`.

- [ ] **Step 1: Add the test**

```python
def test_measure_full_history_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs the real promotion training path (spawn forced, train+validation)
    and returns measured peaks — the curve measurement, exercised at toy scale."""
    from nmr.promote import measure_full_history_peak

    monkeypatch.setenv("NMR_FULL_HISTORY_SPAWN_MIN_BYTES", "1")
    data = _make_data(tmp_path / "data")
    stored = _stored_config_dict(data)
    child_ws, child_commit, parent_ws, parent_commit, rows = measure_full_history_peak(
        stored,
        feature_cols=["f1", "f2"],
        target_cols=["target"],
        weights=[1.0],
        data_dir=data,
        seed=42,
    )
    assert rows > 0
    assert child_ws is None or child_ws > 0
    assert child_commit is None or child_commit > 0
    assert parent_ws is None or parent_ws > 0
    assert parent_commit is None or parent_commit > 0
```

- [ ] **Step 2: Run to verify it passes**

Run: `./.venv/Scripts/python -m pytest -q tests/test_promote.py::test_measure_full_history_peak -p no:cacheprovider`
Expected: `1 passed` (spawn fit on 128 rows, a few seconds)

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_promote.py
git add tests/test_promote.py
git commit -m "test: cover measure_full_history_peak entry point"
```

---

### Task 10: `rehearse_promotion` — env restore, stale pointer, and refusal/acceptance-failure branches

**Files:**
- Modify: `tests/test_promote.py`
- Test: `tests/test_promote.py`

**Interfaces:**
- Consumes: `nmr.promote.rehearse_promotion(run_id, family, *, models_dir, registry_dir, rehearsal_data_root, train_eras, validation_eras)` (promote.py:749); `nmr.promote.PromotionResult` (promote.py:95); `CURRENT_POINTER_NAME` already imported in the test file (test_promote.py:30). The acceptance-failure test monkeypatches `nmr.submission.accept_promoted_artifact` (imported inside `rehearse_promotion`, promote.py:883).
- Produces: nothing used by other tasks.

- [ ] **Step 1: Add the four tests**

```python
def test_rehearse_restores_env_and_removes_stale_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env override is restored to its prior value and a stale current.json
    left by an earlier rehearsal is removed (a rehearsal is never the full version)."""
    monkeypatch.setenv("NMR_FULL_HISTORY_SPAWN_MIN_BYTES", "7777")
    data = _make_data(tmp_path / "data")
    registry = tmp_path / "registry"
    _write_registry(registry, stored_config=_stored_config_dict(data))
    full_dir = tmp_path / "models" / "brb1-lgbm-v6" / "full"
    full_dir.mkdir(parents=True)
    (full_dir / CURRENT_POINTER_NAME).write_text(
        json.dumps({"run_id": "c" * 64}), encoding="utf-8"
    )
    result = rehearse_promotion(
        _RID,
        "brb1-lgbm-v6",
        models_dir=tmp_path / "models",
        registry_dir=registry,
        rehearsal_data_root=tmp_path / "rehearsal",
        train_eras=6,
        validation_eras=6,
    )
    assert result.acceptance_passed is True
    assert os.environ["NMR_FULL_HISTORY_SPAWN_MIN_BYTES"] == "7777"
    assert not (full_dir / CURRENT_POINTER_NAME).exists()


def test_rehearse_manifest_without_config_refused(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    run_dir = registry / _RID
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": _RID, "manifest": {}, "scorecard": _passing_scorecard()}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no config dict"):
        rehearse_promotion(
            _RID, "brb1-lgbm-v6", models_dir=tmp_path / "models", registry_dir=registry
        )


def _fake_promotion_result(tmp_path: Path) -> PromotionResult:
    artifact = tmp_path / "fake_predict.pkl"
    artifact.write_bytes(b"not-a-real-model")
    return PromotionResult(
        artifact_path=artifact,
        manifest_path=tmp_path / "manifest.json",
        run_id=_RID,
        family="brb1-lgbm-v6",
        tier4_gate_passed=False,
        override_used=True,
    )


def test_rehearse_missing_feature_cols_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The post-promotion feature_cols check (promote.py:870-872) — the promotion
    itself is stubbed out so the test never fits a model."""
    from nmr.promote import rehearse_promotion

    data = _make_data(tmp_path / "data")
    registry = tmp_path / "registry"
    _write_registry(
        registry, stored_config=_stored_config_dict(data), feature_cols=[]
    )
    fake = _fake_promotion_result(tmp_path)
    monkeypatch.setattr("nmr.promote.promote_full_version", lambda *a, **k: fake)
    with pytest.raises(ValueError, match="no feature_cols"):
        rehearse_promotion(
            _RID,
            "brb1-lgbm-v6",
            models_dir=tmp_path / "models",
            registry_dir=registry,
            rehearsal_data_root=tmp_path / "rehearsal",
            train_eras=6,
            validation_eras=6,
        )


def test_rehearse_acceptance_failure_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The Phase-D acceptance criterion is NOT overridable: a failed raw-contract
    validation is logged at ERROR and re-raised (promote.py:883-894)."""
    data = _make_data(tmp_path / "data")
    registry = tmp_path / "registry"
    _write_registry(registry, stored_config=_stored_config_dict(data))
    fake = _fake_promotion_result(tmp_path)
    monkeypatch.setattr("nmr.promote.promote_full_version", lambda *a, **k: fake)

    def _boom(*args, **kwargs) -> None:
        raise ValueError("boom")

    monkeypatch.setattr("nmr.submission.accept_promoted_artifact", _boom)
    with (
        caplog.at_level(logging.ERROR, logger="nmr.promote"),
        pytest.raises(ValueError, match="boom"),
    ):
        rehearse_promotion(
            _RID,
            "brb1-lgbm-v6",
            models_dir=tmp_path / "models",
            registry_dir=registry,
            rehearsal_data_root=tmp_path / "rehearsal",
            train_eras=6,
            validation_eras=6,
        )
    assert "acceptance FAILED" in caplog.text
```

Add to the top import block: `import os` and `from nmr.promote import PromotionResult, promote_full_version, rehearse_promotion` (extend the existing `from nmr.promote import ...` line at test_promote.py:31).

- [ ] **Step 2: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_promote.py -k "rehearse" -p no:cacheprovider`
Expected: `5 passed` (4 new + existing `test_rehearse_promotion_end_to_end`)

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_promote.py
git add tests/test_promote.py
git commit -m "test: cover rehearse_promotion env/pointer/refusal/acceptance branches"
```

---

### Task 11: models.py — anchor/CV degenerate paths (lines 218, 250, 283)

**Files:**
- Modify: `tests/test_models.py` (append; file is CRLF — Edit preserves it)
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `ModelOrchestrator` (models.py:195), `Fold(index, train_eras, val_eras)` (splitter.py:30), existing helpers `_model_frame`, `_tiny_model_params`, `_walk_forward_splitter` (test_models.py:20-58).

- [ ] **Step 1: Add the three tests**

```python
def test_train_anchor_fold_requires_exactly_one_fold() -> None:
    orch = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=7,
    )
    with pytest.raises(ValueError, match="exactly one fold"):
        orch.train_anchor_fold(
            _model_frame(n_eras=10, rows_per_era=6),
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            splitter=_walk_forward_splitter(),  # 3 folds, not 1
            era_col="era",
        )


def test_cross_validation_rejects_overlapping_validation_eras() -> None:
    from nmr.splitter import Fold

    class _OverlappingSplitter:
        purge_eras = 1

        def split(self, eras):
            return [Fold(0, ("1", "2"), ("4", "5")), Fold(1, ("3",), ("4", "5"))]

    orch = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=7,
    )
    with pytest.raises(ValueError, match="disjoint across folds"):
        orch.train_cross_validation(
            _model_frame(n_eras=5, rows_per_era=6),
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            splitter=_OverlappingSplitter(),
            era_col="era",
        )


def test_cross_validation_no_folds_raises() -> None:
    class _EmptySplitter:
        purge_eras = 1

        def split(self, eras):
            return []

    orch = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=7,
    )
    with pytest.raises(ValueError, match="No folds produced"):
        orch.train_cross_validation(
            _model_frame(n_eras=4, rows_per_era=6),
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            splitter=_EmptySplitter(),
            era_col="era",
        )
```

- [ ] **Step 2: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_models.py -k "exactly_one_fold or overlapping_validation_eras or no_folds" -p no:cacheprovider`
Expected: `3 passed`

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_models.py
git add tests/test_models.py
git commit -m "test: cover orchestrator anchor/CV degenerate-fold paths"
```

---

### Task 12: models.py — empty-frame paths in full-history and fold fitting (lines 342, 345, 379, 391)

**Files:**
- Modify: `tests/test_models.py` (append)
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `ModelOrchestrator.train_full_history` (models.py:291), `_fit_predict_fold` (models.py:365), `Fold` (splitter.py:30), existing helpers as Task 11.

- [ ] **Step 1: Add the four tests**

```python
def test_train_full_history_all_null_targets_raises() -> None:
    df = _model_frame(n_eras=4, rows_per_era=4).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("target")
    )
    orch = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=7,
    )
    with pytest.raises(ValueError, match="No usable training rows"):
        orch.train_full_history(
            df, feature_cols=["f1", "f2", "f3"], target_col="target", in_process=True
        )


def test_train_full_history_spawn_without_data_config_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_full_history's own guard (models.py:344-349): the spawn path demands
    the DataConfig before the subprocess machinery is even reached."""
    monkeypatch.setenv("NMR_FULL_HISTORY_SPAWN_MIN_BYTES", "1")
    orch = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=7,
    )
    with pytest.raises(ValueError, match="DataConfig"):
        orch.train_full_history(
            _model_frame(n_eras=4, rows_per_era=4),
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
        )


def test_fit_predict_fold_empty_slice_raises() -> None:
    from nmr.splitter import Fold

    orch = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=7,
    )
    fold = Fold(0, ("99",), ("98",))  # eras absent from the frame
    with pytest.raises(ValueError, match="Degenerate training slice"):
        orch._fit_predict_fold(
            _model_frame(n_eras=4, rows_per_era=4),
            fold=fold,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            era_col="era",
            purge_eras=1,
        )


def test_fit_predict_fold_all_null_targets_raises() -> None:
    from nmr.splitter import Fold

    df = _model_frame(n_eras=4, rows_per_era=4).with_columns(
        pl.when(pl.col("era").is_in(["1", "2"]))
        .then(None)
        .otherwise(pl.col("target"))
        .alias("target")
    )
    orch = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=7,
    )
    fold = Fold(0, ("1", "2"), ("3",))
    with pytest.raises(ValueError, match="No usable training rows"):
        orch._fit_predict_fold(
            df,
            fold=fold,
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            era_col="era",
            purge_eras=0,  # gap 3-2=1 > 0, so the leakage assertion passes
        )
```

- [ ] **Step 2: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_models.py -k "all_null_targets or spawn_without_data_config or empty_slice" -p no:cacheprovider`
Expected: `4 passed`

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_models.py
git add tests/test_models.py
git commit -m "test: cover empty-frame paths in full-history and fold fitting"
```

---

### Task 13: models.py — GPU-candidate dedupe and xgboost param translation (lines 548, 602-609)

**Files:**
- Modify: `tests/test_models.py` (append)
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `ModelOrchestrator._device_candidate_params(use_gpu, n_features)` (models.py:532), `_resolved_params(use_gpu, n_features)` (models.py:551), `ModelConfig`.

- [ ] **Step 1: Add the two tests**

```python
def test_device_candidate_params_dedupes_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gpu_params == cpu_params → a single candidate (models.py:547-548)."""
    orch = ModelOrchestrator(
        ModelConfig(
            backend="lightgbm", preset="fast", params=_tiny_model_params(), device="auto"
        ),
        seed=7,
    )
    same = {"dummy": 1}
    monkeypatch.setattr(orch, "_resolved_params", lambda **kwargs: dict(same))
    candidates = orch._device_candidate_params(use_gpu=True, n_features=3)
    assert candidates == [same]


def test_xgboost_param_translation_branches() -> None:
    """num_leaves → max_leaves+lossguide; bare max_leaves → lossguide;
    min_data_in_leaf → min_child_weight (models.py:597-610)."""
    orch_nl = ModelOrchestrator(
        ModelConfig(backend="xgboost", preset="fast", params={"num_leaves": 15}),
        seed=7,
    )
    p = orch_nl._resolved_params(use_gpu=False, n_features=10)
    assert p["grow_policy"] == "lossguide"
    assert p["max_leaves"] == 15

    orch_ml = ModelOrchestrator(
        ModelConfig(backend="xgboost", preset="fast", params={"max_leaves": 15}),
        seed=7,
    )
    p = orch_ml._resolved_params(use_gpu=False, n_features=10)
    assert p["grow_policy"] == "lossguide"
    assert p["max_leaves"] == 15

    orch_md = ModelOrchestrator(
        ModelConfig(
            backend="xgboost", preset="fast", params={"min_data_in_leaf": 20}
        ),
        seed=7,
    )
    p = orch_md._resolved_params(use_gpu=False, n_features=10)
    assert p["min_child_weight"] == 20.0
```

- [ ] **Step 2: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_models.py -k "dedupes_identical or param_translation" -p no:cacheprovider`
Expected: `2 passed`

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_models.py
git add tests/test_models.py
git commit -m "test: cover device-candidate dedupe and xgboost param translation"
```

---

### Task 14: models.py — leakage assertions and subprocess failure (lines 630, 632, 637, 723)

**Files:**
- Modify: `tests/test_models.py` (append)
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `ModelOrchestrator._assert_fold_is_leakage_safe(fold, purge_eras)` (models.py:626), `_fit_full_history_subprocess(train_df, *, feature_cols, target_col, era_col, data)` (models.py:662), `DataConfig(version=..., data_dir=...)`, `Fold`.

- [ ] **Step 1: Add the two tests**

```python
def test_assert_fold_is_leakage_safe_raises() -> None:
    from nmr.splitter import Fold

    orch = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=7,
    )
    with pytest.raises(ValueError, match="reuses eras"):
        orch._assert_fold_is_leakage_safe(Fold(0, ("1", "2", "3"), ("3", "4")), purge_eras=1)
    with pytest.raises(ValueError, match="degenerate"):
        orch._assert_fold_is_leakage_safe(Fold(0, (), ("3",)), purge_eras=1)
    with pytest.raises(ValueError, match="strictly time-ordered"):
        orch._assert_fold_is_leakage_safe(Fold(0, ("3",), ("2",)), purge_eras=1)


def test_full_history_subprocess_child_error_raises(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child that reports ("error", ...) — here because its data dir does not
    exist — must surface as RuntimeError in the parent (models.py:722-725)."""
    from nmr.config import DataConfig

    monkeypatch.setenv("NMR_FULL_HISTORY_SPAWN_MIN_BYTES", "1")
    orch = ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=7,
    )
    data_cfg = DataConfig(version="vtest", data_dir=tmp_path / "missing")
    with pytest.raises(RuntimeError, match="subprocess fit failed"):
        orch._fit_full_history_subprocess(
            _model_frame(n_eras=4, rows_per_era=4),
            feature_cols=["f1", "f2", "f3"],
            target_col="target",
            era_col="era",
            data=data_cfg,
        )
```

- [ ] **Step 2: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_models.py -k "leakage_safe or child_error" -p no:cacheprovider`
Expected: `2 passed`

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_models.py
git add tests/test_models.py
git commit -m "test: cover leakage assertions and subprocess-failure path"
```

---

### Task 15: models.py — memory helpers and the spawn worker in-process (lines 844, 856-857, 869-909, 931-964)

**Files:**
- Modify: `tests/test_models.py` (append)
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `nmr.models._peak_memory_counters()` (models.py:794), `_peak_rss_bytes()` (models.py:854), `_machine_memory_limits()` (models.py:860), `_full_history_fit_worker(spec: dict, out_q)` (models.py:921). Spec schema (models.py:688-709): `{"data": {version, feature_set, feature_subset, targets, data_dir, supplemental_feature_sets}, "feature_cols", "target_col", "era_col", "backend", "preset", "params", "seed", "include_validation"}`. `queue.Queue` works in-process (the worker only calls `out_q.put`).
- Note: this task adds *body* coverage for the worker (in a spawned child, coverage never counts it). The spawn **transport** itself is already tested with real `multiprocessing` spawns: `tests/test_models.py:695` (worker via `ctx.Process`), `tests/test_models.py:779` (`_fit_full_history_subprocess` happy path), `tests/test_promote.py:342` (spawn-forced promote) — no new transport test is needed.
- Produces: nothing used by other tasks.

- [ ] **Step 1: Add the four tests**

```python
def test_machine_memory_limits_shape() -> None:
    from nmr.models import _machine_memory_limits

    physical, commit_limit = _machine_memory_limits()
    assert physical is None or physical > 0
    assert commit_limit is None or commit_limit > 0


def test_peak_memory_counters_and_wrapper() -> None:
    from nmr.models import _peak_memory_counters, _peak_rss_bytes

    working_set, commit = _peak_memory_counters()
    assert working_set is None or working_set > 0
    assert commit is None or commit > 0
    assert _peak_rss_bytes() == working_set


@pytest.mark.skipif(sys.platform != "win32", reason="Windows K32GetProcessMemoryInfo branch")
def test_peak_memory_counters_zero_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """K32GetProcessMemoryInfo returning 0 → (None, None) (models.py:844)."""
    import ctypes

    from nmr.models import _peak_memory_counters

    kernel32 = ctypes.windll.kernel32
    fake = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)(
        lambda *args: 0
    )
    monkeypatch.setattr(kernel32, "K32GetProcessMemoryInfo", fake)
    assert _peak_memory_counters() == (None, None)


def test_full_history_fit_worker_in_process_ok_and_error(tmp_path) -> None:
    """The spawned worker is a plain function: call it in-process so its code is
    actually counted by coverage (in a spawned child it never would be)."""
    import json
    import queue

    import cloudpickle

    from nmr.models import _full_history_fit_worker

    data_root = tmp_path / "data" / "vtest"
    data_root.mkdir(parents=True)
    (data_root / "features.json").write_text(
        json.dumps({"feature_sets": {"small": ["f1", "f2", "f3"]}, "targets": ["target"]}),
        encoding="utf-8",
    )
    df = _model_frame(n_eras=10, rows_per_era=6)
    df.write_parquet(data_root / "train.parquet")

    spec = {
        "data": {
            "version": "vtest",
            "feature_set": "small",
            "feature_subset": None,
            "targets": ("target",),
            "data_dir": str(data_root.parent),
            "supplemental_feature_sets": None,
        },
        "feature_cols": ["f1", "f2", "f3"],
        "target_col": "target",
        "era_col": "era",
        "backend": "lightgbm",
        "preset": "fast",
        "params": _tiny_model_params(),
        "seed": 7,
        "include_validation": False,
    }
    q = queue.Queue()
    _full_history_fit_worker(spec, q)
    status, payload = q.get_nowait()
    assert status == "ok", payload
    model_bytes, working_set, commit = payload
    assert working_set is None or working_set > 0
    assert commit is None or commit > 0
    model = cloudpickle.loads(model_bytes)
    preds = np.asarray(model.predict(df.select(["f1", "f2", "f3"]).to_numpy()), dtype=float)
    assert preds.shape == (df.height,)
    assert np.isfinite(preds).all()

    broken = {**spec, "data": {**spec["data"], "data_dir": str(tmp_path / "nope")}}
    q2 = queue.Queue()
    _full_history_fit_worker(broken, q2)
    status2, payload2 = q2.get_nowait()
    assert status2 == "error"
    assert "FileNotFoundError" in payload2
```

Add `import sys` to the top import block of `tests/test_models.py`.

- [ ] **Step 2: Run to verify**

Run: `./.venv/Scripts/python -m pytest -q tests/test_models.py -k "memory or fit_worker_in_process" -p no:cacheprovider`
Expected: `4 passed` (the skipif test runs on this Windows box)

- [ ] **Step 3: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check tests/test_models.py
git add tests/test_models.py
git commit -m "test: cover memory helpers and the spawn worker in-process"
```

---

### Task 16: Verify the 90% target and the full fast gate

**Files:** none (verification only)

**Interfaces:**
- Consumes: the working coverage command from Global Constraints.

- [ ] **Step 1: Measure per-module coverage (statements + branches)**

Run: `./.venv/Scripts/python -m pytest tests/test_promote.py tests/test_models.py -q --no-header -p no:cacheprovider --cov=nmr --cov-branch --cov-report=term-missing 2>&1 | grep -E "promote\.py|models\.py"`
Expected: `nmr\promote.py` row shows ≥ 90% statements and `nmr\models.py` row shows ≥ 90% statements. (Baseline before this plan: 72% / 83%.) Record the branch percentages too — they are reported, not gated. If promote.py is below 90%, re-run the grep without the tail and list which lines remain missed, then report — do not silently accept.

- [ ] **Step 2: Full fast gate**

```bash
./.venv/Scripts/python -m ruff check .
./.venv/Scripts/python -m pytest -q -p no:cacheprovider
```

Expected: `ruff` clean, `865 passed` (all prior tests) + the ~25 new tests from this plan — total ≥ 890, zero failures, no new skips.

- [ ] **Step 3: Commit any stragglers**

If any test file edits remain uncommitted, commit them with a `test:` message describing the remaining coverage.

---

## Self-Review Notes

- **Spec coverage:** every missing-line block listed in the goal (promote.py 162-163, 167, 200-203, 223, 260-297, 299-331, 343-369, 395, 429-439, 477, 487, 530-531, 541, 545, 655, 677, 723-740, 781, 815, 823-824, 872, 892-894; models.py 218, 250, 283, 342, 345, 379, 391, 548, 602-609, 630, 632, 637, 723, 844, 856-857, 869-909, 931-964) maps to a task. Deliberately left uncovered: promote.py 395 is covered incidentally by Task 12's models work? No — 395 is `_full_history_frame`'s missing-file raise, covered by Task 8's missing-asset test only if routed through `_full_history_frame`. Correction: Task 8 covers `_build_truncated_data` (655, 677); promote.py 395 is `_full_history_frame`'s FileNotFoundError — add a note: Task 9's `measure_full_history_peak` uses `_full_history_frame`, and a missing-asset variant of Task 9 is NOT planned; accept 395 as residual or extend Task 8 with a direct `_full_history_frame` call in Task 8 Step 1 if 90% is not reached. models.py 454/461 (`_predict_model_chunked` defensive empties) are accepted residual — they are unreachable through the public API by construction.
- **Placeholder scan:** no TBDs; all commands and code are literal.
- **Type consistency:** `PromotionResult` fields used in Task 10 match promote.py:95-106 (artifact_path, manifest_path, run_id, family, tier4_gate_passed, override_used — peak fields default None). `Fold(0, ("1", "2"), ("3",))` matches splitter.py:30-33. Estimate/curve JSON keys match promote.py:257-331 verbatim.
