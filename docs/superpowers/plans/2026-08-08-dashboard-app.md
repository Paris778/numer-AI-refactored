# Streamlit + Plotly Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved dashboard design: an interactive Streamlit+Plotly app (`dashboard_app.py`) over the registry scorecards, benchmark CSV, and campaign logs — read-only, thin control plane, pure tested data-shaping helpers + a thin render layer.

**Architecture:** `dashboard_app.py` at the repo root. Pure data-shaping functions (unit-tested via `tests/test_scripts.py`, mirroring the `generate_dashboard.py` precedent) build DataFrames from `artifacts/registry/` run.json files, `artifacts/benchmark_scores*.csv`, and `artifacts/campaigns/*.json`; the Streamlit render layer only wires sidebar filters + Plotly charts + dataframes. Business logic beyond column selection reuses `nmr` public APIs (`nmr.meta.fleet_summary`, scorecard cells); no metrics are computed in the script (AGENTS §2.1).

**Tech Stack:** Python 3.11+ (venv 3.12), Polars/pandas, **Streamlit 1.61.1 + Plotly 6.6.0** (user-granted dependencies; plotly already installed as a transitive dep, pinned at the installed version).

## Global Constraints

- `nmr/` is the only tested boundary; `dashboard_app.py` is a thin control plane — no business logic (no metric formulas, no transforms, no registry writes). All data shaping is column selection/rename/join; the one computation (`fleet_summary`) lives in `nmr/meta.py`.
- TDD: no production code without a failing test first (pure helpers); the render layer is verified by an import-time smoke test.
- **Dependency exception (user-granted 2026-08-08):** `streamlit==1.61.1`, `plotly==6.6.0` pinned in `requirements.txt`.
- Read-only app: it never writes the registry, campaigns, or artifacts.
- No metric/purge/scorecard changes; nothing enters `canonical_scorecards_bytes`; no run_id impact.
- Doc SSOT same-change-set; AGENTS ≤ 32 KB; test-count claims synced numeric-only (precedent).
- Git flow (user-authorized pattern): work on branch `dashboard-app` (created), commits per task, `main` untouched until the user chooses integration (standing preference: local merge). No push.
- Verification honesty: run commands, read output, report truthfully.

## File Structure

| File | Responsibility |
|---|---|
| `dashboard_app.py` (new) | Pure shaping helpers + thin Streamlit render layer |
| `requirements.txt` (modify) | Pin `streamlit==1.61.1`, `plotly==6.6.0` |
| `tests/test_scripts.py` (modify) | Helper tests + import smoke |
| `README.md`, `ARCHITECTURE.md`, `AGENTS.md` (modify) | Tree + quickstart, §O row, toolkit row |
| `nmr/` | UNTOUCHED (read-only reuse of public APIs) |

---

### Task 1: Pin + install `streamlit==1.61.1`, `plotly==6.6.0`

**Files:** `requirements.txt`

- [ ] **Step 1: Pin** — append `streamlit==1.61.1` and `plotly==6.6.0` to `requirements.txt`.
- [ ] **Step 2: Install** — `.venv/Scripts/python -m pip install streamlit==1.61.1 plotly==6.6.0`.
- [ ] **Step 3: Verify** — `.venv/Scripts/python -c "import streamlit, plotly; print(streamlit.__version__, plotly.__version__)"` → `1.61.1 6.6.0`.
- [ ] **Step 4: Commit** — `build(deps): pin streamlit + plotly (user-granted exception)` on `dashboard-app`.

---

### Task 2: `dashboard_app.py` — pure data-shaping helpers

**Files:**
- Create: `dashboard_app.py` (helpers only; a stub `main()`/render guarded by `__main__` can exist but must be minimal)
- Test: `tests/test_scripts.py` (append)

**Interfaces:**
- Consumes: registry run.json layout (`metrics`, `manifest.config`, `scorecard` cells — the shape `RunRegistry.list()` returns), benchmark CSV columns (as `generate_dashboard._load_benchmarks` normalizes), campaign log payloads (`campaign_id/name/configs/runs`).
- Produces (all pure, unit-tested):
  - `load_registry_frame(registry_dir: Path) -> pl.DataFrame` — columns: `model_id, source ("trained"|"trained_legacy"), run_name, backend, preset, feature_set, feature_subset, n_targets, targets, neutralization_proportion, oof_device, corr, corr_ci_low, corr_ci_high, corr_sharpe_ac, corr_sharpe_ac_ci_low, corr_sharpe_ac_ci_high, max_drawdown, std_corr, deflated_sharpe, max_feature_exposure, has_bmc, has_horizon, has_perturb, has_regime, run_dir`. Legacy runs (no scorecard) fall back to `metrics` for corr/sharpe/dd with `source="trained_legacy"` (mirror `generate_dashboard._load_registry_runs` semantics; explicit None checks — a legitimate `0.0` must not fall through).
  - `load_benchmarks(path: Path) -> pl.DataFrame` — normalize the benchmark CSV to the same leaderboard columns with `source="benchmark"` (mirror `generate_dashboard._load_benchmarks`; missing file → empty frame).
  - `merge_leaderboard(registry: pl.DataFrame, benchmarks: pl.DataFrame) -> pl.DataFrame` — row-concat; benchmark rows carry their columns.
  - `load_campaigns(campaigns_dir: Path) -> pl.DataFrame` — flatten each `*.json` log: `campaign_id, name, config_path, run_id, status, error`; missing dir → empty frame.
  - `robustness_matrix(registry: pl.DataFrame) -> pl.DataFrame` — `model_id, has_bmc, has_horizon, has_perturb, has_regime, max_feature_exposure, std_corr, max_drawdown` (numeric casts for the heatmap).
  - `champion_run_id(registry_dir: Path) -> str | None` — read `champion.json` (missing/corrupt → None).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_scripts.py
import json
import polars as pl

import dashboard_app


def _registry_entry(run_id: str, *, scorecard: bool = True) -> dict:
    entry = {
        "run_id": run_id,
        "metrics": {"mean": 0.1, "std": 0.2, "sharpe": 0.5, "max_drawdown": 0.05},
        "manifest": {
            "oof_device": "cpu",
            "config": {
                "data": {"feature_set": "small", "feature_subset": None},
                "model": {"backend": "lightgbm", "preset": "fast"},
                "risk": {"neutralization_proportion": 1.0},
            },
        },
        "scorecard": None if not scorecard else {
            "corr": 0.12, "corr_ci_low": 0.05, "corr_ci_high": 0.19, "corr_n_eras": 30,
            "corr_sharpe_ac": 0.8, "corr_sharpe_ac_ci_low": 0.6, "corr_sharpe_ac_ci_high": 1.0,
            "max_drawdown": 0.1, "std_corr": 0.2, "deflated_sharpe": 0.97,
            "max_feature_exposure": 0.3, "bmc": 0.02, "horizon_model_sharpe_20": 0.5,
            "perturb_ceiling_stability": 0.9, "regime_count": 3,
        },
    }
    return entry


def _write_registry(tmp_path, entries) -> None:
    for entry in entries:
        run_dir = tmp_path / entry["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps(entry), encoding="utf-8")


def test_registry_frame_columns_and_source_tagging(tmp_path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64), _registry_entry("b" * 64, scorecard=False)])
    frame = dashboard_app.load_registry_frame(tmp_path)
    assert frame.height == 2
    assert set(frame.columns) >= {
        "model_id", "source", "backend", "preset", "feature_set", "feature_subset",
        "oof_device", "neutralization_proportion", "corr", "corr_sharpe_ac",
        "corr_sharpe_ac_ci_low", "corr_sharpe_ac_ci_high", "max_drawdown",
        "deflated_sharpe", "has_bmc", "has_horizon", "has_perturb", "has_regime",
    }
    rows = {r["model_id"]: r for r in frame.to_dicts()}
    assert rows["a" * 64]["source"] == "trained"
    assert rows["a" * 64]["corr"] == 0.12
    assert rows["a" * 64]["has_bmc"] is True
    assert rows["b" * 64]["source"] == "trained_legacy"
    assert rows["b" * 64]["corr"] == 0.1          # legacy falls back to metrics.mean
    assert rows["b" * 64]["has_bmc"] is False


def test_registry_frame_zero_value_not_treated_as_legacy(tmp_path) -> None:
    entry = _registry_entry("c" * 64)
    entry["scorecard"]["corr"] = 0.0               # legitimate 0.0 must NOT fall through
    _write_registry(tmp_path, [entry])
    frame = dashboard_app.load_registry_frame(tmp_path)
    assert frame.row(0, named=True)["corr"] == 0.0


def test_benchmarks_and_merge(tmp_path) -> None:
    bench_path = tmp_path / "benchmark_scores.csv"
    bench_path.write_text(
        "model_id,corr,corr_sharpe_ac,std_corr,max_drawdown,strategy_group,horizon_target_name\n"
        "bench_a,0.05,0.5,0.3,0.2,linear,cyrusd\n",
        encoding="utf-8",
    )
    benchmarks = dashboard_app.load_benchmarks(bench_path)
    assert benchmarks.height == 1
    assert benchmarks.row(0, named=True)["source"] == "benchmark"
    assert dashboard_app.load_benchmarks(tmp_path / "missing.csv").height == 0

    _write_registry(tmp_path, [_registry_entry("d" * 64)])
    registry = dashboard_app.load_registry_frame(tmp_path)
    merged = dashboard_app.merge_leaderboard(registry, benchmarks)
    assert merged.height == 2
    assert set(merged.get_column("source").to_list()) == {"trained", "benchmark"}


def test_campaigns_flatten(tmp_path) -> None:
    campaigns_dir = tmp_path / "campaigns"
    campaigns_dir.mkdir()
    (campaigns_dir / "abc.json").write_text(
        json.dumps({
            "campaign_id": "abc", "name": "camp",
            "configs": [{"path": "a.yaml", "sha256": "x" * 64}],
            "runs": [
                {"config_path": "a.yaml", "run_id": "e" * 64, "status": "recorded", "error": None},
                {"config_path": "a.yaml", "run_id": None, "status": "error", "error": "boom"},
            ],
        }),
        encoding="utf-8",
    )
    frame = dashboard_app.load_campaigns(campaigns_dir)
    assert frame.height == 2
    assert set(frame.columns) == {"campaign_id", "name", "config_path", "run_id", "status", "error"}
    assert frame.get_column("status").to_list() == ["recorded", "error"]
    assert dashboard_app.load_campaigns(tmp_path / "missing").height == 0


def test_robustness_matrix_and_champion(tmp_path) -> None:
    _write_registry(tmp_path, [_registry_entry("f" * 64)])
    matrix = dashboard_app.robustness_matrix(dashboard_app.load_registry_frame(tmp_path))
    assert matrix.height == 1
    assert {"has_bmc", "has_horizon", "has_perturb", "has_regime", "max_feature_exposure"} <= set(matrix.columns)
    assert dashboard_app.champion_run_id(tmp_path) is None      # no champion.json
    (tmp_path / "champion.json").write_text(json.dumps({"run_id": "f" * 64}), encoding="utf-8")
    assert dashboard_app.champion_run_id(tmp_path) == "f" * 64
```

- [ ] **Step 2: Run test to verify it fails** — `.venv/Scripts/python -m pytest tests/test_scripts.py -q` — FAIL: `dashboard_app` module missing.
- [ ] **Step 3: Implement** the helpers per the Interfaces contract (mirror `generate_dashboard._load_registry_runs`/`_load_benchmarks` column semantics; polars; explicit None checks for scorecard 0.0).
- [ ] **Step 4: Run test to verify it passes** — all `tests/test_scripts.py` green.
- [ ] **Step 5: Commit** — `feat(dashboard): add pure data-shaping helpers` on `dashboard-app`.

---

### Task 3: `dashboard_app.py` — Streamlit render layer + import smoke

**Files:**
- Modify: `dashboard_app.py` (append the render layer)
- Test: `tests/test_scripts.py` (append import smoke)

**Interfaces:**
- Consumes: Task 2 helpers + `nmr.meta.fleet_summary` + registry entries.
- Produces: `main()` guarded by `if __name__ == "__main__": main()` — thin Streamlit UI: sidebar filters (backend/preset/source), leaderboard (Plotly bar + CI error bars, champion highlighted), run-detail expander, fleet-summary table (via `nmr.meta.fleet_summary` on registry entries), campaign table, robustness heatmap (Plotly). Read-only; no writes.

- [ ] **Step 1: Write the failing test (import smoke)**

```python
# append to tests/test_scripts.py
def test_dashboard_app_imports_without_launching() -> None:
    import dashboard_app  # noqa: F401  (module-level import must not launch the server)

    assert callable(dashboard_app.main)
```

- [ ] **Step 2: Run test to verify it fails** — FAIL: `main` not defined (or module import error if the render layer references streamlit at module top — if so, guard the streamlit import so the module imports headless; the render functions are only called from `main()`).
- [ ] **Step 3: Implement** the render layer: thin, no business logic — it calls the Task 2 helpers + `fleet_summary` and passes frames to streamlit/plotly. `import streamlit as st` and `import plotly.express as px` at module top are fine (the smoke test imports the module; streamlit imports headless OK — verify; if `st.set_page_config` must run first in a script context, keep it inside `main()`).
- [ ] **Step 4: Run test to verify it passes** — all `tests/test_scripts.py` green.
- [ ] **Step 5: Commit** — `feat(dashboard): add streamlit render layer` on `dashboard-app`.

---

### Task 4: Docs + count sync

**Files:** `README.md`, `ARCHITECTURE.md`, `AGENTS.md`, count claims

- [ ] **Step 1: `README.md`** — tree entry: `├── dashboard_app.py          # interactive dashboard — streamlit run dashboard_app.py (registry/benchmarks/campaigns, read-only)`; quickstart: one line under the existing CLI section: `streamlit run dashboard_app.py   # interactive leaderboard + fleet + campaign views`.
- [ ] **Step 2: `ARCHITECTURE.md` §O** — table row: `[dashboard_app.py](dashboard_app.py)` — "Streamlit+Plotly interactive dashboard over registry scorecards, benchmark CSV, campaign logs, and `fleet_summary` (§Q); read-only; launch: `streamlit run dashboard_app.py`. Pure shaping helpers are unit-tested (tests/test_scripts.py)."
- [ ] **Step 3: `AGENTS.md`** — toolkit row: `Inspect runs / campaigns interactively | dashboard_app.py — streamlit run (read-only)`.
- [ ] **Step 4: Count sync** — compute the new total (403 + tests added in Tasks 2–3) and update the numeric claims in AGENTS.md/README.md/CONTRIBUTING.md (numeric-only, precedent).
- [ ] **Step 5: Run** — focused `tests/test_scripts.py` + full `pytest -q` + `tests/test_docs_hygiene.py -q`; all green.
- [ ] **Step 6: Commit** — `docs: document dashboard app (README, ARCH §O, AGENTS toolkit)` on `dashboard-app`.

---

### Task 5: Full verification gate

- [ ] **Step 1: Full suite** — `.venv/Scripts/python -m pytest -q` — PASS, exact count reported.
- [ ] **Step 2: Headless launch smoke** — `.venv/Scripts/python -c "import dashboard_app; assert callable(dashboard_app.main)"` — PASS (the app must import headless; no server launch in tests).
- [ ] **Step 3: Doc-SSOT scan** — AGENTS ≤ 32 KB; no duplicated facts across the four docs; `dashboard_app.py` referenced consistently.
- [ ] **Step 4: Record** — report truthfully.

---

## Self-Review

**Spec coverage** (design → task): deps → T1; shaping helpers → T2; render layer + smoke → T3; docs + count → T4; gate → T5. All five views (leaderboard, run detail, fleet, campaigns, robustness matrix) map to helpers/T2–T3. Read-only + thin-control-plane constraints hold throughout. **Placeholder scan:** none — helper signatures and test code are concrete; the render layer is specified by view list. **Type consistency:** helper signatures match across T2 tests/impl and T3 render calls; `fleet_summary` reused from `nmr.meta` (existing signature).
