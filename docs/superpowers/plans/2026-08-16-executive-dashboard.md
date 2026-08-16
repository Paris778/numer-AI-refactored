# Executive Model Performance Report — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, offline HTML performance report (`artifacts/dashboard.html`) for the portfolio owner, driven by a new pure engine `nmr/dashboard.py`, a plotly chart layer `dashboard_charts.py`, a rewritten `generate_dashboard.py`, and a rewired `dashboard_app.py`.

**Architecture:** One pure data engine in `nmr/` (polars/json/numpy only — no plotly/streamlit imports) feeds a top-level plotly figure layer; both presenters (static HTML primary, Streamlit secondary) consume the same engine + figures. Capital cells missing from legacy run.json scorecards are recomputed at report time from stored parquets via the oracle-parity `nmr.evaluation` / `nmr.payout` paths; the registry is never written.

**Tech Stack:** Python 3.12, Polars (primary), numpy, plotly 5.x (figures + inline JS), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-16-executive-dashboard-design.md` (read first — every "Decision" row is binding).

## Global Constraints

- Run all tests as `./.venv/Scripts/python -m pytest <args>`; lint as `./.venv/Scripts/python -m ruff check .` (config `ruff.toml`: E/F/I/UP, line-length 120). Never use the `Scripts/pip` shim.
- Business logic lives in `nmr/` only; top-level scripts are thin control planes. `nmr/` must never import `streamlit` or `plotly`.
- Registry immutability: no code in this plan ever writes under `artifacts/registry/`.
- Determinism: no wall-clock timestamps in the HTML; era lists always sorted numerically via `nmr.evaluation.sorted_era_labels`; stable sort orders (metric desc, `run_id` tiebreak); benchmark/registry scan order is `sorted()` by path.
- Explicit-None discipline: a legitimate scorecard `0.0` must never fall through to legacy train-OOF `metrics`.
- Gate thresholds are read from `configs/benchmarks/tier4_gate.yaml` via `nmr.benchmark.load_benchmark_file` — never hardcoded. Hard-hurdle semantics mirror `assert_tier4_gate`: `>=` for corr/sharpe/fnc/dsr/gtp, strict `>` for `cagr_1y`, `<=` for turnover only when `turnover_mean is not None`.
- Recompute inputs: `kelly_fraction` receives the **raw** payout series; `annual_compounded_return` / `gain_to_pain_ratio` receive the **clipped** series.
- Stored-first sentinel: a run's capital block is trusted iff `cagr_1y`, `gain_to_pain_ratio`, and `kelly_fraction` are **all** non-null in its run.json scorecard.
- Commit steps below require the user's explicit go-ahead (repo rule). If the user has not authorized commits, skip the commit step and continue to the next task.
- `tests/test_package_api.py` enforces that every public name in an `nmr` module's `__all__` is re-exported from `nmr/__init__.py` on every commit: each task that adds names to `nmr/dashboard.py.__all__` must update `nmr/__init__.py` (import block + `__all__`, alphabetical) in the same commit. `tests/test_docs_hygiene.py` enforces that the "N tests" claims in `AGENTS.md`, `README.md`, and `CONTRIBUTING.md` match `pytest --collect-only`; bump all three in the same commit that changes the test count.

---

## File Structure

- Create: `nmr/dashboard.py` — pure engine: schema, path resolution, leaderboard load, gate status, capital recompute, payout timeseries.
- Create: `dashboard_charts.py` — plotly figure builders (presentation only).
- Create: `tests/test_dashboard.py` — engine + charts + HTML contract tests.
- Modify: `generate_dashboard.py` — full rewrite into the executive HTML compiler.
- Modify: `dashboard_app.py` — thin rewire: `load_registry_frame` / `load_benchmarks` delegate to the engine; everything else unchanged.
- Modify: `tests/test_scripts.py` — replace the two obsolete `generate_dashboard` internals tests.
- Modify: `nmr/__init__.py` — export the new engine symbols (imports **and** `__all__`).
- Modify: `AGENTS.md` — one toolkit row for the dashboard engine (keep within the 32 KB budget).
- Modify: `ARCHITECTURE.md` — short module spec for `nmr/dashboard.py` + `dashboard_charts.py`.

---

### Task 1: Engine skeleton — schema, constants, `resolve_benchmark_path`

**Files:**
- Create: `nmr/dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Produces: `nmr.dashboard.UNIFIED_SCHEMA` (`pl.Schema`), `REPORTS_DIR: Path`, `LEGACY_BENCHMARK_PATH: Path`, `DEFAULT_REGISTRY_DIR: Path`, `DEFAULT_DATA_DIR: Path`, `DEFAULT_GATE_PATH: Path`, `resolve_benchmark_path(benchmark_path: Path | None | bool = None, reports_dir: Path | None = None, legacy_path: Path | None = None) -> Path | None`. ``benchmark_path=False`` disables the chain (test isolation).
- Consumed by: Tasks 2–6, 10, 11.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import nmr.dashboard as dash


def test_resolve_benchmark_path_prefers_given_existing(tmp_path: Path) -> None:
    given = tmp_path / "reports" / "benchmark_hierarchy_scorecard.csv"
    given.parent.mkdir(parents=True)
    given.write_text("x", encoding="utf-8")
    assert dash.resolve_benchmark_path(given) == given


def test_resolve_benchmark_path_chain_falls_back(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    smoke = reports / "benchmark_hierarchy_scorecard_smoke.csv"
    smoke.write_text("x", encoding="utf-8")
    legacy = tmp_path / "benchmark_scores.csv"
    legacy.write_text("x", encoding="utf-8")
    # given path missing -> chain: full (missing) -> smoke (hit)
    assert dash.resolve_benchmark_path(reports / "nope.csv", reports_dir=reports) == smoke
    smoke.unlink()
    assert dash.resolve_benchmark_path(reports / "nope.csv", reports_dir=reports) == legacy
    legacy.unlink()
    assert dash.resolve_benchmark_path(reports / "nope.csv", reports_dir=reports) is None


def test_resolve_benchmark_path_false_disables_chain(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "benchmark_hierarchy_scorecard_smoke.csv").write_text("x", encoding="utf-8")
    assert dash.resolve_benchmark_path(False, reports_dir=reports) is None


def test_unified_schema_contains_leaderboard_projection() -> None:
    # dashboard_app._LEADERBOARD_SCHEMA is a subset the app wrapper projects onto.
    required = {
        "model_id", "source", "run_name", "backend", "preset", "feature_set",
        "feature_subset", "n_targets", "targets", "neutralization_proportion",
        "oof_device", "corr", "corr_ci_low", "corr_ci_high",
        "corr_sharpe_ac", "corr_sharpe_ac_ci_low", "corr_sharpe_ac_ci_high",
        "max_drawdown", "std_corr", "deflated_sharpe", "max_feature_exposure",
        "has_bmc", "has_horizon", "has_perturb", "has_regime", "run_dir",
    }
    assert required <= set(dash.UNIFIED_SCHEMA.names())
    # capital cells required by the executive table
    for col in ("cagr_1y", "gain_to_pain_ratio", "kelly_fraction", "mmc_down",
                "fnc", "mmc", "mean_payout", "n_eras", "tier", "turnover_mean"):
        assert col in dash.UNIFIED_SCHEMA.names()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -v`
Expected: FAIL — `ModuleNotFoundError: nmr.dashboard`.

- [ ] **Step 3: Write minimal implementation**

```python
# nmr/dashboard.py
"""Pure analytical engine for the executive performance dashboard.

Registry scans, benchmark reconciliation, gate projection, capital-cell
recompute, and payout timeseries extraction. Plotly/Streamlit-free; every
function here is covered by tests/test_dashboard.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from nmr.config import REPO_ROOT

logger = logging.getLogger("nmr.dashboard")

__all__ = [
    "UNIFIED_SCHEMA",
    "evaluate_gate_status",
    "extract_payout_timeseries",
    "load_benchmark_frame",
    "load_unified_leaderboard",
    "reconcile_capital_metrics",
    "resolve_benchmark_path",
]

REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"
LEGACY_BENCHMARK_PATH = REPO_ROOT / "artifacts" / "benchmark_scores.csv"
DEFAULT_REGISTRY_DIR = REPO_ROOT / "artifacts" / "registry"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "v5.3"
DEFAULT_GATE_PATH = REPO_ROOT / "configs" / "benchmarks" / "tier4_gate.yaml"

# Superset of dashboard_app._LEADERBOARD_SCHEMA plus the capital-readiness,
# gate, and tier columns consumed by the executive report.
UNIFIED_SCHEMA = pl.Schema(
    {
        "model_id": pl.String, "source": pl.String, "run_name": pl.String,
        "backend": pl.String, "preset": pl.String, "feature_set": pl.String,
        "feature_subset": pl.String, "n_targets": pl.Int64, "targets": pl.String,
        "neutralization_proportion": pl.Float64, "oof_device": pl.String,
        "corr": pl.Float64, "corr_ci_low": pl.Float64, "corr_ci_high": pl.Float64,
        "corr_n_eras": pl.Int64,
        "corr_sharpe_ac": pl.Float64, "corr_sharpe_ac_ci_low": pl.Float64,
        "corr_sharpe_ac_ci_high": pl.Float64, "corr_sharpe_ac_n_eras": pl.Int64,
        "std_corr": pl.Float64, "max_drawdown": pl.Float64,
        "deflated_sharpe": pl.Float64, "fnc": pl.Float64, "mmc": pl.Float64,
        "mmc_sharpe_ac": pl.Float64, "bmc": pl.Float64, "cwmm": pl.Float64,
        "mean_payout": pl.Float64,
        "cagr_1y": pl.Float64, "gain_to_pain_ratio": pl.Float64,
        "kelly_fraction": pl.Float64, "mmc_down": pl.Float64,
        "mmc_down_reason": pl.String, "turnover_mean": pl.Float64,
        "n_eras": pl.Int64, "rank_scalar": pl.Float64, "cvar5": pl.Float64,
        "burn_rate": pl.Float64, "max_feature_exposure": pl.Float64,
        "has_bmc": pl.Boolean, "has_horizon": pl.Boolean,
        "has_perturb": pl.Boolean, "has_regime": pl.Boolean,
        "tier": pl.Int64, "run_dir": pl.String,
    }
)


def resolve_benchmark_path(
    benchmark_path: Path | None | bool = None,
    reports_dir: Path | None = None,
    legacy_path: Path | None = None,
) -> Path | None:
    """Resolve the benchmark scorecard CSV via the fallback chain.

    Chain: given path (if it exists) -> full hierarchy CSV -> smoke CSV ->
    legacy CSV -> None. ``benchmark_path=False`` is an explicit directive to
    disable benchmark loading entirely (test isolation).
    """
    if benchmark_path is False:
        return None
    if benchmark_path is not None:
        given = Path(benchmark_path)
        if given.exists():
            return given
    reports = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    legacy = Path(legacy_path) if legacy_path is not None else LEGACY_BENCHMARK_PATH
    for candidate in (
        reports / "benchmark_hierarchy_scorecard.csv",
        reports / "benchmark_hierarchy_scorecard_smoke.csv",
        legacy,
    ):
        if candidate.exists():
            return candidate
    return None
```

(Only Task 1 symbols are implemented here; later tasks add their functions to this same file.)

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit** (skip if commits not authorized)

```bash
git add nmr/dashboard.py tests/test_dashboard.py
git commit -m "feat(dashboard): engine skeleton with unified schema and benchmark path chain"
```

---

### Task 2: `load_benchmark_frame` — benchmark CSV → unified rows

**Files:**
- Modify: `nmr/dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `UNIFIED_SCHEMA`, `resolve_benchmark_path` (Task 1).
- Produces: `load_benchmark_frame(benchmark_path: Path) -> pl.DataFrame` — benchmark rows in the unified schema; empty schema frame when the file is missing or a parsed-but-empty CSV (zero-byte/corrupt files raise — fail loud). Missing CSV columns map to `None` (no exceptions), except `std_corr`/`max_drawdown` which keep the legacy `0.0` default; `run_name` = `strategy_group`, falling back to `"tier{int(tier)}"` when `tier` is present, else `"benchmark"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py`:

```python
def _write_benchmark_csv(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "bench.csv"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_benchmark_frame_full_and_minimal(tmp_path: Path) -> None:
    full = _write_benchmark_csv(
        tmp_path,
        "model_id,corr,corr_ci_low,corr_ci_high,corr_sharpe_ac,"
        "corr_sharpe_ac_ci_low,corr_sharpe_ac_ci_high,std_corr,max_drawdown,"
        "deflated_sharpe,fnc,cagr_1y,gain_to_pain_ratio,kelly_fraction,"
        "mmc_down,strategy_group,tier\n"
        "v53_lgbm_ender60,0.029,0.022,0.036,0.78,0.61,0.95,0.02,0.04,"
        "1.0,0.027,4.88,44.28,1.0,0.009,ref,4\n",
    )
    frame = dash.load_benchmark_frame(full)
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["model_id"] == "v53_lgbm_ender60"
    assert row["source"] == "benchmark"
    assert row["run_name"] == "ref"
    assert row["tier"] == 4
    assert row["cagr_1y"] == pytest.approx(4.88)
    assert row["gain_to_pain_ratio"] == pytest.approx(44.28)
    assert row["corr_sharpe_ac_ci_low"] == pytest.approx(0.61)

    minimal = _write_benchmark_csv(
        tmp_path,
        "model_id,corr,corr_sharpe_ac,std_corr,max_drawdown,strategy_group\n"
        "bench_a,0.05,0.5,0.3,0.2,linear\n",
    )
    row = dash.load_benchmark_frame(minimal).row(0, named=True)
    assert row["corr_sharpe_ac_ci_low"] is None
    assert row["fnc"] is None
    assert row["has_bmc"] is False


def test_load_benchmark_frame_missing_file_returns_empty_schema_frame(tmp_path: Path) -> None:
    frame = dash.load_benchmark_frame(tmp_path / "missing.csv")
    assert frame.height == 0
    assert frame.schema == dash.UNIFIED_SCHEMA
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k benchmark_frame -v`
Expected: FAIL — `AttributeError: module 'nmr.dashboard' has no attribute 'load_benchmark_frame'`.

- [ ] **Step 3: Write minimal implementation**

Append to `nmr/dashboard.py`:

```python
def load_benchmark_frame(benchmark_path: Path) -> pl.DataFrame:
    """Normalize a benchmark scorecard CSV into unified-schema rows.

    Mirrors the legacy ``dashboard_app.load_benchmarks`` column semantics but
    carries the full scorecard mapping (fnc, deflated_sharpe, capital cells,
    CIs). A missing file, or a parsed-but-empty CSV, returns the empty schema
    frame (zero-byte/corrupt files raise — fail loud).
    """
    path = Path(benchmark_path)
    if not path.exists():
        return pl.DataFrame(schema=UNIFIED_SCHEMA)
    df = pl.read_csv(path)
    if df.height == 0:
        return pl.DataFrame(schema=UNIFIED_SCHEMA)

    rows: list[dict] = []
    for row in df.to_dicts():
        tier_value = row.get("tier")
        rows.append(
            {
                "model_id": row.get("model_id"),
                "source": "benchmark",
                "run_name": row.get("strategy_group")
                or (
                    f"tier{int(tier_value)}"
                    if tier_value is not None
                    else "benchmark"
                ),
                "backend": "benchmark",
                "preset": "benchmark",
                "feature_set": "all",
                "feature_subset": None,
                "n_targets": 1,
                "targets": row.get("horizon_target_name") or "target",
                "neutralization_proportion": None,
                "oof_device": None,
                "corr": row.get("corr"),
                "corr_ci_low": row.get("corr_ci_low"),
                "corr_ci_high": row.get("corr_ci_high"),
                "corr_n_eras": row.get("corr_n_eras"),
                "corr_sharpe_ac": row.get("corr_sharpe_ac"),
                "corr_sharpe_ac_ci_low": row.get("corr_sharpe_ac_ci_low"),
                "corr_sharpe_ac_ci_high": row.get("corr_sharpe_ac_ci_high"),
                "corr_sharpe_ac_n_eras": row.get("corr_sharpe_ac_n_eras"),
                "std_corr": row.get("std_corr", 0.0),
                "max_drawdown": row.get("max_drawdown", 0.0),
                "deflated_sharpe": row.get("deflated_sharpe"),
                "fnc": row.get("fnc"),
                "mmc": row.get("mmc"),
                "mmc_sharpe_ac": row.get("mmc_sharpe_ac"),
                "bmc": row.get("bmc"),
                "cwmm": row.get("cwmm"),
                "mean_payout": row.get("mean_payout"),
                "cagr_1y": row.get("cagr_1y"),
                "gain_to_pain_ratio": row.get("gain_to_pain_ratio"),
                "kelly_fraction": row.get("kelly_fraction"),
                "mmc_down": row.get("mmc_down"),
                "mmc_down_reason": row.get("mmc_down_reason"),
                "turnover_mean": row.get("turnover_mean"),
                "n_eras": row.get("n_eras"),
                "rank_scalar": row.get("rank_scalar"),
                "cvar5": row.get("cvar5"),
                "burn_rate": row.get("burn_rate"),
                "max_feature_exposure": row.get("max_feature_exposure"),
                "has_bmc": row.get("bmc") is not None,
                "has_horizon": row.get("horizon_model_sharpe_20") is not None,
                "has_perturb": row.get("perturb_ceiling_stability") is not None,
                "has_regime": row.get("regime_count") is not None,
                "tier": tier_value,
                "run_dir": str(path),
            }
        )
    return pl.DataFrame(rows, schema=UNIFIED_SCHEMA, strict=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k benchmark_frame -v`
Expected: PASS.

- [ ] **Step 5: Commit** (skip if commits not authorized)

```bash
git add nmr/dashboard.py tests/test_dashboard.py
git commit -m "feat(dashboard): benchmark CSV loader with full scorecard mapping"
```

---

### Task 3: `load_unified_leaderboard` — registry scan + merge

**Files:**
- Modify: `nmr/dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `UNIFIED_SCHEMA`, `resolve_benchmark_path`, `load_benchmark_frame`.
- Produces: `load_unified_leaderboard(registry_dir: Path, benchmark_path: Path | None | bool = None, reports_dir: Path | None = None) -> pl.DataFrame` — registry rows (`source="trained"` when scorecard present else `"trained_legacy"`) concatenated with benchmark rows from the resolved chain (only when resolution succeeds). Corrupt `run.json` → skipped; empty → empty schema frame. Pass ``benchmark_path=False`` for registry-only loads so live repo benchmark CSVs never leak into test frames.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py` (add `import json` at the top of the file):

```python
def _registry_entry(run_id: str, *, scorecard: bool = True) -> dict:
    entry = {
        "run_id": run_id,
        "metrics": {"mean": 0.1, "std": 0.2, "sharpe": 0.5, "max_drawdown": 0.05},
        "manifest": {
            "oof_device": "cpu",
            "config": {
                "data": {"feature_set": "small", "feature_subset": None,
                         "targets": ["target"]},
                "model": {"backend": "lightgbm", "preset": "fast"},
                "risk": {"neutralization_proportion": 1.0},
                "run": {"name": "sample-run"},
            },
        },
        "scorecard": None if not scorecard else {
            "corr": 0.12, "corr_ci_low": 0.05, "corr_ci_high": 0.19, "corr_n_eras": 30,
            "corr_sharpe_ac": 0.8, "corr_sharpe_ac_ci_low": 0.6, "corr_sharpe_ac_ci_high": 1.0,
            "max_drawdown": 0.1, "std_corr": 0.2, "deflated_sharpe": 0.97,
            "max_feature_exposure": 0.3, "bmc": 0.02, "fnc": 0.05, "n_eras": 30,
            "cagr_1y": 1.5, "gain_to_pain_ratio": 2.0, "kelly_fraction": 0.4,
            "mmc_down": 0.01, "mmc_down_reason": None,
        },
    }
    return entry


def _write_registry(tmp_path: Path, entries: list[dict]) -> None:
    for entry in entries:
        run_dir = tmp_path / entry["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps(entry), encoding="utf-8")


def test_load_unified_leaderboard_registry_only(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64), _registry_entry("b" * 64, scorecard=False)])
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    assert frame.height == 2
    rows = {r["model_id"]: r for r in frame.to_dicts()}
    assert rows["a" * 64]["source"] == "trained"
    assert rows["a" * 64]["corr"] == 0.12
    assert rows["b" * 64]["source"] == "trained_legacy"
    assert rows["b" * 64]["corr"] == 0.1  # legacy falls back to metrics.mean
    assert rows["a" * 64]["cagr_1y"] == 1.5  # stored capital block carried through


def test_load_unified_leaderboard_zero_scorecard_value_not_legacy(tmp_path: Path) -> None:
    entry = _registry_entry("c" * 64)
    entry["scorecard"]["corr"] = 0.0  # legitimate 0.0 must NOT fall through
    _write_registry(tmp_path, [entry])
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    assert frame.row(0, named=True)["corr"] == 0.0
    assert frame.row(0, named=True)["source"] == "trained"


def test_load_unified_leaderboard_corrupt_run_json_skipped(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("d" * 64)])
    bad_dir = tmp_path / ("e" * 64)
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "run.json").write_text("{not json", encoding="utf-8")
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    assert frame.height == 1


def test_load_unified_leaderboard_merges_benchmarks(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("f" * 64)])
    bench = _write_benchmark_csv(
        tmp_path,
        "model_id,corr,corr_sharpe_ac,std_corr,max_drawdown,strategy_group,tier\n"
        "bench_a,0.05,0.5,0.3,0.2,linear,1\n",
    )
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=bench)
    assert frame.height == 2
    assert set(frame.get_column("source").to_list()) == {"trained", "benchmark"}


def test_load_unified_leaderboard_empty_registry_returns_schema_frame(tmp_path: Path) -> None:
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    assert frame.height == 0
    assert frame.schema == dash.UNIFIED_SCHEMA
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k unified_leaderboard -v`
Expected: FAIL — `AttributeError: ... has no attribute 'load_unified_leaderboard'`.

- [ ] **Step 3: Write minimal implementation**

Append to `nmr/dashboard.py` (add `import json` to the imports at the top):

```python
def load_unified_leaderboard(
    registry_dir: Path,
    benchmark_path: Path | None | bool = None,
    reports_dir: Path | None = None,
) -> pl.DataFrame:
    """Load registry runs and (optionally) benchmark rows into one frame.

    Explicit-None discipline: a scorecard value of 0.0 is real and must not
    fall through to the legacy train-OOF ``metrics``. Corrupt ``run.json``
    files are skipped. ``benchmark_path=False`` disables benchmark loading
    (registry-only); otherwise a missing path falls through the resolution
    chain.
    """
    rows: list[dict] = []
    registry = Path(registry_dir)
    for run_file in sorted(registry.glob("*/run.json")):
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics") or {}
        manifest = payload.get("manifest") or {}
        cfg = manifest.get("config") or {}
        data_cfg = cfg.get("data") or {}
        model_cfg = cfg.get("model") or {}
        run_cfg = cfg.get("run") or {}
        risk_cfg = cfg.get("risk") or {}

        scorecard = payload.get("scorecard") or {}
        sc_corr = scorecard.get("corr")
        sc_sharpe = scorecard.get("corr_sharpe_ac")
        sc_std = scorecard.get("std_corr")
        sc_dd = scorecard.get("max_drawdown")
        rows.append(
            {
                "model_id": payload.get("run_id") or run_file.parent.name,
                "source": "trained" if scorecard else "trained_legacy",
                "run_name": run_cfg.get("name", "unknown"),
                "backend": model_cfg.get("backend", "unknown"),
                "preset": model_cfg.get("preset", "unknown"),
                "feature_set": data_cfg.get("feature_set", "unknown"),
                "feature_subset": data_cfg.get("feature_subset"),
                "n_targets": len(data_cfg.get("targets", [])),
                "targets": ", ".join(data_cfg.get("targets", [])),
                "neutralization_proportion": risk_cfg.get("neutralization_proportion"),
                "oof_device": manifest.get("oof_device"),
                "corr": float(sc_corr if sc_corr is not None else metrics.get("mean", 0.0)),
                "corr_ci_low": scorecard.get("corr_ci_low"),
                "corr_ci_high": scorecard.get("corr_ci_high"),
                "corr_n_eras": scorecard.get("corr_n_eras"),
                "corr_sharpe_ac": float(
                    sc_sharpe if sc_sharpe is not None else metrics.get("sharpe", 0.0)
                ),
                "corr_sharpe_ac_ci_low": scorecard.get("corr_sharpe_ac_ci_low"),
                "corr_sharpe_ac_ci_high": scorecard.get("corr_sharpe_ac_ci_high"),
                "corr_sharpe_ac_n_eras": scorecard.get("corr_sharpe_ac_n_eras"),
                "std_corr": float(sc_std if sc_std is not None else metrics.get("std", 0.0)),
                "max_drawdown": float(
                    sc_dd if sc_dd is not None else metrics.get("max_drawdown", 0.0)
                ),
                "deflated_sharpe": scorecard.get("deflated_sharpe"),
                "fnc": scorecard.get("fnc"),
                "mmc": scorecard.get("mmc"),
                "mmc_sharpe_ac": scorecard.get("mmc_sharpe_ac"),
                "bmc": scorecard.get("bmc"),
                "cwmm": scorecard.get("cwmm"),
                "mean_payout": scorecard.get("mean_payout"),
                "cagr_1y": scorecard.get("cagr_1y"),
                "gain_to_pain_ratio": scorecard.get("gain_to_pain_ratio"),
                "kelly_fraction": scorecard.get("kelly_fraction"),
                "mmc_down": scorecard.get("mmc_down"),
                "mmc_down_reason": scorecard.get("mmc_down_reason"),
                "turnover_mean": scorecard.get("turnover_mean"),
                "n_eras": scorecard.get("n_eras"),
                "rank_scalar": scorecard.get("rank_scalar"),
                "cvar5": scorecard.get("cvar5"),
                "burn_rate": scorecard.get("burn_rate"),
                "max_feature_exposure": scorecard.get("max_feature_exposure"),
                "has_bmc": scorecard.get("bmc") is not None,
                "has_horizon": scorecard.get("horizon_model_sharpe_20") is not None,
                "has_perturb": scorecard.get("perturb_ceiling_stability") is not None,
                "has_regime": scorecard.get("regime_count") is not None,
                "tier": None,
                "run_dir": str(run_file.parent),
            }
        )

    resolved = resolve_benchmark_path(benchmark_path, reports_dir=reports_dir)
    if resolved is not None:
        rows.extend(load_benchmark_frame(resolved).to_dicts())

    if not rows:
        return pl.DataFrame(schema=UNIFIED_SCHEMA)
    return pl.DataFrame(rows, schema=UNIFIED_SCHEMA, strict=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k unified_leaderboard -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit** (skip if commits not authorized)

```bash
git add nmr/dashboard.py tests/test_dashboard.py
git commit -m "feat(dashboard): unified leaderboard loader (registry + benchmarks)"
```

---

### Task 4: `evaluate_gate_status`

**Files:**
- Modify: `nmr/dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `UNIFIED_SCHEMA`, `nmr.benchmark.load_benchmark_file`, real `configs/benchmarks/tier4_gate.yaml`.
- Produces: `evaluate_gate_status(leaderboard: pl.DataFrame, gate_config_path: Path, champion_path: Path) -> pl.DataFrame` with columns `model_id: str`, `status: str` (`CHAMPION` | `CAPITAL READY` | `RESEARCH` | `GATE HURDLE` | `BENCHMARK`), and `gate_<field>` booleans for `corr`, `corr_sharpe_ac`, `fnc`, `deflated_sharpe`, `gain_to_pain_ratio`, `cagr_1y`, `turnover_mean`. Receipts are `None` when the cell is missing; turnover is exempt when `None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py`:

```python
from nmr.config import REPO_ROOT

_GATE_YAML = REPO_ROOT / "configs" / "benchmarks" / "tier4_gate.yaml"


def _status_frame(tmp_path: Path, rows: list[dict]) -> pl.DataFrame:
    frame = pl.DataFrame(rows, schema=dash.UNIFIED_SCHEMA, strict=False)
    return dash.evaluate_gate_status(frame, _GATE_YAML, tmp_path / "champion.json")


def test_gate_status_research_and_capital_ready(tmp_path: Path) -> None:
    base = {
        "model_id": "r1", "source": "trained", "corr": 0.01,
        "corr_sharpe_ac": 0.2, "fnc": 0.001, "deflated_sharpe": 0.5,
        "gain_to_pain_ratio": 1.0, "cagr_1y": 0.1, "turnover_mean": None,
    }
    frame = _status_frame(tmp_path, [base])
    assert frame.row(0, named=True)["status"] == "RESEARCH"
    assert frame.row(0, named=True)["gate_corr"] is False

    passing = dict(base)
    passing.update({"model_id": "r2", "corr": 0.03, "corr_sharpe_ac": 0.8,
                    "fnc": 0.03, "deflated_sharpe": 0.96,
                    "gain_to_pain_ratio": 1.6, "cagr_1y": 0.01})
    frame = _status_frame(tmp_path, [passing])
    assert frame.row(0, named=True)["status"] == "CAPITAL READY"
    assert frame.row(0, named=True)["gate_corr"] is True
    assert frame.row(0, named=True)["gate_cagr_1y"] is True  # strict > 0.0


def test_gate_status_champion_via_pointer(tmp_path: Path) -> None:
    (tmp_path / "champion.json").write_text(
        json.dumps({"run_id": "ch" * 32}), encoding="utf-8"
    )
    frame = _status_frame(tmp_path, [{"model_id": "ch" * 32, "source": "trained",
                                      "corr": 0.0, "corr_sharpe_ac": 0.0,
                                      "fnc": 0.0, "deflated_sharpe": 0.0,
                                      "gain_to_pain_ratio": 0.0,
                                      "cagr_1y": 0.0, "turnover_mean": None}])
    assert frame.row(0, named=True)["status"] == "CHAMPION"


def test_gate_status_benchmark_rows_never_capital_ready(tmp_path: Path) -> None:
    ref = {"model_id": "v53_lgbm_ender60", "source": "benchmark", "corr": 0.029,
           "corr_sharpe_ac": 0.78, "fnc": 0.027, "deflated_sharpe": 1.0,
           "gain_to_pain_ratio": 44.0, "cagr_1y": 4.88, "turnover_mean": None}
    other = dict(ref, model_id="null_constant_05", corr=0.0, corr_sharpe_ac=0.0)
    frame = _status_frame(tmp_path, [ref, other])
    statuses = {r["model_id"]: r["status"] for r in frame.to_dicts()}
    assert statuses["v53_lgbm_ender60"] == "GATE HURDLE"
    assert statuses["null_constant_05"] == "BENCHMARK"
    ref_row = frame.filter(pl.col("model_id") == "v53_lgbm_ender60").row(0, named=True)
    assert ref_row["gate_corr_sharpe_ac"] is True
    assert ref_row["gate_turnover_mean"] is None  # turnover absent -> exempt


def test_gate_status_turnover_violation_when_present(tmp_path: Path) -> None:
    row = {"model_id": "r3", "source": "trained", "corr": 0.03,
           "corr_sharpe_ac": 0.8, "fnc": 0.03, "deflated_sharpe": 0.96,
           "gain_to_pain_ratio": 1.6, "cagr_1y": 0.01, "turnover_mean": 0.9}
    frame = _status_frame(tmp_path, [row])
    out = frame.row(0, named=True)
    assert out["gate_turnover_mean"] is False  # 0.9 > 0.35
    assert out["status"] == "RESEARCH"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k gate_status -v`
Expected: FAIL — `AttributeError: ... has no attribute 'evaluate_gate_status'`.

- [ ] **Step 3: Write minimal implementation**

Append to `nmr/dashboard.py`:

```python
_GATE_THRESHOLD_ATTRS = {
    "corr": "corr_min",
    "corr_sharpe_ac": "corr_sharpe_ac_min",
    "fnc": "fnc_min",
    "deflated_sharpe": "deflated_sharpe_min",
    "gain_to_pain_ratio": "gain_to_pain_min",
    "cagr_1y": "cagr_min",
}
_GATE_FIELDS = ("corr", "corr_sharpe_ac", "fnc", "deflated_sharpe",
                "gain_to_pain_ratio", "cagr_1y", "turnover_mean")
_STATUS_SCHEMA = pl.Schema(
    {"model_id": pl.String, "status": pl.String,
     **{f"gate_{f}": pl.Boolean for f in _GATE_FIELDS}}
)


def _read_champion_pointer(champion_path: Path) -> str | None:
    """Opaque champion pointer; missing or corrupt file -> None."""
    path = Path(champion_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    return run_id if isinstance(run_id, str) else None


def _gate_receipt(field: str, row: dict, gate) -> bool | None:
    value = row.get(field)
    if value is None:
        return None
    measured = float(value)
    if field == "turnover_mean":
        return measured <= float(gate.turnover_max)
    if field == "cagr_1y":
        return measured > float(gate.cagr_min)  # strict, mirrors assert_tier4_gate
    return measured >= float(getattr(gate, _GATE_THRESHOLD_ATTRS[field]))


def evaluate_gate_status(
    leaderboard: pl.DataFrame,
    gate_config_path: Path,
    champion_path: Path,
) -> pl.DataFrame:
    """Project each row against the tier-4 gate (read-only, never enforces).

    Status ladder: benchmark rows are exempt (``GATE HURDLE`` for the gate
    file's reference column, ``BENCHMARK`` otherwise); registry rows are
    ``CHAMPION`` (champion.json pointer), ``CAPITAL READY`` (all hard
    hurdles), or ``RESEARCH``. Per-field receipts mirror
    ``assert_tier4_gate``: >= for most fields, strict > for cagr_1y, turnover
    exempt when None.
    """
    from nmr.benchmark import load_benchmark_file

    file_cfg = load_benchmark_file(gate_config_path)
    gate = file_cfg.gate
    if gate is None:
        raise ValueError(f"gate config {gate_config_path} has no gate section")
    reference_column = file_cfg.reference_column
    champion_id = _read_champion_pointer(champion_path)

    rows: list[dict] = []
    for row in leaderboard.to_dicts():
        model_id = row["model_id"]
        if row["source"] == "benchmark":
            status = "GATE HURDLE" if model_id == reference_column else "BENCHMARK"
        elif champion_id is not None and model_id == champion_id:
            status = "CHAMPION"
        elif (
            all(
                _gate_receipt(f, row, gate) is True
                for f in _GATE_FIELDS
                if f != "turnover_mean"
            )
            # turnover is exempt only when None; a measured violation blocks the gate
            and _gate_receipt("turnover_mean", row, gate) is not False
        ):
            status = "CAPITAL READY"
        else:
            status = "RESEARCH"
        rows.append(
            {"model_id": model_id, "status": status,
             **{f"gate_{f}": _gate_receipt(f, row, gate) for f in _GATE_FIELDS}}
        )
    if not rows:
        return pl.DataFrame(schema=_STATUS_SCHEMA)
    return pl.DataFrame(rows, schema=_STATUS_SCHEMA, strict=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k gate_status -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit** (skip if commits not authorized)

```bash
git add nmr/dashboard.py tests/test_dashboard.py
git commit -m "feat(dashboard): gate status projection from tier4_gate.yaml"
```

---

### Task 5: `reconcile_capital_metrics`

**Files:**
- Modify: `nmr/dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `UNIFIED_SCHEMA`; `nmr.evaluation.{EvaluationEngine, downside_era_indices}`; `nmr.payout.{payout_series, annual_compounded_return, gain_to_pain_ratio, kelly_fraction}`; public alias `MMC_DOWN_MIN_ERAS` from `nmr.scorecard` (the alias is added to `nmr/scorecard.py` in this task — see the implementer note in Step 3).
- Produces: `reconcile_capital_metrics(leaderboard: pl.DataFrame, registry_dir: Path, data_dir: Path) -> pl.DataFrame` — same schema; fills `cagr_1y`, `gain_to_pain_ratio`, `kelly_fraction`, `mmc_down` (+ `mmc_down_reason`) for trained/trained_legacy rows lacking the stored block; leaves cells untouched otherwise; missing data assets or missing `validation_preds.parquet` → cells stay `None` with a warning, no exception.
- Private helpers also produced here (reused by Task 6 — `_load_shared_lookups` and `_per_era_metrics`; `_has_stored_capital_block` is Task-5-local): `_has_stored_capital_block(row: dict) -> bool`, `_load_shared_lookups(data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, list[str]] | None` (returns `(targets_86, meta, meta_eras)`), `_per_era_metrics(preds_path: Path, targets_86: pl.DataFrame, meta: pl.DataFrame) -> tuple[dict[str, float], dict[str, float], dict[str, float]]` returning `(corr, mmc, meta_corr)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py` (no new imports — the tests use only `json`, `pl`, `pytest`, and `dash` already imported):

```python
def _synthetic_data_dir(tmp_path: Path) -> Path:
    """era/id/target + meta parquets over 3 eras with a perfectly
    correlated predictor so recomputed values are exactly known."""
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    for era in ("0001", "0002", "0003"):
        for i in range(10):
            t = float(i)
            rows.append({"era": era, "id": f"{era}_{i:03d}", "target": t})
    targets = pl.DataFrame(rows)
    meta = targets.select(
        [pl.col("era"), pl.col("id"), pl.col("target").alias("numerai_meta_model")]
    )
    targets.write_parquet(data / "validation.parquet")
    meta.write_parquet(data / "meta_model.parquet")
    return data


def _write_preds(run_dir: Path, scale: float) -> None:
    preds = [
        {"era": era, "id": f"{era}_{i:03d}", "prediction": scale * float(i)}
        for era in ("0001", "0002", "0003")
        for i in range(10)
    ]
    pl.DataFrame(preds).write_parquet(run_dir / "validation_preds.parquet")


def test_reconcile_capital_metrics_recomputes_missing_block(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64, scorecard=True)])
    entry = json.loads((tmp_path / ("a" * 64) / "run.json").read_text(encoding="utf-8"))
    del entry["scorecard"]["cagr_1y"]
    del entry["scorecard"]["gain_to_pain_ratio"]
    del entry["scorecard"]["kelly_fraction"]
    del entry["scorecard"]["mmc_down"]
    (tmp_path / ("a" * 64) / "run.json").write_text(json.dumps(entry), encoding="utf-8")
    _write_preds(tmp_path / ("a" * 64), scale=1.0)
    data = _synthetic_data_dir(tmp_path)

    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    out = dash.reconcile_capital_metrics(frame, tmp_path, data)
    row = out.row(0, named=True)
    assert row["cagr_1y"] is not None
    assert row["gain_to_pain_ratio"] is not None
    assert row["kelly_fraction"] is not None
    # perfect corr with target and meta -> payout == 0.05 clipped every era
    # repo formula: prod(1+r)^(52/n) - 1 => (1.05^3)^(52/3) - 1 = 1.05^52 - 1
    assert row["cagr_1y"] == pytest.approx((1.05 ** 52) - 1.0, rel=1e-6)
    assert row["gain_to_pain_ratio"] == float("inf")  # no losing eras


def test_reconcile_capital_metrics_stored_block_wins(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("b" * 64)])
    _write_preds(tmp_path / ("b" * 64), scale=0.0)  # junk preds must be ignored
    data = _synthetic_data_dir(tmp_path)
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    out = dash.reconcile_capital_metrics(frame, tmp_path, data)
    row = out.row(0, named=True)
    assert row["cagr_1y"] == 1.5       # stored value untouched
    assert row["gain_to_pain_ratio"] == 2.0
    assert row["kelly_fraction"] == 0.4
    assert row["mmc_down"] == 0.01


def test_reconcile_capital_metrics_missing_preds_degrades(tmp_path: Path) -> None:
    entry = _registry_entry("c" * 64)
    del entry["scorecard"]["cagr_1y"]
    del entry["scorecard"]["gain_to_pain_ratio"]
    del entry["scorecard"]["kelly_fraction"]
    _write_registry(tmp_path, [entry])
    data = _synthetic_data_dir(tmp_path)  # no validation_preds.parquet written
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    out = dash.reconcile_capital_metrics(frame, tmp_path, data)
    row = out.row(0, named=True)
    assert row["cagr_1y"] is None
    assert row["kelly_fraction"] is None


def test_reconcile_capital_metrics_missing_data_assets_noop(tmp_path: Path) -> None:
    entry = _registry_entry("d" * 64)
    del entry["scorecard"]["cagr_1y"]
    del entry["scorecard"]["gain_to_pain_ratio"]
    del entry["scorecard"]["kelly_fraction"]
    _write_registry(tmp_path, [entry])
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    out = dash.reconcile_capital_metrics(frame, tmp_path, tmp_path / "no-data")
    assert out.row(0, named=True)["cagr_1y"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k reconcile -v`
Expected: FAIL — `AttributeError: ... has no attribute 'reconcile_capital_metrics'`.

- [ ] **Step 3: Write minimal implementation**

Append to `nmr/dashboard.py` (add `import numpy as np` and the nmr imports at top):

```python
from nmr.evaluation import EvaluationEngine, downside_era_indices
from nmr.payout import (
    annual_compounded_return,
    gain_to_pain_ratio,
    kelly_fraction,
    payout_series,
)
from nmr.scorecard import MMC_DOWN_MIN_ERAS

_CAPITAL_SCALAR_CELLS = ("cagr_1y", "gain_to_pain_ratio", "kelly_fraction")


def _has_stored_capital_block(row: dict) -> bool:
    return all(row.get(c) is not None for c in _CAPITAL_SCALAR_CELLS)


def _load_shared_lookups(
    data_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]] | None:
    """Load the 86-era meta-overlap targets + meta lookups once.

    Returns ``(targets_86, meta, meta_eras)`` or None when either data asset
    is missing.
    """
    data = Path(data_dir)
    targets_path = data / "validation.parquet"
    meta_path = data / "meta_model.parquet"
    if not (targets_path.exists() and meta_path.exists()):
        return None
    targets = pl.read_parquet(targets_path, columns=["era", "id", "target"])
    meta = pl.read_parquet(
        meta_path, columns=["era", "id", "numerai_meta_model"]
    )
    meta_eras = sorted(meta.get_column("era").unique().to_list(), key=int)
    targets_86 = targets.filter(pl.col("era").is_in(meta_eras))
    return targets_86, meta, meta_eras


def _per_era_metrics(
    preds_path: Path,
    targets_86: pl.DataFrame,
    meta: pl.DataFrame,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Per-era CORR, MMC, and meta-CORR for one stored predictions file.

    Joins on [era, id] against the shared lookups — the meta inner join
    restricts to the standard 86-era overlap window.
    """
    preds = pl.read_parquet(preds_path, columns=["era", "id", "prediction"])
    joined = (
        preds.join(targets_86, on=["era", "id"], how="inner")
        .join(meta, on=["era", "id"], how="inner")
    )
    engine = EvaluationEngine()
    corr = engine.per_era_corr(joined, pred_col="prediction", target_col="target")
    mmc = engine.per_era_mmc(
        joined, pred_col="prediction",
        meta_col="numerai_meta_model", target_col="target",
    )
    meta_corr = engine.per_era_corr(
        joined, pred_col="numerai_meta_model", target_col="target"
    )
    return corr, mmc, meta_corr


def reconcile_capital_metrics(
    leaderboard: pl.DataFrame,
    registry_dir: Path,
    data_dir: Path,
) -> pl.DataFrame:
    """Fill missing capital cells by recomputing from stored parquets.

    Stored-first: rows whose scorecard carries all three scalar capital cells
    are trusted verbatim (including a stored ``mmc_down=None`` with reason).
    Everything else for trained/trained_legacy rows is recomputed via the
    oracle-parity evaluation/payout paths. Registry files are never written.
    """
    rows = leaderboard.to_dicts()
    needs_recompute = [
        row for row in rows
        if row["source"] in ("trained", "trained_legacy")
        and not _has_stored_capital_block(row)
    ]
    if not needs_recompute:
        return leaderboard

    lookups = _load_shared_lookups(data_dir)
    if lookups is None:
        logger.warning(
            "nmr.dashboard: v5.3 targets/meta_model missing at %s; "
            "capital cells left None", data_dir,
        )
        return leaderboard
    targets_86, meta, _ = lookups

    for row in rows:
        if not (
            row["source"] in ("trained", "trained_legacy")
            and not _has_stored_capital_block(row)
        ):
            continue
        preds_path = Path(row["run_dir"]) / "validation_preds.parquet"
        if not preds_path.exists():
            logger.warning(
                "nmr.dashboard: %s has no validation_preds.parquet; "
                "capital cells left None", row["model_id"],
            )
            continue
        corr, mmc, meta_corr = _per_era_metrics(preds_path, targets_86, meta)
        series = payout_series(corr, mmc)
        row["cagr_1y"] = annual_compounded_return(series.clipped)
        row["gain_to_pain_ratio"] = gain_to_pain_ratio(series.clipped)
        row["kelly_fraction"] = kelly_fraction(series.raw)
        downside = downside_era_indices(meta_corr)
        if len(downside) >= MMC_DOWN_MIN_ERAS:
            row["mmc_down"] = float(np.mean([mmc[e] for e in downside]))
            row["mmc_down_reason"] = None
        else:
            row["mmc_down"] = None
            row["mmc_down_reason"] = "insufficient_downside_eras"

    return pl.DataFrame(rows, schema=UNIFIED_SCHEMA, strict=False)
```

Implementer note: `nmr/scorecard.py` must first gain a public alias directly below its private constant: `MMC_DOWN_MIN_ERAS = _MMC_DOWN_MIN_ERAS  # public alias for cross-module use` (one line, no `__all__` change needed).

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k reconcile -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit** (skip if commits not authorized)

```bash
git add nmr/dashboard.py nmr/scorecard.py tests/test_dashboard.py
git commit -m "feat(dashboard): capital-cell recompute with stored-first fallback"
```

---

### Task 6: `extract_payout_timeseries`

**Files:**
- Modify: `nmr/dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `_load_shared_lookups`, `_per_era_metrics`, `payout_series`, `annual_compounded_return`, `sorted_era_labels`.
- Produces: `extract_payout_timeseries(registry_dir: Path, data_dir: Path, run_ids: Sequence[str], include_tier4_ref: bool = True, tier4_column: str = "v53_lgbm_ender60") -> dict` — shape per spec §4: `{"eras": [...], "meta_downside_mask": [...], "series": {model_id: {"label", "cumulative_wealth", "drawdown", "cagr", "mdd"}}}`. Deterministic: model ids sorted, eras numeric-sorted, arrays aligned to `eras`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py` (add `import hashlib, json` already present; `hashlib` new):

```python
def test_extract_payout_timeseries_shape_and_determinism(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64), _registry_entry("b" * 64)])
    for run_id, scale in (("a" * 64, 1.0), ("b" * 64, -0.5)):
        _write_preds(tmp_path / run_id, scale=scale)
    data = _synthetic_data_dir(tmp_path)

    payload = dash.extract_payout_timeseries(
        tmp_path, data, run_ids=["b" * 64, "a" * 64], include_tier4_ref=False
    )
    assert payload["eras"] == ["0001", "0002", "0003"]  # numeric order
    assert len(payload["meta_downside_mask"]) == 3
    assert set(payload["series"]) == {"a" * 64, "b" * 64}
    for series in payload["series"].values():
        assert len(series["cumulative_wealth"]) == 3
        assert len(series["drawdown"]) == 3
        assert series["mdd"] <= 0.0
        assert isinstance(series["cagr"], float)
        assert series["label"]

    # determinism: identical payload hash across repeated runs and insertion orders
    again = dash.extract_payout_timeseries(
        tmp_path, data, run_ids=["a" * 64, "b" * 64], include_tier4_ref=False
    )
    assert json.dumps(again, sort_keys=True) == json.dumps(payload, sort_keys=True)

    # perfect-correlation series: wealth compounds at +5% per era, drawdown == 0
    perfect = payload["series"]["a" * 64]
    assert perfect["cumulative_wealth"][-1] == pytest.approx(1.05**3, abs=1e-9)
    assert perfect["drawdown"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert perfect["mdd"] == pytest.approx(0.0, abs=1e-12)


def test_extract_payout_timeseries_missing_run_skipped(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64)])
    _write_preds(tmp_path / ("a" * 64), scale=1.0)
    data = _synthetic_data_dir(tmp_path)
    payload = dash.extract_payout_timeseries(
        tmp_path, data, run_ids=["a" * 64, "9" * 64], include_tier4_ref=False
    )
    assert set(payload["series"]) == {"a" * 64}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k extract_payout -v`
Expected: FAIL — `AttributeError: ... has no attribute 'extract_payout_timeseries'`.

- [ ] **Step 3: Write minimal implementation**

Append to `nmr/dashboard.py` (add `from collections.abc import Sequence` and `from nmr.evaluation import sorted_era_labels` to imports):

```python
def _series_label(registry_dir: Path, run_id: str) -> str:
    run_file = Path(registry_dir) / run_id / "run.json"
    name = "unknown"
    if run_file.exists():
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
        if isinstance(payload, dict):
            manifest = payload.get("manifest") or {}
            name = (manifest.get("config") or {}).get("run", {}).get("name", "unknown")
    return f"{name} · {run_id[:8]}"


def _series_from_metrics(
    corr: dict[str, float], mmc: dict[str, float], axis_eras: list[str], label: str
) -> dict:
    """Aligned wealth/drawdown arrays over ``axis_eras`` (fail-loud on gaps)."""
    missing = set(axis_eras) - set(corr) - set(mmc)
    if set(corr) != set(axis_eras) or set(mmc) != set(axis_eras) or missing:
        raise ValueError(
            f"series {label!r} does not cover the full era axis "
            f"(missing {sorted(missing, key=int)[:5]}...)"
        )
    # Order dicts by the axis explicitly: wealth compounding and drawdown
    # watermarks are sequence-sensitive (defense in depth — payout_series
    # sorts numerically, but this keeps the invariant local).
    ordered_corr = {era: float(corr[era]) for era in axis_eras}
    ordered_mmc = {era: float(mmc[era]) for era in axis_eras}
    pay = payout_series(ordered_corr, ordered_mmc)
    wealth = np.cumprod(1.0 + pay.clipped)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return {
        "label": label,
        "cumulative_wealth": [float(v) for v in wealth],
        "drawdown": [float(v) for v in drawdown],
        "cagr": float(annual_compounded_return(pay.clipped)),
        "mdd": float(np.min(drawdown)),
    }


def extract_payout_timeseries(
    registry_dir: Path,
    data_dir: Path,
    run_ids: Sequence[str],
    include_tier4_ref: bool = True,
    tier4_column: str = "v53_lgbm_ender60",
) -> dict[str, Any]:
    """Per-era payout/wealth/drawdown series over the standard 86-era window.

    Numeric era ordering throughout (``sorted_era_labels``); arrays aligned
    to the shared axis; deterministic key order (sorted run ids).
    """
    lookups = _load_shared_lookups(data_dir)
    if lookups is None:
        logger.warning(
            "nmr.dashboard: data assets missing at %s; "
            "returning empty timeseries", data_dir,
        )
        return {"eras": [], "meta_downside_mask": [], "series": {}}
    targets_86, meta, meta_eras = lookups
    axis = sorted_era_labels(meta_eras)

    # meta-model downside mask (strict CORR_meta < 0), computed once
    meta_only = meta.filter(pl.col("era").is_in(axis))
    meta_joined = meta_only.join(targets_86, on=["era", "id"], how="inner")
    meta_corr = EvaluationEngine().per_era_corr(
        meta_joined, pred_col="numerai_meta_model", target_col="target"
    )
    mask = [bool(meta_corr[e] < 0.0) for e in axis]

    series: dict[str, dict] = {}
    for run_id in sorted(set(run_ids)):
        preds_path = Path(registry_dir) / run_id / "validation_preds.parquet"
        if not preds_path.exists():
            logger.warning("nmr.dashboard: skipping missing preds %s", preds_path)
            continue
        corr, mmc, _ = _per_era_metrics(preds_path, targets_86, meta)
        series[run_id] = _series_from_metrics(
            corr, mmc, axis, _series_label(registry_dir, run_id)
        )

    if include_tier4_ref:
        bench_path = Path(data_dir) / "validation_benchmark_models.parquet"
        if bench_path.exists():
            bench = pl.read_parquet(
                bench_path, columns=["era", "id", tier4_column]
            ).filter(pl.col("era").is_in(axis))
            ref_joined = (
                bench.join(targets_86, on=["era", "id"], how="inner")
                .join(meta, on=["era", "id"], how="inner")
            )
            engine = EvaluationEngine()
            ref_corr = engine.per_era_corr(
                ref_joined, pred_col=tier4_column, target_col="target"
            )
            ref_mmc = engine.per_era_mmc(
                ref_joined, pred_col=tier4_column,
                meta_col="numerai_meta_model", target_col="target",
            )
            series[tier4_column] = _series_from_metrics(
                ref_corr, ref_mmc, axis, tier4_column
            )
        else:
            logger.warning("nmr.dashboard: %s missing; tier-4 curve omitted", bench_path)

    return {"eras": axis, "meta_downside_mask": mask, "series": series}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k extract_payout -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit** (skip if commits not authorized)

```bash
git add nmr/dashboard.py tests/test_dashboard.py
git commit -m "feat(dashboard): payout timeseries extraction with downside mask"
```

---

### Task 7: Real-data acceptance tests

**Files:**
- Modify: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6; real assets under `data/v5.3/` and `artifacts/registry/` (skip-marked, CI-safe).

- [ ] **Step 1: Write the tests** (skipif convention identical to `tests/test_parity.py:94-102`)

Append to `tests/test_dashboard.py`:

```python
_REAL_VALIDATION = Path("data/v5.3/validation.parquet")
_REAL_META = Path("data/v5.3/meta_model.parquet")
_REAL_BENCH = Path("data/v5.3/validation_benchmark_models.parquet")
_REAL_REGISTRY = Path("artifacts/registry")
_SMOKE_CSV = Path("artifacts/reports/benchmark_hierarchy_scorecard_smoke.csv")
_HAS_REAL = (
    _REAL_VALIDATION.exists()
    and _REAL_META.exists()
    and _REAL_BENCH.exists()
    and _REAL_REGISTRY.exists()
    and any(_REAL_REGISTRY.glob("*/run.json"))
)


@pytest.mark.skipif(not _HAS_REAL, reason="real registry/v5.3 data absent; skipped in CI")
def test_real_recompute_matches_stored_corr() -> None:
    frame = dash.load_unified_leaderboard(_REAL_REGISTRY, benchmark_path=False)
    row = frame.sort("corr_sharpe_ac", descending=True, nulls_last=True).row(0, named=True)
    lookups = dash._load_shared_lookups(Path("data/v5.3"))
    assert lookups is not None
    targets_86, meta, _ = lookups
    preds_path = Path(row["run_dir"]) / "validation_preds.parquet"
    corr, _, _ = dash._per_era_metrics(preds_path, targets_86, meta)
    assert len(corr) == row["corr_n_eras"]
    assert float(np.mean(list(corr.values()))) == pytest.approx(row["corr"], abs=1e-4)


@pytest.mark.skipif(not (_HAS_REAL and _SMOKE_CSV.exists()),
                    reason="real v5.3 data + smoke benchmark CSV absent; skipped in CI")
def test_real_tier4_cagr_matches_smoke_csv() -> None:
    lookups = dash._load_shared_lookups(Path("data/v5.3"))
    assert lookups is not None
    targets_86, meta, _ = lookups
    bench = pl.read_parquet(
        _REAL_BENCH, columns=["era", "id", "v53_lgbm_ender60"]
    )
    axis = sorted(
        meta.get_column("era").unique().to_list(), key=int
    )
    joined = (
        bench.filter(pl.col("era").is_in(axis))
        .join(targets_86, on=["era", "id"], how="inner")
        .join(meta, on=["era", "id"], how="inner")
    )
    engine = nmr_evaluation.EvaluationEngine()  # import nmr.evaluation as nmr_evaluation at top
    corr = engine.per_era_corr(joined, pred_col="v53_lgbm_ender60", target_col="target")
    mmc = engine.per_era_mmc(
        joined, pred_col="v53_lgbm_ender60",
        meta_col="numerai_meta_model", target_col="target",
    )
    pay = payout.payout_series(corr, mmc)  # import nmr.payout as payout at top
    recomputed = payout.annual_compounded_return(pay.clipped)
    stored = dash.load_benchmark_frame(_SMOKE_CSV).filter(
        pl.col("model_id") == "v53_lgbm_ender60"
    ).row(0, named=True)["cagr_1y"]
    assert float(recomputed) == pytest.approx(float(stored), abs=1e-6)


@pytest.mark.skipif(not _HAS_REAL, reason="real registry/v5.3 data absent; skipped in CI")
def test_real_reconcile_populates_all_capital_columns() -> None:
    frame = dash.load_unified_leaderboard(_REAL_REGISTRY, benchmark_path=False)
    out = dash.reconcile_capital_metrics(frame, _REAL_REGISTRY, Path("data/v5.3"))
    trained = out.filter(pl.col("source").is_in(["trained", "trained_legacy"]))
    assert trained.height > 0
    for row in trained.to_dicts():
        assert row["cagr_1y"] is not None
        assert row["gain_to_pain_ratio"] is not None
        assert row["kelly_fraction"] is not None
```

(Implementer note: at the top of the file add `import nmr.evaluation as nmr_evaluation` and `import nmr.payout as payout`.)

- [ ] **Step 2: Run the tests**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k real_ -v`
Expected: PASS on this machine (data present); SKIP in CI. If `test_real_recompute_matches_stored_corr` fails on tolerance, inspect the diff and widen `abs` only with evidence in the commit message.

- [ ] **Step 3: Commit** (skip if commits not authorized)

```bash
git add tests/test_dashboard.py
git commit -m "test(dashboard): real-data acceptance for recompute parity and tier-4 CAGR"
```

---

### Task 8: Package exports + docs hygiene

**Files:**
- Modify: `nmr/__init__.py`
- Modify: `AGENTS.md` (toolkit table — one row)
- Modify: `ARCHITECTURE.md` (short module entry)
- Test: `tests/test_dashboard.py` (export smoke test)

**Interfaces:**
- Produces: `from nmr import UNIFIED_SCHEMA, evaluate_gate_status, extract_payout_timeseries, load_benchmark_frame, load_unified_leaderboard, reconcile_capital_metrics, resolve_benchmark_path` (all also in `nmr.__all__`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py`:

```python
def test_dashboard_symbols_exported_from_package() -> None:
    import nmr

    for name in (
        "UNIFIED_SCHEMA",
        "evaluate_gate_status",
        "extract_payout_timeseries",
        "load_benchmark_frame",
        "load_unified_leaderboard",
        "reconcile_capital_metrics",
        "resolve_benchmark_path",
    ):
        assert getattr(nmr, name) is not None, f"nmr.{name} not exported"
        assert name in nmr.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k exported -v`
Expected: FAIL — `AttributeError`.

- [ ] **Step 3: Add exports to `nmr/__init__.py`**

Insert after the `.data import IngestionAgent` line (alphabetical position, following the file's sorted order):

```python
from .dashboard import (
    UNIFIED_SCHEMA,
    evaluate_gate_status,
    extract_payout_timeseries,
    load_benchmark_frame,
    load_unified_leaderboard,
    reconcile_capital_metrics,
    resolve_benchmark_path,
)
```

And insert into `__all__` (alphabetical, after `"CURRENT_DATA_VERSION",`):

```python
    "UNIFIED_SCHEMA",
```

and after `"deflated_sharpe_fleet",`:

```python
    "evaluate_gate_status",
    "extract_payout_timeseries",
```

and after `"load_benchmark_file",`:

```python
    "load_benchmark_frame",
```

and after `"load_benchmark_suite_config",`:

```python
    "load_unified_leaderboard",
```

and after `"reconcile"`... there is no `reconcile` entry — add `"reconcile_capital_metrics",` alphabetically after `"promotion_verdict",` (actual correct slot: `reconcile_` < `regime_`), and `"resolve_benchmark_path",` after `"resolve_bandwidth",`.

- [ ] **Step 4: Run tests**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k exported tests/test_package_api.py -v`
Expected: PASS (export smoke + existing package API contract tests stay green).

- [ ] **Step 5: Update `AGENTS.md` toolkit table**

Add one row to the §6 table (after the benchmark row):

```markdown
| Change the executive dashboard data engine (leaderboard, gate projection, capital recompute, payout timeseries) | `nmr/dashboard.py` + `dashboard_charts.py` + `generate_dashboard.py` (spec: `docs/superpowers/specs/2026-08-16-executive-dashboard-design.md`) |
```

- [ ] **Step 6: Update `ARCHITECTURE.md`**

Add a short entry to the module registry section (match the surrounding style): `nmr/dashboard.py` — executive report engine: unified leaderboard schema (superset of the Streamlit leaderboard schema), tier-4 gate projection from `configs/benchmarks/tier4_gate.yaml`, stored-first capital recompute over the 86-era meta-overlap window, payout timeseries with downside mask; plotly/streamlit-free. Plus one line for `dashboard_charts.py` (top-level control plane, plotly figures only).

- [ ] **Step 7: Commit** (skip if commits not authorized)

```bash
git add nmr/__init__.py AGENTS.md ARCHITECTURE.md tests/test_dashboard.py
git commit -m "feat(dashboard): export engine API and update docs"
```

---

### Task 9: `dashboard_charts.py` — plotly figure builders

**Files:**
- Create: `dashboard_charts.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: none from nmr (pure presentation; inputs are plain frames/dicts).
- Produces:
  - `build_leaderboard_bar_chart(df: pl.DataFrame, *, hurdle_sharpe: float) -> go.Figure` — requires columns `label, corr_sharpe_ac, corr_sharpe_ac_ci_low, corr_sharpe_ac_ci_high, champion` (bool). Ascending y-order (best on top), asymmetric error bars, dashed red hurdle line.
  - `build_cumulative_wealth_chart(payload: dict) -> go.Figure` — `payload["eras"]`, `payload["meta_downside_mask"]`, `payload["series"]` (each `{"label", "cumulative_wealth"}`); `vrect` spans over consecutive `True` mask runs.
  - `build_drawdown_chart(payload: dict) -> go.Figure` — underwater traces with `fill="tozeroy"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py` (add `import dashboard_charts as charts` at top):

```python
def _bar_input() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"label": "run-a · aaaaaaaa", "corr_sharpe_ac": 0.8,
             "corr_sharpe_ac_ci_low": 0.6, "corr_sharpe_ac_ci_high": 1.0,
             "champion": True},
            {"label": "bench · bbbbbbbb", "corr_sharpe_ac": 0.5,
             "corr_sharpe_ac_ci_low": None, "corr_sharpe_ac_ci_high": None,
             "champion": False},
        ]
    )


def _ts_payload() -> dict:
    return {
        "eras": ["0001", "0002", "0003", "0004"],
        "meta_downside_mask": [True, True, False, True],
        "series": {
            "a": {"label": "run-a", "cumulative_wealth": [1.0, 1.05, 1.10, 1.08],
                  "drawdown": [0.0, 0.0, 0.0, -0.018]},
        },
    }


def test_leaderboard_chart_traces_and_hurdle_line() -> None:
    fig = charts.build_leaderboard_bar_chart(_bar_input(), hurdle_sharpe=0.78)
    assert len(fig.data) == 2
    # first trace is the last row (ascending order -> best on top)
    assert fig.data[0].y[0] == "bench · bbbbbbbb"
    # hurdle line is a layout shape
    shapes = [s for s in fig.layout.shapes if s.type == "line"]
    assert any(abs(s.x0 - 0.78) < 1e-9 for s in shapes)


def test_wealth_chart_downside_vrects() -> None:
    fig = charts.build_cumulative_wealth_chart(_ts_payload())
    assert len(fig.data) == 1
    assert fig.data[0].y[-1] == 1.08
    vrects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(vrects) == 2          # [0001..0002] and [0004..0004]
    assert vrects[0].x0 == "0001" and vrects[0].x1 == "0002"
    assert vrects[1].x0 == "0004" and vrects[1].x1 == "0004"


def test_drawdown_chart_underwater_fill() -> None:
    fig = charts.build_drawdown_chart(_ts_payload())
    assert len(fig.data) == 1
    assert fig.data[0].fill == "tozeroy"
    assert fig.data[0].y[-1] == pytest.approx(-0.018)


def test_timeseries_charts_empty_payload_render_annotation() -> None:
    empty = {"eras": [], "meta_downside_mask": [], "series": {}}
    for builder in (charts.build_cumulative_wealth_chart, charts.build_drawdown_chart):
        fig = builder(empty)
        assert len(fig.data) == 0
        assert fig.layout.annotations
        assert "unavailable" in fig.layout.annotations[0].text.lower()


def test_leaderboard_chart_empty_frame_render_annotation() -> None:
    fig = charts.build_leaderboard_bar_chart(
        pl.DataFrame(schema={"label": pl.String, "corr_sharpe_ac": pl.Float64,
                             "corr_sharpe_ac_ci_low": pl.Float64,
                             "corr_sharpe_ac_ci_high": pl.Float64,
                             "champion": pl.Boolean}),
        hurdle_sharpe=0.78,
    )
    assert len(fig.data) == 0
    assert fig.layout.annotations
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k chart -v`
Expected: FAIL — `ModuleNotFoundError: dashboard_charts`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard_charts.py
"""Plotly figure builders for the executive dashboard (presentation only).

Thin presentation layer: consumes clean frames/dicts from ``nmr.dashboard``
and returns configured ``plotly.graph_objects.Figure`` instances. No metric
math, no file I/O, no registry access.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import polars as pl

_HURDLE_COLOR = "#f85149"
_HURDLE_ANNOTATION = "tier-4 hurdle"
_DOWNSIDE_FILL = "rgba(248, 81, 73, 0.10)"


def build_leaderboard_bar_chart(
    df: pl.DataFrame, *, hurdle_sharpe: float
) -> go.Figure:
    """Horizontal Sharpe bars, best on top, with asymmetric CIs + hurdle line."""
    fig = go.Figure()
    if df.height == 0:
        fig.add_annotation(text="No models recorded yet", showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig
    for row in df.sort("corr_sharpe_ac", descending=False, nulls_last=True).to_dicts():
        value = row["corr_sharpe_ac"]
        error_x = None
        if value is not None and row["corr_sharpe_ac_ci_low"] is not None \
                and row["corr_sharpe_ac_ci_high"] is not None:
            error_x = {
                "type": "data",
                "symmetric": False,
                "array": [row["corr_sharpe_ac_ci_high"] - value],
                "arrayminus": [value - row["corr_sharpe_ac_ci_low"]],
            }
        fig.add_trace(
            go.Bar(
                name=row["label"],
                x=[value],
                y=[row["label"]],
                orientation="h",
                error_x=error_x,
                marker_pattern_shape="/" if row["champion"] else "",
                hovertemplate=(
                    f"{row['label']}<br>Sharpe (AC): %{{x:.3f}}<extra></extra>"
                ),
            )
        )
    fig.add_vline(
        x=hurdle_sharpe,
        line_dash="dash",
        line_color=_HURDLE_COLOR,
        annotation_text=f"{_HURDLE_ANNOTATION} {hurdle_sharpe:.2f}",
    )
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="CORR Sharpe (autocorrelation-adjusted)",
    )
    return fig


def _downside_spans(eras: list[str], mask: list[bool]) -> list[tuple[str, str]]:
    spans: list[tuple[str, str]] = []
    start: int | None = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        if not flag and start is not None:
            spans.append((eras[start], eras[index - 1]))
            start = None
    if start is not None:
        spans.append((eras[start], eras[-1]))
    return spans


def build_cumulative_wealth_chart(payload: dict[str, Any]) -> go.Figure:
    """Cumulative wealth curves with shaded meta-model drawdown eras."""
    eras = payload["eras"]
    fig = go.Figure()
    if not payload.get("eras"):
        fig.add_annotation(
            text="Timeseries data unavailable without local v5.3 assets",
            showarrow=False,
        )
        fig.update_layout(template="plotly_dark")
        return fig
    for series in payload["series"].values():
        fig.add_trace(
            go.Scatter(
                name=series["label"],
                x=eras,
                y=series["cumulative_wealth"],
                mode="lines",
                hovertemplate="%{y:.4f}<extra>" + series["label"] + "</extra>",
            )
        )
    for x0, x1 in _downside_spans(eras, payload["meta_downside_mask"]):
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=_DOWNSIDE_FILL,
            line_width=0, layer="below",
        )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Era",
        yaxis_title="Cumulative wealth (1.0 stake)",
        legend=dict(orientation="h"),
    )
    return fig


def build_drawdown_chart(payload: dict[str, Any]) -> go.Figure:
    """Underwater drawdown curves, filled red to zero."""
    eras = payload["eras"]
    fig = go.Figure()
    if not payload.get("eras"):
        fig.add_annotation(
            text="Timeseries data unavailable without local v5.3 assets",
            showarrow=False,
        )
        fig.update_layout(template="plotly_dark")
        return fig
    for series in payload["series"].values():
        fig.add_trace(
            go.Scatter(
                name=series["label"],
                x=eras,
                y=series["drawdown"],
                mode="lines",
                fill="tozeroy",
                fillcolor=_DOWNSIDE_FILL,
                hovertemplate="%{y:.2%}<extra>" + series["label"] + "</extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Era",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h"),
    )
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k chart -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit** (skip if commits not authorized)

```bash
git add dashboard_charts.py tests/test_dashboard.py
git commit -m "feat(dashboard): plotly chart layer (leaderboard, wealth, drawdown)"
```

---

### Task 10: `generate_dashboard.py` rewrite

**Files:**
- Modify: `generate_dashboard.py` (full rewrite)
- Modify: `tests/test_scripts.py` (remove the two obsolete `_build_html`/`_rank_models` tests)
- Test: `tests/test_dashboard.py` (HTML contract tests)

**Interfaces:**
- Consumes: `nmr.dashboard.{DEFAULT_DATA_DIR, DEFAULT_GATE_PATH, DEFAULT_REGISTRY_DIR, evaluate_gate_status, extract_payout_timeseries, load_unified_leaderboard, reconcile_capital_metrics}`, `nmr.benchmark.load_benchmark_file` (for the hurdle), `dashboard_charts`, `plotly.offline.get_plotlyjs`, `plotly.io.to_html`.
- Produces: `generate_dashboard(*, registry_dir: Path | None = None, benchmark_path: Path | None | bool = None, output_path: Path | None = None, open_browser: bool = True) -> Path`; internal `_build_html(leaderboard, champion, kpis, figures, registry_dir, technical_entries) -> str` (the gate hurdle flows in via `kpis["hurdle_sharpe"]`, not a separate parameter); `main() -> int`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard.py`:

```python
import generate_dashboard


def _charts_for_test() -> dict:
    bar = charts.build_leaderboard_bar_chart(_bar_input(), hurdle_sharpe=0.78)
    return {"leaderboard": bar, "wealth": charts.build_cumulative_wealth_chart(_ts_payload()),
            "drawdown": charts.build_drawdown_chart(_ts_payload())}


def _kpis_for_test() -> dict:
    return {
        "champion_label": "None Designated", "champion_detail": "(Unallocated)",
        "top_contender_label": "sample-run · aaaaaaaa",
        "top_contender_sharpe": 0.8, "hurdle_sharpe": 0.78,
        "gap": 0.02, "fleet_best_cagr": 1.5, "worst_drawdown": -0.05,
        "capital_ready_count": 0, "fleet_count": 1, "data_version": "v5.3",
        "n_eras": 86,
    }


def test_html_escapes_user_strings_and_single_plotly_engine(tmp_path: Path) -> None:
    rows = pl.DataFrame(
        [{"model_id": "<script>alert(1)</script>", "source": "trained",
          "run_name": '"><img src=x onerror=alert(2)>', "corr_sharpe_ac": 0.8,
          "corr_sharpe_ac_ci_low": 0.6, "corr_sharpe_ac_ci_high": 1.0,
          "cagr_1y": 1.5, "gain_to_pain_ratio": 2.0, "kelly_fraction": 0.4,
          "mmc_down": 0.01, "deflated_sharpe": 0.97, "max_drawdown": 0.1,
          "fnc": 0.05, "corr": 0.12, "status": "RESEARCH", "tier": None,
          "gate_corr_sharpe_ac": False, "gate_cagr_1y": None}]
    )
    html_text = generate_dashboard._build_html(
        leaderboard=rows, champion=None, kpis=_kpis_for_test(),
        figures=_charts_for_test(),
        registry_dir=tmp_path,
        technical_entries=[],
    )
    assert '"><img src=x' not in html_text            # hostile run_name escaped, never raw
    assert "&lt;img src=x onerror=alert(2)&gt;" in html_text
    # engine embed marker (the bundle itself contains many "window.Plotly"
    # literals, so count the template's own marker, not bundle internals)
    assert html_text.count("<!-- plotly-engine-embed -->") == 1
    assert "<script src" not in html_text            # zero external script tags (offline)
    assert html_text.count("Plotly.newPlot(") == 3   # three figures, no engine per figure
    assert 'class="num gate-fail"' in html_text   # failing gate cell tinted
    assert "badge research" in html_text          # status badge pill rendered
```

```python
def test_generate_dashboard_end_to_end_synthetic(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64)])
    out = generate_dashboard.generate_dashboard(
        registry_dir=tmp_path, benchmark_path=False,
        output_path=tmp_path / "dashboard.html", open_browser=False,
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "sample-run" in text
    # the synthetic fixture genuinely clears the real tier-4 gate (corr 0.12,
    # sharpe 0.8, fnc 0.05, dsr 0.97, gtp 2.0, cagr 1.5) -> CAPITAL READY badge
    assert "CAPITAL READY" in text
    assert "<!-- plotly-engine-embed -->" in text
    # size is unbounded by ruling (full plotly engine inline, ~4.9 MB)
```

(Implementer note: `generate_dashboard()` uses the repo defaults for data/gate paths; with a synthetic tmp registry and real repo data present, capital recompute may run — both paths are valid, the assertions hold either way.)

- [ ] **Step 2: Update `tests/test_scripts.py`**

Delete `test_dashboard_escapes_html_interpolation` and `test_dashboard_ranks_trained_and_benchmark_on_same_sharpe` (superseded by the tests above). Replace them with:

```python
def test_generate_dashboard_import_surface() -> None:
    assert callable(generate_dashboard.generate_dashboard)
    assert callable(generate_dashboard.main)
```

Keep the module-level `import generate_dashboard` line.

- [ ] **Step 3: Run tests to verify they fail**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k "html or end_to_end" tests/test_scripts.py -k "dashboard" -v`
Expected: FAIL — `generate_dashboard._build_html` has the old signature / missing symbols.

- [ ] **Step 4: Write the new `generate_dashboard.py`**

```python
"""Compile the executive HTML performance report from the shared engine.

Thin control plane only: data comes from ``nmr.dashboard``, figures from
``dashboard_charts``, HTML from the template below. No metric math here.
"""

from __future__ import annotations

import html
import json
import webbrowser
from pathlib import Path

import plotly.io as pio
import polars as pl
from plotly.offline import get_plotlyjs

import dashboard_charts as charts
from nmr.benchmark import load_benchmark_file
from nmr.config import REPO_ROOT
from nmr.dashboard import (
    DEFAULT_DATA_DIR,
    DEFAULT_GATE_PATH,
    DEFAULT_REGISTRY_DIR,
    evaluate_gate_status,
    extract_payout_timeseries,
    load_unified_leaderboard,
    reconcile_capital_metrics,
)

_EXEC_COLUMNS = ("cagr_1y", "corr_sharpe_ac", "corr_sharpe_ac_ci_low",
                 "corr_sharpe_ac_ci_high", "max_drawdown", "gain_to_pain_ratio",
                 "mmc_down", "deflated_sharpe")


def _fmt(value, *, pct: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:  # NaN
        return "—"
    if number == float("inf"):
        return "∞"
    if pct:
        return f"{number:.2%}"
    return f"{number:.4f}"


def _bar_label(row: dict) -> str:
    model_id = row["model_id"] or "?"
    if row["source"] == "benchmark":
        return f"{row['run_name']} · {model_id}"
    return f"{row['run_name']} · {model_id[:8]}"


def _bar_input(leaderboard: pl.DataFrame, champion: str | None) -> pl.DataFrame:
    top = leaderboard.sort("corr_sharpe_ac", descending=True, nulls_last=True).head(10)
    return pl.DataFrame(
        [
            {
                "label": _bar_label(row),
                "corr_sharpe_ac": row["corr_sharpe_ac"],
                "corr_sharpe_ac_ci_low": row["corr_sharpe_ac_ci_low"],
                "corr_sharpe_ac_ci_high": row["corr_sharpe_ac_ci_high"],
                "champion": row["model_id"] == champion,
            }
            for row in top.to_dicts()
        ]
    )


def _champion_id(registry_dir: Path) -> str | None:
    champion_path = registry_dir / "champion.json"
    if not champion_path.exists():
        return None
    try:
        payload = json.loads(champion_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    return run_id if isinstance(run_id, str) else None


def _kpi_cards(leaderboard: pl.DataFrame, champion: str | None,
               hurdle_sharpe: float) -> dict:
    fleet = leaderboard.filter(pl.col("source").is_in(["trained", "trained_legacy"]))
    top = fleet.sort("corr_sharpe_ac", descending=True, nulls_last=True).head(1)
    top_row = top.row(0, named=True) if top.height else None
    cagr_values = [
        row["cagr_1y"] for row in fleet.to_dicts()
        if row["cagr_1y"] is not None
    ]
    return {
        "champion_label": "None Designated" if champion is None
                          else _bar_label(leaderboard.filter(
                              pl.col("model_id") == champion).row(0, named=True)),
        "champion_detail": "(Unallocated)" if champion is None else "Active",
        "top_contender_label": _bar_label(top_row) if top_row else "—",
        "top_contender_sharpe": top_row["corr_sharpe_ac"] if top_row else None,
        "hurdle_sharpe": hurdle_sharpe,
        "gap": (top_row["corr_sharpe_ac"] - hurdle_sharpe)
               if top_row and top_row["corr_sharpe_ac"] is not None else None,
        "fleet_best_cagr": max(cagr_values) if cagr_values else None,
        "worst_drawdown": min(
            [row["max_drawdown"] for row in fleet.to_dicts()
             if row["max_drawdown"] is not None],
            default=None,
        ),
        "capital_ready_count": fleet.join(
            leaderboard.select(["model_id", "status"]), on="model_id", how="left"
        ).filter(pl.col("status") == "CAPITAL READY").height,
        "fleet_count": fleet.height,
        "data_version": "v5.3",
        "n_eras": leaderboard.get_column("n_eras").drop_nulls().max()
                  if leaderboard.height else None,
    }


def _table_rows(leaderboard: pl.DataFrame, champion: str | None) -> list[dict]:
    rows = leaderboard.to_dicts()
    champion_rows = [r for r in rows if champion is not None and r["model_id"] == champion]
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
    return champion_rows + fleet_rows + bench_rows


_STATUS_BADGE = {
    "CHAMPION": "champion",
    "CAPITAL READY": "ready",
    "RESEARCH": "research",
    "GATE HURDLE": "hurdle",
    "BENCHMARK": "benchmark",
}


def _status_badge(status: str) -> str:
    cls = _STATUS_BADGE.get(status, "research")
    return f'<span class="badge {cls}">{html.escape(status)}</span>'


def _td_gate(value_str: str, gate_pass: bool | None) -> str:
    if gate_pass is False:
        return f'<td class="num gate-fail">{value_str}</td>'
    return f'<td class="num">{value_str}</td>'


def _row_html(row: dict) -> str:
    status = _status_badge(row.get("status", "RESEARCH"))
    sharpe = _fmt(row.get("corr_sharpe_ac"))
    ci = "—"
    if row.get("corr_sharpe_ac_ci_low") is not None and row.get("corr_sharpe_ac_ci_high") is not None:
        ci = f"[{_fmt(row['corr_sharpe_ac_ci_low'])}–{_fmt(row['corr_sharpe_ac_ci_high'])}]"
    return (
        "<tr>"
        f"<td>{status}</td>"
        f"<td>{html.escape(_bar_label(row))}</td>"
        f"{_td_gate(_fmt(row.get('cagr_1y'), pct=True), row.get('gate_cagr_1y'))}"
        f"{_td_gate(sharpe, row.get('gate_corr_sharpe_ac'))}"
        f"<td class=\"num\">{ci}</td>"
        f"<td class=\"num\">{_fmt(row.get('max_drawdown'), pct=True)}</td>"
        f"{_td_gate(_fmt(row.get('gain_to_pain_ratio')), row.get('gate_gain_to_pain_ratio'))}"
        f"<td class=\"num\">{_fmt(row.get('mmc_down'))}</td>"
        f"{_td_gate(_fmt(row.get('deflated_sharpe')), row.get('gate_deflated_sharpe'))}"
        "</tr>"
    )


def _technical_entries(registry_dir: Path) -> list[dict]:
    entries = []
    for run_file in sorted(registry_dir.glob("*/run.json")):
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        manifest = payload.get("manifest") or {}
        cfg = manifest.get("config") or {}
        run_cfg = cfg.get("run") or {}
        entries.append(
            {
                "label": f"{run_cfg.get('name', 'unknown')} · "
                         f"{str(payload.get('run_id') or run_file.parent.name)[:8]}",
                "summary": {
                    "backend": (cfg.get("model") or {}).get("backend"),
                    "preset": (cfg.get("model") or {}).get("preset"),
                    "feature_set": (cfg.get("data") or {}).get("feature_set"),
                    "feature_subset": (cfg.get("data") or {}).get("feature_subset"),
                    "neutralization_proportion": (cfg.get("risk") or {}).get(
                        "neutralization_proportion"
                    ),
                    "seed": run_cfg.get("seed"),
                    "device": manifest.get("oof_device"),
                    "targets": (cfg.get("data") or {}).get("targets"),
                },
                "json_text": json.dumps(payload, indent=2, sort_keys=True),
            }
        )
    return entries


def _build_html(leaderboard: pl.DataFrame, champion: str | None, kpis: dict,
                figures: dict, registry_dir: Path,
                technical_entries: list[dict]) -> str:
    """Assemble the full HTML document (single plotly engine in <head>)."""
    engine_js = get_plotlyjs()
    figure_html = {
        name: pio.to_html(fig, include_plotlyjs=False, full_html=False)
        for name, fig in figures.items()
    }
    rows_html = "".join(_row_html(row) for row in _table_rows(leaderboard, champion))
    accordion = ""
    for entry in technical_entries:
        accordion += (
            "<details><summary>"
            f"{html.escape(entry['label'])} — technical &amp; audit</summary>"
            f"<pre>{html.escape(entry['json_text'])}</pre></details>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NumerAI Executive Performance Report</title>
<style>
  body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, 'Segoe UI', sans-serif; margin: 0; padding: 1.5rem; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0 2rem; }}
  .kpi {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem; }}
  .kpi .label {{ font-size: 0.8rem; color: #8b949e; text-transform: uppercase; }}
  .kpi .value {{ font-size: 1.4rem; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; background: #161b22; border: 1px solid #30363d; }}
  th, td {{ padding: 0.6rem 0.8rem; border-bottom: 1px solid #30363d; text-align: left; }}
  th {{ background: #21262d; font-size: 0.8rem; text-transform: uppercase; }}
  .num {{ font-variant-numeric: tabular-nums; text-align: right; }}
  .gate-fail {{ color: #f85149; font-weight: 500; }}
  .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }}
  .badge.champion {{ background: rgba(137, 87, 229, 0.2); color: #a371f7; border: 1px solid #8957e5; }}
  .badge.ready {{ background: rgba(46, 160, 67, 0.2); color: #3fb950; border: 1px solid #2ea043; }}
  .badge.research {{ background: rgba(110, 118, 129, 0.2); color: #8b949e; border: 1px solid #30363d; }}
  .badge.hurdle {{ background: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid #da3633; }}
  .badge.benchmark {{ background: rgba(137, 87, 229, 0.12); color: #a371f7; border: 1px solid #30363d; }}
  details {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 0.5rem 1rem; margin: 0.5rem 0; }}
  summary {{ cursor: pointer; }}
  pre {{ white-space: pre-wrap; font-size: 0.75rem; }}
  h1, h2 {{ color: #e6edf3; }}
</style>
<!-- plotly-engine-embed -->
<script>{engine_js}</script>
</head>
<body>
<h1>🏆 NumerAI Executive Performance Report</h1>
<p>Evaluation window: {kpis['n_eras']} overlap eras · data version {kpis['data_version']}</p>
<div class="kpis">
  <div class="kpi"><div class="label">Active Champion</div><div class="value">{html.escape(kpis['champion_label'])}</div><div>{html.escape(kpis['champion_detail'])}</div></div>
  <div class="kpi"><div class="label">Top Research Contender</div><div class="value">{html.escape(kpis['top_contender_label'])}</div><div>Sharpe {_fmt(kpis['top_contender_sharpe'])} vs hurdle {_fmt(kpis['hurdle_sharpe'])}</div></div>
  <div class="kpi"><div class="label">Fleet Best Return (CAGR)</div><div class="value">{_fmt(kpis['fleet_best_cagr'], pct=True)}</div></div>
  <div class="kpi"><div class="label">Worst Fleet Drawdown</div><div class="value">{_fmt(kpis['worst_drawdown'], pct=True)}</div></div>
  <div class="kpi"><div class="label">Capital Readiness</div><div class="value">{kpis['capital_ready_count']} / {kpis['fleet_count']}</div></div>
</div>
<h2>1. Cumulative Wealth &amp; Downside Protection</h2>
{figure_html['wealth']}
<h2>2. Risk-Adjusted Return Leaderboard</h2>
{figure_html['leaderboard']}
<h2>3. Executive Allocation &amp; Risk Decision Table</h2>
<table>
<thead><tr><th>Status</th><th>Model</th><th>Ann. Return</th><th>Sharpe (AC)</th><th>Sharpe CI</th><th>Max DD</th><th>Gain-to-Pain</th><th>Downside</th><th>Confidence (DSR)</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
<h2>4. Underwater Drawdown</h2>
{figure_html['drawdown']}
<h2>Technical &amp; Audit Metadata</h2>
{accordion}
</body>
</html>"""


def generate_dashboard(
    *,
    registry_dir: Path | None = None,
    benchmark_path: Path | None | bool = None,
    output_path: Path | None = None,
    open_browser: bool = True,
) -> Path:
    """Build the executive HTML report and write it to disk."""
    registry_dir = Path(registry_dir) if registry_dir is not None else DEFAULT_REGISTRY_DIR
    output_path = Path(output_path) if output_path is not None else REPO_ROOT / "artifacts" / "dashboard.html"

    leaderboard = load_unified_leaderboard(registry_dir, benchmark_path=benchmark_path)
    leaderboard = reconcile_capital_metrics(leaderboard, registry_dir, DEFAULT_DATA_DIR)
    statuses = evaluate_gate_status(leaderboard, DEFAULT_GATE_PATH, registry_dir / "champion.json")
    leaderboard = leaderboard.join(statuses, on="model_id", how="left")

    gate_cfg = load_benchmark_file(DEFAULT_GATE_PATH)
    hurdle_sharpe = float(gate_cfg.gate.corr_sharpe_ac_min)

    champion = _champion_id(registry_dir)
    fleet = leaderboard.filter(pl.col("source").is_in(["trained", "trained_legacy"]))
    top_ids = fleet.sort("corr_sharpe_ac", descending=True, nulls_last=True).head(3)
    timeseries = extract_payout_timeseries(
        registry_dir, DEFAULT_DATA_DIR,
        run_ids=top_ids.get_column("model_id").to_list(),
        include_tier4_ref=True,
    )
    figures = {
        "leaderboard": charts.build_leaderboard_bar_chart(
            _bar_input(leaderboard, champion), hurdle_sharpe=hurdle_sharpe
        ),
        "wealth": charts.build_cumulative_wealth_chart(timeseries),
        "drawdown": charts.build_drawdown_chart(timeseries),
    }
    html_text = _build_html(
        leaderboard=leaderboard, champion=champion,
        kpis=_kpi_cards(leaderboard, champion, hurdle_sharpe),
        figures=figures, registry_dir=registry_dir,
        technical_entries=_technical_entries(registry_dir),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    if open_browser:
        webbrowser.open(output_path.as_uri())
    return output_path


def main() -> int:
    output = generate_dashboard()
    print(f"Dashboard written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

IMPORTANT implementer note: `figures` must be the dict `{"leaderboard": ..., "wealth": ..., "drawdown": ...}` (as built in `generate_dashboard()`), so the named interpolation above renders exactly three `Plotly.newPlot` blocks and one engine injection.

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k "html or end_to_end" tests/test_scripts.py -v`
Expected: PASS (new HTML tests + remaining script contract tests).

- [ ] **Step 6: Commit** (skip if commits not authorized)

```bash
git add generate_dashboard.py tests/test_dashboard.py tests/test_scripts.py
git commit -m "feat(dashboard): executive HTML report compiler with single plotly embed"
```

---

### Task 11: `dashboard_app.py` rewiring

**Files:**
- Modify: `dashboard_app.py`

**Interfaces:**
- Consumes: `nmr.dashboard.{load_unified_leaderboard, load_benchmark_frame}`.
- Produces: unchanged public surface — `load_registry_frame`, `load_benchmarks`, `merge_leaderboard`, `load_campaigns`, `robustness_matrix`, `champion_run_id`, `_shaped_leaderboard_pdf`, `_LEADERBOARD_SCHEMA`, `_CAMPAIGN_SCHEMA`, and the five render views. All existing `tests/test_scripts.py` tests must pass unmodified.

- [ ] **Step 1: Rewire the two loaders**

In `dashboard_app.py`, replace the bodies of `load_registry_frame` and `load_benchmarks` with thin delegations (keep their docstrings, updating the "Mirrors generate_dashboard" wording to "Delegates to nmr.dashboard"):

```python
from nmr.dashboard import load_benchmark_frame, load_unified_leaderboard


def load_registry_frame(registry_dir: Path) -> pl.DataFrame:
    """Load all registry runs into a leaderboard frame (engine delegation).

    Projects the engine's unified frame down to ``_LEADERBOARD_SCHEMA`` for
    the Streamlit views; parsing and None-discipline live in
    ``nmr.dashboard.load_unified_leaderboard``.
    """
    frame = load_unified_leaderboard(registry_dir, benchmark_path=False)
    trained = frame.filter(pl.col("source").is_in(["trained", "trained_legacy"]))
    if trained.height == 0:
        return _EMPTY_LEADERBOARD
    return trained.select(_LEADERBOARD_SCHEMA.names())


def load_benchmarks(path: Path) -> pl.DataFrame:
    """Normalize the benchmark CSV to the leaderboard schema (engine delegation)."""
    frame = load_benchmark_frame(path)
    if frame.height == 0:
        return _EMPTY_LEADERBOARD
    return frame.select(_LEADERBOARD_SCHEMA.names())
```

Delete the now-dead `_load_registry_entries`-duplication: keep `_load_registry_entries` (it is used by `render_fleet`), but the old row-building loops inside `load_registry_frame`/`load_benchmarks` are removed. The `_ROBUSTNESS_CELLS`, `_LEADERBOARD_SCHEMA`, `_CAMPAIGN_SCHEMA`, `_EMPTY_*` constants stay.

- [ ] **Step 2: Run the existing contract tests unmodified**

Run: `./.venv/Scripts/python -m pytest tests/test_scripts.py -v`
Expected: PASS — all existing dashboard_app tests green with zero test edits. If any fail, the delegation projection (column names/types) is wrong; fix `_LEADERBOARD_SCHEMA` projection in the wrapper, not the tests.

- [ ] **Step 3: Lint check**

Run: `./.venv/Scripts/python -m ruff check dashboard_app.py nmr/dashboard.py dashboard_charts.py generate_dashboard.py`
Expected: clean (fix unused imports removed by the rewire — e.g. `json` may still be needed by `_read_run_payload`).

- [ ] **Step 4: Commit** (skip if commits not authorized)

```bash
git add dashboard_app.py
git commit -m "refactor(dashboard): dashboard_app delegates to nmr.dashboard engine"
```

---

### Task 12: Final verification gate

**Files:**
- None (verification only)

- [ ] **Step 1: Fast gate**

Run: `./.venv/Scripts/python -m ruff check .`
Expected: clean.

- [ ] **Step 2: Full functional gate**

Run: `./.venv/Scripts/python -m pytest -q`
Expected: full suite green (717+ new tests; real-data tests either pass or SKIP — report which).

- [ ] **Step 3: Real report generation**

Run: `./.venv/Scripts/python generate_dashboard.py`
Expected: exit 0; `artifacts/dashboard.html` written.

- [ ] **Step 4: Artifact inspection**

Run:

```bash
./.venv/Scripts/python -c "
from pathlib import Path
import polars as pl
from nmr.dashboard import load_unified_leaderboard, reconcile_capital_metrics, DEFAULT_DATA_DIR
p = Path('artifacts/dashboard.html')
text = p.read_text(encoding='utf-8')
# size unbounded by ruling (full plotly engine inline, ~4.9 MB)
assert "<script src" not in text, 'external script tag found (must be offline)'
assert text.count('<!-- plotly-engine-embed -->') == 1, 'plotly engine must be embedded exactly once'
frame = load_unified_leaderboard(Path('artifacts/registry'), benchmark_path=False)
out = reconcile_capital_metrics(frame, Path('artifacts/registry'), DEFAULT_DATA_DIR)
missing = [r['model_id'][:8] for r in out.to_dicts() if r['cagr_1y'] is None or r['gain_to_pain_ratio'] is None or r['kelly_fraction'] is None]
assert not missing, f'capital cells missing for: {missing}'
print('dashboard.html OK:', p.stat().st_size, 'bytes,', out.height, 'runs, capital cells complete')
"
```

Expected: prints the OK line; every registry run has non-null `cagr_1y` / `gain_to_pain_ratio` / `kelly_fraction`.

- [ ] **Step 5: Commit** (skip if commits not authorized)

```bash
git add artifacts/dashboard.html  # only if the repo tracks this generated artifact
git commit -m "chore(dashboard): regenerate executive report"
```

If `artifacts/dashboard.html` is git-ignored, skip this step and note it in the final report.

---

## Self-Review Notes (completed by plan author)

- Spec coverage: every spec section maps to tasks — §3 topology → Tasks 1–3, 9; §4 contracts → Tasks 1–6 (incl. sentinel, degradation, benchmark exemption, numeric ordering); §5 charts → Task 9; §6 HTML layout/KPIs/accordion → Task 10; §7 verification → Tasks 7, 12 (plus per-task tests); decisions #11–17 are each realized in the task where they apply; §8 exclusions respected (no gate-engine module, no regime surfaces, no registry writes).
- Placeholder scan: none remain — the Task 10 figures-dict note states a real constraint (the `figures` dict keys), not deferred work. All test code is fully written with correct imports.
- Type consistency: `UNIFIED_SCHEMA` column names are used identically in Tasks 2–6, 10, 11; `evaluate_gate_status` receipt names (`gate_<field>`) match its test; `extract_payout_timeseries` payload keys (`eras`, `meta_downside_mask`, `series.<id>.{label,cumulative_wealth,drawdown,cagr,mdd}`) match the chart builders and their tests; `dashboard_app` projection relies only on schema names listed in Task 1's test.
