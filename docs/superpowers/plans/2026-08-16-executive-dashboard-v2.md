# Executive Dashboard v2 — Multi-Metric Trajectory & Signal Diversification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve the shipped executive report (`artifacts/dashboard.html`) with a 7-metric interactive trajectory chart (embedded vanilla-JS controller) and a pairwise signal-similarity matrix with diversification badge and ensemble-Sharpe card.

**Architecture:** Pure engine functions in `nmr/dashboard.py` (new v2 lookups, `extract_multimetric_timeseries`, `extract_pairwise_similarity_matrix`) feed a JS-controller chart block plus two static plotly figures in `dashboard_charts.py`; `generate_dashboard.py` compiles the v2 layout. `dashboard_app.py` untouched. All v1 invariants hold (offline, deterministic, registry read-only, `nmr/` UI-free).

**Tech Stack:** Python 3.12, Polars, numpy, plotly 6.x (figures + inline engine), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-16-executive-dashboard-v2-design.md` (read first — the Approved Decisions Log is binding; decisions #18–#27 are the round-2 rulings).

## Global Constraints

- Run tests as `./.venv/Scripts/python -m pytest <args>`; lint as `./.venv/Scripts/python -m ruff check .` (E/F/I/UP, line-length 120). Never use the `Scripts/pip` shim.
- Business logic in `nmr/` only; `nmr/` must never import plotly/streamlit (the JS controller is a plain Python string — no import involved). Registry strictly read-only.
- Determinism: no wall-clock anywhere in output; era lists numeric-sorted via `nmr.evaluation.sorted_era_labels`; `json.dumps(..., sort_keys=True)` for the embedded payload; stable model-id ordering (`sorted`).
- Degradation: missing data assets → empty payloads + warnings, never raise; missing horizon target column → zeros + warning for that metric slice; report generation never aborts.
- Payout parity (decision #19): `payout` is anchored to `main_target="target"` — the same anchor as `reconcile_capital_metrics`.
- `tests/test_package_api.py` enforces that every public name in an `nmr` module's `__all__` is re-exported from `nmr/__init__.py` (import block + package `__all__`, alphabetical) **in the same commit** that adds/removes it. `tests/test_docs_hygiene.py` enforces that the "N tests" claims in `AGENTS.md` and `CONTRIBUTING.md` equal `pytest --collect-only` — recompute and bump both in the same commit as every test-count change (README.md has no claims).
- Full suite before every commit: `./.venv/Scripts/python -m pytest -q` must be green (0 failed; skips allowed only for data-absent tests).
- Commit steps require the user's explicit go-ahead (repo rule). If commits are not authorized, skip the commit step and continue.
- Baseline: suite currently at 752 collected. Recomputed per task — never assume.

---

## File Structure

- Modify: `nmr/dashboard.py` — add `_V2Lookups`, `_resolve_horizon_targets`, `_load_v2_lookups`, `extract_multimetric_timeseries`, `extract_pairwise_similarity_matrix`; delete `extract_payout_timeseries` and `_series_from_metrics`; keep `_series_label`, `_load_shared_lookups`, `_per_era_metrics` (v1 tests depend on them).
- Modify: `dashboard_charts.py` — add `multimetric_chart_html`; add `build_similarity_matrix_chart`; adapt `build_drawdown_chart`; delete `build_cumulative_wealth_chart`.
- Modify: `generate_dashboard.py` — v2 wiring: `_diversification_stats`, `_ensemble_sharpe`, `_badge_html`, `_ensemble_card_html`, `_build_html` v2 template, `generate_dashboard()` v2 flow.
- Modify: `tests/test_dashboard.py` — migrate the v1 payout/drawdown tests; add v2 units + real-data acceptance.
- Modify: `nmr/__init__.py` — export `extract_multimetric_timeseries`, `extract_pairwise_similarity_matrix`; remove `extract_payout_timeseries`.
- Modify: `docs/superpowers/specs/2026-08-16-executive-dashboard-v2-design.md` — signature-line sync (similarity returns a 4-tuple).
- Modify: `ARCHITECTURE.md` — §W updated to the v2 surface.
- Modify: `AGENTS.md`, `CONTRIBUTING.md` — test-count claims per task.

---

### Task 1: Engine — target resolution + v2 lookups

**Files:**
- Modify: `nmr/dashboard.py` (append after the existing lookups helpers)
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `nmr.evaluation.sorted_era_labels`; `dataclasses` import (add `import dataclasses` if absent — the file currently imports json/logging/pathlib/numpy/polars/nmr.*; add `from dataclasses import dataclass`).
- Produces (consumed by Tasks 2–3): `_resolve_horizon_targets(schema_cols: Sequence[str]) -> tuple[str, str]`; `_V2Lookups` frozen dataclass with fields `targets: pl.DataFrame, target_20_col: str, target_60_col: str, meta: pl.DataFrame, meta_eras: list[str], benchmarks: pl.DataFrame`; `_load_v2_lookups(data_dir: Path, tier4_column: str) -> _V2Lookups | None` (None when validation.parquet or meta_model.parquet missing).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py` (uses the existing `_synthetic_data_dir` helper from Task 5 of the v1 plan; add a benchmark parquet to it):

```python
def _synthetic_v2_data_dir(tmp_path: Path, *, with_benchmark: bool = True) -> Path:
    data = _synthetic_data_dir(tmp_path)  # era/id/target + meta over 0001..0003
    rows = []
    for era in ("0001", "0002", "0003"):
        for i in range(10):
            rows.append({"era": era, "id": f"{era}_{i:03d}",
                         "v53_lgbm_ender60": 0.5 * float(i)})
    pl.DataFrame(rows).write_parquet(data / "validation_benchmark_models.parquet")
    return data


def test_resolve_horizon_targets_fallback_chain() -> None:
    assert dash._resolve_horizon_targets(["era", "target_ender_20", "target_ender_60"]) == \
        ("target_ender_20", "target_ender_60")
    assert dash._resolve_horizon_targets(["target_cyrusd_20", "target_cyrusd_60"]) == \
        ("target_cyrusd_20", "target_cyrusd_60")
    # both horizons collapse to the generic target when nothing else exists
    assert dash._resolve_horizon_targets(["target"]) == ("target", "target")


def test_load_v2_lookups_deduped_target_columns(tmp_path: Path) -> None:
    data = _synthetic_v2_data_dir(tmp_path)
    lookups = dash._load_v2_lookups(data, tier4_column="v53_lgbm_ender60")
    assert lookups is not None
    assert lookups.meta_eras == ["0001", "0002", "0003"]
    # both horizons resolve to "target" in the synthetic fixture — the read
    # must still succeed (deduped column list, decision #18)
    assert lookups.target_20_col == "target"
    assert lookups.target_60_col == "target"
    assert lookups.targets.columns == ["era", "id", "target"]
    assert lookups.benchmarks.columns == ["era", "id", "v53_lgbm_ender60"]
    assert lookups.benchmarks.height == 30


def test_load_v2_lookups_missing_assets_returns_none(tmp_path: Path) -> None:
    assert dash._load_v2_lookups(tmp_path / "no-data", tier4_column="v53_lgbm_ender60") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k "resolve_horizon or v2_lookups" -v`
Expected: FAIL — `AttributeError: module 'nmr.dashboard' has no attribute '_resolve_horizon_targets'`.

- [ ] **Step 3: Write minimal implementation**

Append to `nmr/dashboard.py` (add `from dataclasses import dataclass` to the imports):

```python
@dataclass(frozen=True)
class _V2Lookups:
    targets: pl.DataFrame
    target_20_col: str
    target_60_col: str
    meta: pl.DataFrame
    meta_eras: list[str]
    benchmarks: pl.DataFrame


def _resolve_horizon_targets(schema_cols: Sequence[str]) -> tuple[str, str]:
    """Resolve the 20D/60D target columns with fallback chains (decision #10)."""
    target_20 = next(
        (c for c in ("target_ender_20", "target_cyrusd_20", "target_20", "target")
         if c in schema_cols),
        "target",
    )
    target_60 = next(
        (c for c in ("target_ender_60", "target_cyrusd_60", "target_60", "target")
         if c in schema_cols),
        target_20,
    )
    return target_20, target_60


def _load_v2_lookups(data_dir: Path, tier4_column: str) -> _V2Lookups | None:
    """Single-pass v2 lookups: deduped targets, meta, benchmarks on meta eras.

    Returns None when validation.parquet or meta_model.parquet is missing;
    benchmarks are optional (empty frame when the file is absent).
    """
    data = Path(data_dir)
    targets_path = data / "validation.parquet"
    meta_path = data / "meta_model.parquet"
    bench_path = data / "validation_benchmark_models.parquet"
    if not (targets_path.exists() and meta_path.exists()):
        return None
    schema_cols = pl.read_parquet_schema(targets_path).names()
    target_20, target_60 = _resolve_horizon_targets(schema_cols)
    # decision #18: deduped — both horizons may resolve to "target"
    target_cols = list(dict.fromkeys(["era", "id", "target", target_20, target_60]))
    targets = pl.read_parquet(targets_path, columns=target_cols)
    meta = pl.read_parquet(meta_path, columns=["era", "id", "numerai_meta_model"])
    meta_eras = sorted_era_labels(meta.get_column("era").unique().to_list())
    targets_86 = targets.filter(pl.col("era").is_in(meta_eras))
    benchmarks = pl.DataFrame(
        schema={"era": pl.String, "id": pl.String, tier4_column: pl.Float64}
    )
    if bench_path.exists():
        benchmarks = pl.read_parquet(
            bench_path, columns=["era", "id", tier4_column]
        ).filter(pl.col("era").is_in(meta_eras))
    return _V2Lookups(
        targets=targets_86,
        target_20_col=target_20,
        target_60_col=target_60,
        meta=meta,
        meta_eras=meta_eras,
        benchmarks=benchmarks,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k "resolve_horizon or v2_lookups" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Full suite + docs count + commit**

Run `./.venv/Scripts/python -m pytest --collect-only -q` → expect 755 (752 + 3); bump every "N tests" claim in `AGENTS.md` and `CONTRIBUTING.md` to the actual number; run the FULL suite (0 failed) and `./.venv/Scripts/python -m ruff check nmr/dashboard.py tests/test_dashboard.py`; commit:

```bash
git add nmr/dashboard.py tests/test_dashboard.py AGENTS.md CONTRIBUTING.md
git commit -m "feat(dashboard): v2 horizon-target resolution and deduped lookups"
```

---

### Task 2: Engine — `extract_multimetric_timeseries` (replaces payout timeseries)

**Files:**
- Modify: `nmr/dashboard.py` (delete `extract_payout_timeseries` + `_series_from_metrics`; add the new function; update `__all__` — remove `"extract_payout_timeseries"`, add `"extract_multimetric_timeseries"`)
- Test: `tests/test_dashboard.py` (delete the v1 `test_extract_payout_timeseries_*` tests — they are superseded)

**Interfaces:**
- Consumes: `_load_v2_lookups`, `_series_label`, `EvaluationEngine`, `payout_series`, `np`.
- Produces: `extract_multimetric_timeseries(registry_dir: Path, data_dir: Path, run_ids: Sequence[str], include_tier4_ref: bool = True, tier4_column: str = "v53_lgbm_ender60") -> dict[str, Any]` — payload exactly per spec §4 (keys `eras`, `meta_downside_mask`, `metrics` with the 7 names, `drawdowns`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py`:

```python
def test_multimetric_payload_shape_and_semantics(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64)])
    _write_preds(tmp_path / ("a" * 64), scale=1.0)
    data = _synthetic_v2_data_dir(tmp_path)

    payload = dash.extract_multimetric_timeseries(
        tmp_path, data, run_ids=["a" * 64], include_tier4_ref=True
    )
    assert payload["eras"] == ["0001", "0002", "0003"]
    assert len(payload["meta_downside_mask"]) == 3
    assert set(payload["metrics"]) == {
        "payout", "corr20", "mmc20", "corr60", "mmc60", "bmc", "cwmm"
    }
    assert set(payload["drawdowns"]) >= {"a" * 64, "v53_lgbm_ender60"}
    for name in payload["metrics"]:
        for model_id in ("a" * 64, "v53_lgbm_ender60"):
            series = payload["metrics"][name][model_id]
            assert len(series["standard"]) == 3
            assert len(series["cumulative"]) == 3
            assert series["label"]
    # payout: perfect corr with target and meta -> r_t = 0.05 clipped every era
    payout = payload["metrics"]["payout"]["a" * 64]
    assert payout["standard"] == pytest.approx([0.05, 0.05, 0.05], abs=1e-9)
    assert payout["cumulative"] == pytest.approx([1.05, 1.05**2, 1.05**3], abs=1e-9)
    # correlation-family cumulative = cumsum (aligned 1:1, no origin point)
    cwmm = payload["metrics"]["cwmm"]["a" * 64]
    assert cwmm["cumulative"][-1] == pytest.approx(sum(cwmm["standard"]), abs=1e-9)
    # BMC short-circuit: the reference measured against itself is all zeros
    bmc_ref = payload["metrics"]["bmc"]["v53_lgbm_ender60"]
    assert bmc_ref["standard"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    # drawdown aligned with payout wealth
    wealth = payout["cumulative"]
    peak = max(wealth[:1])
    assert payload["drawdowns"]["a" * 64][0] == pytest.approx(wealth[0] / peak - 1.0, abs=1e-12)


def test_multimetric_payout_parity_with_reconcile(tmp_path: Path) -> None:
    entry = _registry_entry("b" * 64)
    del entry["scorecard"]["cagr_1y"]
    del entry["scorecard"]["gain_to_pain_ratio"]
    del entry["scorecard"]["kelly_fraction"]
    _write_registry(tmp_path, [entry])
    _write_preds(tmp_path / ("b" * 64), scale=1.0)
    data = _synthetic_v2_data_dir(tmp_path)
    payload = dash.extract_multimetric_timeseries(
        tmp_path, data, run_ids=["b" * 64], include_tier4_ref=False
    )
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    reconciled = dash.reconcile_capital_metrics(frame, data)
    row = reconciled.row(0, named=True)
    # chart payout compounded must equal the table's cagr_1y compounding
    # (both anchored to main_target="target" — decision #19)
    import nmr.payout as payout_mod

    standard = payload["metrics"]["payout"]["b" * 64]["standard"]
    from nmr.payout import annual_compounded_return
    assert annual_compounded_return(standard) == pytest.approx(row["cagr_1y"], abs=1e-6)
    assert row["cagr_1y"] is not None


def test_multimetric_missing_data_assets_empty_payload(tmp_path: Path) -> None:
    payload = dash.extract_multimetric_timeseries(
        tmp_path, tmp_path / "no-data", run_ids=["a" * 64], include_tier4_ref=False
    )
    assert payload == {"eras": [], "meta_downside_mask": [], "metrics": {}, "drawdowns": {}}


def test_multimetric_determinism_and_missing_run_skip(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("c" * 64)])
    _write_preds(tmp_path / ("c" * 64), scale=-0.5)
    data = _synthetic_v2_data_dir(tmp_path)
    a = dash.extract_multimetric_timeseries(
        tmp_path, data, run_ids=["c" * 64, "9" * 64], include_tier4_ref=False
    )
    b = dash.extract_multimetric_timeseries(
        tmp_path, data, run_ids=["9" * 64, "c" * 64], include_tier4_ref=False
    )
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert set(a["metrics"]["payout"]) == {"c" * 64}
```

(Implementer note: `import nmr.payout as payout_mod` in the parity test is unused — do not include it; keep only `from nmr.payout import annual_compounded_return` at the top of the file with the other nmr imports.)

Delete from the file: `test_extract_payout_timeseries_shape_and_determinism`, `test_extract_payout_timeseries_missing_run_skipped` (superseded).

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k multimetric -v`
Expected: FAIL — `AttributeError: ... has no attribute 'extract_multimetric_timeseries'`.

- [ ] **Step 3: Write minimal implementation**

Append to `nmr/dashboard.py` (delete `extract_payout_timeseries` and `_series_from_metrics`; keep `_series_label`):

```python
_METRIC_NAMES = ("payout", "corr20", "mmc20", "corr60", "mmc60", "bmc", "cwmm")


def _cumulative_from_standard(standard: list[float], *, payout: bool) -> list[float]:
    values = np.asarray(standard, dtype=float)
    if payout:
        return [float(v) for v in np.cumprod(1.0 + values)]
    return [float(v) for v in np.cumsum(values)]


def extract_multimetric_timeseries(
    registry_dir: Path,
    data_dir: Path,
    run_ids: Sequence[str],
    include_tier4_ref: bool = True,
    tier4_column: str = "v53_lgbm_ender60",
) -> dict[str, Any]:
    """7-metric per-era trajectories over the standardized meta window.

    Payout is anchored to main_target="target" (decision #19); correlation
    metrics use cumsum, payout uses cumprod (decision #9); the tier-4 BMC is
    short-circuited to zeros (decision #11); missing horizon targets zero
    their slice with a warning (decision #23). Never raises on missing
    assets — returns the empty payload.
    """
    lookups = _load_v2_lookups(data_dir, tier4_column)
    if lookups is None:
        logger.warning(
            "nmr.dashboard: data assets missing at %s; empty timeseries", data_dir
        )
        return {"eras": [], "meta_downside_mask": [], "metrics": {}, "drawdowns": {}}

    axis = lookups.meta_eras
    engine = EvaluationEngine()
    meta_joined = lookups.meta.join(lookups.targets, on=["era", "id"], how="inner")
    meta_corr = engine.per_era_corr(
        meta_joined, pred_col="numerai_meta_model", target_col="target"
    )
    mask = [bool(meta_corr[era] < 0.0) for era in axis]

    metrics: dict[str, dict] = {name: {} for name in _METRIC_NAMES}
    drawdowns: dict[str, list[float]] = {}

    ids = [mid for mid in sorted(set(run_ids)) if mid != tier4_column]
    if include_tier4_ref and lookups.benchmarks.height > 0:
        ids.append(tier4_column)

    for model_id in ids:
        if model_id == tier4_column:
            preds = lookups.benchmarks.select(
                ["era", "id", pl.col(tier4_column).alias("prediction")]
            )
            label = tier4_column
        else:
            preds_path = Path(registry_dir) / model_id / "validation_preds.parquet"
            if not preds_path.exists():
                logger.warning(
                    "nmr.dashboard: skipping missing preds %s", preds_path
                )
                continue
            preds = pl.read_parquet(preds_path, columns=["era", "id", "prediction"])
            label = _series_label(registry_dir, model_id)

        joined = (
            preds.join(lookups.targets, on=["era", "id"], how="inner")
            .join(lookups.meta, on=["era", "id"], how="inner")
        )
        corr_t = engine.per_era_corr(joined, pred_col="prediction", target_col="target")
        mmc_t = engine.per_era_mmc(
            joined, pred_col="prediction",
            meta_col="numerai_meta_model", target_col="target",
        )
        pay = payout_series(corr_t, mmc_t)
        standard = [float(meta_corr.get(era, 0.0)) for era in axis]
        standard = [float(pay.clipped[i]) for i in range(len(axis))]
        metrics["payout"][model_id] = {
            "standard": standard,
            "cumulative": _cumulative_from_standard(standard, payout=True),
            "label": label,
        }
        wealth = np.asarray(metrics["payout"][model_id]["cumulative"], dtype=float)
        peak = np.maximum.accumulate(wealth)
        drawdowns[model_id] = [float(v) for v in wealth / peak - 1.0]

        horizon_metrics = (
            ("corr20", lookups.target_20_col, "corr"),
            ("corr60", lookups.target_60_col, "corr"),
            ("mmc20", lookups.target_20_col, "mmc"),
            ("mmc60", lookups.target_60_col, "mmc"),
        )
        for name, target_col, kind in horizon_metrics:
            if target_col not in joined.columns:
                logger.warning(
                    "nmr.dashboard: horizon target %s missing; %s zeroed",
                    target_col, name,
                )
                zeros = [0.0 for _ in axis]
                metrics[name][model_id] = {
                    "standard": zeros, "cumulative": zeros, "label": label,
                }
                continue
            if kind == "corr":
                per = engine.per_era_corr(
                    joined, pred_col="prediction", target_col=target_col
                )
            else:
                per = engine.per_era_mmc(
                    joined, pred_col="prediction",
                    meta_col="numerai_meta_model", target_col=target_col,
                )
            aligned = [float(per.get(era, 0.0)) for era in axis]
            metrics[name][model_id] = {
                "standard": aligned,
                "cumulative": _cumulative_from_standard(aligned, payout=False),
                "label": label,
            }

        if model_id == tier4_column:
            zeros = [0.0 for _ in axis]
            metrics["bmc"][model_id] = {
                "standard": zeros, "cumulative": zeros, "label": label,
            }
        else:
            joined_b = joined.join(lookups.benchmarks, on=["era", "id"], how="inner")
            per_bmc = engine.per_era_bmc(
                joined_b, pred_col="prediction",
                benchmark_col=tier4_column, target_col="target",
            )
            aligned = [float(per_bmc.get(era, 0.0)) for era in axis]
            metrics["bmc"][model_id] = {
                "standard": aligned,
                "cumulative": _cumulative_from_standard(aligned, payout=False),
                "label": label,
            }

        per_cwmm = engine.per_era_cwmm(
            joined, pred_col="prediction", meta_col="numerai_meta_model"
        )
        aligned = [float(per_cwmm.get(era, 0.0)) for era in axis]
        metrics["cwmm"][model_id] = {
            "standard": aligned,
            "cumulative": _cumulative_from_standard(aligned, payout=False),
            "label": label,
        }

    return {
        "eras": axis,
        "meta_downside_mask": mask,
        "metrics": metrics,
        "drawdowns": drawdowns,
    }
```

IMPORTANT implementer note: the two stray lines `standard = [float(meta_corr.get(era, 0.0)) for era in axis]` followed by `standard = [float(pay.clipped[i]) ...]` contain a leftover — write ONLY the second line (the payout standard is `pay.clipped`): `standard = [float(v) for v in pay.clipped]`. Do not ship the stray meta_corr line.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k multimetric -v`
Expected: PASS (4 tests). If the cwmm identity assertion (`cumulative[-1] == sum(standard)`) fails due to float ordering, replace with `pytest.approx(..., abs=1e-9)`.

- [ ] **Step 5: Full suite + exports + docs count + commit**

`nmr/dashboard.py.__all__`: remove `"extract_payout_timeseries"`, add `"extract_multimetric_timeseries"`. `nmr/__init__.py` same commit: remove `extract_payout_timeseries` from the `.dashboard` import block and package `__all__`; add `extract_multimetric_timeseries` (import block alphabetical after `evaluate_gate_status`; package `__all__` same slot). Update the export test in `tests/test_dashboard.py` (`test_dashboard_symbols_exported_from_package`): drop `extract_payout_timeseries`, add `extract_multimetric_timeseries`. Recompute the collected count (deleting 2 old tests, adding 4 new → net +2 → expect 757) and bump both docs claims. Full suite + ruff; commit:

```bash
git add nmr/dashboard.py nmr/__init__.py tests/test_dashboard.py AGENTS.md CONTRIBUTING.md
git commit -m "feat(dashboard): 7-metric multimetric timeseries replaces payout timeseries"
```

---

### Task 3: Engine — `extract_pairwise_similarity_matrix`

**Files:**
- Modify: `nmr/dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `_load_v2_lookups`, `_series_label`, `Ensembler.rank_normalize`, `np`.
- Produces: `extract_pairwise_similarity_matrix(registry_dir, data_dir, run_ids, include_tier4_ref=True, tier4_column="v53_lgbm_ender60") -> tuple[list[str], list[str], list[list[float]], dict[str, Any]]` — `(labels, run_ids, matrix, stress_stats)` where `stress_stats = {"mean_delta": float | None, "n_pairs": int}` (4-tuple — plan-level refinement of the spec's 3-tuple; the spec line is updated in Task 7).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py`:

```python
def test_similarity_matrix_identity_symmetry_and_clamp(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64), _registry_entry("b" * 64)])
    _write_preds(tmp_path / ("a" * 64), scale=1.0)
    _write_preds(tmp_path / ("b" * 64), scale=2.0)  # scale-shifted copy of a
    data = _synthetic_v2_data_dir(tmp_path)
    labels, ids, matrix, stress = dash.extract_pairwise_similarity_matrix(
        tmp_path, data, run_ids=["b" * 64, "a" * 64], include_tier4_ref=False
    )
    assert ids == ["a" * 64, "b" * 64]          # sorted deterministically
    assert matrix[0][0] == 1.0 and matrix[1][1] == 1.0
    assert matrix[0][1] == pytest.approx(1.0, abs=1e-9)   # rank-gaussian: scale-invariant
    assert matrix[0][1] == pytest.approx(matrix[1][0], abs=1e-12)  # symmetric
    assert all(-1.0 <= v <= 1.0 for row in matrix for v in row)    # clamped
    assert set(stress) == {"mean_delta", "n_pairs"}


def test_similarity_matrix_includes_tier4_from_benchmark_parquet(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("c" * 64)])
    _write_preds(tmp_path / ("c" * 64), scale=1.0)
    data = _synthetic_v2_data_dir(tmp_path)
    labels, ids, matrix, _ = dash.extract_pairwise_similarity_matrix(
        tmp_path, data, run_ids=["c" * 64], include_tier4_ref=True
    )
    # no registry dir exists for the benchmark model — it comes from the parquet
    assert ids == ["c" * 64, "v53_lgbm_ender60"]
    assert matrix[1][1] == 1.0


def test_similarity_matrix_degenerate_constant_predictions(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("d" * 64)])
    rows = [
        {"era": era, "id": f"{era}_{i:03d}", "prediction": 1.0}
        for era in ("0001", "0002", "0003")
        for i in range(10)
    ]
    pl.DataFrame(rows).write_parquet(tmp_path / ("d" * 64) / "validation_preds.parquet")
    data = _synthetic_v2_data_dir(tmp_path)
    labels, ids, matrix, _ = dash.extract_pairwise_similarity_matrix(
        tmp_path, data, run_ids=["d" * 64], include_tier4_ref=False
    )
    assert matrix[0][0] == 1.0
    assert not any(v != v for row in matrix for v in row)  # no NaN


def test_similarity_matrix_missing_data_assets(tmp_path: Path) -> None:
    out = dash.extract_pairwise_similarity_matrix(
        tmp_path, tmp_path / "no-data", run_ids=["a" * 64], include_tier4_ref=False
    )
    assert out == ([], [], [], {"mean_delta": None, "n_pairs": 0})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k similarity -v`
Expected: FAIL — `AttributeError: ... has no attribute 'extract_pairwise_similarity_matrix'`.

- [ ] **Step 3: Write minimal implementation**

Append to `nmr/dashboard.py` (add `from nmr.ensemble import Ensembler` to the nmr import block):

```python
def extract_pairwise_similarity_matrix(
    registry_dir: Path,
    data_dir: Path,
    run_ids: Sequence[str],
    include_tier4_ref: bool = True,
    tier4_column: str = "v53_lgbm_ender60",
) -> tuple[list[str], list[str], list[list[float]], dict[str, Any]]:
    """Pairwise rank-gaussian pooled-Pearson similarity over the meta window.

    Single multi-way inner join across all candidates (decision #12), global
    intersection (not pairwise-complete); the tier-4 candidate is read from
    the benchmark parquet, never a registry dir (decision #20); degenerate
    columns guarded (decision #15); matrix clamped to [-1, 1] (decision #22).
    Returns (labels, run_ids, matrix, stress_stats) with stress_stats =
    {"mean_delta": mean off-diagonal (rho_stress - rho_normal) | None,
     "n_pairs": int} (decision #26).
    """
    lookups = _load_v2_lookups(data_dir, tier4_column)
    if lookups is None:
        logger.warning(
            "nmr.dashboard: data assets missing at %s; empty similarity", data_dir
        )
        return [], [], [], {"mean_delta": None, "n_pairs": 0}

    axis = lookups.meta_eras
    frames: list[pl.DataFrame] = []
    ids_used: list[str] = []
    labels: list[str] = []
    for model_id in sorted(set(run_ids)):
        if model_id == tier4_column:
            continue
        preds_path = Path(registry_dir) / model_id / "validation_preds.parquet"
        if not preds_path.exists():
            logger.warning("nmr.dashboard: skipping missing preds %s", preds_path)
            continue
        frames.append(
            pl.read_parquet(preds_path, columns=["era", "id", "prediction"]).rename(
                {"prediction": model_id}
            )
        )
        ids_used.append(model_id)
        labels.append(_series_label(registry_dir, model_id))
    if include_tier4_ref and lookups.benchmarks.height > 0:
        frames.append(lookups.benchmarks)
        ids_used.append(tier4_column)
        labels.append(tier4_column)
    if not frames:
        return [], [], [], {"mean_delta": None, "n_pairs": 0}

    aligned = frames[0]
    for frame in frames[1:]:
        aligned = aligned.join(frame, on=["era", "id"], how="inner")
    gauss = Ensembler.rank_normalize(aligned, pred_cols=ids_used, era_col="era")

    columns: list[np.ndarray] = []
    for model_id in ids_used:
        arr = gauss.get_column(model_id).to_numpy(dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if np.std(arr) <= 0.0:
            arr = np.zeros_like(arr)
        columns.append(arr)
    stacked = np.vstack(columns)
    matrix = np.clip(np.corrcoef(stacked), -1.0, 1.0)

    meta_corr = EvaluationEngine().per_era_corr(
        lookups.meta.join(lookups.targets, on=["era", "id"], how="inner"),
        pred_col="numerai_meta_model", target_col="target",
    )
    stress_eras = {era for era in axis if meta_corr.get(era, 0.0) < 0.0}
    era_arr = gauss.get_column("era").to_list()
    stress_idx = np.asarray([era in stress_eras for era in era_arr])

    def _mean_offdiag(mat: np.ndarray) -> float | None:
        if mat.shape[0] < 2:
            return None
        upper = [mat[i, j] for i in range(mat.shape[0]) for j in range(i + 1, mat.shape[0])]
        return float(np.mean(upper)) if upper else None

    mean_delta = None
    if stress_idx.sum() >= 5:
        rho_stress = _mean_offdiag(
            np.clip(np.corrcoef(stacked[:, stress_idx]), -1.0, 1.0)
        )
        rho_normal = _mean_offdiag(
            np.clip(np.corrcoef(stacked[:, ~stress_idx]), -1.0, 1.0)
        )
        if rho_stress is not None and rho_normal is not None:
            mean_delta = rho_stress - rho_normal

    n_pairs = len(ids_used) * (len(ids_used) - 1) // 2
    return labels, ids_used, matrix.tolist(), {"mean_delta": mean_delta, "n_pairs": n_pairs}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k similarity -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Full suite + exports + docs count + commit**

`nmr/dashboard.py.__all__`: add `"extract_pairwise_similarity_matrix"` (alphabetical after `extract_multimetric_timeseries`). `nmr/__init__.py` same commit: import block + package `__all__` (same slot). Update the export test. Recompute collected count (4 new tests → expect 761) and bump both docs claims. Full suite + ruff; commit:

```bash
git add nmr/dashboard.py nmr/__init__.py tests/test_dashboard.py AGENTS.md CONTRIBUTING.md
git commit -m "feat(dashboard): pairwise similarity matrix with stress-regime delta"
```

---

### Task 4: Charts — `multimetric_chart_html` (JS controller)

**Files:**
- Modify: `dashboard_charts.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: none from nmr (payload is a plain dict; the module already imports `json`, `html`, `plotly.graph_objects`, `polars`).
- Produces: `multimetric_chart_html(payload: dict[str, Any]) -> str` — the full `<div id="multimetric-chart">` + `<script>` block (raw HTML string, NOT a plotly.py figure; embedded by `generate_dashboard` directly).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py` (add `import dashboard_charts as charts` — already imported from v1):

```python
def test_multimetric_chart_html_embeds_payload_and_controls() -> None:
    payload = {
        "eras": ["0001", "0002"],
        "meta_downside_mask": [True, False],
        "metrics": {"payout": {"a": {"standard": [0.01, 0.02],
                                      "cumulative": [1.01, 1.0302], "label": "run · aaaaaaaa"}},
                    "corr20": {}, "mmc20": {}, "corr60": {}, "mmc60": {}, "bmc": {}, "cwmm": {}},
        "drawdowns": {"a": [0.0, 0.0]},
    }
    block = charts.multimetric_chart_html(payload)
    assert 'id="multimetric-chart"' in block
    assert json.loads(__import__("json").loads.__self__.loads(  # placeholder — see note
        block.split("var payload = ")[1].split(";")[0]
    )) == payload  # exact sorted-key serialization round-trips
    assert block.count("<option") == 7
    assert "Cumulative View" in block and "Standard View" in block
    assert "METRIC_CONFIG" in block
    assert "Cumulative Wealth (1.0 Stake)" in block and "Per-Era Net Return" in block
    assert "updatemenus" not in block
    assert "<script src" not in block


def test_multimetric_chart_html_empty_payload_annotation() -> None:
    block = charts.multimetric_chart_html(
        {"eras": [], "meta_downside_mask": [], "metrics": {}, "drawdowns": {}}
    )
    assert "Timeseries data unavailable without local v5.3 assets" in block
    assert "Plotly" not in block  # no chart is even mounted
```

(Implementer note: the first test's JSON round-trip line is deliberately awkward in this plan — write it plainly:
```python
    import json as _json
    embedded = _json.loads(block.split("var payload = ")[1].split(";")[0])
    assert embedded == payload
```
)

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k multimetric_chart -v`
Expected: FAIL — `AttributeError: module 'dashboard_charts' has no attribute 'multimetric_chart_html'`.

- [ ] **Step 3: Write minimal implementation**

Append to `dashboard_charts.py` (add `import json` to its imports):

```python
def multimetric_chart_html(payload: dict[str, Any]) -> str:
    """Interactive 7-metric trajectory chart: embedded payload + vanilla-JS controller.

    No plotly ``updatemenus`` (state collision); two JS state variables drive
    ``Plotly.react``. ``</`` is escaped in the serialized payload so a hostile
    label can never close the script tag. Deterministic: fixed template +
    sorted-key JSON.
    """
    if not payload.get("eras"):
        return (
            '<div id="multimetric-chart" class="chart-box">'
            "<p>Timeseries data unavailable without local v5.3 assets</p></div>"
        )
    payload_json = json.dumps(payload, sort_keys=True).replace("</", "<\\/")
    return f'''<div id="multimetric-chart" class="chart-box"></div>
<script>
(function () {{
  var root = document.getElementById("multimetric-chart");
  var payload = {payload_json};
  var metricKeys = ["payout", "corr20", "mmc20", "corr60", "mmc60", "bmc", "cwmm"];
  var metricLabels = {{
    payout: "Net Payout Return", corr20: "CORR (20D)", mmc20: "MMC (20D)",
    corr60: "CORR (60D)", mmc60: "MMC (60D)", bmc: "BMC", cwmm: "CWMM"
  }};
  var axisConfig = {{
    payout: {{
      standard: {{title: "Per-Era Net Return", tickformat: ".2%"}},
      cumulative: {{title: "Cumulative Wealth (1.0 Stake)", tickformat: ".3f"}}
    }},
    corr20: {{standard: {{title: "Per-Era CORR (20D)", tickformat: ".4f"}},
              cumulative: {{title: "Cumulative CORR (20D)", tickformat: ".4f"}}}},
    mmc20:  {{standard: {{title: "Per-Era MMC (20D)", tickformat: ".4f"}},
              cumulative: {{title: "Cumulative MMC (20D)", tickformat: ".4f"}}}},
    corr60: {{standard: {{title: "Per-Era CORR (60D)", tickformat: ".4f"}},
              cumulative: {{title: "Cumulative CORR (60D)", tickformat: ".4f"}}}},
    mmc60:  {{standard: {{title: "Per-Era MMC (60D)", tickformat: ".4f"}},
              cumulative: {{title: "Cumulative MMC (60D)", tickformat: ".4f"}}}},
    bmc:    {{standard: {{title: "Per-Era BMC", tickformat: ".4f"}},
              cumulative: {{title: "Cumulative BMC", tickformat: ".4f"}}}},
    cwmm:   {{standard: {{title: "Per-Era CWMM", tickformat: ".4f"}},
              cumulative: {{title: "Cumulative CWMM", tickformat: ".4f"}}}}
  }};
  var currentMetric = "payout";
  var currentView = "standard";
  var mounted = false;

  function esc(s) {{
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }}

  function stressShapes() {{
    var shapes = [];
    var eras = payload.eras;
    var mask = payload.meta_downside_mask || [];
    var start = null;
    for (var i = 0; i <= mask.length; i++) {{
      if (mask[i] && start === null) {{ start = i; }}
      else if (!mask[i] && start !== null) {{
        shapes.push({{type: "rect", xref: "x", yref: "paper", x0: eras[start],
                      x1: eras[i - 1], y0: 0, y1: 1,
                      fillcolor: "rgba(248, 81, 73, 0.10)", line: {{width: 0}},
                      layer: "below"}});
        start = null;
      }}
    }}
    return shapes;
  }}

  function applyState() {{
    var metric = payload.metrics[currentMetric] || {{}};
    var traces = [];
    var ids = Object.keys(metric).sort();
    for (var i = 0; i < ids.length; i++) {{
      var series = metric[ids[i]];
      traces.push({{
        x: payload.eras,
        y: series[currentView],
        mode: "lines",
        name: series.label,
        hovertemplate: "%{{y}}" + "<extra>" + esc(series.label) + "</extra>"
      }});
    }}
    var cfg = axisConfig[currentMetric][currentView];
    var layout = {{
      template: "plotly_dark",
      showlegend: false,
      margin: {{l: 20, r: 20, t: 50, b: 20}},
      xaxis: {{title: "Era"}},
      yaxis: {{title: cfg.title, tickformat: cfg.tickformat}},
      shapes: stressShapes()
    }};
    if (!mounted) {{
      Plotly.newPlot(root, traces, layout);
      mounted = true;
    }} else {{
      Plotly.react(root, traces, layout);
    }}
  }}

  var controls = document.createElement("div");
  controls.style.cssText = "display:flex; gap:1rem; align-items:center; margin-bottom:0.5rem;";
  var select = document.createElement("select");
  for (var i = 0; i < metricKeys.length; i++) {{
    var option = document.createElement("option");
    option.value = metricKeys[i];
    option.textContent = metricLabels[metricKeys[i]];
    select.appendChild(option);
  }}
  select.addEventListener("change", function () {{
    currentMetric = select.value;
    applyState();
  }});
  var stdButton = document.createElement("button");
  stdButton.textContent = "Standard View";
  var cumButton = document.createElement("button");
  cumButton.textContent = "Cumulative View";
  stdButton.addEventListener("click", function () {{ currentView = "standard"; applyState(); }});
  cumButton.addEventListener("click", function () {{ currentView = "cumulative"; applyState(); }});
  controls.appendChild(select);
  controls.appendChild(stdButton);
  controls.appendChild(cumButton);
  root.parentNode.insertBefore(controls, root);
  applyState();
}})();
</script>'''
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k multimetric_chart -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Full suite + docs count + commit**

Recompute the collected count (2 new tests → expect 763) and bump both docs claims. Full suite + ruff on `dashboard_charts.py tests/test_dashboard.py`; commit:

```bash
git add dashboard_charts.py tests/test_dashboard.py AGENTS.md CONTRIBUTING.md
git commit -m "feat(dashboard): JS-controller multimetric chart (no updatemenus)"
```

---

### Task 5: Charts — similarity heatmap + adapted drawdown; delete wealth chart

**Files:**
- Modify: `dashboard_charts.py`
- Test: `tests/test_dashboard.py` (delete the v1 `test_wealth_chart_downside_vrects` — superseded)

**Interfaces:**
- Consumes: none from nmr.
- Produces: `build_similarity_matrix_chart(labels: list[str], matrix: list[list[float]]) -> go.Figure`; `build_drawdown_chart(payload: dict[str, Any]) -> go.Figure` (new payload shape); `build_cumulative_wealth_chart` DELETED.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard.py`:

```python
def test_similarity_chart_heatmap_and_highlight() -> None:
    fig = charts.build_similarity_matrix_chart(
        ["top", "second"], [[1.0, 0.7], [0.7, 1.0]]
    )
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert list(trace.z[0]) == [1.0, 0.7]
    assert "<b>" in trace.text[0][0]      # row/col 0 highlight (decision #25)
    assert "<b>" in trace.text[1][0]
    assert trace.colorscale is not None and "RdBu_r" in str(trace.colorscale)


def test_similarity_chart_empty_matrix_annotation() -> None:
    fig = charts.build_similarity_matrix_chart([], [])
    assert len(fig.data) == 0
    assert "Similarity matrix unavailable without local v5.3 assets" in fig.layout.annotations[0].text


def test_drawdown_chart_v2_payload() -> None:
    payload = {
        "eras": ["0001", "0002"],
        "meta_downside_mask": [False, False],
        "drawdowns": {"a": [0.0, -0.01]},
        "metrics": {"payout": {"a": {"label": "run-a"}}},
    }
    fig = charts.build_drawdown_chart(payload)
    assert len(fig.data) == 1
    assert fig.data[0].y[-1] == pytest.approx(-0.01)
    assert fig.data[0].fill == "tozeroy"
```

Delete `test_wealth_chart_downside_vrects` and the `build_cumulative_wealth_chart` reference in `_charts_for_test` (replace `charts.build_cumulative_wealth_chart(_ts_payload())` with `charts.build_drawdown_chart(_ts_payload())` — the `_ts_payload` helper is superseded for the wealth chart; keep the helper for drawdown, adjusting its keys if needed).

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k "similarity_chart or drawdown_chart" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'build_similarity_matrix_chart'`.

- [ ] **Step 3: Write minimal implementation**

Append/replace in `dashboard_charts.py` (delete `build_cumulative_wealth_chart` and `_downside_spans` if nothing else uses it):

```python
def build_similarity_matrix_chart(
    labels: list[str], matrix: list[list[float]]
) -> go.Figure:
    """Pairwise similarity heatmap with the top-ranked row/col highlighted."""
    fig = go.Figure()
    if not matrix:
        fig.add_annotation(
            text="Similarity matrix unavailable without local v5.3 assets",
            showarrow=False,
        )
        fig.update_layout(template="plotly_dark")
        return fig
    text = [
        [
            f"<b>{v:.3f}</b>" if (i == 0 or j == 0) else f"{v:.3f}"
            for j, v in enumerate(row)
        ]
        for i, row in enumerate(matrix)
    ]
    fig.add_trace(
        go.Heatmap(
            z=matrix, x=labels, y=labels, colorscale="RdBu_r", zmid=0.5,
            text=text, texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
    return fig


def build_drawdown_chart(payload: dict[str, Any]) -> go.Figure:
    """Underwater payout drawdown curves (v2 payload: drawdowns + eras)."""
    fig = go.Figure()
    eras = payload.get("eras") or []
    if not eras:
        fig.add_annotation(
            text="Timeseries data unavailable without local v5.3 assets",
            showarrow=False,
        )
        fig.update_layout(template="plotly_dark")
        return fig
    drawdowns = payload.get("drawdowns") or {}
    payout_metric = (payload.get("metrics") or {}).get("payout", {})
    for model_id in sorted(drawdowns):
        label = payout_metric.get(model_id, {}).get("label", model_id)
        fig.add_trace(
            go.Scatter(
                name=label,
                x=eras,
                y=drawdowns[model_id],
                mode="lines",
                fill="tozeroy",
                fillcolor=_DOWNSIDE_FILL,
                hovertemplate="%{y:.2%}<extra>" + html.escape(label) + "</extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Era",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
    )
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k "similarity_chart or drawdown_chart or chart" -v`
Expected: PASS (all chart tests; the leaderboard tests remain green).

- [ ] **Step 5: Full suite + docs count + commit**

Recompute the collected count (2 added − 1 deleted → net +1 → expect 764) and bump both docs claims. Full suite + ruff; commit:

```bash
git add dashboard_charts.py tests/test_dashboard.py AGENTS.md CONTRIBUTING.md
git commit -m "feat(dashboard): similarity heatmap with highlight; drawdown on v2 payload"
```

---

### Task 6: `generate_dashboard.py` — v2 wiring, badge, ensemble card, layout

**Files:**
- Modify: `generate_dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `extract_multimetric_timeseries`, `extract_pairwise_similarity_matrix` (replaces `extract_payout_timeseries`), `multimetric_chart_html`, `build_similarity_matrix_chart`, `build_drawdown_chart`, `build_leaderboard_bar_chart`.
- Produces: `_diversification_stats(matrix) -> dict`, `_ensemble_sharpe(payout_metric: dict) -> float | None`, `_badge_html(stats, stress) -> str`, `_ensemble_card_html(value, n_models) -> str`; updated `_build_html` (sections 1/3/5; four render calls); updated `generate_dashboard()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard.py`:

```python
def test_diversification_stats_thresholds() -> None:
    low = generate_dashboard._diversification_stats(
        [[1.0, 0.4, 0.3], [0.4, 1.0, 0.5], [0.3, 0.5, 1.0]]
    )
    assert low["mean_overlap"] == pytest.approx(0.4, abs=1e-9)
    assert low["max_overlap"] == pytest.approx(0.5, abs=1e-9)
    assert low["badge"] == "EXCELLENT DIVERSIFICATION"
    high = generate_dashboard._diversification_stats([[1.0, 0.9], [0.9, 1.0]])
    assert high["badge"] == "HIGH REDUNDANCY"
    mid = generate_dashboard._diversification_stats([[1.0, 0.7], [0.7, 1.0]])
    assert mid["badge"] == "MODERATE OVERLAP"


def test_ensemble_sharpe_card_guard() -> None:
    assert generate_dashboard._ensemble_sharpe({}) is None           # < 2 series
    assert generate_dashboard._ensemble_sharpe({"a": {"standard": [0.01, 0.02]}}) is None
    value = generate_dashboard._ensemble_sharpe({
        "a": {"standard": [0.01, 0.02, 0.03]},
        "b": {"standard": [0.02, 0.01, 0.02]},
    })
    assert isinstance(value, float) and value == value  # finite


def test_build_html_v2_sections_and_four_render_calls(tmp_path: Path) -> None:
    payload = {
        "eras": ["0001", "0002"],
        "meta_downside_mask": [False, False],
        "metrics": {"payout": {"a": {"standard": [0.01, 0.02], "cumulative": [1.01, 1.0302], "label": "run · aaaaaaaa"}},
                    "corr20": {}, "mmc20": {}, "corr60": {}, "mmc60": {}, "bmc": {}, "cwmm": {}},
        "drawdowns": {"a": [0.0, -0.01]},
    }
    rows = pl.DataFrame(
        [{"model_id": "a", "source": "trained", "run_name": "run",
          "corr_sharpe_ac": 0.8, "corr_sharpe_ac_ci_low": 0.6, "corr_sharpe_ac_ci_high": 1.0,
          "cagr_1y": 1.5, "gain_to_pain_ratio": 2.0, "kelly_fraction": 0.4,
          "mmc_down": 0.01, "deflated_sharpe": 0.97, "max_drawdown": 0.1,
          "fnc": 0.05, "corr": 0.12, "status": "RESEARCH", "tier": None}]
    )
    figures = {
        "leaderboard": charts.build_leaderboard_bar_chart(_bar_input(), hurdle_sharpe=0.78),
        "similarity": charts.build_similarity_matrix_chart(["a", "b"], [[1.0, 0.5], [0.5, 1.0]]),
        "drawdown": charts.build_drawdown_chart(payload),
    }
    multimetric_block = charts.multimetric_chart_html(payload)
    html_text = generate_dashboard._build_html(
        leaderboard=rows, champion=None, kpis=_kpis_for_test(),
        figures=figures, multimetric_block=multimetric_block,
        badge_html="<p>BADGE Mean 0.50 Max 0.50 MODERATE OVERLAP</p>",
        ensemble_card_html="<p>ENSEMBLE CARD 1.234</p>",
        registry_dir=tmp_path, technical_entries=[],
    )
    for section in ("ALPHA GENERATION", "SIGNAL DIVERSIFICATION",
                    "CAPITAL DRAWDOWN", "BADGE", "ENSEMBLE CARD"):
        assert section in html_text
    assert html_text.count("<!-- plotly-engine-embed -->") == 1
    assert html_text.split("</script>", 1)[1].count("Plotly.newPlot(") == 4
    assert "<script src" not in html_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k "diversification_stats or ensemble_sharpe or v2_sections" -v`
Expected: FAIL — `AttributeError: module 'generate_dashboard' has no attribute '_diversification_stats'` and the old `_build_html` signature rejects the new kwargs.

- [ ] **Step 3: Write minimal implementation**

In `generate_dashboard.py` (add `import numpy as np`; replace the `extract_payout_timeseries` import with `extract_multimetric_timeseries, extract_pairwise_similarity_matrix`; add `multimetric_chart_html, build_similarity_matrix_chart` imports):

```python
def _diversification_stats(matrix: list[list[float]]) -> dict:
    """Max/mean off-diagonal overlap + badge tier (decision #16)."""
    n = len(matrix)
    off = [matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    mean = float(np.mean(off)) if off else None
    maximum = float(max(off)) if off else None
    if mean is None:
        badge = "—"
    elif mean < 0.65:
        badge = "EXCELLENT DIVERSIFICATION"
    elif mean <= 0.85:
        badge = "MODERATE OVERLAP"
    else:
        badge = "HIGH REDUNDANCY"
    return {"mean_overlap": mean, "max_overlap": maximum, "badge": badge}


def _ensemble_sharpe(payout_metric: dict) -> float | None:
    """Equal-weighted blended Sharpe from per-era payout series (decision #17).

    SR_blended = mean(mu) / sqrt(w^T Sigma w), w uniform; None when fewer
    than 2 usable series or zero variance.
    """
    series = [
        np.asarray(v["standard"], dtype=float)
        for v in payout_metric.values()
        if v.get("standard")
    ]
    if len(series) < 2:
        return None
    stacked = np.vstack(series)
    mu = np.mean(stacked, axis=1)
    weights = np.full(len(mu), 1.0 / len(mu))
    variance = float(weights @ np.cov(stacked) @ weights)
    if variance <= 0.0 or not np.isfinite(variance):
        return None
    return float(np.mean(mu) / np.sqrt(variance))


def _badge_html(stats: dict, stress: dict) -> str:
    delta = stress.get("mean_delta")
    delta_text = "—" if delta is None else f"{delta:+.3f}"
    mean_text = "—" if stats["mean_overlap"] is None else f"{stats['mean_overlap']:.3f}"
    max_text = "—" if stats["max_overlap"] is None else f"{stats['max_overlap']:.3f}"
    return (
        f'<p class="badge-line"><b>{html.escape(stats["badge"])}</b> · '
        f"Mean Overlap {mean_text} · Max Overlap {max_text} · "
        f"Stress-Regime Δρ {delta_text}</p>"
    )


def _ensemble_card_html(value: float | None) -> str:
    text = "—" if value is None else f"{value:.3f}"
    return (
        '<div class="kpi"><div class="label">Equal-Weight Ensemble Sharpe '
        f"(top-3, heuristic)</div><div class=\"value\">{text}</div></div>"
    )
```

Replace `_build_html`'s signature and template:

```python
def _build_html(
    leaderboard: pl.DataFrame,
    champion: str | None,
    kpis: dict,
    figures: dict,
    multimetric_block: str,
    badge_html: str,
    ensemble_card_html: str,
    registry_dir: Path,
    technical_entries: list[dict],
) -> str:
    engine_js = get_plotlyjs()
    figure_html = {
        name: pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=name)
        for name, fig in figures.items()
    }
    ...
```

Template body (only the changed sections; keep the v1 head/CSS/KPI/table/accordion structure):

```
<h2>1. Alpha Generation &amp; Multi-Metric Performance Trajectory</h2>
{multimetric_block}
<h2>2. Risk-Adjusted Return Leaderboard</h2>
{figure_html['leaderboard']}
<h2>3. Signal Diversification &amp; Pairwise Similarity Matrix</h2>
{badge_html}
{figure_html['similarity']}
{ensemble_card_html}
<h2>4. Executive Allocation &amp; Risk Decision Table</h2>
<table>... (unchanged v1 rows_html) ...</table>
<h2>5. Capital Drawdown (Underwater Trajectory)</h2>
{figure_html['drawdown']}
<h2>Technical &amp; Audit Metadata</h2>
{accordion}
```

Rewrite `generate_dashboard()`:

```python
    leaderboard = load_unified_leaderboard(registry_dir, benchmark_path=benchmark_path)
    leaderboard = reconcile_capital_metrics(leaderboard, DEFAULT_DATA_DIR)
    statuses = evaluate_gate_status(leaderboard, DEFAULT_GATE_PATH, registry_dir / "champion.json")
    leaderboard = leaderboard.join(statuses, on="model_id", how="left")

    gate_cfg = load_benchmark_file(DEFAULT_GATE_PATH)
    assert gate_cfg.reference_column is not None
    tier4_column = str(gate_cfg.reference_column)
    hurdle_sharpe = float(gate_cfg.gate.corr_sharpe_ac_min)

    champion = read_champion_pointer(registry_dir / "champion.json")
    fleet = leaderboard.filter(pl.col("source").is_in(["trained", "trained_legacy"]))
    top3 = fleet.sort("corr_sharpe_ac", descending=True, nulls_last=True).head(3)
    top3_ids = top3.get_column("model_id").to_list()

    payload = extract_multimetric_timeseries(
        registry_dir, DEFAULT_DATA_DIR, run_ids=top3_ids,
        include_tier4_ref=True, tier4_column=tier4_column,
    )
    top5_ids = fleet.sort("corr_sharpe_ac", descending=True, nulls_last=True).head(5) \
        .get_column("model_id").to_list()
    labels, sim_ids, matrix, stress = extract_pairwise_similarity_matrix(
        registry_dir, DEFAULT_DATA_DIR, run_ids=top5_ids,
        include_tier4_ref=True, tier4_column=tier4_column,
    )
    stats = _diversification_stats(matrix)
    payout_metric = (payload.get("metrics") or {}).get("payout") or {}
    top3_payout = {
        mid: payout_metric[mid]
        for mid in top3_ids
        if mid in payout_metric
    }
    ensemble_value = _ensemble_sharpe(top3_payout)

    figures = {
        "leaderboard": charts.build_leaderboard_bar_chart(
            _bar_input(leaderboard, champion), hurdle_sharpe=hurdle_sharpe
        ),
        "similarity": charts.build_similarity_matrix_chart(labels, matrix),
        "drawdown": charts.build_drawdown_chart(payload),
    }
    html_text = _build_html(
        leaderboard=leaderboard, champion=champion,
        kpis=_kpi_cards(leaderboard, champion, hurdle_sharpe),
        figures=figures,
        multimetric_block=charts.multimetric_chart_html(payload),
        badge_html=_badge_html(stats, stress),
        ensemble_card_html=_ensemble_card_html(ensemble_value),
        registry_dir=registry_dir,
        technical_entries=_technical_entries(registry_dir),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k "diversification_stats or ensemble_sharpe or v2_sections or html" -v`
Expected: PASS. The existing `test_generate_dashboard_end_to_end_synthetic` must also still pass — if it asserts old section names or render counts, update it: "CAPITAL READY" stays, `split("</script>", 1)[1].count("Plotly.newPlot(") == 4`.

- [ ] **Step 5: Full suite + docs count + commit**

Recompute the collected count (3 new tests → expect 767) and bump both docs claims. Full suite + ruff; commit:

```bash
git add generate_dashboard.py tests/test_dashboard.py AGENTS.md CONTRIBUTING.md
git commit -m "feat(dashboard): v2 layout — multimetric chart, similarity section, badge, ensemble card"
```

---

### Task 7: Real-data acceptance + docs sync

**Files:**
- Modify: `tests/test_dashboard.py`
- Modify: `docs/superpowers/specs/2026-08-16-executive-dashboard-v2-design.md` (signature line: similarity returns a 4-tuple)
- Modify: `ARCHITECTURE.md` (§W v2 surface)

**Interfaces:**
- Consumes: everything from Tasks 1–6; real assets (skip-marked like `tests/test_parity.py`).

- [ ] **Step 1: Write the real-data tests**

Append to `tests/test_dashboard.py` (reuses `_HAS_REAL` from v1; `import nmr.payout as payout` already imported):

```python
@pytest.mark.skipif(not _HAS_REAL, reason="real registry/v5.3 data absent; skipped in CI")
def test_real_multimetric_payload_and_payout_parity() -> None:
    frame = dash.load_unified_leaderboard(_REAL_REGISTRY, benchmark_path=False)
    top = frame.sort("corr_sharpe_ac", descending=True, nulls_last=True).row(0, named=True)
    payload = dash.extract_multimetric_timeseries(
        _REAL_REGISTRY, Path("data/v5.3"), run_ids=[top["model_id"]],
        include_tier4_ref=False,
    )
    assert len(payload["eras"]) == 86
    for name in ("payout", "corr20", "mmc20", "corr60", "mmc60", "bmc", "cwmm"):
        series = payload["metrics"][name][top["model_id"]]
        assert len(series["standard"]) == 86
        assert len(series["cumulative"]) == 86
    # chart payout compounding == table cagr_1y (same "target" anchor)
    reconciled = dash.reconcile_capital_metrics(frame, Path("data/v5.3"))
    row = reconciled.filter(pl.col("model_id") == top["model_id"]).row(0, named=True)
    from nmr.payout import annual_compounded_return
    assert annual_compounded_return(
        payload["metrics"]["payout"][top["model_id"]]["standard"]
    ) == pytest.approx(row["cagr_1y"], rel=1e-6)


@pytest.mark.skipif(not _HAS_REAL, reason="real registry/v5.3 data absent; skipped in CI")
def test_real_similarity_matrix_top5_with_tier4() -> None:
    frame = dash.load_unified_leaderboard(_REAL_REGISTRY, benchmark_path=False)
    top5 = (
        frame.sort("corr_sharpe_ac", descending=True, nulls_last=True)
        .head(5).get_column("model_id").to_list()
    )
    labels, ids, matrix, stress = dash.extract_pairwise_similarity_matrix(
        _REAL_REGISTRY, Path("data/v5.3"), run_ids=top5,
        include_tier4_ref=True,
    )
    assert len(ids) == 6 and "v53_lgbm_ender60" in ids
    for i in range(len(ids)):
        assert matrix[i][i] == pytest.approx(1.0, abs=1e-12)
        for j in range(len(ids)):
            assert -1.0 <= matrix[i][j] <= 1.0
    assert set(stress) == {"mean_delta", "n_pairs"}
```

- [ ] **Step 2: Run them**

Run: `./.venv/Scripts/python -m pytest tests/test_dashboard.py -k real_ -v`
Expected: PASS on this machine (data present), SKIP in CI. Do not widen tolerances silently — report diffs if they fail.

- [ ] **Step 3: Docs sync**

- `docs/superpowers/specs/2026-08-16-executive-dashboard-v2-design.md` §4: change the similarity return annotation to `-> tuple[list[str], list[str], list[list[float]], dict[str, Any]]` and add: "(plan-level refinement: the stress-regime delta rides along as a fourth element — `{"mean_delta": float | None, "n_pairs": int}` — rather than a separate function.)"
- `ARCHITECTURE.md` §W: replace the v1 description sentence with: "executive report engine — unified leaderboard, tier-4 gate projection, stored-first capital recompute over the 86-era meta-overlap window, 7-metric multimetric timeseries (payout anchored to `target`; BMC self-guard; zeroed missing horizons), pairwise rank-gaussian similarity matrix with stress-regime delta; plotly/streamlit-free. `dashboard_charts.py` adds the JS-controller multimetric chart (no `updatemenus`) and the similarity heatmap."

- [ ] **Step 4: Full suite + docs count + commit**

Recompute the collected count (2 new tests → expect 769) and bump both docs claims. Full suite + ruff; commit:

```bash
git add tests/test_dashboard.py docs/superpowers/specs/2026-08-16-executive-dashboard-v2-design.md ARCHITECTURE.md AGENTS.md CONTRIBUTING.md
git commit -m "test(dashboard): v2 real-data acceptance; docs sync for v2 surface"
```

---

### Task 8: Final verification gate

**Files:** none (verification only)

- [ ] **Step 1: Lint** — `./.venv/Scripts/python -m ruff check .` → clean.
- [ ] **Step 2: Full suite** — `./.venv/Scripts/python -m pytest -q` → green (expect 769; report skips).
- [ ] **Step 3: Real compile** — `./.venv/Scripts/python generate_dashboard.py` → exit 0; `artifacts/dashboard.html` written.
- [ ] **Step 4: Inspection** — run:

```bash
./.venv/Scripts/python -c "
from pathlib import Path
text = Path('artifacts/dashboard.html').read_text(encoding='utf-8')
assert text.count('<!-- plotly-engine-embed -->') == 1
assert '<script src' not in text
assert text.split('</script>', 1)[1].count('Plotly.newPlot(') == 4
for section in ('ALPHA GENERATION', 'SIGNAL DIVERSIFICATION', 'CAPITAL DRAWDOWN',
                'Equal-Weight Ensemble Sharpe', 'Stress-Regime'):
    assert section in text, section
print('v2 dashboard.html OK:', len(text), 'bytes')
"
```

Expected: prints the OK line.

- [ ] **Step 5: Manual browser check** — open `artifacts/dashboard.html` once in a browser and verify: the metric dropdown switches all 7 metrics, the Standard/Cumulative toggle switches views with correct y-axis titles (e.g. "Per-Era Net Return" vs "Cumulative Wealth (1.0 Stake)"), the stress shading renders, the similarity heatmap shows the bold top row/col, and the badge/ensemble card values are sensible. Record the result in the final report (this check cannot be automated).

- [ ] **Step 6: Commit** (only if the HTML is tracked — it is git-ignored, so skip and note it)

---

## Self-Review Notes (completed by plan author)

- Spec coverage: decisions #1–#27 all map to tasks — #10/#18 → Task 1; #3/#9/#11/#13/#14/#19/#23 → Task 2; #2/#5/#12/#15/#20/#22/#26 → Task 3; #4/#21/#24 → Task 4; #8/#25 → Task 5; #16/#17/#27 + layout §6 → Task 6; §7 real-data + docs → Task 7; gates → Task 8.
- Placeholder scan: the two marked implementer notes (the stray `meta_corr` line in Task 2 and the JSON round-trip line in Task 4) contain the corrected plain code inline — do not ship the marked placeholder forms.
- Type consistency: `_V2Lookups` field names match all uses in Tasks 2–3; payload keys (`eras`, `meta_downside_mask`, `metrics.<name>.<id>.{standard,cumulative,label}`, `drawdowns`) match Task 4's controller and Task 5's drawdown chart; `extract_pairwise_similarity_matrix`'s 4-tuple order `(labels, run_ids, matrix, stress_stats)` matches Task 3 tests, Task 6 wiring, and Task 7 docs sync; `_build_html`'s new kwargs (`multimetric_block`, `badge_html`, `ensemble_card_html`) match its only caller and its test.
