from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from plotly.colors import diverging

import generate_dashboard
import nmr.dashboard as dash
import nmr.evaluation as nmr_evaluation
import nmr.payout as payout
from dashboard_ui import charts
from nmr.config import REPO_ROOT
from nmr.payout import annual_compounded_return


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
    # given path missing -> chain: full (missing) -> smoke (hit)
    assert dash.resolve_benchmark_path(reports / "nope.csv", reports_dir=reports) == smoke
    smoke.unlink()
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
    bench_row = frame.filter(pl.col("source") == "benchmark").row(0, named=True)
    assert bench_row["family"] is None
    assert bench_row["training_scope"] is None
    assert bench_row["has_full_version"] is False


def test_load_unified_leaderboard_empty_registry_returns_schema_frame(tmp_path: Path) -> None:
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    assert frame.height == 0
    assert frame.schema == dash.UNIFIED_SCHEMA


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
        ["era", "id", pl.col("target").alias("numerai_meta_model")]
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
    out = dash.reconcile_capital_metrics(frame, data)
    row = out.row(0, named=True)
    assert row["cagr_1y"] is not None
    assert row["gain_to_pain_ratio"] is not None
    assert row["kelly_fraction"] is not None
    # perfect corr with target and meta -> payout == 0.05 clipped every era
    assert row["cagr_1y"] == pytest.approx((1.05 ** 52) - 1.0, abs=1e-6)
    assert row["gain_to_pain_ratio"] == float("inf")  # no losing eras


def test_reconcile_capital_metrics_stored_block_wins(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("b" * 64)])
    _write_preds(tmp_path / ("b" * 64), scale=0.0)  # junk preds must be ignored
    data = _synthetic_data_dir(tmp_path)
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    out = dash.reconcile_capital_metrics(frame, data)
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
    out = dash.reconcile_capital_metrics(frame, data)
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
    out = dash.reconcile_capital_metrics(frame, tmp_path / "no-data")
    assert out.row(0, named=True)["cagr_1y"] is None


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
    standard = payload["metrics"]["payout"]["b" * 64]["standard"]
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


def test_multimetric_payout_aligned_when_model_misses_an_era(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("e" * 64)])
    rows = [
        {"era": era, "id": f"{era}_{i:03d}", "prediction": float(i)}
        for era in ("0001", "0003")  # era 0002 deliberately missing
        for i in range(10)
    ]
    pl.DataFrame(rows).write_parquet(tmp_path / ("e" * 64) / "validation_preds.parquet")
    data = _synthetic_v2_data_dir(tmp_path)
    payload = dash.extract_multimetric_timeseries(
        tmp_path, data, run_ids=["e" * 64], include_tier4_ref=False
    )
    payout = payload["metrics"]["payout"]["e" * 64]
    assert len(payout["standard"]) == 3          # aligned 1:1 with the axis
    assert payout["standard"][1] == 0.0          # missing era zero-filled
    assert len(payload["drawdowns"]["e" * 64]) == 3
    for name in ("corr20", "mmc20", "corr60", "mmc60", "bmc", "cwmm"):
        assert len(payload["metrics"][name]["e" * 64]["standard"]) == 3


def test_multimetric_timeseries_benchmarks_absent_degrades(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64)])
    _write_preds(tmp_path / ("a" * 64), scale=1.0)
    data = _synthetic_v2_data_dir(tmp_path, with_benchmark=False)
    with caplog.at_level(logging.WARNING, logger="nmr.dashboard"):
        payload = dash.extract_multimetric_timeseries(
            tmp_path, data, run_ids=["a" * 64], include_tier4_ref=True
        )
    # no benchmark parquet -> tier-4 reference dropped; every model slice still
    # renders, with BMC zero-filled (degrade, never raise — decision #23)
    assert set(payload["metrics"]) == {
        "payout", "corr20", "mmc20", "corr60", "mmc60", "bmc", "cwmm"
    }
    assert set(payload["metrics"]["payout"]) == {"a" * 64}
    for name in ("payout", "corr20", "mmc20", "corr60", "mmc60", "bmc", "cwmm"):
        series = payload["metrics"][name]["a" * 64]
        assert len(series["standard"]) == 3
        assert len(series["cumulative"]) == 3
    assert payload["metrics"]["bmc"]["a" * 64]["standard"] == pytest.approx(
        [0.0, 0.0, 0.0], abs=1e-12
    )
    assert payload["metrics"]["bmc"]["a" * 64]["cumulative"] == pytest.approx(
        [0.0, 0.0, 0.0], abs=1e-12
    )
    assert "a" * 64 in payload["drawdowns"]
    assert "bmc zeroed" in caplog.text


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
    out = dash.reconcile_capital_metrics(frame, Path("data/v5.3"))
    trained = out.filter(pl.col("source").is_in(["trained", "trained_legacy"]))
    assert trained.height > 0
    for row in trained.to_dicts():
        assert row["cagr_1y"] is not None
        assert row["gain_to_pain_ratio"] is not None
        assert row["kelly_fraction"] is not None


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
    _labels, ids, matrix, stress = dash.extract_pairwise_similarity_matrix(
        _REAL_REGISTRY, Path("data/v5.3"), run_ids=top5,
        include_tier4_ref=True,
    )
    assert len(ids) == 6 and "v53_lgbm_ender60" in ids
    for i in range(len(ids)):
        assert matrix[i][i] == pytest.approx(1.0, abs=1e-12)
        for j in range(len(ids)):
            assert -1.0 <= matrix[i][j] <= 1.0
    assert set(stress) == {"mean_delta", "n_pairs"}


def test_dashboard_symbols_exported_from_package() -> None:
    import nmr

    for name in (
        "UNIFIED_SCHEMA",
        "evaluate_gate_status",
        "extract_multimetric_timeseries",
        "extract_pairwise_similarity_matrix",
        "load_benchmark_frame",
        "load_unified_leaderboard",
        "read_champion_pointer",
        "reconcile_capital_metrics",
        "resolve_benchmark_path",
    ):
        assert getattr(nmr, name) is not None, f"nmr.{name} not exported"
        assert name in nmr.__all__


def _bar_input() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"label": "run-a · aaaaaaaa", "corr_sharpe_ac": 0.8,
             "corr_sharpe_ac_ci_low": 0.6, "corr_sharpe_ac_ci_high": 1.0,
             "champion": True, "cagr_1y": 1.5, "max_drawdown": 0.2,
             "deflated_sharpe": 0.97},
            {"label": "bench · bbbbbbbb", "corr_sharpe_ac": 0.5,
             "corr_sharpe_ac_ci_low": None, "corr_sharpe_ac_ci_high": None,
             "champion": False},
        ]
    )


def _ts_payload() -> dict:
    return {
        "eras": ["0001", "0002", "0003", "0004"],
        "meta_downside_mask": [True, True, False, True],
        "drawdowns": {"a": [0.0, 0.0, 0.0, -0.018]},
        "metrics": {"payout": {"a": {"label": "run-a"}}},
    }


def _multimetric_payload() -> dict:
    return {
        "eras": ["0001", "0002"],
        "meta_downside_mask": [False, False],
        "metrics": {"payout": {"a": {"standard": [0.01, 0.02],
                                      "cumulative": [1.01, 1.0302],
                                      "label": "run · aaaaaaaa"}},
                    "corr20": {}, "mmc20": {}, "corr60": {}, "mmc60": {},
                    "bmc": {}, "cwmm": {}},
        "drawdowns": {"a": [0.0, -0.01]},
    }


def test_leaderboard_chart_traces_and_hurdle_line() -> None:
    fig = charts.build_leaderboard_bar_chart(_bar_input(), hurdle_sharpe=0.78)
    assert len(fig.data) == 2
    # first trace is the last row (ascending order -> best on top)
    assert fig.data[0].y[0] == "bench · bbbbbbbb"
    # hurdle line is a layout shape
    shapes = [s for s in fig.layout.shapes if s.type == "line"]
    assert any(abs(s.x0 - 0.78) < 1e-9 for s in shapes)


def test_leaderboard_chart_hover_fields() -> None:
    fig = charts.build_leaderboard_bar_chart(_bar_input(), hurdle_sharpe=0.78)
    # ascending sort: bench row (no hover values) first, run-a row second
    rich = fig.data[1]
    assert "Ann. Return" in rich.hovertemplate
    assert list(rich.customdata) == [[1.5, 0.2, 0.97]]


def test_chart_hovertemplate_escapes_labels() -> None:
    payload = _ts_payload()
    payload["metrics"]["payout"]["a"]["label"] = "run-<img src=x>"
    fig = charts.build_drawdown_chart(payload)
    assert "&lt;img" in fig.data[0].hovertemplate
    assert "<img src=x>" not in fig.data[0].hovertemplate


def test_drawdown_chart_underwater_fill() -> None:
    fig = charts.build_drawdown_chart(_ts_payload())
    assert len(fig.data) == 1
    assert fig.data[0].fill == "tozeroy"
    assert fig.data[0].y[-1] == pytest.approx(-0.018)


def test_timeseries_charts_empty_payload_render_annotation() -> None:
    empty = {"eras": [], "meta_downside_mask": [], "drawdowns": {}, "metrics": {}}
    fig = charts.build_drawdown_chart(empty)
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


def _charts_for_test() -> dict:
    bar = charts.build_leaderboard_bar_chart(_bar_input(), hurdle_sharpe=0.78)
    similarity = charts.build_similarity_matrix_chart(
        ["run-a · aaaaaaaa"], [[1.0]]
    )
    return {"leaderboard": bar, "similarity": similarity,
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
        multimetric_block=charts.multimetric_chart_html(_multimetric_payload()),
        badge_html="<p>BADGE MODERATE OVERLAP</p>",
        ensemble_card_html="<p>ENSEMBLE CARD —</p>",
        registry_dir=tmp_path,
        technical_entries=[],
    )
    assert '"><img src=x' not in html_text            # hostile run_name escaped, never raw
    assert "&lt;img src=x onerror=alert(2)&gt;" in html_text
    # engine embed marker (the bundle itself contains many "window.Plotly"
    # literals, so count the template's own marker, not bundle internals)
    assert html_text.count("<!-- plotly-engine-embed -->") == 1
    assert "<script src" not in html_text            # zero external script tags (offline)
    # four figure render calls (3 pio figures + the multimetric JS controller),
    # counted AFTER the engine block so the plotly.js bundle's own
    # "Plotly.newPlot(...)" example string is excluded
    assert html_text.split("</script>", 1)[1].count("Plotly.newPlot(") == 4
    assert 'class="num gate-fail"' in html_text   # failing gate cell tinted
    assert "badge research" in html_text          # status badge pill rendered


def test_kpi_cards_stale_champion_pointer_degrades(caplog: pytest.LogCaptureFixture) -> None:
    frame = pl.DataFrame(
        [{"model_id": "a" * 64, "source": "trained", "run_name": "sample-run",
          "corr_sharpe_ac": 0.8, "cagr_1y": 1.5, "max_drawdown": 0.1,
          "status": "RESEARCH", "n_eras": 30}]
    )
    with caplog.at_level(logging.WARNING, logger="generate_dashboard"):
        kpis = generate_dashboard._kpi_cards(
            frame, champion="9" * 64, hurdle_sharpe=0.78
        )
    assert kpis["champion_label"] == "None Designated"
    assert kpis["champion_detail"] == "(Unallocated)"
    assert "champion" in caplog.text and "not found in leaderboard" in caplog.text


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
    # v2 layout sections always present, even with degraded data payloads
    for section in ("ALPHA GENERATION", "SIGNAL DIVERSIFICATION", "CAPITAL DRAWDOWN"):
        assert section in text
    # four Plotly mounts (3 figures + the multimetric JS controller); without
    # local v5.3 assets (CI) the controller degrades to an annotation and only
    # the three pio figures mount
    expected_renders = 4 if Path("data/v5.3/validation.parquet").exists() else 3
    assert text.split("</script>", 1)[1].count("Plotly.newPlot(") == expected_renders
    # size is unbounded by ruling (full plotly engine inline, ~4.9 MB)


def test_build_html_deterministic_across_calls(tmp_path: Path) -> None:
    rows = pl.DataFrame(
        [{"model_id": "a" * 64, "source": "trained", "run_name": "sample-run",
          "corr_sharpe_ac": 0.8, "corr_sharpe_ac_ci_low": 0.6,
          "corr_sharpe_ac_ci_high": 1.0, "cagr_1y": 1.5,
          "gain_to_pain_ratio": 2.0, "kelly_fraction": 0.4, "mmc_down": 0.01,
          "deflated_sharpe": 0.97, "max_drawdown": 0.1, "fnc": 0.05,
          "corr": 0.12, "status": "RESEARCH", "tier": None}]
    )
    kwargs = dict(
        leaderboard=rows, champion=None, kpis=_kpis_for_test(),
        figures=_charts_for_test(),
        multimetric_block=charts.multimetric_chart_html(_multimetric_payload()),
        badge_html="<p>BADGE MODERATE OVERLAP</p>",
        ensemble_card_html="<p>ENSEMBLE CARD —</p>",
        registry_dir=tmp_path, technical_entries=[],
    )
    first = generate_dashboard._build_html(**kwargs)
    second = generate_dashboard._build_html(**kwargs)
    assert first == second  # byte-identical, no random div ids


def _synthetic_v2_data_dir(tmp_path: Path, *, with_benchmark: bool = True) -> Path:
    data = _synthetic_data_dir(tmp_path)  # era/id/target + meta over 0001..0003
    if with_benchmark:
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


def test_similarity_stress_delta_finite_under_degenerate_stress(tmp_path: Path) -> None:
    # meta flipped negative in era 0001 -> a stress era exists; a constant-
    # prediction model makes the stress-subset column degenerate.
    data = _synthetic_data_dir(tmp_path)
    meta_path = data / "meta_model.parquet"
    rows = []
    for era in ("0001", "0002", "0003"):
        for i in range(10):
            sign = -1.0 if era == "0001" else 1.0
            rows.append({"era": era, "id": f"{era}_{i:03d}",
                         "numerai_meta_model": sign * float(i)})
    pl.DataFrame(rows).write_parquet(meta_path)
    pl.DataFrame(
        [{"era": era, "id": f"{era}_{i:03d}", "v53_lgbm_ender60": float(i)}
         for era in ("0001", "0002", "0003") for i in range(10)]
    ).write_parquet(data / "validation_benchmark_models.parquet")
    _write_registry(tmp_path, [_registry_entry("f" * 64)])
    _write_preds(tmp_path / ("f" * 64), scale=0.0)
    _labels, ids, matrix, stress = dash.extract_pairwise_similarity_matrix(
        tmp_path, data, run_ids=["f" * 64], include_tier4_ref=True
    )
    # mean_delta may be None (insufficient stress rows) but never NaN
    assert stress["mean_delta"] is None or stress["mean_delta"] == stress["mean_delta"]
    assert all(-1.0 <= v <= 1.0 for row in matrix for v in row)


def test_similarity_matrix_missing_data_assets(tmp_path: Path) -> None:
    out = dash.extract_pairwise_similarity_matrix(
        tmp_path, tmp_path / "no-data", run_ids=["a" * 64], include_tier4_ref=False
    )
    assert out == ([], [], [], {"mean_delta": None, "n_pairs": 0})


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
    data_start = block.index('id="dashboard-multimetric-data"') + len('id="dashboard-multimetric-data">')
    data_end = block.index("</script>", data_start)
    embedded = json.loads(block[data_start:data_end])
    assert embedded == payload  # exact sorted-key serialization round-trips
    assert "var payload = {" not in block  # payload no longer a JS literal
    assert block.count("<option") == 7
    assert "Cumulative View" in block and "Standard View" in block
    assert "METRIC_CONFIG" in block  # app.js inlined
    assert "Cumulative Wealth (1.0 Stake)" in block and "Per-Era Net Return" in block
    # hover values carry a per-metric format matching the axis tickformat (M3)
    assert "hoverformat" in block
    assert 'hovertemplate: "%{y:"' in block
    assert "updatemenus" not in block
    assert "<script src" not in block


def test_multimetric_chart_html_empty_payload_annotation() -> None:
    block = charts.multimetric_chart_html(
        {"eras": [], "meta_downside_mask": [], "metrics": {}, "drawdowns": {}}
    )
    assert "Timeseries data unavailable without local v5.3 assets" in block
    assert "Plotly" not in block  # no chart is even mounted


def test_similarity_chart_heatmap_and_highlight() -> None:
    fig = charts.build_similarity_matrix_chart(
        ["top", "second"], [[1.0, 0.7], [0.7, 1.0]]
    )
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert list(trace.z[0]) == [1.0, 0.7]
    assert "<b>" in trace.text[0][0]      # row/col 0 highlight (decision #25)
    assert "<b>" in trace.text[1][0]
    # brief asserts "RdBu_r" in str(trace.colorscale); plotly 6.x expands named
    # colorscales server-side (the name is never stored), so assert the exact
    # expansion of RdBu_r plus the diverging midpoint instead
    assert trace.zmid == 0.5
    assert [color for _, color in trace.colorscale] == diverging.RdBu_r


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
    assert fig.layout.legend.orientation == "h"   # horizontal legend restored


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
    assert generate_dashboard._ensemble_sharpe({}) is None           # no series
    assert generate_dashboard._ensemble_sharpe({"a": {"standard": [0.01, 0.02]}}) is None
    # decision #27: N_fleet < 3 renders "—", even with 2 usable series
    assert generate_dashboard._ensemble_sharpe({
        "a": {"standard": [0.01, 0.02, 0.03]},
        "b": {"standard": [0.02, 0.01, 0.02]},
    }) is None
    value = generate_dashboard._ensemble_sharpe({
        "a": {"standard": [0.01, 0.02, 0.03]},
        "b": {"standard": [0.02, 0.01, 0.02]},
        "c": {"standard": [0.01, 0.01, 0.02]},
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
        **_lb_row("a" * 64, "trained", "brb1-xgb-v6", 0.5, has_full=True),
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


def test_multimetric_chart_embeds_data_node_and_app_js_once() -> None:
    payload = {
        "eras": ["0001", "0002"],
        "meta_downside_mask": [False, False],
        "metrics": {"payout": {"a": {"standard": [0.01, 0.02],
                                     "cumulative": [1.01, 1.0302], "label": "r"}}},
        "drawdowns": {"a": [0.0, 0.0]},
    }
    block = charts.multimetric_chart_html(payload)
    assert block.count('id="dashboard-multimetric-data"') == 1
    assert block.count("<script") == 2  # data node + app.js
    assert block.count("var METRIC_CONFIG = {") == 1  # app.js inlined exactly once
    # a marker from the controller body is present (dataNode read)
    assert 'getElementById("dashboard-multimetric-data")' in block
