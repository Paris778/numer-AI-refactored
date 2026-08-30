from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import nmr.dashboard as dash
import nmr.evaluation as nmr_evaluation
import nmr.payout as payout
from dashboard_ui import report as generate_dashboard
from nmr import lifecycle, paths
from nmr.config import REPO_ROOT
from nmr.deployment import serialize_predict
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
    assert (
        dash.resolve_benchmark_path(reports / "nope.csv", reports_dir=reports) == smoke
    )
    smoke.unlink()
    assert (
        dash.resolve_benchmark_path(reports / "nope.csv", reports_dir=reports) is None
    )


def test_resolve_benchmark_path_false_disables_chain(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "benchmark_hierarchy_scorecard_smoke.csv").write_text(
        "x", encoding="utf-8"
    )
    assert dash.resolve_benchmark_path(False, reports_dir=reports) is None


def test_unified_schema_contains_leaderboard_projection() -> None:
    # dashboard_app._LEADERBOARD_SCHEMA is a subset the app wrapper projects onto.
    required = {
        "model_id",
        "source",
        "run_name",
        "backend",
        "preset",
        "feature_set",
        "feature_subset",
        "n_targets",
        "targets",
        "neutralization_proportion",
        "oof_device",
        "corr",
        "corr_ci_low",
        "corr_ci_high",
        "corr_sharpe_ac",
        "corr_sharpe_ac_ci_low",
        "corr_sharpe_ac_ci_high",
        "max_drawdown",
        "std_corr",
        "deflated_sharpe",
        "max_feature_exposure",
        "has_bmc",
        "has_horizon",
        "has_perturb",
        "has_regime",
        "run_dir",
    }
    assert required <= set(dash.UNIFIED_SCHEMA.names())
    # capital cells required by the executive table
    for col in (
        "cagr_1y",
        "gain_to_pain_ratio",
        "kelly_fraction",
        "mmc_down",
        "fnc",
        "mmc",
        "mean_payout",
        "n_eras",
        "tier",
        "turnover_mean",
    ):
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
        "model_id,corr,corr_sharpe_ac,strategy_group\n" "bench_a,0.05,0.5,linear\n",
    )
    row = dash.load_benchmark_frame(minimal).row(0, named=True)
    assert row["corr_sharpe_ac_ci_low"] is None
    assert row["std_corr"] is None
    assert row["max_drawdown"] is None
    assert row["fnc"] is None
    assert row["has_bmc"] is False


def test_load_benchmark_frame_missing_file_returns_empty_schema_frame(
    tmp_path: Path,
) -> None:
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
                "data": {
                    "feature_set": "small",
                    "feature_subset": None,
                    "targets": ["target"],
                },
                "model": {"backend": "lightgbm", "preset": "fast"},
                "risk": {"neutralization_proportion": 1.0},
                "run": {"name": "sample-run"},
            },
        },
        "scorecard": (
            None
            if not scorecard
            else {
                "corr": 0.12,
                "corr_ci_low": 0.05,
                "corr_ci_high": 0.19,
                "corr_n_eras": 30,
                "corr_sharpe_ac": 0.8,
                "corr_sharpe_ac_ci_low": 0.6,
                "corr_sharpe_ac_ci_high": 1.0,
                "max_drawdown": 0.1,
                "std_corr": 0.2,
                "deflated_sharpe": 0.97,
                "max_feature_exposure": 0.3,
                "bmc": 0.02,
                "fnc": 0.05,
                "n_eras": 30,
                "cagr_1y": 1.5,
                "gain_to_pain_ratio": 2.0,
                "kelly_fraction": 0.4,
                "mmc_down": 0.01,
                "mmc_down_reason": None,
            }
        ),
    }
    return entry


def _write_registry(tmp_path: Path, entries: list[dict]) -> None:
    for entry in entries:
        run_dir = tmp_path / entry["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps(entry), encoding="utf-8")


def test_load_unified_leaderboard_registry_only(tmp_path: Path) -> None:
    _write_registry(
        tmp_path,
        [_registry_entry("a" * 64), _registry_entry("b" * 64, scorecard=False)],
    )
    frame = dash.load_unified_leaderboard(
        tmp_path, benchmark_path=False, models_dir=tmp_path / "models"
    )
    assert frame.height == 2
    rows = {r["model_id"]: r for r in frame.to_dicts()}
    assert rows["a" * 64]["source"] == "trained"
    assert rows["a" * 64]["corr"] == 0.12
    assert rows["b" * 64]["source"] == "trained_legacy"
    assert rows["b" * 64]["corr"] == 0.1  # legacy falls back to metrics.mean
    assert rows["a" * 64]["cagr_1y"] == 1.5  # stored capital block carried through

    incomplete = _registry_entry("c" * 64, scorecard=False)
    incomplete["metrics"] = {}
    incomplete_dir = tmp_path / "incomplete"
    _write_registry(incomplete_dir, [incomplete])
    incomplete_frame = dash.load_unified_leaderboard(
        incomplete_dir, benchmark_path=False, models_dir=incomplete_dir / "models"
    )
    incomplete_row = incomplete_frame.row(0, named=True)
    assert incomplete_row["corr"] is None
    assert incomplete_row["corr_sharpe_ac"] is None
    assert incomplete_row["std_corr"] is None
    assert incomplete_row["max_drawdown"] is None


def test_load_unified_leaderboard_zero_scorecard_value_not_legacy(
    tmp_path: Path,
) -> None:
    entry = _registry_entry("c" * 64)
    entry["scorecard"]["corr"] = 0.0  # legitimate 0.0 must NOT fall through
    _write_registry(tmp_path, [entry])
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    assert frame.row(0, named=True)["corr"] == 0.0
    assert frame.row(0, named=True)["source"] == "trained"


def test_load_unified_leaderboard_exposure_definition_boundary(tmp_path: Path) -> None:
    """SEV-1 #14 boundary: legacy rows (no scorecard_prediction_scale marker)
    measured exposure on unranked preds (~machine epsilon) and must not sit
    silently beside post-fix rows — nulled with a documented reason."""
    legacy = _registry_entry("a" * 64)
    postfix = _registry_entry("b" * 64)
    postfix["manifest"]["scorecard_prediction_scale"] = "percentile_rank"
    _write_registry(tmp_path, [legacy, postfix])
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    rows = {r["model_id"]: r for r in frame.to_dicts()}
    assert rows["a" * 64]["max_feature_exposure"] is None
    assert rows["a" * 64]["max_feature_exposure_reason"] == "pre_rank_fix_definition"
    assert rows["b" * 64]["max_feature_exposure"] == 0.3
    assert rows["b" * 64]["max_feature_exposure_reason"] is None


def test_load_unified_leaderboard_corrupt_run_json_skipped(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("d" * 64)])
    bad_dir = tmp_path / ("e" * 64)
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "run.json").write_text("{not json", encoding="utf-8")
    frame = dash.load_unified_leaderboard(
        tmp_path, benchmark_path=False, models_dir=tmp_path / "models"
    )
    assert frame.height == 1


def test_load_unified_leaderboard_merges_benchmarks(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("f" * 64)])
    bench = _write_benchmark_csv(
        tmp_path,
        "model_id,corr,corr_sharpe_ac,std_corr,max_drawdown,strategy_group,tier\n"
        "bench_a,0.05,0.5,0.3,0.2,linear,1\n",
    )
    frame = dash.load_unified_leaderboard(
        tmp_path, benchmark_path=bench, models_dir=tmp_path / "models"
    )
    assert frame.height == 2
    assert set(frame.get_column("source").to_list()) == {"trained", "benchmark"}
    bench_row = frame.filter(pl.col("source") == "benchmark").row(0, named=True)
    assert bench_row["family"] is None
    assert bench_row["training_scope"] is None
    assert bench_row["has_full_version"] is False
    assert bench_row["max_feature_exposure_reason"] == "unranked_predictions"


def test_load_unified_leaderboard_empty_registry_returns_schema_frame(
    tmp_path: Path,
) -> None:
    frame = dash.load_unified_leaderboard(
        tmp_path, benchmark_path=False, models_dir=tmp_path / "models"
    )
    assert frame.height == 0
    assert frame.schema == dash.UNIFIED_SCHEMA


_GATE_YAML = REPO_ROOT / "configs" / "benchmarks" / "tier4_gate.yaml"


def _status_frame(tmp_path: Path, rows: list[dict]) -> pl.DataFrame:
    frame = pl.DataFrame(rows, schema=dash.UNIFIED_SCHEMA, strict=False)
    return dash.evaluate_gate_status(frame, _GATE_YAML, tmp_path / "champion.json")


def test_gate_status_research_and_capital_ready(tmp_path: Path) -> None:
    base = {
        "model_id": "r1",
        "source": "trained",
        "corr": 0.01,
        "corr_sharpe_ac": 0.2,
        "fnc": 0.001,
        "deflated_sharpe": 0.5,
        "gain_to_pain_ratio": 1.0,
        "cagr_1y": 0.1,
        "turnover_mean": None,
    }
    frame = _status_frame(tmp_path, [base])
    assert frame.row(0, named=True)["status"] == "RESEARCH"
    assert frame.row(0, named=True)["gate_corr"] is False

    passing = dict(base)
    passing.update(
        {
            "model_id": "r2",
            "corr": 0.03,
            "corr_sharpe_ac": 0.8,
            "fnc": 0.03,
            "deflated_sharpe": 0.96,
            "gain_to_pain_ratio": 1.6,
            "cagr_1y": 0.01,
        }
    )
    frame = _status_frame(tmp_path, [passing])
    assert frame.row(0, named=True)["status"] == "CAPITAL READY"
    assert frame.row(0, named=True)["gate_corr"] is True
    assert frame.row(0, named=True)["gate_cagr_1y"] is True  # strict > 0.0


def test_gate_status_champion_via_pointer(tmp_path: Path) -> None:
    (tmp_path / "champion.json").write_text(
        json.dumps({"run_id": "ch" * 32}), encoding="utf-8"
    )
    frame = _status_frame(
        tmp_path,
        [
            {
                "model_id": "ch" * 32,
                "source": "trained",
                "corr": 0.0,
                "corr_sharpe_ac": 0.0,
                "fnc": 0.0,
                "deflated_sharpe": 0.0,
                "gain_to_pain_ratio": 0.0,
                "cagr_1y": 0.0,
                "turnover_mean": None,
            }
        ],
    )
    assert frame.row(0, named=True)["status"] == "RESEARCH"


def test_gate_status_champion_requires_pointer_and_capital_gate(tmp_path: Path) -> None:
    champion_id = "ch" * 32
    (tmp_path / "champion.json").write_text(
        json.dumps({"run_id": champion_id}), encoding="utf-8"
    )
    frame = _status_frame(
        tmp_path,
        [
            {
                "model_id": champion_id,
                "source": "trained",
                "corr": 0.03,
                "corr_sharpe_ac": 0.8,
                "fnc": 0.03,
                "deflated_sharpe": 0.96,
                "gain_to_pain_ratio": 1.6,
                "cagr_1y": 0.01,
                "turnover_mean": None,
            }
        ],
    )
    assert frame.row(0, named=True)["status"] == "CHAMPION"


def test_gate_status_benchmark_rows_never_capital_ready(tmp_path: Path) -> None:
    ref = {
        "model_id": "v53_lgbm_ender60",
        "source": "benchmark",
        "corr": 0.029,
        "corr_sharpe_ac": 0.78,
        "fnc": 0.027,
        "deflated_sharpe": 1.0,
        "gain_to_pain_ratio": 44.0,
        "cagr_1y": 4.88,
        "turnover_mean": None,
    }
    other = dict(ref, model_id="null_constant_05", corr=0.0, corr_sharpe_ac=0.0)
    frame = _status_frame(tmp_path, [ref, other])
    statuses = {r["model_id"]: r["status"] for r in frame.to_dicts()}
    assert statuses["v53_lgbm_ender60"] == "GATE HURDLE"
    assert statuses["null_constant_05"] == "BENCHMARK"
    ref_row = frame.filter(pl.col("model_id") == "v53_lgbm_ender60").row(0, named=True)
    assert ref_row["gate_corr_sharpe_ac"] is True
    assert ref_row["gate_turnover_mean"] is None  # turnover absent -> exempt


def test_gate_status_turnover_violation_when_present(tmp_path: Path) -> None:
    row = {
        "model_id": "r3",
        "source": "trained",
        "corr": 0.03,
        "corr_sharpe_ac": 0.8,
        "fnc": 0.03,
        "deflated_sharpe": 0.96,
        "gain_to_pain_ratio": 1.6,
        "cagr_1y": 0.01,
        "turnover_mean": 0.9,
    }
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
    meta = targets.select(["era", "id", pl.col("target").alias("numerai_meta_model")])
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
    assert row["cagr_1y"] == pytest.approx((1.05**52) - 1.0, abs=1e-6)
    assert row["gain_to_pain_ratio"] == float("inf")  # no losing eras


def test_reconcile_capital_metrics_stored_block_wins(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("b" * 64)])
    _write_preds(tmp_path / ("b" * 64), scale=0.0)  # junk preds must be ignored
    data = _synthetic_data_dir(tmp_path)
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    out = dash.reconcile_capital_metrics(frame, data)
    row = out.row(0, named=True)
    assert row["cagr_1y"] == 1.5  # stored value untouched
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
        "payout",
        "corr20",
        "mmc20",
        "corr60",
        "mmc60",
        "bmc",
        "cwmm",
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
    assert payload["drawdowns"]["a" * 64][0] == pytest.approx(
        wealth[0] / peak - 1.0, abs=1e-12
    )


def test_multimetric_renders_both_tier4_reference_columns(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64)])
    _write_preds(tmp_path / ("a" * 64), scale=1.0)
    data = _synthetic_v2_data_dir(tmp_path)
    # extend the fixture benchmark parquet with the second official column
    rows = []
    for era in ("0001", "0002", "0003"):
        for i in range(10):
            rows.append(
                {
                    "era": era,
                    "id": f"{era}_{i:03d}",
                    "v53_lgbm_ender60": 0.5 * float(i),
                    "v53_lgbm_ender20": 0.25 * float(i),
                }
            )
    pl.DataFrame(rows).write_parquet(data / "validation_benchmark_models.parquet")

    payload = dash.extract_multimetric_timeseries(
        tmp_path, data, run_ids=["a" * 64], include_tier4_ref=True
    )
    # both official benchmark models render as reference curves
    for name in ("payout", "corr20", "mmc20", "corr60", "mmc60", "cwmm"):
        assert "v53_lgbm_ender60" in payload["metrics"][name]
        assert "v53_lgbm_ender20" in payload["metrics"][name]
    # every tier-4 reference BMC slice is short-circuited to zeros (decision #11)
    for ref in ("v53_lgbm_ender60", "v53_lgbm_ender20"):
        bmc = payload["metrics"]["bmc"][ref]
        assert bmc["standard"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    # the trained run still gets a BMC slice keyed against the primary
    # (its values are degenerate here — the perfect-signal fixture leaves zero
    # residual variance after target orthogonalization — so presence is the
    # contract, not magnitude)
    assert "a" * 64 in payload["metrics"]["bmc"]


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
    assert payload == {
        "eras": [],
        "meta_downside_mask": [],
        "metrics": {},
        "drawdowns": {},
    }


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
    assert len(payout["standard"]) == 3  # aligned 1:1 with the axis
    assert payout["standard"][1] is None  # missing era remains unavailable
    assert payout["cumulative"] == pytest.approx([1.05, None, None])
    assert payload["drawdowns"]["e" * 64] == [0.0, None, None]
    assert len(payload["drawdowns"]["e" * 64]) == 3
    for name in ("corr20", "mmc20", "corr60", "mmc60", "bmc", "cwmm"):
        assert len(payload["metrics"][name]["e" * 64]["standard"]) == 3


def test_multimetric_reference_bmc_preserves_missing_eras(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("f" * 64)])
    data = _synthetic_v2_data_dir(tmp_path)
    benchmark_path = data / "validation_benchmark_models.parquet"
    benchmarks = pl.read_parquet(benchmark_path).filter(pl.col("era") != "0002")
    benchmarks.write_parquet(benchmark_path)

    payload = dash.extract_multimetric_timeseries(
        tmp_path,
        data,
        run_ids=[],
        include_tier4_ref=True,
        tier4_columns=("v53_lgbm_ender60",),
    )

    assert payload["metrics"]["bmc"]["v53_lgbm_ender60"]["standard"] == [
        0.0,
        None,
        0.0,
    ]


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
    # renders, with BMC unavailable (degrade, never raise — decision #23)
    assert set(payload["metrics"]) == {
        "payout",
        "corr20",
        "mmc20",
        "corr60",
        "mmc60",
        "bmc",
        "cwmm",
    }
    assert set(payload["metrics"]["payout"]) == {"a" * 64}
    for name in ("payout", "corr20", "mmc20", "corr60", "mmc60", "bmc", "cwmm"):
        series = payload["metrics"][name]["a" * 64]
        assert len(series["standard"]) == 3
        assert len(series["cumulative"]) == 3
    assert payload["metrics"]["bmc"]["a" * 64]["standard"] == [None, None, None]
    assert payload["metrics"]["bmc"]["a" * 64]["cumulative"] == [None, None, None]
    assert "a" * 64 in payload["drawdowns"]
    assert "bmc unavailable" in caplog.text


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


@pytest.mark.skipif(
    not _HAS_REAL, reason="real registry/v5.3 data absent; skipped in CI"
)
def test_real_recompute_matches_stored_corr() -> None:
    frame = dash.load_unified_leaderboard(_REAL_REGISTRY, benchmark_path=False)
    row = frame.sort("corr_sharpe_ac", descending=True, nulls_last=True).row(
        0, named=True
    )
    lookups = dash._load_shared_lookups(Path("data/v5.3"))
    assert lookups is not None
    targets_86, meta, _ = lookups
    preds_path = Path(row["run_dir"]) / "validation_preds.parquet"
    corr, _, _ = dash._per_era_metrics(preds_path, targets_86, meta)
    assert len(corr) == row["corr_n_eras"]
    assert float(np.mean(list(corr.values()))) == pytest.approx(row["corr"], abs=1e-4)


@pytest.mark.skipif(
    not (_HAS_REAL and _SMOKE_CSV.exists()),
    reason="real v5.3 data + smoke benchmark CSV absent; skipped in CI",
)
def test_real_tier4_cagr_matches_smoke_csv() -> None:
    lookups = dash._load_shared_lookups(Path("data/v5.3"))
    assert lookups is not None
    targets_86, meta, _ = lookups
    bench = pl.read_parquet(_REAL_BENCH, columns=["era", "id", "v53_lgbm_ender60"])
    axis = sorted(meta.get_column("era").unique().to_list(), key=int)
    joined = (
        bench.filter(pl.col("era").is_in(axis))
        .join(targets_86, on=["era", "id"], how="inner")
        .join(meta, on=["era", "id"], how="inner")
    )
    engine = (
        nmr_evaluation.EvaluationEngine()
    )  # import nmr.evaluation as nmr_evaluation at top
    corr = engine.per_era_corr(joined, pred_col="v53_lgbm_ender60", target_col="target")
    mmc = engine.per_era_mmc(
        joined,
        pred_col="v53_lgbm_ender60",
        meta_col="numerai_meta_model",
        target_col="target",
    )
    pay = payout.payout_series(corr, mmc)  # import nmr.payout as payout at top
    recomputed = payout.annual_compounded_return(pay.clipped)
    stored = (
        dash.load_benchmark_frame(_SMOKE_CSV)
        .filter(pl.col("model_id") == "v53_lgbm_ender60")
        .row(0, named=True)["cagr_1y"]
    )
    assert float(recomputed) == pytest.approx(float(stored), abs=1e-6)


@pytest.mark.skipif(
    not _HAS_REAL, reason="real registry/v5.3 data absent; skipped in CI"
)
def test_real_reconcile_populates_all_capital_columns() -> None:
    frame = dash.load_unified_leaderboard(_REAL_REGISTRY, benchmark_path=False)
    out = dash.reconcile_capital_metrics(frame, Path("data/v5.3"))
    trained = out.filter(pl.col("source").is_in(["trained", "trained_legacy"]))
    assert trained.height > 0
    for row in trained.to_dicts():
        assert row["cagr_1y"] is not None
        assert row["gain_to_pain_ratio"] is not None
        assert row["kelly_fraction"] is not None


@pytest.mark.skipif(
    not _HAS_REAL, reason="real registry/v5.3 data absent; skipped in CI"
)
def test_real_multimetric_payload_and_payout_parity() -> None:
    frame = dash.load_unified_leaderboard(_REAL_REGISTRY, benchmark_path=False)
    top = frame.sort("corr_sharpe_ac", descending=True, nulls_last=True).row(
        0, named=True
    )
    payload = dash.extract_multimetric_timeseries(
        _REAL_REGISTRY,
        Path("data/v5.3"),
        run_ids=[top["model_id"]],
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


@pytest.mark.skipif(
    not _HAS_REAL, reason="real registry/v5.3 data absent; skipped in CI"
)
def test_real_similarity_matrix_top5_with_tier4() -> None:
    frame = dash.load_unified_leaderboard(_REAL_REGISTRY, benchmark_path=False)
    top5 = (
        frame.sort("corr_sharpe_ac", descending=True, nulls_last=True)
        .head(5)
        .get_column("model_id")
        .to_list()
    )
    _labels, ids, matrix, stress = dash.extract_pairwise_similarity_matrix(
        _REAL_REGISTRY,
        Path("data/v5.3"),
        run_ids=top5,
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


def test_kpi_cards_stale_champion_pointer_degrades(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = pl.DataFrame(
        [
            {
                "model_id": "a" * 64,
                "source": "trained",
                "run_name": "sample-run",
                "corr_sharpe_ac": 0.8,
                "cagr_1y": 1.5,
                "max_drawdown": 0.1,
                "status": "RESEARCH",
                "n_eras": 30,
            }
        ]
    )
    with caplog.at_level(logging.WARNING, logger="generate_dashboard"):
        kpis = generate_dashboard._kpi_cards(
            frame, champion="9" * 64, hurdle_sharpe=0.78
        )
    assert kpis["champion_label"] == "None Designated"
    assert kpis["champion_detail"] == "(Unallocated)"
    assert "champion" in caplog.text and "not found in leaderboard" in caplog.text


def _synthetic_v2_data_dir(tmp_path: Path, *, with_benchmark: bool = True) -> Path:
    data = _synthetic_data_dir(tmp_path)  # era/id/target + meta over 0001..0003
    if with_benchmark:
        rows = []
        for era in ("0001", "0002", "0003"):
            for i in range(10):
                rows.append(
                    {
                        "era": era,
                        "id": f"{era}_{i:03d}",
                        "v53_lgbm_ender60": 0.5 * float(i),
                    }
                )
        pl.DataFrame(rows).write_parquet(data / "validation_benchmark_models.parquet")
    return data


def test_resolve_horizon_targets_fallback_chain() -> None:
    assert dash._resolve_horizon_targets(
        ["era", "target_ender_20", "target_ender_60"]
    ) == ("target_ender_20", "target_ender_60")
    assert dash._resolve_horizon_targets(["target_cyrusd_20", "target_cyrusd_60"]) == (
        "target_cyrusd_20",
        "target_cyrusd_60",
    )
    # both horizons collapse to the generic target when nothing else exists
    assert dash._resolve_horizon_targets(["target"]) == ("target", "target")


def test_load_v2_lookups_deduped_target_columns(tmp_path: Path) -> None:
    data = _synthetic_v2_data_dir(tmp_path)
    lookups = dash._load_v2_lookups(data, tier4_columns=("v53_lgbm_ender60",))
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
    assert (
        dash._load_v2_lookups(tmp_path / "no-data", tier4_columns=("v53_lgbm_ender60",))
        is None
    )


def test_similarity_matrix_identity_symmetry_and_clamp(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64), _registry_entry("b" * 64)])
    _write_preds(tmp_path / ("a" * 64), scale=1.0)
    _write_preds(tmp_path / ("b" * 64), scale=2.0)  # scale-shifted copy of a
    data = _synthetic_v2_data_dir(tmp_path)
    labels, ids, matrix, stress = dash.extract_pairwise_similarity_matrix(
        tmp_path, data, run_ids=["b" * 64, "a" * 64], include_tier4_ref=False
    )
    assert ids == ["a" * 64, "b" * 64]  # sorted deterministically
    assert matrix[0][0] == 1.0 and matrix[1][1] == 1.0
    assert matrix[0][1] == pytest.approx(
        1.0, abs=1e-9
    )  # rank-gaussian: scale-invariant
    assert matrix[0][1] == pytest.approx(matrix[1][0], abs=1e-12)  # symmetric
    assert all(-1.0 <= v <= 1.0 for row in matrix for v in row)  # clamped
    assert set(stress) == {"mean_delta", "n_pairs"}


def test_similarity_matrix_includes_tier4_from_benchmark_parquet(
    tmp_path: Path,
) -> None:
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
            rows.append(
                {
                    "era": era,
                    "id": f"{era}_{i:03d}",
                    "numerai_meta_model": sign * float(i),
                }
            )
    pl.DataFrame(rows).write_parquet(meta_path)
    pl.DataFrame(
        [
            {"era": era, "id": f"{era}_{i:03d}", "v53_lgbm_ender60": float(i)}
            for era in ("0001", "0002", "0003")
            for i in range(10)
        ]
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
    assert generate_dashboard._ensemble_sharpe({}) is None  # no series
    assert (
        generate_dashboard._ensemble_sharpe({"a": {"standard": [0.01, 0.02]}}) is None
    )
    # decision #27: N_fleet < 3 renders "—", even with 2 usable series
    assert (
        generate_dashboard._ensemble_sharpe(
            {
                "a": {"standard": [0.01, 0.02, 0.03]},
                "b": {"standard": [0.02, 0.01, 0.02]},
            }
        )
        is None
    )
    value = generate_dashboard._ensemble_sharpe(
        {
            "a": {"standard": [0.01, 0.02, 0.03]},
            "b": {"standard": [0.02, 0.01, 0.02]},
            "c": {"standard": [0.01, 0.01, 0.02]},
        }
    )
    assert isinstance(value, float) and value == value  # finite


def test_unified_schema_has_family_columns() -> None:
    for col in ("family", "training_scope", "has_full_version"):
        assert col in dash.UNIFIED_SCHEMA.names()


def _write_export_slot(
    root: Path,
    family: str,
    scope: str,
    run_id: str,
    *,
    config: dict | None = None,
    scorecard: dict | None = None,
    write_run_record: bool = True,
) -> Path:
    """Write a VALID export slot at ``root/<family>/exports/<scope>/<run_id>``.

    Validity (``lifecycle.valid_export``): export.json identity binding plus a
    hash-verified loadable predict.pkl — ``serialize_predict`` writes the
    sha256 sibling manifest so the loadability predicate holds. Partial slots
    require ``scorecard.json``; ``scorecard`` supplies the cross-check record
    (defaults to an empty schema stub whose missing ``scorecard`` block leaves
    the unified metric cells null). A run record is written by default
    (identity binding, 2026-08-26 review BLOCKING 2) — pass
    ``write_run_record=False`` to construct an orphan.
    """
    slot = root / family / "exports" / scope / run_id
    slot.mkdir(parents=True, exist_ok=True)

    def dummy_predict(live_features, live_benchmark_models=None):
        return live_features

    serialize_predict(dummy_predict, path=slot / "predict.pkl", feature_names=["f1"])
    payload = {
        "family": family,
        "training_scope": scope,
        "promoted_from_run_id": run_id,
        "promoted_at": "2026-08-17T12:00:00Z",
        "config": config
        or {
            "run": {"name": family},
            "data": {
                "feature_set": "all",
                "feature_subset": "medium",
                "targets": ["target"],
            },
            "model": {"backend": "xgboost", "preset": "fast"},
        },
    }
    (slot / "export.json").write_text(json.dumps(payload), encoding="utf-8")
    if scope == "partial":
        (slot / "scorecard.json").write_text(
            json.dumps(
                scorecard if scorecard is not None else {"schema_version": 3},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    if write_run_record:
        _write_experiments_run(
            root,
            family,
            {
                "run_id": run_id,
                "manifest": {"config": {"run": {"name": family}}},
                "scorecard": {},
            },
        )
    return slot


def _write_family_meta(
    root: Path,
    family: str,
    *,
    display_name: str | None = None,
    staked: dict | None = None,
) -> None:
    meta = root / family / "meta.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(
        json.dumps({"display_name": display_name or family, "staked": staked}),
        encoding="utf-8",
    )


def _write_current_pointer(root: Path, family: str, run_id: str) -> None:
    pointer = root / family / "exports" / "full" / "current.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        json.dumps({"run_id": run_id, "promoted_at": "2026-08-17T12:00:00Z"}),
        encoding="utf-8",
    )


def _write_experiments_run(root: Path, family: str, entry: dict) -> None:
    run_dir = root / family / "runs" / entry["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(entry), encoding="utf-8")


def _full_export_config(family: str) -> dict:
    return {
        "run": {"name": family},
        "data": {
            "feature_set": "all",
            "feature_subset": "medium",
            "targets": ["target"],
        },
        "model": {"backend": "xgboost", "preset": "fast"},
    }


def test_load_unified_leaderboard_family_columns_and_full_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry"
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", experiments)
    entry = _registry_entry("a" * 64)
    entry["manifest"]["config"]["run"]["name"] = "brb1-xgb-v6"
    _write_registry(registry, [entry])
    _write_family_meta(experiments, "brb1-xgb-v6", display_name="Brb1 XGB v6")
    _write_export_slot(experiments, "brb1-xgb-v6", "full", "a" * 64)
    _write_current_pointer(experiments, "brb1-xgb-v6", "a" * 64)
    frame = dash.load_unified_leaderboard(registry, benchmark_path=False)
    rows = {r["model_id"]: r for r in frame.to_dicts()}
    trained = rows["a" * 64]
    assert trained["family"] == "brb1-xgb-v6"
    assert trained["training_scope"] == "research"
    assert trained["has_full_version"] is True
    assert trained["display_name"] == "Brb1 XGB v6"
    assert trained["lifecycle_stage"] == "full"
    assert trained["current_full_status"] == "full"
    full_id = "brb1-xgb-v6::full::" + "a" * 64
    full = rows[full_id]
    assert full["source"] == "full"
    assert full["run_name"] == "brb1-xgb-v6"
    assert full["training_scope"] == "full"
    assert full["has_full_version"] is False
    assert full["corr"] is None
    assert full["corr_sharpe_ac"] is None
    assert full["backend"] == "xgboost"
    assert full["feature_subset"] == "medium"
    assert full["display_name"] == "Brb1 XGB v6"
    # experiment layout: the full row's run_dir is the immutable export slot.
    assert full["run_dir"] == str(
        experiments / "brb1-xgb-v6" / "exports" / "full" / ("a" * 64)
    )


def test_load_unified_leaderboard_scan_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry"
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", experiments)
    _write_registry(registry, [_registry_entry("a" * 64)])
    _write_family_meta(experiments, "sample-run")
    _write_export_slot(experiments, "sample-run", "full", "a" * 64)
    _write_export_slot(experiments, "sample-run", "full", "b" * 64)
    _write_export_slot(experiments, "sample-run", "partial", "a" * 64)
    calls: list[tuple[str, str]] = []
    real_scan = dash.lifecycle.scan_valid_exports

    def counting_scan(family: str, scope: str):
        calls.append((family, scope))
        return real_scan(family, scope)

    monkeypatch.setattr(dash.lifecycle, "scan_valid_exports", counting_scan)
    dash.load_unified_leaderboard(registry, benchmark_path=False, models_dir=experiments)
    # Per (family, scope) the dashboard scans once; lifecycle internals
    # (current_full_status / derive_stage) add one more each — never per slot.
    assert calls.count(("sample-run", "full")) == 2
    assert calls.count(("sample-run", "partial")) == 2


def test_load_unified_leaderboard_missing_models_dir(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64)])
    frame = dash.load_unified_leaderboard(
        tmp_path, benchmark_path=False, models_dir=tmp_path / "nope"
    )
    assert frame.height == 1
    assert frame.row(0, named=True)["has_full_version"] is False


def test_load_unified_leaderboard_orphan_export_rejected(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BLOCKING 2: an export whose run_id has no run record is an orphan —
    rejected upstream by lifecycle.valid_export, never render-valid (the old
    behavior warned but still rendered)."""
    registry = tmp_path / "registry"
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", experiments)
    _write_registry(registry, [_registry_entry("a" * 64)])
    _write_family_meta(experiments, "orphan-family")
    _write_export_slot(
        experiments, "orphan-family", "full", "f" * 64, write_run_record=False
    )
    frame = dash.load_unified_leaderboard(registry, benchmark_path=False)
    rows = {r["model_id"]: r for r in frame.to_dicts()}
    assert "orphan-family::full::" + "f" * 64 not in rows  # rejected, not rendered
    # The family has no valid exports -> the orphan slot does not produce a
    # degraded/full row; it simply does not exist in the unified frame.
    assert "orphan-family" not in " ".join(rows.keys())


def test_unified_schema_has_lifecycle_columns() -> None:
    for col in ("display_name", "lifecycle_stage", "current_full_status", "stale"):
        assert col in dash.UNIFIED_SCHEMA.names()


def test_load_unified_leaderboard_reads_experiments_run_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry"
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", experiments)
    entry = _registry_entry("c" * 64)
    entry["manifest"]["config"]["run"]["name"] = "fam-exp"
    _write_experiments_run(experiments, "fam-exp", entry)
    frame = dash.load_unified_leaderboard(
        registry, benchmark_path=False, models_dir=experiments
    )
    rows = {r["model_id"]: r for r in frame.to_dicts()}
    assert "c" * 64 in rows
    assert rows["c" * 64]["source"] == "trained"
    assert rows["c" * 64]["family"] == "fam-exp"
    assert rows["c" * 64]["run_dir"] == str(
        experiments / "fam-exp" / "runs" / ("c" * 64)
    )
    assert rows["c" * 64]["lifecycle_stage"] == "research"


def test_load_unified_leaderboard_full_row_export_version_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Landmine regression guard: full rows read ExportVersion fields
    (run_id / slot_dir), never the removed promoted_from_run_id / manifest_path."""
    registry = tmp_path / "registry"
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", experiments)
    _write_registry(registry, [_registry_entry("a" * 64)])
    _write_family_meta(experiments, "sample-run")
    _write_export_slot(experiments, "sample-run", "full", "a" * 64)
    _write_current_pointer(experiments, "sample-run", "a" * 64)
    frame = dash.load_unified_leaderboard(registry, benchmark_path=False)
    full_id = "sample-run::full::" + "a" * 64
    full = {r["model_id"]: r for r in frame.to_dicts()}[full_id]
    assert full["run_dir"] == str(experiments / "sample-run" / "exports" / "full" / ("a" * 64))
    assert full["feature_set"] == "all"
    assert full["n_targets"] == 1
    assert full["targets"] == "target"


def test_load_unified_leaderboard_partial_rows_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry"
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", experiments)
    _write_registry(registry, [_registry_entry("a" * 64)])
    _write_family_meta(experiments, "sample-run")
    _write_export_slot(experiments, "sample-run", "partial", "a" * 64)
    frame = dash.load_unified_leaderboard(
        registry, benchmark_path=False, models_dir=experiments
    )
    rows = {r["model_id"]: r for r in frame.to_dicts()}
    partial = rows["sample-run::partial::" + "a" * 64]
    assert partial["source"] == "partial"
    assert partial["training_scope"] == "partial"
    assert partial["corr"] is None
    assert partial["lifecycle_stage"] == "partial"
    assert dash.dashboard_cohort(partial) == "partial"


def test_partial_rows_not_evaluable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry"
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", experiments)
    entry = _registry_entry("a" * 64)
    entry["scorecard"]["mmc"] = 0.2
    _write_registry(registry, [entry])
    _write_family_meta(experiments, "sample-run")
    _write_export_slot(experiments, "sample-run", "full", "a" * 64)
    _write_export_slot(experiments, "sample-run", "partial", "a" * 64)
    frame = dash.load_unified_leaderboard(
        registry, benchmark_path=False, models_dir=experiments
    )
    ranked = frame.filter(dash.EVALUABLE_ROWS)
    ids = ranked.get_column("model_id").to_list()
    assert "a" * 64 in ids
    assert not any("::partial::" in mid for mid in ids)
    assert not any("::full::" in mid for mid in ids)
    # rank_leaderboard renders diagnostic rows unranked (rank None) — the
    # evaluable path never gives them a rank
    ranked = dash.rank_leaderboard(frame)
    ranked_by_id = {r["model_id"]: r for r in ranked.to_dicts()}
    assert ranked_by_id["a" * 64]["rank"] is not None
    assert ranked_by_id["sample-run::full::" + "a" * 64]["rank"] is None
    assert ranked_by_id["sample-run::partial::" + "a" * 64]["rank"] is None


def _partial_scorecard() -> dict:
    return {
        "schema_version": 3,
        "scope": "partial",
        "scorecard": {
            "corr": {"value": 0.11, "ci_low": 0.04, "ci_high": 0.18, "n_eras": 22},
            "mmc": {"value": 0.09, "ci_low": -0.01, "ci_high": 0.19, "n_eras": 22},
            "corr_sharpe_ac": {
                "value": 0.62,
                "ci_low": 0.4,
                "ci_high": 0.84,
                "n_eras": 22,
            },
            "fnc": 0.021,
            "n_eras": 22,
            "deflated_sharpe": 0.91,
            "max_drawdown": 0.13,
            "burn_rate": 0.5,
        },
    }


def test_partial_row_carries_cross_check_metric_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §8 + §12 #15: a partial row's metric cells equal its slot's
    scorecard.json values, yet it stays diagnostic (never ranked)."""
    registry = tmp_path / "registry"
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", experiments)
    run_id = "a" * 64
    entry = _registry_entry(run_id)
    entry["scorecard"]["mmc"] = 0.2
    _write_registry(registry, [entry])
    _write_family_meta(experiments, "sample-run")
    _write_export_slot(
        experiments,
        "sample-run",
        "partial",
        run_id,
        scorecard=_partial_scorecard(),
    )
    frame = dash.load_unified_leaderboard(
        registry, benchmark_path=False, models_dir=experiments
    )
    rows = {r["model_id"]: r for r in frame.to_dicts()}
    partial = rows[f"sample-run::partial::{run_id}"]
    assert partial["source"] == "partial"
    assert partial["corr"] == pytest.approx(0.11)
    assert partial["corr_ci_low"] == pytest.approx(0.04)
    assert partial["corr_ci_high"] == pytest.approx(0.18)
    assert partial["corr_n_eras"] == 22
    assert partial["mmc"] == pytest.approx(0.09)
    # the mmc CI cells have no unified column — they render via the detail
    assert partial["corr_sharpe_ac"] == pytest.approx(0.62)
    assert partial["corr_sharpe_ac_ci_low"] == pytest.approx(0.4)
    assert partial["corr_sharpe_ac_ci_high"] == pytest.approx(0.84)
    assert partial["corr_sharpe_ac_n_eras"] == 22
    assert partial["fnc"] == pytest.approx(0.021)
    assert partial["n_eras"] == 22
    assert partial["deflated_sharpe"] == pytest.approx(0.91)
    assert partial["max_drawdown"] == pytest.approx(0.13)
    assert partial["burn_rate"] == pytest.approx(0.5)
    # cross-check metrics must never earn the partial row a rank
    ranked = dash.rank_leaderboard(frame)
    ranked_by_id = {r["model_id"]: r for r in ranked.to_dicts()}
    assert ranked_by_id[run_id]["rank"] == 1
    assert ranked_by_id[f"sample-run::partial::{run_id}"]["rank"] is None
    assert ranked_by_id[f"sample-run::partial::{run_id}"]["cohort"] == "partial"


def test_payload_carries_lifecycle_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry"
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", experiments)
    entry = _registry_entry("a" * 64)
    entry["manifest"]["config"]["run"]["name"] = "fam1"
    entry["scorecard"]["mmc"] = 0.2
    _write_registry(registry, [entry])
    _write_family_meta(experiments, "fam1", display_name="Fam One")
    _write_export_slot(experiments, "fam1", "full", "a" * 64)
    _write_export_slot(experiments, "fam1", "partial", "a" * 64)
    _write_current_pointer(experiments, "fam1", "a" * 64)
    frame = dash.load_unified_leaderboard(
        registry, benchmark_path=False, models_dir=experiments
    )
    payload = dash.build_tournament_payload(frame, evaluation_eras=["0001"])
    row_fields = payload["row_fields"]
    payload_rows = [
        {**dict(zip(row_fields, record)), "model_id": payload["model_ids"][index]}
        for index, record in enumerate(payload["rows"])
    ]
    family_rows = [r for r in payload_rows if r["family"] == "fam1"]
    assert family_rows
    for r in family_rows:
        assert r["lifecycle_stage"] in lifecycle.LIFECYCLE_STAGES
        assert r["display_name"] == "Fam One"  # from meta.json
    partial_rows = [r for r in payload_rows if "::partial::" in r["model_id"]]
    assert partial_rows and all(r["source"] == "partial" for r in partial_rows)
    assert all(r["model_id"].startswith("fam1::") for r in partial_rows)


def test_stale_staked_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = tmp_path / "registry"
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", experiments)
    _write_registry(registry, [_registry_entry("a" * 64)])
    staked = {
        "run_id": "dead" * 16,
        "scope": "full",
        "numerai_model_id": "m1",
        "staked_at": "2026-08-26T11:00:00+00:00",
        "status": "active",
    }
    _write_family_meta(experiments, "sample-run", staked=staked)
    _write_export_slot(experiments, "sample-run", "full", "a" * 64)
    _write_current_pointer(experiments, "sample-run", "a" * 64)
    frame = dash.load_unified_leaderboard(registry, benchmark_path=False)
    rows = {r["model_id"]: r for r in frame.to_dicts()}
    # active staked record referencing an invalid export => underlying stage + stale
    assert rows["a" * 64]["lifecycle_stage"] == "full"
    assert rows["a" * 64]["stale"] is True
    assert rows["sample-run::full::" + "a" * 64]["stale"] is True


def test_valid_staked_record_not_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = tmp_path / "registry"
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", experiments)
    _write_registry(registry, [_registry_entry("a" * 64)])
    staked = {
        "run_id": "a" * 64,
        "scope": "full",
        "numerai_model_id": "m1",
        "staked_at": "2026-08-26T11:00:00+00:00",
        "status": "active",
    }
    _write_family_meta(experiments, "sample-run", staked=staked)
    _write_export_slot(experiments, "sample-run", "full", "a" * 64)
    _write_current_pointer(experiments, "sample-run", "a" * 64)
    frame = dash.load_unified_leaderboard(registry, benchmark_path=False)
    rows = {r["model_id"]: r for r in frame.to_dicts()}
    assert rows["a" * 64]["lifecycle_stage"] == "staked"
    assert rows["a" * 64]["stale"] is False


def test_series_label_prefers_display_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry"
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", experiments)
    entry = _registry_entry("a" * 64)
    entry["manifest"]["config"]["run"]["name"] = "brb1-xgb-v6"
    _write_registry(registry, [entry])
    _write_family_meta(experiments, "brb1-xgb-v6", display_name="Brb1 XGB v6")
    assert dash._series_label(registry, "a" * 64) == "Brb1 XGB v6 · " + "a" * 8
    # no experiments meta -> falls back to the run name
    assert dash._series_label(registry, "z" * 64) == "unknown · " + "z" * 8


def test_gate_status_full_rows_stamped_full(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        [
            {
                "model_id": "brb1-xgb-v6::full",
                "source": "full",
                "corr": None,
                "corr_sharpe_ac": None,
                "fnc": None,
                "deflated_sharpe": None,
                "gain_to_pain_ratio": None,
                "cagr_1y": None,
                "turnover_mean": None,
            }
        ],
        schema=dash.UNIFIED_SCHEMA,
        strict=False,
    )
    out = dash.evaluate_gate_status(frame, _GATE_YAML, tmp_path / "champion.json").row(
        0, named=True
    )
    assert out["status"] == "FULL"
    assert out["gate_corr"] is None
    assert out["gate_turnover_mean"] is None


def test_evaluable_rows_predicate() -> None:
    """EVALUABLE_ROWS (spec §8): trained + benchmark rows are evaluable;
    diagnostic full/partial rows are not."""
    frame = pl.DataFrame(
        [
            {"model_id": "a", "source": "trained"},
            {"model_id": "b", "source": "benchmark"},
            {"model_id": "c::full", "source": "full"},
            {"model_id": "d::partial", "source": "partial"},
        ],
        schema=dash.UNIFIED_SCHEMA,
        strict=False,
    )
    keep = frame.filter(dash.EVALUABLE_ROWS).get_column("model_id").to_list()
    assert keep == ["a", "b"]


def test_dashboard_metric_specs_make_direction_and_default_explicit() -> None:
    specs = {spec.name: spec for spec in dash.DASHBOARD_METRICS}

    assert dash.DEFAULT_RANK_METRIC == "mmc"
    assert specs["cagr_1y"].label == "Profitability (CAGR 1Y)"
    assert specs["mmc"].window == "standardized meta-overlap"
    assert specs["mmc"].higher_is_better is True
    assert specs["corr"].higher_is_better is True
    assert specs["max_drawdown"].higher_is_better is False
    assert specs["turnover_mean"].higher_is_better is False
    assert specs["std_corr"].higher_is_better is False


def test_dashboard_cohort_uses_source_and_benchmark_tier() -> None:
    assert dash.dashboard_cohort({"source": "trained"}) == "trained"
    assert dash.dashboard_cohort({"source": "trained_legacy"}) == "trained"
    assert dash.dashboard_cohort({"source": "full"}) == "full"
    assert dash.dashboard_cohort({"source": "partial"}) == "partial"
    assert dash.dashboard_cohort({"source": "benchmark", "tier": 0}) == "heuristic"
    assert dash.dashboard_cohort({"source": "benchmark", "tier": 2}) == "heuristic"
    assert dash.dashboard_cohort({"source": "benchmark", "tier": 3}) == "benchmark"
    assert dash.dashboard_cohort({"source": "benchmark", "tier": 4}) == "benchmark"
    assert dash.dashboard_cohort({"source": "benchmark", "tier": None}) == "benchmark"


def test_rank_leaderboard_is_direction_aware_null_safe_and_deterministic() -> None:
    rows = [
        {"model_id": "b", "source": "trained", "mmc": 0.2, "max_drawdown": 0.1},
        {"model_id": "a", "source": "trained", "mmc": 0.2, "max_drawdown": 0.2},
        {"model_id": "null", "source": "trained", "mmc": None, "max_drawdown": None},
        {
            "model_id": "h",
            "source": "benchmark",
            "tier": 1,
            "mmc": 0.3,
            "max_drawdown": 0.3,
        },
        {"model_id": "full", "source": "full", "mmc": 9.0, "max_drawdown": 9.0},
    ]
    frame = pl.DataFrame(rows, schema=dash.UNIFIED_SCHEMA, strict=False)

    ranked = dash.rank_leaderboard(frame, metric="mmc")
    assert ranked.get_column("model_id").to_list() == ["h", "a", "b", "null", "full"]
    assert ranked.get_column("rank").to_list() == [1, 2, 3, None, None]
    assert ranked.get_column("cohort").to_list() == [
        "heuristic",
        "trained",
        "trained",
        "trained",
        "full",
    ]

    lower = dash.rank_leaderboard(frame, metric="max_drawdown")
    assert lower.get_column("model_id").to_list() == ["b", "a", "h", "null", "full"]
    assert lower.get_column("rank").to_list() == [1, 2, 3, None, None]


def test_rank_map_covers_each_metric_and_excludes_full_rows() -> None:
    frame = pl.DataFrame(
        [
            {"model_id": "a", "source": "trained", "mmc": 0.2, "corr": 0.1},
            {
                "model_id": "b",
                "source": "benchmark",
                "tier": 3,
                "mmc": 0.3,
                "corr": 0.05,
            },
            {"model_id": "full", "source": "full", "mmc": 9.0, "corr": 9.0},
        ],
        schema=dash.UNIFIED_SCHEMA,
        strict=False,
    )

    ranks = dash.rank_map_by_metric(frame)
    assert ranks["a"]["mmc"] == 2
    assert ranks["b"]["mmc"] == 1
    assert "full" not in ranks
    assert ranks["a"]["corr"] == 1


def test_ml_advantage_reports_directional_edges_and_missing_baselines() -> None:
    frame = pl.DataFrame(
        [
            {"model_id": "trained", "source": "trained", "mmc": 0.20},
            {"model_id": "heuristic", "source": "benchmark", "tier": 2, "mmc": 0.10},
            {"model_id": "benchmark", "source": "benchmark", "tier": 4, "mmc": 0.05},
        ],
        schema=dash.UNIFIED_SCHEMA,
        strict=False,
    )
    advantage = dash.compute_ml_advantage(frame, metric="mmc")

    assert advantage["trained"]["model_id"] == "trained"
    assert advantage["heuristic"]["model_id"] == "heuristic"
    assert advantage["benchmark"]["model_id"] == "benchmark"
    assert advantage["vs_heuristic"]["absolute_edge"] == pytest.approx(0.10)
    assert advantage["vs_heuristic"]["relative_percent"] == pytest.approx(100.0)
    assert advantage["vs_benchmark"]["absolute_edge"] == pytest.approx(0.15)

    missing = frame.filter(pl.col("source") == "trained")
    missing_advantage = dash.compute_ml_advantage(missing, metric="mmc")
    assert missing_advantage["vs_heuristic"]["absolute_edge"] is None
    assert missing_advantage["vs_heuristic"]["reason"] == "no_heuristic_baseline"


def test_tournament_payload_contains_ranked_rows_details_and_offline_metadata(
    tmp_path: Path,
) -> None:
    trained_id = "a" * 64
    trained_entry = _registry_entry(trained_id)
    trained_entry["scorecard"]["mmc"] = 0.20
    _write_registry(tmp_path, [trained_entry])
    benchmark = _write_benchmark_csv(
        tmp_path,
        "model_id,corr,mmc,corr_sharpe_ac,strategy_group,tier\n"
        "simple_baseline,0.01,0.02,0.1,ridge,1\n",
    )
    frame = dash.load_unified_leaderboard(
        tmp_path, benchmark_path=benchmark, models_dir=tmp_path / "models"
    ).with_columns(pl.lit("CAPITAL READY").alias("status"))

    payload = dash.build_tournament_payload(
        frame,
        champion_id=trained_id,
        evaluation_eras=["0001", "0002"],
    )

    assert payload["meta"] == {
        "suite_version": "v2",
        "data_version": "v5.3",
        "evaluation_window": {"start": "0001", "end": "0002", "n_eras": 2},
        "offline_only": True,
        "default_rank_metric": "mmc",
        "metric_window": "standardized meta-overlap",
    }
    row_map = [dict(zip(payload["row_fields"], row)) for row in payload["rows"]]
    assert [row["cohort"] for row in row_map] == ["trained", "heuristic"]
    assert payload["model_ids"] == [trained_id, "simple_baseline"]
    assert payload["rank_values"][0][payload["metric_fields"].index("mmc")] == 1
    assert row_map[0]["champion"] is True
    trained_index = next(
        i for i, model_id in enumerate(payload["model_ids"]) if model_id == trained_id
    )
    corr_index = payload["metric_fields"].index("corr")
    assert payload["row_value_scale"] == 1_000_000
    assert row_map[trained_index]["values"][corr_index] == 120_000
    assert payload["strict_beats"]["mmc"] == []
    seed_index = payload["provenance_fields"].index("seed")
    provenance_pairs = payload["details"][trained_index][1]
    assert all(pair[0] != seed_index for pair in provenance_pairs)
    assert payload["advantage"]["vs_heuristic"]["absolute_edge"] == pytest.approx(0.18)


def test_tournament_payload_keeps_sub_micro_ranks_distinct() -> None:
    frame = pl.DataFrame(
        [
            {"model_id": "a", "source": "trained", "mmc": 0.1000004},
            {"model_id": "b", "source": "trained", "mmc": 0.1},
            {"model_id": "benchmark", "source": "benchmark", "tier": 4, "mmc": 0.1},
        ],
        schema=dash.UNIFIED_SCHEMA,
        strict=False,
    )

    payload = dash.build_tournament_payload(frame)
    mmc_index = payload["metric_fields"].index("mmc")
    row_map = [dict(zip(payload["row_fields"], row)) for row in payload["rows"]]

    assert payload["model_ids"] == ["a", "b", "benchmark"]
    assert [payload["rank_values"][index][mmc_index] for index in range(3)] == [1, 2, 3]
    assert [row["values"][mmc_index] for row in row_map] == [100_000, 100_000, 100_000]
    assert payload["strict_beats"]["mmc"] == ["a"]


def test_tournament_payload_is_deterministic_and_does_not_expose_run_paths(
    tmp_path: Path,
) -> None:
    run_id = "b" * 64
    _write_registry(tmp_path, [_registry_entry(run_id)])
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)

    first = dash.build_tournament_payload(frame, evaluation_eras=["0001"])
    second = dash.build_tournament_payload(frame, evaluation_eras=["0001"])

    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
    encoded = json.dumps(first, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert "run_dir" not in encoded


def test_load_model_detail_keeps_full_scorecard_fields(tmp_path: Path) -> None:
    run_id = "d" * 64
    _write_registry(tmp_path, [_registry_entry(run_id)])
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    detail = dash.load_model_detail(frame.row(0, named=True))

    assert detail["scorecard"]["corr"] == 0.12
    assert detail["scorecard"]["gain_to_pain_ratio"] == 2.0
    assert detail["scorecard"]["mmc_down"] == 0.01
    assert detail["provenance"]["oof_device"] == "cpu"


def test_load_model_detail_partial_renders_cross_check_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §8 + §12 #15: a partial row's detail carries the slot's cross-check
    scorecard (flat-mapped onto the standard detail key convention)."""
    registry = tmp_path / "registry"
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", experiments)
    run_id = "a" * 64
    _write_registry(registry, [_registry_entry(run_id)])
    _write_family_meta(experiments, "sample-run")
    _write_export_slot(
        experiments,
        "sample-run",
        "partial",
        run_id,
        scorecard=_partial_scorecard(),
    )
    frame = dash.load_unified_leaderboard(
        registry, benchmark_path=False, models_dir=experiments
    )
    partial_row = frame.filter(pl.col("source") == "partial").row(0, named=True)
    detail = dash.load_model_detail(partial_row)

    assert detail["source"] == "partial"
    assert detail["cohort"] == "partial"
    assert detail["reason"] is None
    assert detail["evidence_ref"] == (
        f"experiments/sample-run/exports/partial/{run_id}/scorecard.json"
    )
    assert detail["scorecard"]["corr"] == pytest.approx(0.11)
    assert detail["scorecard"]["corr_ci_low"] == pytest.approx(0.04)
    assert detail["scorecard"]["corr_ci_high"] == pytest.approx(0.18)
    assert detail["scorecard"]["corr_n_eras"] == 22
    assert detail["scorecard"]["mmc"] == pytest.approx(0.09)
    assert detail["scorecard"]["mmc_ci_low"] == pytest.approx(-0.01)
    assert detail["scorecard"]["mmc_ci_high"] == pytest.approx(0.19)
    assert detail["scorecard"]["mmc_n_eras"] == 22
    assert detail["scorecard"]["corr_sharpe_ac"] == pytest.approx(0.62)
    assert detail["scorecard"]["corr_sharpe_ac_ci_low"] == pytest.approx(0.4)
    assert detail["scorecard"]["corr_sharpe_ac_ci_high"] == pytest.approx(0.84)
    assert detail["scorecard"]["fnc"] == pytest.approx(0.021)
    assert detail["scorecard"]["deflated_sharpe"] == pytest.approx(0.91)
    assert detail["scorecard"]["max_drawdown"] == pytest.approx(0.13)
    assert detail["scorecard"]["burn_rate"] == pytest.approx(0.5)


def _lb_row(
    model_id: str,
    source: str,
    run_name: str,
    sharpe: float | None = None,
    has_full: bool = False,
) -> dict:
    return {
        "model_id": model_id,
        "source": source,
        "run_name": run_name,
        "corr_sharpe_ac": sharpe,
        "has_full_version": has_full,
    }


def test_generate_dashboard_table_rows_grouping() -> None:
    rows = [
        _lb_row("ch" * 32, "trained", "champ-run", 0.9),
        _lb_row("a" * 64, "trained", "brb1-xgb-v6", 0.5, has_full=True),
        _lb_row("brb1-xgb-v6::full", "full", "brb1-xgb-v6"),
        _lb_row("bench_a", "benchmark", "ref", 0.78),
    ]
    frame = pl.DataFrame(rows, schema=dash.UNIFIED_SCHEMA, strict=False)
    ordered = generate_dashboard._table_rows(frame, champion="ch" * 32)
    kinds = ["header" if r.get("_group_header") else r["source"] for r in ordered]
    assert kinds == ["trained", "header", "full", "trained", "benchmark"]


def test_generate_dashboard_row_html_full_chip() -> None:
    row = {
        **_lb_row("a" * 64, "trained", "brb1-xgb-v6", 0.5, has_full=True),
        "status": "RESEARCH",
        "cagr_1y": None,
        "corr_sharpe_ac_ci_low": None,
        "corr_sharpe_ac_ci_high": None,
        "max_drawdown": None,
        "gain_to_pain_ratio": None,
        "mmc_down": None,
        "deflated_sharpe": None,
        "gate_cagr_1y": None,
        "gate_corr_sharpe_ac": None,
        "gate_gain_to_pain_ratio": None,
        "gate_deflated_sharpe": None,
    }
    html_out = generate_dashboard._row_html(row)
    assert 'class="badge full">FULL</span>' in html_out


def test_generate_dashboard_bar_input_includes_benchmark_excludes_diagnostic_rows() -> None:
    rows = [
        _lb_row("a" * 64, "trained", "r1", 0.5),
        _lb_row("bench_a", "benchmark", "ref", 0.78),
        _lb_row("brb1-xgb-v6::full", "full", "brb1-xgb-v6"),
        _lb_row("brb1-xgb-v6::partial", "partial", "brb1-xgb-v6", 0.9),
    ]
    frame = pl.DataFrame(rows, schema=dash.UNIFIED_SCHEMA, strict=False)
    out = generate_dashboard._bar_input(frame, champion=None)
    # benchmark rows are reference curves on the chart (EVALUABLE_ROWS);
    # full/partial diagnostic rows never appear even with metric cells
    assert out.get_column("label").to_list() == ["ref · bench_a", "r1 · " + "a" * 8]


def test_dashboard_app_load_registry_frame_includes_full_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dashboard_ui import app

    registry = tmp_path / "registry"
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", experiments)
    entry = _registry_entry("a" * 64)
    entry["manifest"]["config"]["run"]["name"] = "brb1-xgb-v6"
    _write_registry(registry, [entry])
    _write_family_meta(experiments, "brb1-xgb-v6")
    _write_export_slot(experiments, "brb1-xgb-v6", "full", "a" * 64)
    _write_current_pointer(experiments, "brb1-xgb-v6", "a" * 64)
    frame = app.load_registry_frame(registry, models_dir=experiments)
    sources = frame.get_column("source").to_list()
    assert "full" in sources
    assert "trained" in sources
    full_id = "brb1-xgb-v6::full::" + "a" * 64
    full_row = frame.filter(pl.col("model_id") == full_id).row(0, named=True)
    assert full_row["backend"] == "xgboost"  # filled from export.json snapshot
    assert full_row["has_full_version"] is False


def test_dashboard_app_shaped_leaderboard_pins_full_rows_first() -> None:
    from dashboard_ui import app

    rows = [
        {
            "model_id": "a" * 64,
            "source": "trained",
            "run_name": "r1",
            "corr_sharpe_ac": 0.5,
            "corr_sharpe_ac_ci_low": None,
            "corr_sharpe_ac_ci_high": None,
        },
        {
            "model_id": "brb1-xgb-v6::full",
            "source": "full",
            "run_name": "brb1-xgb-v6",
            "corr_sharpe_ac": None,
            "corr_sharpe_ac_ci_low": None,
            "corr_sharpe_ac_ci_high": None,
        },
    ]
    frame = pl.DataFrame(rows, schema=dash.UNIFIED_SCHEMA, strict=False)
    pdf = app._shaped_leaderboard_pdf(frame, champion=None)
    assert list(pdf["model_id"]) == ["brb1-xgb-v6::full", "a" * 64]
    assert "_is_full" not in pdf.columns


def test_dashboard_app_robustness_matrix_includes_benchmark_excludes_diagnostic_rows() -> None:
    from dashboard_ui import app

    rows = [
        {
            "model_id": "a" * 64,
            "source": "trained",
            "run_name": "r1",
            "corr_sharpe_ac": 0.5,
            "has_bmc": True,
            "has_horizon": False,
            "has_perturb": True,
            "has_regime": False,
            "max_feature_exposure": 0.3,
            "std_corr": 0.2,
            "max_drawdown": 0.1,
        },
        {
            "model_id": "bench_a",
            "source": "benchmark",
            "run_name": "ref",
            "corr_sharpe_ac": 0.78,
            "has_bmc": False,
            "has_horizon": None,
            "has_perturb": None,
            "has_regime": None,
            "max_feature_exposure": None,
            "std_corr": None,
            "max_drawdown": None,
        },
        {
            "model_id": "brb1-xgb-v6::full",
            "source": "full",
            "run_name": "brb1-xgb-v6",
            "corr_sharpe_ac": None,
            "has_bmc": None,
            "has_horizon": None,
            "has_perturb": None,
            "has_regime": None,
            "max_feature_exposure": None,
            "std_corr": None,
            "max_drawdown": None,
        },
        {
            "model_id": "brb1-xgb-v6::partial",
            "source": "partial",
            "run_name": "brb1-xgb-v6",
            "corr_sharpe_ac": 0.9,
            "has_bmc": None,
            "has_horizon": None,
            "has_perturb": None,
            "has_regime": None,
            "max_feature_exposure": None,
            "std_corr": None,
            "max_drawdown": None,
        },
    ]
    frame = pl.DataFrame(rows, schema=dash.UNIFIED_SCHEMA, strict=False)
    matrix = app.robustness_matrix(frame)
    # benchmark reference curves participate; diagnostic full/partial do not
    assert matrix.get_column("model_id").to_list() == ["a" * 64, "bench_a"]

def test_reconcile_capital_metrics_uses_historical_pf_csv(tmp_path: Path) -> None:
    """A present payout-factor CSV must change the recomputed capital cells:
    with the clip-saturated perfect-corr fixture, PF=0.01 pulls the per-era
    returns inside the clip band, lowering the annualized return below the
    PF=1.0 clip-saturated baseline."""
    _write_registry(tmp_path, [_registry_entry("a" * 64, scorecard=True)])
    entry = json.loads((tmp_path / ("a" * 64) / "run.json").read_text(encoding="utf-8"))
    for field in ("cagr_1y", "gain_to_pain_ratio", "kelly_fraction", "mmc_down"):
        entry["scorecard"].pop(field, None)
    (tmp_path / ("a" * 64) / "run.json").write_text(json.dumps(entry), encoding="utf-8")
    _write_preds(tmp_path / ("a" * 64), scale=1.0)
    data = _synthetic_data_dir(tmp_path)
    (data / "payout_factor_historic.csv").write_text(
        "round,status,close,resolve,pf\n"
        "1,Resolved,Jan 01 2025,Feb 01 2025,0.01\n"
        "2,Resolved,Jan 02 2025,Feb 02 2025,0.01\n"
        "3,Resolved,Jan 03 2025,Feb 03 2025,0.01\n",
        encoding="utf-8",
    )
    frame = dash.load_unified_leaderboard(tmp_path, benchmark_path=False)
    out = dash.reconcile_capital_metrics(frame, data)
    row = out.row(0, named=True)
    assert row["cagr_1y"] is not None
    # PF=1.0 (no CSV) clips every era at 0.05 -> (1.05^52 - 1); PF=0.01 scales
    # the raw payouts down inside the band, so the annualized return must be
    # strictly lower.
    assert row["cagr_1y"] < (1.05**52) - 1.0
    assert row["cagr_1y"] > 0.0


def test_benchmark_tier4_labels_unique_per_model(tmp_path: Path) -> None:
    """The tier-4 reference pair shares the generic "tier4" strategy label;
    derived display labels must be unique per model id."""
    csv_path = tmp_path / "bench.csv"
    csv_path.write_text(
        "model_id,strategy_group,tier,horizon_target_name,mean_payout\n"
        "v53_lgbm_ender20,tier4,4,ender,0.01\n"
        "v53_lgbm_ender60,tier4,4,ender,0.02\n"
        "null_constant_05,null,0,,0.0\n"
        , encoding="utf-8",
    )
    frame = dash.load_benchmark_frame(csv_path)
    labels = frame.select(["model_id", "run_name"]).to_dicts()
    by_id = {row["model_id"]: row["run_name"] for row in labels}
    assert by_id["v53_lgbm_ender20"] == "Tier 4 " + chr(0xB7) +  " Ender 20D"
    assert by_id["v53_lgbm_ender60"] == "Tier 4 " + chr(0xB7) +  " Ender 60D"
    assert len({row["run_name"] for row in labels}) == len(labels)
