from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import nmr.dashboard as dash
from nmr.config import REPO_ROOT


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
    assert dash.resolve_benchmark_path(reports / "nope.csv", reports_dir=reports, legacy_path=legacy) == smoke
    smoke.unlink()
    assert dash.resolve_benchmark_path(reports / "nope.csv", reports_dir=reports, legacy_path=legacy) == legacy
    legacy.unlink()
    assert dash.resolve_benchmark_path(reports / "nope.csv", reports_dir=reports, legacy_path=legacy) is None


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
