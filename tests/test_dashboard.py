from __future__ import annotations

from pathlib import Path

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
