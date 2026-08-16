from __future__ import annotations

from pathlib import Path

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
