"""Contract tests for control-plane scripts (F-018)."""

from __future__ import annotations

import pandas as pd
import polars as pl

import benchmark_runner
import generate_dashboard
import train_first_model  # noqa: F401  (import-time smoke)


class _StubSuite:
    """Public-surface stub: only iter_baseline_predictions exists."""

    def __init__(self) -> None:
        self.frame = pl.DataFrame(
            {"era": ["1", "1"], "id": ["a", "b"], "prediction": [0.1, 0.2]}
        )

    def iter_baseline_predictions(self, *, include_classical, min_train_eras):
        yield ("constant-0.5", "null", self.frame, 77)
        if include_classical:
            yield ("linear", "classical", self.frame, 81)


def test_candidate_strategies_consumes_only_public_api() -> None:
    suite = _StubSuite()
    benchmarks = pl.DataFrame(
        {"era": ["1", "1"], "id": ["a", "b"], "bench_a": [0.3, 0.4]}
    )
    contexts = list(
        benchmark_runner._candidate_strategies(suite, benchmarks, seed=77, min_train_eras=2, fast_mode=False)
    )
    assert [ctx.model_id for ctx in contexts] == ["constant-0.5", "linear", "bench_a"]
    assert contexts[0].seed == 77
    assert contexts[1].seed == 81
    assert contexts[2].seed == 83  # benchmark_model rows keep seed + 6 (bootstrap CIs)


def test_dashboard_escapes_html_interpolation() -> None:
    df = pd.DataFrame(
        [
            {
                "model_id": "<script>alert(1)</script>",
                "source": "trained",
                "run_name": '"><img src=x onerror=alert(2)>',
                "feature_set": "small",
                "backend": "lightgbm",
                "preset": "fast",
                "n_targets": 1,
                "targets": "target",
                "mean": 0.1,
                "std": 0.2,
                "sharpe": 0.5,
                "max_drawdown": 0.05,
                "rank": 1,
            }
        ]
    )
    html = generate_dashboard._build_html(
        df,
        benchmark_path=__import__("pathlib").Path("benchmark_scores.csv"),
        registry_dir=__import__("pathlib").Path("registry"),
        legacy=pd.DataFrame(),
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_dashboard_ranks_trained_and_benchmark_on_same_sharpe() -> None:
    trained = pd.DataFrame(
        [
            {
                "model_id": "trained_a", "source": "trained", "run_name": "t",
                "feature_set": "small", "backend": "lgbm", "preset": "fast",
                "n_targets": 1, "targets": "target", "mean": 0.1, "std": 0.1,
                "sharpe": 1.5, "max_drawdown": 0.1,
            }
        ]
    )
    benchmark = pd.DataFrame(
        [
            {
                "model_id": "bench_a", "source": "benchmark", "run_name": "b",
                "feature_set": "all", "backend": "benchmark", "preset": "benchmark",
                "n_targets": 1, "targets": "target", "mean": 0.05, "std": 0.1,
                "sharpe": 0.5, "max_drawdown": 0.2,
            }
        ]
    )
    ranked = generate_dashboard._rank_models(pd.concat([trained, benchmark], ignore_index=True))
    assert ranked.iloc[0]["model_id"] == "trained_a"


def test_run_campaign_imports_as_control_plane() -> None:
    import run_campaign  # noqa: F401  (import-time smoke)


import json

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


def test_dashboard_app_imports_without_launching() -> None:
    # Module-level import must be side-effect free: streamlit/plotly import
    # headless, no server is launched, and every `st.*` call stays inside
    # main()/view functions. `main` was already callable in the Task 2 stub,
    # so the real contract is the five render views per the Task 3 brief.
    import dashboard_app  # noqa: F401

    assert callable(dashboard_app.main)
    for view in (
        "render_leaderboard",
        "render_run_detail",
        "render_fleet",
        "render_campaigns",
        "render_robustness_matrix",
    ):
        assert callable(getattr(dashboard_app, view)), f"missing render view: {view}"
