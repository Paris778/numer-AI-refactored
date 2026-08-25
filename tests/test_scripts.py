"""Contract tests for control-plane scripts (F-018)."""

from __future__ import annotations

import json

import benchmark_runner
import generate_dashboard
import train_first_model  # noqa: F401  (import-time smoke)


def test_benchmark_runner_import_surface() -> None:
    # Task 9 rewrote the runner CLI around BenchmarkHierarchy; the runner
    # must stay importable with its argument parser present.
    assert callable(benchmark_runner._parse_args)


def test_generate_dashboard_import_surface() -> None:
    assert callable(generate_dashboard.generate_dashboard)
    assert callable(generate_dashboard.main)


def test_run_campaign_imports_as_control_plane() -> None:
    import run_campaign  # noqa: F401  (import-time smoke)


def test_promote_model_import_surface() -> None:
    import promote_model  # noqa: F401  (import-time smoke)

    assert callable(promote_model.main)


def test_rehearse_promotion_import_surface() -> None:
    import rehearse_promotion  # noqa: F401  (import-time smoke)

    assert callable(rehearse_promotion.main)


def test_real_data_gate_import_surface() -> None:
    import scripts.real_data_gate as real_data_gate  # noqa: F401

    assert callable(real_data_gate.main)


from dashboard_ui import (
    app as dashboard_app,
)  # noqa: E402  (lazy: streamlit is heavy at module load)


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
                "horizon_model_sharpe_20": 0.5,
                "perturb_ceiling_stability": 0.9,
                "regime_count": 3,
            }
        ),
    }
    return entry


def _write_registry(tmp_path, entries) -> None:
    for entry in entries:
        run_dir = tmp_path / entry["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps(entry), encoding="utf-8")


def test_registry_frame_columns_and_source_tagging(tmp_path) -> None:
    _write_registry(
        tmp_path,
        [_registry_entry("a" * 64), _registry_entry("b" * 64, scorecard=False)],
    )
    # Isolated models dir: the real artifacts/models/ may legitimately hold a
    # promoted full version (the D7 rehearsal artifact) — tests must not scan it.
    frame = dashboard_app.load_registry_frame(tmp_path, models_dir=tmp_path / "models")
    assert frame.height == 2
    assert set(frame.columns) >= {
        "model_id",
        "source",
        "backend",
        "preset",
        "feature_set",
        "feature_subset",
        "oof_device",
        "neutralization_proportion",
        "corr",
        "corr_sharpe_ac",
        "corr_sharpe_ac_ci_low",
        "corr_sharpe_ac_ci_high",
        "max_drawdown",
        "deflated_sharpe",
        "has_bmc",
        "has_horizon",
        "has_perturb",
        "has_regime",
    }
    rows = {r["model_id"]: r for r in frame.to_dicts()}
    assert rows["a" * 64]["source"] == "trained"
    assert rows["a" * 64]["corr"] == 0.12
    assert rows["a" * 64]["has_bmc"] is True
    assert rows["b" * 64]["source"] == "trained_legacy"
    assert rows["b" * 64]["corr"] == 0.1  # legacy falls back to metrics.mean
    assert rows["b" * 64]["has_bmc"] is False


def test_registry_frame_zero_value_not_treated_as_legacy(tmp_path) -> None:
    entry = _registry_entry("c" * 64)
    entry["scorecard"]["corr"] = 0.0  # legitimate 0.0 must NOT fall through
    _write_registry(tmp_path, [entry])
    frame = dashboard_app.load_registry_frame(tmp_path)
    assert frame.row(0, named=True)["corr"] == 0.0


def test_leaderboard_bar_labels_unique_on_run_name_collision(tmp_path) -> None:
    # Real-data collision: both registry runs share run_name
    # "first-competitive-lgbm-small" (their config name), so run_name alone
    # cannot key the bars — px.bar would draw two overlapping bars at one
    # x-tick. Labels must be unique per run while keeping the readable name.
    a = _registry_entry("a" * 64)
    b = _registry_entry("b" * 64, scorecard=False)
    for entry in (a, b):
        entry["manifest"]["config"]["run"] = {"name": "same-config-name"}
    _write_registry(tmp_path, [a, b])
    frame = dashboard_app.load_registry_frame(tmp_path, models_dir=tmp_path / "models")
    assert frame.height == 2
    assert frame.get_column("run_name").n_unique() == 1  # collision is real
    pdf = dashboard_app._shaped_leaderboard_pdf(frame, champion=None)
    labels = pdf["label"].tolist()
    assert len(labels) == 2
    assert len(set(labels)) == 2  # unique bar keys
    assert all("same-config-name" in label for label in labels)  # readable name kept


def test_benchmarks_and_merge(tmp_path) -> None:
    bench_path = tmp_path / "bench.csv"
    bench_path.write_text(
        "model_id,corr,corr_sharpe_ac,corr_sharpe_ac_ci_low,corr_sharpe_ac_ci_high,std_corr,max_drawdown,strategy_group,horizon_target_name\n"
        "bench_a,0.05,0.5,0.4,0.6,0.3,0.2,linear,cyrusd\n",
        encoding="utf-8",
    )
    benchmarks = dashboard_app.load_benchmarks(bench_path)
    assert benchmarks.height == 1
    assert benchmarks.row(0, named=True)["source"] == "benchmark"
    assert benchmarks.row(0, named=True)["corr_sharpe_ac_ci_low"] == 0.4
    assert benchmarks.row(0, named=True)["corr_sharpe_ac_ci_high"] == 0.6
    # Absent CI columns fall back to None (real CSVs carry them, so None must
    # remain the behavior only when the data is genuinely missing).
    bench_no_ci = tmp_path / "benchmark_no_ci.csv"
    bench_no_ci.write_text(
        "model_id,corr,corr_sharpe_ac,std_corr,max_drawdown,strategy_group,horizon_target_name\n"
        "bench_b,0.05,0.5,0.3,0.2,linear,cyrusd\n",
        encoding="utf-8",
    )
    no_ci = dashboard_app.load_benchmarks(bench_no_ci).row(0, named=True)
    assert no_ci["corr_sharpe_ac_ci_low"] is None
    assert no_ci["corr_sharpe_ac_ci_high"] is None
    assert dashboard_app.load_benchmarks(tmp_path / "missing.csv").height == 0

    _write_registry(tmp_path, [_registry_entry("d" * 64)])
    registry = dashboard_app.load_registry_frame(
        tmp_path, models_dir=tmp_path / "models"
    )
    merged = dashboard_app.merge_leaderboard(registry, benchmarks)
    assert merged.height == 2
    assert set(merged.get_column("source").to_list()) == {"trained", "benchmark"}


def test_campaigns_flatten(tmp_path) -> None:
    campaigns_dir = tmp_path / "campaigns"
    campaigns_dir.mkdir()
    (campaigns_dir / "abc.json").write_text(
        json.dumps(
            {
                "campaign_id": "abc",
                "name": "camp",
                "configs": [{"path": "a.yaml", "sha256": "x" * 64}],
                "runs": [
                    {
                        "config_path": "a.yaml",
                        "run_id": "e" * 64,
                        "status": "recorded",
                        "error": None,
                    },
                    {
                        "config_path": "a.yaml",
                        "run_id": None,
                        "status": "error",
                        "error": "boom",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    frame = dashboard_app.load_campaigns(campaigns_dir)
    assert frame.height == 2
    assert set(frame.columns) == {
        "campaign_id",
        "name",
        "config_path",
        "run_id",
        "status",
        "error",
    }
    assert frame.get_column("status").to_list() == ["recorded", "error"]
    assert dashboard_app.load_campaigns(tmp_path / "missing").height == 0


def test_robustness_matrix_and_champion(tmp_path) -> None:
    _write_registry(tmp_path, [_registry_entry("f" * 64)])
    matrix = dashboard_app.robustness_matrix(
        dashboard_app.load_registry_frame(tmp_path)
    )
    assert matrix.height == 1
    assert {
        "has_bmc",
        "has_horizon",
        "has_perturb",
        "has_regime",
        "max_feature_exposure",
    } <= set(matrix.columns)
    assert dashboard_app.champion_run_id(tmp_path) is None  # no champion.json
    (tmp_path / "champion.json").write_text(
        json.dumps({"run_id": "f" * 64}), encoding="utf-8"
    )
    assert dashboard_app.champion_run_id(tmp_path) == "f" * 64


def test_champion_run_id_corrupt_json_returns_none(tmp_path) -> None:
    (tmp_path / "champion.json").write_text("{not json", encoding="utf-8")
    assert dashboard_app.champion_run_id(tmp_path) is None


def test_load_registry_frame_empty_dir_returns_schema_frame(tmp_path) -> None:
    frame = dashboard_app.load_registry_frame(tmp_path, models_dir=tmp_path / "models")
    assert frame.height == 0
    assert frame.schema == dashboard_app._LEADERBOARD_SCHEMA


def test_campaigns_null_runs_returns_empty_frame(tmp_path) -> None:
    campaigns_dir = tmp_path / "campaigns"
    campaigns_dir.mkdir()
    (campaigns_dir / "null-runs.json").write_text(
        json.dumps({"campaign_id": "null-runs", "name": "camp", "runs": None}),
        encoding="utf-8",
    )
    frame = dashboard_app.load_campaigns(campaigns_dir)
    assert frame.height == 0
    assert frame.schema == dashboard_app._CAMPAIGN_SCHEMA


def test_dashboard_app_imports_without_launching() -> None:
    # Module-level import must be side-effect free: Streamlit imports headless,
    # no server is launched, and the host delegates to one shared renderer.
    from dashboard_ui import app as dashboard_app

    assert callable(dashboard_app.main)
    assert callable(dashboard_app.render_tournament)


def test_benchmark_runner_cli_defaults() -> None:
    import benchmark_runner

    args = benchmark_runner._parse_args_with(
        ["--data-dir", "data/v5.3", "--seed", "42", "--n-boot", "1000"]
    )
    assert args.seed == 42
    assert args.n_boot == 1000
    assert args.output.name == "benchmark_hierarchy_scorecard.csv"
    assert "reports" in args.output.parts
    assert args.configs.name == "benchmarks"
    assert args.fast_mode is False


def test_benchmark_runner_cli_fast_mode_and_horizon() -> None:
    import benchmark_runner

    args = benchmark_runner._parse_args_with(["--fast-mode", "--horizon", "60D"])
    assert args.fast_mode is True
    assert args.horizon == "60D"


def test_dashboard_app_has_no_plotly_reference() -> None:
    import inspect

    from dashboard_ui import app as dashboard_app

    assert "plotly" not in inspect.getsource(dashboard_app).lower()
