"""Contract tests for control-plane scripts (F-018)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmark_runner
import generate_dashboard
import train_first_model  # noqa: F401  (import-time smoke)
from nmr import paths


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


from dashboard_ui import (  # noqa: E402  (lazy: streamlit is heavy at module load)
    app as dashboard_app,
)


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


def test_train_first_model_registry_points_at_experiments_root() -> None:
    # Task 11: the comparison/champion registry reads the experiments layout.
    assert train_first_model._build_registry()._root == paths.EXPERIMENTS_ROOT


def test_run_campaign_default_registry_root_is_experiments() -> None:
    import run_campaign

    args = run_campaign._parse_args(["--config", "a.yaml", "--name", "camp"])
    assert Path(args.registry) == paths.EXPERIMENTS_ROOT


def test_run_campaign_records_via_experiment_store(tmp_path, monkeypatch) -> None:
    import polars as pl

    import run_campaign
    from nmr.evaluation import MetricSummary
    from nmr.runner import ExperimentRunner, RunResult

    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")

    def fake_run(self, *, deploy: bool = False) -> RunResult:
        return RunResult(
            run_id="a" * 64,
            oof=pl.DataFrame({"id": ["x"], "era": ["1"], "prediction": [0.5]}),
            metrics=MetricSummary(mean=0.1, std=0.2, sharpe=0.5, max_drawdown=0.05),
            artifact=None,
            manifest={"run_id": "a" * 64, "oof_device": "cpu"},
        )

    monkeypatch.setattr(ExperimentRunner, "run", fake_run)
    monkeypatch.setattr(
        ExperimentRunner, "compute_run_id", staticmethod(lambda config: "a" * 64)
    )
    monkeypatch.setattr(
        ExperimentRunner,
        "_compute_run_id",
        staticmethod(lambda config, **_: "a" * 64),
    )
    cfg = tmp_path / "a.yaml"
    cfg.write_text("run:\n  name: x\n", encoding="utf-8")
    rc = run_campaign.main([
        "--config", str(cfg), "--name", "camp",
        "--registry", str(paths.EXPERIMENTS_ROOT),
        "--campaigns-dir", str(tmp_path / "campaigns"),
    ])
    assert rc == 0
    # Recording lands in the experiments layout (not artifacts/registry).
    assert (
        paths.EXPERIMENTS_ROOT / "x" / "runs" / ("a" * 64) / "run.json"
    ).is_file()


def test_promote_model_champion_resolves_from_paths_pointer(
    tmp_path, monkeypatch
) -> None:
    import promote_model
    from nmr import experiment_store
    from nmr.registry import RunRegistry

    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    experiment_store.record_run("fam-a", "a" * 64, {"scorecard": {}})
    RunRegistry(paths.EXPERIMENTS_ROOT).promote("a" * 64, "fam-a")
    # Task 11: champion resolution reads paths.champion_path() only.
    assert promote_model._resolve_champion_run_id() == "a" * 64


def test_promote_model_missing_champion_fails_loud(tmp_path, monkeypatch) -> None:
    import promote_model

    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    with pytest.raises(FileNotFoundError, match="no champion"):
        promote_model._resolve_champion_run_id()


def test_promote_model_champion_family_mismatch_raises(
    tmp_path, monkeypatch
) -> None:
    """SECONDARY 6: with --champion the pointer's experiment_slug is
    authoritative for promotion — a user-supplied --family that disagrees is
    a clear error, never a silent promotion under another family."""
    import promote_model
    from nmr import experiment_store
    from nmr.registry import RunRegistry

    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    experiment_store.record_run("champ-fam", "a" * 64, {"scorecard": {}})
    RunRegistry(paths.EXPERIMENTS_ROOT).promote("a" * 64, "champ-fam")

    # --champion + a wrong --family raises before any promotion.
    with pytest.raises(ValueError, match="does not match the champion's family"):
        promote_model.main(
            ["--champion", "--family", "other-fam", "--override-gate"]
        )

    # --champion + the matching --family resolves the champion run + slug.
    run_id, slug = promote_model._resolve_champion()
    assert run_id == "a" * 64
    assert slug == "champ-fam"


def test_promote_and_rehearse_clis_reject_legacy_dir_args() -> None:
    # Task 11: models_dir/registry_dir are gone from the promotion CLIs.
    import promote_model
    import rehearse_promotion

    with pytest.raises(SystemExit):
        promote_model._parse_args(
            ["--run-id", "a" * 64, "--family", "fam", "--models-dir", "x"]
        )
    with pytest.raises(SystemExit):
        rehearse_promotion._parse_args(
            ["--run-id", "a" * 64, "--family", "fam", "--registry-dir", "x"]
        )


def test_dashboard_defaults_point_at_experiments_root() -> None:
    from nmr import dashboard

    assert dashboard.DEFAULT_REGISTRY_DIR == paths.EXPERIMENTS_ROOT
    assert dashboard_app._DEFAULT_REGISTRY_DIR == paths.EXPERIMENTS_ROOT
