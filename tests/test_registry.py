"""Tests for the cross-family run registry and champion pointer.

Runs live under ``experiments/<slug>/runs/<run_id>/run.json``; the registry
iterates families for comparison and owns the atomic ``champion.json``
pointer. Tests monkeypatch ``paths.EXPERIMENTS_ROOT`` to ``tmp_path`` so every
write stays inside the fixture.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from nmr import experiment_store, paths
from nmr.evaluation import MetricSummary
from nmr.registry import RunRegistry
from nmr.runner import RunResult
from nmr.scorecard import MetricCell, MetricScorecard


@pytest.fixture
def registry(tmp_path, monkeypatch) -> RunRegistry:
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    return RunRegistry(tmp_path / "experiments")


def _result(run_id: str, sharpe: float, name: str = "fam-a") -> RunResult:
    return RunResult(
        run_id=run_id,
        oof=pl.DataFrame(
            {
                "id": ["a", "b"],
                "era": ["1", "1"],
                "prediction": [0.1, 0.9],
            }
        ),
        metrics=MetricSummary(mean=0.1, std=0.2, sharpe=sharpe, max_drawdown=0.05),
        artifact=None,
        manifest={"config": {"run": {"name": name}}},
    )


def _scorecard(sharpe_ac: float, *, max_drawdown: float = 0.1) -> MetricScorecard:
    def cell(v: float) -> MetricCell:
        return MetricCell(value=v, ci_low=None, ci_high=None, n_eras=10)

    return MetricScorecard(
        model_id="m", n_eras=10, rank_scalar=0.0, deflated_sharpe=0.0,
        mean_payout=cell(0.0), corr=cell(0.0), mmc=cell(0.0), fnc=0.0,
        corr_sharpe_ac=cell(sharpe_ac), cvar5=0.0, max_drawdown=max_drawdown,
        burn_rate=0.0, mmc_sharpe_ac=0.0, sortino=0.0, calmar=0.0,
        std_corr=0.1, max_burn_streak=0, time_to_recovery=0,
        horizon_stability=None, horizon_reason=None, regime_corr=None,
        regime_reason=None, perturbation=None, max_feature_exposure=0.0,
        bmc=None, bmc_reason=None, cwmm=None, cwmm_reason=None,
        book_correlation=None,
        cagr_1y=0.0, gain_to_pain_ratio=0.0, kelly_fraction=0.0,
        mmc_down=None, mmc_down_n_eras=0, mmc_down_reason=None,
        turnover_mean=None, turnover_std=None, turnover_reason=None,
        sim_portfolio_cagr=0.0, sim_portfolio_mdd=0.0,
        sim_capital_utilization=0.0,
        metric_timing_seconds=None, eval_total_seconds=0.0,
    )


def _result_with_scorecard(
    run_id: str, sharpe_ac: float, *, name: str = "fam-a", max_drawdown: float = 0.1
) -> RunResult:
    result = _result(run_id, sharpe=0.5, name=name)
    return RunResult(
        run_id=result.run_id, oof=result.oof, metrics=result.metrics,
        artifact=result.artifact, manifest=result.manifest,
        scorecard=_scorecard(sharpe_ac, max_drawdown=max_drawdown),
    )


def test_cross_family_list_and_best(registry) -> None:
    experiment_store.record_run(
        "fam-a",
        "a" * 64,
        {"manifest": {"config": {"run": {"name": "fam-a"}}}, "scorecard": {"corr_sharpe_ac": 0.5}},
    )
    experiment_store.record_run(
        "fam-b",
        "b" * 64,
        {"manifest": {"config": {"run": {"name": "fam-b"}}}, "scorecard": {"corr_sharpe_ac": 0.9}},
    )
    assert set(registry.list()) == {"a" * 64, "b" * 64}
    assert registry.best() == ("b" * 64, "fam-b")


def test_champion_pointer_has_slug(registry) -> None:
    experiment_store.record_run("fam-a", "a" * 64, {"scorecard": {}})
    path = registry.promote("a" * 64, "fam-a")
    payload = json.loads(path.read_text())
    assert payload == {
        "run_id": "a" * 64,
        "experiment_slug": "fam-a",
        "promoted_at": payload["promoted_at"],
    }


def test_record_writes_legacy_layout_and_parquets(registry) -> None:
    result = _result_with_scorecard("a" * 64, sharpe_ac=0.8)
    result = RunResult(
        run_id=result.run_id, oof=result.oof, metrics=result.metrics,
        artifact=result.artifact, manifest=result.manifest,
        scorecard=result.scorecard,
        validation_predictions=pl.DataFrame(
            {"era": ["0575", "0575"], "id": ["x", "y"], "prediction": [0.2, 0.8]}
        ),
    )
    # Compat (removed in Task 11): record() writes the legacy single-pool
    # layout directly under the registry root.
    run_dir = registry.record(result)
    assert run_dir == paths.EXPERIMENTS_ROOT / ("a" * 64)
    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "a" * 64
    assert payload["manifest"]["config"]["run"]["name"] == "fam-a"
    assert payload["scorecard"]["corr_sharpe_ac"] == 0.8
    assert pl.read_parquet(run_dir / "oof.parquet").height == 2
    persisted = pl.read_parquet(run_dir / "validation_preds.parquet")
    assert persisted["prediction"].to_list() == [0.2, 0.8]
    # no validation predictions -> no file
    assert not (paths.EXPERIMENTS_ROOT / ("d" * 64) / "validation_preds.parquet").exists()


def test_record_is_idempotent(registry) -> None:
    result = _result("a" * 64, sharpe=0.7)
    run_dir = registry.record(result)
    original = (run_dir / "run.json").read_text(encoding="utf-8")
    assert registry.record(result) == run_dir
    assert (run_dir / "run.json").read_text(encoding="utf-8") == original


def test_record_tolerates_manifest_without_config_run_name(registry) -> None:
    # test_campaign.py stubs produce manifests without config.run.name; the
    # compat record() must not require a slug (slug matters only at promote).
    result = _result("a" * 64, sharpe=0.7)
    result = RunResult(
        run_id=result.run_id, oof=result.oof, metrics=result.metrics,
        artifact=result.artifact, manifest={"run_id": result.run_id},
    )
    run_dir = registry.record(result)
    assert (run_dir / "run.json").is_file()


def test_record_atomic_write_failure_keeps_previous_run_json(
    registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _result("a" * 64, sharpe=0.7)
    run_dir = registry.record(result)
    stable_json = (run_dir / "run.json").read_text(encoding="utf-8")

    import nmr._atomicio as atomicio_module

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(atomicio_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        registry.record(result)

    assert (run_dir / "run.json").read_text(encoding="utf-8") == stable_json


def test_list_returns_all_run_ids_sorted(registry) -> None:
    experiment_store.record_run("fam-b", "b" * 64, {"scorecard": {}})
    experiment_store.record_run("fam-a", "a" * 64, {"scorecard": {}})
    assert registry.list() == ["a" * 64, "b" * 64]


def test_best_skips_runs_without_metric_and_empty_registry(registry) -> None:
    assert registry.best() is None
    experiment_store.record_run(
        "fam-a", "a" * 64, {"scorecard": {"corr_sharpe_ac": 0.3}}
    )
    experiment_store.record_run("fam-b", "b" * 64, {"scorecard": {}})  # no metric
    assert registry.best() == ("a" * 64, "fam-a")
    assert registry.best("mmc") is None  # no run carries mmc


def test_promote_resolves_slug_by_scanning_when_omitted(registry) -> None:
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    path = registry.promote("a" * 64)  # compat: slug=None
    assert json.loads(path.read_text(encoding="utf-8"))["experiment_slug"] == "fam-a"
    assert registry.promote("a" * 64) == path  # idempotent


def test_promote_unknown_run_fails_loud(registry) -> None:
    with pytest.raises(FileNotFoundError, match="no run record"):
        registry.promote("a" * 64, "fam-a")
    with pytest.raises(ValueError, match="not found"):
        registry.promote("a" * 64)


def test_promote_resolution_ambiguous_raises(registry) -> None:
    experiment_store.record_run("fam-a", "a" * 64, {"scorecard": {}})
    experiment_store.record_run("fam-b", "a" * 64, {"scorecard": {}})
    with pytest.raises(ValueError, match="ambiguous"):
        registry.promote("a" * 64)


def test_promote_rejects_non_hex_run_id(registry) -> None:
    with pytest.raises(ValueError, match="run_id"):
        registry.promote("../../etc/passwd", "fam-a")


def test_promote_if_better_promotes_only_strictly_better(registry) -> None:
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    path, promoted = registry.promote_if_better("a" * 64)
    assert promoted is True
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "a" * 64

    registry.record(_result_with_scorecard("b" * 64, sharpe_ac=0.9))
    path, promoted = registry.promote_if_better("b" * 64)
    assert promoted is True
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "b" * 64

    registry.record(_result_with_scorecard("c" * 64, sharpe_ac=0.85))
    _, promoted = registry.promote_if_better("c" * 64)
    assert promoted is False
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "b" * 64


def test_promote_if_better_explicit_slug_cross_family(registry) -> None:
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.8, name="fam-a"))
    registry.record(_result_with_scorecard("b" * 64, sharpe_ac=0.9, name="fam-b"))
    registry.promote("a" * 64, "fam-a")
    path, promoted = registry.promote_if_better("b" * 64, "fam-b")
    assert promoted is True
    assert json.loads(path.read_text(encoding="utf-8"))["experiment_slug"] == "fam-b"


def test_promote_if_better_direction_lower_is_better_for_drawdown(registry) -> None:
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.5, max_drawdown=0.2))
    registry.promote("a" * 64)

    # Higher drawdown is WORSE on max_drawdown -> must not promote.
    registry.record(_result_with_scorecard("b" * 64, sharpe_ac=0.5, max_drawdown=0.4))
    _, promoted = registry.promote_if_better("b" * 64, metric="max_drawdown")
    assert promoted is False

    registry.record(_result_with_scorecard("c" * 64, sharpe_ac=0.5, max_drawdown=0.1))
    _, promoted = registry.promote_if_better("c" * 64, metric="max_drawdown")
    assert promoted is True


def test_promote_if_better_champion_without_metric_is_displaced(registry) -> None:
    registry.record(_result("a" * 64, sharpe=0.9))  # champion without scorecard
    registry.promote("a" * 64)

    registry.record(_result_with_scorecard("b" * 64, sharpe_ac=0.4))
    _, promoted = registry.promote_if_better("b" * 64)
    assert promoted is True


def test_promote_if_better_refuses_candidate_without_metric(registry) -> None:
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    registry.promote("a" * 64)
    registry.record(_result("b" * 64, sharpe=9.9))  # legacy candidate, no scorecard
    with pytest.raises(ValueError, match="scorecard"):
        registry.promote_if_better("b" * 64)


def test_promote_if_better_unknown_metric_raises(registry) -> None:
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    with pytest.raises(ValueError, match="metric"):
        registry.promote_if_better("a" * 64, metric="nope")


def test_resolve_champion(registry) -> None:
    assert registry.resolve_champion() is None
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    registry.promote("a" * 64, "fam-a")
    assert registry.resolve_champion() == ("a" * 64, "fam-a")


def test_resolve_champion_dangling_pointer_fails_loud(registry) -> None:
    champion = paths.champion_path()
    champion.parent.mkdir(parents=True, exist_ok=True)
    champion.write_text(
        json.dumps({"run_id": "b" * 64, "experiment_slug": "fam-a"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dangles"):
        registry.resolve_champion()


def test_resolve_champion_corrupt_pointer_fails_loud(registry) -> None:
    champion = paths.champion_path()
    champion.parent.mkdir(parents=True, exist_ok=True)
    champion.write_text(json.dumps({"run_id": "b" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        registry.resolve_champion()


def test_promote_if_better_dangling_champion_pointer_fails_loud(registry) -> None:
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    paths.champion_path().write_text(
        json.dumps({"run_id": "b" * 64, "experiment_slug": "fam-a"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dangles"):
        registry.promote_if_better("a" * 64)


def test_champion_pointer_write_is_atomic(
    registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    registry.promote("a" * 64, "fam-a")
    stable = paths.champion_path().read_text(encoding="utf-8")

    import nmr._atomicio as atomicio_module

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(atomicio_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        registry.promote("a" * 64, "fam-a")

    assert paths.champion_path().read_text(encoding="utf-8") == stable
