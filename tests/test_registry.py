"""Tests for run registry persistence and promotion semantics."""

from __future__ import annotations

import json

import polars as pl
import pytest

from nmr.evaluation import MetricSummary
from nmr.registry import RunRegistry
from nmr.runner import RunResult
from nmr.scorecard import MetricCell, MetricScorecard


def _result(run_id: str, sharpe: float) -> RunResult:
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
        manifest={"run_id": run_id},
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
    run_id: str, sharpe_ac: float, *, max_drawdown: float = 0.1
) -> RunResult:
    result = _result(run_id, sharpe=0.5)
    return RunResult(
        run_id=result.run_id, oof=result.oof, metrics=result.metrics,
        artifact=result.artifact, manifest=result.manifest,
        scorecard=_scorecard(sharpe_ac, max_drawdown=max_drawdown),
    )


def test_record_is_idempotent_and_writes_json_atomically(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    result = _result("a" * 64, sharpe=0.7)

    run_dir = registry.record(result)
    original = (run_dir / "run.json").read_text(encoding="utf-8")
    run_dir_again = registry.record(result)
    repeated = (run_dir_again / "run.json").read_text(encoding="utf-8")

    assert run_dir == run_dir_again
    assert original == repeated


def test_atomic_write_failure_keeps_previous_run_json(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = RunRegistry(tmp_path)
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


def test_list_best_and_promote_are_deterministic_and_idempotent(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.record(_result("a" * 64, sharpe=0.3))
    registry.record(_result("b" * 64, sharpe=0.8))

    listed = registry.list()
    assert {entry["run_id"] for entry in listed} == {"a" * 64, "b" * 64}

    best = registry.best("sharpe")
    assert best is not None
    assert best["run_id"] == "b" * 64

    run_b_json_before = (tmp_path / ("b" * 64) / "run.json").read_text(
        encoding="utf-8"
    )
    champion_path = registry.promote("b" * 64)
    champion_again = registry.promote("b" * 64)
    assert champion_path == champion_again
    assert json.loads(champion_path.read_text(encoding="utf-8")) == {
        "run_id": "b" * 64
    }

    run_b_json_after = (tmp_path / ("b" * 64) / "run.json").read_text(
        encoding="utf-8"
    )
    assert run_b_json_before == run_b_json_after


def test_best_returns_none_for_empty_registry(tmp_path) -> None:
    assert RunRegistry(tmp_path).best() is None


def test_promote_if_better_promotes_only_strictly_better(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    registry.promote("a" * 64)

    registry.record(_result_with_scorecard("b" * 64, sharpe_ac=0.9))
    path, promoted = registry.promote_if_better("b" * 64)
    assert promoted is True
    assert json.loads(path.read_text(encoding="utf-8")) == {"run_id": "b" * 64}

    registry.record(_result_with_scorecard("c" * 64, sharpe_ac=0.85))
    _, promoted = registry.promote_if_better("c" * 64)
    assert promoted is False
    assert json.loads(path.read_text(encoding="utf-8")) == {"run_id": "b" * 64}


def test_promote_if_better_direction_lower_is_better_for_drawdown(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.5, max_drawdown=0.2))
    registry.promote("a" * 64)

    # Higher drawdown is WORSE on max_drawdown -> must not promote.
    registry.record(_result_with_scorecard("b" * 64, sharpe_ac=0.5, max_drawdown=0.4))
    _, promoted = registry.promote_if_better("b" * 64, metric="max_drawdown")
    assert promoted is False

    registry.record(_result_with_scorecard("c" * 64, sharpe_ac=0.5, max_drawdown=0.1))
    _, promoted = registry.promote_if_better("c" * 64, metric="max_drawdown")
    assert promoted is True


def test_promote_if_better_legacy_champion_is_displaced(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.record(_result("a" * 64, sharpe=0.9))  # no scorecard
    registry.promote("a" * 64)

    registry.record(_result_with_scorecard("b" * 64, sharpe_ac=0.4))
    _, promoted = registry.promote_if_better("b" * 64)
    assert promoted is True


def test_promote_if_better_refuses_legacy_candidate(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    registry.promote("a" * 64)
    registry.record(_result("b" * 64, sharpe=9.9))  # legacy candidate, no scorecard
    with pytest.raises(ValueError, match="scorecard"):
        registry.promote_if_better("b" * 64)


def test_promote_rejects_non_hex_run_id(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    with pytest.raises(ValueError, match="run_id"):
        registry.promote("../../etc/passwd")


def test_promote_if_better_unknown_metric_raises(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    with pytest.raises(ValueError, match="metric"):
        registry.promote_if_better("a" * 64, metric="nope")


def test_best_validates_metric_name(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    registry.record(_result("a" * 64, sharpe=0.5))
    with pytest.raises(ValueError, match="metric"):
        registry.best("nope")


def test_promote_if_better_corrupted_champion_pointer_is_treated_as_no_champion(
    tmp_path,
) -> None:
    registry = RunRegistry(tmp_path)
    registry.record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    champion_path = tmp_path / "champion.json"
    champion_path.write_text(json.dumps({}), encoding="utf-8")  # no run_id key

    path, promoted = registry.promote_if_better("a" * 64)

    assert promoted is True
    assert json.loads(path.read_text(encoding="utf-8")) == {"run_id": "a" * 64}


def test_record_persists_validation_predictions(tmp_path) -> None:
    registry = RunRegistry(tmp_path)
    result = _result("c" * 64, sharpe=0.4)
    result = RunResult(
        run_id=result.run_id,
        oof=result.oof,
        metrics=result.metrics,
        artifact=result.artifact,
        manifest=result.manifest,
        validation_predictions=pl.DataFrame(
            {"era": ["0575", "0575"], "id": ["x", "y"], "prediction": [0.2, 0.8]}
        ),
    )
    run_dir = registry.record(result)
    persisted = pl.read_parquet(run_dir / "validation_preds.parquet")
    assert persisted.height == 2
    assert persisted["prediction"].to_list() == [0.2, 0.8]
    # no validation predictions -> no file
    assert not (tmp_path / ("d" * 64) / "validation_preds.parquet").exists()
