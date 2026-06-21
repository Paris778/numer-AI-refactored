"""Unit tests for EvaluationEngine summary and boundary behavior."""

from __future__ import annotations

import math

import polars as pl
import pytest

from nmr.evaluation import EvaluationEngine, MetricSummary


def test_summarize_known_values() -> None:
    engine = EvaluationEngine("custom")
    summary = engine.summarize({"1": 1.0, "2": -2.0, "3": 1.0})

    assert isinstance(summary, MetricSummary)
    assert summary.mean == pytest.approx(0.0)
    assert summary.std == pytest.approx(math.sqrt(2.0))
    assert summary.sharpe == pytest.approx(0.0)
    assert summary.max_drawdown == pytest.approx(2.0)


def test_summarize_zero_std_sets_sharpe_to_zero() -> None:
    engine = EvaluationEngine("custom")
    summary = engine.summarize({"1": 2.0, "2": 2.0, "3": 2.0})
    assert summary.std == pytest.approx(0.0)
    assert summary.sharpe == pytest.approx(0.0)
    assert summary.max_drawdown == pytest.approx(0.0)


def test_summarize_requires_non_empty_mapping() -> None:
    engine = EvaluationEngine("custom")
    with pytest.raises(ValueError, match="at least one era score"):
        engine.summarize({})


def test_per_era_keys_are_numeric_ordered() -> None:
    engine = EvaluationEngine("custom")
    df = pl.DataFrame(
        {
            "era": ["10", "2", "1", "2", "10", "1"],
            "pred": [0.9, 0.3, 0.2, 0.8, 0.1, 0.7],
            "target": [0.2, 0.5, 0.7, 0.1, 0.9, 0.3],
        }
    )

    result = engine.per_era_corr(df, pred_col="pred", target_col="target")
    assert list(result) == ["1", "2", "10"]


@pytest.mark.parametrize("backend", ["custom", "official"])
def test_degenerate_corr_returns_zero(backend: str) -> None:
    engine = EvaluationEngine(backend)
    df = pl.DataFrame(
        {
            "era": ["1", "1", "1", "1"],
            "pred": [0.5, 0.5, 0.5, 0.5],
            "target": [0.1, 0.2, 0.3, 0.4],
        }
    )
    result = engine.per_era_corr(df, pred_col="pred", target_col="target")
    assert result == {"1": 0.0}


@pytest.mark.parametrize("backend", ["custom", "official"])
def test_degenerate_fnc_returns_zero(backend: str) -> None:
    engine = EvaluationEngine(backend)
    df = pl.DataFrame(
        {
            "era": ["1", "1", "1", "1"],
            "pred": [0.5, 0.5, 0.5, 0.5],
            "target": [0.1, 0.2, 0.3, 0.4],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [2.0, 1.0, 4.0, 3.0],
        }
    )
    result = engine.per_era_fnc(
        df,
        pred_col="pred",
        feature_cols=["f1", "f2"],
        target_col="target",
    )
    assert result == {"1": 0.0}


@pytest.mark.parametrize("backend", ["custom", "official"])
def test_degenerate_mmc_returns_zero(backend: str) -> None:
    engine = EvaluationEngine(backend)
    df = pl.DataFrame(
        {
            "era": ["1", "1", "1", "1"],
            "pred": [0.5, 0.5, 0.5, 0.5],
            "meta": [0.2, 0.4, 0.6, 0.8],
            "target": [0.1, 0.2, 0.3, 0.4],
        }
    )
    result = engine.per_era_mmc(
        df,
        pred_col="pred",
        meta_col="meta",
        target_col="target",
    )
    assert result == {"1": 0.0}


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend="):
        EvaluationEngine("bogus")


def test_empty_feature_cols_raise() -> None:
    engine = EvaluationEngine("custom")
    df = pl.DataFrame({"era": ["1"], "pred": [0.1], "target": [0.2]})
    with pytest.raises(ValueError, match="feature_cols"):
        engine.per_era_fnc(df, pred_col="pred", feature_cols=[], target_col="target")
