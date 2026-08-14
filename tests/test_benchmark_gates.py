"""Gate, orthogonality, and inference-helper tests for the benchmark module.

These cover the mathematically meaningful parts of ``nmr.benchmark`` that the
real-data slice tests reach only indirectly: the null-floor / monotone gates,
book orthogonality (correlation of candidate scores vs a book of scores), and
the tutorial id-column inference (F-024 fallback warning).
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import pytest

from nmr.benchmark import (
    NULL_BASELINES,
    BenchmarkSuite,
    _infer_id_column,
    assert_null_floor,
    assert_slice1_monotone,
)
from nmr.scorecard import MetricCell, MetricScorecard


def _scorecard(
    *,
    rank_scalar: float = 0.0,
    corr: float = 0.0,
    mmc: float = 0.0,
    fnc: float = 0.0,
    corr_sharpe_ac: float = 0.0,
    bmc: MetricCell | None = None,
    cwmm: MetricCell | None = None,
) -> MetricScorecard:
    cell = lambda v: MetricCell(value=v, ci_low=None, ci_high=None, n_eras=10)
    return MetricScorecard(
        model_id="m", n_eras=10, rank_scalar=rank_scalar, deflated_sharpe=0.0,
        mean_payout=cell(0.0), corr=cell(corr), mmc=cell(mmc), fnc=fnc,
        corr_sharpe_ac=cell(corr_sharpe_ac), cvar5=0.0, max_drawdown=0.0,
        burn_rate=0.0, mmc_sharpe_ac=0.0, sortino=0.0, calmar=0.0,
        std_corr=0.1, max_burn_streak=0, time_to_recovery=0,
        horizon_stability=None, horizon_reason=None, regime_corr=None,
        regime_reason=None, perturbation=None, max_feature_exposure=0.0,
        bmc=bmc, bmc_reason=None, cwmm=cwmm, cwmm_reason=None,
        book_correlation=None,
        cagr_1y=0.0, gain_to_pain_ratio=0.0, kelly_fraction=0.0,
        mmc_down=None, mmc_down_n_eras=0, mmc_down_reason=None,
        turnover_mean=None, turnover_std=None, turnover_reason=None,
        sim_portfolio_cagr=0.0, sim_portfolio_mdd=0.0,
        sim_capital_utilization=0.0,
        metric_timing_seconds=None, eval_total_seconds=0.0,
    )


def _null_scorecards() -> dict[str, MetricScorecard]:
    return {name: _scorecard() for name in NULL_BASELINES}


def test_assert_null_floor_passes_zero_baselines() -> None:
    assert_null_floor(_null_scorecards())


def test_assert_null_floor_rejects_high_corr() -> None:
    scorecards = _null_scorecards()
    scorecards["constant-0.5"] = _scorecard(corr=0.2)
    with pytest.raises(ValueError, match="Null floor violation"):
        assert_null_floor(scorecards)


def test_assert_null_floor_requires_all_baselines() -> None:
    scorecards = {name: _scorecard() for name in NULL_BASELINES[:-1]}
    with pytest.raises(ValueError, match="Missing null baseline"):
        assert_null_floor(scorecards)


def test_assert_null_floor_honors_custom_tolerance() -> None:
    scorecards = _null_scorecards()
    scorecards["constant-0.5"] = _scorecard(corr=0.2)
    assert_null_floor(scorecards, metric_tolerances={"corr": 0.25})


def test_assert_slice1_monotone_orderings() -> None:
    # null floor = 0.0 (all null baselines rank_scalar 0); hello = 0.05;
    # sunshine = 0.10 -> valid ordering.
    scorecards = {
        **{name: _scorecard() for name in NULL_BASELINES},
        "hello-numerai": _scorecard(rank_scalar=0.05),
        "sunshine": _scorecard(rank_scalar=0.10),
    }
    assert_slice1_monotone(scorecards)

    scorecards["hello-numerai"] = _scorecard(rank_scalar=-0.02)
    with pytest.raises(ValueError, match="hello below null floor"):
        assert_slice1_monotone(scorecards)

    scorecards["hello-numerai"] = _scorecard(rank_scalar=0.12)
    with pytest.raises(ValueError, match="sunshine below hello"):
        assert_slice1_monotone(scorecards)


def test_assert_slice1_monotone_missing_models() -> None:
    with pytest.raises(ValueError, match="Missing required scorecard"):
        assert_slice1_monotone({"hello-numerai": _scorecard()})


def _orthogonality_suite() -> BenchmarkSuite:
    rows = []
    for era in range(1, 13):
        for idx in range(4):
            rows.append(
                {
                    "era": str(era),
                    "id": f"{era}_{idx}",
                    "f1": float(idx) / 10.0,
                    "f2": float((idx % 2)) / 10.0,
                    "target": float(era) / 100.0,
                }
            )
    frame = pl.DataFrame(rows)
    return BenchmarkSuite(
        meta_model=frame.select(["era", "id"]).with_columns(
            pl.lit(0.1).alias("numerai_meta_model")
        ),
        benchmarks=pl.DataFrame(
            {"era": [], "id": [], "bench": []},
            schema={"era": pl.String, "id": pl.String, "bench": pl.Float64},
        ),
        features=frame.select(["era", "id", "f1", "f2"]),
        targets=frame.select(["era", "id", "target"]),
        n_trials=1,
        seed=7,
        horizon="20D",
        n_boot=5,
        min_overlap_eras=2,
    )


def test_book_orthogonality_self_correlation_is_one() -> None:
    suite = _orthogonality_suite()
    x = np.linspace(-1.0, 1.0, 30)  # n >= MIN_OVERLAP_ERAS (20)
    out = suite.compute_book_orthogonality(x, x, seed=7, n_boot=20)
    assert out["rho_global"] == pytest.approx(1.0, abs=1e-12)
    assert out["rho_tail"] == pytest.approx(1.0, abs=1e-12)
    assert out["spread"] == pytest.approx(0.0, abs=1e-12)
    assert out["n_eras"] == 30.0


def test_book_orthogonality_negative_uncorrelated_and_deterministic() -> None:
    suite = _orthogonality_suite()
    rng = np.random.default_rng(5)
    cand = np.linspace(-1.0, 1.0, 30)
    book = rng.normal(size=30)
    a = suite.compute_book_orthogonality(cand, -cand, seed=11, n_boot=20)
    b = suite.compute_book_orthogonality(cand, -cand, seed=11, n_boot=20)
    assert a["rho_global"] == pytest.approx(-1.0, abs=1e-12)
    assert a == b  # deterministic


def test_book_orthogonality_rejects_length_mismatch_and_short_input() -> None:
    suite = _orthogonality_suite()
    with pytest.raises(ValueError, match="same length"):
        suite.compute_book_orthogonality(np.ones(5), np.ones(4), seed=1, n_boot=5)
    with pytest.raises(ValueError, match="Non-vacuity"):
        suite.compute_book_orthogonality(np.ones(5), np.ones(5), seed=1, n_boot=5)


def test_infer_id_column_aliases_and_fallback_warning(caplog) -> None:
    assert _infer_id_column(["prediction", "era", "ID"], pred_col="prediction", era_col="era") == "ID"
    assert _infer_id_column(["prediction", "era"], pred_col="prediction", era_col="era") is None

    with caplog.at_level(logging.WARNING, logger="nmr.benchmark"):
        inferred = _infer_id_column(
            ["prediction", "era", "user_id"], pred_col="prediction", era_col="era"
        )
    assert inferred == "user_id"
    assert any("no known id alias" in record.message for record in caplog.records)
