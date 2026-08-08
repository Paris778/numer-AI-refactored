from __future__ import annotations

import polars as pl
import pytest

from nmr.evaluation import MIN_OVERLAP_ERAS, NonVacuityError
from nmr.meta import fleet_summary, paired_era_comparison, promotion_verdict


def _frame(n_eras: int = 24) -> pl.DataFrame:
    rows = []
    for era in range(1, n_eras + 1):
        for idx in range(10):
            rows.append({"era": str(era), "id": f"{era}_{idx}", "prediction": idx * 0.1})
    return pl.DataFrame(rows)


def _era_index_metric(frame: pl.DataFrame) -> dict[str, float]:
    """Deterministic per-era metric: the era number itself."""
    return {
        str(era): float(era)
        for era in frame.get_column("era").unique().sort().to_list()
    }


def test_paired_comparison_estimates_mean_difference_with_ci() -> None:
    a = _frame()
    b = _frame()
    result = paired_era_comparison(
        a, b, metric_fn=_era_index_metric, seed=7, n_boot=50,
    )
    assert result.mean_diff == pytest.approx(0.0, abs=1e-9)
    assert result.n_eras == 24
    assert result.device_mismatch is False
    assert result.ci_low <= result.mean_diff <= result.ci_high
    assert result.alpha == 0.05 and result.n_boot == 50


def test_paired_comparison_sign_using_prediction_means() -> None:
    def mean_pred(frame: pl.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        for era in frame.get_column("era").unique().to_list():
            out[str(era)] = float(
                frame.filter(pl.col("era") == era).get_column("prediction").mean()
            )
        return out

    a = _frame()  # prediction = idx * 0.1 -> era mean 0.45
    b = a.with_columns((pl.col("prediction") + 1.0).alias("prediction"))  # era mean 1.45
    result = paired_era_comparison(a, b, metric_fn=mean_pred, seed=7, n_boot=50)
    assert result.mean_diff == pytest.approx(-1.0, abs=1e-9)  # a - b == -1.0


def test_paired_comparison_bootstrap_deterministic_under_seed() -> None:
    a, b = _frame(), _frame()
    r1 = paired_era_comparison(a, b, metric_fn=_era_index_metric, seed=11, n_boot=200)
    r2 = paired_era_comparison(a, b, metric_fn=_era_index_metric, seed=11, n_boot=200)
    assert r1 == r2  # same seed -> identical CI (cross-process determinism)


def test_paired_comparison_intersects_eras_and_raises_below_overlap_floor() -> None:
    a = _frame(n_eras=24)
    b = _frame(n_eras=10)  # overlap = 10 < MIN_OVERLAP_ERAS
    with pytest.raises(NonVacuityError):
        paired_era_comparison(a, b, metric_fn=_era_index_metric, seed=7)


def test_paired_comparison_device_mismatch_flag() -> None:
    result = paired_era_comparison(
        _frame(), _frame(), metric_fn=_era_index_metric, seed=7,
        device_a="gpu", device_b="cpu",
    )
    assert result.device_mismatch is True
    same = paired_era_comparison(
        _frame(), _frame(), metric_fn=_era_index_metric, seed=7,
        device_a="cpu", device_b="cpu",
    )
    assert same.device_mismatch is False


def test_paired_comparison_raises_when_era_col_missing() -> None:
    a = _frame()
    renamed = _frame().rename({"era": "epoch"})  # lacks default era_col "era"
    with pytest.raises(ValueError, match="oof_b"):
        paired_era_comparison(a, renamed, metric_fn=_era_index_metric, seed=7)
    with pytest.raises(ValueError, match="oof_a"):
        paired_era_comparison(renamed, a, metric_fn=_era_index_metric, seed=7)
    with pytest.raises(ValueError, match="oof_a.*oof_b"):
        paired_era_comparison(renamed, renamed, metric_fn=_era_index_metric, seed=7)


def test_paired_comparison_honors_renamed_era_col() -> None:
    def metric_on_epoch(frame: pl.DataFrame) -> dict[str, float]:
        return {
            str(era): float(era)
            for era in frame.get_column("epoch").unique().sort().to_list()
        }

    a = _frame().rename({"era": "epoch"})
    b = _frame().rename({"era": "epoch"})
    result = paired_era_comparison(
        a, b, metric_fn=metric_on_epoch, era_col="epoch", seed=7, n_boot=50,
    )
    assert result.n_eras == 24
    assert result.mean_diff == pytest.approx(0.0, abs=1e-9)


def _entry(run_id: str, metric: str = "corr_sharpe_ac", *, value: float | None = None,
           lo: float | None = None, hi: float | None = None) -> dict:
    scorecard: dict = {}
    if value is not None:
        scorecard[metric] = value
    if lo is not None:
        scorecard[f"{metric}_ci_low"] = lo
    if hi is not None:
        scorecard[f"{metric}_ci_high"] = hi
    return {"run_id": run_id, "scorecard": scorecard}


def test_verdict_promotes_when_candidate_ci_clears_champion() -> None:
    champion = _entry("c" * 64, value=0.10, lo=0.05, hi=0.15)
    candidate = _entry("d" * 64, value=0.25, lo=0.20, hi=0.30)
    assert promotion_verdict(candidate, champion) == "promote"


def test_verdict_holds_when_candidate_ci_below_champion() -> None:
    champion = _entry("c" * 64, value=0.25, lo=0.20, hi=0.30)
    candidate = _entry("d" * 64, value=0.10, lo=0.05, hi=0.15)
    assert promotion_verdict(candidate, champion) == "hold"


def test_verdict_cautions_on_ci_overlap() -> None:
    champion = _entry("c" * 64, value=0.18, lo=0.10, hi=0.26)
    candidate = _entry("d" * 64, value=0.20, lo=0.14, hi=0.27)
    assert promotion_verdict(candidate, champion) == "caution"


def test_verdict_cautions_when_ci_unavailable() -> None:
    champion = _entry("c" * 64, value=0.10)
    candidate = _entry("d" * 64, value=0.25)
    assert promotion_verdict(candidate, champion) == "caution"


def test_verdict_promotes_without_champion() -> None:
    candidate = _entry("d" * 64, value=0.25, lo=0.20, hi=0.30)
    assert promotion_verdict(candidate, None) == "promote"


def test_verdict_lower_is_better_for_max_drawdown() -> None:
    champion = _entry("c" * 64, metric="max_drawdown", value=0.20, lo=0.18, hi=0.22)
    candidate = _entry("d" * 64, metric="max_drawdown", value=0.10, lo=0.08, hi=0.12)
    assert promotion_verdict(candidate, champion, metric="max_drawdown") == "promote"


def test_verdict_directions_match_registry_semantics() -> None:
    from nmr.meta import _VERDICT_DIRECTIONS
    from nmr.registry import _SCORECARD_METRIC_DIRECTION

    assert set(_VERDICT_DIRECTIONS) <= set(_SCORECARD_METRIC_DIRECTION)
    for metric, higher_is_better in _VERDICT_DIRECTIONS.items():
        assert _SCORECARD_METRIC_DIRECTION[metric] == higher_is_better


def test_verdict_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="metric"):
        promotion_verdict(_entry("d" * 64), None, metric="bogus")


def _full_entry(run_id: str, sharpe_ac: float) -> dict:
    return {
        "run_id": run_id,
        "manifest": {
            "oof_device": "cpu",
            "config": {
                "data": {"feature_set": "small", "feature_subset": None},
                "model": {"preset": "fast"},
                "risk": {"neutralization_proportion": 1.0},
            },
        },
        "scorecard": {
            "corr_sharpe_ac": sharpe_ac,
            "corr_sharpe_ac_ci_low": sharpe_ac - 0.05,
            "corr_sharpe_ac_ci_high": sharpe_ac + 0.05,
            "corr_sharpe_ac_n_eras": 30,
            "deflated_sharpe": 0.98,
            "max_feature_exposure": 0.3,
            "bmc": 0.02,
            "horizon_model_sharpe_20": 0.5,
            "perturb_ceiling_stability": 0.9,
            "regime_count": 3,
        },
    }


def test_fleet_summary_columns_and_flags() -> None:
    runs = [_full_entry("a" * 64, 0.12), _full_entry("b" * 64, 0.05)]
    frame = fleet_summary(runs, n_trials=2)
    assert frame.height == 2
    assert set(frame.columns) >= {
        "run_id", "metric", "metric_ci_low", "metric_ci_high", "metric_n_eras",
        "deflated_sharpe", "dsr_pass", "max_feature_exposure", "oof_device",
        "preset", "feature_set", "feature_subset", "neutralization_proportion",
        "has_bmc", "has_horizon", "has_perturb", "has_regime",
        "policy_n_trials", "policy_dsr_confidence",
    }
    first = frame.filter(pl.col("run_id") == "a" * 64).row(0, named=True)
    assert first["dsr_pass"] is True
    assert first["oof_device"] == "cpu"
    assert first["preset"] == "fast"
    assert first["feature_set"] == "small"
    assert first["has_bmc"] is True and first["has_horizon"] is True
    assert first["has_perturb"] is True and first["has_regime"] is True
    assert first["policy_n_trials"] == 2
    # sorted by metric desc, run_id tiebreak
    assert frame.get_column("run_id").to_list() == ["a" * 64, "b" * 64]


def test_fleet_summary_flags_legacy_runs_without_scorecard() -> None:
    legacy = {
        "run_id": "c" * 64,
        "manifest": {"oof_device": "cpu", "config": {
            "data": {"feature_set": "all", "feature_subset": None},
            "model": {"preset": "deep"},
            "risk": {"neutralization_proportion": 0.5},
        }},
        "scorecard": None,
    }
    frame = fleet_summary([legacy], n_trials=1)
    row = frame.row(0, named=True)
    assert row["metric"] is None
    assert row["dsr_pass"] is False
    assert row["has_bmc"] is False
    assert row["preset"] == "deep" and row["neutralization_proportion"] == 0.5


def test_fleet_summary_validates_policy_arguments() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        fleet_summary([], n_trials=0)
    with pytest.raises(ValueError, match="dsr_confidence"):
        fleet_summary([], n_trials=1, dsr_confidence=1.5)
