from __future__ import annotations

import polars as pl
import pytest

from nmr.evaluation import MIN_OVERLAP_ERAS, NonVacuityError
from nmr.meta import paired_era_comparison


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
