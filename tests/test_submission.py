"""Tests for submission building and oracle-backed validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nmr._transforms import tie_kept_rank
from nmr.deployment import serialize_predict
from nmr.submission import (
    accept_promoted_artifact,
    build_submission,
    validate_submission,
    write_submission,
)


def _raw_predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["id3", "id1", "id2", "id4"],
            "prediction": [3.5, -1.2, 7.1, 0.3],
        }
    )


def test_build_submission_is_deterministic_and_bounded() -> None:
    first = build_submission(_raw_predictions())
    second = build_submission(_raw_predictions())

    assert first.equals(second)
    assert first.columns == ["id", "prediction"]
    assert first.get_column("prediction").min() > 0.0
    assert first.get_column("prediction").max() < 1.0
    assert first.get_column("id").to_list() == sorted(first.get_column("id").to_list())


def test_validate_submission_accepts_valid_submission() -> None:
    submission = build_submission(_raw_predictions())
    validate_submission(submission, live_ids=submission.get_column("id").to_list())


@pytest.mark.parametrize(
    ("submission", "live_ids", "message"),
    [
        (
            pl.DataFrame({"id": ["id1", "id2"], "prediction": [0.1, None]}),
            ["id1", "id2"],
            "invalid_submission_values",
        ),
        (
            pl.DataFrame({"id": ["id1", "id2"], "prediction": [0.1, 1.2]}),
            ["id1", "id2"],
            "invalid_submission_values",
        ),
        (
            pl.DataFrame({"id": ["id1", "id1"], "prediction": [0.1, 0.2]}),
            ["id1", "id2"],
            "invalid_submission_ids",
        ),
        (
            pl.DataFrame({"id": ["id1"], "prediction": [0.1]}),
            ["id1", "id2"],
            "invalid_submission_ids",
        ),
        (
            pl.DataFrame(
                {"id": ["id1", "id2", "extra"], "prediction": [0.1, 0.9, 0.4]}
            ),
            ["id1", "id2"],
            "invalid_submission_ids",
        ),
    ],
)
def test_validate_submission_rejects_invalid_cases(
    submission: pl.DataFrame,
    live_ids: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_submission(submission, live_ids=live_ids)


def test_write_submission_writes_csv(tmp_path) -> None:
    submission = build_submission(_raw_predictions())
    path = write_submission(submission, tmp_path / "submission.csv")

    assert path == tmp_path / "submission.csv"
    assert Path(path).exists()
    written = pl.read_csv(path)
    assert written.equals(submission)


def test_build_submission_ranks_inputs_with_exact_zero_and_one_to_open_interval() -> (
    None
):
    raw = pl.DataFrame(
        {
            "id": ["id1", "id2", "id3", "id4"],
            "prediction": [0.0, 1.0, 0.25, 0.75],
        }
    )
    submission = build_submission(raw)

    assert submission.get_column("prediction").min() > 0.0
    assert submission.get_column("prediction").max() < 1.0
    validate_submission(submission, live_ids=submission.get_column("id").to_list())


def _ranked_predict_fn() -> object:
    def predict(
        live_features: pd.DataFrame,
        live_benchmark_models: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        del live_benchmark_models
        signal = (
            np.asarray(live_features.iloc[:, 0], dtype=float)
            - np.asarray(live_features.iloc[:, 1], dtype=float)
        )
        # The fixed closure's final per-era (0,1) step.
        return pd.DataFrame(
            {"prediction": tie_kept_rank(signal)}, index=live_features.index
        )

    return predict


def _unranked_predict_fn() -> object:
    """The pre-fix closure: neutralize and stop — output is unbounded."""

    def predict(
        live_features: pd.DataFrame,
        live_benchmark_models: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        del live_benchmark_models
        signal = (
            np.asarray(live_features.iloc[:, 0], dtype=float)
            - np.asarray(live_features.iloc[:, 1], dtype=float)
        )
        return pd.DataFrame({"prediction": signal}, index=live_features.index)

    return predict


def _live_fixture(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        },
        index=[f"id_{i}" for i in range(n)],
    )


def test_accept_promoted_artifact_valid(tmp_path: Path) -> None:
    artifact = serialize_predict(
        _ranked_predict_fn(), path=tmp_path / "predict.pkl", feature_names=["f1", "f2"]
    )
    live = _live_fixture()
    bench = pd.DataFrame({"dummy": [0.1] * live.shape[0]}, index=live.index)
    raw = accept_promoted_artifact(
        artifact.path, live_features=live, live_benchmark_models=bench
    )
    assert list(raw.columns) == ["prediction"]
    values = raw["prediction"].to_numpy()
    assert ((values > 0) & (values < 1)).all()
    assert np.isfinite(values).all()


def test_accept_promoted_artifact_rejects_unranked_output(tmp_path: Path) -> None:
    """The masking trap + SEV-1 #14 regression guard: the raw output must pass
    the validator UNAIDED. If the closure's (0,1) step is ever removed, this
    gate fails — it must never repair the output before validating it."""
    artifact = serialize_predict(
        _unranked_predict_fn(), path=tmp_path / "predict.pkl", feature_names=["f1", "f2"]
    )
    with pytest.raises(ValueError, match=r"\(0,1\)"):
        accept_promoted_artifact(
            artifact.path, live_features=_live_fixture(), live_benchmark_models=None
        )


def test_accept_promoted_artifact_rejects_nonfinite_and_wrong_columns(
    tmp_path: Path,
) -> None:
    def nan_predict(
        live_features: pd.DataFrame,
        live_benchmark_models: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        del live_benchmark_models
        values = np.full(len(live_features), np.nan)
        return pd.DataFrame({"prediction": values}, index=live_features.index)

    artifact = serialize_predict(
        nan_predict, path=tmp_path / "predict.pkl", feature_names=["f1", "f2"]
    )
    with pytest.raises(ValueError, match="finite"):
        accept_promoted_artifact(
            artifact.path, live_features=_live_fixture(), live_benchmark_models=None
        )

    def wrong_cols_predict(
        live_features: pd.DataFrame,
        live_benchmark_models: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        del live_benchmark_models
        return pd.DataFrame(
            {"not_prediction": [0.5] * len(live_features)}, index=live_features.index
        )

    artifact2 = serialize_predict(
        wrong_cols_predict, path=tmp_path / "predict2.pkl", feature_names=["f1", "f2"]
    )
    with pytest.raises(ValueError, match="prediction"):
        accept_promoted_artifact(
            artifact2.path, live_features=_live_fixture(), live_benchmark_models=None
        )
