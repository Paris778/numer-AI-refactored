"""Tests for submission building and oracle-backed validation."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nmr.submission import build_submission, validate_submission, write_submission


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
