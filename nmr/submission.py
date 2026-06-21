"""Submission building and oracle-backed Numerai validation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from nmr._transforms import tie_kept_rank

__all__ = ["build_submission", "validate_submission", "write_submission"]


def build_submission(
    predictions: pl.DataFrame,
    *,
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Return a deterministic `id,prediction` submission frame.

    Predictions are always converted to percentile ranks in `(0, 1)`.
    IDs must be unique and predictions must be finite.
    """
    if id_col not in predictions.columns or pred_col not in predictions.columns:
        raise ValueError(f"predictions must contain {id_col!r} and {pred_col!r}")

    submission = predictions.select([id_col, pred_col]).rename(
        {id_col: "id", pred_col: "prediction"}
    )
    if submission.get_column("id").is_null().any():
        raise ValueError("submission ids must not contain nulls")
    if submission.get_column("id").n_unique() != submission.height:
        raise ValueError("submission ids must be unique")

    pred_values = submission.get_column("prediction").cast(pl.Float64).to_numpy()
    if not np.all(np.isfinite(pred_values)):
        raise ValueError("submission predictions must be finite")

    pred_values = tie_kept_rank(pred_values)

    return (
        submission.with_columns(
            [
                pl.col("id").cast(pl.Utf8),
                pl.Series("prediction", np.asarray(pred_values, dtype=float)),
            ]
        )
        .sort("id")
        .select(["id", "prediction"])
    )


def validate_submission(submission: pl.DataFrame, *, live_ids: Sequence[str]) -> None:
    """Validate a submission against Numerai's official local validator."""
    from numerai_tools.submissions import validate_submission_numerai

    pdf = submission.select(["id", "prediction"]).to_pandas()
    universe = pd.Series([str(value) for value in live_ids], name="id")

    try:
        _, _, filtered_sub, invalid_tickers = validate_submission_numerai(universe, pdf)
    except AssertionError as exc:
        raise ValueError(str(exc)) from exc

    if invalid_tickers:
        extras = sorted(str(value) for value in invalid_tickers)
        raise ValueError(
            "invalid_submission_ids: ids outside live universe detected: "
            f"{extras[:5]}"
        )

    filtered_ids = set(filtered_sub["id"].astype(str).tolist())
    expected_ids = set(universe.astype(str).tolist())
    missing_ids = sorted(expected_ids.difference(filtered_ids))
    if missing_ids:
        raise ValueError(
            "invalid_submission_ids: missing live ids in submission: "
            f"{missing_ids[:5]}"
        )


def write_submission(submission: pl.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.select(["id", "prediction"]).write_csv(output_path)
    return output_path
