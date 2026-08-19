"""Submission building, oracle-backed validation, and artifact acceptance.

Role split (D5/D6' of the audit remediation):
- ``validate_submission`` — the Phase D acceptance gate: wraps Numerai's
  official local validator. Every promoted artifact's RAW output must pass it.
- ``accept_promoted_artifact`` — loads a deploy artifact, calls the real
  contract with both arguments, and runs the raw ``prediction`` column through
  the official validator. The raw output is never passed through
  ``build_submission`` (that would re-rank and repair it before the validator
  sees it — reproducing SEV-1 #14 with a green test on top).
- ``build_submission`` / ``write_submission`` — the manual-CSV contingency
  channel (operational fallback if Model Uploads is unavailable).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from nmr._transforms import tie_kept_rank

__all__ = [
    "accept_promoted_artifact",
    "build_submission",
    "validate_submission",
    "write_submission",
]


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


def accept_promoted_artifact(
    artifact_path: str | Path,
    *,
    live_features: pd.DataFrame,
    live_benchmark_models: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Acceptance gate (SEV-1 #14): validate a promoted artifact's RAW contract
    output against the official validator.

    Loads the artifact via ``load_predict``, invokes the real contract with
    BOTH arguments — ``predict(live_features, live_benchmark_models)``, the
    second sourced from ``live_benchmark_models.parquet`` — and runs the raw
    returned ``prediction`` column through ``numerai_tools`` validation
    (headers, values strictly in (0,1), ids against the live universe).

    **The masking trap (enforced):** the raw output is NEVER passed through
    ``build_submission`` — that would re-rank to (0,1) and repair the output
    before the validator sees it, reproducing the original defect with a green
    test on top. For Model Uploads the closure's return value is consumed
    verbatim, so the raw output IS the submission.

    Returns the raw predictions frame for diagnostics. Raises ``ValueError``
    on any contract violation. Failure here is NOT overridable by the
    promotion writer's ``override_gate`` — that flag covers tier-4
    *performance*, never contract *validity*.
    """
    from nmr.deployment import load_predict

    predict_fn = load_predict(artifact_path)
    preds = predict_fn(live_features, live_benchmark_models)
    if not isinstance(preds, pd.DataFrame):
        raise ValueError(
            f"predict must return a pandas DataFrame, got {type(preds).__name__}"
        )
    if list(preds.columns) != ["prediction"]:
        raise ValueError(
            "predict must return exactly a 'prediction' column, "
            f"got {list(preds.columns)}"
        )
    if not preds.index.equals(live_features.index):
        raise ValueError(
            "predict must preserve the live feature index/row order "
            "(raw output is consumed verbatim)"
        )
    values = preds["prediction"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "invalid_submission_values: raw predictions must be finite"
        )
    if not ((values > 0) & (values < 1)).all():
        raise ValueError(
            "invalid_submission_values: raw predictions must be strictly in "
            "(0,1); the deploy closure's per-era tie_kept_rank step is missing?"
        )
    live_ids = [str(value) for value in live_features.index.to_list()]
    submission = pl.DataFrame(
        {
            "id": live_ids,
            "prediction": values.tolist(),
        }
    )
    # Official validator: values, headers, ids against the live universe.
    validate_submission(submission, live_ids=live_ids)
    return preds


def write_submission(submission: pl.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.select(["id", "prediction"]).write_csv(output_path)
    return output_path
