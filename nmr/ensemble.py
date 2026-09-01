"""Rank-domain blending for leakage-safe Numerai ensembles."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy.optimize import nnls

from nmr._transforms import rank_gaussianize, rank_gaussianize_unit_variance

logger = logging.getLogger("nmr.ensemble")

__all__ = ["Ensembler"]

# Ridge weight-learning regularization (audit SEV-3 "no magic values"): the
# ridge penalty on the mean-squared-error objective over the rank-gaussianized
# OOF design. Normalizing X'X and X'y by row count keeps alpha's meaning stable
# when the same era population is replicated or the validation window grows.
_RIDGE_ALPHA = 1.0


class Ensembler:
    @staticmethod
    def rank_normalize(
        df: pl.DataFrame,
        *,
        pred_cols: Sequence[str],
        era_col: str = "era",
    ) -> pl.DataFrame:
        pred_list = _validate_pred_cols(pred_cols)
        return _apply_per_era(
            df, pred_cols=pred_list, era_col=era_col, transform=_rank_block
        )

    def blend(
        self,
        df: pl.DataFrame,
        *,
        pred_cols: Sequence[str],
        weights: Sequence[float] | None = None,
        era_col: str = "era",
        out_col: str = "prediction",
    ) -> pl.DataFrame:
        pred_list = _validate_pred_cols(pred_cols)
        weight_array = _resolve_weights(pred_list, weights)
        normalized = self.rank_normalize(df, pred_cols=pred_list, era_col=era_col)

        return _apply_per_era(
            normalized,
            pred_cols=pred_list,
            era_col=era_col,
            transform=lambda block, *, pred_cols: _blend_block(
                block,
                pred_cols=pred_cols,
                weights=weight_array,
                out_col=out_col,
            ),
        )

    def learn_weights(
        self,
        oof_df: pl.DataFrame,
        *,
        pred_cols: Sequence[str],
        target_col: str,
        era_col: str = "era",
        method: str = "ridge",
    ) -> tuple[float, ...]:
        logger.info(
            "[learn_weights] learning weights for %d predictions via %s",
            len(pred_cols),
            method,
        )
        pred_list = _validate_pred_cols(pred_cols)
        clean = oof_df.select([era_col, *pred_list, target_col]).drop_nulls()
        if clean.is_empty():
            raise ValueError("oof_df has no usable rows after dropping nulls")

        normalized = self.rank_normalize(clean, pred_cols=pred_list, era_col=era_col)
        design = normalized.select(pred_list).cast(pl.Float64).to_numpy()
        target = normalized.get_column(target_col).cast(pl.Float64).to_numpy()

        finite_mask = np.isfinite(target)
        for idx in range(design.shape[1]):
            finite_mask &= np.isfinite(design[:, idx])
        design = design[finite_mask]
        target = target[finite_mask]
        if len(target) == 0:
            raise ValueError("oof_df has no finite rows for weight learning")

        if method == "ridge":
            alpha = _RIDGE_ALPHA
            sample_count = float(len(target))
            gram = (design.T @ design) / sample_count
            rhs = (design.T @ target) / sample_count
            weights = np.linalg.solve(
                gram + alpha * np.eye(gram.shape[0], dtype=float),
                rhs,
            )
        elif method in {"non_negative", "nnls"}:
            weights = nnls(design, target)[0]
        else:
            raise ValueError("method must be 'ridge' or 'non_negative'")

        logger.info(
            "[learn_weights] weights learned: %s", [round(float(w), 4) for w in weights]
        )
        return tuple(float(value) for value in weights)


def _validate_pred_cols(pred_cols: Sequence[str]) -> list[str]:
    pred_list = list(pred_cols)
    if not pred_list:
        raise ValueError("pred_cols must contain at least one prediction column")
    return pred_list


def _resolve_weights(
    pred_cols: Sequence[str], weights: Sequence[float] | None
) -> np.ndarray:
    if weights is None:
        return np.full(len(pred_cols), 1.0 / len(pred_cols), dtype=float)

    weight_array = np.asarray(list(weights), dtype=float)
    if len(weight_array) != len(pred_cols):
        raise ValueError("weights length must match pred_cols length")
    if not np.all(np.isfinite(weight_array)):
        raise ValueError("weights must be finite")
    return weight_array


def _apply_per_era(
    df: pl.DataFrame,
    *,
    pred_cols: Sequence[str],
    era_col: str,
    transform,
) -> pl.DataFrame:
    indexed = df.with_row_index("__row_idx")
    parts: list[pl.DataFrame] = []
    for era_df in indexed.partition_by(era_col, as_dict=False, maintain_order=True):
        parts.append(transform(era_df, pred_cols=pred_cols))
    return pl.concat(parts, how="vertical").sort("__row_idx").drop("__row_idx")


def _rank_block(era_df: pl.DataFrame, *, pred_cols: Sequence[str]) -> pl.DataFrame:
    transformed = {}
    for col in pred_cols:
        values = era_df.get_column(col).cast(pl.Float64).to_numpy()
        transformed[col] = pl.Series(
            col,
            rank_gaussianize_unit_variance(values),
        )
    return era_df.with_columns([transformed[col] for col in pred_cols])


def _blend_block(
    era_df: pl.DataFrame,
    *,
    pred_cols: Sequence[str],
    weights: np.ndarray,
    out_col: str,
) -> pl.DataFrame:
    matrix = era_df.select(list(pred_cols)).cast(pl.Float64).to_numpy()
    combined = matrix.dot(weights)
    blended = rank_gaussianize(combined)
    return era_df.with_columns(pl.Series(out_col, blended))
