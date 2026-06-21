"""Evaluation engine with custom and official backends.

The engine computes per-era CORR, MMC, and FNC, then summarizes the resulting
time series. The `custom` backend is a NumPy reimplementation of the canonical
math; the `official` backend is a thin wrapper over `numerai_tools.scoring`.

Degenerate eras are normalized at the engine boundary:
- fewer than 2 usable rows -> 0.0
- zero-variance predictions/targets/meta -> 0.0
- non-finite oracle/custom results -> 0.0

Summary Sharpe uses population standard deviation (`ddof=0`). When `std == 0`,
Sharpe is defined as `0.0`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from nmr._transforms import power_1_5, rank_gaussianize

__all__ = ["MIN_OVERLAP_ERAS", "MetricSummary", "EvaluationEngine"]

_VALID_BACKENDS = ("custom", "official")
MIN_OVERLAP_ERAS = 20


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    sharpe: float
    max_drawdown: float


class EvaluationEngine:
    """Dual-backend per-era metric engine."""

    def __init__(self, backend: str = "custom") -> None:
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"backend={backend!r} not in {_VALID_BACKENDS}")
        self._backend = backend

    def per_era_corr(
        self,
        df: pl.DataFrame,
        *,
        pred_col: str,
        target_col: str,
        era_col: str = "era",
    ) -> dict[str, float]:
        return self._per_era_metric(
            df,
            era_col=era_col,
            required_cols=[pred_col, target_col],
            score_fn=lambda era_df: self._corr_single_era(
                era_df, pred_col=pred_col, target_col=target_col
            ),
        )

    def per_era_mmc(
        self,
        df: pl.DataFrame,
        *,
        pred_col: str,
        meta_col: str,
        target_col: str,
        era_col: str = "era",
    ) -> dict[str, float]:
        return self._per_era_metric(
            df,
            era_col=era_col,
            required_cols=[pred_col, meta_col, target_col],
            score_fn=lambda era_df: self._mmc_single_era(
                era_df,
                pred_col=pred_col,
                meta_col=meta_col,
                target_col=target_col,
            ),
        )

    def per_era_fnc(
        self,
        df: pl.DataFrame,
        *,
        pred_col: str,
        feature_cols: Sequence[str],
        target_col: str,
        era_col: str = "era",
    ) -> dict[str, float]:
        feature_list = list(feature_cols)
        if not feature_list:
            raise ValueError("feature_cols must contain at least one feature")
        return self._per_era_metric(
            df,
            era_col=era_col,
            required_cols=[pred_col, target_col, *feature_list],
            score_fn=lambda era_df: self._fnc_single_era(
                era_df,
                pred_col=pred_col,
                feature_cols=feature_list,
                target_col=target_col,
            ),
        )

    def per_era_bmc(
        self,
        df: pl.DataFrame,
        *,
        pred_col: str,
        benchmark_col: str,
        target_col: str,
        era_col: str = "era",
        min_overlap_eras: int = MIN_OVERLAP_ERAS,
    ) -> dict[str, float]:
        self._validate_required_columns(
            df, [era_col, pred_col, benchmark_col, target_col]
        )
        overlap_eras = self._resolve_overlap_eras(
            df,
            era_col=era_col,
            coverage_col=benchmark_col,
            min_overlap_eras=min_overlap_eras,
        )

        scores: dict[str, float] = {}
        for era in overlap_eras:
            era_df = df.filter(pl.col(era_col) == era)
            clean_df = self._clean_frame(era_df, [pred_col, benchmark_col, target_col])
            score = self._mmc_single_era(
                clean_df,
                pred_col=pred_col,
                meta_col=benchmark_col,
                target_col=target_col,
            )
            scores[era] = self._normalize_score(score)
        return scores

    def per_era_cwmm(
        self,
        df: pl.DataFrame,
        *,
        pred_col: str,
        meta_col: str,
        era_col: str = "era",
        min_overlap_eras: int = MIN_OVERLAP_ERAS,
    ) -> dict[str, float]:
        self._validate_required_columns(df, [era_col, pred_col, meta_col])
        overlap_eras = self._resolve_overlap_eras(
            df,
            era_col=era_col,
            coverage_col=meta_col,
            min_overlap_eras=min_overlap_eras,
        )

        scores: dict[str, float] = {}
        for era in overlap_eras:
            era_df = df.filter(pl.col(era_col) == era)
            clean_df = self._clean_frame(era_df, [pred_col, meta_col])
            score = self._cwmm_single_era(
                clean_df,
                pred_col=pred_col,
                meta_col=meta_col,
            )
            scores[era] = self._normalize_score(score)
        return scores

    def summarize(self, per_era: Mapping[str, float]) -> MetricSummary:
        if not per_era:
            raise ValueError("per_era must contain at least one era score")

        ordered_values = np.array(
            [per_era[era] for era in self._sorted_labels(per_era.keys())],
            dtype=float,
        )
        mean = float(np.mean(ordered_values))
        std = float(np.std(ordered_values, ddof=0))
        sharpe = 0.0 if std == 0.0 else float(mean / std)

        cumulative = np.cumsum(ordered_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = float(np.max(drawdowns))

        return MetricSummary(
            mean=mean,
            std=std,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
        )

    def _per_era_metric(
        self,
        df: pl.DataFrame,
        *,
        era_col: str,
        required_cols: Sequence[str],
        score_fn,
    ) -> dict[str, float]:
        eras = self._sorted_labels(df.get_column(era_col).to_list())
        scores: dict[str, float] = {}
        for era in eras:
            era_df = df.filter(pl.col(era_col) == era)
            clean_df = self._clean_frame(era_df, required_cols)
            scores[era] = self._normalize_score(score_fn(clean_df))
        return scores

    def _clean_frame(
        self,
        df: pl.DataFrame,
        columns: Sequence[str],
    ) -> pl.DataFrame:
        clean_df = df.select(list(columns)).drop_nulls()
        if clean_df.is_empty():
            return clean_df

        mask = np.ones(clean_df.height, dtype=bool)
        for col in columns:
            values = clean_df.get_column(col).to_numpy()
            if np.issubdtype(values.dtype, np.number):
                mask &= np.isfinite(values)

        if mask.all():
            return clean_df
        return clean_df.filter(pl.Series("mask", mask))

    def _corr_single_era(
        self,
        df: pl.DataFrame,
        *,
        pred_col: str,
        target_col: str,
    ) -> float:
        pred = self._column_values(df, pred_col)
        target = self._column_values(df, target_col)
        if self._should_short_circuit(pred, target):
            return 0.0
        if self._backend == "custom":
            return self._custom_corr(pred, target)
        return self._official_corr(df, pred_col=pred_col, target_col=target_col)

    def _mmc_single_era(
        self,
        df: pl.DataFrame,
        *,
        pred_col: str,
        meta_col: str,
        target_col: str,
    ) -> float:
        pred = self._column_values(df, pred_col)
        meta = self._column_values(df, meta_col)
        target = self._column_values(df, target_col)
        if self._should_short_circuit(pred, meta, target):
            return 0.0
        if self._backend == "custom":
            return self._custom_mmc(pred, meta, target)
        return self._official_mmc(
            df,
            pred_col=pred_col,
            meta_col=meta_col,
            target_col=target_col,
        )

    def _fnc_single_era(
        self,
        df: pl.DataFrame,
        *,
        pred_col: str,
        feature_cols: Sequence[str],
        target_col: str,
    ) -> float:
        pred = self._column_values(df, pred_col)
        target = self._column_values(df, target_col)
        if self._should_short_circuit(pred, target):
            return 0.0
        if self._backend == "custom":
            return self._custom_fnc(
                pred, self._feature_matrix(df, feature_cols), target
            )
        return self._official_fnc(
            df,
            pred_col=pred_col,
            feature_cols=feature_cols,
            target_col=target_col,
        )

    def _cwmm_single_era(
        self,
        df: pl.DataFrame,
        *,
        pred_col: str,
        meta_col: str,
    ) -> float:
        pred = self._column_values(df, pred_col)
        meta = self._column_values(df, meta_col)
        if self._should_short_circuit(pred, meta):
            return 0.0
        # There is no numerai_tools oracle for prediction-vs-prediction CWMM;
        # both backends intentionally use the same custom transform geometry.
        return self._custom_cwmm(pred, meta)

    def _official_corr(
        self,
        df: pl.DataFrame,
        *,
        pred_col: str,
        target_col: str,
    ) -> float:
        from numerai_tools.scoring import numerai_corr

        pdf = df.to_pandas()
        preds = pdf[[pred_col]]
        target = pdf[target_col].rename(target_col)
        return float(numerai_corr(preds, target)[pred_col])

    def _official_mmc(
        self,
        df: pl.DataFrame,
        *,
        pred_col: str,
        meta_col: str,
        target_col: str,
    ) -> float:
        from numerai_tools.scoring import correlation_contribution

        pdf = df.to_pandas()
        preds = pdf[[pred_col]]
        meta = pdf[meta_col].rename(meta_col)
        target = pdf[target_col].rename(target_col)
        return float(correlation_contribution(preds, meta, target)[pred_col])

    def _official_fnc(
        self,
        df: pl.DataFrame,
        *,
        pred_col: str,
        feature_cols: Sequence[str],
        target_col: str,
    ) -> float:
        from numerai_tools.scoring import feature_neutral_corr

        pdf = df.to_pandas()
        preds = pdf[[pred_col]]
        features = pdf[list(feature_cols)]
        target = pdf[target_col].rename(target_col)
        return float(feature_neutral_corr(preds, features, target)[pred_col])

    def _custom_corr(self, pred: np.ndarray, target: np.ndarray) -> float:
        transformed_pred = power_1_5(rank_gaussianize(pred))
        transformed_target = power_1_5(target - np.mean(target))
        return self._pearson_corr(transformed_pred, transformed_target)

    def _custom_mmc(
        self,
        pred: np.ndarray,
        meta: np.ndarray,
        target: np.ndarray,
    ) -> float:
        ranked_pred = rank_gaussianize(pred)
        ranked_meta = rank_gaussianize(meta)
        denominator = float(ranked_meta @ ranked_meta)
        if denominator == 0.0:
            return 0.0

        neutral_pred = ranked_pred - ranked_meta * (
            (ranked_pred @ ranked_meta) / denominator
        )

        live_target = target.astype(float, copy=True)
        if np.all((live_target >= 0.0) & (live_target <= 1.0)):
            # Match numerai_tools: bucket-style targets on [0, 1] are rescaled
            # to [0, 4] before centering inside correlation_contribution.
            live_target = live_target * 4.0
        live_target = live_target - np.mean(live_target)
        return float((live_target @ neutral_pred) / len(live_target))

    def _custom_fnc(
        self,
        pred: np.ndarray,
        features: np.ndarray,
        target: np.ndarray,
    ) -> float:
        ranked_pred = rank_gaussianize(pred).reshape(-1, 1)
        neutralizers = np.hstack([features, np.ones((len(features), 1), dtype=float)])
        least_squares = np.linalg.lstsq(neutralizers, ranked_pred, rcond=1e-6)[0]
        neutral_pred = ranked_pred - neutralizers.dot(least_squares)
        std = np.std(neutral_pred, axis=0)
        if std[0] == 0.0 or not np.isfinite(std[0]):
            return 0.0
        normalized_pred = (neutral_pred / std).ravel()
        return self._custom_corr(normalized_pred, target)

    def _custom_cwmm(self, pred: np.ndarray, meta: np.ndarray) -> float:
        transformed_pred = power_1_5(rank_gaussianize(pred))
        transformed_meta = power_1_5(rank_gaussianize(meta))
        return self._pearson_corr(transformed_pred, transformed_meta)

    def _column_values(self, df: pl.DataFrame, col: str) -> np.ndarray:
        return df.get_column(col).cast(pl.Float64).to_numpy()

    def _feature_matrix(
        self, df: pl.DataFrame, feature_cols: Sequence[str]
    ) -> np.ndarray:
        return df.select(list(feature_cols)).cast(pl.Float64).to_numpy()

    def _should_short_circuit(self, *arrays: np.ndarray) -> bool:
        if not arrays or len(arrays[0]) < 2:
            return True
        for arr in arrays:
            if len(arr) < 2 or not np.all(np.isfinite(arr)):
                return True
            if np.std(arr) == 0.0:
                return True
        return False

    def _normalize_score(self, value: float) -> float:
        return 0.0 if not math.isfinite(value) else float(value)

    def _validate_required_columns(
        self, df: pl.DataFrame, required_cols: Sequence[str]
    ) -> None:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _resolve_overlap_eras(
        self,
        df: pl.DataFrame,
        *,
        era_col: str,
        coverage_col: str,
        min_overlap_eras: int,
    ) -> list[str]:
        if min_overlap_eras < 1:
            raise ValueError("min_overlap_eras must be >= 1")

        eras = self._sorted_labels(df.get_column(era_col).to_list())
        overlap_eras: list[str] = []
        for era in eras:
            era_values = (
                df.filter(pl.col(era_col) == era)
                .select(pl.col(coverage_col).cast(pl.Float64, strict=False))
                .drop_nulls()
                .get_column(coverage_col)
                .to_numpy()
            )
            if era_values.size == 0:
                continue
            if np.isfinite(era_values).any():
                overlap_eras.append(era)

        if len(overlap_eras) < min_overlap_eras:
            raise ValueError(
                "Non-vacuity violation: intersection yielded only "
                f"{len(overlap_eras)} eras; minimum required {min_overlap_eras}."
            )
        return overlap_eras

    def _sorted_labels(self, labels: Sequence[str]) -> list[str]:
        numeric_to_label: dict[int, str] = {}
        for label in labels:
            try:
                numeric_label = int(label)
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric era label {label!r}; evaluation requires chronological eras"
                ) from exc
            numeric_to_label.setdefault(numeric_label, label)
        return [numeric_to_label[num] for num in sorted(numeric_to_label)]

    def _pearson_corr(self, left: np.ndarray, right: np.ndarray) -> float:
        left_centered = left - np.mean(left)
        right_centered = right - np.mean(right)
        denominator = float(
            np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
        )
        if denominator == 0.0:
            return 0.0
        return float((left_centered @ right_centered) / denominator)
