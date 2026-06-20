from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl
from scipy import stats

from src.risk import CachedPseudoInverse, NeutralizationEngine


@dataclass(frozen=True)
class EraMetricRow:
    era: str
    corr: float
    fnc: float | None
    benchmark_corr: float | None
    feature_exposure: float | None
    n_rows: int


@dataclass(frozen=True)
class EvaluationSummary:
    mean_corr: float
    sharpe_corr: float
    max_drawdown_corr: float
    mean_fnc: float | None
    mean_benchmark_corr: float | None
    max_feature_exposure: float | None
    eras_evaluated: int


@dataclass(frozen=True)
class FastFailGateResult:
    passed: bool
    failures: tuple[str, ...]


class EvaluationEngine:
    def __init__(
        self,
        *,
        era_col: str = "era",
        prediction_col: str = "prediction",
        target_col: str = "target",
        id_col: str = "id",
        epsilon: float = 1e-12,
    ) -> None:
        self.era_col = era_col
        self.prediction_col = prediction_col
        self.target_col = target_col
        self.id_col = id_col
        self.epsilon = epsilon

    def evaluate_eras(
        self,
        df: pl.DataFrame,
        *,
        feature_columns: Sequence[str] | None = None,
        benchmark_prediction_col: str | None = None,
        neutralization_engine: NeutralizationEngine | None = None,
        neutralization_subset_name: str | None = None,
        neutralization_proportion: float = 1.0,
    ) -> list[EraMetricRow]:
        frame = self._validate_frame(
            df,
            feature_columns=feature_columns,
            benchmark_prediction_col=benchmark_prediction_col,
        )
        feature_names = tuple(feature_columns or ())
        metrics: list[EraMetricRow] = []

        ordered = frame.sort(self.era_col)
        for era_value, era_frame in ordered.group_by(self.era_col, maintain_order=True):
            era = str(era_value[0] if isinstance(era_value, tuple) else era_value)
            predictions = era_frame.get_column(self.prediction_col).to_numpy()
            targets = era_frame.get_column(self.target_col).to_numpy()
            corr = self.numerai_corr(predictions, targets)

            benchmark_corr = None
            if benchmark_prediction_col is not None:
                benchmark_predictions = era_frame.get_column(
                    benchmark_prediction_col
                ).to_numpy()
                benchmark_corr = self.prediction_correlation(
                    predictions,
                    benchmark_predictions,
                )

            feature_exposure = None
            fnc = None
            if feature_names:
                feature_matrix = era_frame.select(list(feature_names)).to_numpy()
                feature_exposure = self.compute_max_feature_exposure(
                    predictions,
                    feature_matrix,
                    epsilon=self.epsilon,
                )
                if neutralization_engine is not None:
                    fnc = self._compute_fnc(
                        era_frame=era_frame,
                        era=era,
                        predictions=predictions,
                        targets=targets,
                        feature_matrix=feature_matrix,
                        feature_names=feature_names,
                        engine=neutralization_engine,
                        subset_name=neutralization_subset_name,
                        proportion=neutralization_proportion,
                    )

            metrics.append(
                EraMetricRow(
                    era=era,
                    corr=corr,
                    fnc=fnc,
                    benchmark_corr=benchmark_corr,
                    feature_exposure=feature_exposure,
                    n_rows=era_frame.height,
                )
            )

        return metrics

    def summarize(self, era_metrics: Sequence[EraMetricRow]) -> EvaluationSummary:
        if not era_metrics:
            raise ValueError("era_metrics must be non-empty")

        corrs = np.asarray([row.corr for row in era_metrics], dtype=np.float64)
        fncs = [row.fnc for row in era_metrics if row.fnc is not None]
        benchmark_corrs = [
            row.benchmark_corr for row in era_metrics if row.benchmark_corr is not None
        ]
        feature_exposures = [
            row.feature_exposure
            for row in era_metrics
            if row.feature_exposure is not None
        ]

        return EvaluationSummary(
            mean_corr=float(np.mean(corrs)),
            sharpe_corr=self.safe_sharpe(corrs, epsilon=self.epsilon),
            max_drawdown_corr=self.compute_max_drawdown(corrs),
            mean_fnc=(float(np.mean(fncs)) if fncs else None),
            mean_benchmark_corr=(
                float(np.mean(benchmark_corrs)) if benchmark_corrs else None
            ),
            max_feature_exposure=(
                float(np.max(feature_exposures)) if feature_exposures else None
            ),
            eras_evaluated=len(era_metrics),
        )

    def fast_fail_gate(
        self,
        summary: EvaluationSummary,
        *,
        min_mean_corr: float | None = None,
        min_sharpe_corr: float | None = None,
        max_drawdown_corr: float | None = None,
        max_feature_exposure: float | None = None,
        max_benchmark_corr: float | None = None,
        min_mean_fnc: float | None = None,
    ) -> FastFailGateResult:
        failures: list[str] = []

        if min_mean_corr is not None and summary.mean_corr < min_mean_corr:
            failures.append(
                f"mean_corr {summary.mean_corr:.6f} is below minimum {min_mean_corr:.6f}"
            )
        if min_sharpe_corr is not None and summary.sharpe_corr < min_sharpe_corr:
            failures.append(
                f"sharpe_corr {summary.sharpe_corr:.6f} is below minimum {min_sharpe_corr:.6f}"
            )
        if (
            max_drawdown_corr is not None
            and summary.max_drawdown_corr > max_drawdown_corr
        ):
            failures.append(
                f"max_drawdown_corr {summary.max_drawdown_corr:.6f} exceeds maximum {max_drawdown_corr:.6f}"
            )
        if (
            max_feature_exposure is not None
            and summary.max_feature_exposure is not None
            and summary.max_feature_exposure > max_feature_exposure
        ):
            failures.append(
                f"max_feature_exposure {summary.max_feature_exposure:.6f} exceeds maximum {max_feature_exposure:.6f}"
            )
        if (
            max_benchmark_corr is not None
            and summary.mean_benchmark_corr is not None
            and summary.mean_benchmark_corr > max_benchmark_corr
        ):
            failures.append(
                f"mean_benchmark_corr {summary.mean_benchmark_corr:.6f} exceeds maximum {max_benchmark_corr:.6f}"
            )
        if (
            min_mean_fnc is not None
            and summary.mean_fnc is not None
            and summary.mean_fnc < min_mean_fnc
        ):
            failures.append(
                f"mean_fnc {summary.mean_fnc:.6f} is below minimum {min_mean_fnc:.6f}"
            )

        return FastFailGateResult(passed=not failures, failures=tuple(failures))

    def numerai_corr(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        epsilon: float | None = None,
    ) -> float:
        eps = self.epsilon if epsilon is None else epsilon
        predictions = self._coerce_vector(preds, name="preds")
        target_values = self._coerce_vector(targets, name="targets")
        self._ensure_matching_lengths(predictions, target_values)
        gaussianized_predictions = self.rank_to_gaussian(predictions, epsilon=eps)
        return self.tail_correlation(
            gaussianized_predictions,
            target_values,
            epsilon=eps,
        )

    def prediction_correlation(
        self,
        preds_a: np.ndarray,
        preds_b: np.ndarray,
        epsilon: float | None = None,
    ) -> float:
        eps = self.epsilon if epsilon is None else epsilon
        left = self.rank_to_gaussian(
            self._coerce_vector(preds_a, name="preds_a"), epsilon=eps
        )
        right = self.rank_to_gaussian(
            self._coerce_vector(preds_b, name="preds_b"), epsilon=eps
        )
        self._ensure_matching_lengths(left, right)
        left_tail = np.sign(left) * np.abs(left) ** 1.5
        right_tail = np.sign(right) * np.abs(right) ** 1.5
        return self.safe_pearson(left_tail, right_tail, epsilon=eps)

    def tail_correlation(
        self,
        prediction_signal: np.ndarray,
        targets: np.ndarray,
        *,
        epsilon: float | None = None,
    ) -> float:
        eps = self.epsilon if epsilon is None else epsilon
        signal = self._coerce_vector(prediction_signal, name="prediction_signal")
        target_values = self._coerce_vector(targets, name="targets")
        self._ensure_matching_lengths(signal, target_values)
        centered_target = target_values - np.mean(target_values)
        prediction_tail = np.sign(signal) * np.abs(signal) ** 1.5
        target_tail = np.sign(centered_target) * np.abs(centered_target) ** 1.5
        return self.safe_pearson(prediction_tail, target_tail, epsilon=eps)

    @staticmethod
    def safe_pearson(
        x: np.ndarray,
        y: np.ndarray,
        epsilon: float = 1e-12,
    ) -> float:
        left = np.asarray(x, dtype=np.float64).reshape(-1)
        right = np.asarray(y, dtype=np.float64).reshape(-1)
        if left.shape[0] != right.shape[0]:
            raise ValueError("x and y must have matching lengths")
        left_centered = left - np.mean(left)
        right_centered = right - np.mean(right)
        left_std = float(np.std(left_centered))
        right_std = float(np.std(right_centered))
        if left_std < epsilon or right_std < epsilon:
            return 0.0
        covariance = float(np.mean(left_centered * right_centered))
        return covariance / max(left_std * right_std, epsilon)

    @staticmethod
    def rank_to_gaussian(preds: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
        values = np.asarray(preds, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError("preds must be non-empty")
        ranked = (stats.rankdata(values, method="average") - 0.5) / values.size
        clipped = np.clip(ranked, epsilon, 1.0 - epsilon)
        return np.asarray(stats.norm.ppf(clipped), dtype=np.float64)

    @staticmethod
    def safe_sharpe(values: np.ndarray, epsilon: float = 1e-12) -> float:
        series = np.asarray(values, dtype=np.float64).reshape(-1)
        return float(np.mean(series) / max(float(np.std(series)), epsilon))

    @staticmethod
    def compute_max_feature_exposure(
        preds: np.ndarray,
        features: np.ndarray,
        epsilon: float = 1e-12,
    ) -> float:
        prediction_values = np.asarray(preds, dtype=np.float64).reshape(-1)
        feature_matrix = np.asarray(features, dtype=np.float64)
        if feature_matrix.ndim != 2:
            raise ValueError("features must be a 2D array")
        if feature_matrix.shape[0] != prediction_values.shape[0]:
            raise ValueError("features row count must match preds length")
        if feature_matrix.shape[1] == 0:
            return 0.0

        prediction_centered = prediction_values - np.mean(prediction_values)
        feature_centered = feature_matrix - np.mean(feature_matrix, axis=0)
        prediction_std = float(np.std(prediction_values))
        feature_std = np.std(feature_matrix, axis=0)

        if prediction_std < epsilon:
            return 0.0

        covariance = (prediction_centered @ feature_centered) / prediction_values.shape[
            0
        ]
        correlations = covariance / (prediction_std * np.maximum(feature_std, epsilon))
        return float(np.max(np.abs(correlations)))

    @staticmethod
    def compute_max_drawdown(era_corrs: np.ndarray) -> float:
        series = np.asarray(era_corrs, dtype=np.float64).reshape(-1)
        if series.size == 0:
            return 0.0
        cumulative = np.cumsum(series)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return float(np.max(drawdowns))

    def _compute_fnc(
        self,
        *,
        era_frame: pl.DataFrame,
        era: str,
        predictions: np.ndarray,
        targets: np.ndarray,
        feature_matrix: np.ndarray,
        feature_names: Sequence[str],
        engine: NeutralizationEngine,
        subset_name: str | None,
        proportion: float,
    ) -> float:
        gaussianized_predictions = self.rank_to_gaussian(
            predictions, epsilon=self.epsilon
        )
        pseudo_inverse = None
        if subset_name is not None:
            cached = self._load_cached_pseudo_inverse(
                era_frame=era_frame,
                era=era,
                feature_names=feature_names,
                engine=engine,
                subset_name=subset_name,
            )
            if cached is not None:
                pseudo_inverse = cached.pseudo_inverse

        neutralized = engine.neutralize_tensor(
            gaussianized_predictions,
            feature_matrix,
            proportion=proportion,
            pseudo_inverse=pseudo_inverse,
        )
        neutralized_std = float(np.std(neutralized))
        if neutralized_std < self.epsilon:
            normalized = np.zeros_like(neutralized)
        else:
            normalized = neutralized / neutralized_std
        return self.numerai_corr(normalized, targets, epsilon=self.epsilon)

    def _load_cached_pseudo_inverse(
        self,
        *,
        era_frame: pl.DataFrame,
        era: str,
        feature_names: Sequence[str],
        engine: NeutralizationEngine,
        subset_name: str,
    ) -> CachedPseudoInverse | None:
        try:
            cached = engine.load_cached_pseudo_inverse(subset_name, era)
        except FileNotFoundError:
            return None

        if cached.feature_names != tuple(feature_names):
            raise ValueError(
                f"Cached feature names for era '{era}' do not match evaluation feature columns"
            )

        if self.id_col in era_frame.columns:
            row_ids = tuple(era_frame.get_column(self.id_col).to_list())
            if cached.row_ids != row_ids:
                raise ValueError(
                    f"Cached row ids for era '{era}' do not match evaluation rows"
                )

        return cached

    def _validate_frame(
        self,
        df: pl.DataFrame,
        *,
        feature_columns: Sequence[str] | None,
        benchmark_prediction_col: str | None,
    ) -> pl.DataFrame:
        required = {self.era_col, self.prediction_col, self.target_col}
        if feature_columns is not None:
            required.update(feature_columns)
        if benchmark_prediction_col is not None:
            required.add(benchmark_prediction_col)
        missing = sorted(required.difference(df.columns))
        if missing:
            missing_columns = ", ".join(missing)
            raise KeyError(f"Missing required columns: {missing_columns}")
        return df

    @staticmethod
    def _coerce_vector(values: np.ndarray, *, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size == 0:
            raise ValueError(f"{name} must be non-empty")
        return vector

    @staticmethod
    def _ensure_matching_lengths(left: np.ndarray, right: np.ndarray) -> None:
        if left.shape[0] != right.shape[0]:
            raise ValueError("Input vectors must have matching lengths")
