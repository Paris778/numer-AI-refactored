from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl
from numerai_tools.scoring import (
    correlation_contribution as official_correlation_contribution,
)
from numerai_tools.scoring import feature_neutral_corr as official_feature_neutral_corr
from numerai_tools.scoring import (
    max_feature_correlation as official_max_feature_correlation,
)
from numerai_tools.scoring import numerai_corr as official_numerai_corr
from scipy import optimize, stats

from src.risk import CachedPseudoInverse, NeutralizationEngine


@dataclass(frozen=True)
class EraMetricRow:
    era: str
    corr: float
    fnc: float | None
    mmc: float | None
    benchmark_corr: float | None
    feature_exposure: float | None
    n_rows: int


@dataclass(frozen=True)
class EvaluationSummary:
    mean_corr: float
    sharpe_corr: float
    max_drawdown_corr: float
    mean_fnc: float | None
    mean_mmc: float | None
    mean_benchmark_corr: float | None
    max_feature_exposure: float | None
    eras_evaluated: int


@dataclass(frozen=True)
class FastFailGateResult:
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class TargetWeightOptimizationResult:
    target_weights: dict[str, float]
    mean_corr_by_target: dict[str, float]
    sharpe_corr_by_target: dict[str, float]
    volatility_by_target: dict[str, float]
    prediction_correlation_matrix: pd.DataFrame
    covariance_matrix: pd.DataFrame
    theoretical_mean_corr: float
    theoretical_sigma: float
    theoretical_sharpe: float


class EvaluationEngine:
    def __init__(
        self,
        *,
        era_col: str = "era",
        prediction_col: str = "prediction",
        target_col: str = "target",
        id_col: str = "id",
        epsilon: float = 1e-12,
        backend: str = "custom",
    ) -> None:
        if backend not in {"custom", "official"}:
            raise ValueError("backend must be 'custom' or 'official'")
        self.era_col = era_col
        self.prediction_col = prediction_col
        self.target_col = target_col
        self.id_col = id_col
        self.epsilon = epsilon
        self.backend = backend

    def evaluate_eras(
        self,
        df: pl.DataFrame,
        *,
        feature_columns: Sequence[str] | None = None,
        benchmark_prediction_col: str | None = None,
        benchmark_mmc_col: str | None = None,
        neutralization_engine: NeutralizationEngine | None = None,
        neutralization_subset_name: str | None = None,
        neutralization_proportion: float = 1.0,
    ) -> list[EraMetricRow]:
        frame = self._validate_frame(
            df,
            feature_columns=feature_columns,
            benchmark_prediction_col=benchmark_prediction_col,
            benchmark_mmc_col=benchmark_mmc_col,
        )
        feature_names = tuple(feature_columns or ())
        if self.backend == "official":
            return self._evaluate_eras_official(
                frame,
                feature_names=feature_names,
                benchmark_prediction_col=benchmark_prediction_col,
                benchmark_mmc_col=benchmark_mmc_col,
            )

        return self._evaluate_eras_custom(
            frame,
            feature_names=feature_names,
            benchmark_prediction_col=benchmark_prediction_col,
            benchmark_mmc_col=benchmark_mmc_col,
            neutralization_engine=neutralization_engine,
            neutralization_subset_name=neutralization_subset_name,
            neutralization_proportion=neutralization_proportion,
        )

    def _evaluate_eras_custom(
        self,
        frame: pl.DataFrame,
        *,
        feature_names: tuple[str, ...],
        benchmark_prediction_col: str | None,
        benchmark_mmc_col: str | None,
        neutralization_engine: NeutralizationEngine | None,
        neutralization_subset_name: str | None,
        neutralization_proportion: float,
    ) -> list[EraMetricRow]:
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
                    predictions, benchmark_predictions
                )

            mmc = None
            if benchmark_mmc_col is not None:
                mmc = self._official_mmc_from_era_frame(
                    era_frame,
                    benchmark_mmc_col=benchmark_mmc_col,
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
                    mmc=mmc,
                    benchmark_corr=benchmark_corr,
                    feature_exposure=feature_exposure,
                    n_rows=era_frame.height,
                )
            )

        return metrics

    def _evaluate_eras_official(
        self,
        frame: pl.DataFrame,
        *,
        feature_names: tuple[str, ...],
        benchmark_prediction_col: str | None,
        benchmark_mmc_col: str | None,
    ) -> list[EraMetricRow]:
        metrics: list[EraMetricRow] = []

        ordered = frame.sort(self.era_col)
        for era_value, era_frame in ordered.group_by(self.era_col, maintain_order=True):
            era = str(era_value[0] if isinstance(era_value, tuple) else era_value)
            prediction_frame, target_series, feature_frame, benchmark_series = (
                self._official_inputs(
                    era_frame,
                    feature_names=feature_names,
                    benchmark_prediction_col=benchmark_prediction_col,
                )
            )

            corr = self._sanitize_metric(
                float(official_numerai_corr(prediction_frame, target_series).iloc[0])
            )

            benchmark_corr = None
            if benchmark_series is not None:
                benchmark_corr = self.prediction_correlation(
                    prediction_frame[self.prediction_col].to_numpy(),
                    benchmark_series.to_numpy(),
                )

            mmc = None
            if benchmark_mmc_col is not None:
                mmc = self._official_mmc(
                    prediction_frame,
                    target_series,
                    pd.Series(
                        era_frame.get_column(benchmark_mmc_col).to_numpy(),
                        index=prediction_frame.index,
                        name=benchmark_mmc_col,
                    ),
                )

            fnc = None
            feature_exposure = None
            if feature_frame is not None:
                fnc = self._official_feature_neutral_corr(
                    prediction_frame,
                    feature_frame,
                    target_series,
                )
                feature_exposure = self._official_max_feature_exposure(
                    prediction_frame[self.prediction_col],
                    feature_frame,
                )

            metrics.append(
                EraMetricRow(
                    era=era,
                    corr=corr,
                    fnc=fnc,
                    mmc=mmc,
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
        mmcs = [row.mmc for row in era_metrics if row.mmc is not None]
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
            mean_mmc=(float(np.mean(mmcs)) if mmcs else None),
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
        min_mean_mmc: float | None = None,
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
        if (
            min_mean_mmc is not None
            and summary.mean_mmc is not None
            and summary.mean_mmc < min_mean_mmc
        ):
            failures.append(
                f"mean_mmc {summary.mean_mmc:.6f} is below minimum {min_mean_mmc:.6f}"
            )

        return FastFailGateResult(passed=not failures, failures=tuple(failures))

    def optimize_target_weights(
        self,
        target_predictions: dict[str, np.ndarray],
        targets: np.ndarray,
        eras: Sequence[str],
        *,
        initial_weights: Sequence[float] | None = None,
    ) -> TargetWeightOptimizationResult:
        if not target_predictions:
            raise ValueError("target_predictions must be non-empty")

        target_names = tuple(target_predictions)
        target_values = self._coerce_vector(targets, name="targets")
        era_values = np.asarray(eras, dtype=str).reshape(-1)
        self._ensure_matching_lengths(target_values, era_values)

        prediction_frame = pd.DataFrame(
            {
                target_name: self._coerce_vector(
                    target_predictions[target_name],
                    name=f"target_predictions[{target_name!r}]",
                )
                for target_name in target_names
            }
        )
        for column_name in prediction_frame.columns:
            self._ensure_matching_lengths(
                prediction_frame[column_name].to_numpy(),
                target_values,
            )

        mean_corr_by_target: dict[str, float] = {}
        sharpe_corr_by_target: dict[str, float] = {}
        volatility_by_target: dict[str, float] = {}
        mean_vector: list[float] = []
        volatility_vector: list[float] = []
        for target_name in target_names:
            era_corrs = self._compute_era_corrs(
                prediction_frame[target_name].to_numpy(),
                target_values,
                era_values,
            )
            mean_corr = float(np.mean(era_corrs))
            volatility = float(np.std(era_corrs))
            sharpe_corr = self.safe_sharpe(era_corrs, epsilon=self.epsilon)
            mean_corr_by_target[target_name] = mean_corr
            sharpe_corr_by_target[target_name] = sharpe_corr
            volatility_by_target[target_name] = volatility
            mean_vector.append(mean_corr)
            volatility_vector.append(volatility)

        correlation_matrix = prediction_frame.corr(method="spearman").fillna(0.0)
        mean_array = np.asarray(mean_vector, dtype=np.float64)
        volatility_array = np.asarray(volatility_vector, dtype=np.float64)
        covariance_values = np.outer(
            volatility_array, volatility_array
        ) * correlation_matrix.to_numpy(dtype=np.float64)
        covariance_values = (covariance_values + covariance_values.T) / 2.0
        covariance_matrix = pd.DataFrame(
            covariance_values,
            index=target_names,
            columns=target_names,
        )

        if len(target_names) == 1:
            weights = np.array([1.0], dtype=np.float64)
        else:
            if initial_weights is not None and len(initial_weights) != len(
                target_names
            ):
                raise ValueError(
                    "initial_weights length must match target_predictions length"
                )
            starting_weights = self._resolve_initial_weights(
                initial_weights,
                mean_array,
            )
            optimization = optimize.minimize(
                self._negative_portfolio_sharpe,
                x0=starting_weights,
                args=(mean_array, covariance_values),
                method="SLSQP",
                bounds=[(0.0, 1.0)] * len(target_names),
                constraints=[
                    {
                        "type": "eq",
                        "fun": lambda weights: float(np.sum(weights) - 1.0),
                    }
                ],
                options={"maxiter": 500, "ftol": 1e-12},
            )
            if not optimization.success:
                raise ValueError(
                    "target weight optimization failed: " f"{optimization.message}"
                )
            weights = np.asarray(optimization.x, dtype=np.float64)
            weights = np.clip(weights, 0.0, None)
            weights = weights / max(float(weights.sum()), self.epsilon)

        theoretical_mean_corr = float(weights @ mean_array)
        theoretical_variance = float(weights @ covariance_values @ weights)
        theoretical_sigma = float(np.sqrt(max(theoretical_variance, self.epsilon)))
        theoretical_sharpe = theoretical_mean_corr / max(
            theoretical_sigma,
            self.epsilon,
        )
        target_weights = {
            target_name: float(weight)
            for target_name, weight in zip(target_names, weights, strict=True)
        }

        return TargetWeightOptimizationResult(
            target_weights=target_weights,
            mean_corr_by_target=mean_corr_by_target,
            sharpe_corr_by_target=sharpe_corr_by_target,
            volatility_by_target=volatility_by_target,
            prediction_correlation_matrix=correlation_matrix,
            covariance_matrix=covariance_matrix,
            theoretical_mean_corr=theoretical_mean_corr,
            theoretical_sigma=theoretical_sigma,
            theoretical_sharpe=theoretical_sharpe,
        )

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
            gaussianized_predictions, target_values, epsilon=eps
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
    def safe_pearson(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-12) -> float:
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

    def _official_inputs(
        self,
        era_frame: pl.DataFrame,
        *,
        feature_names: Sequence[str],
        benchmark_prediction_col: str | None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None, pd.Series | None]:
        id_column = self.id_col if self.id_col in era_frame.columns else None
        columns = [self.prediction_col, self.target_col, *feature_names]
        if benchmark_prediction_col is not None:
            columns.append(benchmark_prediction_col)
        if id_column is not None:
            columns.insert(0, id_column)

        ordered = era_frame.select(columns)
        if id_column is not None:
            ordered = ordered.sort(id_column)
            index = ordered.get_column(id_column).to_list()
        else:
            index = list(range(ordered.height))

        prediction_frame = pd.DataFrame(
            {self.prediction_col: ordered.get_column(self.prediction_col).to_numpy()},
            index=index,
        )
        target_series = pd.Series(
            ordered.get_column(self.target_col).to_numpy(),
            index=index,
            name=self.target_col,
        )
        feature_frame = None
        if feature_names:
            feature_frame = pd.DataFrame(
                ordered.select(list(feature_names)).to_numpy(),
                index=index,
                columns=list(feature_names),
            )
        benchmark_series = None
        if benchmark_prediction_col is not None:
            benchmark_series = pd.Series(
                ordered.get_column(benchmark_prediction_col).to_numpy(),
                index=index,
                name=benchmark_prediction_col,
            )
        return prediction_frame, target_series, feature_frame, benchmark_series

    def _official_feature_neutral_corr(
        self,
        prediction_frame: pd.DataFrame,
        feature_frame: pd.DataFrame,
        target_series: pd.Series,
    ) -> float:
        prediction_values = prediction_frame[self.prediction_col].to_numpy()
        if np.std(prediction_values) < self.epsilon:
            return 0.0
        try:
            value = float(
                official_feature_neutral_corr(
                    prediction_frame,
                    feature_frame,
                    target_series,
                ).iloc[0]
            )
        except (AssertionError, ValueError, ZeroDivisionError):
            return 0.0
        return self._sanitize_metric(value)

    def _official_max_feature_exposure(
        self,
        prediction_series: pd.Series,
        feature_frame: pd.DataFrame,
    ) -> float:
        if float(prediction_series.std()) < self.epsilon:
            return 0.0
        try:
            _, value = official_max_feature_correlation(
                prediction_series, feature_frame
            )
        except (AssertionError, KeyError, ValueError, ZeroDivisionError):
            return 0.0
        return self._sanitize_metric(float(value))

    def _official_mmc_from_era_frame(
        self,
        era_frame: pl.DataFrame,
        *,
        benchmark_mmc_col: str,
    ) -> float:
        prediction_frame = pd.DataFrame(
            {self.prediction_col: era_frame.get_column(self.prediction_col).to_numpy()},
            index=era_frame.get_column(self.id_col).to_list(),
        )
        target_series = pd.Series(
            era_frame.get_column(self.target_col).to_numpy(),
            index=prediction_frame.index,
            name=self.target_col,
        )
        benchmark_series = pd.Series(
            era_frame.get_column(benchmark_mmc_col).to_numpy(),
            index=prediction_frame.index,
            name=benchmark_mmc_col,
        )
        return self._official_mmc(prediction_frame, target_series, benchmark_series)

    def _official_mmc(
        self,
        prediction_frame: pd.DataFrame,
        target_series: pd.Series,
        benchmark_series: pd.Series,
    ) -> float:
        try:
            value = float(
                official_correlation_contribution(
                    prediction_frame,
                    benchmark_series,
                    target_series,
                ).iloc[0]
            )
        except (AssertionError, ValueError, ZeroDivisionError):
            return 0.0
        return self._sanitize_metric(value)

    def _validate_frame(
        self,
        df: pl.DataFrame,
        *,
        feature_columns: Sequence[str] | None,
        benchmark_prediction_col: str | None,
        benchmark_mmc_col: str | None,
    ) -> pl.DataFrame:
        required = {self.era_col, self.prediction_col, self.target_col}
        if feature_columns is not None:
            required.update(feature_columns)
        if benchmark_prediction_col is not None:
            required.add(benchmark_prediction_col)
        if benchmark_mmc_col is not None:
            required.add(benchmark_mmc_col)
        missing = sorted(required.difference(df.columns))
        if missing:
            raise KeyError(f"Missing required columns: {', '.join(missing)}")
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

    def _compute_era_corrs(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        eras: np.ndarray,
    ) -> np.ndarray:
        prediction_values = self._coerce_vector(predictions, name="predictions")
        target_values = self._coerce_vector(targets, name="targets")
        era_values = np.asarray(eras, dtype=str).reshape(-1)
        self._ensure_matching_lengths(prediction_values, target_values)
        self._ensure_matching_lengths(prediction_values, era_values)

        era_corrs = [
            self.numerai_corr(
                prediction_values[era_values == era],
                target_values[era_values == era],
                epsilon=self.epsilon,
            )
            for era in pd.unique(era_values)
        ]
        return np.asarray(era_corrs, dtype=np.float64)

    def _resolve_initial_weights(
        self,
        initial_weights: Sequence[float] | None,
        mean_corrs: np.ndarray,
    ) -> np.ndarray:
        if initial_weights is not None:
            weights = np.asarray(initial_weights, dtype=np.float64)
        else:
            positive_means = np.clip(mean_corrs, 0.0, None)
            if float(positive_means.sum()) > self.epsilon:
                weights = positive_means / float(positive_means.sum())
            else:
                weights = np.full(mean_corrs.shape[0], 1.0 / mean_corrs.shape[0])

        if np.any(weights < 0.0):
            raise ValueError("initial_weights must be non-negative")
        weight_sum = float(weights.sum())
        if weight_sum <= self.epsilon:
            raise ValueError("initial_weights must sum to a positive value")
        return weights / weight_sum

    def _negative_portfolio_sharpe(
        self,
        weights: np.ndarray,
        mean_corrs: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> float:
        portfolio_mean = float(weights @ mean_corrs)
        portfolio_variance = float(weights @ covariance_matrix @ weights)
        portfolio_sigma = float(np.sqrt(max(portfolio_variance, self.epsilon)))
        return -(portfolio_mean / max(portfolio_sigma, self.epsilon))

    @staticmethod
    def _sanitize_metric(value: float) -> float:
        return 0.0 if not np.isfinite(value) else float(value)
