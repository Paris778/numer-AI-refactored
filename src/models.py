from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from src.features import PurgedEraSplitter
from src.protocols import Predictor

FrameLike = pl.DataFrame | pl.LazyFrame
EstimatorType = lgb.LGBMRegressor | xgb.XGBRegressor


@dataclass(frozen=True)
class TrainingFoldResult:
    fold_number: int
    train_eras: tuple[str, ...]
    validation_eras: tuple[str, ...]
    validation_row_indices: np.ndarray
    validation_predictions: np.ndarray
    best_iteration: int
    fit_seconds: float
    backend: str


@dataclass(frozen=True)
class AnchorTrainingResult:
    validation_predictions: np.ndarray
    validation_ids: np.ndarray
    validation_eras: np.ndarray
    best_iteration: int
    fit_seconds: float
    backend: str
    predictor: "MultiTargetPredictor"

    @property
    def model_count(self) -> int:
        return self.predictor.model_count


@dataclass(frozen=True)
class CrossValidationResult:
    oof_predictions: np.ndarray
    fold_results: list[TrainingFoldResult]
    models: list[EstimatorType]
    predictor: "MultiTargetPredictor"
    backend: str
    total_fit_seconds: float

    @property
    def model_count(self) -> int:
        return len(self.models)


@dataclass(frozen=True)
class MultiTargetPredictor(Predictor):
    feature_names: tuple[str, ...]
    target_models: dict[str, tuple[EstimatorType, ...]]

    @property
    def model_count(self) -> int:
        return sum(len(models) for models in self.target_models.values())

    def predict_all(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        missing = sorted(set(self.feature_names).difference(df.columns))
        if missing:
            raise KeyError(
                f"Prediction dataframe is missing required features: {', '.join(missing)}"
            )
        ordered = df.loc[:, self.feature_names]
        outputs: dict[str, np.ndarray] = {}
        for target_name, models in self.target_models.items():
            for index, model in enumerate(models, start=1):
                outputs[f"{target_name}__model_{index:02d}"] = np.asarray(
                    model.predict(ordered),
                    dtype=np.float64,
                )
        return outputs

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return ModelOrchestrator.rank_average_predictions(self.predict_all(df).values())


@dataclass(frozen=True)
class WeightedMultiTargetPredictor(MultiTargetPredictor):
    target_weights: dict[str, float]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw_predictions = self.predict_all(df)
        weights = self._expanded_model_weights(raw_predictions)
        return ModelOrchestrator.weighted_rank_average_predictions(
            raw_predictions.values(),
            weights,
        )

    def _expanded_model_weights(
        self,
        raw_predictions: dict[str, np.ndarray],
    ) -> list[float]:
        weights: list[float] = []
        for model_key in raw_predictions:
            target_name = model_key.split("__model_", 1)[0]
            try:
                weights.append(self.target_weights[target_name])
            except KeyError as exc:
                raise KeyError(f"Missing target weight for '{target_name}'") from exc
        return weights


class ModelOrchestrator:
    """Gradient-boosted model manager for anchor and full-CV research loops."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        target_column: str | None = None,
        target_columns: Sequence[str] | None = None,
        target_weights: dict[str, float] | Sequence[float] | None = None,
        model_library: str = "lightgbm",
        model_params: dict[str, Any] | None = None,
        prefer_gpu: bool = True,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ) -> None:
        if not feature_names:
            raise ValueError("feature_names must be non-empty")
        if model_library not in {"lightgbm", "xgboost"}:
            raise ValueError("model_library must be 'lightgbm' or 'xgboost'")
        if target_columns is None and target_column is None:
            raise ValueError("target_column or target_columns must be provided")

        resolved_targets = (
            tuple(target_columns)
            if target_columns is not None
            else (str(target_column),)
        )
        if not resolved_targets:
            raise ValueError("target_columns must be non-empty")

        self.feature_names = tuple(feature_names)
        self.target_columns = tuple(resolved_targets)
        self.target_column = self.target_columns[0]
        self.target_weights = self._resolve_target_weights(target_weights)
        self.model_library = model_library
        self.prefer_gpu = prefer_gpu
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.model_params = dict(model_params or {})

    def train_anchor_fold(
        self,
        train_df: FrameLike,
        val_df: FrameLike,
        *,
        id_col: str = "id",
        era_col: str = "era",
    ) -> AnchorTrainingResult:
        train_frame = self._collect_required_columns(
            train_df, id_col=id_col, era_col=era_col
        )
        val_frame = self._collect_required_columns(
            val_df, id_col=id_col, era_col=era_col
        )

        target_models: dict[str, tuple[EstimatorType, ...]] = {}
        best_iterations: list[int] = []
        fit_seconds_total = 0.0
        backend = ""
        for target_name in self.target_columns:
            estimator, backend, fit_seconds = self._fit_with_fallback(
                train_frame,
                val_frame,
                target_column=target_name,
            )
            target_models[target_name] = (estimator,)
            best_iterations.append(self._best_iteration(estimator))
            fit_seconds_total += fit_seconds

        predictor = self._build_predictor(target_models)
        validation_predictions = predictor.predict(self._feature_frame(val_frame))

        return AnchorTrainingResult(
            validation_predictions=np.asarray(validation_predictions, dtype=np.float64),
            validation_ids=np.asarray(
                val_frame.get_column(id_col).to_list(), dtype=str
            ),
            validation_eras=np.asarray(
                val_frame.get_column(era_col).to_list(), dtype=str
            ),
            best_iteration=max(best_iterations),
            fit_seconds=fit_seconds_total,
            backend=backend,
            predictor=predictor,
        )

    def train_cross_validation(
        self,
        df: FrameLike,
        splitter: PurgedEraSplitter,
        *,
        id_col: str = "id",
        era_col: str = "era",
    ) -> CrossValidationResult:
        base_lazy = (
            self._ensure_lazy_frame(df)
            .select([id_col, era_col, *self.target_columns, *self.feature_names])
            .with_row_index("__row_idx")
        )
        row_count = base_lazy.select(pl.len()).collect().item()
        oof_predictions = np.full(row_count, np.nan, dtype=np.float64)
        fold_results: list[TrainingFoldResult] = []
        models: list[EstimatorType] = []
        target_models_accumulator: dict[str, list[EstimatorType]] = {
            target_name: [] for target_name in self.target_columns
        }

        total_fit_seconds = 0.0
        backend_label = ""
        for fold_number, (train_eras, validation_eras) in enumerate(
            splitter.split(base_lazy, era_col=era_col),
            start=1,
        ):
            train_frame = base_lazy.filter(pl.col(era_col).is_in(train_eras)).collect()
            val_frame = base_lazy.filter(
                pl.col(era_col).is_in(validation_eras)
            ).collect()

            prediction_blocks: list[np.ndarray] = []
            best_iterations: list[int] = []
            fold_fit_seconds = 0.0
            backend = ""
            for target_name in self.target_columns:
                estimator, backend, fit_seconds = self._fit_with_fallback(
                    train_frame,
                    val_frame,
                    target_column=target_name,
                )
                prediction_blocks.append(
                    np.asarray(
                        estimator.predict(self._feature_frame(val_frame)),
                        dtype=np.float64,
                    )
                )
                target_models_accumulator[target_name].append(estimator)
                models.append(estimator)
                best_iterations.append(self._best_iteration(estimator))
                fold_fit_seconds += fit_seconds

            predictions = self.weighted_rank_average_predictions(
                prediction_blocks,
                self._target_weight_vector(self.target_columns),
            )
            row_indices = val_frame.get_column("__row_idx").to_numpy()
            oof_predictions[row_indices] = np.asarray(predictions, dtype=np.float64)

            fold_results.append(
                TrainingFoldResult(
                    fold_number=fold_number,
                    train_eras=tuple(train_eras.tolist()),
                    validation_eras=tuple(validation_eras.tolist()),
                    validation_row_indices=row_indices,
                    validation_predictions=np.asarray(predictions, dtype=np.float64),
                    best_iteration=max(best_iterations),
                    fit_seconds=fold_fit_seconds,
                    backend=backend,
                )
            )
            total_fit_seconds += fold_fit_seconds
            backend_label = backend

        predictor = self._build_predictor(
            {
                target_name: tuple(target_models)
                for target_name, target_models in target_models_accumulator.items()
            }
        )

        return CrossValidationResult(
            oof_predictions=oof_predictions,
            fold_results=fold_results,
            models=models,
            predictor=predictor,
            backend=backend_label,
            total_fit_seconds=total_fit_seconds,
        )

    def predict_ensemble(
        self,
        df: FrameLike,
        *,
        models: Sequence[EstimatorType],
    ) -> np.ndarray:
        if not models:
            raise ValueError("models must be non-empty")
        frame = self._collect_prediction_columns(df)
        features = self._feature_frame(frame)
        predictions = [
            np.asarray(model.predict(features), dtype=np.float64) for model in models
        ]
        return self.rank_average_predictions(predictions)

    @staticmethod
    def rank_average_predictions(predictions: Sequence[np.ndarray]) -> np.ndarray:
        return ModelOrchestrator.weighted_rank_average_predictions(predictions)

    @staticmethod
    def weighted_rank_average_predictions(
        predictions: Sequence[np.ndarray],
        weights: Sequence[float] | None = None,
    ) -> np.ndarray:
        if not predictions:
            raise ValueError("predictions must be non-empty")
        if weights is not None and len(weights) != len(predictions):
            raise ValueError("weights length must match predictions length")

        ranked = pd.DataFrame(
            {
                f"prediction_{index:02d}": np.asarray(
                    prediction, dtype=np.float64
                ).reshape(-1)
                for index, prediction in enumerate(predictions, start=1)
            }
        ).rank(pct=True, method="average")

        if weights is None:
            normalized_weights = np.full(len(predictions), 1.0 / len(predictions))
        else:
            normalized_weights = np.asarray(weights, dtype=np.float64)
            if np.any(normalized_weights < 0.0):
                raise ValueError("weights must be non-negative")
            weight_sum = float(normalized_weights.sum())
            if weight_sum <= 0.0:
                raise ValueError("weights must sum to a positive value")
            normalized_weights = normalized_weights / weight_sum

        averaged = ranked.to_numpy() @ normalized_weights
        averaged_series = pd.Series(averaged)
        return np.asarray(
            averaged_series.rank(pct=True, method="first"),
            dtype=np.float64,
        )

    def _build_predictor(
        self,
        target_models: dict[str, tuple[EstimatorType, ...]],
    ) -> MultiTargetPredictor:
        if self.target_weights is None:
            return MultiTargetPredictor(
                feature_names=self.feature_names,
                target_models=target_models,
            )
        return WeightedMultiTargetPredictor(
            feature_names=self.feature_names,
            target_models=target_models,
            target_weights=self.target_weights,
        )

    def _resolve_target_weights(
        self,
        target_weights: dict[str, float] | Sequence[float] | None,
    ) -> dict[str, float] | None:
        if target_weights is None:
            return None
        if isinstance(target_weights, dict):
            missing = sorted(set(self.target_columns).difference(target_weights))
            if missing:
                raise ValueError(
                    f"target_weights is missing entries for: {', '.join(missing)}"
                )
            return {
                target_name: float(target_weights[target_name])
                for target_name in self.target_columns
            }

        if len(target_weights) != len(self.target_columns):
            raise ValueError("target_weights length must match target_columns length")
        return {
            target_name: float(weight)
            for target_name, weight in zip(
                self.target_columns, target_weights, strict=True
            )
        }

    def _target_weight_vector(self, target_names: Sequence[str]) -> list[float]:
        if self.target_weights is None:
            return [1.0 / len(target_names)] * len(target_names)
        return [self.target_weights[target_name] for target_name in target_names]

    def _collect_required_columns(
        self,
        df: FrameLike,
        *,
        id_col: str = "id",
        era_col: str = "era",
        require_target: bool = True,
    ) -> pl.DataFrame:
        columns = [id_col, era_col, *self.feature_names]
        if require_target:
            columns.extend(self.target_columns)
        return self._ensure_lazy_frame(df).select(columns).collect()

    def _collect_prediction_columns(
        self,
        df: FrameLike,
        *,
        id_col: str = "id",
    ) -> pl.DataFrame:
        columns = [id_col, *self.feature_names]
        return self._ensure_lazy_frame(df).select(columns).collect()

    @staticmethod
    def _ensure_lazy_frame(df: FrameLike) -> pl.LazyFrame:
        return df.lazy() if isinstance(df, pl.DataFrame) else df

    def _feature_frame(self, frame: pl.DataFrame) -> pd.DataFrame:
        return frame.select(self.feature_names).to_pandas()

    def _fit_with_fallback(
        self,
        train_frame: pl.DataFrame,
        val_frame: pl.DataFrame,
        *,
        target_column: str,
    ) -> tuple[EstimatorType, str, float]:
        train_x = self._feature_frame(train_frame)
        train_y = train_frame.get_column(target_column).to_numpy()
        val_x = self._feature_frame(val_frame)
        val_y = val_frame.get_column(target_column).to_numpy()

        backends = [self.prefer_gpu, False] if self.prefer_gpu else [False]
        last_error: Exception | None = None
        for use_gpu in backends:
            estimator = self._build_estimator(use_gpu=use_gpu)
            start = perf_counter()
            try:
                self._fit_estimator(estimator, train_x, train_y, val_x, val_y)
                fit_seconds = perf_counter() - start
                return estimator, self._backend_name(use_gpu), fit_seconds
            except Exception as exc:
                last_error = exc
                if not use_gpu:
                    break

        if last_error is not None:
            raise last_error
        raise RuntimeError("Model fitting failed without raising an explicit exception")

    def _build_estimator(self, *, use_gpu: bool) -> EstimatorType:
        params = self._resolved_params(use_gpu=use_gpu)
        if self.model_library == "lightgbm":
            return lgb.LGBMRegressor(**params)
        return xgb.XGBRegressor(**params)

    def _fit_estimator(
        self,
        estimator: EstimatorType,
        train_x: pd.DataFrame,
        train_y: np.ndarray,
        val_x: pd.DataFrame,
        val_y: np.ndarray,
    ) -> None:
        if self.model_library == "lightgbm":
            estimator.fit(
                train_x,
                train_y,
                eval_set=[(val_x, val_y)],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False)
                ],
            )
            return
        estimator.fit(
            train_x,
            train_y,
            eval_set=[(val_x, val_y)],
            verbose=False,
        )

    def _resolved_params(self, *, use_gpu: bool) -> dict[str, Any]:
        defaults = self._default_params(use_gpu=use_gpu)
        resolved = dict(defaults)
        resolved.update(self.model_params)
        if self.model_library == "xgboost":
            resolved.setdefault("early_stopping_rounds", self.early_stopping_rounds)
        return resolved

    def _default_params(self, *, use_gpu: bool) -> dict[str, Any]:
        if self.model_library == "lightgbm":
            return {
                "objective": "regression",
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "verbosity": -1,
                "device_type": "gpu" if use_gpu else "cpu",
            }

        return {
            "objective": "reg:squarederror",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
            "tree_method": "hist",
            "device": "cuda" if use_gpu else "cpu",
            "eval_metric": "rmse",
        }

    def _backend_name(self, use_gpu: bool) -> str:
        return f"{self.model_library}-{'gpu' if use_gpu else 'cpu'}"

    @staticmethod
    def _best_iteration(estimator: EstimatorType) -> int:
        best_iteration = getattr(estimator, "best_iteration_", None)
        if best_iteration is None:
            best_iteration = getattr(estimator, "best_iteration", None)
        if best_iteration is None:
            return int(getattr(estimator, "n_estimators", 0))
        return int(best_iteration)
