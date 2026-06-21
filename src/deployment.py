from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cloudpickle
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from src.evaluation import EvaluationEngine, EvaluationSummary
from src.models import EstimatorType, ModelOrchestrator
from src.protocols import Predictor, Transformer

LoadedModelType = lgb.Booster | xgb.Booster


@dataclass(frozen=True)
class StressTestResult:
    clean_summary: EvaluationSummary
    stressed_summary: EvaluationSummary
    degradation_pct: float
    passed: bool
    clean_predictions: np.ndarray
    stressed_predictions: np.ndarray


@dataclass(frozen=True)
class NumeraiPayloadBundle:
    predict: Callable[[pd.DataFrame, pd.DataFrame | None], pd.DataFrame]
    payload_path: Path


class AdversarialStressTester:
    def __init__(
        self,
        evaluation_engine: EvaluationEngine,
        *,
        degradation_threshold: float = 0.40,
        random_state: int = 42,
    ) -> None:
        self.evaluation_engine = evaluation_engine
        self.degradation_threshold = degradation_threshold
        self.random_state = random_state

    def evaluate_noise_resilience(
        self,
        df: pl.DataFrame,
        models: Sequence[EstimatorType | LoadedModelType] | Predictor,
        feature_names: Sequence[str],
        *,
        noise_std_ratio: float = 0.05,
        benchmark_prediction_col: str | None = None,
        neutralization_engine=None,
        neutralization_subset_name: str | None = None,
    ) -> StressTestResult:
        if noise_std_ratio < 0.0:
            raise ValueError("noise_std_ratio must be non-negative")
        frame = self._validate_frame(df, feature_names)
        clean_predictions = self._predict_ensemble(frame, models, feature_names)
        clean_frame = frame.with_columns(
            pl.Series(
                self.evaluation_engine.prediction_col,
                clean_predictions,
                dtype=pl.Float64,
            )
        )
        clean_summary = self.evaluation_engine.summarize(
            self.evaluation_engine.evaluate_eras(
                clean_frame,
                feature_columns=feature_names,
                benchmark_prediction_col=benchmark_prediction_col,
                neutralization_engine=neutralization_engine,
                neutralization_subset_name=neutralization_subset_name,
            )
        )

        rng = np.random.default_rng(self.random_state)
        noisy_frame = self._inject_noise(frame, feature_names, noise_std_ratio, rng)
        stressed_predictions = self._predict_ensemble(
            noisy_frame, models, feature_names
        )
        stressed_eval_frame = noisy_frame.with_columns(
            pl.Series(
                self.evaluation_engine.prediction_col,
                stressed_predictions,
                dtype=pl.Float64,
            )
        )
        stressed_summary = self.evaluation_engine.summarize(
            self.evaluation_engine.evaluate_eras(
                stressed_eval_frame,
                feature_columns=feature_names,
                benchmark_prediction_col=benchmark_prediction_col,
                neutralization_engine=neutralization_engine,
                neutralization_subset_name=neutralization_subset_name,
            )
        )

        degradation_pct = self._degradation_pct(
            clean_summary.mean_corr,
            stressed_summary.mean_corr,
        )
        return StressTestResult(
            clean_summary=clean_summary,
            stressed_summary=stressed_summary,
            degradation_pct=degradation_pct,
            passed=degradation_pct <= self.degradation_threshold * 100.0,
            clean_predictions=clean_predictions,
            stressed_predictions=stressed_predictions,
        )

    @staticmethod
    def _degradation_pct(clean_mean_corr: float, stressed_mean_corr: float) -> float:
        denominator = max(abs(clean_mean_corr), 1e-12)
        drop = max(clean_mean_corr - stressed_mean_corr, 0.0)
        return float((drop / denominator) * 100.0)

    def _inject_noise(
        self,
        df: pl.DataFrame,
        feature_names: Sequence[str],
        noise_std_ratio: float,
        rng: np.random.Generator,
    ) -> pl.DataFrame:
        feature_matrix = df.select(list(feature_names)).to_numpy().astype(np.float64)
        feature_std = np.std(feature_matrix, axis=0)
        noise_scale = noise_std_ratio * feature_std
        noise = rng.normal(
            loc=0.0,
            scale=np.broadcast_to(noise_scale, feature_matrix.shape),
            size=feature_matrix.shape,
        )
        noisy_matrix = feature_matrix + noise
        noisy_columns = [
            pl.Series(name, noisy_matrix[:, idx], dtype=pl.Float64)
            for idx, name in enumerate(feature_names)
        ]
        return df.with_columns(noisy_columns)

    def _predict_ensemble(
        self,
        df: pl.DataFrame,
        models: Sequence[EstimatorType | LoadedModelType] | Predictor,
        feature_names: Sequence[str],
    ) -> np.ndarray:
        if hasattr(models, "predict") and not isinstance(models, Sequence):
            feature_frame = df.select(list(feature_names)).to_pandas()
            return np.asarray(models.predict(feature_frame), dtype=np.float64)
        if not models:
            raise ValueError("models must be non-empty")
        feature_frame = df.select(list(feature_names)).to_pandas()
        predictions = [
            self._predict_single_model(model, feature_frame) for model in models
        ]
        return ModelOrchestrator.rank_average_predictions(predictions)

    @staticmethod
    def _predict_single_model(
        model: Predictor | EstimatorType | LoadedModelType,
        feature_frame: pd.DataFrame,
    ) -> np.ndarray:
        if hasattr(model, "predict") and not isinstance(
            model, (lgb.LGBMRegressor, lgb.Booster, xgb.XGBRegressor, xgb.Booster)
        ):
            return np.asarray(model.predict(feature_frame), dtype=np.float64)
        if isinstance(model, lgb.LGBMRegressor):
            return np.asarray(model.predict(feature_frame), dtype=np.float64)
        if isinstance(model, lgb.Booster):
            return np.asarray(model.predict(feature_frame), dtype=np.float64)
        if isinstance(model, xgb.XGBRegressor):
            return np.asarray(model.predict(feature_frame), dtype=np.float64)
        if isinstance(model, xgb.Booster):
            return np.asarray(
                model.predict(
                    xgb.DMatrix(
                        feature_frame, feature_names=list(feature_frame.columns)
                    )
                ),
                dtype=np.float64,
            )
        raise TypeError(f"Unsupported model type: {type(model)!r}")

    def _validate_frame(
        self,
        df: pl.DataFrame,
        feature_names: Sequence[str],
    ) -> pl.DataFrame:
        required = {
            self.evaluation_engine.era_col,
            self.evaluation_engine.target_col,
            self.evaluation_engine.id_col,
            *feature_names,
        }
        missing = sorted(required.difference(df.columns))
        if missing:
            raise KeyError(f"Missing required columns: {', '.join(missing)}")
        return df


class DeploymentHarness:
    def build_numerai_payload(
        self,
        *,
        feature_pipeline: Transformer,
        predictor_ensemble: Predictor,
        feature_names: Sequence[str],
    ) -> Callable[[pd.DataFrame, pd.DataFrame | None], pd.DataFrame]:
        if not feature_names:
            raise ValueError("feature_names must be non-empty")

        locked_features = tuple(feature_names)

        def predict(
            live_features: pd.DataFrame,
            live_benchmark_models: pd.DataFrame | None = None,
        ) -> pd.DataFrame:
            if not isinstance(live_features, pd.DataFrame):
                raise TypeError("live_features must be a pandas DataFrame")
            missing_features = sorted(
                set(locked_features).difference(live_features.columns)
            )
            if missing_features:
                raise KeyError(
                    f"Live dataframe is missing required features: {', '.join(missing_features)}"
                )
            ordered_features = live_features.loc[:, locked_features]
            transformed = feature_pipeline.transform(pl.from_pandas(ordered_features))
            transformed_pandas = transformed.select(list(locked_features)).to_pandas()
            if hasattr(predictor_ensemble, "predict_all"):
                raw_predictions = predictor_ensemble.predict_all(transformed_pandas)
            else:
                raw_predictions = {
                    "model_01": np.asarray(
                        predictor_ensemble.predict(transformed_pandas),
                        dtype=np.float64,
                    )
                }
            ranked = pd.DataFrame(raw_predictions, index=ordered_features.index).rank(
                pct=True,
                method="average",
            )
            final_predictions = ranked.mean(axis=1).rank(pct=True, method="first")
            return pd.DataFrame(
                {"prediction": final_predictions.values},
                index=live_features.index,
            )

        return predict

    def serialize_payload(
        self,
        payload: Callable[[pd.DataFrame, pd.DataFrame | None], pd.DataFrame],
        target_path: Path,
    ) -> NumeraiPayloadBundle:
        payload_path = Path(target_path).expanduser().resolve()
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        with payload_path.open("wb") as file_handle:
            cloudpickle.dump(payload, file_handle)
        return NumeraiPayloadBundle(predict=payload, payload_path=payload_path)

    def load_payload(
        self,
        payload_path: Path,
    ) -> Callable[[pd.DataFrame, pd.DataFrame | None], pd.DataFrame]:
        resolved = Path(payload_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Payload is missing at {resolved}")
        with resolved.open("rb") as file_handle:
            payload = cloudpickle.load(file_handle)
        if not callable(payload):
            raise TypeError("Loaded payload is not callable")
        return payload

    def predict_live(
        self,
        live_df: pl.DataFrame,
        payload_path: Path,
        *,
        live_benchmark_models: pd.DataFrame | None = None,
    ) -> pl.DataFrame:
        if "id" not in live_df.columns:
            raise KeyError("Live dataframe must include an 'id' column")
        payload = self.load_payload(payload_path)
        live_pandas = live_df.to_pandas().set_index("id")
        output = payload(live_pandas, live_benchmark_models)
        if not isinstance(output, pd.DataFrame):
            raise TypeError("Payload predict callable must return a pandas DataFrame")
        if "prediction" not in output.columns:
            raise KeyError("Payload output must contain a 'prediction' column")
        return pl.DataFrame(
            {
                "id": output.index.to_list(),
                "prediction": output["prediction"].to_numpy(dtype=np.float64),
            }
        )
