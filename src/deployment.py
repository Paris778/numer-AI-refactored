from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from src.evaluation import EvaluationEngine, EvaluationSummary
from src.models import EstimatorType

LoadedModelType = lgb.Booster | xgb.Booster


@dataclass(frozen=True)
class StressTestResult:
    clean_summary: EvaluationSummary
    stressed_summary: EvaluationSummary
    degradation_pct: float
    passed: bool
    clean_predictions: np.ndarray
    stressed_predictions: np.ndarray


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
        models: Sequence[EstimatorType | LoadedModelType],
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
        models: Sequence[EstimatorType | LoadedModelType],
        feature_names: Sequence[str],
    ) -> np.ndarray:
        if not models:
            raise ValueError("models must be non-empty")
        feature_frame = df.select(list(feature_names)).to_pandas()
        predictions = np.vstack(
            [self._predict_single_model(model, feature_frame) for model in models]
        )
        return predictions.mean(axis=0)

    @staticmethod
    def _predict_single_model(
        model: EstimatorType | LoadedModelType,
        feature_frame: pd.DataFrame,
    ) -> np.ndarray:
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
    def serialize_candidate(
        self,
        target_dir: Path,
        models: Sequence[EstimatorType | LoadedModelType],
        feature_names: Sequence[str],
        config_metadata: dict[str, Any],
        *,
        requirements_path: Path | None = None,
    ) -> Path:
        if not models:
            raise ValueError("models must be non-empty")
        if not feature_names:
            raise ValueError("feature_names must be non-empty")

        artifact_dir = Path(target_dir).expanduser().resolve()
        artifact_dir.mkdir(parents=True, exist_ok=False)
        model_library = self._infer_model_library(models)
        model_files: list[str] = []
        model_hashes: dict[str, str] = {}

        for index, model in enumerate(models, start=1):
            model_path = artifact_dir / self._model_filename(model_library, index)
            self._save_native_model(model, model_path, model_library)
            model_files.append(model_path.name)
            model_hashes[model_path.name] = self._sha256_file(model_path)

        manifest = {
            "manifest_version": 1,
            "model_library": model_library,
            "feature_names": list(feature_names),
            "model_files": model_files,
            "model_hashes": model_hashes,
            "environment": self._environment_fingerprint(
                requirements_path=requirements_path
            ),
            "config_metadata": config_metadata,
        }
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return artifact_dir

    def predict_live(self, live_df: pl.DataFrame, artifact_dir: Path) -> pl.DataFrame:
        artifact_root = Path(artifact_dir).expanduser().resolve()
        manifest_path = artifact_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest is missing at {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        feature_names = tuple(manifest["feature_names"])
        model_library = str(manifest["model_library"])
        self._verify_model_hashes(artifact_root, manifest)
        if "id" not in live_df.columns:
            raise KeyError("Live dataframe must include an 'id' column")
        missing_features = sorted(set(feature_names).difference(live_df.columns))
        if missing_features:
            raise KeyError(
                f"Live dataframe is missing required features: {', '.join(missing_features)}"
            )

        reordered = live_df.select(["id", *feature_names])
        feature_frame = reordered.select(list(feature_names)).to_pandas()
        models = [
            self._load_native_model(artifact_root / model_file, model_library)
            for model_file in manifest["model_files"]
        ]
        predictions = np.vstack(
            [
                AdversarialStressTester._predict_single_model(model, feature_frame)
                for model in models
            ]
        ).mean(axis=0)
        return pl.DataFrame(
            {
                "id": reordered.get_column("id").to_list(),
                "prediction": predictions.astype(np.float64),
            }
        )

    @staticmethod
    def _infer_model_library(models: Sequence[EstimatorType | LoadedModelType]) -> str:
        libraries = {DeploymentHarness._single_model_library(model) for model in models}
        if len(libraries) != 1:
            raise ValueError("Mixed model libraries are not supported in one artifact")
        return libraries.pop()

    @staticmethod
    def _single_model_library(model: EstimatorType | LoadedModelType) -> str:
        if isinstance(model, (lgb.LGBMRegressor, lgb.Booster)):
            return "lightgbm"
        if isinstance(model, (xgb.XGBRegressor, xgb.Booster)):
            return "xgboost"
        raise TypeError(f"Unsupported model type: {type(model)!r}")

    @staticmethod
    def _model_filename(model_library: str, index: int) -> str:
        suffix = "txt" if model_library == "lightgbm" else "json"
        return f"model_{index:02d}.{suffix}"

    @staticmethod
    def _save_native_model(
        model: EstimatorType | LoadedModelType,
        path: Path,
        model_library: str,
    ) -> None:
        if model_library == "lightgbm":
            booster = model.booster_ if isinstance(model, lgb.LGBMRegressor) else model
            booster.save_model(str(path))
            return
        if model_library == "xgboost":
            booster = (
                model.get_booster() if isinstance(model, xgb.XGBRegressor) else model
            )
            booster.save_model(str(path))
            return
        raise ValueError(f"Unsupported model library: {model_library}")

    @staticmethod
    def _load_native_model(path: Path, model_library: str) -> LoadedModelType:
        if model_library == "lightgbm":
            return lgb.Booster(model_file=str(path))
        if model_library == "xgboost":
            booster = xgb.Booster()
            booster.load_model(str(path))
            return booster
        raise ValueError(f"Unsupported model library: {model_library}")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _verify_model_hashes(
        self, artifact_root: Path, manifest: dict[str, Any]
    ) -> None:
        model_files = manifest.get("model_files")
        model_hashes = manifest.get("model_hashes")
        if not isinstance(model_files, list) or not isinstance(model_hashes, dict):
            raise ValueError("Manifest is missing model_files or model_hashes")

        for model_file in model_files:
            model_path = artifact_root / model_file
            if not model_path.exists():
                raise FileNotFoundError(f"Model file is missing at {model_path}")
            expected_hash = model_hashes.get(model_file)
            if not isinstance(expected_hash, str):
                raise ValueError(
                    f"Manifest is missing hash entry for model file '{model_file}'"
                )
            actual_hash = self._sha256_file(model_path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"SHA-256 mismatch for model file '{model_file}': expected {expected_hash}, got {actual_hash}"
                )

    @staticmethod
    def _environment_fingerprint(
        *,
        requirements_path: Path | None,
    ) -> dict[str, Any]:
        resolved_requirements = (
            Path(requirements_path).expanduser().resolve()
            if requirements_path is not None
            else (Path.cwd() / "requirements.txt").resolve()
        )
        requirements_sha256 = None
        if resolved_requirements.exists():
            requirements_sha256 = DeploymentHarness._sha256_file(resolved_requirements)

        critical_packages = ("polars", "numpy", "lightgbm", "xgboost", "scipy")
        package_versions = {
            package_name: importlib.metadata.version(package_name)
            for package_name in critical_packages
        }

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "package_versions": package_versions,
            "requirements_path": (
                str(resolved_requirements) if resolved_requirements.exists() else None
            ),
            "requirements_sha256": requirements_sha256,
        }
