"""Deterministic model orchestration for leakage-safe era validation.

`ModelOrchestrator` is the narrow training boundary for tree models. It only
does three things:

- resolve canonical preset params from `ModelConfig`
- fit one model per leakage-safe fold from `PurgedEraSplitter`
- emit raw, out-of-fold predictions as a Polars frame

The splitter owns chronology and purge semantics. This module consumes those
folds directly and refuses to widen scope into ranking, ensembling, or scoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from nmr.config import ModelConfig
from nmr.splitter import Fold, PurgedEraSplitter

__all__ = ["CVResult", "ModelOrchestrator"]


_CANONICAL_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 5,
        "num_leaves": (2**5) - 1,
        "colsample_bytree": 0.1,
    },
    "standard": {
        "n_estimators": 20000,
        "learning_rate": 0.001,
        "max_depth": 6,
        "num_leaves": 2**6,
        "colsample_bytree": 0.1,
    },
    "deep": {
        "n_estimators": 30000,
        "learning_rate": 0.001,
        "max_depth": 10,
        "num_leaves": 1024,
        "colsample_bytree": 0.1,
        "min_data_in_leaf": 10000,
    },
}


@dataclass(frozen=True)
class CVResult:
    oof: pl.DataFrame
    models: tuple[object, ...]


class ModelOrchestrator:
    def __init__(self, config: ModelConfig, *, seed: int = 42) -> None:
        self._config = config
        self._seed = seed

    def train_anchor_fold(
        self,
        df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        target_col: str,
        splitter: PurgedEraSplitter,
        era_col: str = "era",
    ) -> tuple[object, pl.DataFrame]:
        folds = splitter.split(df.get_column(era_col).to_list())
        if len(folds) != 1:
            raise ValueError(
                "train_anchor_fold requires exactly one fold; use anchor splitting"
            )

        fold = folds[0]
        return self._fit_predict_fold(
            df,
            fold=fold,
            feature_cols=feature_cols,
            target_col=target_col,
            era_col=era_col,
        )

    def train_cross_validation(
        self,
        df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        target_col: str,
        splitter: PurgedEraSplitter,
        era_col: str = "era",
    ) -> CVResult:
        folds = splitter.split(df.get_column(era_col).to_list())
        models: list[object] = []
        oof_parts: list[pl.DataFrame] = []
        seen_val_eras: set[str] = set()

        for fold in folds:
            overlap = seen_val_eras & set(fold.val_eras)
            if overlap:
                raise ValueError(
                    f"Validation eras must be disjoint across folds, got {sorted(overlap)}"
                )

            model, fold_predictions = self._fit_predict_fold(
                df,
                fold=fold,
                feature_cols=feature_cols,
                target_col=target_col,
                era_col=era_col,
            )
            models.append(model)
            oof_parts.append(fold_predictions)
            seen_val_eras.update(fold.val_eras)

        if not oof_parts:
            raise ValueError("No folds produced OOF predictions")

        oof = pl.concat(oof_parts, how="vertical")
        return CVResult(oof=oof, models=tuple(models))

    def _fit_predict_fold(
        self,
        df: pl.DataFrame,
        *,
        fold: Fold,
        feature_cols: Sequence[str],
        target_col: str,
        era_col: str,
    ) -> tuple[object, pl.DataFrame]:
        self._assert_fold_is_leakage_safe(fold)
        train_df = df.filter(pl.col(era_col).is_in(fold.train_eras))
        val_df = df.filter(pl.col(era_col).is_in(fold.val_eras))
        if train_df.is_empty() or val_df.is_empty():
            raise ValueError(f"Degenerate training slice for fold {fold.index}")

        model = self._fit_model(
            features=self._feature_frame(train_df, feature_cols=feature_cols),
            target=train_df.get_column(target_col).to_numpy(),
        )
        prediction = self._predict_model(
            model,
            features=self._feature_frame(val_df, feature_cols=feature_cols),
        )
        pred_frame = val_df.select(["id", era_col]).rename({era_col: "era"})
        pred_frame = pred_frame.with_columns(
            pl.Series("prediction", np.asarray(prediction, dtype=float).reshape(-1))
        )
        return model, pred_frame

    def _feature_frame(
        self, df: pl.DataFrame, *, feature_cols: Sequence[str]
    ) -> pd.DataFrame:
        feature_frame = df.select(feature_cols).to_pandas()
        return feature_frame.loc[:, list(feature_cols)]

    def _fit_model(self, *, features: pd.DataFrame, target: np.ndarray) -> object:
        candidate_params = self._device_candidate_params()
        last_error: Exception | None = None

        for params in candidate_params:
            model = self._build_model(params)
            try:
                model.fit(features, target)
                return model
            except Exception as exc:
                last_error = exc

        assert last_error is not None
        raise last_error

    def _predict_model(self, model: object, *, features: pd.DataFrame) -> np.ndarray:
        prediction = model.predict(features)
        return np.asarray(prediction, dtype=float).reshape(-1)

    def _device_candidate_params(self) -> list[dict[str, Any]]:
        cpu_params = self._resolved_params(use_gpu=False)
        gpu_params = self._resolved_params(use_gpu=True)
        if gpu_params == cpu_params:
            return [cpu_params]
        return [gpu_params, cpu_params]

    def _resolved_params(self, *, use_gpu: bool) -> dict[str, Any]:
        base = dict(_CANONICAL_PRESETS[self._config.preset])
        base.update(self._config.params)

        if self._config.backend == "lightgbm":
            params = {
                "objective": "regression",
                "random_state": self._seed,
                "n_jobs": 1,
                "deterministic": True,
                "force_col_wise": True,
                "verbosity": -1,
                **base,
            }
            params["device_type"] = "gpu" if use_gpu else "cpu"
            return params

        params = {
            "objective": "reg:squarederror",
            "random_state": self._seed,
            "seed": self._seed,
            "n_jobs": 1,
            "verbosity": 0,
            "subsample": 1.0,
            "colsample_bylevel": 1.0,
            **base,
        }
        num_leaves = params.pop("num_leaves", None)
        min_data_in_leaf = params.pop("min_data_in_leaf", None)
        if num_leaves is not None:
            params.setdefault("grow_policy", "lossguide")
            params.setdefault("max_leaves", num_leaves)
        if min_data_in_leaf is not None:
            params.setdefault("min_child_weight", float(min_data_in_leaf))
        params["tree_method"] = "gpu_hist" if use_gpu else "hist"
        return params

    def _build_model(self, params: dict[str, Any]) -> object:
        if self._config.backend == "lightgbm":
            return lgb.LGBMRegressor(**params)
        return xgb.XGBRegressor(**params)

    def _assert_fold_is_leakage_safe(self, fold: Fold) -> None:
        train_eras = {int(era) for era in fold.train_eras}
        val_eras = {int(era) for era in fold.val_eras}
        if train_eras & val_eras:
            raise ValueError(f"Fold {fold.index} reuses eras across train/val")

        train_max = max(train_eras)
        val_min = min(val_eras)
        purge_buffer = set(range(train_max + 1, val_min))
        if train_eras & purge_buffer:
            raise ValueError(f"Fold {fold.index} leaked purge buffer into training")
        if val_eras & purge_buffer:
            raise ValueError(f"Fold {fold.index} leaked purge buffer into validation")
