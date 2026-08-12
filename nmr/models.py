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

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from nmr.config import ModelConfig
from nmr.splitter import Fold, PurgedEraSplitter

logger = logging.getLogger("nmr.models")

__all__ = ["CVResult", "ModelOrchestrator", "coerce_float32_features"]


_FIT_PROGRESS_PERIOD = 100  # print one training progress line every N iterations
_PREDICT_ERA_BATCH = 20  # eras per predict call (bounds GPU VRAM / RAM)

# Dtypes that are exactly representable in float32 (the Numerai v5.x integer
# feature bins). Casting these to a single Float32 block BEFORE pandas makes
# LightGBM/XGBoost's `to_numpy(dtype=float32)` a zero-copy view instead of a
# full dense copy — at 3,555 features × 2.1M rows that copy is ~28 GiB (the
# lgbm_v1 campaign OOM). Float64 columns are left untouched (precision).
_EXACT_FLOAT32_DTYPES = frozenset(
    {
        pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32, pl.Float32
    }
)


def coerce_float32_features(
    df: pl.DataFrame, feature_cols: Sequence[str]
) -> pl.DataFrame:
    """Return ``df`` with exactly-representable feature columns cast to Float32.

    Pure dtype normalization: no value changes (integer bins are exact in
    float32). Mixed or Float64 schemas are returned unchanged so full
    precision is never silently dropped. The result is a single-block pandas
    frame downstream, which removes the backend's dense float32 copy.
    """
    selected = df.select(feature_cols)
    schema = selected.schema
    if all(dt in _EXACT_FLOAT32_DTYPES for dt in schema.values()):
        return selected.cast(pl.Float32)
    return selected


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


def resolve_model_params(preset: str, params: dict[str, Any]) -> dict[str, Any]:
    """Resolve preset defaults overridden by explicit ``params``.

    Single source of truth for preset+params resolution (ARCHITECTURE.md §S):
    ``model.params`` wins over ``_CANONICAL_PRESETS[preset]``. Used by
    :meth:`ModelOrchestrator._resolved_params` and the Bayesian sweep anchor.
    """
    resolved = dict(_CANONICAL_PRESETS[preset])
    resolved.update(params)
    return resolved


def _translate_catboost(
    resolved: dict[str, Any], *, seed: int, use_gpu: bool
) -> dict[str, Any]:
    """Translate generic preset knobs to CatBoost names and apply the fixed contract.

    Renames ``n_estimators``/``colsample_bytree``/``max_depth`` to
    ``iterations``/``rsm``/``depth``, drops ``num_leaves`` (symmetric
    depth-limited trees only), passes every other key through unchanged, then
    overlays the deterministic contract. Contract values win over user params.
    Only kwargs verified present in the pinned catboost 1.2.10 are emitted.
    """
    rename = {
        "n_estimators": "iterations",
        "colsample_bytree": "rsm",
        "max_depth": "depth",
    }
    params = {rename.get(key, key): value for key, value in resolved.items()}
    params.pop("num_leaves", None)
    params.update(
        {
            "loss_function": "RMSE",
            "random_seed": seed,
            "thread_count": 1,
            "verbose": False,
            "allow_writing_files": False,  # CatBoost writes files by default; disable
            "task_type": "GPU" if use_gpu else "CPU",
        }
    )
    if use_gpu:
        params["devices"] = "0"
    return params


@dataclass(frozen=True)
class CVResult:
    oof: pl.DataFrame
    models: tuple[object, ...]


class ModelOrchestrator:
    def __init__(self, config: ModelConfig, *, seed: int = 42) -> None:
        self._config = config
        self._seed = seed
        self.resolved_device: str | None = None

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
            purge_eras=splitter.purge_eras,
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
        logger.info("[train_cross_validation] %s: %d folds", target_col, len(folds))
        models: list[object] = []
        oof_parts: list[pl.DataFrame] = []
        seen_val_eras: set[str] = set()

        for fold in folds:
            overlap = seen_val_eras & set(fold.val_eras)
            if overlap:
                raise ValueError(
                    f"Validation eras must be disjoint across folds, got {sorted(overlap)}"
                )

            logger.info(
                "[train_cross_validation] %s: fold %d/%d train_eras=%d val_eras=%d",
                target_col,
                fold.index + 1,
                len(folds),
                len(fold.train_eras),
                len(fold.val_eras),
            )
            t0 = time.time()
            model, fold_predictions = self._fit_predict_fold(
                df,
                fold=fold,
                feature_cols=feature_cols,
                target_col=target_col,
                era_col=era_col,
                purge_eras=splitter.purge_eras,
            )
            logger.info(
                "[train_cross_validation] %s: fold %d/%d trained in %.1fs",
                target_col,
                fold.index + 1,
                len(folds),
                time.time() - t0,
            )
            models.append(model)
            oof_parts.append(fold_predictions)
            seen_val_eras.update(fold.val_eras)

        if not oof_parts:
            raise ValueError("No folds produced OOF predictions")

        oof = pl.concat(oof_parts, how="vertical")
        logger.info(
            "[train_cross_validation] %s: OOF complete rows=%d", target_col, oof.height
        )
        return CVResult(oof=oof, models=tuple(models))

    def train_full_history(
        self,
        df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        target_col: str,
        era_col: str = "era",
    ) -> object:
        """Fit a single CPU-only model on every era (deployment/validation artifact).

        CPU-only by design: determinism is per-device and the deployed model must
        reproduce identically on any hosted runtime (which may lack a GPU).
        """
        train_df = df.filter(pl.col(era_col).is_not_null())
        train_df = train_df.filter(
            pl.col(target_col).is_not_null() & pl.col(target_col).is_finite()
        )
        dropped = df.height - train_df.height
        if dropped:
            logger.warning(
                "[train_full_history] dropped %d rows with null/non-finite %s targets",
                dropped,
                target_col,
            )
        if train_df.is_empty():
            raise ValueError("No usable training rows after null-target filtering")
        model = self._fit_model(
            features=self._feature_frame(train_df, feature_cols=feature_cols),
            target=train_df.get_column(target_col).to_numpy(),
            use_gpu=False,
        )
        logger.info(
            "[train_full_history] %s: fitted on %d rows (all eras)", target_col, train_df.height
        )
        return model

    def _fit_predict_fold(
        self,
        df: pl.DataFrame,
        *,
        fold: Fold,
        feature_cols: Sequence[str],
        target_col: str,
        era_col: str,
        purge_eras: int,
    ) -> tuple[object, pl.DataFrame]:
        self._assert_fold_is_leakage_safe(fold, purge_eras=purge_eras)
        train_df = df.filter(pl.col(era_col).is_in(fold.train_eras))
        val_df = df.filter(pl.col(era_col).is_in(fold.val_eras))
        if train_df.is_empty() or val_df.is_empty():
            raise ValueError(f"Degenerate training slice for fold {fold.index}")

        train_df = train_df.filter(
            pl.col(target_col).is_not_null() & pl.col(target_col).is_finite()
        )
        dropped = df.filter(pl.col(era_col).is_in(fold.train_eras)).height - train_df.height
        if dropped:
            logger.warning(
                "[_fit_predict_fold] %s fold %d: dropped %d rows with null/non-finite targets",
                target_col, fold.index, dropped,
            )
        if train_df.is_empty():
            raise ValueError(f"No usable training rows for fold {fold.index} after null filtering")

        logger.info(
            "[_fit_predict_fold] %s fold %d: fitting model on %d rows",
            target_col,
            fold.index,
            train_df.height,
        )
        t0 = time.time()
        model = self._fit_model(
            features=self._feature_frame(train_df, feature_cols=feature_cols),
            target=train_df.get_column(target_col).to_numpy(),
            use_gpu=self._config.device != "cpu",
        )
        logger.info(
            "[_fit_predict_fold] %s fold %d: fit complete in %.1fs",
            target_col,
            fold.index,
            time.time() - t0,
        )
        prediction = self._predict_model_chunked(model, val_df, feature_cols)
        pred_frame = val_df.select(["id", era_col]).rename({era_col: "era"})
        pred_frame = pred_frame.with_columns(
            pl.Series("prediction", np.asarray(prediction, dtype=float).reshape(-1))
        )
        return model, pred_frame

    def _feature_frame(
        self, df: pl.DataFrame, *, feature_cols: Sequence[str]
    ) -> np.ndarray:
        """Float32 feature matrix, zero-copy from the coerced polars frame.

        Skipping pandas entirely: polars→pandas goes through pyarrow and
        allocates a full second copy (~36 GiB at 3,555 × 2.1M — the lgbm_v1
        full-history OOM), while polars→numpy is zero-copy for the uniform
        Float32 block (verified empirically). Values are bit-identical to the
        pandas path: Int8 bins are exact in float32 and Float64 frames pass
        through untouched.
        """
        return coerce_float32_features(df, feature_cols).to_numpy()

    def _predict_model(self, model: object, *, features: np.ndarray) -> np.ndarray:
        prediction = model.predict(features)
        return np.asarray(prediction, dtype=float).reshape(-1)

    def _predict_model_chunked(
        self, model: object, val_df: pl.DataFrame, feature_cols: Sequence[str]
    ) -> np.ndarray:
        """Fold/val predict in era-batches to bound predict-time memory.

        At 3,555 features a full fold-val matrix is ~4.9 GiB float32 — above
        the 4 GiB GPU VRAM (xgb_v1 CUDA OOM) and heavy on RAM for CPU
        predictors. Per-era predictions are row-wise identical to the full
        frame, so concatenating era-batches is bit-identical to one call.
        """
        if val_df.is_empty():
            return np.zeros(0, dtype=float)
        eras = list(dict.fromkeys(val_df.get_column("era").to_list()))
        parts: list[np.ndarray] = []
        for start in range(0, len(eras), _PREDICT_ERA_BATCH):
            batch_eras = set(eras[start : start + _PREDICT_ERA_BATCH])
            part = val_df.filter(pl.col("era").is_in(batch_eras))
            if part.is_empty():
                continue
            parts.append(
                self._predict_model(
                    model, features=self._feature_frame(part, feature_cols=feature_cols)
                )
            )
        if not parts:
            return np.zeros(0, dtype=float)
        return np.concatenate(parts)

    def _fit_model(
        self, *, features: np.ndarray, target: np.ndarray, use_gpu: bool = True
    ) -> object:
        candidate_params = self._device_candidate_params(use_gpu=use_gpu)
        last_error: Exception | None = None
        if self._config.backend == "lightgbm":
            backend_errors = (ValueError, TypeError, lgb.basic.LightGBMError)
        elif self._config.backend == "catboost":
            backend_errors = (ValueError, TypeError, catboost.CatBoostError)
        else:
            backend_errors = (ValueError, TypeError, xgb.core.XGBoostError)

        for params in candidate_params:
            model = self._build_model(params)
            try:
                self._fit_with_progress(model, features, target)
            except backend_errors as exc:
                logger.warning(
                    "[fit] %s fit failed (%s: %s); trying next candidate",
                    self._config.backend, type(exc).__name__, exc,
                )
                last_error = exc
                continue
            self.resolved_device = (
                "gpu"
                if params.get("device_type") == "gpu"
                or params.get("device") == "cuda"
                or params.get("task_type") == "GPU"
                else "cpu"
            )
            return model

        assert last_error is not None
        raise last_error

    def _fit_with_progress(
        self, model: object, features: np.ndarray, target: np.ndarray
    ) -> None:
        """Fit with progress markers on stdout.

        LightGBM and CatBoost expose per-iteration hooks; the installed
        xgboost 3.x sklearn wrapper does not (no ``callbacks`` fit argument),
        so xgboost gets start/elapsed markers instead. Markers are
        output-only: they never touch the model's numeric results or the
        params dict, so determinism guarantees are unchanged.
        """
        backend = self._config.backend
        period = _FIT_PROGRESS_PERIOD
        if backend == "lightgbm":
            def _lgb_progress(env: Any) -> None:
                iteration = env.iteration + 1
                if iteration == 1 or iteration % period == 0:
                    print(f"[fit] lightgbm iteration {iteration}", flush=True)

            model.fit(features, target, callbacks=[_lgb_progress])
        elif backend == "xgboost":
            started = time.monotonic()
            print("[fit] xgboost training started", flush=True)
            model.fit(features, target)
            print(
                f"[fit] xgboost training done ({time.monotonic() - started:.1f}s)",
                flush=True,
            )
        else:  # catboost: period-based verbose logging at fit time
            model.fit(features, target, verbose=period)

    def _predict_model(self, model: object, *, features: np.ndarray) -> np.ndarray:
        prediction = model.predict(features)
        return np.asarray(prediction, dtype=float).reshape(-1)

    def _device_candidate_params(self, *, use_gpu: bool) -> list[dict[str, Any]]:
        if not use_gpu:
            return [self._resolved_params(use_gpu=False)]
        if self._config.backend == "catboost":
            # CPU-only by design: catboost rejects `rsm` on GPU (non-pairwise
            # modes) and every canonical preset ships colsample_bytree -> rsm,
            # so a GPU candidate can never fit. Never attempt one.
            return [self._resolved_params(use_gpu=False)]
        gpu_params = self._resolved_params(use_gpu=True)
        if self._config.device == "gpu":
            # Forced GPU: a failing GPU fit raises — no silent CPU fallback.
            return [gpu_params]
        cpu_params = self._resolved_params(use_gpu=False)
        if gpu_params == cpu_params:
            return [cpu_params]
        return [gpu_params, cpu_params]

    def _resolved_params(self, *, use_gpu: bool) -> dict[str, Any]:
        base = resolve_model_params(self._config.preset, self._config.params)

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

        if self._config.backend == "catboost":
            base = resolve_model_params(self._config.preset, self._config.params)
            return _translate_catboost(base, seed=self._seed, use_gpu=use_gpu)

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
        # xgboost >= 3.0 unified GPU acceleration under device='cuda';
        # tree_method='gpu_hist' was removed and raises Invalid Input.
        params["tree_method"] = "hist"
        params["device"] = "cuda" if use_gpu else "cpu"
        return params

    def _build_model(self, params: dict[str, Any]) -> object:
        if self._config.backend == "lightgbm":
            return lgb.LGBMRegressor(**params)
        if self._config.backend == "catboost":
            return catboost.CatBoostRegressor(**params)
        return xgb.XGBRegressor(**params)

    def _assert_fold_is_leakage_safe(self, fold: Fold, *, purge_eras: int) -> None:
        train_eras = {int(era) for era in fold.train_eras}
        val_eras = {int(era) for era in fold.val_eras}
        if train_eras & val_eras:
            raise ValueError(f"Fold {fold.index} reuses eras across train/val")
        if not train_eras or not val_eras:
            raise ValueError(f"Fold {fold.index} is degenerate")

        train_max = max(train_eras)
        val_min = min(val_eras)
        if train_max >= val_min:
            raise ValueError(f"Fold {fold.index} is not strictly time-ordered")
        if val_min - train_max <= purge_eras:
            raise ValueError(
                f"Fold {fold.index} violates purge invariant: gap "
                f"{val_min - train_max} <= purge_eras={purge_eras}"
            )
