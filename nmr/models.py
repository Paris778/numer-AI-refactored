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

import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import catboost
import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb

from nmr._atomicio import atomic_write_bytes
from nmr.config import DataConfig, ModelConfig
from nmr.splitter import Fold, PurgedEraSplitter

logger = logging.getLogger("nmr.models")

__all__ = ["CVResult", "ModelOrchestrator", "coerce_float32_features"]


_FIT_PROGRESS_PERIOD = 100  # print one training progress line every N iterations
_PREDICT_ERA_BATCH = 20  # eras per predict call (bounds GPU VRAM / RAM)

# Full-history fits whose float32 matrix exceeds this run in a fresh process
# (see ModelOrchestrator.train_full_history docstring for the VA-accumulation
# rationale). 8 GiB covers medium (780) and smaller; all (3,555) spawns.
_FULL_HISTORY_SUBPROCESS_MIN_BYTES = 8 * 2**30

# Parent-side guard for the spawned full-history fit: poll the child's
# liveness while waiting on the result queue so a child that dies before
# reporting raises promptly instead of blocking the parent forever. A
# still-alive child is waited on indefinitely — a legitimate full-history
# fit can run for hours, so no overall timeout is applied.
_SUBPROCESS_POLL_INTERVAL_SECONDS = 5.0
_SUBPROCESS_DRAIN_GRACE_SECONDS = 5.0

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


_COLSAMPLE_FLOOR_MIN = 0.1
_COLSAMPLE_FLOOR_TARGET_FEATURES = 10
_COLSAMPLE_FLOOR_EPS = 1e-7

# LightGBM's sampling aliases form one _ConfigAliases group in the installed
# sklearn wrapper (verified on lightgbm 4.6.0); unknown kwargs flow through
# **kwargs into the native engine, so every present member must be floored.
_LGBM_COLSAMPLE_ALIASES = ("colsample_bytree", "feature_fraction", "sub_feature")


def _colsample_floor(n_features: int) -> float:
    """Dynamic lower bound for the feature-sampling fraction.

    Small feature sets must not be crippled by sampling ~1 feature per tree:
    ``min(1.0, max(0.1, min(10, |S|) / |S| + 1e-7))`` guarantees at least
    ``min(10, |S|)`` candidate features per split. The 1e-7 expansion guards
    the float32 truncation hazard in the C++ backends (``static_cast<int>(
    n_features * fraction)`` in single precision can land infinitesimally
    below an integer boundary, e.g. ``float32(10/42) * 42 == 9.9999999...``).
    The expansion sits *inside* the ``max(0.1, ...)`` bound, so ``|S| >= 100``
    keeps the floor at exactly 0.1 (large-set configs stay bit-identical).
    """
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    raw_floor = (
        min(float(_COLSAMPLE_FLOOR_TARGET_FEATURES), float(n_features))
        / float(n_features)
    )
    return float(min(1.0, max(_COLSAMPLE_FLOOR_MIN, raw_floor + _COLSAMPLE_FLOOR_EPS)))


def _raise_to_colsample_floor(value: float, n_features: int) -> float:
    """Raise-only application of the sampling floor (a floor is never a ceiling)."""
    return float(min(1.0, max(value, _colsample_floor(n_features))))


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
        # Measured peak RSS + commit charge of the last spawned full-history
        # fit (bytes), or None when unknown / not spawned. Commit is the
        # quantity that gates the promotion (the full-universe thrash was a
        # commit-limit crossing). Read by the promotion rehearsal.
        self.last_full_history_peak_bytes: int | None = None
        self.last_full_history_peak_commit_bytes: int | None = None

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

    def _cv_fold_parts(
        self,
        df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        target_col: str,
        splitter: PurgedEraSplitter,
        era_col: str = "era",
        checkpoint_dir: Path | None = None,
    ) -> tuple[list[object | None], list[pl.DataFrame]]:
        """Shared fold loop: fit or load each fold; (models, oof_parts).

        ``checkpoint_dir=None`` = the legacy fit-everything path (models has no
        None entries). With a checkpoint dir, existing fold parquets are loaded
        (models entry None — the OOF-only caller discards models) and new folds
        are fitted then atomically persisted. The checkpoint root carries a
        manifest.json with code+device identity; a mismatch raises.
        """
        # Local import: nmr._oof imports nmr.models at module top, so a
        # module-level import here would be circular.
        from nmr._oof import (
            _KNOWN_RESOLVED_DEVICES,
            _fitting_code_sha256,
            _write_frame_atomic,
        )

        folds = splitter.split(df.get_column(era_col).to_list())
        logger.info("[train_cross_validation] %s: %d folds", target_col, len(folds))
        models: list[object | None] = []
        oof_parts: list[pl.DataFrame] = []
        seen_val_eras: set[str] = set()

        manifest_path = checkpoint_dir / "manifest.json" if checkpoint_dir else None
        manifest_written = False
        if manifest_path is not None:
            if manifest_path.exists():
                stored = json.loads(manifest_path.read_text(encoding="utf-8"))
                if stored.get("code_sha256") != _fitting_code_sha256():
                    raise ValueError(
                        "OOF checkpoint code_sha256 mismatch: fitting code changed "
                        f"since the checkpoints were written ({manifest_path}). "
                        "Delete the oof_checkpoints directory to force a full refit."
                    )
                stored_device = stored.get("device")
                if self.resolved_device is not None:
                    # Same-process reuse: the device is known — exact compare.
                    if stored_device != str(self.resolved_device):
                        raise ValueError(
                            "OOF checkpoint device mismatch: checkpoints were "
                            f"fitted on device {stored_device!r}, current device "
                            f"is {str(self.resolved_device)!r}. Delete the "
                            "oof_checkpoints directory to force a full refit."
                        )
                elif stored_device not in _KNOWN_RESOLVED_DEVICES:
                    # Fresh orchestrator (device unknown pre-fit): the
                    # authoritative check runs at the first fitted fold; here
                    # a manifest recording anything but a real fit device is
                    # rejected loudly instead of accepted vacuously.
                    raise ValueError(
                        "OOF checkpoint device mismatch: manifest records "
                        f"unknown device {stored_device!r} ({manifest_path}). "
                        "Delete the oof_checkpoints directory to force a full refit."
                    )
            else:
                # PINNED DECISION (review): the manifest is written at the FIRST
                # fitted fold, never here — resolved_device is None until a fit
                # completes, and an early write would record "None", making the
                # device guard pass vacuously on resume.
                existing_parts = any(
                    manifest_path.parent.rglob("fold_*.parquet")
                ) if manifest_path.parent.exists() else False
                if existing_parts:
                    raise ValueError(
                        "OOF checkpoint tree has fold parts but no manifest.json "
                        f"({manifest_path}) — inconsistent state. Delete the "
                        "oof_checkpoints directory to force a full refit."
                    )

        for fold in folds:
            overlap = seen_val_eras & set(fold.val_eras)
            if overlap:
                raise ValueError(
                    f"Validation eras must be disjoint across folds, got {sorted(overlap)}"
                )
            part_path = (
                checkpoint_dir / target_col / f"fold_{fold.index + 1:02d}.parquet"
                if checkpoint_dir is not None else None
            )
            if part_path is not None and part_path.exists():
                try:
                    fold_predictions = pl.read_parquet(part_path)
                except Exception as exc:
                    raise ValueError(
                        f"corrupt OOF checkpoint {part_path}: {exc}"
                    ) from exc
                models.append(None)
                logger.info(
                    "[train_cross_validation] %s: fold %d/%d loaded from checkpoint %s",
                    target_col, fold.index + 1, len(folds), part_path,
                )
            else:
                logger.info(
                    "[train_cross_validation] %s: fold %d/%d train_eras=%d val_eras=%d",
                    target_col, fold.index + 1, len(folds),
                    len(fold.train_eras), len(fold.val_eras),
                )
                t0 = time.time()
                model, fold_predictions = self._fit_predict_fold(
                    df, fold=fold, feature_cols=feature_cols,
                    target_col=target_col, era_col=era_col,
                    purge_eras=splitter.purge_eras,
                )
                logger.info(
                    "[train_cross_validation] %s: fold %d/%d trained in %.1fs",
                    target_col, fold.index + 1, len(folds), time.time() - t0,
                )
                models.append(model)
                if part_path is not None:
                    # PINNED DECISION: manifest is written at the first fitted
                    # fold (device is only known post-fit) and BEFORE the first
                    # fold part, so a crash between the two writes is safe. On
                    # resume, the first fitted fold is where the authoritative
                    # device check runs — the device is only known post-fit.
                    if manifest_path is not None and not manifest_written:
                        resolved_device = str(self.resolved_device)
                        if manifest_path.exists():
                            stored = json.loads(
                                manifest_path.read_text(encoding="utf-8")
                            )
                            if stored.get("device") != resolved_device:
                                raise ValueError(
                                    "OOF checkpoint device mismatch: checkpoints "
                                    f"were fitted on device {stored.get('device')!r}, "
                                    f"current device is {resolved_device!r}. "
                                    "Delete the oof_checkpoints directory to "
                                    "force a full refit."
                                )
                        else:
                            manifest_path.parent.mkdir(parents=True, exist_ok=True)
                            atomic_write_bytes(
                                manifest_path,
                                json.dumps(
                                    {
                                        "code_sha256": _fitting_code_sha256(),
                                        "device": resolved_device,
                                    },
                                    sort_keys=True,
                                ).encode("utf-8"),
                            )
                        manifest_written = True
                    part_path.parent.mkdir(parents=True, exist_ok=True)
                    _write_frame_atomic(fold_predictions, part_path)
            oof_parts.append(fold_predictions)
            seen_val_eras.update(fold.val_eras)

        if not oof_parts:
            raise ValueError("No folds produced OOF predictions")
        return models, oof_parts

    def train_cross_validation(
        self,
        df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        target_col: str,
        splitter: PurgedEraSplitter,
        era_col: str = "era",
    ) -> CVResult:
        models, oof_parts = self._cv_fold_parts(
            df, feature_cols=feature_cols, target_col=target_col,
            splitter=splitter, era_col=era_col, checkpoint_dir=None,
        )
        if any(m is None for m in models):  # unreachable defensive guard
            raise ValueError("checkpoint-less CV produced a None model entry")
        oof = pl.concat(oof_parts, how="vertical")
        logger.info(
            "[train_cross_validation] %s: OOF complete rows=%d", target_col, oof.height
        )
        return CVResult(oof=oof, models=tuple(models))

    def train_oof_with_checkpoints(
        self,
        df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        target_col: str,
        splitter: PurgedEraSplitter,
        era_col: str = "era",
        checkpoint_dir: Path,
    ) -> pl.DataFrame:
        """Checkpoint-aware OOF training; returns OOF only (models discarded).

        See the checkpoint spec (2026-08-20-oof-checkpoint-resume) for the
        resume contract and code/device identity rules.
        """
        _, oof_parts = self._cv_fold_parts(
            df, feature_cols=feature_cols, target_col=target_col,
            splitter=splitter, era_col=era_col, checkpoint_dir=checkpoint_dir,
        )
        oof = pl.concat(oof_parts, how="vertical")
        logger.info(
            "[train_oof_with_checkpoints] %s: OOF complete rows=%d", target_col, oof.height
        )
        return oof

    def train_full_history(
        self,
        df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        target_col: str,
        era_col: str = "era",
        in_process: bool = False,
        data: DataConfig | None = None,
        include_validation: bool = False,
    ) -> object:
        """Fit a single CPU-only model on every era (deployment/validation artifact).

        CPU-only by design: determinism is per-device and the deployed model must
        reproduce identically on any hosted runtime (which may lack a GPU).

        ``include_validation`` (promotion writer only): when the fit spawns a
        subprocess, the child re-reads BOTH ``train.parquet`` and
        ``validation.parquet`` (concatenated) so a full version trained on
        train+validation sees the validation eras the research run never did.
        The in-process path fits whatever frame the caller passes, so callers
        pass the concat directly there.

        Memory discipline (2026-08-12, measured curve 2026-08-18): when the
        float32 feature matrix would exceed ``_FULL_HISTORY_SUBPROCESS_MIN_BYTES``
        the fit runs in a freshly spawned process with its own address space.
        The full-version (train+validation) memory profile is MEASURED, not
        guessed — see ``artifacts/reports/ram_curve.json`` and the AGENTS.md
        hazard: at medium (780 features, 6.85M rows) combined commit
        extrapolates to ~61-65 GiB and combined working set to 86-90% of
        physical RAM (marginal-to-infeasible on this box; the promotion RAM
        guard refuses with the measured numbers). Earlier conflicting
        full-universe figures (40-45 vs ~71 GiB) are retired in favor of that
        curve. Running the fit in the long-lived campaign process stacks it on
        the run's accumulated commit (CV + neutralization) and crosses the
        machine's commit limit — Windows then thrashes (measured: 1.1 iters/s
        vs ~50). The child re-reads the data itself and returns the pickled
        booster; results are bit-identical (same code path, same seed).
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
        if not in_process and self._should_spawn_full_history(train_df, feature_cols):
            if data is None:
                raise ValueError(
                    "subprocess full-history fit requires the DataConfig — pass "
                    "data=<ExperimentConfig.data> (the orchestrator only holds "
                    "the ModelConfig)"
                )
            return self._fit_full_history_subprocess(
                train_df, feature_cols=feature_cols, target_col=target_col,
                era_col=era_col, data=data,
                include_validation=include_validation,
            )
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
        candidate_params = self._device_candidate_params(
            use_gpu=use_gpu, n_features=int(features.shape[1])
        )
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

    def _device_candidate_params(
        self, *, use_gpu: bool, n_features: int
    ) -> list[dict[str, Any]]:
        if not use_gpu:
            return [self._resolved_params(use_gpu=False, n_features=n_features)]
        if self._config.backend == "catboost":
            # CPU-only by design: catboost rejects `rsm` on GPU (non-pairwise
            # modes) and every canonical preset ships colsample_bytree -> rsm,
            # so a GPU candidate can never fit. Never attempt one.
            return [self._resolved_params(use_gpu=False, n_features=n_features)]
        gpu_params = self._resolved_params(use_gpu=True, n_features=n_features)
        if self._config.device == "gpu":
            # Forced GPU: a failing GPU fit raises — no silent CPU fallback.
            return [gpu_params]
        cpu_params = self._resolved_params(use_gpu=False, n_features=n_features)
        if gpu_params == cpu_params:
            return [cpu_params]
        return [gpu_params, cpu_params]

    def _resolved_params(
        self, *, use_gpu: bool, n_features: int
    ) -> dict[str, Any]:
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
            # Floor every present member of the LightGBM sampling-alias group
            # ({colsample_bytree, feature_fraction, sub_feature} — one
            # _ConfigAliases group in the installed wrapper, and unknown
            # kwargs flow through **kwargs into the native engine). Flooring
            # all present members is precedence-proof.
            for alias in _LGBM_COLSAMPLE_ALIASES:
                if alias in params:
                    params[alias] = _raise_to_colsample_floor(
                        float(params[alias]), n_features
                    )
            return params

        if self._config.backend == "catboost":
            base = resolve_model_params(self._config.preset, self._config.params)
            translated = _translate_catboost(base, seed=self._seed, use_gpu=use_gpu)
            translated["rsm"] = _raise_to_colsample_floor(
                float(translated["rsm"]), n_features
            )
            return translated

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
        elif "max_leaves" in params:
            # A config specifying max_leaves directly (without num_leaves) means
            # leaf-wise growth: under XGBoost's default depthwise policy
            # max_leaves is silently inert (audit SEV-3: the tier-2 XGB cell's
            # max_leaves: 15 was a no-op). An explicit grow_policy wins.
            params.setdefault("grow_policy", "lossguide")
        if min_data_in_leaf is not None:
            params.setdefault("min_child_weight", float(min_data_in_leaf))
        params["colsample_bytree"] = _raise_to_colsample_floor(
            float(params["colsample_bytree"]), n_features
        )
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


    def _should_spawn_full_history(
        self, train_df: pl.DataFrame, feature_cols: Sequence[str]
    ) -> bool:
        """True when the float32 feature matrix would exceed the spawn threshold.

        Env override ``NMR_FULL_HISTORY_SPAWN_MIN_BYTES`` lowers the threshold
        so the D7 truncated rehearsal (and tests) can force the fresh-process
        path at small scale — the least-exercised code in this module.
        """
        threshold = int(
            os.environ.get(
                "NMR_FULL_HISTORY_SPAWN_MIN_BYTES",
                str(_FULL_HISTORY_SUBPROCESS_MIN_BYTES),
            )
        )
        return train_df.height * len(feature_cols) * 4 > threshold

    def _fit_full_history_subprocess(
        self,
        train_df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        target_col: str,
        era_col: str,
        data: DataConfig,
        include_validation: bool = False,
    ) -> object:
        """Fit the full-history model in a fresh process (bounded commit).

        ``include_validation`` (promotion writer): the child re-reads
        train+validation so the full version sees the validation eras.
        """
        if data is None:
            raise ValueError(
                "subprocess full-history fit requires the DataConfig — pass "
                "data=<ExperimentConfig.data>"
            )
        import multiprocessing as mp

        import cloudpickle

        ctx = mp.get_context("spawn")
        out_q = ctx.Queue()
        spec = {
            "data": {
                "version": data.version,
                "feature_set": data.feature_set,
                "feature_subset": data.feature_subset,
                "targets": (target_col,),
                "data_dir": str(data.data_dir),
                "supplemental_feature_sets": (
                    str(data.supplemental_feature_sets)
                    if data.supplemental_feature_sets is not None
                    else None
                ),
            },
            "feature_cols": list(feature_cols),
            "target_col": target_col,
            "era_col": era_col,
            "backend": self._config.backend,
            "preset": self._config.preset,
            "params": dict(self._config.params),
            "seed": self._seed,
            "include_validation": include_validation,
        }
        logger.info(
            "[train_full_history] spawning fresh-process fit "
            "(%d rows x %d features > %d GiB float32)",
            train_df.height, len(feature_cols),
            _FULL_HISTORY_SUBPROCESS_MIN_BYTES // 2**30,
        )
        proc = ctx.Process(target=_full_history_fit_worker, args=(spec, out_q))
        proc.start()
        try:
            status, payload = _receive_subprocess_result(out_q, proc)
        finally:
            proc.join()
        if proc.exitcode != 0 or status != "ok":
            raise RuntimeError(
                f"full-history subprocess fit failed (exit={proc.exitcode}): {payload}"
            )
        self.resolved_device = "cpu"  # train_full_history is CPU-only by design
        model_bytes, working_set, commit = payload
        self.last_full_history_peak_bytes = working_set
        self.last_full_history_peak_commit_bytes = commit
        return cloudpickle.loads(model_bytes)


def construct_tree_model(
    backend: str,
    params: Mapping[str, Any],
    *,
    seed: int,
    n_features: int,
    device: str = "cpu",
    extra_params: Mapping[str, Any] | None = None,
) -> object:
    """Build a deterministic, CPU-default tree estimator from raw params.

    Applies the same backend param mapping, colsample flooring, and
    determinism flags as ``ModelOrchestrator``. Used by the benchmark
    hierarchy so benchmark cells never hand-duplicate param resolution.
    ``extra_params`` are merged AFTER resolution so constructor-only kwargs
    (e.g. XGBoost ``early_stopping_rounds``) bypass param validation.
    """
    config = ModelConfig(
        backend=backend,
        preset="fast",
        params=dict(params),
        device=device,
    )
    orchestrator = ModelOrchestrator(config, seed=seed)
    resolved = orchestrator._resolved_params(
        use_gpu=device != "cpu", n_features=int(n_features)
    )
    if extra_params:
        resolved.update(dict(extra_params))
    return orchestrator._build_model(resolved)


def _receive_subprocess_result(out_q, proc) -> tuple:
    """Block until the spawned child reports ``(status, payload)`` — or dies.

    Polls ``proc.is_alive()`` while waiting on the queue, so a child that
    crashes before enqueuing its result raises promptly instead of leaving
    the parent blocked forever on an unanswered ``get()`` (a documented OOM
    failure mode in long campaign jobs). A still-alive child is waited on
    indefinitely: a legitimate full-history fit runs for hours, so no overall
    timeout is applied by design.
    """
    from queue import Empty

    result = None
    while proc.is_alive():
        try:
            result = out_q.get(timeout=_SUBPROCESS_POLL_INTERVAL_SECONDS)
            break
        except Empty:
            continue
    if result is None:
        # Child exited; a result may still be in flight through the queue's
        # feeder thread (child put, then died between parent polls).
        try:
            result = out_q.get(timeout=_SUBPROCESS_DRAIN_GRACE_SECONDS)
        except Empty:
            pass
    if result is None:
        raise RuntimeError(
            "full-history subprocess died without reporting a result "
            f"(exitcode={proc.exitcode})"
        )
    return result


def _peak_memory_counters() -> tuple[int | None, int | None]:
    """Peak (working set, commit charge) of the calling process, stdlib-only.

    Windows: ``PROCESS_MEMORY_COUNTERS`` gives both ``PeakWorkingSetSize``
    (RSS) and ``PeakPagefileUsage`` (peak commit charge). COMMIT is the
    quantity that produced the documented full-universe thrash (~71 GiB
    commit) — the RAM guard gates on commit, never on working set. Unix has no
    commit counter; the resource fallback reports ``ru_maxrss`` as working set
    only (commit is None). Returns ``(None, None)`` when unavailable.
    """
    try:
        import ctypes  # Windows

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        # Typed signatures are REQUIRED on 64-bit Windows: GetCurrentProcess
        # returns a HANDLE, and without restype=c_void_p ctypes truncates it to
        # c_int, making the subsequent memory-info call fail (measured as a
        # 0-byte peak — the exact failure the rehearsal is meant to catch).
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_size_t,
        ]
        kernel32.K32GetProcessMemoryInfo.restype = ctypes.c_int
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        process = kernel32.GetCurrentProcess()
        if kernel32.K32GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            return (
                int(counters.PeakWorkingSetSize),
                int(counters.PeakPagefileUsage),
            )
        return None, None
    except Exception:  # pragma: no cover - platform-dependent, best-effort
        try:
            import resource  # Unix: ru_maxrss (KiB on Linux) — working set only

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, None
        except Exception:
            return None, None


def _peak_rss_bytes() -> int | None:
    """Backward-compatible wrapper: peak working set only."""
    working_set, _ = _peak_memory_counters()
    return working_set


def _machine_memory_limits() -> tuple[int | None, int | None]:
    """Physical RAM and commit limit of THIS machine, stdlib-only.

    Returns ``(physical_total_bytes, commit_limit_bytes)``. Windows:
    ``GetPerformanceInfo`` (fields are in pages; ``PageSize`` converts).
    Unix: physical via ``sysconf``; commit limit N/A (None). The two metrics
    guard different failure modes: commit vs commit limit (hard OOM), working
    set vs physical RAM (thrash — the documented 1.1 iters/s collapse).
    """
    try:
        import ctypes  # Windows: GetPerformanceInfo (psapi.dll / K32GetPerformanceInfo)

        class _PerformanceInformation(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("CommitTotal", ctypes.c_size_t),
                ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t),
                ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t),
                ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t),
                ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t),
                ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_ulong),
                ("ProcessCount", ctypes.c_ulong),
                ("ThreadCount", ctypes.c_ulong),
            ]

        info = _PerformanceInformation()
        info.cb = ctypes.sizeof(_PerformanceInformation)
        fn = None
        for dll_name, fn_name in (
            ("psapi", "GetPerformanceInfo"),
            ("kernel32", "K32GetPerformanceInfo"),
        ):
            try:
                dll = getattr(ctypes.windll, dll_name)
                candidate = getattr(dll, fn_name)
                candidate.argtypes = [ctypes.POINTER(_PerformanceInformation), ctypes.c_ulong]
                candidate.restype = ctypes.c_int
                fn = candidate
                break
            except (AttributeError, OSError):
                continue
        if fn is not None and fn(ctypes.byref(info), info.cb):
            page = int(info.PageSize)
            return int(info.PhysicalTotal) * page, int(info.CommitLimit) * page
        return None, None
    except Exception:  # pragma: no cover - platform-dependent, best-effort
        try:
            import os as _os

            pages = _os.sysconf("SC_PHYS_PAGES")
            page_size = _os.sysconf("SC_PAGE_SIZE")
            return int(pages) * int(page_size), None
        except Exception:
            return None, None


def _full_history_fit_worker(spec: dict, out_q) -> None:
    """Spawned-process fitter for memory-bounded full-history training.

    Runs in a fresh address space: re-loads the train split via
    ``IngestionAgent`` (plus the validation split when ``include_validation``
    is set — the promotion writer's full version trains on train+validation),
    fits the same CPU-only model (identical code path and seed as the
    in-process variant), and returns the cloudpickled booster plus the
    measured peak RSS (for the promotion rehearsal's RAM extrapolation).
    """
    try:
        import cloudpickle
        import polars as pl

        from nmr.config import DataConfig, ModelConfig, set_global_seeds
        from nmr.data import IngestionAgent

        set_global_seeds(spec["seed"])
        data = DataConfig(**spec["data"])
        agent = IngestionAgent(data)
        columns = ["era", "id", *spec["feature_cols"], spec["target_col"]]
        df = agent.load("train", columns=columns)
        if spec.get("include_validation"):
            df = pl.concat([df, agent.load("validation", columns=columns)])
        orch = ModelOrchestrator(
            ModelConfig(
                backend=spec["backend"],
                preset=spec["preset"],
                params=spec["params"],
                device="cpu",
            ),
            seed=spec["seed"],
        )
        model = orch.train_full_history(
            df,
            feature_cols=spec["feature_cols"],
            target_col=spec["target_col"],
            era_col=spec["era_col"],
            in_process=True,
        )
        working_set, commit = _peak_memory_counters()
        out_q.put(("ok", (cloudpickle.dumps(model), working_set, commit)))
    except Exception as exc:  # surface the child's failure loudly in the parent
        out_q.put(("error", repr(exc)))


