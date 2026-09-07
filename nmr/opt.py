"""Bayesian hyperparameter optimization via Optuna (user-granted dependency).

``bayesian_sweep`` is the single Optuna-integration point. Space definitions are
declarative dicts (ARCHITECTURE.md §S); the objective is harness-internal
(``research._held_out_metric``); sweeps are seeded, single-threaded, and
deterministic per environment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import optuna
import polars as pl
import psutil
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge

from nmr._atomicio import atomic_write_text
from nmr.config import ExperimentConfig
from nmr.hardware import discover_hardware
from nmr.models import resolve_model_params
from nmr.payout import CLASSIC_ATOMIC_ENDER60_R1343_V1
from nmr.research import (
    SweepResult,
    _held_out_metric_full,
    _override_config,
    metric_direction,
)
from nmr.scorecard import evaluate_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger("nmr.opt")

__all__ = [
    "HGBHPOConfig",
    "HGBHPOResult",
    "VotingHPOConfig",
    "bayesian_hgb_sweep",
    "bayesian_voting_sweep",
    "bayesian_sweep",
]

_VALID_METRICS = ("mean", "std", "sharpe", "max_drawdown", "corr_sharpe_ac")
_PAYOUT_TARGET = "target_ender_60"
_JSON_PRIMITIVES = (str, int, float, bool)


@dataclass(frozen=True)
class _SpaceParam:
    kind: Literal["float", "int", "categorical"]
    name: str
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    step: int | None = None
    choices: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class HGBHPOConfig:
    """Configuration for a deterministic HGB Bayesian search."""

    data_dir: Path
    output: Path
    n_trials: int = 24
    n_startup_trials: int = 8
    max_train_rows: int | None = 250_000
    max_validation_rows: int | None = 300_000
    purge_eras: int = 8
    seed: int = 42
    target_col: str = "target"
    score_window: str = "meta"
    tuning_fraction: float = 0.75
    compute_fnc: bool = True
    full_confirmation: bool = True
    resume: bool = True
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.n_trials, bool) or self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if isinstance(self.n_startup_trials, bool) or self.n_startup_trials < 1:
            raise ValueError("n_startup_trials must be >= 1")
        if self.n_jobs != 1:
            raise ValueError(
                f"n_jobs must be 1 (parallel trials break TPE determinism); got {self.n_jobs}"
            )
        if self.score_window not in ("meta", "validation"):
            raise ValueError("score_window must be 'meta' or 'validation'")
        if not isinstance(self.target_col, str) or not self.target_col:
            raise ValueError("target_col must be a non-empty string")
        if not 0.5 <= float(self.tuning_fraction) < 1.0:
            raise ValueError("tuning_fraction must be in [0.5, 1.0)")
        if isinstance(self.purge_eras, bool) or self.purge_eras < 0:
            raise ValueError("purge_eras must be a non-negative int")
        for name in ("max_train_rows", "max_validation_rows"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive int or None")


@dataclass(frozen=True)
class HGBHPOResult:
    """Trial table, selected parameters, and optional uncapped confirmation."""

    trials: pl.DataFrame
    best_params: dict[str, Any]
    best_value: float
    confirmation: dict[str, Any]


@dataclass(frozen=True)
class VotingHPOConfig(HGBHPOConfig):
    """Configuration for Bayesian tuning of the Ridge+HGB vote."""


def _parse_space(space: dict[str, dict[str, Any]]) -> list[_SpaceParam]:
    if not space:
        raise ValueError("search space is empty; must contain at least one parameter")
    parsed: list[_SpaceParam] = []
    for name, spec in space.items():
        if not isinstance(spec, dict):
            raise ValueError(f"parameter {name!r}: spec must be a dict")
        unknown = set(spec) - {"kind", "low", "high", "log", "step", "choices"}
        if unknown:
            raise ValueError(f"parameter {name!r}: unknown keys {sorted(unknown)}")
        kind = spec.get("kind")
        if kind not in ("float", "int", "categorical"):
            raise ValueError(f"parameter {name!r}: kind must be float/int/categorical")
        if kind in ("float", "int"):
            low, high = spec.get("low"), spec.get("high")
            if low is None or high is None or low > high:
                raise ValueError(f"parameter {name!r}: low/high bounds invalid")
            raw_log = spec.get("log", False)
            if not isinstance(raw_log, bool):
                raise ValueError(f"parameter {name!r}: 'log' must be a boolean")
            log = raw_log
            if log and low <= 0:
                raise ValueError(
                    f"parameter {name!r}: 'low' must be > 0 when log=True, got {low}"
                )
            step = spec.get("step")
            if kind == "float" and step is not None:
                raise ValueError(
                    f"parameter {name!r}: 'step' is only valid for int params"
                )
            if step is not None and (not isinstance(step, int) or step < 1):
                raise ValueError(f"parameter {name!r}: step must be a positive int")
            if log and step is not None:
                raise ValueError(
                    f"parameter {name!r}: log=True and step are mutually exclusive"
                )
            parsed.append(
                _SpaceParam(
                    kind=kind, name=name, low=low, high=high, log=log, step=step
                )
            )
        else:
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"parameter {name!r}: choices must be a non-empty list"
                )
            if not all(isinstance(c, _JSON_PRIMITIVES) for c in choices):
                raise ValueError(
                    f"parameter {name!r}: categorical choices must be str/int/float/bool"
                )
            parsed.append(
                _SpaceParam(kind="categorical", name=name, choices=list(choices))
            )
    return parsed


def _suggest(trial: optuna.Trial, param: _SpaceParam) -> Any:
    if param.kind == "float":
        return trial.suggest_float(param.name, param.low, param.high, log=param.log)
    if param.kind == "int":
        kwargs: dict[str, Any] = {"log": param.log}
        if param.step is not None:
            kwargs["step"] = param.step
        return trial.suggest_int(param.name, param.low, param.high, **kwargs)
    return trial.suggest_categorical(param.name, list(param.choices))


def _hgb_search_space() -> dict[str, dict[str, Any]]:
    """Return the bounded HGB search space used by ``bayesian_hgb_sweep``."""
    return {
        "learning_rate": {
            "kind": "float",
            "low": 0.01,
            "high": 0.3,
            "log": True,
        },
        "max_iter": {
            "kind": "int",
            "low": 40,
            "high": 240,
            "step": 20,
        },
        "max_leaf_nodes": {
            "kind": "int",
            "low": 7,
            "high": 127,
            "log": True,
        },
        "min_samples_leaf": {
            "kind": "int",
            "low": 10,
            "high": 1000,
            "log": True,
        },
        "l2_regularization": {
            "kind": "float",
            "low": 1.0e-8,
            "high": 100.0,
            "log": True,
        },
        "max_bins": {"kind": "categorical", "choices": [64, 128, 192, 255]},
        "max_features": {
            "kind": "float",
            "low": 0.5,
            "high": 1.0,
        },
        "max_depth": {"kind": "categorical", "choices": [0, 3, 5, 7, 10]},
    }


def _hgb_estimator(
    params: dict[str, Any], *, seed: int
) -> HistGradientBoostingRegressor:
    depth = int(params["max_depth"])
    return HistGradientBoostingRegressor(
        learning_rate=float(params["learning_rate"]),
        max_iter=int(params["max_iter"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        max_bins=int(params["max_bins"]),
        max_features=float(params["max_features"]),
        max_depth=None if depth == 0 else depth,
        early_stopping=False,
        random_state=seed,
    )


def _voting_search_space() -> dict[str, dict[str, Any]]:
    space = dict(_hgb_search_space())
    space["ridge_weight"] = {"kind": "float", "low": 0.0, "high": 1.0}
    return space


def _voting_estimator(params: dict[str, Any], *, seed: int) -> VotingRegressor:
    hgb_params = {name: params[name] for name in _hgb_search_space()}
    ridge = Ridge(alpha=1.0, solver="lsqr", tol=1.0e-3, random_state=seed)
    return VotingRegressor(
        estimators=[
            ("ridge", ridge),
            ("hist", _hgb_estimator(hgb_params, seed=seed)),
        ],
        weights=[float(params["ridge_weight"]), 1.0 - float(params["ridge_weight"])],
        n_jobs=1,
    )


def _rss_gib() -> float:
    return round(psutil.Process().memory_info().rss / (1024**3), 4)


def _hgb_run_identity(config: HGBHPOConfig) -> str:
    payload = {
        "data_dir": str(config.data_dir.resolve()),
        "n_trials": config.n_trials,
        "n_startup_trials": config.n_startup_trials,
        "max_train_rows": config.max_train_rows,
        "max_validation_rows": config.max_validation_rows,
        "purge_eras": config.purge_eras,
        "seed": config.seed,
        "target_col": config.target_col,
        "score_window": config.score_window,
        "tuning_fraction": config.tuning_fraction,
        "compute_fnc": config.compute_fnc,
        "full_confirmation": config.full_confirmation,
        "search_space": _hgb_search_space(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _voting_run_identity(config: VotingHPOConfig) -> str:
    payload = {
        "model": "VotingRegressor",
        "base_identity": _hgb_run_identity(config),
        "search_space": _voting_search_space(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hgb_json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


def _hgb_checkpoint_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".work") / "trials.json"


def _load_hgb_trials(path: Path, identity: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("run_identity") != identity:
        raise ValueError(
            f"HGB HPO checkpoint identity mismatch: {path}; delete it to restart"
        )
    rows = payload.get("trials")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid HGB HPO trial checkpoint: {path}")
    return [dict(row) for row in rows]


def _write_hgb_trials(path: Path, identity: str, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        json.dumps(
            {"run_identity": identity, "trials": rows},
            sort_keys=True,
            indent=2,
            default=_hgb_json_default,
        ),
    )


def _hgb_trials_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    frame = pl.DataFrame(rows).with_columns(
        pl.col("metric_value").cast(pl.Float64),
        pl.col("fit_seconds").cast(pl.Float64),
        pl.col("predict_seconds").cast(pl.Float64),
        pl.col("score_seconds").cast(pl.Float64),
        pl.col("total_seconds").cast(pl.Float64),
        pl.col("peak_rss_gib").cast(pl.Float64),
    )
    completed = frame.filter(pl.col("status") == "completed")
    if completed.height:
        best = float(completed.get_column("metric_value").max())
        frame = frame.with_columns(
            (
                (pl.col("status") == "completed") & (pl.col("metric_value") == best)
            ).alias("best_trial")
        )
    else:
        frame = frame.with_columns(pl.lit(False).alias("best_trial"))
    return frame.sort("trial_id")


def _hgb_data_slice(data: Any, mask: np.ndarray) -> Any:
    return replace(
        data,
        X_validation=data.X_validation[mask],
        y_validation=data.y_validation[mask],
        eras_validation=tuple(
            str(value) for value in np.asarray(data.eras_validation, dtype=object)[mask]
        ),
    )


def _hgb_validation_masks(
    data: Any, tuning_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    ordered_eras = sorted(set(data.eras_validation), key=int)
    if len(ordered_eras) < 2:
        raise ValueError("HGB HPO requires at least two validation eras")
    tuning_count = max(
        1, min(len(ordered_eras) - 1, int(len(ordered_eras) * tuning_fraction))
    )
    tuning_eras = set(ordered_eras[:tuning_count])
    era_values = np.asarray(data.eras_validation, dtype=object)
    tuning_mask = np.isin(era_values, list(tuning_eras))
    return tuning_mask, ~tuning_mask


def _add_hgb_study_trials(
    study: optuna.Study,
    rows: list[dict[str, Any]],
    distributions: dict[str, optuna.distributions.BaseDistribution],
) -> None:
    for row in rows:
        params = json.loads(str(row["params_json"]))
        state = (
            optuna.trial.TrialState.COMPLETE
            if row.get("status") == "completed" and row.get("metric_value") is not None
            else optuna.trial.TrialState.PRUNED
        )
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=(
                    float(row["metric_value"])
                    if state == optuna.trial.TrialState.COMPLETE
                    else None
                ),
                state=state,
            )
        )


def bayesian_sweep(
    base_config: ExperimentConfig,
    space: dict[str, dict[str, Any]],
    *,
    n_trials: int,
    seed: int,
    metric: str = "sharpe",
    n_startup_trials: int = 10,
    enqueue_base_config: bool = True,
    n_jobs: int = 1,
) -> SweepResult:
    """Bayesian hyperparameter sweep over ``space`` around ``base_config``.

    Seeded TPE sampler (``TPESampler(seed=..., n_startup_trials=...)`` —
    deterministic-by-default since Optuna 4.x, which removed the 3.x
    ``deterministic`` flag; verified on 4.9.0), single-threaded (``n_jobs`` must
    be 1 — parallel trials break TPE determinism), in-memory storage.
    Trial 0 evaluates the resolved baseline (preset defaults + ``model.params``,
    intersected with the space) when ``enqueue_base_config`` is true.
    Returns the standard :class:`SweepResult` (ARCHITECTURE.md §S).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_startup_trials < 1:
        raise ValueError("n_startup_trials must be >= 1")
    if n_jobs != 1:
        raise ValueError(
            f"n_jobs must be 1 (parallel trials break TPE determinism); got {n_jobs}"
        )
    if metric not in _VALID_METRICS:
        raise ValueError(f"metric={metric!r} not in {sorted(_VALID_METRICS)}")
    direction = metric_direction(metric)

    parsed = _parse_space(space)
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=n_startup_trials
        ),
        storage=optuna.storages.InMemoryStorage(),
    )
    if enqueue_base_config:
        resolved = resolve_model_params(
            base_config.model.preset, base_config.model.params
        )
        anchor = {p.name: resolved[p.name] for p in parsed if p.name in resolved}
        if anchor:
            study.enqueue_trial(anchor)

    moments_by_trial: dict[int, object] = {}

    def objective(trial: optuna.Trial) -> float:
        params = {p.name: _suggest(trial, p) for p in parsed}
        cfg = _override_config(base_config, params)
        try:
            value, moments = _held_out_metric_full(cfg, metric_name=metric)
            moments_by_trial[trial.number] = moments
        except Exception as exc:
            logger.error("[bayesian_sweep] trial %s failed: %s", trial.number, exc)
            raise optuna.exceptions.TrialPruned(f"trial failed: {exc}") from exc
        finally:
            gc.collect()
        return float(value)

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    rows = []
    for t in study.trials:
        value = t.value if t.state == optuna.trial.TrialState.COMPLETE else None
        moments = moments_by_trial.get(t.number)
        rows.append(
            {
                "trial_id": t.number,
                "params_json": json.dumps(t.params, sort_keys=True),
                "metric_value": value,
                "metric": metric,
                "ic_sharpe": getattr(moments, "ic_sharpe", None),
                "ic_skew": getattr(moments, "ic_skew", None),
                "ic_kurt": getattr(moments, "ic_kurt", None),
                "ic_n_eras": getattr(moments, "ic_n_eras", None),
                "ic_std": getattr(moments, "ic_std", None),
            }
        )
    # Explicit Float64: when every trial fails, `rows` has only None metric
    # values and polars infers the column as Null dtype — diverging from
    # HyperparameterSweep.run's always-Float64 SweepResult contract.
    trial_df = (
        pl.DataFrame(rows)
        .with_columns(pl.col("metric_value").cast(pl.Float64))
        .sort(
            ["metric_value", "trial_id"],
            descending=[direction == "maximize", False],
            nulls_last=True,
        )
    )
    best = study.best_trial if len(study.best_trials) > 0 else None
    return SweepResult(
        trials=trial_df,
        best_params=best.params if best is not None else {},
        best_value=float(best.value) if best is not None else float("nan"),
        is_capital=False,
        proxy_metric=metric,
        proxy_split="held_out_80_20",
        proxy_target=base_config.evaluation.main_target,
        selection_bias=False,
    )


def sweep_dsr(trials: pl.DataFrame) -> pl.DataFrame:
    """Post-hoc sweep-aware DSR over COMPLETE trials with held-out moments.

    Requires the moment columns emitted by ``HyperparameterSweep`` /
    ``bayesian_sweep`` (``ic_sharpe``, ``ic_skew``, ``ic_kurt``, ``ic_n_eras``,
    ``ic_std``). Valid trials: finite moments, ``ic_std > 0``, ``ic_n_eras >= 4``.
    Returns ``trials`` with ``dsr_sweep_aware``, ``dsr_pass_sweep`` (>= 0.95),
    ``dsr_reason``, ``dsr_n_trials``, ``dsr_trials_sr_var`` appended. Guard A:
    zero cross-trial Sharpe variance (or fewer than 2 valid trials) yields
    None DSR with the fleet reason — never a crash, never an analytic fallback.
    """
    from nmr.inference import deflated_sharpe_fleet

    required = {
        "trial_id",
        "ic_sharpe",
        "ic_skew",
        "ic_kurt",
        "ic_n_eras",
        "ic_std",
    }
    missing = required - set(trials.columns)
    if missing:
        raise ValueError(f"trials missing required columns: {sorted(missing)}")

    valid_mask = (
        trials["ic_sharpe"].is_not_null()
        & trials["ic_skew"].is_not_null()
        & trials["ic_kurt"].is_not_null()
        & trials["ic_std"].is_not_null()
        & (trials["ic_std"] > 0.0)
        & (trials["ic_n_eras"] >= 4)
    )
    for col in ("ic_sharpe", "ic_skew", "ic_kurt", "ic_std"):
        valid_mask &= trials[col].is_finite()

    idxs = np.flatnonzero(valid_mask.to_numpy())
    dsr_arr = np.full(trials.height, np.nan)
    reason_arr = np.full(trials.height, None, dtype=object)
    n_trials = int(idxs.size)
    trials_var: float | None = None
    if n_trials:
        sharpe_vec = trials["ic_sharpe"].to_numpy()[idxs].astype(float)
        if n_trials >= 2:
            trials_var = float(np.var(sharpe_vec, ddof=1))
        dsr, reasons = deflated_sharpe_fleet(
            sharpe_vec,
            skew=trials["ic_skew"].to_numpy()[idxs].astype(float),
            kurt=trials["ic_kurt"].to_numpy()[idxs].astype(float),
            n_obs=trials["ic_n_eras"].to_numpy()[idxs].astype(float),
        )
        dsr_arr[idxs] = dsr
        reason_arr[idxs] = reasons

    return trials.with_columns(
        [
            pl.Series("dsr_sweep_aware", dsr_arr)
            .fill_nan(None)
            .alias("dsr_sweep_aware"),
            pl.Series("dsr_pass_sweep", (dsr_arr >= 0.95).astype(bool)),
            pl.Series("dsr_reason", reason_arr).alias("dsr_reason"),
            pl.Series(
                "dsr_n_trials",
                [n_trials if n_trials >= 2 else None] * trials.height,
                dtype=pl.Int64,
            ).alias("dsr_n_trials"),
            pl.Series(
                "dsr_trials_sr_var",
                [trials_var] * trials.height,
                dtype=pl.Float64,
            ).alias("dsr_trials_sr_var"),
        ]
    )


def _hgb_confirmation_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_confirmation{output.suffix}")


def _hgb_base_params() -> dict[str, Any]:
    return {
        "learning_rate": 0.1,
        "max_iter": 60,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0e-8,
        "max_bins": 255,
        "max_features": 1.0,
        "max_depth": 0,
    }


def _voting_base_params() -> dict[str, Any]:
    params = _hgb_base_params()
    params["ridge_weight"] = 0.5
    return params


def _atomic_payout_metrics(
    predictions: pl.DataFrame,
    *,
    meta_model: pl.DataFrame,
    targets: pl.DataFrame,
    features: pl.DataFrame,
    benchmarks: pl.DataFrame | None,
    seed: int,
    model_id: str,
) -> dict[str, Any]:
    """Return the canonical full scorecard for one validation prediction set."""
    if _PAYOUT_TARGET not in targets.columns:
        raise ValueError(f"scorecard targets must include {_PAYOUT_TARGET!r}")
    benchmark_col = (
        "v53_lgbm_ender60"
        if benchmarks is not None and "v53_lgbm_ender60" in benchmarks.columns
        else None
    )
    scorecard = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=benchmarks,
        features=features,
        targets=targets,
        n_trials=1,
        seed=seed,
        payout_policy=CLASSIC_ATOMIC_ENDER60_R1343_V1,
        horizon="60D",
        main_target=_PAYOUT_TARGET,
        benchmark_col=benchmark_col,
        backend="custom",
        model_id=model_id,
    )
    return scorecard.to_frame().to_dicts()[0]


def _confirmation_payout_metrics(
    config: HGBHPOConfig,
    data: Any,
    predictions: np.ndarray,
    *,
    model_id: str,
) -> dict[str, Any]:
    if (
        not data.validation_ids
        or data.validation_scorecard_features is None
        or not data.validation_scorecard_targets
    ):
        raise ValueError("confirmation data is missing full scorecard inputs")
    meta_path = config.data_dir / "meta_model.parquet"
    if not meta_path.is_file():
        raise FileNotFoundError(f"full scorecard evaluation requires {meta_path}")
    predictions_frame = pl.DataFrame(
        {
            "era": data.eras_validation,
            "id": data.validation_ids,
            "prediction": predictions,
        }
    )
    target_columns: dict[str, Any] = {
        "era": data.eras_validation,
        "id": data.validation_ids,
    }
    target_columns.update(data.validation_scorecard_targets)
    targets_frame = pl.DataFrame(target_columns)
    feature_columns: dict[str, Any] = {
        "era": data.eras_validation,
        "id": data.validation_ids,
    }
    feature_columns.update(
        {
            name: data.validation_scorecard_features[:, index]
            for index, name in enumerate(data.validation_feature_names)
        }
    )
    features_frame = pl.DataFrame(feature_columns)
    meta_model = pl.read_parquet(meta_path).select(["era", "id", "numerai_meta_model"])
    benchmark_path = config.data_dir / "validation_benchmark_models.parquet"
    benchmarks = pl.read_parquet(benchmark_path) if benchmark_path.is_file() else None
    return _atomic_payout_metrics(
        predictions_frame,
        meta_model=meta_model,
        targets=targets_frame,
        features=features_frame,
        benchmarks=benchmarks,
        seed=config.seed,
        model_id=model_id,
    )


def _confirm_hgb(
    config: HGBHPOConfig,
    best_params: dict[str, Any],
) -> dict[str, Any]:
    from nmr.sklearn_breadth import _catalog, _prepare_data, _score_predictions

    method = next(
        spec for spec in _catalog() if spec.name == "HistGradientBoostingRegressor"
    )
    started = time.perf_counter()
    data = _prepare_data(
        data_dir=config.data_dir,
        method=method,
        max_train_rows=None,
        max_validation_rows=None,
        purge_eras=config.purge_eras,
        score_window=config.score_window,
        target_col=config.target_col,
        payout_target_col=_PAYOUT_TARGET,
        scorecard_target_cols=("target_ender_20", _PAYOUT_TARGET),
    )
    model = _hgb_estimator(best_params, seed=config.seed)
    fit_started = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    fit_seconds = round(time.perf_counter() - fit_started, 6)
    predict_started = time.perf_counter()
    predictions = np.asarray(model.predict(data.X_validation), dtype=float)
    predict_seconds = round(time.perf_counter() - predict_started, 6)
    if not np.all(np.isfinite(predictions)):
        raise ValueError("best HGB estimator returned non-finite predictions")

    tuning_mask, holdout_mask = _hgb_validation_masks(data, config.tuning_fraction)
    score_started = time.perf_counter()
    full_score = _score_predictions(
        data,
        predictions,
        compute_fnc=config.compute_fnc,
    )
    tuning_score = _score_predictions(
        _hgb_data_slice(data, tuning_mask),
        predictions[tuning_mask],
        compute_fnc=False,
    )
    holdout_score = _score_predictions(
        _hgb_data_slice(data, holdout_mask),
        predictions[holdout_mask],
        compute_fnc=False,
    )
    payout_metrics = _confirmation_payout_metrics(
        config,
        data,
        predictions,
        model_id="sklearn::hpo::hgb",
    )
    score_seconds = round(time.perf_counter() - score_started, 6)
    result: dict[str, Any] = {
        "status": "completed",
        "target_col": config.target_col,
        "best_params_json": json.dumps(best_params, sort_keys=True),
        "load_seconds": data.load_seconds,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "score_seconds": score_seconds,
        "total_seconds": round(time.perf_counter() - started, 6),
        "peak_rss_gib": _rss_gib(),
        "train_rows": int(data.X_train.shape[0]),
        "validation_rows": int(data.X_validation.shape[0]),
        "train_eras": len(set(data.eras_train)),
        "validation_eras": len(set(data.eras_validation)),
        "feature_count": data.feature_count,
        "tuning_corr_mean": tuning_score["corr_mean"],
        "tuning_corr_sharpe": tuning_score["corr_sharpe"],
        "tuning_n_eras": tuning_score["corr_n_eras"],
        "holdout_corr_mean": holdout_score["corr_mean"],
        "holdout_corr_sharpe": holdout_score["corr_sharpe"],
        "holdout_n_eras": holdout_score["corr_n_eras"],
        "full_corr_mean": full_score["corr_mean"],
        "full_corr_sharpe": full_score["corr_sharpe"],
        "full_fnc_mean": full_score["fnc_mean"],
        "full_fnc_sharpe": full_score["fnc_sharpe"],
        "full_n_eras": full_score["corr_n_eras"],
        "error_type": None,
        "error_message": None,
    }
    result.update(payout_metrics)
    return result


def _confirm_voting(
    config: VotingHPOConfig,
    best_params: dict[str, Any],
) -> dict[str, Any]:
    from nmr.sklearn_breadth import _catalog, _prepare_data, _score_predictions

    method = next(
        spec for spec in _catalog() if spec.name == "HistGradientBoostingRegressor"
    )
    started = time.perf_counter()
    data = _prepare_data(
        data_dir=config.data_dir,
        method=method,
        max_train_rows=None,
        max_validation_rows=None,
        purge_eras=config.purge_eras,
        score_window=config.score_window,
        target_col=config.target_col,
        payout_target_col=_PAYOUT_TARGET,
        scorecard_target_cols=("target_ender_20", _PAYOUT_TARGET),
    )
    model = _voting_estimator(best_params, seed=config.seed)
    fit_started = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    fit_seconds = round(time.perf_counter() - fit_started, 6)
    predict_started = time.perf_counter()
    predictions = np.asarray(model.predict(data.X_validation), dtype=float)
    predict_seconds = round(time.perf_counter() - predict_started, 6)
    if not np.all(np.isfinite(predictions)):
        raise ValueError("best VotingRegressor returned non-finite predictions")

    tuning_mask, holdout_mask = _hgb_validation_masks(data, config.tuning_fraction)
    score_started = time.perf_counter()
    full_score = _score_predictions(
        data,
        predictions,
        compute_fnc=config.compute_fnc,
    )
    tuning_score = _score_predictions(
        _hgb_data_slice(data, tuning_mask),
        predictions[tuning_mask],
        compute_fnc=False,
    )
    holdout_score = _score_predictions(
        _hgb_data_slice(data, holdout_mask),
        predictions[holdout_mask],
        compute_fnc=False,
    )
    payout_metrics = _confirmation_payout_metrics(
        config,
        data,
        predictions,
        model_id="sklearn::hpo::voting",
    )
    score_seconds = round(time.perf_counter() - score_started, 6)
    result = {
        "status": "completed",
        "target_col": config.target_col,
        "best_params_json": json.dumps(best_params, sort_keys=True),
        "load_seconds": data.load_seconds,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "score_seconds": score_seconds,
        "total_seconds": round(time.perf_counter() - started, 6),
        "peak_rss_gib": _rss_gib(),
        "train_rows": int(data.X_train.shape[0]),
        "validation_rows": int(data.X_validation.shape[0]),
        "train_eras": len(set(data.eras_train)),
        "validation_eras": len(set(data.eras_validation)),
        "feature_count": data.feature_count,
        "tuning_corr_mean": tuning_score["corr_mean"],
        "tuning_corr_sharpe": tuning_score["corr_sharpe"],
        "tuning_n_eras": tuning_score["corr_n_eras"],
        "holdout_corr_mean": holdout_score["corr_mean"],
        "holdout_corr_sharpe": holdout_score["corr_sharpe"],
        "holdout_n_eras": holdout_score["corr_n_eras"],
        "full_corr_mean": full_score["corr_mean"],
        "full_corr_sharpe": full_score["corr_sharpe"],
        "full_fnc_mean": full_score["fnc_mean"],
        "full_fnc_sharpe": full_score["fnc_sharpe"],
        "full_n_eras": full_score["corr_n_eras"],
        "error_type": None,
        "error_message": None,
    }
    result.update(payout_metrics)
    return result


def bayesian_hgb_sweep(config: HGBHPOConfig) -> HGBHPOResult:
    """Tune HGB on a temporal tuning slice and confirm the winner uncapped.

    The optimizer maximizes mean per-era CORR on the first part of the selected
    validation window. The trailing eras are held out for a post-search check,
    and a separate uncapped fit reports the full-window confirmation metrics.
    Optuna remains single-threaded and each completed trial is atomically
    checkpointed before the next trial begins.
    """
    from nmr.sklearn_breadth import _catalog, _prepare_data, _score_predictions

    method = next(
        spec for spec in _catalog() if spec.name == "HistGradientBoostingRegressor"
    )
    space = _hgb_search_space()
    parsed = _parse_space(space)
    distributions: dict[str, optuna.distributions.BaseDistribution] = {}
    for param in parsed:
        if param.kind == "float":
            distributions[param.name] = optuna.distributions.FloatDistribution(
                float(param.low), float(param.high), log=param.log
            )
        elif param.kind == "int":
            distributions[param.name] = optuna.distributions.IntDistribution(
                int(param.low),
                int(param.high),
                log=param.log,
                step=param.step or 1,
            )
        else:
            distributions[param.name] = optuna.distributions.CategoricalDistribution(
                list(param.choices)
            )

    identity = _hgb_run_identity(config)
    checkpoint_path = _hgb_checkpoint_path(config.output)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_hgb_trials(checkpoint_path, identity) if config.resume else []
    data = _prepare_data(
        data_dir=config.data_dir,
        method=method,
        max_train_rows=config.max_train_rows,
        max_validation_rows=config.max_validation_rows,
        purge_eras=config.purge_eras,
        score_window=config.score_window,
        target_col=config.target_col,
    )
    tuning_mask, _ = _hgb_validation_masks(data, config.tuning_fraction)
    tuning_data = _hgb_data_slice(data, tuning_mask)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=config.seed,
            n_startup_trials=config.n_startup_trials,
        ),
        storage=optuna.storages.InMemoryStorage(),
    )
    if rows:
        _add_hgb_study_trials(study, rows, distributions)
    else:
        study.enqueue_trial(_hgb_base_params())

    def objective(trial: optuna.Trial) -> float:
        started = time.perf_counter()
        params = {param.name: _suggest(trial, param) for param in parsed}
        row: dict[str, Any] = {
            "trial_id": trial.number,
            "status": "running",
            "params_json": json.dumps(params, sort_keys=True),
            "metric": "corr_mean",
            "metric_value": None,
            "tuning_corr_sharpe": None,
            "tuning_n_eras": None,
            "fit_seconds": None,
            "predict_seconds": None,
            "score_seconds": None,
            "total_seconds": None,
            "peak_rss_gib": None,
            "error_type": None,
            "error_message": None,
        }
        try:
            model = _hgb_estimator(params, seed=config.seed)
            fit_started = time.perf_counter()
            model.fit(data.X_train, data.y_train)
            row["fit_seconds"] = round(time.perf_counter() - fit_started, 6)
            predict_started = time.perf_counter()
            predictions = np.asarray(
                model.predict(tuning_data.X_validation), dtype=float
            )
            row["predict_seconds"] = round(time.perf_counter() - predict_started, 6)
            if not np.all(np.isfinite(predictions)):
                raise ValueError("HGB trial returned non-finite predictions")
            score_started = time.perf_counter()
            score = _score_predictions(
                tuning_data,
                predictions,
                compute_fnc=False,
            )
            row["score_seconds"] = round(time.perf_counter() - score_started, 6)
            row["metric_value"] = float(score["corr_mean"])
            row["tuning_corr_sharpe"] = float(score["corr_sharpe"])
            row["tuning_n_eras"] = int(score["corr_n_eras"])
            row["status"] = "completed"
            return float(score["corr_mean"])
        except Exception as exc:
            row.update(
                status="failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise optuna.exceptions.TrialPruned(str(exc)) from exc
        finally:
            row["total_seconds"] = round(time.perf_counter() - started, 6)
            row["peak_rss_gib"] = _rss_gib()
            rows.append(row)
            _write_hgb_trials(checkpoint_path, identity, rows)
            gc.collect()
            logger.info(
                "[hgb_hpo] trial=%d status=%s corr=%s fit=%ss total=%ss",
                trial.number,
                row["status"],
                row["metric_value"],
                row["fit_seconds"],
                row["total_seconds"],
            )

    remaining = max(0, config.n_trials - len(rows))
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=config.n_jobs)

    trial_frame = _hgb_trials_frame(rows)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    trial_frame.write_csv(config.output)
    complete = [row for row in rows if row.get("status") == "completed"]
    if complete:
        best_row = max(
            complete,
            key=lambda row: (float(row["metric_value"]), -int(row["trial_id"])),
        )
        best_params = json.loads(str(best_row["params_json"]))
        best_value = float(best_row["metric_value"])
    else:
        best_params, best_value = {}, float("nan")

    data = None
    tuning_data = None
    gc.collect()
    if best_params and config.full_confirmation:
        confirmation = _confirm_hgb(config, best_params)
    elif best_params:
        confirmation = {"status": "skipped", "reason": "full_confirmation_disabled"}
    else:
        confirmation = {"status": "unavailable", "reason": "no_completed_trials"}

    confirmation_path = _hgb_confirmation_path(config.output)
    pl.DataFrame([confirmation]).write_csv(confirmation_path)
    metadata = {
        "run_identity": identity,
        "method": "HistGradientBoostingRegressor",
        "target_col": config.target_col,
        "n_trials": config.n_trials,
        "completed_trials": len(complete),
        "best_params": best_params,
        "best_tuning_corr_mean": best_value,
        "confirmation_path": str(confirmation_path),
        "gpu_available": bool(discover_hardware().gpus),
        "gpu_used": False,
        "gpu_reason": "scikit-learn HistGradientBoostingRegressor has no CUDA execution path",
    }
    atomic_write_text(
        config.output.with_suffix(config.output.suffix + ".json"),
        json.dumps(metadata, sort_keys=True, indent=2, default=_hgb_json_default),
    )
    return HGBHPOResult(
        trials=trial_frame,
        best_params=best_params,
        best_value=best_value,
        confirmation=confirmation,
    )


def bayesian_voting_sweep(config: VotingHPOConfig) -> HGBHPOResult:
    """Tune the Ridge+HGB vote and confirm the selected ensemble uncapped."""
    from nmr.sklearn_breadth import _catalog, _prepare_data, _score_predictions

    method = next(
        spec for spec in _catalog() if spec.name == "HistGradientBoostingRegressor"
    )
    space = _voting_search_space()
    parsed = _parse_space(space)
    distributions: dict[str, optuna.distributions.BaseDistribution] = {}
    for param in parsed:
        if param.kind == "float":
            distributions[param.name] = optuna.distributions.FloatDistribution(
                float(param.low), float(param.high), log=param.log
            )
        elif param.kind == "int":
            distributions[param.name] = optuna.distributions.IntDistribution(
                int(param.low),
                int(param.high),
                log=param.log,
                step=param.step or 1,
            )
        else:
            distributions[param.name] = optuna.distributions.CategoricalDistribution(
                list(param.choices)
            )

    identity = _voting_run_identity(config)
    checkpoint_path = _hgb_checkpoint_path(config.output)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_hgb_trials(checkpoint_path, identity) if config.resume else []
    data = _prepare_data(
        data_dir=config.data_dir,
        method=method,
        max_train_rows=config.max_train_rows,
        max_validation_rows=config.max_validation_rows,
        purge_eras=config.purge_eras,
        score_window=config.score_window,
        target_col=config.target_col,
    )
    tuning_mask, _ = _hgb_validation_masks(data, config.tuning_fraction)
    tuning_data = _hgb_data_slice(data, tuning_mask)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=config.seed,
            n_startup_trials=config.n_startup_trials,
        ),
        storage=optuna.storages.InMemoryStorage(),
    )
    if rows:
        _add_hgb_study_trials(study, rows, distributions)
    else:
        study.enqueue_trial(_voting_base_params())

    def objective(trial: optuna.Trial) -> float:
        started = time.perf_counter()
        params = {param.name: _suggest(trial, param) for param in parsed}
        row: dict[str, Any] = {
            "trial_id": trial.number,
            "status": "running",
            "params_json": json.dumps(params, sort_keys=True),
            "metric": "corr_mean",
            "metric_value": None,
            "tuning_corr_sharpe": None,
            "tuning_n_eras": None,
            "fit_seconds": None,
            "predict_seconds": None,
            "score_seconds": None,
            "total_seconds": None,
            "peak_rss_gib": None,
            "error_type": None,
            "error_message": None,
        }
        try:
            model = _voting_estimator(params, seed=config.seed)
            fit_started = time.perf_counter()
            model.fit(data.X_train, data.y_train)
            row["fit_seconds"] = round(time.perf_counter() - fit_started, 6)
            predict_started = time.perf_counter()
            predictions = np.asarray(
                model.predict(tuning_data.X_validation), dtype=float
            )
            row["predict_seconds"] = round(time.perf_counter() - predict_started, 6)
            if not np.all(np.isfinite(predictions)):
                raise ValueError(
                    "VotingRegressor trial returned non-finite predictions"
                )
            score_started = time.perf_counter()
            score = _score_predictions(
                tuning_data,
                predictions,
                compute_fnc=False,
            )
            row["score_seconds"] = round(time.perf_counter() - score_started, 6)
            row["metric_value"] = float(score["corr_mean"])
            row["tuning_corr_sharpe"] = float(score["corr_sharpe"])
            row["tuning_n_eras"] = int(score["corr_n_eras"])
            row["status"] = "completed"
            return float(score["corr_mean"])
        except Exception as exc:
            row.update(
                status="failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise optuna.exceptions.TrialPruned(str(exc)) from exc
        finally:
            row["total_seconds"] = round(time.perf_counter() - started, 6)
            row["peak_rss_gib"] = _rss_gib()
            rows.append(row)
            _write_hgb_trials(checkpoint_path, identity, rows)
            gc.collect()
            logger.info(
                "[voting_hpo] trial=%d status=%s corr=%s fit=%ss total=%ss",
                trial.number,
                row["status"],
                row["metric_value"],
                row["fit_seconds"],
                row["total_seconds"],
            )

    remaining = max(0, config.n_trials - len(rows))
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=config.n_jobs)

    trial_frame = _hgb_trials_frame(rows)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    trial_frame.write_csv(config.output)
    complete = [row for row in rows if row.get("status") == "completed"]
    if complete:
        best_row = max(
            complete,
            key=lambda row: (float(row["metric_value"]), -int(row["trial_id"])),
        )
        best_params = json.loads(str(best_row["params_json"]))
        best_value = float(best_row["metric_value"])
    else:
        best_params, best_value = {}, float("nan")

    data = None
    tuning_data = None
    gc.collect()
    if best_params and config.full_confirmation:
        confirmation = _confirm_voting(config, best_params)
    elif best_params:
        confirmation = {"status": "skipped", "reason": "full_confirmation_disabled"}
    else:
        confirmation = {"status": "unavailable", "reason": "no_completed_trials"}

    confirmation_path = _hgb_confirmation_path(config.output)
    pl.DataFrame([confirmation]).write_csv(confirmation_path)
    metadata = {
        "run_identity": identity,
        "method": "VotingRegressor",
        "base_estimators": ["Ridge", "HistGradientBoostingRegressor"],
        "target_col": config.target_col,
        "n_trials": config.n_trials,
        "completed_trials": len(complete),
        "best_params": best_params,
        "best_tuning_corr_mean": best_value,
        "confirmation_path": str(confirmation_path),
        "gpu_available": bool(discover_hardware().gpus),
        "gpu_used": False,
        "gpu_reason": "scikit-learn regressors have no CUDA execution path",
    }
    atomic_write_text(
        config.output.with_suffix(config.output.suffix + ".json"),
        json.dumps(metadata, sort_keys=True, indent=2, default=_hgb_json_default),
    )
    return HGBHPOResult(
        trials=trial_frame,
        best_params=best_params,
        best_value=best_value,
        confirmation=confirmation,
    )


def _hgb_optional_rows(value: str) -> int | None:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("row limit must be >= 0")
    return None if parsed == 0 else parsed


def hgb_hpo_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimization for HGB on NumerAI medium."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "v5.3")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "reports" / "sklearn_hgb_hpo_trials.csv",
    )
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--n-startup-trials", type=int, default=8)
    parser.add_argument("--max-train-rows", type=_hgb_optional_rows, default=250_000)
    parser.add_argument(
        "--max-validation-rows",
        type=_hgb_optional_rows,
        default=300_000,
    )
    parser.add_argument("--purge-eras", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", dest="target_col", default="target")
    parser.add_argument(
        "--score-window", choices=("meta", "validation"), default="meta"
    )
    parser.add_argument("--tuning-fraction", type=float, default=0.75)
    parser.add_argument("--no-fnc", action="store_true")
    parser.add_argument("--no-confirmation", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    config = HGBHPOConfig(
        data_dir=args.data_dir,
        output=args.output,
        n_trials=args.n_trials,
        n_startup_trials=args.n_startup_trials,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        purge_eras=args.purge_eras,
        seed=args.seed,
        target_col=args.target_col,
        score_window=args.score_window,
        tuning_fraction=args.tuning_fraction,
        compute_fnc=not args.no_fnc,
        full_confirmation=not args.no_confirmation,
        resume=not args.no_resume,
    )
    result = bayesian_hgb_sweep(config)
    logger.info(
        "[hgb_hpo] completed_trials=%d best_tuning_corr=%s confirmation=%s",
        result.trials.filter(pl.col("status") == "completed").height,
        result.best_value,
        result.confirmation.get("status"),
    )
    return 0 if result.best_params else 1


def voting_hpo_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimization for VotingRegressor on NumerAI medium."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "v5.3")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "reports" / "sklearn_voting_hpo_trials.csv",
    )
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--n-startup-trials", type=int, default=8)
    parser.add_argument("--max-train-rows", type=_hgb_optional_rows, default=250_000)
    parser.add_argument(
        "--max-validation-rows",
        type=_hgb_optional_rows,
        default=300_000,
    )
    parser.add_argument("--purge-eras", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", dest="target_col", default="target")
    parser.add_argument(
        "--score-window", choices=("meta", "validation"), default="meta"
    )
    parser.add_argument("--tuning-fraction", type=float, default=0.75)
    parser.add_argument("--no-fnc", action="store_true")
    parser.add_argument("--no-confirmation", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    config = VotingHPOConfig(
        data_dir=args.data_dir,
        output=args.output,
        n_trials=args.n_trials,
        n_startup_trials=args.n_startup_trials,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        purge_eras=args.purge_eras,
        seed=args.seed,
        target_col=args.target_col,
        score_window=args.score_window,
        tuning_fraction=args.tuning_fraction,
        compute_fnc=not args.no_fnc,
        full_confirmation=not args.no_confirmation,
        resume=not args.no_resume,
    )
    result = bayesian_voting_sweep(config)
    logger.info(
        "[voting_hpo] completed_trials=%d best_tuning_corr=%s confirmation=%s",
        result.trials.filter(pl.col("status") == "completed").height,
        result.best_value,
        result.confirmation.get("status"),
    )
    return 0 if result.best_params else 1
