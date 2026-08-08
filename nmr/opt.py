"""Bayesian hyperparameter optimization via Optuna (user-granted dependency).

``bayesian_sweep`` is the single Optuna-integration point. Space definitions are
declarative dicts (ARCHITECTURE.md §S); the objective is harness-internal
(``research._held_out_metric``); sweeps are seeded, single-threaded, and
deterministic per environment.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import optuna
import polars as pl

from nmr.config import ExperimentConfig
from nmr.models import resolve_model_params
from nmr.research import SweepResult, _held_out_metric, _override_config

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger("nmr.opt")

__all__ = ["bayesian_sweep"]

_VALID_METRICS = ("mean", "std", "sharpe", "max_drawdown", "corr_sharpe_ac")
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
            log = bool(spec.get("log", False))
            if log and low <= 0:
                raise ValueError(
                    f"parameter {name!r}: 'low' must be > 0 when log=True, got {low}"
                )
            step = spec.get("step")
            if step is not None and (not isinstance(step, int) or step < 1):
                raise ValueError(f"parameter {name!r}: step must be a positive int")
            if log and step is not None:
                raise ValueError(
                    f"parameter {name!r}: log=True and step are mutually exclusive"
                )
            parsed.append(
                _SpaceParam(kind=kind, name=name, low=low, high=high,
                            log=log, step=step)
            )
        else:
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"parameter {name!r}: choices must be a non-empty list")
            if not all(isinstance(c, _JSON_PRIMITIVES) for c in choices):
                raise ValueError(
                    f"parameter {name!r}: categorical choices must be str/int/float/bool"
                )
            parsed.append(_SpaceParam(kind="categorical", name=name, choices=list(choices)))
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

    Seeded TPE sampler (``TPESampler(seed=...)`` — deterministic-by-default
    since Optuna 4.x, which removed the 3.x ``deterministic`` flag; verified
    on 4.9.0), single-threaded (``n_jobs`` must be 1 — parallel trials break
    TPE determinism), in-memory storage.
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

    parsed = _parse_space(space)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=optuna.storages.InMemoryStorage(),
    )
    if enqueue_base_config:
        resolved = resolve_model_params(base_config.model.preset, base_config.model.params)
        anchor = {p.name: resolved[p.name] for p in parsed if p.name in resolved}
        if anchor:
            study.enqueue_trial(anchor)

    def objective(trial: optuna.Trial) -> float:
        params = {p.name: _suggest(trial, p) for p in parsed}
        cfg = _override_config(base_config, params)
        try:
            value = _held_out_metric(cfg, metric_name=metric)
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
        rows.append(
            {
                "trial_id": t.number,
                "params_json": json.dumps(t.params, sort_keys=True),
                "metric_value": value,
                "metric": metric,
            }
        )
    trial_df = pl.DataFrame(rows).sort(
        ["metric_value", "trial_id"], descending=[True, False], nulls_last=True
    )
    best = study.best_trial if len(study.best_trials) > 0 else None
    return SweepResult(
        trials=trial_df,
        best_params=best.params if best is not None else {},
        best_value=float(best.value) if best is not None else float("nan"),
    )
