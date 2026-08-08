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
