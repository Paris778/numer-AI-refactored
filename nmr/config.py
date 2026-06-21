"""Typed experiment configuration: YAML -> dataclasses, determinism, path resolution.

This module is the single source of truth for how an experiment is parameterized.
Every downstream slice consumes a frozen :class:`ExperimentConfig`; nothing else
reads YAML directly. Keeping configuration typed and immutable makes runs
reproducible and makes invalid experiments fail loudly at load time.
"""

from __future__ import annotations

import dataclasses
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = the parent of the `nmr` package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

VALID_FEATURE_SETS = ("small", "medium", "all")
VALID_MODEL_BACKENDS = ("lightgbm", "xgboost")
VALID_MODEL_PRESETS = ("fast", "standard", "deep")
VALID_EVAL_BACKENDS = ("custom", "official")
VALID_SPLIT_SCHEMES = ("walk_forward", "anchor")

__all__ = [
    "REPO_ROOT",
    "DataConfig",
    "SplitConfig",
    "ModelConfig",
    "EvalConfig",
    "RunConfig",
    "ExperimentConfig",
    "load_config",
    "set_global_seeds",
]


def _resolve_path(p: str | Path) -> Path:
    """Resolve a path against the repo root unless it is already absolute."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


@dataclass(frozen=True)
class DataConfig:
    """Dataset selection: version, feature set, and target columns."""

    version: str = "v5.2"
    feature_set: str = "small"
    targets: tuple[str, ...] = ("target",)
    data_dir: Path = REPO_ROOT / "data"

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "data_dir", _resolve_path(self.data_dir))
        if self.feature_set not in VALID_FEATURE_SETS:
            raise ValueError(
                f"feature_set={self.feature_set!r} not in {VALID_FEATURE_SETS}"
            )
        if not self.targets:
            raise ValueError("data.targets must contain at least one target")

    def path(self, filename: str) -> Path:
        """Absolute path to a dataset file for this version, e.g. ``train.parquet``."""
        return self.data_dir / self.version / filename


@dataclass(frozen=True)
class SplitConfig:
    """Era-grouped, leakage-safe validation splitting."""

    scheme: str = "walk_forward"
    purge_eras: int = 8  # 20D targets; use 16 for 60D horizons
    embargo_eras: int = 4
    n_folds: int = 4

    def __post_init__(self) -> None:
        if self.scheme not in VALID_SPLIT_SCHEMES:
            raise ValueError(
                f"split.scheme={self.scheme!r} not in {VALID_SPLIT_SCHEMES}"
            )
        if self.purge_eras < 0 or self.embargo_eras < 0:
            raise ValueError("purge_eras and embargo_eras must be >= 0")
        if self.n_folds < 1:
            raise ValueError("split.n_folds must be >= 1")


@dataclass(frozen=True)
class ModelConfig:
    """Model backend, parameter preset, and explicit param overrides."""

    backend: str = "lightgbm"
    preset: str = "fast"
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in VALID_MODEL_BACKENDS:
            raise ValueError(
                f"model.backend={self.backend!r} not in {VALID_MODEL_BACKENDS}"
            )
        if self.preset not in VALID_MODEL_PRESETS:
            raise ValueError(
                f"model.preset={self.preset!r} not in {VALID_MODEL_PRESETS}"
            )


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation backend (fast custom vs official oracle) and metric selection."""

    backend: str = "custom"
    main_target: str = "target"
    metrics: tuple[str, ...] = ("corr", "mmc", "fnc", "sharpe")

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if self.backend not in VALID_EVAL_BACKENDS:
            raise ValueError(
                f"evaluation.backend={self.backend!r} not in {VALID_EVAL_BACKENDS}"
            )
        if not self.metrics:
            raise ValueError("evaluation.metrics must contain at least one metric")


@dataclass(frozen=True)
class RunConfig:
    """Run identity, determinism seed, and artifact output location."""

    name: str = "default"
    seed: int = 42
    artifacts_dir: Path = REPO_ROOT / "artifacts"

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts_dir", _resolve_path(self.artifacts_dir))
        if not self.name:
            raise ValueError("run.name must be a non-empty string")


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration aggregating all layers."""

    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    run: RunConfig = field(default_factory=RunConfig)


_SECTIONS = {
    "data": DataConfig,
    "split": SplitConfig,
    "model": ModelConfig,
    "evaluation": EvalConfig,
    "run": RunConfig,
}


def _build(cls: type, data: dict[str, Any]):
    """Construct a config dataclass, rejecting unknown keys with a clear error."""
    if not isinstance(data, dict):
        raise ValueError(
            f"{cls.__name__} section must be a mapping, got {type(data).__name__}"
        )
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an :class:`ExperimentConfig` from a YAML file.

    Omitted sections and fields fall back to typed defaults. Unknown keys and
    invalid values raise ``ValueError`` so misconfigured experiments fail fast.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping")
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    return ExperimentConfig(
        **{
            section: _build(cls, raw.get(section, {}))
            for section, cls in _SECTIONS.items()
        }
    )


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and the hash seed for reproducible runs.

    Model backends (LightGBM/XGBoost) receive their seed via model params, not here.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is a hard dependency, but stay resilient.
        pass
