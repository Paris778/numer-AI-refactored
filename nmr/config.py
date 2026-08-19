"""Typed experiment configuration: YAML -> dataclasses, determinism, path resolution.

This module is the single source of truth for how an experiment is parameterized.
Every downstream slice consumes a frozen :class:`ExperimentConfig`; nothing else
reads YAML directly. Keeping configuration typed and immutable makes runs
reproducible and makes invalid experiments fail loudly at load time.
"""

from __future__ import annotations

import dataclasses
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = the parent of the `nmr` package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

VALID_FEATURE_SETS = ("small", "medium", "all")
VALID_MODEL_BACKENDS = ("lightgbm", "xgboost", "catboost")
VALID_MODEL_PRESETS = ("fast", "standard", "deep")
VALID_MODEL_DEVICES = ("auto", "gpu", "cpu")
VALID_EVAL_BACKENDS = ("custom", "official")
VALID_EVAL_METRICS = ("corr", "mmc", "fnc", "sharpe")
VALID_SPLIT_SCHEMES = ("walk_forward", "anchor")
VALID_ENSEMBLE_METHODS = ("ridge", "non_negative")
VALID_HORIZONS = ("20D", "60D")
# Leakage law (AGENTS.md §4): 8-era purge for 20D targets, 16 for 60D. These
# are MINIMUMs — stricter purges are allowed, weaker ones are a correctness
# bug and rejected at load time.
PURGE_ERAS_20D = 8
PURGE_ERAS_60D = 16
# Horizon-encoding target-name convention (ARCHITECTURE.md §K): targets end
# with ``20``/``60`` (e.g. ``target_cyrusd_20``, ``ender60``) and must agree
# with the declared data.horizon.
_HORIZON_TARGET_RE = re.compile(r"(20|60)$")


def _validate_purge_vs_horizon(config: ExperimentConfig) -> None:
    """Horizon/target-name consistency at config load (leakage law, AGENTS.md §4).

    The purge FLOOR is data-aware and enforced at run time via
    :func:`enforce_purge_horizon_law` (a real 574-era dataset must use ≥ 8
    purge for 20D / ≥ 16 for 60D; small synthetic datasets are governed by the
    splitter's own geometry). Here we validate that target names encoding a
    horizon (``target_<name>_20/60``, ``ender60``) agree with the declared
    ``data.horizon`` — mixed-horizon sets fail loud.
    """
    for target in config.data.targets:
        match = _HORIZON_TARGET_RE.search(target)
        if not match:
            continue
        encoded = f"{match.group(1)}D"
        if encoded != config.data.horizon:
            raise ValueError(
                f"data.targets entry {target!r} encodes horizon {encoded} but "
                f"data.horizon={config.data.horizon}; set the horizon explicitly "
                "or fix the target"
            )


def enforce_purge_horizon_law(era_count: int, config: ExperimentConfig) -> None:
    """Enforce the purge/horizon floor once the era count is known.

    ``era_count`` is the number of distinct training eras. The convention floor
    (8 for 20D, 16 for 60D) applies when the dataset has at least twice the
    floor's eras (the real-data regime); smaller datasets are governed by the
    splitter's own geometry and may legitimately use smaller purges.
    """
    floor = PURGE_ERAS_20D if config.data.horizon == "20D" else PURGE_ERAS_60D
    if era_count >= 2 * floor and config.split.purge_eras < floor:
        raise ValueError(
            f"split.purge_eras={config.split.purge_eras} < minimum {floor} for "
            f"data.horizon={config.data.horizon} on {era_count} eras "
            f"(leakage law: {PURGE_ERAS_20D} eras for 20D targets, "
            f"{PURGE_ERAS_60D} for 60D)"
        )

__all__ = [
    "REPO_ROOT",
    "DataConfig",
    "SplitConfig",
    "ModelConfig",
    "EvalConfig",
    "RiskConfig",
    "EnsembleConfig",
    "RunConfig",
    "ExperimentConfig",
    "PURGE_ERAS_20D",
    "PURGE_ERAS_60D",
    "VALID_HORIZONS",
    "config_from_dict",
    "enforce_purge_horizon_law",
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

    version: str = "v5.3"
    feature_set: str = "small"
    feature_subset: str | None = None
    supplemental_feature_sets: Path | None = None
    targets: tuple[str, ...] = ("target",)
    horizon: str = "20D"
    data_dir: Path = REPO_ROOT / "data"

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "data_dir", _resolve_path(self.data_dir))
        if self.horizon not in VALID_HORIZONS:
            raise ValueError(
                f"data.horizon={self.horizon!r} not in {VALID_HORIZONS}"
            )
        if self.supplemental_feature_sets is not None:
            object.__setattr__(
                self,
                "supplemental_feature_sets",
                _resolve_path(self.supplemental_feature_sets),
            )
            if not str(self.supplemental_feature_sets).strip():
                raise ValueError(
                    "data.supplemental_feature_sets must be a non-empty path when provided"
                )
        if self.feature_set not in VALID_FEATURE_SETS:
            raise ValueError(
                f"feature_set={self.feature_set!r} not in {VALID_FEATURE_SETS}"
            )
        if self.feature_subset is not None and not self.feature_subset:
            raise ValueError(
                "data.feature_subset must be a non-empty string when provided"
            )
        if not self.targets:
            raise ValueError("data.targets must contain at least one target")

    @property
    def resolved_feature_set(self) -> str:
        """Feature set actually used: explicit ``feature_subset`` wins over ``feature_set``.

        ``feature_subset`` names are validated against ``features.json`` at
        ingestion time (fail loud, fail late — ``IngestionAgent.features``).
        """
        return self.feature_subset if self.feature_subset is not None else self.feature_set

    def path(self, filename: str) -> Path:
        """Absolute path to a dataset file for this version, e.g. ``train.parquet``."""
        return self.data_dir / self.version / filename


@dataclass(frozen=True)
class SplitConfig:
    """Era-grouped, leakage-safe validation splitting."""

    scheme: str = "walk_forward"
    purge_eras: int = 8  # 20D targets; use 16 for 60D horizons
    embargo_eras: int = 0
    n_folds: int = 4

    def __post_init__(self) -> None:
        if self.scheme not in VALID_SPLIT_SCHEMES:
            raise ValueError(
                f"split.scheme={self.scheme!r} not in {VALID_SPLIT_SCHEMES}"
            )
        if self.embargo_eras != 0:
            # A2 (audit SEV-3): the knob was validated, documented, and
            # structurally inert — a config knob that lies. It is now rejected
            # at load: purge_eras is the active leakage buffer, and a non-zero
            # embargo was never used by fold geometry.
            raise ValueError(
                "split.embargo_eras must be 0: the knob is structurally inert "
                "(purge_eras is the active leakage buffer); non-zero values "
                "were never used by fold geometry and are rejected at load"
            )
        if self.purge_eras < 0:
            raise ValueError("purge_eras must be >= 0")
        if self.n_folds < 1:
            raise ValueError("split.n_folds must be >= 1")


@dataclass(frozen=True)
class ModelConfig:
    """Model backend, parameter preset, explicit overrides, and device policy.

    ``device``: ``auto`` (default — GPU-first with CPU fallback in CV, the
    legacy behavior), ``gpu`` (force GPU for CV/experimentation; a failed GPU
    fit raises — no silent fallback), ``cpu`` (never attempt GPU). The
    deployment artifact path (``train_full_history``) is always CPU by
    invariant: determinism is per-device and the hosted runtime may lack a
    GPU.
    """

    backend: str = "lightgbm"
    preset: str = "fast"
    params: dict[str, Any] = field(default_factory=dict)
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.backend not in VALID_MODEL_BACKENDS:
            raise ValueError(
                f"model.backend={self.backend!r} not in {VALID_MODEL_BACKENDS}"
            )
        if self.preset not in VALID_MODEL_PRESETS:
            raise ValueError(
                f"model.preset={self.preset!r} not in {VALID_MODEL_PRESETS}"
            )
        if self.device not in VALID_MODEL_DEVICES:
            raise ValueError(
                f"model.device={self.device!r} not in {VALID_MODEL_DEVICES}"
            )


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation backend (fast custom vs official oracle) and metric selection."""

    backend: str = "custom"
    main_target: str = "target"
    metrics: tuple[str, ...] = ("corr", "mmc", "fnc", "sharpe")
    validation_scorecard: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if self.backend not in VALID_EVAL_BACKENDS:
            raise ValueError(
                f"evaluation.backend={self.backend!r} not in {VALID_EVAL_BACKENDS}"
            )
        if not self.metrics:
            raise ValueError("evaluation.metrics must contain at least one metric")
        unknown = sorted(set(self.metrics) - set(VALID_EVAL_METRICS))
        if unknown:
            raise ValueError(
                f"evaluation.metrics contains unknown names {unknown}; "
                f"valid metrics: {VALID_EVAL_METRICS}"
            )


@dataclass(frozen=True)
class RiskConfig:
    """Risk transforms: neutralization strength and cache budget."""

    neutralization_proportion: float = 1.0
    cache_max_bytes: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.neutralization_proportion <= 1.0:
            raise ValueError("risk.neutralization_proportion must be in [0, 1]")
        if self.cache_max_bytes is not None and self.cache_max_bytes < 0:
            raise ValueError("risk.cache_max_bytes must be >= 0 or None")


@dataclass(frozen=True)
class EnsembleConfig:
    """Ensemble weight-learning method (applies to the OOF blend)."""

    method: str = "ridge"

    def __post_init__(self) -> None:
        if self.method not in VALID_ENSEMBLE_METHODS:
            raise ValueError(
                f"ensemble.method={self.method!r} not in {VALID_ENSEMBLE_METHODS}"
            )


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
    risk: RiskConfig = field(default_factory=RiskConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    run: RunConfig = field(default_factory=RunConfig)


_SECTIONS = {
    "data": DataConfig,
    "split": SplitConfig,
    "model": ModelConfig,
    "evaluation": EvalConfig,
    "risk": RiskConfig,
    "ensemble": EnsembleConfig,
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


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """Build a validated :class:`ExperimentConfig` from a raw section mapping.

    Shared by :func:`load_config` (YAML) and the promotion writer
    (``nmr/promote.py`` reconstructing a stored run's config after
    normalization). Unknown sections/keys and invalid values raise
    ``ValueError`` so misconfigured experiments fail fast.
    """
    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping")
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    config = ExperimentConfig(
        **{
            section: _build(cls, raw.get(section, {}))
            for section, cls in _SECTIONS.items()
        }
    )
    _validate_purge_vs_horizon(config)
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an :class:`ExperimentConfig` from a YAML file.

    Omitted sections and fields fall back to typed defaults. Unknown keys and
    invalid values raise ``ValueError`` so misconfigured experiments fail fast.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return config_from_dict(raw)


def set_global_seeds(seed: int) -> None:
    """Seed Python and NumPy for reproducible runs.

    Note: ``PYTHONHASHSEED`` is NOT set here — CPython fixes hash randomization
    at interpreter startup, so a runtime assignment affects only subprocesses
    (none are spawned). Model backends (LightGBM/XGBoost) receive their seed via
    model params, not here.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is a hard dependency, but stay resilient.
        pass
