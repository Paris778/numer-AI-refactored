"""Untiered benchmark fleet: community & tutorial model recreation layer.

Fleet cells are benchmark models without a tier assignment: they are scored
through the same ``evaluate_model`` pipeline as the 5-tier hierarchy and
their measured scorecards place them against the tier ladder indirectly.
Spec: docs/superpowers/specs/2026-08-19-benchmark-fleet-design.md.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import polars as pl
import yaml

from nmr.benchmark import (
    DEFAULT_BENCHMARK_PURGE_ERAS,
    DEFAULT_BENCHMARK_SEED,
    VALID_INPUT_SPACES,
    _freeze_mapping,
    _reject_unknown_keys,
    generate_canonical_predictions,
    train_validation_purged_split,
)
from nmr.features import feature_stability_screen
from nmr.risk import NeutralizationEngine

logger = logging.getLogger("nmr.benchmark_fleet")

# Generator/runner/placement names (generate_*, fleet_frame, BenchmarkFleet, ...)
# join this list in the tasks that define them.
__all__ = [
    "FleetCellConfig",
    "FleetFileConfig",
    "VALID_FLEET_MODEL_KINDS",
    "VALID_FLEET_NEUTRALIZATION",
    "VALID_FLEET_NEUTRALIZER_SELECTIONS",
    "generate_fleet_lightgbm_predictions",
    "generate_lagged_target_predictions",
    "load_fleet_config",
    "load_fleet_suite_config",
]

VALID_FLEET_MODEL_KINDS: tuple[str, ...] = (
    "target_lag_mean", "lightgbm", "xgboost", "mlp", "ridge_stack",
)
VALID_FLEET_NEUTRALIZATION: tuple[float | None, ...] = (None, 0.25, 0.35, 0.5, 1.0)
VALID_FLEET_NEUTRALIZER_SELECTIONS: tuple[str, ...] = ("none", "riskiest_50")
DEFAULT_NEUTRALIZER_COUNT: int = 50


@dataclasses.dataclass(frozen=True)
class FleetCellConfig:
    benchmark_id: str
    source: str
    input_space: str
    model_kind: str
    targets: tuple[str, ...] = ("target",)
    target_weights: Mapping[str, float] | None = None
    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    seed: int = DEFAULT_BENCHMARK_SEED
    neutralization: float | None = None
    neutralizer_selection: str = "none"
    neutralizer_count: int = DEFAULT_NEUTRALIZER_COUNT
    fast_mode_params: Mapping[str, Any] | None = None
    anchors: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.benchmark_id or not isinstance(self.benchmark_id, str):
            raise ValueError(f"benchmark_id must be a non-empty string: {self.benchmark_id!r}")
        if not self.source or not isinstance(self.source, str):
            raise ValueError(f"source must be a non-empty string: {self.source!r}")
        if self.input_space not in VALID_INPUT_SPACES:
            raise ValueError(
                f"input_space={self.input_space!r} not in {VALID_INPUT_SPACES}"
            )
        if self.model_kind not in VALID_FLEET_MODEL_KINDS:
            raise ValueError(
                f"model_kind={self.model_kind!r} not in {VALID_FLEET_MODEL_KINDS}"
            )
        if not isinstance(self.targets, tuple) or not self.targets:
            raise ValueError("targets must be a non-empty tuple")
        if not all(isinstance(t, str) and t for t in self.targets):
            raise ValueError(f"targets must be non-empty strings: {self.targets!r}")
        if self.model_kind == "target_lag_mean":
            if self.input_space != "none":
                raise ValueError(
                    "target_lag_mean requires input_space='none', "
                    f"got {self.input_space!r}"
                )
            if len(self.targets) != 1:
                raise ValueError(
                    "target_lag_mean requires exactly one single target, "
                    f"got {self.targets!r}"
                )
        if self.model_kind == "ridge_stack":
            for key in ("mode", "main_target", "specialists"):
                if key not in self.params:
                    raise ValueError(
                        f"ridge_stack requires params.{key}"
                    )
            if self.params["mode"] not in ("fixed", "search"):
                raise ValueError(
                    f"ridge_stack params.mode must be 'fixed' or 'search', "
                    f"got {self.params['mode']!r}"
                )
        if self.neutralization not in VALID_FLEET_NEUTRALIZATION:
            raise ValueError(
                f"neutralization={self.neutralization!r} not in "
                f"{VALID_FLEET_NEUTRALIZATION}"
            )
        if self.neutralizer_selection not in VALID_FLEET_NEUTRALIZER_SELECTIONS:
            raise ValueError(
                f"neutralizer_selection={self.neutralizer_selection!r} not in "
                f"{VALID_FLEET_NEUTRALIZER_SELECTIONS}"
            )
        if not isinstance(self.neutralizer_count, int) or isinstance(self.neutralizer_count, bool) \
                or self.neutralizer_count < 1:
            raise ValueError(
                f"neutralizer_count must be a positive int, got {self.neutralizer_count!r}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        object.__setattr__(self, "params", _freeze_mapping(self.params, name="params"))
        if self.target_weights is not None:
            weights = dict(self.target_weights)
            for target in weights:
                if target not in self.targets:
                    raise ValueError(
                        f"target_weights key {target!r} not in targets {self.targets!r}"
                    )
            if not all(isinstance(w, (int, float)) and not isinstance(w, bool)
                       and float(w) > 0.0 for w in weights.values()):
                raise ValueError("target_weights must be positive numbers")
            object.__setattr__(
                self, "target_weights",
                _freeze_mapping(self.target_weights, name="target_weights"),
            )
        if self.fast_mode_params is not None:
            object.__setattr__(
                self, "fast_mode_params",
                _freeze_mapping(self.fast_mode_params, name="fast_mode_params"),
            )
        if self.anchors is not None:
            object.__setattr__(
                self, "anchors",
                _freeze_mapping(self.anchors, name="anchors"),
            )


@dataclasses.dataclass(frozen=True)
class FleetFileConfig:
    cells: tuple[FleetCellConfig, ...] = ()

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("fleet config requires non-empty cells")
        ids = [cell.benchmark_id for cell in self.cells]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate benchmark ids in file: {ids}")


def load_fleet_config(path: str | Path) -> FleetFileConfig:
    """Load and validate a single fleet config file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"fleet config must be a mapping, got {type(raw).__name__}")
    _reject_unknown_keys(FleetFileConfig, raw)
    if not isinstance(raw.get("cells", []), list):
        raise ValueError("cells must be a list")
    cells: list[FleetCellConfig] = []
    for data in raw.get("cells", []):
        if not isinstance(data, dict):
            raise ValueError(f"fleet cell must be a mapping, got {type(data).__name__}")
        data = dict(data)
        if "neutralization" in data and data["neutralization"] == "none":
            data["neutralization"] = None
        if isinstance(data.get("targets"), list):
            data["targets"] = tuple(data["targets"])
        _reject_unknown_keys(FleetCellConfig, data)
        cells.append(FleetCellConfig(**data))
    return FleetFileConfig(cells=tuple(cells))


def load_fleet_suite_config(config_dir: str | Path) -> tuple[FleetCellConfig, ...]:
    """Load every *.yaml in config_dir, dedupe ids, sort by benchmark_id."""
    directory = Path(config_dir)
    files = sorted(p for p in directory.glob("*.yaml"))
    if not files:
        raise ValueError(f"no fleet config files found in {directory}")
    all_cells: list[FleetCellConfig] = []
    for path in files:
        all_cells.extend(load_fleet_config(path).cells)
    ids = [cell.benchmark_id for cell in all_cells]
    if len(set(ids)) != len(ids):
        seen = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate benchmark ids across configs: {seen}")
    return tuple(sorted(all_cells, key=lambda c: c.benchmark_id))


def generate_lagged_target_predictions(
    train: pl.DataFrame,
    val_index: pl.DataFrame,
    *,
    target: str,
    window: int = 1,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Silly baseline: per validation era, predict the trailing-train target mean.

    All rows pooled across the trailing ``window`` train eras; train targets
    only, so the prediction is leak-safe by construction.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError(f"window must be a positive int, got {window!r}")
    if target not in train.columns:
        raise ValueError(f"train missing target column: {target!r}")
    for col in (era_col, id_col):
        if col not in val_index.columns:
            raise ValueError(f"val_index missing required column: {col!r}")

    train_eras = sorted(train.get_column(era_col).unique().to_list())
    val_eras = sorted(val_index.get_column(era_col).unique().to_list())
    if not train_eras or not val_eras:
        raise ValueError("train and val_index must each contain at least one era")
    if max(int(e) for e in train_eras) >= min(int(e) for e in val_eras):
        raise ValueError("train eras must be strictly earlier than validation eras")
    if window > len(train_eras):
        raise ValueError(
            f"window={window} exceeds available train eras ({len(train_eras)})"
        )

    trailing = train_eras[-window:]
    pooled = (
        train.filter(pl.col(era_col).is_in(trailing))
        .select(pl.col(target).cast(pl.Float64))
        .drop_nulls()
    )
    if pooled.is_empty():
        raise ValueError(f"trailing train eras have no finite {target!r} rows")
    value = float(pooled.get_column(target).mean())

    return (
        val_index.select([era_col, id_col])
        .sort([era_col, id_col])
        .with_columns(pl.lit(value, dtype=pl.Float64).alias(pred_col))
    )


def _select_riskiest_features(
    train: pl.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    count: int,
    era_col: str = "era",
) -> list[str]:
    """Top-``count`` features by cross-regime drift (framework risk screen).

    Ranked by ``cross_regime_variance`` descending, nulls last, feature name
    ascending as the deterministic tie-break. Documented deviation from the
    notebooks' ``get_biggest_change_features`` (same intent — most unstable
    features — via the framework-tested screen).
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(f"count must be a positive int, got {count!r}")
    screen = feature_stability_screen(
        train, feature_cols=list(feature_cols), target_col=target_col,
        era_col=era_col,
    )
    # polars >= 1.41 dropped Expr.nulls_last()/Expr.desc(); the kwargs form
    # below is the equivalent (variance desc with nulls last, name asc).
    ranked = screen.sort(
        by=["cross_regime_variance", "feature"],
        descending=[True, False],
        nulls_last=[True, False],
    )
    return ranked.get_column("feature").to_list()[:count]


def generate_fleet_lightgbm_predictions(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    targets: Sequence[str],
    feature_cols: Sequence[str],
    params: Mapping[str, Any],
    seed: int,
    neutralization: float = 0.0,
    neutralizer_selection: str = "none",
    neutralizer_count: int = DEFAULT_NEUTRALIZER_COUNT,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Fleet LightGBM: canonical fits + optional riskiest-feature neutralization."""
    if neutralizer_selection not in VALID_FLEET_NEUTRALIZER_SELECTIONS:
        raise ValueError(
            f"neutralizer_selection={neutralizer_selection!r} not in "
            f"{VALID_FLEET_NEUTRALIZER_SELECTIONS}"
        )
    trimmed_train_eras, _ = train_validation_purged_split(
        train.get_column(era_col).unique().to_list(),
        val.get_column(era_col).unique().to_list(),
        purge_eras=purge_eras,
    )
    train_rows = train.filter(pl.col(era_col).is_in(trimmed_train_eras))

    if neutralizer_selection == "riskiest_50":
        neutralizer_cols = _select_riskiest_features(
            train_rows, feature_cols=feature_cols,
            target_col=list(targets)[0], count=neutralizer_count, era_col=era_col,
        )
    else:
        neutralizer_cols = list(feature_cols)

    out = generate_canonical_predictions(
        train, val, targets=list(targets), feature_cols=list(feature_cols),
        params=params, seed=seed, neutralization=0.0,
        purge_eras=purge_eras, era_col=era_col, id_col=id_col, pred_col=pred_col,
    )

    if float(neutralization) > 0.0:
        with_features = out.join(
            val.select([era_col, id_col, *neutralizer_cols]),
            on=[era_col, id_col], how="inner",
        )
        out = NeutralizationEngine().neutralize(
            with_features, pred_col=pred_col, feature_cols=neutralizer_cols,
            era_col=era_col, proportion=float(neutralization),
        ).select([era_col, id_col, pred_col]).sort([era_col, id_col])
    return out
