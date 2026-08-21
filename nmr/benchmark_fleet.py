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

import numpy as np
import polars as pl
import yaml
from lightgbm import early_stopping
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from nmr.benchmark import (
    DEFAULT_BENCHMARK_PURGE_ERAS,
    DEFAULT_BENCHMARK_SEED,
    VALID_INPUT_SPACES,
    BenchmarkData,
    Tier4GateConfig,
    _freeze_mapping,
    _reject_unknown_keys,
    _standardize_feature_block,
    generate_canonical_predictions,
    resolve_benchmark_feature_cols,
    scorecards_to_frame,
    tier4_gate_verdict,
    train_validation_purged_split,
)
from nmr.ensemble import Ensembler
from nmr.features import feature_stability_screen
from nmr.models import construct_tree_model
from nmr.risk import NeutralizationEngine
from nmr.scorecard import MetricScorecard, evaluate_model

logger = logging.getLogger("nmr.benchmark_fleet")

# Generators, config loaders, the fleet runner, placement, and frame writers.
__all__ = [
    "BenchmarkFleet",
    "FleetCellConfig",
    "FleetFileConfig",
    "FleetResult",
    "VALID_FLEET_MODEL_KINDS",
    "VALID_FLEET_NEUTRALIZATION",
    "VALID_FLEET_NEUTRALIZER_SELECTIONS",
    "fleet_frame",
    "fleet_placement",
    "generate_fleet_lightgbm_predictions",
    "generate_fleet_xgb_predictions",
    "generate_lagged_target_predictions",
    "generate_mlp_predictions",
    "generate_ridge_stack_predictions",
    "load_fleet_config",
    "load_fleet_suite_config",
    "load_tier_rungs_from_csv",
    "select_fleet_cells",
    "write_fleet_csv",
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


_ES_KEYS = ("early_stopping_rounds", "holdout_era_frac")


def generate_fleet_xgb_predictions(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    targets: Sequence[str],
    feature_cols: Sequence[str],
    params: Mapping[str, Any],
    seed: int,
    neutralization: float = 0.0,
    target_weights: Mapping[str, float] | None = None,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Fleet XGBoost: per-target fits, optional early stopping, weighted blend.

    Optional ``neutralization`` (proportion in (0, 1]) applies the framework
    ``NeutralizationEngine`` to the final rank-gaussianized blend over all
    ``feature_cols``.
    """
    if not targets or not feature_cols:
        raise ValueError("targets and feature_cols must be non-empty")
    weights = dict(target_weights) if target_weights else {}
    for target in weights:
        if target not in targets:
            raise ValueError(
                f"target_weights key {target!r} not in targets {list(targets)!r}"
            )
    trimmed_train_eras, _ = train_validation_purged_split(
        train.get_column(era_col).unique().to_list(),
        val.get_column(era_col).unique().to_list(),
        purge_eras=purge_eras,
    )
    train_rows = train.filter(pl.col(era_col).is_in(trimmed_train_eras))
    val_rows = val.sort([era_col, id_col])
    missing_feats = [c for c in feature_cols if c not in train.columns or c not in val.columns]
    if missing_feats:
        raise ValueError(f"missing feature columns: {missing_feats}")

    es_rounds = params.get("early_stopping_rounds")
    holdout_frac = float(params.get("holdout_era_frac", 0.1))
    fit_eras = list(trimmed_train_eras)
    holdout_eras: list[str] = []
    if es_rounds is not None and len(fit_eras) >= 4:
        n_hold = max(1, int(round(holdout_frac * len(fit_eras))))
        if n_hold >= len(fit_eras):
            n_hold = len(fit_eras) // 2
        holdout_eras = fit_eras[-n_hold:]
        fit_eras = fit_eras[:-n_hold]
    model_params = {k: v for k, v in params.items() if k not in _ES_KEYS}

    x_fit = train_rows.filter(pl.col(era_col).is_in(fit_eras)) \
        .select(feature_cols).cast(pl.Float32).to_pandas()
    x_val = val_rows.select(feature_cols).cast(pl.Float32).to_pandas()
    x_hold = None
    if holdout_eras:
        hold_rows = train_rows.filter(pl.col(era_col).is_in(holdout_eras))
        x_hold = hold_rows.select(feature_cols).cast(pl.Float32).to_pandas()

    component_preds: dict[str, np.ndarray] = {}
    for index, target in enumerate(targets):
        if target not in train.columns:
            raise ValueError(f"missing target column: {target!r}")
        y = train_rows.filter(pl.col(era_col).is_in(fit_eras)) \
            .get_column(target).cast(pl.Float64).to_numpy()
        mask = np.isfinite(y)
        if mask.sum() < 2:
            raise ValueError(
                f"target {target!r} has fewer than 2 finite train rows after purge"
            )
        extra = (
            {"early_stopping_rounds": int(es_rounds)} if es_rounds is not None else None
        )
        model = construct_tree_model(
            "xgboost", model_params, seed=seed + index,
            n_features=len(feature_cols), device="cpu", extra_params=extra,
        )
        if x_hold is not None:
            y_hold = train_rows.filter(pl.col(era_col).is_in(holdout_eras)) \
                .get_column(target).cast(pl.Float64).to_numpy()
            mask_h = np.isfinite(y_hold)
            model.fit(
                x_fit[mask], y[mask],
                eval_set=[(x_hold[mask_h], y_hold[mask_h])],
                verbose=False,
            )
        else:
            model.fit(x_fit[mask], y[mask])
        component_preds[target] = np.asarray(model.predict(x_val), dtype=float)

    val_index = val_rows.select([era_col, id_col])
    frame = val_index.with_columns(
        [pl.Series(target, component_preds[target]) for target in targets]
    )
    if weights:
        total = sum(weights.get(t, 0.0) for t in targets)
        blend_weights = [weights.get(t, 0.0) / total for t in targets]
    else:
        blend_weights = [1.0 / len(targets)] * len(targets)
    ensembler = Ensembler()
    blended = ensembler.blend(
        Ensembler.rank_normalize(frame, pred_cols=list(targets), era_col=era_col),
        pred_cols=list(targets), weights=blend_weights,
        era_col=era_col, out_col=pred_col,
    )
    gaussianized = Ensembler.rank_normalize(
        blended, pred_cols=[pred_col], era_col=era_col
    )
    out = gaussianized.select([era_col, id_col, pred_col]).sort([era_col, id_col])
    if float(neutralization) > 0.0:
        neutralizer_cols = list(feature_cols)
        with_features = out.join(
            val.select([era_col, id_col, *neutralizer_cols]),
            on=[era_col, id_col], how="inner",
        )
        out = NeutralizationEngine().neutralize(
            with_features, pred_col=pred_col, feature_cols=neutralizer_cols,
            era_col=era_col, proportion=float(neutralization),
        ).select([era_col, id_col, pred_col]).sort([era_col, id_col])
    return out


_VALID_MLP_PARAM_KEYS: tuple[str, ...] = (
    "hidden_layer_sizes", "activation", "solver", "alpha", "learning_rate_init",
    "batch_size", "max_iter", "early_stopping", "n_iter_no_change",
    "validation_fraction",
)


def generate_mlp_predictions(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    target: str,
    feature_cols: Sequence[str],
    params: Mapping[str, Any],
    seed: int,
    neutralization: float = 0.0,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Fleet MLP (sklearn MLPRegressor): standardized features, fixed seed.

    Optional ``neutralization`` (proportion in (0, 1]) applies the framework
    ``NeutralizationEngine`` to the final rank-gaussianized output over all
    ``feature_cols``.
    """
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")
    unknown = sorted(set(params) - set(_VALID_MLP_PARAM_KEYS))
    if unknown:
        raise ValueError(f"unknown mlp param keys: {unknown}")
    trimmed_train_eras, _ = train_validation_purged_split(
        train.get_column(era_col).unique().to_list(),
        val.get_column(era_col).unique().to_list(),
        purge_eras=purge_eras,
    )
    train_rows = train.filter(pl.col(era_col).is_in(trimmed_train_eras))
    val_rows = val.sort([era_col, id_col])
    if target not in train.columns:
        raise ValueError(f"missing target column: {target!r}")
    missing_feats = [c for c in feature_cols if c not in train.columns or c not in val.columns]
    if missing_feats:
        raise ValueError(f"missing feature columns: {missing_feats}")

    x_train = train_rows.select(feature_cols).cast(pl.Float32).to_numpy(writable=True)
    x_val = val_rows.select(feature_cols).cast(pl.Float32).to_numpy(writable=True)
    y = train_rows.get_column(target).cast(pl.Float64).to_numpy()
    mask = np.isfinite(y)
    if mask.sum() < 2:
        raise ValueError(
            f"target {target!r} has fewer than 2 finite train rows after purge"
        )
    x_train, x_val = _standardize_feature_block(x_train, x_val)

    model = MLPRegressor(**dict(params), random_state=seed)
    model.fit(x_train[mask], y[mask])
    raw = np.asarray(model.predict(x_val), dtype=float)

    frame = val_rows.select([era_col, id_col]).with_columns(pl.Series(pred_col, raw))
    blended = Ensembler().blend(
        Ensembler.rank_normalize(frame, pred_cols=[pred_col], era_col=era_col),
        pred_cols=[pred_col], weights=[1.0], era_col=era_col, out_col=pred_col,
    )
    out = blended.select([era_col, id_col, pred_col]).sort([era_col, id_col])
    if float(neutralization) > 0.0:
        neutralizer_cols = list(feature_cols)
        with_features = out.join(
            val.select([era_col, id_col, *neutralizer_cols]),
            on=[era_col, id_col], how="inner",
        )
        out = NeutralizationEngine().neutralize(
            with_features, pred_col=pred_col, feature_cols=neutralizer_cols,
            era_col=era_col, proportion=float(neutralization),
        ).select([era_col, id_col, pred_col]).sort([era_col, id_col])
    return out


def _stack_partitions(
    trimmed: Sequence[str],
    *,
    meta_tail_pct: float,
    specialists: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Split purged train eras into specialist-train and meta-tail partitions.

    The boundary gets a horizon-aware purge buffer mirroring the splitter
    convention: 16 eras when any 60D specialist is present, else 8.
    """
    if not 0.0 < float(meta_tail_pct) < 1.0:
        raise ValueError(f"meta_tail_pct must be in (0, 1), got {meta_tail_pct!r}")
    eras = list(trimmed)
    n_meta = max(1, int(round(float(meta_tail_pct) * len(eras))))
    stack_purge = 16 if any(str(t).endswith("_60") for t in specialists) else 8
    if len(eras) - n_meta - stack_purge < 2:
        raise ValueError(
            "not enough train eras for stack split: "
            f"eras={len(eras)}, meta_tail={n_meta}, purge={stack_purge}"
        )
    meta = eras[-n_meta:]
    spec = eras[: len(eras) - n_meta - stack_purge]
    return spec, meta


def generate_ridge_stack_predictions(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    main_target: str,
    specialists: Sequence[str],
    feature_cols: Sequence[str],
    params: Mapping[str, Any],
    seed: int,
    neutralization: float = 0.0,
    val_targets: pl.DataFrame | None = None,
    benchmarks: pl.DataFrame | None = None,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Two-layer ridge stacking: per-target specialists -> meta ridge.

    Fixed mode: one Ridge per specialist (``params.alpha``), meta Ridge
    (``params.meta_alpha``) on per-era-ranked tail OOF predictions.
    Search mode (v1.5.1) delegates to :func:`_ridge_stack_search`.
    """
    mode = params.get("mode")
    if mode not in ("fixed", "search"):
        raise ValueError(f"params.mode must be 'fixed' or 'search', got {mode!r}")
    if not specialists or not feature_cols:
        raise ValueError("specialists and feature_cols must be non-empty")
    if main_target not in train.columns:
        raise ValueError(f"train missing main target column: {main_target!r}")

    trimmed_train_eras, _ = train_validation_purged_split(
        train.get_column(era_col).unique().to_list(),
        val.get_column(era_col).unique().to_list(),
        purge_eras=purge_eras,
    )
    train_rows = train.filter(pl.col(era_col).is_in(trimmed_train_eras))
    val_rows = val.sort([era_col, id_col])
    missing_feats = [c for c in feature_cols if c not in train.columns or c not in val.columns]
    if missing_feats:
        raise ValueError(f"missing feature columns: {missing_feats}")

    # v1.5.1 NaN strategy: fill features with 0.5 (neutral) before any fit.
    if bool(params.get("nan_fill", False)):
        train_rows = train_rows.with_columns(
            [pl.col(c).fill_null(0.5) for c in feature_cols]
        )
        val_rows = val_rows.with_columns(
            [pl.col(c).fill_null(0.5) for c in feature_cols]
        )

    if mode == "search":
        return _ridge_stack_search(
            train_rows, val_rows, main_target=main_target,
            specialists=list(specialists), feature_cols=list(feature_cols),
            params=params, seed=seed, neutralization=float(neutralization),
            val_targets=val_targets, benchmarks=benchmarks,
            era_col=era_col, id_col=id_col, pred_col=pred_col,
        )

    spec_eras, meta_eras = _stack_partitions(
        trimmed_train_eras,
        meta_tail_pct=float(params["meta_tail_pct"]),
        specialists=list(specialists),
    )
    spec_rows = train_rows.filter(pl.col(era_col).is_in(spec_eras))
    meta_rows = train_rows.filter(pl.col(era_col).is_in(meta_eras))
    alpha = float(params["alpha"])
    meta_alpha = float(params["meta_alpha"])

    x_spec = spec_rows.select(feature_cols).cast(pl.Float32).to_numpy()
    x_meta = meta_rows.select(feature_cols).cast(pl.Float32).to_numpy()
    x_val = val_rows.select(feature_cols).cast(pl.Float32).to_numpy()

    meta_pred_frames: list[pl.DataFrame] = []
    val_pred_frames: list[pl.DataFrame] = []
    for target in specialists:
        if target not in train.columns:
            raise ValueError(f"train missing specialist target column: {target!r}")
        y = spec_rows.get_column(target).cast(pl.Float64).to_numpy()
        mask = np.isfinite(y)
        if mask.sum() < 2:
            raise ValueError(f"specialist {target!r} has <2 finite train rows")
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=seed)
        model.fit(x_spec[mask], y[mask])
        meta_pred_frames.append(
            meta_rows.select([era_col, id_col]).with_columns(
                pl.Series(target, np.asarray(model.predict(x_meta), dtype=float))
            )
        )
        val_pred_frames.append(
            val_rows.select([era_col, id_col]).with_columns(
                pl.Series(target, np.asarray(model.predict(x_val), dtype=float))
            )
        )

    meta_ranked = meta_pred_frames[0]
    for part in meta_pred_frames[1:]:
        meta_ranked = meta_ranked.join(part, on=[era_col, id_col], how="inner")
    meta_ranked = Ensembler.rank_normalize(
        meta_ranked, pred_cols=list(specialists), era_col=era_col
    )
    val_ranked = val_pred_frames[0]
    for part in val_pred_frames[1:]:
        val_ranked = val_ranked.join(part, on=[era_col, id_col], how="inner")
    val_ranked = Ensembler.rank_normalize(
        val_ranked, pred_cols=list(specialists), era_col=era_col
    )

    meta_y = meta_rows.select([era_col, id_col, main_target]).drop_nulls()
    meta_X = meta_ranked.join(meta_y, on=[era_col, id_col], how="inner")
    if meta_X.height < 2:
        raise ValueError("fewer than 2 aligned meta-train rows")
    meta_model = Ridge(alpha=meta_alpha, fit_intercept=True, random_state=seed)
    meta_model.fit(
        meta_X.select(specialists).cast(pl.Float32).to_numpy(),
        meta_X.get_column(main_target).cast(pl.Float64).to_numpy(),
    )
    raw = np.asarray(
        meta_model.predict(val_ranked.select(specialists).cast(pl.Float32).to_numpy()),
        dtype=float,
    )
    frame = val_ranked.select([era_col, id_col]).with_columns(pl.Series(pred_col, raw))
    out = Ensembler().blend(
        Ensembler.rank_normalize(frame, pred_cols=[pred_col], era_col=era_col),
        pred_cols=[pred_col], weights=[1.0], era_col=era_col, out_col=pred_col,
    ).select([era_col, id_col, pred_col]).sort([era_col, id_col])

    if float(neutralization) > 0.0:
        with_features = out.join(
            val.select([era_col, id_col, *feature_cols]),
            on=[era_col, id_col], how="inner",
        )
        out = NeutralizationEngine().neutralize(
            with_features, pred_col=pred_col, feature_cols=list(feature_cols),
            era_col=era_col, proportion=float(neutralization),
        ).select([era_col, id_col, pred_col]).sort([era_col, id_col])
    return out


def _era_sharpe(preds: np.ndarray, eras: Sequence[str], target: np.ndarray) -> float:
    """Mean/std(ddof=0) of per-era Pearson CORR (notebook `_compute_era_sharpe`)."""
    frame = pl.DataFrame(
        {"prediction": preds, "era": list(eras), "target": target}
    ).drop_nulls()
    era_corrs: list[float] = []
    for _, era_frame in frame.group_by("era", maintain_order=True):
        if era_frame["prediction"].n_unique() < 2 or era_frame["target"].n_unique() < 2:
            continue
        p = era_frame.get_column("prediction").to_numpy()
        t = era_frame.get_column("target").to_numpy()
        era_corrs.append(float(np.corrcoef(p, t)[0, 1]))
    if not era_corrs:
        return -np.inf
    arr = np.asarray(era_corrs, dtype=float)
    return float(arr.mean() / (arr.std(ddof=0) + 1.0e-12))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return 0.0
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _rank_values_per_era(values: np.ndarray, eras: Sequence[str]) -> np.ndarray:
    """Per-era rank-gaussianize a raw vector (Ensembler semantics)."""
    frame = pl.DataFrame({"__v": values, "era": list(eras)})
    ranked = Ensembler.rank_normalize(frame, pred_cols=["__v"], era_col="era")
    return ranked.get_column("__v").to_numpy()


def _per_era_corrs(values: np.ndarray, eras: Sequence[str], target: np.ndarray) -> np.ndarray:
    frame = pl.DataFrame(
        {"__v": values, "era": list(eras), "target": target}
    ).drop_nulls()
    out: list[float] = []
    for _, era_frame in frame.group_by("era", maintain_order=True):
        if era_frame["__v"].n_unique() < 2 or era_frame["target"].n_unique() < 2:
            continue
        out.append(float(np.corrcoef(
            era_frame.get_column("__v").to_numpy(),
            era_frame.get_column("target").to_numpy(),
        )[0, 1]))
    return np.asarray(out, dtype=float)


def _ridge_stack_search(
    train_rows: pl.DataFrame,
    val_rows: pl.DataFrame,
    *,
    main_target: str,
    specialists: list[str],
    feature_cols: list[str],
    params: Mapping[str, Any],
    seed: int,
    neutralization: float,
    val_targets: pl.DataFrame | None,
    benchmarks: pl.DataFrame | None,
    era_col: str,
    id_col: str,
    pred_col: str,
) -> pl.DataFrame:
    """v1.5.1-style config-driven specialist/meta search (selection-biased).

    Candidate selection uses validation (as the notebook did) — the runner
    flags the resulting scorecard with ``selection_bias: true``.
    """
    if val_targets is None:
        raise ValueError("search mode requires val_targets (selection uses validation)")
    if benchmarks is None:
        raise ValueError("search mode requires benchmarks (decorr sweep)")
    benchmark_col = str(params["benchmark_col"])
    if benchmark_col not in benchmarks.columns:
        raise ValueError(f"benchmarks missing column: {benchmark_col!r}")

    # 1. Target quality filter (coverage, corr to main, priority hints, top-k).
    weights = dict(params["snnr_weights"])
    quality: list[tuple[float, float, float, str]] = []
    for target in specialists:
        if target not in train_rows.columns:
            continue
        series = train_rows.get_column(target)
        coverage = float(series.drop_nulls().len() / max(1, series.len()))
        if coverage < float(params["min_coverage"]):
            continue
        aligned = train_rows.select([main_target, target]).drop_nulls()
        corr = _pearson(
            aligned.get_column(main_target).to_numpy(),
            aligned.get_column(target).to_numpy(),
        )
        if abs(corr) < float(params["min_abs_main_corr"]):
            continue
        hint_bonus = 1.0 if target in list(params.get("priority_hints", [])) else 0.0
        quality.append((hint_bonus, float(weights.get(target, 0.0)), corr, target))
    quality.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
    selected = [row[3] for row in quality[: int(params["top_k"])]]
    if not selected:
        raise ValueError("no auxiliary targets survive the quality filter")

    spec_eras, meta_eras = _stack_partitions(
        sorted(train_rows.get_column(era_col).unique().to_list()),
        meta_tail_pct=float(params["meta_tail_pct"]),
        specialists=selected,
    )
    spec_rows = train_rows.filter(pl.col(era_col).is_in(spec_eras))
    meta_rows = train_rows.filter(pl.col(era_col).is_in(meta_eras))
    x_spec = spec_rows.select(feature_cols).cast(pl.Float32).to_numpy()
    x_meta = meta_rows.select(feature_cols).cast(pl.Float32).to_numpy()
    x_val = val_rows.select(feature_cols).cast(pl.Float32).to_numpy()

    # 2. Specialist alpha search: Sharpe on the meta tail.
    meta_main = meta_rows.get_column(main_target).cast(pl.Float64).to_numpy()
    meta_era_list = meta_rows.get_column(era_col).to_list()
    kept: list[tuple[str, float, object]] = []  # (target, alpha, model)
    for target in selected:
        y = spec_rows.get_column(target).cast(pl.Float64).to_numpy()
        mask = np.isfinite(y)
        if mask.sum() < 2:
            continue
        best: tuple[float, float, object] | None = None
        for alpha in params["specialist_alpha_grid"]:
            model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=seed)
            model.fit(x_spec[mask], y[mask])
            meta_raw = np.asarray(model.predict(x_meta), dtype=float)
            meta_ranked = _rank_values_per_era(meta_raw, meta_era_list)
            sharpe = _era_sharpe(meta_ranked, meta_era_list, meta_main)
            if best is None or sharpe > best[0]:
                best = (sharpe, float(alpha), model)
        if best is not None and best[0] >= float(params["specialist_sharpe_floor"]):
            kept.append((target, best[1], best[2]))
    if len(kept) < int(params["min_specialists"]):
        raise ValueError(
            f"only {len(kept)} specialists survive the Sharpe floor, "
            f"need >= min_specialists={params['min_specialists']}"
        )

    # 3. Meta features: per-era-ranked specialist predictions on meta tail + val.
    meta_X = meta_rows.select([era_col, id_col])
    val_X = val_rows.select([era_col, id_col])
    for target, _, model in kept:
        meta_X = meta_X.with_columns(
            pl.Series(target, np.asarray(model.predict(x_meta), dtype=float))
        )
        val_X = val_X.with_columns(
            pl.Series(target, np.asarray(model.predict(x_val), dtype=float))
        )
    kept_cols = [t for t, _, _ in kept]
    meta_X = Ensembler.rank_normalize(meta_X, pred_cols=kept_cols, era_col=era_col)
    val_X = Ensembler.rank_normalize(val_X, pred_cols=kept_cols, era_col=era_col)
    meta_y = meta_rows.select([era_col, id_col, main_target]).drop_nulls()
    meta_fit = meta_X.join(meta_y, on=[era_col, id_col], how="inner")
    if meta_fit.height < 2:
        raise ValueError("fewer than 2 aligned meta-train rows")
    meta_fit_X = meta_fit.select(kept_cols).cast(pl.Float32).to_numpy()
    meta_fit_y = meta_fit.get_column(main_target).cast(pl.Float64).to_numpy()
    val_meta_X = val_X.select(kept_cols).cast(pl.Float32).to_numpy()

    # 4. Meta candidates: non-negative ridge grid + shallow LGBM (internal es split).
    candidates: dict[str, np.ndarray] = {}
    for alpha in params["meta_alpha_grid"]:
        try:
            model = Ridge(alpha=float(alpha), positive=True, random_state=seed)
            model.fit(meta_fit_X, meta_fit_y)
        except TypeError:
            model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=seed)
            model.fit(meta_fit_X, meta_fit_y)
        candidates[f"ridge|alpha={float(alpha):.4g}"] = np.asarray(
            model.predict(val_meta_X), dtype=float
        )
    lgbm_params = dict(params["meta_lgbm_params"])
    es_rounds = int(lgbm_params.pop("early_stopping_rounds", 50))
    valid_tail_pct = float(lgbm_params.pop("valid_tail_pct", 0.2))
    min_valid_eras = int(lgbm_params.pop("min_valid_eras", 5))
    meta_era_sorted = meta_fit.get_column(era_col).unique().sort().to_list()
    if len(meta_era_sorted) > 1:
        n_valid = max(min_valid_eras, int(round(len(meta_era_sorted) * valid_tail_pct)))
        n_valid = min(n_valid, max(1, len(meta_era_sorted) - 1))
        valid_eras = set(meta_era_sorted[-n_valid:])
        is_valid = meta_fit.get_column(era_col).is_in(list(valid_eras)).to_numpy()
    else:
        is_valid = np.zeros(meta_fit.height, dtype=bool)
    lgbm_model = construct_tree_model(
        "lightgbm", lgbm_params, seed=seed,
        n_features=len(kept_cols), device="cpu",
    )
    if is_valid.any():
        lgbm_model.fit(
            meta_fit_X[~is_valid], meta_fit_y[~is_valid],
            eval_set=[(meta_fit_X[is_valid], meta_fit_y[is_valid])],
            callbacks=[early_stopping(es_rounds, verbose=False)],
        )
    else:
        lgbm_model.fit(meta_fit_X, meta_fit_y)
    candidates["lgbm"] = np.asarray(lgbm_model.predict(val_meta_X), dtype=float)

    # 5. Post-processing sweeps: benchmark decorr x neutralization; selection on
    #    validation mean CORR vs the main target (documented selection bias).
    val_main = val_targets.join(
        val_rows.select([era_col, id_col]), on=[era_col, id_col], how="inner"
    ).sort([era_col, id_col])
    val_main_y = val_main.get_column(main_target).cast(pl.Float64).to_numpy()
    val_era_list = val_main.get_column(era_col).to_list()
    bench_sorted = benchmarks.sort([era_col, id_col])
    bench_ranked = _rank_values_per_era(
        bench_sorted.get_column(benchmark_col).to_numpy(),
        bench_sorted.get_column(era_col).to_list(),
    )
    best: tuple[float, str, float, float, np.ndarray] | None = None
    for key in sorted(candidates):  # deterministic iteration order
        base = candidates[key]
        for decorr in params["decorr_grid"]:
            decorrelated = base - float(decorr) * bench_ranked
            for neu in params["neutralization_grid"]:
                ranked = _rank_values_per_era(decorrelated, val_era_list)
                era_corrs = _per_era_corrs(ranked, val_era_list, val_main_y)
                score = float(np.mean(era_corrs)) if era_corrs.size else -np.inf
                if best is None or score > best[0]:
                    best = (score, key, float(decorr), float(neu), decorrelated)
    if best is None:
        raise ValueError("no post-processing candidate survived")
    selected_raw = best[4]

    frame = val_main.select([era_col, id_col]).with_columns(
        pl.Series(pred_col, selected_raw)
    )
    if best[3] > 0.0:
        with_features = frame.join(
            val_rows.select([era_col, id_col, *feature_cols]),
            on=[era_col, id_col], how="inner",
        )
        frame = NeutralizationEngine().neutralize(
            with_features, pred_col=pred_col, feature_cols=feature_cols,
            era_col=era_col, proportion=best[3],
        ).select([era_col, id_col, pred_col])
    out = Ensembler().blend(
        Ensembler.rank_normalize(frame, pred_cols=[pred_col], era_col=era_col),
        pred_cols=[pred_col], weights=[1.0], era_col=era_col, out_col=pred_col,
    )
    return out.select([era_col, id_col, pred_col]).sort([era_col, id_col])


@dataclasses.dataclass(frozen=True)
class FleetResult:
    scorecards: Mapping[str, MetricScorecard]
    sources: Mapping[str, str]
    placements: Mapping[str, str]
    gate_verdicts: Mapping[str, Mapping[str, bool | None]]
    selection_bias: Mapping[str, bool]


def fleet_placement(corr: float, rungs: Mapping[int, float]) -> str:
    """Place a measured CORR against the per-tier max-corr ladder rungs."""
    if not rungs:
        raise ValueError("rungs must be non-empty")
    if (
        not isinstance(corr, (int, float)) or isinstance(corr, bool)
        or not -1.0 <= float(corr) <= 1.0
    ):
        raise ValueError(f"corr must be a float in [-1, 1], got {corr!r}")
    ordered = sorted(rungs.items())
    if corr < ordered[0][1]:
        return "below tier 0"
    if corr > ordered[-1][1]:
        return f"above tier {ordered[-1][0]}"
    for index in range(len(ordered) - 1):
        tier_low, value_low = ordered[index]
        tier_high, value_high = ordered[index + 1]
        if value_low <= corr < value_high:
            return f"tier{tier_low}..tier{tier_high}"
    return f"tier{ordered[-1][0]}"


class BenchmarkFleet:
    """Untiered fleet of benchmark models (spec 2026-08-19-benchmark-fleet-design)."""

    def __init__(
        self,
        *,
        spec: tuple[FleetCellConfig, ...],
        data: BenchmarkData,
        seed: int = DEFAULT_BENCHMARK_SEED,
        horizon: str = "20D",
        n_boot: int = 1000,
        min_overlap_eras: int = 20,
        fast_mode: bool = False,
    ) -> None:
        if not spec:
            raise ValueError("fleet spec has no cells")
        self._spec = spec
        self._data = data
        self._seed = int(seed)
        self._horizon = horizon
        self._n_boot = int(n_boot)
        self._min_overlap_eras = int(min_overlap_eras)
        self._fast_mode = bool(fast_mode)
        self._schema_cols = pl.read_parquet_schema(data.validation_path).names()
        self._target_cols = ["era", "id"] + [
            c for c in self._schema_cols if c == "target" or c.startswith("target_")
        ]

    def _feature_cols(self, cell: FleetCellConfig) -> list[str]:
        return resolve_benchmark_feature_cols(
            self._data.features_json, cell.input_space, self._schema_cols
        )

    def _cell_params(self, cell: FleetCellConfig) -> dict[str, Any]:
        params = dict(cell.params)
        if self._fast_mode and cell.fast_mode_params:
            params.update(dict(cell.fast_mode_params))
        return params

    def _predictions_for_cell(
        self, cell: FleetCellConfig
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        feature_cols = self._feature_cols(cell)
        params = self._cell_params(cell)
        val_id = pl.read_parquet(self._data.validation_path, columns=["era", "id"])

        if cell.input_space == "none":
            small_cols = resolve_benchmark_feature_cols(
                self._data.features_json, "small", self._schema_cols
            )
            val_features = pl.read_parquet(
                self._data.validation_path, columns=["era", "id", *small_cols]
            )
        else:
            val_features = pl.read_parquet(
                self._data.validation_path,
                columns=["era", "id", *feature_cols],
            )

        if cell.model_kind == "target_lag_mean":
            train_targets = pl.read_parquet(
                self._data.train_path, columns=["era", cell.targets[0]]
            )
            preds = generate_lagged_target_predictions(
                train_targets, val_id, target=cell.targets[0],
                window=int(params.get("window", 1)),
            )
            return preds, val_features

        if cell.model_kind == "ridge_stack":
            stack_targets = [
                str(params["main_target"]), *map(str, params["specialists"])
            ]
            train = pl.read_parquet(
                self._data.train_path,
                columns=["era", "id", *feature_cols, *stack_targets],
            )
            is_search = params.get("mode") == "search"
            val_cols = ["era", "id", *feature_cols]
            if is_search:
                val_cols.append(str(params["main_target"]))
            val = pl.read_parquet(
                self._data.validation_path, columns=val_cols
            )
            preds = generate_ridge_stack_predictions(
                train, val,
                main_target=str(params["main_target"]),
                specialists=[str(t) for t in params["specialists"]],
                feature_cols=feature_cols, params=params, seed=cell.seed,
                neutralization=float(cell.neutralization or 0.0),
                val_targets=(
                    val.select(["era", "id", str(params["main_target"])])
                    if is_search else None
                ),
                benchmarks=self._data.benchmarks,
            )
            return preds, val_features

        train = pl.read_parquet(
            self._data.train_path,
            columns=["era", "id", *feature_cols, *cell.targets],
        )
        val = pl.read_parquet(
            self._data.validation_path, columns=["era", "id", *feature_cols]
        )
        if cell.model_kind == "lightgbm":
            preds = generate_fleet_lightgbm_predictions(
                train, val, targets=list(cell.targets), feature_cols=feature_cols,
                params=params, seed=cell.seed,
                neutralization=float(cell.neutralization or 0.0),
                neutralizer_selection=cell.neutralizer_selection,
                neutralizer_count=cell.neutralizer_count,
            )
        elif cell.model_kind == "xgboost":
            preds = generate_fleet_xgb_predictions(
                train, val, targets=list(cell.targets), feature_cols=feature_cols,
                params=params, seed=cell.seed,
                neutralization=float(cell.neutralization or 0.0),
                target_weights=(
                    dict(cell.target_weights) if cell.target_weights else None
                ),
            )
        elif cell.model_kind == "mlp":
            preds = generate_mlp_predictions(
                train, val, target=cell.targets[0], feature_cols=feature_cols,
                params=params, seed=cell.seed,
                neutralization=float(cell.neutralization or 0.0),
            )
        else:
            raise ValueError(f"Unsupported fleet model kind: {cell.model_kind!r}")
        return preds, val_features

    def run(
        self,
        *,
        tier_rungs: Mapping[int, float],
        gate: Tier4GateConfig | None,
    ) -> FleetResult:
        scorecards: dict[str, MetricScorecard] = {}
        sources: dict[str, str] = {}
        selection_bias: dict[str, bool] = {}
        val_targets = pl.read_parquet(
            self._data.validation_path, columns=self._target_cols
        )
        for cell in self._spec:
            logger.info("[fleet] %s (kind=%s)", cell.benchmark_id, cell.model_kind)
            preds, val_features = self._predictions_for_cell(cell)
            scorecards[cell.benchmark_id] = evaluate_model(
                preds,
                meta_model=self._data.meta_model,
                benchmarks=self._data.benchmarks,
                features=val_features,
                targets=val_targets,
                n_trials=1,
                seed=cell.seed,
                horizon=self._horizon,
                main_target="target",
                benchmark_col=None,
                n_boot=self._n_boot,
                min_overlap_eras=self._min_overlap_eras,
                model_id=cell.benchmark_id,
            )
            sources[cell.benchmark_id] = cell.source
            selection_bias[cell.benchmark_id] = (
                cell.model_kind == "ridge_stack"
                and cell.params.get("mode") == "search"
            )
            if cell.anchors:
                measured = float(scorecards[cell.benchmark_id].corr.value)
                for key, anchor in cell.anchors.items():
                    logger.info(
                        "    anchor %s=%.4f (measured=%.6f)",
                        key, float(anchor), measured,
                    )
        placements = (
            {
                mid: fleet_placement(float(card.corr.value), tier_rungs)
                for mid, card in scorecards.items()
            }
            if tier_rungs
            else {}
        )
        gate_verdicts: dict[str, Mapping[str, bool | None]] = {}
        for mid, card in scorecards.items():
            gate_verdicts[mid] = (
                tier4_gate_verdict(card, gate) if gate is not None else {}
            )
        return FleetResult(
            scorecards=scorecards,
            sources=sources,
            placements=placements,
            gate_verdicts=gate_verdicts,
            selection_bias=selection_bias,
        )


def load_tier_rungs_from_csv(path: str | Path) -> dict[int, float]:
    """Per-tier max mean-CORR rungs from a hierarchy scorecard CSV.

    ``benchmark_runner.py --only-fleet`` sources placement rungs from the
    last completed hierarchy run instead of a live one.
    """
    frame = pl.read_csv(Path(path))
    for col in ("tier", "corr"):
        if col not in frame.columns:
            raise ValueError(f"hierarchy scorecard CSV missing column: {col!r}")
    rungs: dict[int, float] = {}
    tiers = sorted(frame.get_column("tier").unique().to_list())
    for tier in tiers:
        rungs[int(tier)] = float(
            frame.filter(pl.col("tier") == tier).get_column("corr").max()
        )
    return rungs


def select_fleet_cells(
    cells: tuple[FleetCellConfig, ...], requested: tuple[str, ...]
) -> tuple[FleetCellConfig, ...]:
    """Filter fleet cells by benchmark_id; unknown ids raise (fail loud)."""
    if not requested:
        return cells
    by_id = {cell.benchmark_id: cell for cell in cells}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"unknown fleet ids: {unknown}")
    return tuple(by_id[cell_id] for cell_id in requested)


def fleet_frame(result: FleetResult) -> pl.DataFrame:
    """Scorecard rows + placement/selection-bias/gate-verdict columns."""
    frame = scorecards_to_frame(result.scorecards)
    extra = pl.DataFrame({
        "model_id": list(result.scorecards.keys()),
        "source": [result.sources[mid] for mid in result.scorecards],
        "placement": [result.placements[mid] for mid in result.scorecards],
        "selection_bias": [
            result.selection_bias[mid] for mid in result.scorecards
        ],
    })
    out = frame.join(extra, on="model_id", how="left")
    verdict_fields = (
        "corr", "corr_sharpe_ac", "fnc", "deflated_sharpe",
        "gain_to_pain_ratio", "cagr_1y", "turnover_mean",
    )
    # Gate series must follow the joined frame's row order, which
    # ``scorecards_to_frame`` sorts by model_id.
    sorted_ids = sorted(result.scorecards)
    for field in verdict_fields:
        out = out.with_columns(
            pl.Series(
                f"gate_{field}",
                [result.gate_verdicts[mid].get(field) for mid in sorted_ids],
            )
        )
    return out.sort("model_id")


def write_fleet_csv(result: FleetResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"output_path must be a .csv file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fleet_frame(result).write_csv(path)
    return path
