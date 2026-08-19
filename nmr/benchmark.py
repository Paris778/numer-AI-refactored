"""E6 benchmark infrastructure: config-driven 5-tier benchmark hierarchy.

The 5-tier ladder ("the line in the sand") is declared in YAML config
files, generated deterministically by tier (nulls, ridge, shallow trees,
canonical baselines, and the tier-4 reference), scored against the shared
evaluation suite, and gated by hard thresholds (tier-0 null floor, tier-4
production gate, cross-tier monotonicity). Canonical scorecard bytes
support cross-process determinism checks.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import polars as pl
import yaml
from sklearn.linear_model import Ridge

from nmr.ensemble import Ensembler
from nmr.features import resolve_feature_sets, resolve_small_feature_set
from nmr.models import construct_tree_model
from nmr.risk import NeutralizationEngine
from nmr.scorecard import MetricScorecard, evaluate_model

logger = logging.getLogger("nmr.benchmark")

__all__ = [
    "BenchmarkCellConfig",
    "BenchmarkData",
    "BenchmarkFileConfig",
    "BenchmarkHierarchy",
    "BenchmarkHierarchyResult",
    "BenchmarkSuiteSpec",
    "Tier4GateConfig",
    "VALID_BENCHMARK_TIERS",
    "assert_hierarchy_monotone",
    "assert_tier0_null_floor",
    "assert_tier4_gate",
    "canonical_scorecards_bytes",
    "gate_report_frame",
    "generate_canonical_predictions",
    "generate_null_predictions",
    "generate_ridge_predictions",
    "generate_tree_predictions",
    "hierarchy_frame",
    "load_benchmark_data",
    "load_benchmark_file",
    "load_benchmark_suite_config",
    "resolve_benchmark_feature_cols",
    "score_benchmark_column",
    "scorecards_sha256",
    "scorecards_to_frame",
    "tier4_gate_verdict",
    "tier_max_corrs",
    "train_validation_purged_split",
    "write_scorecards_csv",
]

# ---------------------------------------------------------------------------
# 5-tier benchmark hierarchy: config schema (spec:
# docs/superpowers/specs/2026-08-15-benchmark-hierarchy-design.md)
# ---------------------------------------------------------------------------

VALID_BENCHMARK_TIERS: tuple[int, ...] = (0, 1, 2, 3, 4)
VALID_INPUT_SPACES: tuple[str, ...] = ("none", "small", "medium")
VALID_BENCHMARK_MODEL_KINDS: tuple[str, ...] = (
    "null_constant_05",
    "null_uniform_rand",
    "null_gaussian_rand",
    "null_feature_mean",
    "ridge",
    "lightgbm",
    "xgboost",
)
NULL_KINDS: tuple[str, ...] = (
    "null_constant_05",
    "null_uniform_rand",
    "null_gaussian_rand",
    "null_feature_mean",
)
NULL_FLOOR_KINDS: tuple[str, ...] = (
    "null_constant_05",
    "null_uniform_rand",
    "null_gaussian_rand",
)
DEFAULT_BENCHMARK_SEED: int = 42
DEFAULT_BENCHMARK_PURGE_ERAS: int = 8


def _reject_unknown_keys(cls: type, data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError(
            f"{cls.__name__} section must be a mapping, got {type(data).__name__}"
        )
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")


def _freeze_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    out = dict(value)
    for key in out:
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings, got {key!r}")
    return MappingProxyType(out)


@dataclasses.dataclass(frozen=True)
class Tier4GateConfig:
    corr_min: float
    corr_sharpe_ac_min: float
    fnc_min: float
    deflated_sharpe_min: float
    gain_to_pain_min: float
    cagr_min: float
    turnover_max: float

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Tier4GateConfig.{field.name} must be numeric, got {value!r}"
                )
            if not float(value) == float(value):  # NaN check
                raise ValueError(f"Tier4GateConfig.{field.name} must be finite")
        if not (-1.0 <= self.corr_min <= 1.0):
            raise ValueError(f"corr_min out of range: {self.corr_min!r}")
        if self.turnover_max < 0.0:
            raise ValueError(f"turnover_max must be >= 0: {self.turnover_max!r}")


@dataclasses.dataclass(frozen=True)
class BenchmarkCellConfig:
    benchmark_id: str
    input_space: str
    model_kind: str
    tier: int
    targets: tuple[str, ...] = ("target",)
    params: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    seed: int = DEFAULT_BENCHMARK_SEED
    neutralization: float = 0.0
    anchors: Mapping[str, float] | None = None
    fast_mode_params: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.benchmark_id or not isinstance(self.benchmark_id, str):
            raise ValueError(f"benchmark_id must be a non-empty string: {self.benchmark_id!r}")
        if self.tier not in VALID_BENCHMARK_TIERS:
            raise ValueError(f"tier={self.tier!r} not in {VALID_BENCHMARK_TIERS}")
        if self.input_space not in VALID_INPUT_SPACES:
            raise ValueError(
                f"input_space={self.input_space!r} not in {VALID_INPUT_SPACES}"
            )
        if self.model_kind not in VALID_BENCHMARK_MODEL_KINDS:
            raise ValueError(
                f"model_kind={self.model_kind!r} not in {VALID_BENCHMARK_MODEL_KINDS}"
            )
        if self.model_kind == "null_feature_mean" and self.input_space != "small":
            raise ValueError(
                "null_feature_mean requires input_space='small', "
                f"got {self.input_space!r}"
            )
        if self.model_kind in NULL_KINDS and self.input_space != "none" \
                and self.model_kind != "null_feature_mean":
            raise ValueError(
                f"{self.model_kind} requires input_space='none', "
                f"got {self.input_space!r}"
            )
        if not isinstance(self.targets, tuple) or not self.targets:
            raise ValueError("targets must be a non-empty tuple")
        if not all(isinstance(t, str) and t for t in self.targets):
            raise ValueError(f"targets must be non-empty strings: {self.targets!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        if not 0.0 <= float(self.neutralization) <= 1.0:
            raise ValueError(
                f"neutralization must be in [0, 1], got {self.neutralization!r}"
            )
        object.__setattr__(self, "params", _freeze_mapping(self.params, name="params"))
        if self.anchors is not None:
            anchors = _freeze_mapping(self.anchors, name="anchors")
            for key, value in anchors.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"anchor {key!r} must be numeric, got {value!r}")
            object.__setattr__(self, "anchors", anchors)
        if self.fast_mode_params is not None:
            object.__setattr__(
                self,
                "fast_mode_params",
                _freeze_mapping(self.fast_mode_params, name="fast_mode_params"),
            )


@dataclasses.dataclass(frozen=True)
class BenchmarkFileConfig:
    tier: int
    cells: tuple[BenchmarkCellConfig, ...] = ()
    reference_column: str | None = None
    gate: Tier4GateConfig | None = None

    def __post_init__(self) -> None:
        if self.tier not in VALID_BENCHMARK_TIERS:
            raise ValueError(
                f"tier={self.tier!r} not in {VALID_BENCHMARK_TIERS}"
            )
        if self.tier == 4:
            if self.gate is None:
                raise ValueError("tier 4 config requires a 'gate' section")
            if not self.reference_column:
                raise ValueError("tier 4 config requires a non-empty reference_column")
        else:
            if not self.cells:
                raise ValueError(
                    f"tier {self.tier} config requires non-empty cells"
                )
            if self.gate is not None:
                raise ValueError(f"gate section only allowed for tier 4, got tier {self.tier}")
        ids = [cell.benchmark_id for cell in self.cells]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate benchmark ids in file: {ids}")


def _build_benchmark_cell(data: Any, tier: int) -> BenchmarkCellConfig:
    if not isinstance(data, dict):
        raise ValueError(
            f"benchmark cell must be a mapping, got {type(data).__name__}"
        )
    if "benchmark_id" not in data:
        raise ValueError(f"benchmark cell missing 'benchmark_id': {data!r}")
    if "tier" in data and int(data["tier"]) != int(tier):
        raise ValueError(
            f"cell tier {data['tier']!r} conflicts with file tier {tier!r}"
        )
    data["tier"] = int(tier)
    _reject_unknown_keys(BenchmarkCellConfig, data)
    if isinstance(data.get("targets"), list):
        data["targets"] = tuple(data["targets"])
    return BenchmarkCellConfig(**data)


def load_benchmark_file(path: str | Path) -> BenchmarkFileConfig:
    """Load and validate a single benchmark tier config file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"benchmark config must be a mapping, got {type(raw).__name__}")
    _reject_unknown_keys(BenchmarkFileConfig, raw)
    if not isinstance(raw.get("cells", []), list):
        raise ValueError("cells must be a list")
    gate_raw = raw.get("gate")
    gate = None
    if gate_raw is not None:
        _reject_unknown_keys(Tier4GateConfig, gate_raw)
        gate = Tier4GateConfig(**gate_raw)
    return BenchmarkFileConfig(
        tier=int(raw["tier"]),
        cells=tuple(
            _build_benchmark_cell(c, int(raw["tier"]))
            for c in raw.get("cells", [])
        ),
        reference_column=raw.get("reference_column"),
        gate=gate,
    )


@dataclasses.dataclass(frozen=True)
class BenchmarkSuiteSpec:
    cells: tuple[BenchmarkCellConfig, ...]
    gate: Tier4GateConfig | None
    reference_column: str | None


def load_benchmark_suite_config(config_dir: str | Path) -> BenchmarkSuiteSpec:
    """Load every *.yaml file in config_dir and aggregate into a suite spec."""
    directory = Path(config_dir)
    files = sorted(p for p in directory.glob("*.yaml"))
    if not files:
        raise ValueError(f"no benchmark config files found in {directory}")
    all_cells: list[BenchmarkCellConfig] = []
    gate: Tier4GateConfig | None = None
    reference_column: str | None = None
    for path in files:
        file_cfg = load_benchmark_file(path)
        if file_cfg.gate is not None:
            if gate is not None:
                raise ValueError("multiple tier-4 gate configs found")
            gate = file_cfg.gate
            reference_column = file_cfg.reference_column
        all_cells.extend(file_cfg.cells)
    ids = [cell.benchmark_id for cell in all_cells]
    if len(set(ids)) != len(ids):
        seen = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate benchmark ids across configs: {seen}")
    all_cells.sort(key=lambda c: (c.tier, c.benchmark_id))
    return BenchmarkSuiteSpec(
        cells=tuple(all_cells),
        gate=gate,
        reference_column=reference_column,
    )


def _ordered_numeric_eras(eras: Sequence[str]) -> list[str]:
    """Dedupe, validate, and numerically sort era labels."""
    if not eras:
        raise ValueError("era universe is empty")
    mapping: dict[int, str] = {}
    for era in eras:
        if not isinstance(era, str):
            raise ValueError(
                f"Era labels must be strings, got {type(era).__name__}"
            )
        try:
            era_num = int(era)
        except ValueError as exc:
            raise ValueError(f"Non-numeric era label {era!r}") from exc
        if era_num in mapping and mapping[era_num] != era:
            raise ValueError(
                "Inconsistent zero-padding in era labels: "
                f"{mapping[era_num]!r} vs {era!r}"
            )
        mapping[era_num] = era
    labels = [mapping[num] for num in sorted(mapping)]
    widths = {len(label) for label in labels}
    if len(widths) != 1 or any(
        label != str(int(label)).zfill(len(labels[0])) for label in labels
    ):
        raise ValueError(
            "Inconsistent zero-padding in era labels: " + ", ".join(labels)
        )
    return labels


def train_validation_purged_split(
    train_eras: Sequence[str],
    val_eras: Sequence[str],
    *,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the purged train->validation era partition for benchmark fits.

    Mirrors ``PurgedEraSplitter`` invariants for the fixed one-shot split:
    the final ``purge_eras`` train eras are excluded (the purge buffer), the
    remaining train eras strictly precede validation eras, and exactly
    ``purge_eras`` eras separate the trimmed train tail from validation.
    """
    if isinstance(purge_eras, bool) or not isinstance(purge_eras, int) or purge_eras < 0:
        raise ValueError(f"purge_eras must be a non-negative int, got {purge_eras!r}")

    ordered_train = _ordered_numeric_eras(train_eras)
    ordered_val = _ordered_numeric_eras(val_eras)

    overlap = set(ordered_train) & set(ordered_val)
    if overlap:
        raise ValueError(f"train/validation era overlap: {sorted(overlap)[:5]}")

    if len(ordered_train) <= purge_eras:
        raise ValueError(
            "Not enough train eras after purge: "
            f"train={len(ordered_train)}, purge={purge_eras}"
        )

    trimmed = ordered_train[: len(ordered_train) - purge_eras]
    train_max = int(trimmed[-1])
    val_min = int(ordered_val[0])
    if train_max >= val_min:
        raise ValueError(
            "train eras must be strictly earlier than validation eras: "
            f"max(train)={train_max} >= min(val)={val_min}"
        )

    gap_width = val_min - train_max - 1
    if gap_width != purge_eras:
        raise ValueError(
            f"purge buffer is not exactly {purge_eras} eras wide: got {gap_width} "
            f"(max(train)={train_max}, min(val)={val_min})"
        )

    return tuple(trimmed), tuple(ordered_val)


def generate_null_predictions(
    prediction_index: pl.DataFrame,
    *,
    kind: str,
    seed: int,
    features: pl.DataFrame | None = None,
    feature_cols: Sequence[str] = (),
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Generate deterministic tier-0 null predictions on the prediction index."""
    if kind not in NULL_KINDS:
        raise ValueError(f"Unknown null kind {kind!r}; expected one of {NULL_KINDS}")
    missing_keys = [c for c in (era_col, id_col) if c not in prediction_index.columns]
    if missing_keys:
        raise ValueError(f"prediction_index missing required columns: {missing_keys}")

    index = (
        prediction_index.select([era_col, id_col])
        .unique()
        .sort([era_col, id_col])
    )
    n = index.height
    rng = np.random.default_rng(seed)

    if kind == "null_constant_05":
        values = np.full(n, 0.5, dtype=float)
    elif kind == "null_uniform_rand":
        values = rng.uniform(0.0, 1.0, n)
    elif kind == "null_gaussian_rand":
        values = np.clip(rng.normal(0.5, 0.15, n), 0.0, 1.0)
    else:  # null_feature_mean
        if features is None:
            raise ValueError("null_feature_mean requires a features frame")
        if not feature_cols:
            raise ValueError("null_feature_mean requires at least one feature column")
        missing_feats = [c for c in feature_cols if c not in features.columns]
        if missing_feats:
            raise ValueError(f"features missing columns: {missing_feats}")
        joined = index.join(
            features.select([era_col, id_col, *feature_cols]),
            on=[era_col, id_col],
            how="inner",
        )
        if joined.height != n:
            raise ValueError(
                f"null_feature_mean join dropped {n - joined.height} rows"
            )
        values = (
            joined.select(
                pl.mean_horizontal(
                    [pl.col(c).cast(pl.Float64, strict=False) for c in feature_cols]
                )
            )
            .to_series()
            .to_numpy()
        )

    return index.with_columns(pl.Series(pred_col, values))


def _standardize_feature_block(
    train_values: np.ndarray, val_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize with train statistics; zero-variance features -> 0.0.

    Float32 end-to-end with in-place updates: inputs are float32 arrays and are
    mutated in place (and returned) so no large float64 temporaries are created.
    Statistics are computed as float64 on the float32 block (tiny per-column
    transient), then downcast before the in-place subtract/multiply.
    """
    mu = np.mean(train_values, axis=0, dtype=np.float64)
    sigma = np.std(train_values, axis=0, dtype=np.float64)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    scale = np.where((sigma > 0.0) & np.isfinite(sigma), 1.0 / sigma, 0.0)
    mu32 = mu.astype(np.float32)
    scale32 = scale.astype(np.float32)
    np.subtract(train_values, mu32, out=train_values)
    np.multiply(train_values, scale32, out=train_values)
    np.subtract(val_values, mu32, out=val_values)
    np.multiply(val_values, scale32, out=val_values)
    return train_values, val_values


def generate_ridge_predictions(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    targets: Sequence[str],
    feature_cols: Sequence[str],
    alpha: float,
    seed: int,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Fit purged Ridge models per target and blend in rank-Gaussian domain."""
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha < 0:
        raise ValueError(f"alpha must be a non-negative number, got {alpha!r}")
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")
    if not targets:
        raise ValueError("targets must be non-empty")

    trimmed_train_eras, val_eras = train_validation_purged_split(
        train.get_column(era_col).unique().to_list(),
        val.get_column(era_col).unique().to_list(),
        purge_eras=purge_eras,
    )

    train_rows = train.filter(pl.col(era_col).is_in(trimmed_train_eras))
    val_rows = val.sort([era_col, id_col])
    missing_feats = [c for c in feature_cols if c not in train.columns or c not in val.columns]
    if missing_feats:
        raise ValueError(f"missing feature columns: {missing_feats}")

    x_train_raw = train_rows.select(feature_cols).cast(pl.Float32).to_numpy(writable=True)
    x_val_raw = val_rows.select(feature_cols).cast(pl.Float32).to_numpy(writable=True)

    # Extract targets and the val index up front, then release the polars frames
    # before standardization so peak memory stays float32-bound.
    y_by_target: dict[str, np.ndarray] = {}
    for target in targets:
        if target not in train.columns:
            raise ValueError(f"missing target column: {target!r}")
        y_by_target[target] = train_rows.get_column(target).cast(pl.Float64).to_numpy()
    val_index = val_rows.select([era_col, id_col])
    del train_rows, val_rows
    gc.collect()

    x_train, x_val = _standardize_feature_block(x_train_raw, x_val_raw)

    component_preds: dict[str, np.ndarray] = {}
    for target in targets:
        y = y_by_target[target]
        mask = np.isfinite(y)
        if mask.sum() < 2:
            raise ValueError(
                f"target {target!r} has fewer than 2 finite train rows after purge"
            )
        model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=seed)
        model.fit(x_train[mask], y[mask])
        component_preds[target] = np.asarray(model.predict(x_val), dtype=float)

    frame = val_index.with_columns(
        [pl.Series(target, component_preds[target]) for target in targets]
    )
    weights = [1.0 / len(targets)] * len(targets)
    ensembler = Ensembler()
    blended = ensembler.blend(
        Ensembler.rank_normalize(frame, pred_cols=list(targets), era_col=era_col),
        pred_cols=list(targets),
        weights=weights,
        era_col=era_col,
        out_col=pred_col,
    )
    # Ensembler.blend re-gaussianizes with plain rank_gaussianize (no unit-variance
    # standardization), so small eras come out with std slightly below 1.0. Apply the
    # unit-variance rank-Gaussian form once more: the blend output is already
    # gaussianized ranks, so this rescales only (rank order unchanged) and guarantees
    # every era has mean 0 and std 1, per the tier contract.
    gaussianized = Ensembler.rank_normalize(
        blended, pred_cols=[pred_col], era_col=era_col
    )
    return gaussianized.select([era_col, id_col, pred_col]).sort([era_col, id_col])


def generate_tree_predictions(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    target: str,
    feature_cols: Sequence[str],
    backend: str,
    params: Mapping[str, Any],
    seed: int,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Fit one shallow tree on purged train eras and predict validation rows."""
    if backend not in ("lightgbm", "xgboost"):
        raise ValueError(f"Unsupported tree backend: {backend!r}")
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")

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

    x_train = train_rows.select(feature_cols).cast(pl.Float32).to_pandas()
    y = train_rows.get_column(target).cast(pl.Float64).to_numpy()
    mask = np.isfinite(y)
    if mask.sum() < 2:
        raise ValueError(f"target {target!r} has fewer than 2 finite train rows after purge")
    x_val = val_rows.select(feature_cols).cast(pl.Float32).to_pandas()

    model = construct_tree_model(
        backend, dict(params), seed=seed, n_features=len(feature_cols),
        device="cpu",
    )
    model.fit(x_train[mask], y[mask])
    raw = np.asarray(model.predict(x_val), dtype=float)

    frame = val_rows.select([era_col, id_col]).with_columns(pl.Series(pred_col, raw))
    blended = Ensembler().blend(
        Ensembler.rank_normalize(frame, pred_cols=[pred_col], era_col=era_col),
        pred_cols=[pred_col],
        weights=[1.0],
        era_col=era_col,
        out_col=pred_col,
    )
    return blended.select([era_col, id_col, pred_col]).sort([era_col, id_col])


def generate_canonical_predictions(
    train: pl.DataFrame,
    val: pl.DataFrame,
    *,
    targets: Sequence[str],
    feature_cols: Sequence[str],
    params: Mapping[str, Any],
    seed: int,
    neutralization: float,
    purge_eras: int = DEFAULT_BENCHMARK_PURGE_ERAS,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Tier-3 canonical baselines: LightGBM fits + optional neutralization."""
    if not targets:
        raise ValueError("targets must be non-empty")
    if not 0.0 <= float(neutralization) <= 1.0:
        raise ValueError(f"neutralization must be in [0, 1], got {neutralization!r}")

    if len(targets) == 1:
        out = generate_tree_predictions(
            train, val, target=targets[0], feature_cols=feature_cols,
            backend="lightgbm", params=params, seed=seed,
            purge_eras=purge_eras, era_col=era_col, id_col=id_col,
            pred_col=pred_col,
        )
    else:
        parts: list[pl.DataFrame] = []
        for index, target in enumerate(targets):
            parts.append(
                generate_tree_predictions(
                    train, val, target=target, feature_cols=feature_cols,
                    backend="lightgbm", params=params, seed=seed + index,
                    purge_eras=purge_eras, era_col=era_col, id_col=id_col,
                    pred_col=pred_col,
                ).rename({pred_col: f"__component_{index}"})
            )
        stacked = parts[0]
        for part in parts[1:]:
            stacked = stacked.join(part, on=[era_col, id_col], how="inner")
        component_cols = [f"__component_{index}" for index in range(len(targets))]
        weights = [1.0 / len(targets)] * len(targets)
        ensembler = Ensembler()
        out = ensembler.blend(
            Ensembler.rank_normalize(
                stacked, pred_cols=component_cols, era_col=era_col
            ),
            pred_cols=component_cols,
            weights=weights,
            era_col=era_col,
            out_col=pred_col,
        ).select([era_col, id_col, pred_col]).sort([era_col, id_col])

    if float(neutralization) > 0.0:
        # NeutralizationEngine requires the feature columns present in-frame.
        with_features = out.join(
            val.select([era_col, id_col, *feature_cols]),
            on=[era_col, id_col],
            how="inner",
        )
        engine = NeutralizationEngine()
        out = engine.neutralize(
            with_features,
            pred_col=pred_col,
            feature_cols=list(feature_cols),
            era_col=era_col,
            proportion=float(neutralization),
        ).select([era_col, id_col, pred_col]).sort([era_col, id_col])
    return out


def score_benchmark_column(
    benchmarks: pl.DataFrame,
    *,
    column: str,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Wrap a benchmark-model column as a predictions frame."""
    if column not in benchmarks.columns:
        raise ValueError(f"Unknown benchmark column {column!r}")
    missing = [c for c in (era_col, id_col) if c not in benchmarks.columns]
    if missing:
        raise ValueError(f"benchmarks missing required columns: {missing}")
    return (
        benchmarks.select([era_col, id_col, pl.col(column).alias(pred_col)])
        .drop_nulls()
        .with_columns(pl.col(pred_col).cast(pl.Float64, strict=False))
        .filter(pl.col(pred_col).is_finite())
        .sort([era_col, id_col])
    )


def assert_tier0_null_floor(
    scorecards: Mapping[str, MetricScorecard],
    *,
    corr_tol: float = 0.005,
    sharpe_tol: float = 0.15,
) -> None:
    """Tier-0 sanity gate: null baselines must score at the statistical floor.

    Checks |corr| and |corr_sharpe_ac| for the three structural null kinds
    (constant-0.5, uniform-random, gaussian-random). ``deflated_sharpe`` is
    deliberately excluded: it has no constant null value on v5.3 (degenerate
    denominator behavior; measured null DSRs span 0.11-1.0).
    ``null_feature_mean`` is not structural noise (v5.3 corr 0.00294,
    sharpe 0.257) and is excluded as well.
    """
    for name in NULL_FLOOR_KINDS:
        if name not in scorecards:
            raise ValueError(f"Missing null baseline scorecard {name!r}")

    for name in NULL_FLOOR_KINDS:
        score = scorecards[name]
        _assert_scorecard_finite(score, model_id=name)
        checks = (
            ("corr", float(score.corr.value), float(corr_tol)),
            ("corr_sharpe_ac", float(score.corr_sharpe_ac.value), float(sharpe_tol)),
        )
        for metric_name, observed, tolerance in checks:
            if abs(observed) > tolerance:
                raise ValueError(
                    "Null floor violation for "
                    f"{name}.{metric_name}: |{observed:.8f}| > {tolerance:.8f}"
                )


def _tier4_gate_rows(
    scorecard: MetricScorecard, gate: Tier4GateConfig
) -> list[tuple[str, float | None, float, bool | None]]:
    """(field, observed, threshold, strict) rows for the 7 tier-4 fields.

    ``strict=None`` marks display-only fields (deflated_sharpe, A6) that are
    never pass/fail; ``observed=None`` marks structurally unavailable fields
    (turnover on v5.3).
    """
    card = scorecard
    return [
        ("corr", float(card.corr.value), float(gate.corr_min), False),
        ("corr_sharpe_ac", float(card.corr_sharpe_ac.value),
         float(gate.corr_sharpe_ac_min), False),
        ("fnc", float(card.fnc), float(gate.fnc_min), False),
        ("deflated_sharpe", float(card.deflated_sharpe),
         float(gate.deflated_sharpe_min), None),
        ("gain_to_pain_ratio", float(card.gain_to_pain_ratio),
         float(gate.gain_to_pain_min), False),
        ("cagr_1y", float(card.cagr_1y), float(gate.cagr_min), True),
        ("turnover_mean",
         None if card.turnover_mean is None else float(card.turnover_mean),
         float(gate.turnover_max), False),
    ]


def tier4_gate_verdict(
    scorecard: MetricScorecard, gate: Tier4GateConfig
) -> dict[str, bool | None]:
    """Per-threshold pass/fail booleans; None = unavailable/display-only."""
    _assert_scorecard_finite(scorecard, model_id=scorecard.model_id)
    verdict: dict[str, bool | None] = {}
    for field, observed, threshold, strict in _tier4_gate_rows(scorecard, gate):
        if observed is None or strict is None:
            verdict[field] = None
        elif field == "turnover_mean":
            verdict[field] = observed <= threshold
        elif strict:
            verdict[field] = observed > threshold
        else:
            verdict[field] = observed >= threshold
    return verdict


def assert_tier4_gate(scorecard: MetricScorecard, gate: Tier4GateConfig) -> None:
    """Production capital gate: reject candidates below the 7 hard thresholds.

    ``turnover_mean`` is structurally unavailable on v5.3 (disjoint era
    universes — consecutive validation eras share zero ids), so a ``None``
    turnover is reported by ``gate_report_frame`` but is not a hard failure.
    """
    _assert_scorecard_finite(scorecard, model_id=scorecard.model_id)
    violations: list[str] = []
    for field, observed, threshold, strict in _tier4_gate_rows(scorecard, gate):
        if observed is None or strict is None:
            continue
        if field == "turnover_mean":
            # turnover is an upper bound (<=), not a lower bound (>=); the
            # generic branches below would report a low turnover as a
            # violation, which would change hard-gate behavior.
            if observed > threshold:
                violations.append(
                    "turnover_mean: "
                    f"observed={observed:.8f}, need <= {threshold:.4f}"
                )
        elif strict:
            if observed <= threshold:
                violations.append(
                    f"{field}: observed={observed:.8f}, need > {threshold:.8f}"
                )
        elif observed < threshold:
            violations.append(
                f"{field}: observed={observed:.8f}, need >= {threshold:.8f}"
            )
    if violations:
        raise ValueError(
            f"Tier-4 gate violations for {scorecard.model_id!r}: "
            + "; ".join(violations)
        )


def tier_max_corrs(
    scorecards: Mapping[str, MetricScorecard],
    tier_of: Mapping[str, int],
) -> dict[int, float]:
    """Per-tier max of mean CORR (the monotonicity ladder metric)."""
    tiers = sorted(set(tier_of.values()))
    out: dict[int, float] = {}
    for tier in tiers:
        members = [mid for mid, t in tier_of.items() if t == tier]
        missing = [mid for mid in members if mid not in scorecards]
        if missing:
            raise ValueError(f"Missing scorecards for tier {tier}: {missing}")
        out[tier] = max(float(scorecards[mid].corr.value) for mid in members)
    return out


def assert_hierarchy_monotone(
    scorecards: Mapping[str, MetricScorecard],
    *,
    tier_of: Mapping[str, int],
    metric: str = "corr",
    atol: float = 1e-5,
) -> None:
    """Assert escalating tier ordering (T0 < T1 < T2 < T3 <= T4).

    Per-tier scalar = max over members of ``score.corr.value`` (default) or
    ``score.rank_scalar``. Evidence: on the v5.3 86-era meta overlap,
    rank_scalar noise spread swamps the null-vs-ridge rung (tier0 0.0092 >
    tier1 -0.0005), while mean corr orders all five tiers cleanly
    (0.00294 < 0.00478 < 0.00741 < 0.00952 <= 0.02927).
    """
    if metric not in ("corr", "rank_scalar"):
        raise ValueError(f"metric must be 'corr' or 'rank_scalar', got {metric!r}")
    tiers_present = sorted(set(tier_of.values()))
    if tiers_present != [0, 1, 2, 3, 4]:
        raise ValueError(f"tier_of must cover all tiers 0..4, got {tiers_present}")

    scalar_by_tier: dict[int, float]
    if metric == "corr":
        scalar_by_tier = tier_max_corrs(scorecards, tier_of)
    else:
        scalar_by_tier = {}
        for tier in (0, 1, 2, 3, 4):
            members = [mid for mid, t in tier_of.items() if t == tier]
            if not members:
                raise ValueError(f"No scorecards for tier {tier}")
            missing = [mid for mid in members if mid not in scorecards]
            if missing:
                raise ValueError(f"Missing scorecards for tier {tier}: {missing}")
            scalar_by_tier[tier] = max(
                float(scorecards[mid].rank_scalar) for mid in members
            )

    for lower in (0, 1, 2):
        if scalar_by_tier[lower] + atol > scalar_by_tier[lower + 1]:
            raise ValueError(
                "Monotone violation: tier "
                f"{lower}={scalar_by_tier[lower]:.8f} not < tier "
                f"{lower + 1}={scalar_by_tier[lower + 1]:.8f} (atol={atol:.2e})"
            )
    if scalar_by_tier[3] > scalar_by_tier[4] + atol:
        raise ValueError(
            "Monotone violation: tier "
            f"3={scalar_by_tier[3]:.8f} not <= tier "
            f"4={scalar_by_tier[4]:.8f} (atol={atol:.2e})"
        )


def scorecards_to_frame(scorecards: Mapping[str, MetricScorecard]) -> pl.DataFrame:
    if not scorecards:
        raise ValueError("scorecards must be non-empty")

    frames: list[pl.DataFrame] = []
    for model_id in sorted(scorecards):
        frame = scorecards[model_id].to_frame()
        row_model_id = frame.get_column("model_id")[0]
        if row_model_id != model_id:
            raise ValueError(
                "Scorecard model_id mismatch: "
                f"mapping key {model_id!r} != row model_id {row_model_id!r}"
            )
        frames.append(frame)

    return pl.concat(frames, how="vertical_relaxed").sort("model_id")


def write_scorecards_csv(
    scorecards: Mapping[str, MetricScorecard],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"output_path must be a .csv file: {path}")

    frame = scorecards_to_frame(scorecards)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)
    return path


def canonical_scorecards_bytes(
    scorecards: Mapping[str, MetricScorecard],
    fleet_scorecards: Mapping[str, MetricScorecard] | None = None,
) -> bytes:
    """Canonical, timing-stripped scorecard serialization for determinism.

    ``fleet_scorecards`` are merged into the same canonical payload so fleet
    determinism is covered by the same cross-process hash. Id collisions
    between the hierarchy and fleet mappings raise (both are scored domains).
    """
    if fleet_scorecards:
        collision = set(scorecards) & set(fleet_scorecards)
        if collision:
            raise ValueError(
                f"benchmark id collision between hierarchy and fleet: {sorted(collision)}"
            )
        scorecards = {**scorecards, **fleet_scorecards}
    frame = scorecards_to_frame(scorecards).sort("model_id")
    # Timing fields are wall-clock dependent and must not participate in
    # cross-process determinism hashes.
    timing_cols = {
        "quality_metric_total_seconds",
        "quality_metric_timings_json",
    }
    timing_cols.update(c for c in frame.columns if c.startswith("timing_"))
    frame = frame.drop(*sorted(timing_cols & set(frame.columns)))

    payload: dict[str, object] = {}
    for row in frame.iter_rows(named=True):
        model_id = str(row["model_id"])
        payload[model_id] = _sanitize_json_payload(row)

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )
    return encoded.encode("utf-8")


def scorecards_sha256(scorecards: Mapping[str, MetricScorecard]) -> str:
    return hashlib.sha256(canonical_scorecards_bytes(scorecards)).hexdigest()


def _assert_scorecard_finite(score: MetricScorecard, *, model_id: str) -> None:
    row = score.to_frame().row(0, named=True)
    for key, value in row.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"Non-finite value in scorecard {model_id}.{key}: {value}")


def _json_default(value: object) -> object:
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sanitize_json_payload(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _sanitize_json_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_payload(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_json_payload(v) for v in value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if np.isfinite(f):
            return f
        if np.isnan(f):
            return "NaN"
        if f > 0:
            return "Infinity"
        return "-Infinity"
    if isinstance(value, np.integer):
        return int(value)
    return value


# ---------------------------------------------------------------------------
# Hierarchy orchestration
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BenchmarkData:
    meta_model: pl.DataFrame
    benchmarks: pl.DataFrame
    features_json: Path
    train_path: Path
    validation_path: Path


def load_benchmark_data(data_dir: str | Path) -> BenchmarkData:
    """Load the lightweight shared domains; heavy parquets stay lazy."""
    directory = Path(data_dir)
    for name in ("meta_model.parquet", "validation_benchmark_models.parquet",
                 "features.json", "train.parquet", "validation.parquet"):
        if not (directory / name).exists():
            raise FileNotFoundError(f"Missing benchmark data asset: {directory / name}")
    meta_model = pl.read_parquet(directory / "meta_model.parquet").select(
        ["era", "id", "numerai_meta_model"]
    )
    benchmarks = pl.read_parquet(directory / "validation_benchmark_models.parquet")
    return BenchmarkData(
        meta_model=meta_model,
        benchmarks=benchmarks,
        features_json=directory / "features.json",
        train_path=directory / "train.parquet",
        validation_path=directory / "validation.parquet",
    )


def resolve_benchmark_feature_cols(
    features_json: Path,
    input_space: str,
    available: Sequence[str],
) -> list[str]:
    """Resolve feature columns for a benchmark input space, fail-loud."""
    if input_space not in VALID_INPUT_SPACES:
        raise ValueError(f"input_space={input_space!r} not in {VALID_INPUT_SPACES}")
    if input_space == "none":
        return []
    if input_space == "small":
        return resolve_small_feature_set(features_json, available)
    sets = resolve_feature_sets(features_json)
    if "medium" not in sets:
        raise ValueError("features.json has no 'medium' feature set")
    cols = [c for c in sets["medium"] if c in available]
    missing = sorted(set(sets["medium"]) - set(available))
    if missing:
        raise ValueError(
            f"{len(missing)} medium features missing from data columns: "
            f"{missing[:5]}..."
        )
    return cols


@dataclasses.dataclass(frozen=True)
class BenchmarkHierarchyResult:
    scorecards: Mapping[str, MetricScorecard]
    tier_of: Mapping[str, int]
    gate: Tier4GateConfig | None
    null_floor_ok: bool
    null_floor_errors: tuple[str, ...]
    tier4_violations: tuple[str, ...]
    monotone_ok: bool
    monotone_error: str | None


class BenchmarkHierarchy:
    """Config-driven 5-tier benchmark ladder (the line in the sand)."""

    def __init__(
        self,
        *,
        spec: BenchmarkSuiteSpec,
        data: BenchmarkData,
        seed: int = DEFAULT_BENCHMARK_SEED,
        horizon: str = "20D",
        n_boot: int = 1000,
        min_overlap_eras: int = 20,
        fast_mode: bool = False,
    ) -> None:
        if not spec.cells:
            raise ValueError("BenchmarkSuiteSpec has no cells")
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

    def _feature_cols(self, cell: BenchmarkCellConfig) -> list[str]:
        return resolve_benchmark_feature_cols(
            self._data.features_json, cell.input_space, self._schema_cols
        )

    def _cell_params(self, cell: BenchmarkCellConfig) -> dict[str, Any]:
        params = dict(cell.params)
        if self._fast_mode and cell.fast_mode_params:
            params.update(dict(cell.fast_mode_params))
        return params

    def _domain_frames(
        self, cell: BenchmarkCellConfig, feature_cols: list[str]
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        id_era = ["era", "id"]
        train = pl.read_parquet(
            self._data.train_path,
            columns=[*id_era, *feature_cols, *cell.targets],
        )
        val = pl.read_parquet(
            self._data.validation_path,
            columns=[*id_era, *feature_cols],
        )
        return train, val

    def _predictions_for_cell(
        self, cell: BenchmarkCellConfig
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return (predictions, val_feature_frame) for one benchmark cell."""
        feature_cols = self._feature_cols(cell)
        val_id = pl.read_parquet(
            self._data.validation_path, columns=["era", "id"]
        )
        params = self._cell_params(cell)

        if cell.model_kind in NULL_KINDS:
            if cell.model_kind == "null_feature_mean":
                val_features = pl.read_parquet(
                    self._data.validation_path,
                    columns=["era", "id", *feature_cols],
                )
                preds = generate_null_predictions(
                    val_id, kind=cell.model_kind, seed=cell.seed,
                    features=val_features, feature_cols=feature_cols,
                )
            else:
                preds = generate_null_predictions(
                    val_id, kind=cell.model_kind, seed=cell.seed
                )
                small_cols = resolve_benchmark_feature_cols(
                    self._data.features_json, "small", self._schema_cols
                )
                val_features = pl.read_parquet(
                    self._data.validation_path,
                    columns=["era", "id", *small_cols],
                )
            return preds, val_features

        train, val = self._domain_frames(cell, feature_cols)
        if cell.model_kind == "ridge":
            if "alpha" not in params:
                raise ValueError(
                    f"ridge cell {cell.benchmark_id!r} requires params.alpha"
                )
            alpha = float(params["alpha"])
            preds = generate_ridge_predictions(
                train, val, targets=list(cell.targets),
                feature_cols=feature_cols,
                alpha=alpha,
                seed=cell.seed,
            )
        elif cell.model_kind == "lightgbm":
            preds = generate_canonical_predictions(
                train, val, targets=list(cell.targets),
                feature_cols=feature_cols, params=params, seed=cell.seed,
                neutralization=cell.neutralization,
            )
        elif cell.model_kind == "xgboost":
            preds = generate_tree_predictions(
                train, val, target=cell.targets[0],
                feature_cols=feature_cols, backend="xgboost",
                params=params, seed=cell.seed,
            )
        else:
            raise ValueError(f"Unsupported benchmark model kind: {cell.model_kind!r}")
        return preds, val

    def run(self) -> BenchmarkHierarchyResult:
        """Score every cell, the tier-4 reference, and all hard gates."""
        scorecards: dict[str, MetricScorecard] = {}
        tier_of: dict[str, int] = {}
        # The validation targets block is identical for every cell - read it
        # once and share it across all cells (and the tier-4 reference) so a
        # 13-cell hierarchy does not re-stream the targets parquet per cell.
        val_targets = pl.read_parquet(
            self._data.validation_path, columns=self._target_cols
        )

        for cell in self._spec.cells:
            logger.info(
                "[hierarchy] tier %d: %s (kind=%s)", cell.tier,
                cell.benchmark_id, cell.model_kind,
            )
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
                benchmark_col=self._spec.reference_column,
                n_boot=self._n_boot,
                min_overlap_eras=self._min_overlap_eras,
                model_id=cell.benchmark_id,
            )
            tier_of[cell.benchmark_id] = cell.tier
            if cell.anchors:
                measured_corr = float(scorecards[cell.benchmark_id].corr.value)
                logger.info(
                    "[hierarchy] tier %d: %s anchors — measured corr=%.6f",
                    cell.tier, cell.benchmark_id, measured_corr,
                )
                for key, anchor in cell.anchors.items():
                    logger.info(
                        "    anchor %s=%.4f (measured=%.6f)",
                        key, float(anchor), measured_corr,
                    )

        reference_id = "v53_lgbm_ender60"
        if self._spec.reference_column:
            reference_id = self._spec.reference_column
            medium_cols = resolve_benchmark_feature_cols(
                self._data.features_json, "medium", self._schema_cols
            )
            ref_features = pl.read_parquet(
                self._data.validation_path,
                columns=["era", "id", *medium_cols],
            )
            ref_preds = score_benchmark_column(
                self._data.benchmarks, column=self._spec.reference_column
            )
            scorecards[reference_id] = evaluate_model(
                ref_preds,
                meta_model=self._data.meta_model,
                benchmarks=self._data.benchmarks,
                features=ref_features,
                targets=val_targets,
                n_trials=1,
                seed=self._seed,
                horizon=self._horizon,
                main_target="target",
                benchmark_col=self._spec.reference_column,
                n_boot=self._n_boot,
                min_overlap_eras=self._min_overlap_eras,
                model_id=reference_id,
            )
            tier_of[reference_id] = 4

        null_cards = {
            mid: scorecards[mid] for mid in NULL_KINDS if mid in scorecards
        }
        null_floor_ok, null_floor_errors = True, ()
        try:
            assert_tier0_null_floor(null_cards)
        except ValueError as exc:
            null_floor_ok, null_floor_errors = False, (str(exc),)

        tier4_violations: tuple[str, ...] = ()
        if self._spec.gate is not None and reference_id in scorecards:
            try:
                assert_tier4_gate(scorecards[reference_id], self._spec.gate)
            except ValueError as exc:
                tier4_violations = (str(exc),)

        monotone_ok, monotone_error = True, None
        try:
            assert_hierarchy_monotone(scorecards, tier_of=tier_of)
        except ValueError as exc:
            monotone_ok, monotone_error = False, str(exc)

        return BenchmarkHierarchyResult(
            scorecards=scorecards,
            tier_of=tier_of,
            gate=self._spec.gate,
            null_floor_ok=null_floor_ok,
            null_floor_errors=null_floor_errors,
            tier4_violations=tier4_violations,
            monotone_ok=monotone_ok,
            monotone_error=monotone_error,
        )


def hierarchy_frame(result: BenchmarkHierarchyResult) -> pl.DataFrame:
    """Scorecard rows with tier metadata (dashboard-compatible)."""
    frame = scorecards_to_frame(result.scorecards)
    tier_rows = pl.DataFrame(
        {
            "model_id": list(result.tier_of.keys()),
            "tier": [result.tier_of[mid] for mid in result.tier_of.keys()],
        }
    )
    frame = frame.join(tier_rows, on="model_id", how="left").with_columns(
        pl.col("tier").cast(pl.Int64).map_elements(
            lambda t: f"tier{int(t)}", return_dtype=pl.String
        ).alias("strategy_group")
    )
    return frame.sort(["tier", "model_id"])


def gate_report_frame(result: BenchmarkHierarchyResult) -> pl.DataFrame:
    """One row per tier-4 field: threshold vs measured."""
    gate = result.gate
    if gate is None:
        return pl.DataFrame(
            {"model_id": [], "field": [], "threshold": [], "measured": [], "pass": []}
        )
    reference_id = "v53_lgbm_ender60"
    for mid in result.tier_of:
        if result.tier_of[mid] == 4:
            reference_id = mid
            break
    card = result.scorecards[reference_id]
    rows = _tier4_gate_rows(card, gate)
    out_rows = []
    for field, measured, threshold, strict in rows:
        if measured is None or strict is None:
            passed = None
        elif strict:
            passed = measured > threshold
        else:
            passed = measured >= threshold
        out_rows.append({
            "model_id": reference_id,
            "field": field,
            "threshold": threshold,
            "measured": measured,
            "pass": passed,
        })
    return pl.DataFrame(out_rows)
