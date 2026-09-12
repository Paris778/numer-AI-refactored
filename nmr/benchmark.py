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
import hashlib
import json
import logging
import os
import re
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
from nmr.hardware import machine_memory_limits
from nmr.models import construct_tree_model
from nmr.payout import (
    CLASSIC_LEGACY_V1,
    PAYOUT_FACTOR_FILENAME,
    era_payout_factors,
    resolve_payout_policy,
)
from nmr.predictions import validation_key_fingerprint
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
    "NullFloorCalibration",
    "NullFloorConfig",
    "NullFloorSummary",
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
    "load_null_floor_calibration",
    "resolve_benchmark_feature_cols",
    "RidgeMemoryBudget",
    "estimate_ridge_peak_bytes",
    "ridge_memory_budget",
    "score_benchmark_column",
    "scorecards_sha256",
    "scorecards_to_frame",
    "tier4_gate_verdict",
    "tier_max_corrs",
    "train_validation_purged_split",
    "verify_null_floor_window",
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
    payout_policy_id: str
    scoring_target: str
    scoring_horizon: str
    corr_min: float
    corr_sharpe_ac_min: float
    fnc_min: float
    gain_to_pain_min: float

    def __post_init__(self) -> None:
        for name in ("payout_policy_id", "scoring_target", "scoring_horizon"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Tier4GateConfig.{name} must be a non-empty string")
        for field in dataclasses.fields(self):
            if field.name in {"payout_policy_id", "scoring_target", "scoring_horizon"}:
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Tier4GateConfig.{field.name} must be numeric, got {value!r}"
                )
            if not float(value) == float(value):  # NaN check
                raise ValueError(f"Tier4GateConfig.{field.name} must be finite")
        if not (-1.0 <= self.corr_min <= 1.0):
            raise ValueError(f"corr_min out of range: {self.corr_min!r}")


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
            raise ValueError(
                f"benchmark_id must be a non-empty string: {self.benchmark_id!r}"
            )
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
        if (
            self.model_kind in NULL_KINDS
            and self.input_space != "none"
            and self.model_kind != "null_feature_mean"
        ):
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
class NullFloorConfig:
    """Tier-0 null-floor gate configuration (calibration reference).

    The ACTIVE AC-Sharpe threshold comes from the digest-bound calibration
    artifact (pre-registered family-wise quantile), never from a code
    default; the CORR floor stays the structural 0.005 constant.
    """

    calibration: str
    corr_tol: float = 0.005

    def __post_init__(self) -> None:
        if not isinstance(self.calibration, str) or not self.calibration:
            raise ValueError("NullFloorConfig.calibration must be a non-empty string")
        if isinstance(self.corr_tol, bool) or not isinstance(
            self.corr_tol, (int, float)
        ):
            raise ValueError(
                f"NullFloorConfig.corr_tol must be numeric, got {self.corr_tol!r}"
            )
        if not 0.0 < float(self.corr_tol) < 1.0:
            raise ValueError(
                f"NullFloorConfig.corr_tol must be in (0, 1), got {self.corr_tol!r}"
            )


@dataclasses.dataclass(frozen=True)
class BenchmarkFileConfig:
    tier: int
    cells: tuple[BenchmarkCellConfig, ...] = ()
    reference_column: str | None = None
    reference_columns: tuple[str, ...] = ()
    gate: Tier4GateConfig | None = None
    null_floor: NullFloorConfig | None = None

    def __post_init__(self) -> None:
        if self.tier not in VALID_BENCHMARK_TIERS:
            raise ValueError(f"tier={self.tier!r} not in {VALID_BENCHMARK_TIERS}")
        if self.tier == 4:
            if self.gate is None:
                raise ValueError("tier 4 config requires a 'gate' section")
            if not self.reference_column:
                raise ValueError("tier 4 config requires a non-empty reference_column")
            if self.reference_column in self.reference_columns:
                raise ValueError(
                    f"reference_column {self.reference_column!r} must not repeat "
                    f"in reference_columns"
                )
        else:
            if not self.cells:
                raise ValueError(f"tier {self.tier} config requires non-empty cells")
            if self.gate is not None:
                raise ValueError(
                    f"gate section only allowed for tier 4, got tier {self.tier}"
                )
        if self.null_floor is not None and self.tier != 0:
            raise ValueError(
                f"null_floor section only allowed for tier 0, got tier {self.tier}"
            )
        ids = [cell.benchmark_id for cell in self.cells]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate benchmark ids in file: {ids}")


def _build_benchmark_cell(data: Any, tier: int) -> BenchmarkCellConfig:
    if not isinstance(data, dict):
        raise ValueError(f"benchmark cell must be a mapping, got {type(data).__name__}")
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
        raise ValueError(
            f"benchmark config must be a mapping, got {type(raw).__name__}"
        )
    _reject_unknown_keys(BenchmarkFileConfig, raw)
    if not isinstance(raw.get("cells", []), list):
        raise ValueError("cells must be a list")
    gate_raw = raw.get("gate")
    gate = None
    if gate_raw is not None:
        _reject_unknown_keys(Tier4GateConfig, gate_raw)
        gate = Tier4GateConfig(**gate_raw)
    null_floor_raw = raw.get("null_floor")
    null_floor = None
    if null_floor_raw is not None:
        _reject_unknown_keys(NullFloorConfig, null_floor_raw)
        null_floor = NullFloorConfig(**null_floor_raw)
    ref_cols_raw = raw.get("reference_columns", [])
    if not isinstance(ref_cols_raw, list):
        raise ValueError("reference_columns must be a list")
    reference_columns = tuple(ref_cols_raw)
    if any(not isinstance(c, str) or not c for c in reference_columns):
        raise ValueError("reference_columns entries must be non-empty strings")
    return BenchmarkFileConfig(
        tier=int(raw["tier"]),
        cells=tuple(
            _build_benchmark_cell(c, int(raw["tier"])) for c in raw.get("cells", [])
        ),
        reference_column=raw.get("reference_column"),
        reference_columns=reference_columns,
        gate=gate,
        null_floor=null_floor,
    )


@dataclasses.dataclass(frozen=True)
class BenchmarkSuiteSpec:
    cells: tuple[BenchmarkCellConfig, ...]
    gate: Tier4GateConfig | None
    reference_column: str | None
    reference_columns: tuple[str, ...] = ()
    null_floor: NullFloorConfig | None = None
    null_floor_base_dir: str | None = None


def load_benchmark_suite_config(config_dir: str | Path) -> BenchmarkSuiteSpec:
    """Load every *.yaml file in config_dir and aggregate into a suite spec."""
    directory = Path(config_dir)
    files = sorted(p for p in directory.glob("*.yaml"))
    if not files:
        raise ValueError(f"no benchmark config files found in {directory}")
    all_cells: list[BenchmarkCellConfig] = []
    gate: Tier4GateConfig | None = None
    reference_column: str | None = None
    reference_columns: tuple[str, ...] = ()
    null_floor: NullFloorConfig | None = None
    null_floor_base_dir: str | None = None
    for path in files:
        file_cfg = load_benchmark_file(path)
        if file_cfg.gate is not None:
            if gate is not None:
                raise ValueError("multiple tier-4 gate configs found")
            gate = file_cfg.gate
            reference_column = file_cfg.reference_column
            reference_columns = file_cfg.reference_columns
        if file_cfg.null_floor is not None:
            if null_floor is not None:
                raise ValueError("multiple tier-0 null_floor configs found")
            null_floor = file_cfg.null_floor
            null_floor_base_dir = str(path.parent)
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
        reference_columns=reference_columns,
        null_floor=null_floor,
        null_floor_base_dir=null_floor_base_dir,
    )


def _ordered_numeric_eras(eras: Sequence[str]) -> list[str]:
    """Dedupe, validate, and numerically sort era labels."""
    if not eras:
        raise ValueError("era universe is empty")
    mapping: dict[int, str] = {}
    for era in eras:
        if not isinstance(era, str):
            raise ValueError(f"Era labels must be strings, got {type(era).__name__}")
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
    if (
        isinstance(purge_eras, bool)
        or not isinstance(purge_eras, int)
        or purge_eras < 0
    ):
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

    index = prediction_index.select([era_col, id_col]).unique().sort([era_col, id_col])
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
            raise ValueError(f"null_feature_mean join dropped {n - joined.height} rows")
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


# ---------------------------------------------------------------------------
# D1a ridge memory discipline (2026-09-08): the medium ridge cell previously
# materialized a full-width float64 deviation matrix (~15.7 GiB) inside
# np.std and a full-width fancy-index subset copy (~8.4 GiB), on top of the
# sklearn Ridge.fit float64 upcast. Peak measured pre-fix by
# freeze_ridge_reference.py: working set 49.7 GiB / commit 79.0 GiB. The
# memory-safe path below keeps sklearn Ridge untouched and bounds every
# transient: column-block two-pass statistics, a single float32 finite-y fit
# block (no full-width raw numpy block, no subset copy), and era-batched
# validation predicts. Estimated corrected peak ~30 GiB commit.
# ---------------------------------------------------------------------------

# Column-block size for the bounded two-pass statistics (transient =
# n_rows x block_cols x 8 float64, ~0.7 GiB at the medium geometry).
_RIDGE_STANDARDIZE_BLOCK_COLS = 32
# Fixed overhead term: process, polars frames beyond the feature blocks,
# logging, and metric frames (~1 GiB measured-scale slack).
_RIDGE_FIXED_OVERHEAD_BYTES = 1 * 2**30
# Estimator safety factor: the per-array terms deliberately OVERCOUNT the
# measured medium-cell peak (25.1 GiB commit / 16.1 GiB working set per the
# freeze receipt) — the preflight must be conservative, never optimistic.
_RIDGE_ESTIMATE_SAFETY_FACTOR = 1.1
# Configured commit ceiling for a ridge benchmark cell. Derivation: the
# corrected path measures ~42 GiB commit (1.1x-factored estimate ~45.7 GiB)
# on the medium cell; the 52 GiB ceiling admits it with margin, stays well
# below the machine commit limit, and the working-set guard (0.85 of
# physical) remains the binding thrash protection. Env override:
# NMR_RIDGE_COMMIT_CEILING_BYTES (fail loud on invalid values).
_RIDGE_COMMIT_CEILING_BYTES = 52 * 2**30
# Working-set guard fraction (mirrors promote._RAM_WS_FRACTION): the
# estimated peak working set must stay below this fraction of physical RAM
# or the cell refuses before materializing (thrash guard).
_RIDGE_WS_FRACTION = 0.85


def _ridge_commit_ceiling_bytes() -> int:
    """Resolve the ridge commit ceiling (constant or NMR_RIDGE_COMMIT_CEILING_BYTES)."""
    raw = os.environ.get("NMR_RIDGE_COMMIT_CEILING_BYTES")
    if raw is None:
        return _RIDGE_COMMIT_CEILING_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"NMR_RIDGE_COMMIT_CEILING_BYTES must be an integer >= 1, got {raw!r}"
        ) from exc
    if value < 1:
        raise ValueError(
            f"NMR_RIDGE_COMMIT_CEILING_BYTES must be an integer >= 1, got {raw!r}"
        )
    return value


def _block_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column float64 mean/std of a bounded float32 block.

    The float32 block is cast to float64 once (the caller bounds the block
    width); np.std's deviation matrix is then block-sized, never full-width.
    """
    block = values.astype(np.float64)
    return np.mean(block, axis=0), np.std(block, axis=0)


def _finalize_statistics(
    mu: np.ndarray, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the finite/zero-variance guards and downcast to (mu32, scale32)."""
    mu = np.where(np.isfinite(mu), mu, 0.0)
    scale = np.where((sigma > 0.0) & np.isfinite(sigma), 1.0 / sigma, 0.0)
    return mu.astype(np.float32), scale.astype(np.float32)


def _feature_block_statistics(
    train_values: np.ndarray,
    *,
    block_cols: int = _RIDGE_STANDARDIZE_BLOCK_COLS,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded two-pass (mu, scale) statistics of a float32 feature block.

    Iterates column blocks so no full-width float64 temporary is ever
    allocated. Returns (mu32, scale32) with the legacy finite/zero-variance
    guards applied exactly.
    """
    if train_values.ndim != 2 or train_values.shape[0] < 1 or train_values.shape[1] < 1:
        raise ValueError("train_values must be a non-empty 2D array")
    n_features = train_values.shape[1]
    mu = np.empty(n_features, dtype=np.float64)
    sigma = np.empty(n_features, dtype=np.float64)
    for start in range(0, n_features, block_cols):
        stop = min(start + block_cols, n_features)
        mu_b, sigma_b = _block_mean_std(train_values[:, start:stop])
        mu[start:stop] = mu_b
        sigma[start:stop] = sigma_b
    return _finalize_statistics(mu, sigma)


def _apply_standardization(
    values: np.ndarray, mu32: np.ndarray, scale32: np.ndarray
) -> np.ndarray:
    """Standardize a float32 block in place with precomputed statistics."""
    np.subtract(values, mu32, out=values)
    np.multiply(values, scale32, out=values)
    return values


def _polars_feature_statistics(
    frame: pl.DataFrame,
    feature_cols: Sequence[str],
    trimmed_train_eras: Sequence[str] | set[str],
    era_col: str,
    *,
    block_cols: int = _RIDGE_STANDARDIZE_BLOCK_COLS,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded two-pass statistics straight from a polars frame.

    Column blocks are selected AND era-filtered lazily before ``to_numpy``,
    so no full-width numpy feature block is materialized for statistics.
    """
    n_features = len(feature_cols)
    mu = np.empty(n_features, dtype=np.float64)
    sigma = np.empty(n_features, dtype=np.float64)
    for start in range(0, n_features, block_cols):
        cols = list(feature_cols[start : start + block_cols])
        block = (
            frame.lazy()
            .select([era_col, *cols])
            .filter(pl.col(era_col).is_in(trimmed_train_eras))
            .collect()
            .select(cols)
            .cast(pl.Float32)
            .to_numpy(writable=True)
        )
        mu_b, sigma_b = _block_mean_std(block)
        mu[start : start + len(cols)] = mu_b
        sigma[start : start + len(cols)] = sigma_b
        del block
    return _finalize_statistics(mu, sigma)


def _standardize_feature_block(
    train_values: np.ndarray, val_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize with train statistics; zero-variance features -> 0.0.

    Float32 end-to-end with in-place updates. Statistics are computed in
    bounded column blocks (see :func:`_feature_block_statistics`) so no
    full-width float64 temporary exists — the pre-fix ``np.std(dtype=float64)``
    deviation matrix was ~15.7 GiB at the medium geometry.
    """
    mu32, scale32 = _feature_block_statistics(train_values)
    _apply_standardization(train_values, mu32, scale32)
    _apply_standardization(val_values, mu32, scale32)
    return train_values, val_values


@dataclasses.dataclass(frozen=True)
class RidgeMemoryBudget:
    """Preflight memory verdict for one ridge cell (dual-metric)."""

    cell_id: str
    n_train_rows: int
    n_val_rows: int
    n_features: int
    dtype: str
    terms: Mapping[str, float]
    machine_commit_limit_bytes: int | None
    machine_physical_bytes: int | None
    commit_ceiling_bytes: int
    ws_fraction: float
    peak_commit_bytes: int
    peak_ws_bytes: int
    verdict: str
    reasons: tuple[str, ...]

    def describe(self) -> str:
        """One-line human summary for logs (never hashed)."""
        return (
            f"ridge_memory_budget[{self.cell_id}] rows={self.n_train_rows}/"
            f"{self.n_val_rows} features={self.n_features} dtype={self.dtype} "
            f"peak_commit={self.peak_commit_bytes / 2**30:.1f}GiB "
            f"peak_ws={self.peak_ws_bytes / 2**30:.1f}GiB "
            f"ceiling={self.commit_ceiling_bytes / 2**30:.1f}GiB "
            f"ws_fraction={self.ws_fraction} verdict={self.verdict}"
        )


def estimate_ridge_peak_bytes(
    n_train_rows: int,
    n_val_rows: int,
    n_val_eras: int,
    n_features: int,
    *,
    caller_itemsize: int = 1,
) -> dict[str, int]:
    """Pure per-stage peak estimates for the memory-safe sklearn ridge path.

    Terms (all bytes):
      - caller_frames: the caller-held polars frames (train + validation
        feature columns at ``caller_itemsize`` — v5.x integer bins are Int8)
      - stats_transient: bounded column-block statistics (float32 pull +
        float64 cast/deviation for one block)
      - fit_block_temp: per-block raw pull + float32 cast/ops transients
        for one column block of the fit matrix
      - fit_matrix: the C-order float32 standardized fit matrix (sklearn's
        cholesky runs in float32 arithmetic on float32 input — see
        generate_ridge_predictions)
      - val_block_temp: per-block transients for one column block of the
        validation matrix
      - val_matrix: the F-order float32 standardized validation matrix
      - val_upcast: numpy's float32->float64 promotion inside the
        whole-block predict dot (pre-fix semantics; a pre-built float64
        matrix is NOT bit-equal)
      - overhead: fixed process/metric slack
    Peak = caller_frames + max(fit_matrix + fit_block_temp, val_matrix +
    val_upcast + val_block_temp) + overhead. The whole-block predict is
    REQUIRED for bit-identity with the pre-fix path: per-row BLAS results
    depend on the call shape, and the rank-domain blend amplifies any
    last-ulp drift to O(1) in the final output.
    """
    for name, value in (
        ("n_train_rows", n_train_rows),
        ("n_val_rows", n_val_rows),
        ("n_val_eras", n_val_eras),
        ("n_features", n_features),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    caller_frames = (n_train_rows + n_val_rows) * n_features * caller_itemsize
    stats_transient = n_train_rows * min(_RIDGE_STANDARDIZE_BLOCK_COLS, n_features) * 12
    block_cols = min(_RIDGE_STANDARDIZE_BLOCK_COLS, n_features)
    # Block per element: int8 raw (1) + float32 cast (4) + astype temp
    # slack (4) = 9 bytes.
    fit_block_temp = n_train_rows * block_cols * 9
    fit_matrix = n_train_rows * n_features * 4
    val_block_temp = n_val_rows * block_cols * 9
    val_matrix = n_val_rows * n_features * 4
    val_upcast = n_val_rows * n_features * 8
    overhead = _RIDGE_FIXED_OVERHEAD_BYTES
    peak_base = caller_frames + max(
        fit_matrix + fit_block_temp,
        val_matrix + val_upcast + val_block_temp,
    )
    peak_base += overhead
    peak = int(peak_base * _RIDGE_ESTIMATE_SAFETY_FACTOR + 0.5)
    return {
        "caller_frames_bytes": caller_frames,
        "stats_transient_bytes": stats_transient,
        "fit_block_temp_bytes": fit_block_temp,
        "fit_matrix_bytes": fit_matrix,
        "val_block_temp_bytes": val_block_temp,
        "val_matrix_bytes": val_matrix,
        "val_upcast_bytes": val_upcast,
        "overhead_bytes": overhead,
        "safety_factor": _RIDGE_ESTIMATE_SAFETY_FACTOR,
        "peak_commit_bytes": peak,
        "peak_ws_bytes": peak,
    }


def ridge_memory_budget(
    cell_id: str,
    *,
    n_train_rows: int,
    n_val_rows: int,
    n_val_eras: int,
    n_features: int,
    dtype: str = "float64",
    machine_commit_limit_bytes: int | None = None,
    machine_physical_bytes: int | None = None,
    commit_ceiling_bytes: int | None = None,
    ws_fraction: float = _RIDGE_WS_FRACTION,
) -> RidgeMemoryBudget:
    """Dual-metric preflight for one ridge cell (fail before OOM).

    Refuses when ANY guard fails: the configured commit ceiling, the machine
    commit limit, or the working-set fraction of physical RAM. Warns (never
    refuses) when machine limits are unavailable — the configured ceiling
    still applies. Callers raise on ``verdict == "refuse"`` before
    materializing any feature matrix.
    """
    ceiling = (
        _ridge_commit_ceiling_bytes()
        if commit_ceiling_bytes is None
        else commit_ceiling_bytes
    )
    terms = estimate_ridge_peak_bytes(n_train_rows, n_val_rows, n_val_eras, n_features)
    peak_commit = terms["peak_commit_bytes"]
    peak_ws = terms["peak_ws_bytes"]
    refusals: list[str] = []
    if peak_commit > ceiling:
        refusals.append(
            f"estimated peak commit {peak_commit / 2**30:.1f} GiB exceeds the "
            f"configured {ceiling / 2**30:.1f} GiB ceiling"
        )
    if (
        machine_commit_limit_bytes is not None
        and peak_commit > machine_commit_limit_bytes
    ):
        refusals.append(
            f"estimated peak commit {peak_commit / 2**30:.1f} GiB exceeds the "
            f"machine commit limit {machine_commit_limit_bytes / 2**30:.1f} GiB"
        )
    if (
        machine_physical_bytes is not None
        and peak_ws > machine_physical_bytes * ws_fraction
    ):
        refusals.append(
            f"estimated peak working set {peak_ws / 2**30:.1f} GiB exceeds "
            f"{ws_fraction:.2f} of {machine_physical_bytes / 2**30:.1f} GiB "
            "physical RAM"
        )
    warnings: list[str] = []
    if machine_commit_limit_bytes is None or machine_physical_bytes is None:
        warnings.append(
            "machine memory limits unavailable; the configured ceiling still applies"
        )
    verdict = "refuse" if refusals else ("warn" if warnings else "allow")
    return RidgeMemoryBudget(
        cell_id=cell_id,
        n_train_rows=n_train_rows,
        n_val_rows=n_val_rows,
        n_features=n_features,
        dtype=dtype,
        terms=dict(terms),
        machine_commit_limit_bytes=machine_commit_limit_bytes,
        machine_physical_bytes=machine_physical_bytes,
        commit_ceiling_bytes=ceiling,
        ws_fraction=ws_fraction,
        peak_commit_bytes=peak_commit,
        peak_ws_bytes=peak_ws,
        verdict=verdict,
        reasons=tuple([*refusals, *warnings]),
    )


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
    budget_cell_id: str | None = None,
) -> pl.DataFrame:
    """Fit purged Ridge models per target and blend in rank-Gaussian domain.

    Memory-safe (D1a, 2026-09-08): a dual-metric budget refuses before any
    feature matrix is materialized; statistics are computed in bounded
    column blocks straight from the polars frame. The fit and validation
    matrices are float32 assembled in column blocks (no full-width raw
    numpy block, no fancy-index copy, no sorted-frame copy — the pre-fix
    sorted row order is reproduced with a gather index). float32 inputs are
    REQUIRED for bit-identity: sklearn's Ridge cholesky runs in float32
    arithmetic on float32 input, and numpy's float32->float64 promotion
    inside the predict dot is not bit-equal to a pre-built float64 matrix
    (both proven by the frozen-reference drift). The validation predict is
    one whole-block call (exactly the pre-fix call shape), so coefficients
    and predictions are bit-identical to the pre-fix path (proven by the
    frozen-reference test).
    """
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha < 0:
        raise ValueError(f"alpha must be a non-negative number, got {alpha!r}")
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")
    if not targets:
        raise ValueError("targets must be non-empty")

    trimmed_train_eras, _val_eras = train_validation_purged_split(
        train.get_column(era_col).unique().to_list(),
        val.get_column(era_col).unique().to_list(),
        purge_eras=purge_eras,
    )

    missing_feats = [
        c for c in feature_cols if c not in train.columns or c not in val.columns
    ]
    if missing_feats:
        raise ValueError(f"missing feature columns: {missing_feats}")
    for target in targets:
        if target not in train.columns:
            raise ValueError(f"missing target column: {target!r}")

    # Dual-metric memory budget BEFORE materializing any feature matrix
    # (fail before OOM; refusal never silently degrades).
    n_train_rows = (
        train.select([era_col]).filter(pl.col(era_col).is_in(trimmed_train_eras)).height
    )
    n_val_eras = val.get_column(era_col).n_unique()
    physical, commit_limit = machine_memory_limits()
    budget = ridge_memory_budget(
        budget_cell_id or "ridge_predictions",
        n_train_rows=n_train_rows,
        n_val_rows=val.height,
        n_val_eras=n_val_eras,
        n_features=len(feature_cols),
        machine_commit_limit_bytes=commit_limit,
        machine_physical_bytes=physical,
    )
    if budget.verdict == "refuse":
        raise ValueError(
            "ridge memory budget refuses the fit before materializing: "
            + "; ".join(budget.reasons)
            + " | "
            + budget.describe()
        )
    logger.info("[ridge] %s", budget.describe())

    # Bounded column-block statistics straight from the polars frame
    # (no full-width numpy feature block is materialized for statistics).
    mu32, scale32 = _polars_feature_statistics(
        train, feature_cols, trimmed_train_eras, era_col
    )

    # Fit phase (train-only; the validation matrix is not materialized yet).
    # Each target's fit matrix is a C-order float32 matrix assembled in
    # column blocks from the polars frame. It MUST stay float32: sklearn's
    # Ridge.fit on float32 input runs its cholesky solver in float32
    # arithmetic, which no float64 precomputation can reproduce bit-for-bit
    # (proven by the frozen-reference drift) — so the matrix is the exact
    # pre-fix float32 standardized block, just built without full-width
    # temporaries.
    models: dict[str, Ridge] = {}
    for target in targets:
        y = (
            train.lazy()
            .select([era_col, target])
            .filter(
                pl.col(era_col).is_in(trimmed_train_eras) & pl.col(target).is_finite()
            )
            .collect()
            .get_column(target)
            .cast(pl.Float64)
            .to_numpy()
        )
        if y.size < 2:
            raise ValueError(
                f"target {target!r} has fewer than 2 finite train rows after purge"
            )
        x_fit = np.empty((y.size, len(feature_cols)), dtype=np.float32, order="C")
        for start in range(0, len(feature_cols), _RIDGE_STANDARDIZE_BLOCK_COLS):
            stop = min(start + _RIDGE_STANDARDIZE_BLOCK_COLS, len(feature_cols))
            cols = list(feature_cols[start:stop])
            raw_block = (
                train.lazy()
                .filter(
                    pl.col(era_col).is_in(trimmed_train_eras)
                    & pl.col(target).is_finite()
                )
                .select([era_col, *cols])
                .collect()
                .select(cols)
                .to_numpy(writable=True)
            )
            block = raw_block.astype(np.float32)
            np.subtract(block, mu32[start:stop], out=block)
            np.multiply(block, scale32[start:stop], out=block)
            x_fit[:, start:stop] = block
            del raw_block, block
        model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=seed)
        model.fit(x_fit, y)
        del x_fit, y
        models[target] = model

    # Validation phase: the pre-fix sorted row order is reproduced with a
    # gather index (np.std's pairwise reduction inside the rank blend is
    # order-dependent at the last ulp) WITHOUT materializing a sorted copy
    # of the frame. The F-order float32 matrix carries the exact pre-fix
    # standardized values, and the whole-block predict keeps the BLAS call
    # shape bit-identical (float32 input, exactly like pre-fix).
    sort_idx = (
        val.select([era_col, id_col])
        .with_row_index("__ridx")
        .sort([era_col, id_col])
        .get_column("__ridx")
    )
    val_index = val.select([era_col, id_col]).gather(sort_idx)
    x_val = np.empty((val.height, len(feature_cols)), dtype=np.float32, order="F")
    for start in range(0, len(feature_cols), _RIDGE_STANDARDIZE_BLOCK_COLS):
        stop = min(start + _RIDGE_STANDARDIZE_BLOCK_COLS, len(feature_cols))
        cols = list(feature_cols[start:stop])
        raw_block = val.select(cols).gather(sort_idx).to_numpy(writable=True)
        block = raw_block.astype(np.float32)
        np.subtract(block, mu32[start:stop], out=block)
        np.multiply(block, scale32[start:stop], out=block)
        x_val[:, start:stop] = block
        del raw_block, block

    component_frames: dict[str, pl.DataFrame] = {}
    for target in targets:
        component_frames[target] = val_index.with_columns(
            pl.Series(target, np.asarray(models[target].predict(x_val), dtype=float))
        )
    del x_val, sort_idx

    frame = component_frames[targets[0]]
    for target in targets[1:]:
        frame = frame.join(component_frames[target], on=[era_col, id_col], how="inner")
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
    missing_feats = [
        c for c in feature_cols if c not in train.columns or c not in val.columns
    ]
    if missing_feats:
        raise ValueError(f"missing feature columns: {missing_feats}")

    x_train = train_rows.select(feature_cols).cast(pl.Float32).to_pandas()
    y = train_rows.get_column(target).cast(pl.Float64).to_numpy()
    mask = np.isfinite(y)
    if mask.sum() < 2:
        raise ValueError(
            f"target {target!r} has fewer than 2 finite train rows after purge"
        )
    x_val = val_rows.select(feature_cols).cast(pl.Float32).to_pandas()

    model = construct_tree_model(
        backend,
        dict(params),
        seed=seed,
        n_features=len(feature_cols),
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
            train,
            val,
            target=targets[0],
            feature_cols=feature_cols,
            backend="lightgbm",
            params=params,
            seed=seed,
            purge_eras=purge_eras,
            era_col=era_col,
            id_col=id_col,
            pred_col=pred_col,
        )
    else:
        parts: list[pl.DataFrame] = []
        for index, target in enumerate(targets):
            parts.append(
                generate_tree_predictions(
                    train,
                    val,
                    target=target,
                    feature_cols=feature_cols,
                    backend="lightgbm",
                    params=params,
                    seed=seed + index,
                    purge_eras=purge_eras,
                    era_col=era_col,
                    id_col=id_col,
                    pred_col=pred_col,
                ).rename({pred_col: f"__component_{index}"})
            )
        stacked = parts[0]
        for part in parts[1:]:
            stacked = stacked.join(part, on=[era_col, id_col], how="inner")
        component_cols = [f"__component_{index}" for index in range(len(targets))]
        weights = [1.0 / len(targets)] * len(targets)
        ensembler = Ensembler()
        out = (
            ensembler.blend(
                Ensembler.rank_normalize(
                    stacked, pred_cols=component_cols, era_col=era_col
                ),
                pred_cols=component_cols,
                weights=weights,
                era_col=era_col,
                out_col=pred_col,
            )
            .select([era_col, id_col, pred_col])
            .sort([era_col, id_col])
        )

    if float(neutralization) > 0.0:
        # NeutralizationEngine requires the feature columns present in-frame.
        with_features = out.join(
            val.select([era_col, id_col, *feature_cols]),
            on=[era_col, id_col],
            how="inner",
        )
        engine = NeutralizationEngine()
        out = (
            engine.neutralize(
                with_features,
                pred_col=pred_col,
                feature_cols=list(feature_cols),
                era_col=era_col,
                proportion=float(neutralization),
            )
            .select([era_col, id_col, pred_col])
            .sort([era_col, id_col])
        )
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


# ---------------------------------------------------------------------------
# Tier-0 null-floor calibration (2026-09-12). The structural-null AC-Sharpe
# floor is a PRE-REGISTERED empirical family-wise quantile over a fixed seed
# set, stored in a committed digest-bound calibration artifact — not a code
# constant. The calibration study (null_floor_study.py) established that a
# fixed single-seed ±0.15 floor rejects ~half of all seeds at the 86-era
# window, so the gate checks the family-wise max against the calibrated p99
# and anchors determinism on the stored seed-42 expected values. The CORR
# floor stays the structural 0.005 constant; null_feature_mean remains
# excluded (feature-derived, not structural noise).
# ---------------------------------------------------------------------------
_NULL_FLOOR_CALIBRATION_SCHEMA_VERSION = 1
# Determinism anchor tolerance: observed seed-42 values must match the stored
# calibration values to float noise (they are deterministic).
_NULL_FLOOR_ANCHOR_ATOL = 1e-9
_NULL_FLOOR_IDENTITY_KEYS = (
    "payout_policy_id",
    "scoring_target",
    "scoring_horizon",
    "scoring_backend",
)


def _canonical_payload_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclasses.dataclass(frozen=True)
class NullFloorCalibration:
    """Digest-bound Tier-0 null-floor calibration reference (committed)."""

    schema_version: int
    method: str
    data_version: str
    window_key_fingerprint: str
    validation_eras: tuple[str, ...]
    validation_rows: int
    scoring_identity: Mapping[str, Any]
    null_kinds: tuple[str, ...]
    seed_count: int
    selected_quantile: float
    selected_threshold: float
    seed42_expected: Mapping[str, Mapping[str, float]]
    seed42_family_max: float
    study_code_fingerprint: str
    digest: str

    def describe(self) -> str:
        """One-line log summary (never hashed)."""
        return (
            f"tier0 null floor: method={self.method} "
            f"quantile={self.selected_quantile} seeds={self.seed_count} "
            f"threshold={self.selected_threshold:.4f} "
            f"window={self.validation_eras[0]}..{self.validation_eras[-1]} "
            f"({len(self.validation_eras)} eras) "
            f"reference_digest={self.digest[:12]}"
        )


def load_null_floor_calibration(path: str | Path) -> NullFloorCalibration:
    """Load, schema-check, digest-verify, and validate the calibration file.

    Unknown schema versions, tampered payloads, non-structural kind sets, and
    malformed values all REFUSE (fail loud) — a stale or edited calibration
    must never silently authorize a gate decision.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("null-floor calibration must be a JSON object")
    if payload.get("schema_version") != _NULL_FLOOR_CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            "null-floor calibration schema_version "
            f"{payload.get('schema_version')!r} is not supported (expected "
            f"{_NULL_FLOOR_CALIBRATION_SCHEMA_VERSION}); recalibrate"
        )
    digest = payload.get("digest")
    if not isinstance(digest, str) or not digest:
        raise ValueError("null-floor calibration missing digest")
    recomputed = _canonical_payload_digest(
        {k: v for k, v in payload.items() if k != "digest"}
    )
    if recomputed != digest:
        raise ValueError(
            "null-floor calibration digest mismatch: "
            f"file={digest[:12]} recomputed={recomputed[:12]}"
        )
    required = (
        "method",
        "data_version",
        "window_key_fingerprint",
        "validation_eras",
        "validation_rows",
        "scoring_identity",
        "null_kinds",
        "seed_set",
        "selected_quantile",
        "selected_threshold",
        "seed42_expected",
        "seed42_family_max",
        "study_code_fingerprint",
    )
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"null-floor calibration missing fields: {missing}")
    kinds = tuple(payload["null_kinds"])
    if kinds != NULL_FLOOR_KINDS:
        raise ValueError(
            "null-floor calibration null_kinds must be the three structural "
            f"kinds {NULL_FLOOR_KINDS}, got {kinds!r}"
        )
    quantile = float(payload["selected_quantile"])
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"selected_quantile must be in (0, 1], got {quantile!r}")
    threshold = float(payload["selected_threshold"])
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(
            f"selected_threshold must be finite and > 0, got {threshold!r}"
        )
    seed_set = payload["seed_set"]
    if not isinstance(seed_set, Mapping):
        raise ValueError("seed_set must be a mapping")
    seed_count = int(seed_set.get("count", 0))
    if seed_count < 1:
        raise ValueError(f"seed_set.count must be >= 1, got {seed_set!r}")
    expected_raw = payload["seed42_expected"]
    if not isinstance(expected_raw, Mapping):
        raise ValueError("seed42_expected must be a mapping")
    expected: dict[str, dict[str, float]] = {}
    for kind in NULL_FLOOR_KINDS:
        row = expected_raw.get(kind)
        if (
            not isinstance(row, Mapping)
            or "corr" not in row
            or "corr_sharpe_ac" not in row
        ):
            raise ValueError(f"seed42_expected missing values for {kind!r}")
        expected[kind] = {
            "corr": float(row["corr"]),
            "corr_sharpe_ac": float(row["corr_sharpe_ac"]),
        }
    window = tuple(str(e) for e in payload["validation_eras"])
    if not window:
        raise ValueError("validation_eras must be non-empty")
    return NullFloorCalibration(
        schema_version=_NULL_FLOOR_CALIBRATION_SCHEMA_VERSION,
        method=str(payload["method"]),
        data_version=str(payload["data_version"]),
        window_key_fingerprint=str(payload["window_key_fingerprint"]),
        validation_eras=window,
        validation_rows=int(payload["validation_rows"]),
        scoring_identity=dict(payload["scoring_identity"]),
        null_kinds=kinds,
        seed_count=seed_count,
        selected_quantile=quantile,
        selected_threshold=threshold,
        seed42_expected=expected,
        seed42_family_max=float(payload["seed42_family_max"]),
        study_code_fingerprint=str(payload["study_code_fingerprint"]),
        digest=digest,
    )


def verify_null_floor_window(
    calibration: NullFloorCalibration,
    *,
    window_key_fingerprint: str,
    validation_eras: Sequence[str],
    scoring_identity: Mapping[str, Any],
) -> None:
    """Refuse a stale calibration: window and scoring identity must match.

    The standardized comparison window moves after a data refresh; a stale
    null calibration must never authorize a new data snapshot.
    """
    if window_key_fingerprint != calibration.window_key_fingerprint:
        raise ValueError(
            "null-floor calibration belongs to a different data window: "
            f"calibration={calibration.window_key_fingerprint[:12]} "
            f"current={window_key_fingerprint[:12]}; recalibrate before scoring"
        )
    eras = tuple(sorted((str(e) for e in validation_eras), key=int))
    calibrated = tuple(sorted(calibration.validation_eras, key=int))
    if eras != calibrated:
        raise ValueError(
            "null-floor calibration era window does not match the current "
            f"validation window ({len(eras)} vs {len(calibrated)} eras); recalibrate"
        )
    for key in _NULL_FLOOR_IDENTITY_KEYS:
        if str(scoring_identity.get(key)) != str(calibration.scoring_identity.get(key)):
            raise ValueError(
                f"null-floor calibration scoring identity mismatch on {key!r}: "
                f"calibration={calibration.scoring_identity.get(key)!r} "
                f"current={scoring_identity.get(key)!r}; recalibrate"
            )


@dataclasses.dataclass(frozen=True)
class NullFloorSummary:
    """Observed null-floor values plus calibration identity (gate-report source)."""

    method: str
    quantile: float
    seed_count: int
    threshold: float
    corr_tol: float
    reference_fingerprint: str
    observed: Mapping[str, Mapping[str, float]]
    family_max_abs_ac_sharpe: float


def assert_tier0_null_floor(
    scorecards: Mapping[str, MetricScorecard],
    *,
    calibration: NullFloorCalibration,
    corr_tol: float = 0.005,
) -> NullFloorSummary:
    """Tier-0 sanity gate: structural nulls must sit inside the calibrated envelope.

    The AC-Sharpe threshold is the PRE-REGISTERED family-wise quantile from
    the calibration artifact (never a code default); the CORR floor stays
    0.005. Observed seed-42 values are anchored to the calibration's stored
    expected values (determinism check) and the family-wise max |AC-Sharpe|
    must stay at or below the calibrated threshold. ``null_feature_mean`` is
    excluded by construction: the calibration kind set must be exactly the
    three structural nulls or the gate refuses.
    """
    if tuple(calibration.null_kinds) != NULL_FLOOR_KINDS:
        raise ValueError(
            "null-floor calibration kinds must be the three structural nulls; "
            f"got {tuple(calibration.null_kinds)!r}"
        )
    for name in NULL_FLOOR_KINDS:
        if name not in scorecards:
            raise ValueError(f"Missing null baseline scorecard {name!r}")

    observed: dict[str, dict[str, float]] = {}
    for name in NULL_FLOOR_KINDS:
        score = scorecards[name]
        _assert_scorecard_finite(score, model_id=name)
        corr_value = float(score.corr.value)
        ac_value = float(score.corr_sharpe_ac.value)
        expected = calibration.seed42_expected[name]
        if abs(corr_value - float(expected["corr"])) > _NULL_FLOOR_ANCHOR_ATOL:
            raise ValueError(
                f"Null floor determinism anchor failed for {name}.corr: "
                f"observed={corr_value:.12f} "
                f"expected={float(expected['corr']):.12f}"
            )
        if abs(ac_value - float(expected["corr_sharpe_ac"])) > _NULL_FLOOR_ANCHOR_ATOL:
            raise ValueError(
                f"Null floor determinism anchor failed for {name}.corr_sharpe_ac: "
                f"observed={ac_value:.12f} "
                f"expected={float(expected['corr_sharpe_ac']):.12f}"
            )
        observed[name] = {"corr": corr_value, "corr_sharpe_ac": ac_value}
        if abs(corr_value) > float(corr_tol):
            raise ValueError(
                f"Null floor violation for {name}.corr: "
                f"|{corr_value:.8f}| > {float(corr_tol):.8f}"
            )

    family_observed = max(abs(v["corr_sharpe_ac"]) for v in observed.values())
    threshold = float(calibration.selected_threshold)
    if family_observed > threshold:
        worst = max(observed, key=lambda k: abs(observed[k]["corr_sharpe_ac"]))
        raise ValueError(
            "Null floor violation (calibrated family-wise): "
            f"max |corr_sharpe_ac| = {family_observed:.8f} ({worst}) > "
            f"calibrated p{calibration.selected_quantile * 100:.1f} threshold "
            f"{threshold:.8f} ({calibration.seed_count} seeds); recalibrate "
            "only through the pre-registered procedure"
        )
    return NullFloorSummary(
        method=calibration.method,
        quantile=calibration.selected_quantile,
        seed_count=calibration.seed_count,
        threshold=threshold,
        corr_tol=float(corr_tol),
        reference_fingerprint=calibration.digest,
        observed=observed,
        family_max_abs_ac_sharpe=family_observed,
    )


def _tier4_gate_rows(
    scorecard: MetricScorecard, gate: Tier4GateConfig
) -> list[tuple[str, float, float, bool]]:
    """Enforceable `(field, observed, threshold, strict)` gate rows."""
    observed_identity = (
        scorecard.payout_policy_id,
        scorecard.scoring_target,
        scorecard.scoring_horizon,
    )
    expected_identity = (
        gate.payout_policy_id,
        gate.scoring_target,
        gate.scoring_horizon,
    )
    if observed_identity != expected_identity:
        raise ValueError(
            f"tier-4 gate policy mismatch: expected {expected_identity}, "
            f"got {observed_identity}"
        )
    card = scorecard
    return [
        ("corr", float(card.corr.value), float(gate.corr_min), False),
        (
            "corr_sharpe_ac",
            float(card.corr_sharpe_ac.value),
            float(gate.corr_sharpe_ac_min),
            False,
        ),
        ("fnc", float(card.fnc), float(gate.fnc_min), False),
        (
            "gain_to_pain_ratio",
            float(card.gain_to_pain_ratio),
            float(gate.gain_to_pain_min),
            False,
        ),
    ]


def tier4_gate_verdict(
    scorecard: MetricScorecard, gate: Tier4GateConfig
) -> dict[str, bool | None]:
    """Per-threshold pass/fail booleans; None = unavailable/display-only."""
    _assert_scorecard_finite(scorecard, model_id=scorecard.model_id)
    verdict: dict[str, bool | None] = {}
    for field, observed, threshold, strict in _tier4_gate_rows(scorecard, gate):
        if strict:
            verdict[field] = observed > threshold
        else:
            verdict[field] = observed >= threshold
    return verdict


def assert_tier4_gate(scorecard: MetricScorecard, gate: Tier4GateConfig) -> None:
    """Production capital gate: reject candidates below the four hard thresholds.

    Only fields with measurable, policy-comparable evidence are configured.
    """
    _assert_scorecard_finite(scorecard, model_id=scorecard.model_id)
    violations: list[str] = []
    for field, observed, threshold, strict in _tier4_gate_rows(scorecard, gate):
        if strict:
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
    for name in (
        "meta_model.parquet",
        "validation_benchmark_models.parquet",
        "features.json",
        "train.parquet",
        "validation.parquet",
    ):
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
    gated_reference_id: str | None = None
    null_floor_summary: NullFloorSummary | None = None


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
        gate_policy = (
            resolve_payout_policy(spec.gate.payout_policy_id)
            if spec.gate is not None
            else CLASSIC_LEGACY_V1
        )
        self._gate_policy = gate_policy
        self._scoring_target = gate_policy.target or "target"
        self._scoring_horizon = gate_policy.scoring_horizon or self._horizon
        # Tier-0 null-floor calibration: load and verify the committed,
        # digest-bound reference BEFORE any scoring — a stale calibration
        # must never authorize a new data snapshot (fail early).
        self._null_floor_calibration: NullFloorCalibration | None = None
        if spec.null_floor is not None:
            calibration_path = Path(spec.null_floor_base_dir or ".") / (
                spec.null_floor.calibration
            )
            calibration = load_null_floor_calibration(calibration_path)
            verify_null_floor_window(
                calibration,
                window_key_fingerprint=validation_key_fingerprint(
                    data.meta_model.select(["era", "id"])
                ),
                validation_eras=data.meta_model.get_column("era").unique().to_list(),
                scoring_identity={
                    "payout_policy_id": gate_policy.policy_id,
                    "scoring_target": self._scoring_target,
                    "scoring_horizon": self._scoring_horizon,
                    "scoring_backend": "custom",
                },
            )
            self._null_floor_calibration = calibration
            logger.info("[hierarchy] %s", calibration.describe())
        target_cols = ["era", "id", "target"]
        reference = spec.reference_column or ""
        match = re.search(r"_([a-zA-Z0-9]+)(?:20|60)$", reference)
        if match is not None:
            target_name = match.group(1)
            target_cols.extend(
                column
                for column in (
                    f"target_{target_name}_20",
                    f"target_{target_name}_60",
                )
                if column in self._schema_cols
            )
        self._target_cols = list(dict.fromkeys(target_cols))

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
        val_id = pl.read_parquet(self._data.validation_path, columns=["era", "id"])
        params = self._cell_params(cell)

        if cell.model_kind in NULL_KINDS:
            if cell.model_kind == "null_feature_mean":
                val_features = pl.read_parquet(
                    self._data.validation_path,
                    columns=["era", "id", *feature_cols],
                )
                preds = generate_null_predictions(
                    val_id,
                    kind=cell.model_kind,
                    seed=cell.seed,
                    features=val_features,
                    feature_cols=feature_cols,
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
                train,
                val,
                targets=list(cell.targets),
                feature_cols=feature_cols,
                alpha=alpha,
                seed=cell.seed,
                budget_cell_id=cell.benchmark_id,
            )
        elif cell.model_kind == "lightgbm":
            preds = generate_canonical_predictions(
                train,
                val,
                targets=list(cell.targets),
                feature_cols=feature_cols,
                params=params,
                seed=cell.seed,
                neutralization=cell.neutralization,
            )
        elif cell.model_kind == "xgboost":
            preds = generate_tree_predictions(
                train,
                val,
                target=cell.targets[0],
                feature_cols=feature_cols,
                backend="xgboost",
                params=params,
                seed=cell.seed,
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
        pf_map = era_payout_factors(
            Path(self._data.validation_path).parent / PAYOUT_FACTOR_FILENAME
        )
        gate_policy = self._gate_policy
        scoring_target = self._scoring_target
        scoring_horizon = self._scoring_horizon
        scoring_pf = pf_map if gate_policy.fixed_payout_factor is None else None

        for cell in self._spec.cells:
            logger.info(
                "[hierarchy] tier %d: %s (kind=%s)",
                cell.tier,
                cell.benchmark_id,
                cell.model_kind,
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
                payout_policy=gate_policy,
                horizon=scoring_horizon,
                main_target=scoring_target,
                benchmark_col=self._spec.reference_column,
                pf=scoring_pf,
                n_boot=self._n_boot,
                min_overlap_eras=self._min_overlap_eras,
                model_id=cell.benchmark_id,
            )
            tier_of[cell.benchmark_id] = cell.tier
            if cell.anchors:
                measured_corr = float(scorecards[cell.benchmark_id].corr.value)
                logger.info(
                    "[hierarchy] tier %d: %s anchors — measured corr=%.6f",
                    cell.tier,
                    cell.benchmark_id,
                    measured_corr,
                )
                for key, anchor in cell.anchors.items():
                    logger.info(
                        "    anchor %s=%.4f (measured=%.6f)",
                        key,
                        float(anchor),
                        measured_corr,
                    )

        reference_id = "v53_lgbm_ender60"
        if self._spec.reference_column:
            reference_id = self._spec.reference_column
            # Tier-4 reference rows: the gated reference column first (it owns
            # the capital-line gate), then any additional reference columns
            # (e.g. the second official Numerai benchmark model) scored as
            # informational tier-4 rows. Each is evaluated identically; only
            # `reference_id` feeds assert_tier4_gate below.
            reference_columns = [reference_id, *self._spec.reference_columns]
            medium_cols = resolve_benchmark_feature_cols(
                self._data.features_json, "medium", self._schema_cols
            )
            ref_features = pl.read_parquet(
                self._data.validation_path,
                columns=["era", "id", *medium_cols],
            )
            for index, ref_col in enumerate(reference_columns):
                ref_preds = score_benchmark_column(
                    self._data.benchmarks, column=ref_col
                )
                scorecards[ref_col] = evaluate_model(
                    ref_preds,
                    meta_model=self._data.meta_model,
                    benchmarks=self._data.benchmarks,
                    features=ref_features,
                    targets=val_targets,
                    n_trials=1,
                    seed=self._seed + index,
                    payout_policy=gate_policy,
                    horizon=scoring_horizon,
                    main_target=scoring_target,
                    benchmark_col=self._spec.reference_column,
                    pf=scoring_pf,
                    n_boot=self._n_boot,
                    min_overlap_eras=self._min_overlap_eras,
                    model_id=ref_col,
                )
                tier_of[ref_col] = 4

        null_cards = {mid: scorecards[mid] for mid in NULL_KINDS if mid in scorecards}
        null_floor_ok, null_floor_errors = True, ()
        null_floor_summary: NullFloorSummary | None = None
        try:
            if self._null_floor_calibration is None:
                raise ValueError(
                    "no tier-0 null-floor calibration configured for this suite"
                )
            null_floor_summary = assert_tier0_null_floor(
                null_cards,
                calibration=self._null_floor_calibration,
                corr_tol=(
                    self._spec.null_floor.corr_tol
                    if self._spec.null_floor is not None
                    else 0.005
                ),
            )
            logger.info(
                "[hierarchy] null floor passed: family_max|ac|=%.6f <= %.6f "
                "(%s, seeds=%d, ref=%s)",
                null_floor_summary.family_max_abs_ac_sharpe,
                null_floor_summary.threshold,
                null_floor_summary.method,
                null_floor_summary.seed_count,
                null_floor_summary.reference_fingerprint[:12],
            )
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
            gated_reference_id=reference_id if self._spec.gate is not None else None,
            null_floor_summary=null_floor_summary,
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
        pl.col("tier")
        .cast(pl.Int64)
        .map_elements(lambda t: f"tier{int(t)}", return_dtype=pl.String)
        .alias("strategy_group")
    )
    return frame.sort(["tier", "model_id"])


def gate_report_frame(result: BenchmarkHierarchyResult) -> pl.DataFrame:
    """Tier-4 gate rows plus calibrated tier-0 null-floor rows.

    The extra `null_floor_*` columns carry the calibration identity (method,
    quantile, seed count, reference digest) on null-floor rows; tier-4 rows
    leave them null. A suite without a tier-4 gate still reports the
    null-floor rows (the structural gate is tier-0, not tier-4).
    """
    out_rows: list[dict[str, Any]] = []
    gate = result.gate
    if gate is not None:
        reference_id = result.gated_reference_id
        if not reference_id:
            raise ValueError(
                "BenchmarkHierarchyResult.gated_reference_id is required to build "
                "the tier-4 gate report"
            )
        if reference_id not in result.scorecards:
            raise ValueError(
                f"gated_reference_id {reference_id!r} is missing from hierarchy "
                "scorecards"
            )
        card = result.scorecards[reference_id]
        for field, measured, threshold, strict in _tier4_gate_rows(card, gate):
            if measured is None or strict is None:
                passed = None
            elif strict:
                passed = measured > threshold
            else:
                passed = measured >= threshold
            out_rows.append(
                {
                    "model_id": reference_id,
                    "field": field,
                    "threshold": threshold,
                    "measured": measured,
                    "pass": passed,
                    "null_floor_method": None,
                    "null_floor_quantile": None,
                    "null_floor_seed_count": None,
                    "null_floor_reference_fingerprint": None,
                }
            )
    summary = result.null_floor_summary
    if summary is not None:
        for kind, metrics in summary.observed.items():
            for metric in ("corr", "corr_sharpe_ac"):
                measured = float(metrics[metric])
                threshold = (
                    summary.corr_tol if metric == "corr" else float(summary.threshold)
                )
                out_rows.append(
                    {
                        "model_id": kind,
                        "field": metric,
                        "threshold": threshold,
                        "measured": measured,
                        "pass": abs(measured) <= threshold,
                        "null_floor_method": summary.method,
                        "null_floor_quantile": float(summary.quantile),
                        "null_floor_seed_count": int(summary.seed_count),
                        "null_floor_reference_fingerprint": (
                            summary.reference_fingerprint
                        ),
                    }
                )
    return pl.DataFrame(
        out_rows,
        schema={
            "model_id": pl.String,
            "field": pl.String,
            "threshold": pl.Float64,
            "measured": pl.Float64,
            "pass": pl.Boolean,
            "null_floor_method": pl.String,
            "null_floor_quantile": pl.Float64,
            "null_floor_seed_count": pl.Int64,
            "null_floor_reference_fingerprint": pl.String,
        },
    )
