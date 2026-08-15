"""E6 benchmark infrastructure: null floors, tutorial ingestion, integration gates.

This module wires scorecard evaluation into a simple benchmark suite that can:
- generate deterministic null baselines (constant-0.5, uniform-random, gaussian-random),
- ingest tutorial prediction vectors from notebook-adjacent artifacts,
- enforce Slice 1 gates (null floor and monotone sanity),
- produce canonical bytes for cross-process determinism checks.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dataclasses
from types import MappingProxyType

import yaml

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from nmr.evaluation import MIN_OVERLAP_ERAS
from nmr.ensemble import Ensembler
from nmr.inference import block_bootstrap_ci, resolve_block_len
from nmr.models import construct_tree_model
from nmr.risk import NeutralizationEngine
from nmr.scorecard import MetricScorecard, evaluate_model

logger = logging.getLogger("nmr.benchmark")

__all__ = [
    "NULL_BASELINES",
    "TUTORIAL_NOTEBOOK_TO_MODEL_ID",
    "BenchmarkSuite",
    "scorecards_to_frame",
    "write_scorecards_csv",
    "discover_tutorial_notebooks",
    "assert_notebook_prediction_contract",
    "extract_oos_predictions",
    "ingest_tutorial_prediction",
    "ingest_tutorial_prediction_batch",
    "assert_null_floor",
    "assert_slice1_monotone",
    "canonical_scorecards_bytes",
    "scorecards_sha256",
]

NULL_BASELINES: tuple[str, ...] = (
    "constant-0.5",
    "uniform-random",
    "gaussian-random",
)

TUTORIAL_NOTEBOOK_TO_MODEL_ID: dict[str, str] = {
    "1_hello_numerai.ipynb": "hello-numerai",
    "2_feature_neutralization.ipynb": "feature-neutralization",
    "example-model-sunshine.ipynb": "sunshine",
}

_TUTORIAL_NOTEBOOK_ANCHORS: dict[str, tuple[str, ...]] = {
    "1_hello_numerai.ipynb": ("validation", "prediction", "model.predict"),
    "2_feature_neutralization.ipynb": (
        "validation",
        "prediction",
        "model.predict",
    ),
    "example-model-sunshine.ipynb": (
        "all_data.loc[validation_index",
        "prediction",
        "validation_predictions_",
    ),
}


@dataclass(frozen=True)
class _EvalConfig:
    n_trials: int
    seed: int
    horizon: str
    main_target: str
    benchmark_col: str | None
    regime_labels: pl.DataFrame | None
    pf: float
    clip: float
    n_boot: int
    alpha: float
    min_overlap_eras: int
    model_id: str
    era_col: str
    id_col: str
    pred_col: str
    meta_col: str
    trials_sr_var: float | None
    sr0_benchmark: float


class BenchmarkSuite:
    """Evaluate baseline and tutorial predictions with a shared scorecard pipeline."""

    def __init__(
        self,
        *,
        meta_model: pl.DataFrame,
        benchmarks: pl.DataFrame | None,
        features: pl.DataFrame,
        targets: pl.DataFrame,
        n_trials: int,
        seed: int,
        horizon: str = "20D",
        main_target: str = "target",
        benchmark_col: str | None = None,
        regime_labels: pl.DataFrame | None = None,
        pf: float = 1.0,
        clip: float = 0.05,
        n_boot: int = 1000,
        alpha: float = 0.05,
        min_overlap_eras: int = 20,
        era_col: str = "era",
        id_col: str = "id",
        pred_col: str = "prediction",
        meta_col: str = "numerai_meta_model",
        trials_sr_var: float | None = None,
        sr0_benchmark: float = 0.0,
    ) -> None:
        self._eval_cfg = _EvalConfig(
            n_trials=n_trials,
            seed=seed,
            horizon=horizon,
            main_target=main_target,
            benchmark_col=benchmark_col,
            regime_labels=regime_labels,
            pf=pf,
            clip=clip,
            n_boot=n_boot,
            alpha=alpha,
            min_overlap_eras=min_overlap_eras,
            model_id="model",
            era_col=era_col,
            id_col=id_col,
            pred_col=pred_col,
            meta_col=meta_col,
            trials_sr_var=trials_sr_var,
            sr0_benchmark=sr0_benchmark,
        )

        self._meta_model = meta_model
        self._benchmarks = benchmarks
        self._features = features
        self._targets = targets

        self._join_keys = self._resolve_join_keys()
        self._prediction_index = (
            self._targets.select(self._join_keys).unique().sort(self._join_keys)
        )
        self._id_to_era = None
        if id_col in self._join_keys:
            self._id_to_era = self._targets.select([id_col, era_col]).unique()

    def iter_baseline_predictions(
        self,
        *,
        include_classical: bool = False,
        min_train_eras: int = 10,
    ) -> Iterator[tuple[str, str, pl.DataFrame, int]]:
        """Yield (model_id, group, raw_preds, seed) for null/trivial/classical baselines."""
        base = self._eval_cfg.seed
        for idx, baseline in enumerate(NULL_BASELINES):
            r_seed = base + idx
            yield (
                baseline,
                "null",
                self.null_prediction_frame(baseline, seed=r_seed),
                r_seed,
            )
        yield ("trivial", "classical", self._trivial_prediction_frame(), base + 3)
        if include_classical:
            yield (
                "linear",
                "classical",
                self._walk_forward_model_predictions(
                    model_name="linear", min_train_eras=min_train_eras
                ),
                base + 4,
            )
            yield (
                "tree",
                "classical",
                self._walk_forward_model_predictions(
                    model_name="tree", min_train_eras=min_train_eras
                ),
                base + 5,
            )

    def run_classical_baselines(
        self,
        *,
        min_train_eras: int = 10,
    ) -> dict[str, MetricScorecard]:
        """Generate and score S11 classical rungs: trivial, linear, and tree."""
        out: dict[str, MetricScorecard] = {}
        for model_id, group, raw_preds, _seed in self.iter_baseline_predictions(
            include_classical=True, min_train_eras=min_train_eras
        ):
            if group != "classical":
                continue
            # Historical behavior: score with the suite's default seed (no seed= arg).
            out[model_id] = self.evaluate_predictions(raw_preds, model_id=model_id)
        return out

    def compute_book_orthogonality(
        self,
        candidate_scores: pl.Series | np.ndarray,
        book_scores: pl.Series | np.ndarray,
        *,
        seed: int,
        n_boot: int = 1000,
        horizon: str = "20D",
    ) -> dict[str, float | tuple[float, float] | None]:
        """Compute global/tail correlation and spread with joint circular bootstrap.

        The tail mask is recomputed inside each bootstrap replicate on the
        resampled book path, preserving contiguous temporal dependence.
        """

        cand = self._as_finite_vector(candidate_scores, name="candidate_scores")
        book = self._as_finite_vector(book_scores, name="book_scores")
        if cand.shape[0] != book.shape[0]:
            raise ValueError(
                "candidate_scores and book_scores must have the same length"
            )

        n = int(cand.shape[0])
        min_overlap = max(MIN_OVERLAP_ERAS, self._eval_cfg.min_overlap_eras)
        if n < min_overlap:
            raise ValueError(
                "Non-vacuity violation: overlap yielded only "
                f"{n} eras; minimum required {min_overlap}."
            )

        joint = np.column_stack([cand, book])
        block_len = resolve_block_len(n, horizon)

        point = self._orthogonality_stat(joint)
        ci_global = block_bootstrap_ci(
            joint,
            lambda arr: float(self._orthogonality_stat(arr)[0]),
            block_len=block_len,
            n_boot=n_boot,
            seed=seed,
            alpha=self._eval_cfg.alpha,
        )
        ci_tail = block_bootstrap_ci(
            joint,
            lambda arr: float(self._orthogonality_stat(arr)[1]),
            block_len=block_len,
            n_boot=n_boot,
            seed=seed + 1,
            alpha=self._eval_cfg.alpha,
        )
        ci_spread = block_bootstrap_ci(
            joint,
            lambda arr: float(self._orthogonality_stat(arr)[2]),
            block_len=block_len,
            n_boot=n_boot,
            seed=seed + 2,
            alpha=self._eval_cfg.alpha,
        )

        return {
            "rho_global": float(point[0]),
            "rho_tail": float(point[1]),
            "spread": float(point[2]),
            "rho_global_ci": (float(ci_global.lo), float(ci_global.hi)),
            "rho_tail_ci": (float(ci_tail.lo), float(ci_tail.hi)),
            "spread_ci": (float(ci_spread.lo), float(ci_spread.hi)),
            "n_eras": float(n),
            "redundancy_mean": None,
            "redundancy_max": None,
        }

    def run_null_baselines(
        self, *, seed: int | None = None
    ) -> dict[str, MetricScorecard]:
        base_seed = self._eval_cfg.seed if seed is None else int(seed)
        out: dict[str, MetricScorecard] = {}
        for idx, baseline in enumerate(NULL_BASELINES):
            preds = self.null_prediction_frame(baseline, seed=base_seed + idx)
            out[baseline] = self.evaluate_predictions(preds, model_id=baseline)
        return out

    def evaluate_predictions(
        self,
        predictions: pl.DataFrame,
        *,
        model_id: str,
        seed: int | None = None,
    ) -> MetricScorecard:
        normalized = self._normalize_predictions(predictions)
        return self.evaluate_normalized_predictions(
            normalized,
            model_id=model_id,
            seed=seed,
        )

    def evaluate_normalized_predictions(
        self,
        normalized_predictions: pl.DataFrame,
        *,
        model_id: str,
        seed: int | None = None,
    ) -> MetricScorecard:
        run_seed = self._eval_cfg.seed if seed is None else int(seed)
        cfg = self._eval_cfg
        return evaluate_model(
            normalized_predictions,
            meta_model=self._meta_model,
            benchmarks=self._benchmarks,
            features=self._features,
            targets=self._targets,
            n_trials=cfg.n_trials,
            seed=run_seed,
            horizon=cfg.horizon,
            main_target=cfg.main_target,
            benchmark_col=cfg.benchmark_col,
            regime_labels=cfg.regime_labels,
            pf=cfg.pf,
            clip=cfg.clip,
            n_boot=cfg.n_boot,
            alpha=cfg.alpha,
            min_overlap_eras=cfg.min_overlap_eras,
            model_id=model_id,
            era_col=cfg.era_col,
            id_col=cfg.id_col,
            pred_col=cfg.pred_col,
            meta_col=cfg.meta_col,
            trials_sr_var=cfg.trials_sr_var,
            sr0_benchmark=cfg.sr0_benchmark,
        )

    def normalized_era_labels(self, predictions: pl.DataFrame) -> pl.DataFrame:
        """Return per-era row counts from normalized predictions for profiling."""

        normalized = self._normalize_predictions(predictions)
        era_col = self._eval_cfg.era_col
        return (
            normalized.group_by(era_col).len().rename({"len": "n_rows"}).sort(era_col)
        )

    def normalize_predictions(self, predictions: pl.DataFrame) -> pl.DataFrame:
        """Public wrapper for benchmark orchestration that needs normalized frames."""

        return self._normalize_predictions(predictions)

    def evaluate_tutorial_predictions(
        self,
        predictions_by_model_id: Mapping[str, pl.DataFrame],
    ) -> dict[str, MetricScorecard]:
        out: dict[str, MetricScorecard] = {}
        for model_id, frame in predictions_by_model_id.items():
            out[model_id] = self.evaluate_predictions(frame, model_id=model_id)
        return out

    def scorecards_to_frame(
        self,
        scorecards: Mapping[str, MetricScorecard],
    ) -> pl.DataFrame:
        return scorecards_to_frame(scorecards)

    def write_scorecards_csv(
        self,
        scorecards: Mapping[str, MetricScorecard],
        output_path: str | Path,
    ) -> Path:
        return write_scorecards_csv(scorecards, output_path)

    def null_prediction_frame(self, baseline: str, *, seed: int) -> pl.DataFrame:
        pred_col = self._eval_cfg.pred_col
        n = self._prediction_index.height
        if baseline == "constant-0.5":
            values = np.full(n, 0.5, dtype=float)
        elif baseline == "uniform-random":
            values = np.random.default_rng(seed).uniform(0.0, 1.0, n)
        elif baseline == "gaussian-random":
            values = np.random.default_rng(seed).normal(0.0, 1.0, n)
        else:
            raise ValueError(
                f"Unknown baseline {baseline!r}; expected one of {NULL_BASELINES}"
            )

        return self._prediction_index.with_columns(pl.Series(pred_col, values))

    def _resolve_join_keys(self) -> list[str]:
        era_col = self._eval_cfg.era_col
        id_col = self._eval_cfg.id_col
        frames = (self._meta_model, self._features, self._targets)
        if any(era_col not in frame.columns for frame in frames):
            raise ValueError(f"Missing required columns: ['{era_col}']")
        if all(id_col in frame.columns for frame in frames):
            return [era_col, id_col]
        return [era_col]

    def _normalize_predictions(self, predictions: pl.DataFrame) -> pl.DataFrame:
        if not isinstance(predictions, pl.DataFrame):
            raise ValueError("predictions must be a polars DataFrame")

        cfg = self._eval_cfg
        pred_col = cfg.pred_col
        era_col = cfg.era_col
        id_col = cfg.id_col

        frame = predictions
        cols = set(frame.columns)

        if pred_col not in cols:
            raise ValueError(f"Missing required columns: ['{pred_col}']")

        missing_join = [key for key in self._join_keys if key not in cols]
        if missing_join == [era_col] and id_col in cols and self._id_to_era is not None:
            frame = frame.join(self._id_to_era, on=id_col, how="inner")
            cols = set(frame.columns)
            missing_join = [key for key in self._join_keys if key not in cols]

        if missing_join:
            raise ValueError(f"Missing required columns: {missing_join}")

        cleaned = (
            frame.select([*self._join_keys, pred_col])
            .drop_nulls()
            .with_columns(pl.col(pred_col).cast(pl.Float64, strict=False))
            .drop_nulls()
            .filter(pl.col(pred_col).is_finite())
            .sort(self._join_keys)
            .unique(subset=self._join_keys, keep="first")
        )
        if cleaned.height != frame.height:
            logger.warning(
                "[normalize] dropped %d of %d rows (null/non-finite preds or "
                "duplicate (era, id) pairs — first in sorted order kept)",
                frame.height - cleaned.height, frame.height,
            )

        if cleaned.is_empty():
            raise ValueError("No valid prediction rows after normalization")
        return cleaned

    def _trivial_prediction_frame(self) -> pl.DataFrame:
        cfg = self._eval_cfg
        feature_cols = [
            c for c in self._features.columns if c not in set(self._join_keys)
        ]
        if not feature_cols:
            raise ValueError("features must contain at least one feature column")

        frame = self._features.select([*self._join_keys, *feature_cols]).with_columns(
            pl.mean_horizontal(
                [pl.col(c).cast(pl.Float64, strict=False) for c in feature_cols]
            ).alias(cfg.pred_col)
        )
        return frame.select([*self._join_keys, cfg.pred_col]).sort(self._join_keys)

    def _walk_forward_model_predictions(
        self,
        *,
        model_name: str,
        min_train_eras: int,
    ) -> pl.DataFrame:
        cfg = self._eval_cfg
        feature_cols = [
            c for c in self._features.columns if c not in set(self._join_keys)
        ]
        if not feature_cols:
            raise ValueError("features must contain at least one feature column")

        train_frame = (
            self._targets.select([*self._join_keys, cfg.main_target])
            .join(
                self._features.select([*self._join_keys, *feature_cols]),
                on=self._join_keys,
                how="inner",
            )
            .drop_nulls()
        )
        if train_frame.is_empty():
            raise ValueError("No rows available for classical baseline training")

        eras = sorted(train_frame.get_column(cfg.era_col).unique().to_list(), key=int)
        if len(eras) <= min_train_eras:
            raise ValueError(
                "Not enough eras for walk-forward baselines: "
                f"have {len(eras)}, need > {min_train_eras}"
            )

        parts: list[pl.DataFrame] = []
        for idx in range(min_train_eras, len(eras)):
            logger.info(
                "[walk_forward] %s baseline: era %d/%d (train %d eras -> predict %s)",
                model_name, idx - min_train_eras + 1, len(eras) - min_train_eras,
                idx, eras[idx],
            )
            train_eras = eras[:idx]
            test_era = eras[idx]

            train_part = train_frame.filter(pl.col(cfg.era_col).is_in(train_eras))
            test_part = train_frame.filter(pl.col(cfg.era_col) == test_era)
            if train_part.is_empty() or test_part.is_empty():
                continue

            x_train = train_part.select(feature_cols).cast(pl.Float64).to_pandas()
            y_train = train_part.get_column(cfg.main_target).cast(pl.Float64).to_numpy()
            x_test = test_part.select(feature_cols).cast(pl.Float64).to_pandas()

            model = self._build_classical_model(model_name)
            model.fit(x_train, y_train)
            pred = np.asarray(model.predict(x_test), dtype=float)

            parts.append(
                test_part.select(self._join_keys).with_columns(
                    pl.Series(cfg.pred_col, pred)
                )
            )

        if not parts:
            raise ValueError("No walk-forward predictions generated")
        return pl.concat(parts, how="vertical").sort(self._join_keys)

    def _build_classical_model(self, name: str) -> Any:
        if name == "linear":
            return Ridge(alpha=1.0, random_state=self._eval_cfg.seed)

        if name == "tree":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.1,
                random_state=self._eval_cfg.seed,
                n_jobs=1,
                verbose=-1,
            )

        raise ValueError(f"Unknown classical model name {name!r}")

    @staticmethod
    def _as_finite_vector(
        values: pl.Series | np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        arr = (
            values.cast(pl.Float64, strict=False).to_numpy()
            if isinstance(values, pl.Series)
            else np.asarray(values, dtype=float)
        )
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-D")
        if arr.size == 0:
            raise ValueError(f"{name} must be non-empty")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values")
        return arr

    @staticmethod
    def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
        if left.size < 2 or right.size < 2:
            return 0.0
        left_centered = left - np.mean(left)
        right_centered = right - np.mean(right)
        denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
        if denom == 0.0 or not np.isfinite(denom):
            return 0.0
        return float((left_centered @ right_centered) / denom)

    def _orthogonality_stat(self, joint: np.ndarray) -> tuple[float, float, float]:
        if joint.ndim != 2 or joint.shape[1] != 2:
            raise ValueError("joint must be shaped (n, 2)")
        cand = joint[:, 0]
        book = joint[:, 1]
        global_rho = self._safe_corr(cand, book)

        threshold = float(np.quantile(book, 0.10))
        mask = book <= threshold
        tail_rho = self._safe_corr(cand[mask], book[mask])
        spread = float(tail_rho - global_rho)
        return float(global_rho), float(tail_rho), spread


def ingest_tutorial_prediction(
    notebook_path: str | Path,
    prediction_path: str | Path,
    *,
    model_id: str | None = None,
    pred_col: str = "prediction",
    id_col: str = "id",
    era_col: str = "era",
) -> tuple[str, pl.DataFrame]:
    """Load a tutorial prediction vector and normalize it to a standard schema."""

    nb_path = Path(notebook_path)
    pred_path = Path(prediction_path)
    _verify_notebook_contract(nb_path)

    resolved_model_id = model_id
    if resolved_model_id is None:
        resolved_model_id = TUTORIAL_NOTEBOOK_TO_MODEL_ID.get(
            nb_path.name, nb_path.stem
        )

    frame = _read_prediction_file(pred_path)
    columns = set(frame.columns)

    if pred_col not in columns:
        pred_candidate = _infer_prediction_column(
            frame.columns,
            frame.dtypes,
            pred_col=pred_col,
            era_col=era_col,
            id_col=id_col,
        )
        if pred_candidate is None:
            raise ValueError(f"Missing required prediction column {pred_col!r}")
        frame = frame.rename({pred_candidate: pred_col})
        columns = set(frame.columns)

    if id_col not in columns:
        id_candidate = _infer_id_column(
            frame.columns, pred_col=pred_col, era_col=era_col
        )
        if id_candidate is None:
            raise ValueError(f"Missing required id column {id_col!r}")
        frame = frame.rename({id_candidate: id_col})

    keep_cols = [id_col, pred_col]
    if era_col in frame.columns:
        keep_cols = [era_col, *keep_cols]

    normalized = (
        frame.select(keep_cols)
        .drop_nulls()
        .with_columns(pl.col(pred_col).cast(pl.Float64, strict=False))
        .drop_nulls()
        .filter(pl.col(pred_col).is_finite())
    )
    if normalized.is_empty():
        raise ValueError("No valid rows in tutorial prediction artifact")

    return resolved_model_id, normalized


def ingest_tutorial_prediction_batch(
    source_root: str | Path,
    prediction_files: Mapping[str, str | Path],
    *,
    pred_col: str = "prediction",
    id_col: str = "id",
    era_col: str = "era",
) -> dict[str, pl.DataFrame]:
    """Ingest multiple tutorial notebook prediction artifacts in one pass.

    Keys in ``prediction_files`` can be notebook filenames or model ids.
    """

    root = Path(source_root)
    out: dict[str, pl.DataFrame] = {}

    for key, prediction_path in prediction_files.items():
        nb_path = _resolve_notebook_path(root, key)
        model_id = TUTORIAL_NOTEBOOK_TO_MODEL_ID.get(nb_path.name, nb_path.stem)
        model_id, frame = ingest_tutorial_prediction(
            nb_path,
            prediction_path,
            model_id=model_id,
            pred_col=pred_col,
            id_col=id_col,
            era_col=era_col,
        )
        out[model_id] = frame

    return out


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


def discover_tutorial_notebooks(root: str | Path) -> dict[str, Path]:
    """Resolve required Slice 1 tutorial notebook paths from a root directory."""

    root_path = Path(root)
    found: dict[str, Path] = {}
    missing: list[str] = []
    for notebook_name in TUTORIAL_NOTEBOOK_TO_MODEL_ID:
        path = root_path / notebook_name
        if path.exists():
            found[notebook_name] = path
        else:
            missing.append(notebook_name)

    if missing:
        raise FileNotFoundError(
            "Missing tutorial notebooks: " + ", ".join(sorted(missing))
        )
    return found


def assert_notebook_prediction_contract(path: str | Path) -> None:
    """Public contract check for tutorial notebook prediction anchors."""
    _verify_notebook_contract(Path(path))


def extract_oos_predictions(
    source: pl.DataFrame | str | Path,
    *,
    id_to_era: pl.DataFrame | None = None,
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
) -> pl.DataFrame:
    """Normalize raw out-of-sample predictions to [era, id, prediction].

    Supports direct DataFrame input or file paths to csv/parquet artifacts.
    """

    if isinstance(source, pl.DataFrame):
        frame = source.clone()
    else:
        frame = _read_prediction_file(Path(source))

    if id_col not in frame.columns:
        candidate = _infer_id_column(frame.columns, pred_col=pred_col, era_col=era_col)
        if candidate is not None:
            frame = frame.rename({candidate: id_col})

    if pred_col not in frame.columns:
        pred_candidate = _infer_prediction_column(
            frame.columns,
            frame.dtypes,
            pred_col=pred_col,
            era_col=era_col,
            id_col=id_col,
        )
        if pred_candidate is not None:
            frame = frame.rename({pred_candidate: pred_col})

    missing = [name for name in (id_col, pred_col) if name not in frame.columns]
    if missing:
        raise ValueError(f"Predictions are missing required columns: {missing}")

    if era_col not in frame.columns:
        if id_to_era is None:
            raise ValueError(
                f"Predictions are missing {era_col!r}; provide id_to_era mapping"
            )
        required_map = [id_col, era_col]
        missing_map = [name for name in required_map if name not in id_to_era.columns]
        if missing_map:
            raise ValueError(f"id_to_era missing required columns: {missing_map}")
        frame = frame.join(
            id_to_era.select(required_map),
            on=id_col,
            how="left",
        )

    out = (
        frame.select([era_col, id_col, pred_col])
        .drop_nulls()
        .with_columns(pl.col(pred_col).cast(pl.Float64, strict=False))
        .drop_nulls()
        .filter(pl.col(pred_col).is_finite())
        .sort([era_col, id_col])
    )
    if out.is_empty():
        raise ValueError("No usable prediction rows after normalization")
    return out


def assert_null_floor(
    scorecards: Mapping[str, MetricScorecard],
    *,
    tolerance: float = 0.05,
    metric_tolerances: Mapping[str, float] | None = None,
) -> None:
    """Ensure null baselines remain near zero on core skill metrics."""

    tol = float(tolerance)
    metric_tol = dict(metric_tolerances or {})
    for name in NULL_BASELINES:
        if name not in scorecards:
            raise ValueError(f"Missing null baseline scorecard {name!r}")

    for name in NULL_BASELINES:
        score = scorecards[name]
        _assert_scorecard_finite(score, model_id=name)

        checks = {
            "rank_scalar": score.rank_scalar,
            "mean_payout": score.mean_payout.value,
            "corr": score.corr.value,
            "mmc": score.mmc.value,
            "fnc": score.fnc,
            "corr_sharpe_ac": score.corr_sharpe_ac.value,
        }
        if score.bmc is not None:
            checks["bmc"] = score.bmc.value
        if score.cwmm is not None:
            checks["cwmm"] = score.cwmm.value
        for metric_name, value in checks.items():
            threshold = float(metric_tol.get(metric_name, tol))
            if abs(float(value)) > threshold:
                raise ValueError(
                    "Null floor violation for "
                    f"{name}.{metric_name}: observed={value:.8f}, tolerance={threshold:.8f}"
                )


def assert_slice1_monotone(
    scorecards: Mapping[str, MetricScorecard],
    *,
    hello_model_id: str = "hello-numerai",
    sunshine_model_id: str = "sunshine",
    atol: float = 0.0,
) -> None:
    """Check monotone payout-proxy ordering: null floor <= hello <= sunshine."""

    for key in (hello_model_id, sunshine_model_id):
        if key not in scorecards:
            raise ValueError(f"Missing required scorecard {key!r}")

    null_values = [
        float(scorecards[name].rank_scalar)
        for name in NULL_BASELINES
        if name in scorecards
    ]
    if len(null_values) != len(NULL_BASELINES):
        missing = [name for name in NULL_BASELINES if name not in scorecards]
        raise ValueError(f"Missing null baselines for monotone check: {missing}")

    null_floor = max(null_values)
    hello = float(scorecards[hello_model_id].rank_scalar)
    sunshine = float(scorecards[sunshine_model_id].rank_scalar)
    tol = float(atol)

    if hello + tol < null_floor:
        raise ValueError(
            "Monotone violation: hello below null floor "
            f"(hello={hello:.8f}, null_floor={null_floor:.8f}, atol={tol:.8f})"
        )
    if sunshine + tol < hello:
        raise ValueError(
            "Monotone violation: sunshine below hello "
            f"(sunshine={sunshine:.8f}, hello={hello:.8f}, atol={tol:.8f})"
        )


def canonical_scorecards_bytes(scorecards: Mapping[str, MetricScorecard]) -> bytes:
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


def _verify_notebook_contract(notebook_path: Path) -> None:
    notebook_name = notebook_path.name
    if notebook_name not in _TUTORIAL_NOTEBOOK_ANCHORS:
        known = sorted(_TUTORIAL_NOTEBOOK_ANCHORS)
        raise ValueError(f"Notebook {notebook_name!r} not in tutorial roster: {known}")

    try:
        raw = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Failed to parse notebook JSON for {notebook_name!r}"
        ) from exc

    cells = raw.get("cells", []) if isinstance(raw, dict) else []
    source_parts: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        source = cell.get("source", [])
        if isinstance(source, str):
            source_parts.append(source)
        elif isinstance(source, list):
            source_parts.extend(str(line) for line in source)

    text = "\n".join(source_parts)
    anchors = _TUTORIAL_NOTEBOOK_ANCHORS[notebook_name]
    missing = [anchor for anchor in anchors if anchor not in text]
    if missing:
        raise ValueError(
            f"Notebook {notebook_name!r} does not match expected extraction anchors: {missing}"
        )


def _resolve_notebook_path(root: Path, key: str) -> Path:
    candidate = root / key
    if candidate.exists():
        return candidate

    for notebook_name, model_id in TUTORIAL_NOTEBOOK_TO_MODEL_ID.items():
        if key in {notebook_name, model_id, Path(notebook_name).stem}:
            resolved = root / notebook_name
            if not resolved.exists():
                raise FileNotFoundError(f"Tutorial notebook not found: {resolved}")
            return resolved

    raise FileNotFoundError(f"Could not resolve tutorial notebook for key {key!r}")


def _read_prediction_file(path: Path) -> pl.DataFrame:
    suffixes = tuple(s.lower() for s in path.suffixes)
    if suffixes and suffixes[-1] == ".parquet":
        return pl.read_parquet(path)
    if suffixes[-1:] == (".csv",):
        return pl.read_csv(path)
    if suffixes[-2:] == (".csv", ".gz"):
        return pl.read_csv(path)
    raise ValueError(f"Unsupported prediction artifact format: {path}")


def _infer_id_column(
    columns: Sequence[str],
    *,
    pred_col: str,
    era_col: str,
) -> str | None:
    non_metric = [col for col in columns if col not in {pred_col, era_col}]
    if not non_metric:
        return None

    normalized = {col.lower(): col for col in non_metric}
    for alias in ("id", "index", "unnamed: 0", "column_1", ""):
        if alias in normalized:
            return normalized[alias]
    logger.warning(
        "[tutorial] no known id alias in columns %r; inferring first non-metric column %r",
        columns, non_metric[0],
    )
    return non_metric[0]


def _infer_prediction_column(
    columns: Sequence[str],
    dtypes: Sequence[pl.DataType],
    *,
    pred_col: str,
    era_col: str,
    id_col: str,
) -> str | None:
    normalized = {col.lower(): col for col in columns}
    aliases = (
        pred_col.lower(),
        "prediction",
        "predictions",
        "pred",
        "score",
        "scores",
        "model_prediction",
        "numerai_prediction",
    )
    for alias in aliases:
        resolved = normalized.get(alias)
        if resolved is not None and resolved not in {era_col, id_col}:
            return resolved

    numeric_candidates = [
        col
        for col, dtype in zip(columns, dtypes)
        if dtype.is_numeric() and col not in {era_col, id_col}
    ]
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    pred_like = [
        col
        for col in numeric_candidates
        if "pred" in col.lower() or "score" in col.lower()
    ]
    if len(pred_like) == 1:
        return pred_like[0]

    return None


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
    """Standardize with train statistics; zero-variance features -> 0.0."""
    mu = np.mean(train_values, axis=0)
    sigma = np.std(train_values, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    scale = np.where((sigma > 0.0) & np.isfinite(sigma), 1.0 / sigma, 0.0)
    return (train_values - mu) * scale, (val_values - mu) * scale


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

    x_train_raw = train_rows.select(feature_cols).cast(pl.Float64).to_numpy()
    x_val_raw = val_rows.select(feature_cols).cast(pl.Float64).to_numpy()
    x_train, x_val = _standardize_feature_block(x_train_raw, x_val_raw)

    component_preds: dict[str, np.ndarray] = {}
    for target in targets:
        if target not in train.columns:
            raise ValueError(f"missing target column: {target!r}")
        y = train_rows.get_column(target).cast(pl.Float64).to_numpy()
        mask = np.isfinite(y)
        if mask.sum() < 2:
            raise ValueError(
                f"target {target!r} has fewer than 2 finite train rows after purge"
            )
        model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=seed)
        model.fit(x_train[mask], y[mask])
        component_preds[target] = np.asarray(model.predict(x_val), dtype=float)

    frame = val_rows.select([era_col, id_col]).with_columns(
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

    x_train = train_rows.select(feature_cols).cast(pl.Float64).to_pandas()
    y = train_rows.get_column(target).cast(pl.Float64).to_numpy()
    mask = np.isfinite(y)
    if mask.sum() < 2:
        raise ValueError(f"target {target!r} has fewer than 2 finite train rows after purge")
    x_val = val_rows.select(feature_cols).cast(pl.Float64).to_pandas()

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
    sharpe_tol: float = 0.10,
    dsr_tol: float = 0.05,
) -> None:
    """Tier-0 sanity gate: null baselines must score at the statistical floor."""
    for name in NULL_KINDS:
        if name not in scorecards:
            raise ValueError(f"Missing null baseline scorecard {name!r}")

    for name in NULL_KINDS:
        score = scorecards[name]
        _assert_scorecard_finite(score, model_id=name)
        checks = (
            ("corr", float(score.corr.value), float(corr_tol)),
            ("corr_sharpe_ac", float(score.corr_sharpe_ac.value), float(sharpe_tol)),
            ("deflated_sharpe", float(score.deflated_sharpe), float(dsr_tol)),
        )
        for metric_name, observed, tolerance in checks:
            if abs(observed) > tolerance:
                raise ValueError(
                    "Null floor violation for "
                    f"{name}.{metric_name}: |{observed:.8f}| > {tolerance:.8f}"
                )


def assert_tier4_gate(scorecard: MetricScorecard, gate: Tier4GateConfig) -> None:
    """Production capital gate: reject candidates below the 7 hard thresholds."""
    _assert_scorecard_finite(scorecard, model_id=scorecard.model_id)
    violations: list[str] = []

    def _check(field: str, observed: float, threshold: float, strict: bool) -> None:
        if strict:
            if observed <= threshold:
                violations.append(
                    f"{field}: observed={observed:.8f}, need > {threshold:.8f}"
                )
        elif observed < threshold:
            violations.append(
                f"{field}: observed={observed:.8f}, need >= {threshold:.8f}"
            )

    if scorecard.turnover_mean is None:
        violations.append(
            f"turnover_mean: unavailable (reason={scorecard.turnover_reason!r}); "
            f"cannot verify <= {gate.turnover_max:.4f}"
        )
    else:
        if float(scorecard.turnover_mean) > float(gate.turnover_max):
            violations.append(
                "turnover_mean: "
                f"observed={float(scorecard.turnover_mean):.8f}, "
                f"need <= {gate.turnover_max:.4f}"
            )

    _check("corr", float(scorecard.corr.value), float(gate.corr_min), strict=False)
    _check(
        "corr_sharpe_ac",
        float(scorecard.corr_sharpe_ac.value),
        float(gate.corr_sharpe_ac_min),
        strict=False,
    )
    _check("fnc", float(scorecard.fnc), float(gate.fnc_min), strict=False)
    _check(
        "deflated_sharpe",
        float(scorecard.deflated_sharpe),
        float(gate.deflated_sharpe_min),
        strict=False,
    )
    _check(
        "gain_to_pain_ratio",
        float(scorecard.gain_to_pain_ratio),
        float(gate.gain_to_pain_min),
        strict=False,
    )
    _check("cagr_1y", float(scorecard.cagr_1y), float(gate.cagr_min), strict=True)

    if violations:
        raise ValueError(
            f"Tier-4 gate violations for {scorecard.model_id!r}: "
            + "; ".join(violations)
        )


def assert_hierarchy_monotone(
    scorecards: Mapping[str, MetricScorecard],
    *,
    tier_of: Mapping[str, int],
    atol: float = 1e-5,
) -> None:
    """Assert escalating tier ordering on the rank scalar (T0 < T1 < T2 < T3 <= T4)."""
    tiers_present = sorted(set(tier_of.values()))
    if tiers_present != [0, 1, 2, 3, 4]:
        raise ValueError(f"tier_of must cover all tiers 0..4, got {tiers_present}")

    scalar_by_tier: dict[int, float] = {}
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
