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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from nmr.evaluation import MIN_OVERLAP_ERAS
from nmr.inference import block_bootstrap_ci, resolve_block_len
from nmr.scorecard import MetricScorecard, evaluate_model
from sklearn.linear_model import Ridge

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

    def run_classical_baselines(
        self,
        *,
        min_train_eras: int = 10,
    ) -> dict[str, MetricScorecard]:
        """Generate and score S11 classical rungs: trivial, linear, and tree."""

        trivial = self._trivial_prediction_frame()
        linear = self._walk_forward_model_predictions(
            model_name="linear",
            min_train_eras=min_train_eras,
        )
        tree = self._walk_forward_model_predictions(
            model_name="tree",
            min_train_eras=min_train_eras,
        )

        return {
            "trivial": self.evaluate_predictions(trivial, model_id="trivial"),
            "linear": self.evaluate_predictions(linear, model_id="linear"),
            "tree": self.evaluate_predictions(tree, model_id="tree"),
        }

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
        run_seed = self._eval_cfg.seed if seed is None else int(seed)
        cfg = self._eval_cfg
        return evaluate_model(
            normalized,
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
            .unique(subset=self._join_keys, keep="first")
            .sort(self._join_keys)
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
            try:
                from lightgbm import LGBMRegressor

                return LGBMRegressor(
                    n_estimators=120,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    random_state=self._eval_cfg.seed,
                    n_jobs=1,
                    verbose=-1,
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor

                return GradientBoostingRegressor(random_state=self._eval_cfg.seed)

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
        numeric_candidates = [
            col for col, dtype in zip(frame.columns, frame.dtypes) if dtype.is_numeric()
        ]
        if len(numeric_candidates) == 1:
            frame = frame.rename({numeric_candidates[0]: pred_col})
            columns = set(frame.columns)
        else:
            raise ValueError(f"Missing required prediction column {pred_col!r}")

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
        numeric_candidates = [
            col for col, dtype in zip(frame.columns, frame.dtypes) if dtype.is_numeric()
        ]
        if len(numeric_candidates) == 1:
            frame = frame.rename({numeric_candidates[0]: pred_col})

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
    payload = {
        model_id: _sanitize_json_payload(scorecards[model_id].to_frame().to_dicts()[0])
        for model_id in sorted(scorecards)
    }
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

    text = notebook_path.read_text(encoding="utf-8")
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
    return non_metric[0]


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
