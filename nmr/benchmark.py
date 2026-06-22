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

import numpy as np
import polars as pl

from nmr.scorecard import MetricScorecard, evaluate_model

__all__ = [
    "NULL_BASELINES",
    "TUTORIAL_NOTEBOOK_TO_MODEL_ID",
    "BenchmarkSuite",
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
) -> None:
    """Ensure null baselines remain near zero on core skill metrics."""

    tol = float(tolerance)
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
        for metric_name, value in checks.items():
            if abs(float(value)) > tol:
                raise ValueError(
                    "Null floor violation for "
                    f"{name}.{metric_name}: observed={value:.8f}, tolerance={tol:.8f}"
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
        model_id: scorecards[model_id].to_frame().to_dicts()[0]
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
