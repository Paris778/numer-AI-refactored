"""Robustness diagnostics for Evaluation Suite v2.

Tier-2 diagnostics that flag instability for investigation. This module never
implements inference math directly: confidence intervals and autocorrelation-
adjusted Sharpe are delegated to `nmr.inference`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import polars as pl

from nmr._transforms import tie_kept_rank
from nmr.evaluation import MIN_OVERLAP_ERAS, EvaluationEngine, NonVacuityError
from nmr.inference import ac_adjusted_sharpe, block_bootstrap_ci, resolve_block_len

__all__ = [
    "PerturbationResult",
    "HorizonStabilityResult",
    "RegimeCorr",
    "adversarial_perturbation",
    "time_horizon_stability",
    "regime_conditioned_corr",
]


@dataclass(frozen=True)
class PerturbationResult:
    alpha: float
    n_eras: int
    ceiling_stability: float
    manifold_stability: float
    gap: float
    effective_perturb_frac: float


@dataclass(frozen=True)
class HorizonStabilityResult:
    target_name: str
    n_eras: int
    model_sharpe_20: float
    model_sharpe_60: float
    model_decay: float
    benchmark_decay: float
    relative_divergence: float


@dataclass(frozen=True)
class RegimeCorr:
    regime: str
    n_eras: int
    mean_corr: float
    ci_low: float
    ci_high: float


def _seed_from_parts(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _numeric_sorted_eras(labels: list[str]) -> list[str]:
    numeric_to_label: dict[int, str] = {}
    for label in labels:
        try:
            numeric_label = int(label)
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric era label {label!r}; robustness requires chronological eras"
            ) from exc
        numeric_to_label.setdefault(numeric_label, label)
    return [numeric_to_label[num] for num in sorted(numeric_to_label)]


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        return 0.0
    return float((left_centered @ right_centered) / denominator)


def _validate_columns(df: pl.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_bin_features(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    values = df.select(feature_cols).cast(pl.Float64).to_numpy()
    if values.size == 0:
        raise ValueError("features must be Int8 bins in [0,4]")
    finite = np.isfinite(values)
    in_range = (values >= 0.0) & (values <= 4.0)
    integer_bins = np.equal(values, np.floor(values))
    if not (finite.all() and in_range.all() and integer_bins.all()):
        raise ValueError("features must be Int8 bins in [0,4]")
    return values.astype(np.int8, copy=False)


def _validate_blocks(
    feature_cols: list[str],
    blocks: dict[str, list[str]],
) -> list[np.ndarray]:
    if not blocks:
        raise ValueError("blocks must contain at least one block")

    feature_to_idx = {name: idx for idx, name in enumerate(feature_cols)}
    block_indices: list[np.ndarray] = []
    for block_name, cols in blocks.items():
        if not cols:
            raise ValueError(f"block {block_name!r} must contain at least one feature")
        idxs: list[int] = []
        for col in cols:
            if col not in feature_to_idx:
                raise ValueError(
                    f"block {block_name!r} references unknown feature {col!r}"
                )
            idxs.append(feature_to_idx[col])
        block_indices.append(np.asarray(idxs, dtype=int))
    return block_indices


def _build_perturbation_field(
    *,
    n_eval: int,
    n_features: int,
    n_train: int,
    n_blocks: int,
    alpha: float,
    seed: int,
    data_version: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_seed = _seed_from_parts(
        data_version,
        seed,
        n_eval,
        n_features,
        n_train,
        n_blocks,
        alpha,
    )
    rng_cell = np.random.default_rng(_seed_from_parts(base_seed, "cell"))
    rng_block = np.random.default_rng(_seed_from_parts(base_seed, "block"))

    cell_mask = rng_cell.random((n_eval, n_features)) < alpha
    cell_dir = rng_cell.choice(
        np.array([-1, 1], dtype=np.int8), size=(n_eval, n_features)
    )
    block_mask = rng_block.random((n_eval, n_blocks)) < alpha
    block_donors = rng_block.integers(0, n_train, size=(n_eval, n_blocks))
    return cell_mask, cell_dir, block_mask, block_donors


def _perturb_independent(
    clean: np.ndarray,
    *,
    cell_mask: np.ndarray,
    cell_dir: np.ndarray,
) -> tuple[np.ndarray, float]:
    shifted = clean + (cell_mask.astype(np.int8) * cell_dir)
    perturbed = np.clip(shifted, 0, 4).astype(np.int8, copy=False)
    changed = perturbed != clean
    effective_frac = float(np.mean(changed))
    return perturbed, effective_frac


def _perturb_block_swap(
    clean: np.ndarray,
    train: np.ndarray,
    *,
    block_indices: list[np.ndarray],
    block_mask: np.ndarray,
    block_donors: np.ndarray,
) -> np.ndarray:
    perturbed = clean.copy()
    for block_i, idxs in enumerate(block_indices):
        row_mask = block_mask[:, block_i]
        if not np.any(row_mask):
            continue
        rows = np.flatnonzero(row_mask)
        donors = block_donors[rows, block_i]
        perturbed[np.ix_(rows, idxs)] = train[np.ix_(donors, idxs)]
    return perturbed.astype(np.int8, copy=False)


def _predict_vector(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    features: np.ndarray,
) -> np.ndarray:
    pred = np.asarray(predict_fn(features), dtype=float).reshape(-1)
    if pred.shape[0] != features.shape[0]:
        raise ValueError("predict_fn must return a vector with one score per row")
    if not np.all(np.isfinite(pred)):
        raise ValueError("predict_fn output must contain only finite values")
    return pred


def _per_era_spearman_mean(
    era_labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[float, int]:
    unique_eras = _numeric_sorted_eras([str(e) for e in np.unique(era_labels).tolist()])
    scores: list[float] = []
    for era in unique_eras:
        mask = era_labels == era
        lvals = left[mask]
        rvals = right[mask]
        if lvals.size < 2 or rvals.size < 2:
            scores.append(0.0)
            continue
        lrank = tie_kept_rank(lvals)
        rrank = tie_kept_rank(rvals)
        scores.append(_pearson(lrank, rrank))
    return float(np.mean(np.asarray(scores, dtype=float))), len(unique_eras)


def adversarial_perturbation(
    eval_features: pl.DataFrame,
    train_features: pl.DataFrame,
    *,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    feature_cols: list[str],
    blocks: dict[str, list[str]],
    era_col: str = "era",
    alpha: float = 0.1,
    seed: int,
    data_version: str = "v5.2",
) -> PerturbationResult:
    if not (0.1 <= alpha <= 0.25):
        raise ValueError("alpha must satisfy 0.1 <= alpha <= 0.25")

    _validate_columns(eval_features, [era_col, *feature_cols])
    _validate_columns(train_features, feature_cols)
    block_indices = _validate_blocks(feature_cols, blocks)

    eval_frame = eval_features.select([era_col, *feature_cols])
    train_frame = train_features.select(feature_cols)
    eval_bins = _validate_bin_features(eval_frame, feature_cols)
    train_bins = _validate_bin_features(train_frame, feature_cols)
    if train_bins.shape[0] < 1:
        raise ValueError("train_features must contain at least one row")

    cell_mask, cell_dir, block_mask, block_donors = _build_perturbation_field(
        n_eval=eval_bins.shape[0],
        n_features=eval_bins.shape[1],
        n_train=train_bins.shape[0],
        n_blocks=len(block_indices),
        alpha=float(alpha),
        seed=seed,
        data_version=data_version,
    )

    independent_bins, effective_frac = _perturb_independent(
        eval_bins,
        cell_mask=cell_mask,
        cell_dir=cell_dir,
    )
    block_bins = _perturb_block_swap(
        eval_bins,
        train_bins,
        block_indices=block_indices,
        block_mask=block_mask,
        block_donors=block_donors,
    )

    clean_pred = _predict_vector(predict_fn, eval_bins.astype(float, copy=False))
    independent_pred = _predict_vector(
        predict_fn,
        independent_bins.astype(float, copy=False),
    )
    block_pred = _predict_vector(predict_fn, block_bins.astype(float, copy=False))

    era_labels = eval_frame.get_column(era_col).cast(pl.Utf8).to_numpy()
    ceiling_stability, n_eras = _per_era_spearman_mean(
        era_labels,
        clean_pred,
        independent_pred,
    )
    manifold_stability, _ = _per_era_spearman_mean(era_labels, clean_pred, block_pred)
    gap = float(ceiling_stability - manifold_stability)

    return PerturbationResult(
        alpha=float(alpha),
        n_eras=n_eras,
        ceiling_stability=float(ceiling_stability),
        manifold_stability=float(manifold_stability),
        gap=gap,
        effective_perturb_frac=float(effective_frac),
    )


def time_horizon_stability(
    df: pl.DataFrame,
    *,
    pred_col: str,
    benchmark_col: str,
    target_name: str,
    era_col: str = "era",
    min_overlap_eras: int = MIN_OVERLAP_ERAS,
) -> HorizonStabilityResult:
    if min_overlap_eras < 1:
        raise ValueError("min_overlap_eras must be >= 1")

    target_20 = f"target_{target_name}_20"
    target_60 = f"target_{target_name}_60"
    _validate_columns(df, [era_col, pred_col, benchmark_col, target_20, target_60])

    engine = EvaluationEngine("custom")

    # Resolve eras before scoring: unresolved null-target eras must not be
    # injected as synthetic 0.0 corr into horizon statistics.
    all_eras = engine._sorted_labels(df.get_column(era_col).to_list())
    resolved_eras: list[str] = []
    for era in all_eras:
        era_df = df.filter(pl.col(era_col) == era)
        pred_20 = engine._clean_frame(era_df, [pred_col, target_20]).height
        pred_60 = engine._clean_frame(era_df, [pred_col, target_60]).height
        bench_20 = engine._clean_frame(era_df, [benchmark_col, target_20]).height
        bench_60 = engine._clean_frame(era_df, [benchmark_col, target_60]).height
        if min(pred_20, pred_60, bench_20, bench_60) >= 2:
            resolved_eras.append(era)

    model_20 = engine.per_era_corr(
        df, pred_col=pred_col, target_col=target_20, era_col=era_col
    )
    model_60 = engine.per_era_corr(
        df, pred_col=pred_col, target_col=target_60, era_col=era_col
    )
    bench_20 = engine.per_era_corr(
        df,
        pred_col=benchmark_col,
        target_col=target_20,
        era_col=era_col,
    )
    bench_60 = engine.per_era_corr(
        df,
        pred_col=benchmark_col,
        target_col=target_60,
        era_col=era_col,
    )

    overlap = _numeric_sorted_eras(
        list(
            set(model_20)
            & set(model_60)
            & set(bench_20)
            & set(bench_60)
            & set(resolved_eras)
        )
    )
    if len(overlap) < min_overlap_eras:
        raise NonVacuityError(
            "Non-vacuity violation: horizon overlap yielded only "
            f"{len(overlap)} eras; minimum required {min_overlap_eras}."
        )

    model_20_series = np.asarray([model_20[e] for e in overlap], dtype=float)
    model_60_series = np.asarray([model_60[e] for e in overlap], dtype=float)
    bench_20_series = np.asarray([bench_20[e] for e in overlap], dtype=float)
    bench_60_series = np.asarray([bench_60[e] for e in overlap], dtype=float)

    model_sharpe_20 = ac_adjusted_sharpe(model_20_series, horizon="20D")
    model_sharpe_60 = ac_adjusted_sharpe(model_60_series, horizon="60D")
    benchmark_sharpe_20 = ac_adjusted_sharpe(bench_20_series, horizon="20D")
    benchmark_sharpe_60 = ac_adjusted_sharpe(bench_60_series, horizon="60D")

    model_decay = float(model_sharpe_20 - model_sharpe_60)
    benchmark_decay = float(benchmark_sharpe_20 - benchmark_sharpe_60)
    relative_divergence = float(model_decay - benchmark_decay)

    return HorizonStabilityResult(
        target_name=target_name,
        n_eras=len(overlap),
        model_sharpe_20=float(model_sharpe_20),
        model_sharpe_60=float(model_sharpe_60),
        model_decay=model_decay,
        benchmark_decay=benchmark_decay,
        relative_divergence=relative_divergence,
    )


def regime_conditioned_corr(
    df: pl.DataFrame,
    regime_labels: pl.DataFrame,
    *,
    pred_col: str,
    target_col: str,
    regime_col: str,
    era_col: str = "era",
    horizon: Literal["20D", "60D"],
    seed: int,
    n_boot: int = 1000,
    alpha: float = 0.05,
    min_eras_per_regime: int = MIN_OVERLAP_ERAS,
) -> dict[str, RegimeCorr]:
    if min_eras_per_regime < 1:
        raise ValueError("min_eras_per_regime must be >= 1")

    _validate_columns(df, [era_col, pred_col, target_col])
    _validate_columns(regime_labels, [era_col, regime_col])

    engine = EvaluationEngine("custom")
    per_era_corr = engine.per_era_corr(
        df, pred_col=pred_col, target_col=target_col, era_col=era_col
    )
    corr_frame = pl.DataFrame(
        {
            era_col: list(per_era_corr.keys()),
            "corr": list(per_era_corr.values()),
        }
    )
    label_frame = regime_labels.select([era_col, regime_col]).with_columns(
        pl.col(era_col).cast(pl.Utf8),
        pl.col(regime_col).cast(pl.Utf8),
    )
    joined = corr_frame.with_columns(pl.col(era_col).cast(pl.Utf8)).join(
        label_frame,
        on=era_col,
        how="inner",
    )
    if joined.is_empty():
        raise ValueError("No overlap eras between per-era corr and regime labels")

    regimes = sorted(joined.get_column(regime_col).unique().to_list())
    results: dict[str, RegimeCorr] = {}
    for regime in regimes:
        sub = joined.filter(pl.col(regime_col) == regime)
        eras = _numeric_sorted_eras(sub.get_column(era_col).to_list())
        series = np.asarray(
            [
                float(
                    sub.filter(pl.col(era_col) == era).get_column("corr").to_numpy()[0]
                )
                for era in eras
            ],
            dtype=float,
        )
        n = int(series.size)
        if n < min_eras_per_regime:
            raise NonVacuityError(
                f"Non-vacuity violation: regime '{regime}' yielded only {n} eras; "
                f"minimum required {min_eras_per_regime}."
            )
        block_len = resolve_block_len(n, horizon)
        ci = block_bootstrap_ci(
            series,
            lambda a: float(np.mean(a)),
            block_len=block_len,
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
        )
        mean_corr = float(np.mean(series))
        results[regime] = RegimeCorr(
            regime=regime,
            n_eras=n,
            mean_corr=mean_corr,
            ci_low=float(ci.lo),
            ci_high=float(ci.hi),
        )

    return results
