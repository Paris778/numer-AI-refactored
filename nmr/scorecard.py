"""Evaluation Suite v2 scorecard aggregator.

This module composes metrics from E1-E4 and the evaluation engine into a
single structured scorecard. It does not define new statistical metrics.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from nmr.evaluation import (
    MIN_OVERLAP_ERAS,
    EvaluationEngine,
    NonVacuityError,
    downside_era_indices,
    per_era_turnover,
    sorted_era_labels,
)
from nmr.inference import (
    BootstrapCI,
    Horizon,
    ac_adjusted_sharpe,
    block_bootstrap_ci,
    era_series_stats,
    resolve_block_len,
)
from nmr.payout import payout_report
from nmr.research import feature_exposure_report
from nmr.robustness import (
    HorizonStabilityResult,
    PerturbationResult,
    RegimeCorr,
    regime_conditioned_corr,
    time_horizon_stability,
)

logger = logging.getLogger("nmr.scorecard")

_MMC_DOWN_MIN_ERAS = 5
MMC_DOWN_MIN_ERAS = _MMC_DOWN_MIN_ERAS  # public alias for cross-module use

__all__ = [
    "MetricCell",
    "MetricScorecard",
    "CrossCheckResult",
    "evaluate_model",
    "evaluate_cross_check",
]


@dataclass(frozen=True)
class MetricCell:
    value: float
    ci_low: float | None
    ci_high: float | None
    n_eras: int


@dataclass(frozen=True)
class MetricScorecard:
    model_id: str
    n_eras: int
    rank_scalar: float
    deflated_sharpe: float

    mean_payout: MetricCell
    corr: MetricCell
    mmc: MetricCell
    fnc: float
    corr_sharpe_ac: MetricCell
    cvar5: float
    max_drawdown: float
    burn_rate: float

    mmc_sharpe_ac: float
    sortino: float
    calmar: float
    std_corr: float
    max_burn_streak: int
    time_to_recovery: int
    horizon_stability: HorizonStabilityResult | None
    horizon_reason: str | None
    regime_corr: dict[str, RegimeCorr] | None
    regime_reason: str | None
    perturbation: PerturbationResult | None

    max_feature_exposure: float
    bmc: MetricCell | None
    bmc_reason: str | None
    cwmm: MetricCell | None
    cwmm_reason: str | None
    book_correlation: object | None

    cagr_1y: float
    gain_to_pain_ratio: float
    kelly_fraction: float
    mmc_down: float | None
    mmc_down_n_eras: int | None
    mmc_down_reason: str | None
    turnover_mean: float | None
    turnover_std: float | None
    turnover_reason: str | None
    sim_portfolio_cagr: float
    sim_portfolio_mdd: float
    sim_capital_utilization: float

    metric_timing_seconds: dict[str, float] | None
    eval_total_seconds: float
    # A4 (audit SEV-3): era labels whose per-era metrics short-circuited
    # (degenerate — 0.0-normalized at the engine boundary), so they are
    # distinguishable from genuine zero-IC eras. Deterministic; the count
    # enters the canonical frame.
    degenerate_eras: tuple[str, ...] = ()

    @property
    def n_degenerate_eras(self) -> int:
        return len(self.degenerate_eras)

    def to_frame(self) -> pl.DataFrame:
        row: dict[str, Any] = {
            "model_id": self.model_id,
            "n_eras": self.n_eras,
            "rank_scalar": self.rank_scalar,
            "deflated_sharpe": self.deflated_sharpe,
            "fnc": self.fnc,
            "cvar5": self.cvar5,
            "max_drawdown": self.max_drawdown,
            "burn_rate": self.burn_rate,
            "mmc_sharpe_ac": self.mmc_sharpe_ac,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "std_corr": self.std_corr,
            "max_burn_streak": self.max_burn_streak,
            "time_to_recovery": self.time_to_recovery,
            "max_feature_exposure": self.max_feature_exposure,
            "book_correlation": self.book_correlation,
            "cagr_1y": self.cagr_1y,
            "gain_to_pain_ratio": self.gain_to_pain_ratio,
            "kelly_fraction": self.kelly_fraction,
            "mmc_down": self.mmc_down,
            "mmc_down_n_eras": self.mmc_down_n_eras,
            "mmc_down_reason": self.mmc_down_reason,
            "turnover_mean": self.turnover_mean,
            "turnover_std": self.turnover_std,
            "turnover_reason": self.turnover_reason,
            "sim_portfolio_cagr": self.sim_portfolio_cagr,
            "sim_portfolio_mdd": self.sim_portfolio_mdd,
            "sim_capital_utilization": self.sim_capital_utilization,
            "quality_metric_total_seconds": self.eval_total_seconds,
            "n_degenerate_eras": self.n_degenerate_eras,
        }

        metric_timings = self.metric_timing_seconds or {}
        if metric_timings:
            row["quality_metric_timings_json"] = json.dumps(
                {k: float(metric_timings[k]) for k in sorted(metric_timings)},
                sort_keys=True,
            )
            for name in sorted(metric_timings):
                row[f"timing_{name}_seconds"] = float(metric_timings[name])
        else:
            row["quality_metric_timings_json"] = None

        self._flatten_metric_cell(row, "mean_payout", self.mean_payout)
        self._flatten_metric_cell(row, "corr", self.corr)
        self._flatten_metric_cell(row, "mmc", self.mmc)
        self._flatten_metric_cell(row, "corr_sharpe_ac", self.corr_sharpe_ac)
        self._flatten_metric_cell(row, "bmc", self.bmc)
        self._flatten_metric_cell(row, "cwmm", self.cwmm)

        if self.horizon_stability is None:
            row.update(
                {
                    "horizon_target_name": None,
                    "horizon_n_eras": None,
                    "horizon_model_sharpe_20": None,
                    "horizon_model_sharpe_60": None,
                    "horizon_model_decay": None,
                    "horizon_benchmark_decay": None,
                    "horizon_relative_divergence": None,
                    "horizon_reason": self.horizon_reason,
                }
            )
        else:
            h = self.horizon_stability
            row.update(
                {
                    "horizon_target_name": h.target_name,
                    "horizon_n_eras": h.n_eras,
                    "horizon_model_sharpe_20": h.model_sharpe_20,
                    "horizon_model_sharpe_60": h.model_sharpe_60,
                    "horizon_model_decay": h.model_decay,
                    "horizon_benchmark_decay": h.benchmark_decay,
                    "horizon_relative_divergence": h.relative_divergence,
                    "horizon_reason": self.horizon_reason,
                }
            )

        if self.perturbation is None:
            row.update(
                {
                    "perturb_alpha": None,
                    "perturb_n_eras": None,
                    "perturb_ceiling_stability": None,
                    "perturb_manifold_stability": None,
                    "perturb_gap": None,
                    "perturb_effective_perturb_frac": None,
                }
            )
        else:
            p = self.perturbation
            row.update(
                {
                    "perturb_alpha": p.alpha,
                    "perturb_n_eras": p.n_eras,
                    "perturb_ceiling_stability": p.ceiling_stability,
                    "perturb_manifold_stability": p.manifold_stability,
                    "perturb_gap": p.gap,
                    "perturb_effective_perturb_frac": p.effective_perturb_frac,
                }
            )

        if self.regime_corr is None:
            row["regime_count"] = None
            row["regime_min_n_eras"] = None
            row["regime_max_n_eras"] = None
            row["regime_corr_json"] = None
            row["regime_reason"] = self.regime_reason
        else:
            regimes = sorted(self.regime_corr)
            row["regime_count"] = len(regimes)
            counts = [self.regime_corr[r].n_eras for r in regimes]
            row["regime_min_n_eras"] = min(counts)
            row["regime_max_n_eras"] = max(counts)
            row["regime_corr_json"] = json.dumps(
                {
                    regime: {
                        "mean_corr": self.regime_corr[regime].mean_corr,
                        "ci_low": self.regime_corr[regime].ci_low,
                        "ci_high": self.regime_corr[regime].ci_high,
                        "n_eras": self.regime_corr[regime].n_eras,
                    }
                    for regime in regimes
                },
                sort_keys=True,
            )
            row["regime_reason"] = self.regime_reason

        row["bmc_reason"] = self.bmc_reason
        row["cwmm_reason"] = self.cwmm_reason

        return pl.DataFrame([row])

    @staticmethod
    def _flatten_metric_cell(
        row: dict[str, Any],
        name: str,
        cell: MetricCell | None,
    ) -> None:
        if cell is None:
            row[name] = None
            row[f"{name}_ci_low"] = None
            row[f"{name}_ci_high"] = None
            row[f"{name}_n_eras"] = None
            return
        row[name] = cell.value
        row[f"{name}_ci_low"] = cell.ci_low
        row[f"{name}_ci_high"] = cell.ci_high
        row[f"{name}_n_eras"] = cell.n_eras


@dataclass(frozen=True)
class CrossCheckResult:
    """Cross-check evaluation: scorecard + labeled per-era series + raw Sharpe."""

    scorecard: MetricScorecard
    per_era: dict[str, list[dict[str, float | str]]]
    raw_sharpe: float


# Fixed replay constants (spec §7) — the cross-check never varies these.
CROSSCHECK_N_TRIALS = 1


def _sorted_numeric_keys(values: dict[str, float]) -> list[str]:
    """Sort per-era keys chronologically; non-numeric eras raise (fail loud).

    A non-numeric era descriptor (e.g. live ``"X"``) in a scorecard metric
    series is a data-integrity anomaly — silently sorting it last would fold
    a bogus era into the aggregate, so it raises with a clear message instead.
    """
    try:
        return sorted(values, key=lambda x: int(x))
    except ValueError as exc:
        bad = [k for k in values if not str(k).lstrip("-").isdigit()]
        raise ValueError(
            "Non-numeric era keys in scorecard metric series: "
            f"{sorted(bad)} (validation eras must be numeric)"
        ) from exc


def _cell_from_series(
    series_by_era: dict[str, float],
    *,
    horizon: Horizon,
    seed: int,
    n_boot: int,
    alpha: float,
) -> MetricCell:
    keys = _sorted_numeric_keys(series_by_era)
    values = [series_by_era[k] for k in keys]
    stats = era_series_stats(values)
    block_len = resolve_block_len(stats.n, horizon)
    ci = block_bootstrap_ci(
        values,
        lambda a: float(era_series_stats(a).mean),
        block_len=block_len,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )
    return MetricCell(value=stats.mean, ci_low=ci.lo, ci_high=ci.hi, n_eras=stats.n)


def _cell_from_sharpe_series(
    series_by_era: dict[str, float],
    *,
    horizon: Horizon,
    seed: int,
    n_boot: int,
    alpha: float,
) -> MetricCell:
    keys = _sorted_numeric_keys(series_by_era)
    values = [series_by_era[k] for k in keys]
    n = len(values)
    value = ac_adjusted_sharpe(values, horizon=horizon)
    block_len = resolve_block_len(n, horizon)
    ci = block_bootstrap_ci(
        values,
        lambda a: float(ac_adjusted_sharpe(a, horizon=horizon)),
        block_len=block_len,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )
    return MetricCell(value=value, ci_low=ci.lo, ci_high=ci.hi, n_eras=n)


def _metric_cell_from_ci(ci: BootstrapCI, *, n_eras: int) -> MetricCell:
    return MetricCell(value=ci.point, ci_low=ci.lo, ci_high=ci.hi, n_eras=n_eras)


def _infer_horizon_target_name(
    benchmark_col: str | None,
    available_columns: list[str],
) -> str | None:
    if benchmark_col is None:
        return None
    match = re.search(r"_([a-zA-Z0-9]+)(?:20|60)$", benchmark_col)
    if match is None:
        return None
    candidate = match.group(1)
    if (
        f"target_{candidate}_20" in available_columns
        and f"target_{candidate}_60" in available_columns
    ):
        return candidate
    return None


def _build_joined_base(
    predictions: pl.DataFrame,
    *,
    meta_model: pl.DataFrame,
    features: pl.DataFrame,
    targets: pl.DataFrame,
    era_col: str,
    id_col: str,
    pred_col: str,
    meta_col: str,
    main_target: str,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Validate the four input frames and inner-join them into the evaluation
    base — the shared construction path for ``evaluate_model`` and
    ``evaluate_cross_check`` (single implementation, no duplicated math).

    Returns ``(base, join_keys, feature_cols)``. The optional benchmark join
    stays in ``evaluate_model`` — only it accepts benchmarks.
    """
    for name, frame in {
        "predictions": predictions,
        "meta_model": meta_model,
        "features": features,
        "targets": targets,
    }.items():
        if not isinstance(frame, pl.DataFrame):
            raise ValueError(f"{name} must be a polars DataFrame")

    if pred_col not in predictions.columns:
        raise ValueError(f"Missing required columns: ['{pred_col}']")
    if meta_col not in meta_model.columns:
        raise ValueError(f"Missing required columns: ['{meta_col}']")
    if main_target not in targets.columns:
        raise ValueError(f"Missing required columns: ['{main_target}']")

    join_keys = [era_col]
    for frame in (predictions, meta_model, features, targets):
        if era_col not in frame.columns:
            raise ValueError(f"Missing required columns: ['{era_col}']")
    if all(
        id_col in frame.columns
        for frame in (predictions, meta_model, features, targets)
    ):
        join_keys = [era_col, id_col]

    base = (
        predictions.select([*join_keys, pred_col])
        .join(meta_model.select([*join_keys, meta_col]), on=join_keys, how="inner")
        .join(targets, on=join_keys, how="inner")
        .join(features, on=join_keys, how="inner")
    )
    if base.is_empty():
        raise ValueError(
            "No overlap rows after joining predictions, meta_model, targets, and features"
        )

    feature_cols = [c for c in features.columns if c not in set(join_keys)]
    if not feature_cols:
        raise ValueError("features must contain at least one feature column")
    return base, join_keys, feature_cols


def _compute_era_series(
    evaluator: EvaluationEngine,
    base: pl.DataFrame,
    *,
    pred_col: str,
    meta_col: str,
    target_col: str,
    era_col: str,
    feature_cols: Sequence[str],
    mark: Callable[[str, float], None] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Per-era CORR/MMC/FNC series — the single computation path shared by
    ``evaluate_model`` and ``evaluate_cross_check`` (no duplicated math).

    ``mark`` is the caller's per-metric timing callback (``evaluate_model``'s
    ``_mark`` instrumentation); the cross-check passes ``None``. Returns the
    three series keyed by era label, in the engine's chronological order.
    """

    def _timed(step: str, func: Callable[[], dict[str, float]]) -> dict[str, float]:
        t0 = time.perf_counter()
        result = func()
        if mark is not None:
            mark(step, t0)
        return result

    corr_by_era = _timed(
        "corr_by_era",
        lambda: evaluator.per_era_corr(
            base,
            pred_col=pred_col,
            target_col=target_col,
            era_col=era_col,
        ),
    )
    mmc_by_era = _timed(
        "mmc_by_era",
        lambda: evaluator.per_era_mmc(
            base,
            pred_col=pred_col,
            meta_col=meta_col,
            target_col=target_col,
            era_col=era_col,
        ),
    )
    fnc_by_era = _timed(
        "fnc_by_era",
        lambda: evaluator.per_era_fnc(
            base,
            pred_col=pred_col,
            feature_cols=feature_cols,
            target_col=target_col,
            era_col=era_col,
        ),
    )
    return corr_by_era, mmc_by_era, fnc_by_era


def _plain_sharpe(values: Sequence[float]) -> float:
    """Plain per-era Sharpe: mean / std (ddof=0) of the era values, NOT
    AC-adjusted. Zero for a constant series (``era_series_stats`` semantics)."""
    return era_series_stats(values).sharpe


def evaluate_model(
    predictions: pl.DataFrame,
    *,
    meta_model: pl.DataFrame,
    benchmarks: pl.DataFrame | None,
    features: pl.DataFrame,
    targets: pl.DataFrame,
    n_trials: int,
    seed: int,
    horizon: Horizon = "20D",
    main_target: str = "target",
    benchmark_col: str | None = None,
    backend: str = "custom",
    regime_labels: pl.DataFrame | None = None,
    perturbation: PerturbationResult | None = None,
    pf: float = 1.0,
    clip: float = 0.05,
    n_boot: int = 1000,
    alpha: float = 0.05,
    min_overlap_eras: int = MIN_OVERLAP_ERAS,
    model_id: str = "model",
    era_col: str = "era",
    id_col: str = "id",
    pred_col: str = "prediction",
    meta_col: str = "numerai_meta_model",
    trials_sr_var: float | None = None,
    sr0_benchmark: float = 0.0,
) -> MetricScorecard:
    eval_start = time.perf_counter()
    metric_timing_seconds: dict[str, float] = {}

    def _mark(name: str, started: float) -> None:
        metric_timing_seconds[name] = round(time.perf_counter() - started, 6)

    t0 = time.perf_counter()
    base, join_keys, feature_cols = _build_joined_base(
        predictions,
        meta_model=meta_model,
        features=features,
        targets=targets,
        era_col=era_col,
        id_col=id_col,
        pred_col=pred_col,
        meta_col=meta_col,
        main_target=main_target,
    )
    _mark("join_base", t0)

    bench_col = benchmark_col
    if benchmarks is not None:
        if not isinstance(benchmarks, pl.DataFrame):
            raise ValueError("benchmarks must be a polars DataFrame when provided")
        if all(k in benchmarks.columns for k in join_keys):
            if bench_col is None:
                candidate_cols = [
                    c for c in benchmarks.columns if c not in set(join_keys)
                ]
                if candidate_cols:
                    bench_col = candidate_cols[0]
            if bench_col is not None and bench_col in benchmarks.columns:
                t0 = time.perf_counter()
                base = base.join(
                    benchmarks.select([*join_keys, bench_col]),
                    on=join_keys,
                    how="left",
                )
                _mark("join_benchmark", t0)

    evaluator = EvaluationEngine(backend)
    corr_by_era, mmc_by_era, fnc_by_era = _compute_era_series(
        evaluator,
        base,
        pred_col=pred_col,
        meta_col=meta_col,
        target_col=main_target,
        era_col=era_col,
        feature_cols=feature_cols,
        mark=_mark,
    )

    t0 = time.perf_counter()
    meta_corr_by_era = evaluator.per_era_corr(
        base,
        pred_col=meta_col,
        target_col=main_target,
        era_col=era_col,
    )
    _mark("meta_corr_by_era", t0)

    t0 = time.perf_counter()
    downside_eras = downside_era_indices(meta_corr_by_era)
    mmc_down_n = len(downside_eras)
    if mmc_down_n >= _MMC_DOWN_MIN_ERAS:
        mmc_down_value = float(np.mean([mmc_by_era[e] for e in downside_eras]))
        mmc_down_reason = None
    else:
        mmc_down_value = None
        mmc_down_reason = "insufficient_downside_eras"
    _mark("mmc_down", t0)

    t0 = time.perf_counter()
    turnover_mean: float | None = None
    turnover_std: float | None = None
    if id_col in base.columns:
        turnover_by_era = per_era_turnover(
            base,
            pred_col=pred_col,
            era_col=era_col,
            id_col=id_col,
        )
        turnover_values = [
            turnover_by_era[e]
            for e in sorted_era_labels(list(turnover_by_era.keys()))
        ]
        if len(turnover_values) >= 2:
            turnover_mean = float(np.mean(turnover_values))
            turnover_std = float(np.std(turnover_values, ddof=0))
            turnover_reason = None
        else:
            turnover_reason = "insufficient_transitions"
    else:
        turnover_reason = "id column unavailable"
    _mark("turnover", t0)

    t0 = time.perf_counter()
    payout = payout_report(
        corr_by_era,
        mmc_by_era,
        horizon=horizon,
        n_trials=n_trials,
        seed=seed,
        pf=pf,
        clip=clip,
        n_boot=n_boot,
        alpha=alpha,
        trials_sr_var=trials_sr_var,
        sr0_benchmark=sr0_benchmark,
    )
    _mark("payout_report", t0)

    t0 = time.perf_counter()
    corr_cell = _cell_from_series(
        corr_by_era,
        horizon=horizon,
        seed=seed,
        n_boot=n_boot,
        alpha=alpha,
    )
    _mark("corr_cell", t0)

    t0 = time.perf_counter()
    mmc_cell = _cell_from_series(
        mmc_by_era,
        horizon=horizon,
        seed=seed,
        n_boot=n_boot,
        alpha=alpha,
    )
    _mark("mmc_cell", t0)

    t0 = time.perf_counter()
    corr_sharpe_cell = _cell_from_sharpe_series(
        corr_by_era,
        horizon=horizon,
        seed=seed,
        n_boot=n_boot,
        alpha=alpha,
    )
    _mark("corr_sharpe_cell", t0)

    t0 = time.perf_counter()
    fnc_value = era_series_stats(
        [fnc_by_era[k] for k in _sorted_numeric_keys(fnc_by_era)]
    ).mean
    _mark("fnc_summary", t0)

    t0 = time.perf_counter()
    std_corr = era_series_stats(
        [corr_by_era[k] for k in _sorted_numeric_keys(corr_by_era)]
    ).std
    _mark("std_corr_summary", t0)

    t0 = time.perf_counter()
    exposure_df = feature_exposure_report(
        base.select([era_col, pred_col, *feature_cols]),
        feature_cols=feature_cols,
        era_col=era_col,
        pred_col=pred_col,
    )
    _mark("feature_exposure", t0)
    max_feature_exposure = (
        float(exposure_df.get_column("max_abs_exposure").to_list()[0])
        if exposure_df.height > 0
        else 0.0
    )

    horizon_result: HorizonStabilityResult | None = None
    horizon_reason: str | None = None
    if bench_col is not None and bench_col in base.columns:
        target_name = _infer_horizon_target_name(bench_col, base.columns)
        if target_name is not None:
            t0 = time.perf_counter()
            try:
                horizon_result = time_horizon_stability(
                    base,
                    pred_col=pred_col,
                    benchmark_col=bench_col,
                    target_name=target_name,
                    era_col=era_col,
                    min_overlap_eras=min_overlap_eras,
                )
            except NonVacuityError as exc:
                horizon_result = None
                horizon_reason = str(exc)
            _mark("horizon_stability", t0)
        else:
            horizon_reason = "horizon target columns unavailable"
            logger.warning(
                "[scorecard] benchmark column %r present but horizon diagnostics "
                "disabled: no matching target_<name>_20/60 pair found in the "
                "targets frame — load all target_* columns when building "
                "validation payloads",
                bench_col,
            )
    else:
        horizon_reason = "benchmark unavailable"

    regime_result: dict[str, RegimeCorr] | None = None
    regime_reason: str | None = None
    if regime_labels is not None:
        t0 = time.perf_counter()
        try:
            regime_result = regime_conditioned_corr(
                base.select([era_col, pred_col, main_target]),
                regime_labels,
                pred_col=pred_col,
                target_col=main_target,
                regime_col="regime",
                era_col=era_col,
                horizon=horizon,
                seed=seed,
                n_boot=n_boot,
                alpha=alpha,
                min_eras_per_regime=min_overlap_eras,
            )
        except NonVacuityError as exc:
            regime_result = None
            regime_reason = str(exc)
        _mark("regime_conditioned_corr", t0)
    else:
        regime_reason = "regime labels unavailable"

    bmc_cell: MetricCell | None = None
    bmc_reason: str | None = None
    if bench_col is not None and bench_col in base.columns:
        t0 = time.perf_counter()
        try:
            bmc_by_era = evaluator.per_era_bmc(
                base,
                pred_col=pred_col,
                benchmark_col=bench_col,
                target_col=main_target,
                era_col=era_col,
                min_overlap_eras=min_overlap_eras,
            )
            bmc_cell = _cell_from_series(
                bmc_by_era,
                horizon=horizon,
                seed=seed,
                n_boot=n_boot,
                alpha=alpha,
            )
        except NonVacuityError as exc:
            bmc_cell = None
            bmc_reason = str(exc)
        _mark("bmc", t0)
    else:
        bmc_reason = "benchmark unavailable"

    cwmm_cell: MetricCell | None = None
    cwmm_reason: str | None = None
    t0 = time.perf_counter()
    try:
        cwmm_by_era = evaluator.per_era_cwmm(
            base,
            pred_col=pred_col,
            meta_col=meta_col,
            era_col=era_col,
            min_overlap_eras=min_overlap_eras,
        )
        cwmm_cell = _cell_from_series(
            cwmm_by_era,
            horizon=horizon,
            seed=seed,
            n_boot=n_boot,
            alpha=alpha,
        )
    except NonVacuityError as exc:
        cwmm_cell = None
        cwmm_reason = str(exc)
    _mark("cwmm", t0)

    eval_total_seconds = round(time.perf_counter() - eval_start, 6)

    t0 = time.perf_counter()
    # A4 (audit SEV-3): surface the eras that short-circuited (degenerate —
    # 0.0-normalized at the engine boundary) so they are distinguishable from
    # genuine zero-IC eras. Values are unchanged; only visibility is added.
    degenerate_eras = tuple(
        evaluator.degenerate_eras(
            base,
            era_col=era_col,
            pred_col=pred_col,
            target_col=main_target,
            meta_col=meta_col,
            feature_cols=feature_cols,
        )
    )
    _mark("degenerate_eras", t0)

    return MetricScorecard(
        model_id=model_id,
        n_eras=payout.n_eras,
        rank_scalar=payout.mean_payout,
        deflated_sharpe=payout.deflated_sharpe,
        mean_payout=_metric_cell_from_ci(payout.payout_ci, n_eras=payout.n_eras),
        corr=corr_cell,
        mmc=mmc_cell,
        fnc=fnc_value,
        corr_sharpe_ac=corr_sharpe_cell,
        cvar5=payout.cvar5,
        max_drawdown=payout.max_drawdown,
        burn_rate=payout.burn_rate,
        mmc_sharpe_ac=payout.mmc_sharpe,
        sortino=payout.sortino,
        calmar=payout.calmar,
        std_corr=std_corr,
        max_burn_streak=payout.max_burn_streak,
        time_to_recovery=payout.time_to_recovery,
        horizon_stability=horizon_result,
        horizon_reason=horizon_reason,
        regime_corr=regime_result,
        regime_reason=regime_reason,
        perturbation=perturbation,
        max_feature_exposure=max_feature_exposure,
        bmc=bmc_cell,
        bmc_reason=bmc_reason,
        cwmm=cwmm_cell,
        cwmm_reason=cwmm_reason,
        book_correlation=None,
        cagr_1y=payout.cagr_1y,
        gain_to_pain_ratio=payout.gain_to_pain_ratio,
        kelly_fraction=payout.kelly_fraction,
        mmc_down=mmc_down_value,
        mmc_down_n_eras=mmc_down_n,
        mmc_down_reason=mmc_down_reason,
        turnover_mean=turnover_mean,
        turnover_std=turnover_std,
        turnover_reason=turnover_reason,
        sim_portfolio_cagr=payout.overlapping_sim.portfolio_cagr,
        sim_portfolio_mdd=payout.overlapping_sim.portfolio_max_drawdown,
        sim_capital_utilization=payout.overlapping_sim.avg_capital_utilization,
        metric_timing_seconds=metric_timing_seconds,
        eval_total_seconds=eval_total_seconds,
        degenerate_eras=degenerate_eras,
    )


def evaluate_cross_check(
    predictions: pl.DataFrame,
    *,
    meta_model: pl.DataFrame,
    features: pl.DataFrame,
    targets: pl.DataFrame,
    horizon: Horizon = "20D",
    main_target: str = "target",
    seed: int = 42,
) -> CrossCheckResult:
    """Era-grouped official-backend cross-check of a partial export.

    Reuses the exact per-era computation path of ``evaluate_model`` (shared
    internal helpers — no duplicated math) but returns the per-era series and
    raw Sharpe the scorecard itself does not retain. Replay parameters are the
    fixed ``CROSSCHECK_N_TRIALS`` constant plus ``evaluate_model`` defaults
    (official backend forced; column names are ``evaluate_model``'s defaults).
    """
    base, _join_keys, feature_cols = _build_joined_base(
        predictions,
        meta_model=meta_model,
        features=features,
        targets=targets,
        era_col="era",
        id_col="id",
        pred_col="prediction",
        meta_col="numerai_meta_model",
        main_target=main_target,
    )
    evaluator = EvaluationEngine("official")
    corr_by_era, mmc_by_era, fnc_by_era = _compute_era_series(
        evaluator,
        base,
        pred_col="prediction",
        meta_col="numerai_meta_model",
        target_col=main_target,
        era_col="era",
        feature_cols=feature_cols,
    )
    scorecard = evaluate_model(
        predictions,
        meta_model=meta_model,
        benchmarks=None,
        features=features,
        targets=targets,
        n_trials=CROSSCHECK_N_TRIALS,
        seed=seed,
        horizon=horizon,
        main_target=main_target,
        backend="official",
    )
    raw_sharpe = _plain_sharpe(list(corr_by_era.values()))
    per_era = {
        "corr": [
            {"era": str(era), "value": float(value)}
            for era, value in corr_by_era.items()
        ],
        "mmc": [
            {"era": str(era), "value": float(value)}
            for era, value in mmc_by_era.items()
        ],
        "fnc": [
            {"era": str(era), "value": float(value)}
            for era, value in fnc_by_era.items()
        ],
    }
    return CrossCheckResult(scorecard=scorecard, per_era=per_era, raw_sharpe=raw_sharpe)
