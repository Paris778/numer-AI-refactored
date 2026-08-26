"""Pure analytical engine for the executive performance dashboard.

Registry scans, benchmark reconciliation, gate projection, capital-cell
recompute, and payout timeseries extraction. Plotly/Streamlit-free; every
function here is covered by tests/test_dashboard.py.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nmr import lifecycle, paths
from nmr.config import REPO_ROOT
from nmr.ensemble import Ensembler
from nmr.evaluation import EvaluationEngine, downside_era_indices, sorted_era_labels
from nmr.payout import (
    annual_compounded_return,
    gain_to_pain_ratio,
    kelly_fraction,
    payout_series,
)
from nmr.scorecard import MMC_DOWN_MIN_ERAS

logger = logging.getLogger("nmr.dashboard")

__all__ = [
    "DASHBOARD_METRICS",
    "DEFAULT_RANK_METRIC",
    "DashboardMetricSpec",
    "EVALUABLE_ROWS",
    "UNIFIED_SCHEMA",
    "compute_ml_advantage",
    "dashboard_cohort",
    "build_tournament_payload",
    "evaluate_gate_status",
    "extract_multimetric_timeseries",
    "extract_pairwise_similarity_matrix",
    "load_benchmark_frame",
    "load_unified_leaderboard",
    "load_model_detail",
    "read_champion_pointer",
    "reconcile_capital_metrics",
    "rank_leaderboard",
    "rank_map_by_metric",
    "resolve_benchmark_path",
]

REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"
DEFAULT_REGISTRY_DIR = REPO_ROOT / "artifacts" / "registry"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "v5.3"
DEFAULT_GATE_PATH = REPO_ROOT / "configs" / "benchmarks" / "tier4_gate.yaml"

# Superset of dashboard_app._LEADERBOARD_SCHEMA plus the capital-readiness,
# gate, and tier columns consumed by the executive report.
UNIFIED_SCHEMA = pl.Schema(
    {
        "model_id": pl.String,
        "source": pl.String,
        "run_name": pl.String,
        "family": pl.String,
        "display_name": pl.String,
        "lifecycle_stage": pl.String,
        "current_full_status": pl.String,
        "stale": pl.Boolean,
        "training_scope": pl.String,
        "has_full_version": pl.Boolean,
        "backend": pl.String,
        "preset": pl.String,
        "feature_set": pl.String,
        "feature_subset": pl.String,
        "n_targets": pl.Int64,
        "targets": pl.String,
        "neutralization_proportion": pl.Float64,
        "oof_device": pl.String,
        "corr": pl.Float64,
        "corr_ci_low": pl.Float64,
        "corr_ci_high": pl.Float64,
        "corr_n_eras": pl.Int64,
        "corr_sharpe_ac": pl.Float64,
        "corr_sharpe_ac_ci_low": pl.Float64,
        "corr_sharpe_ac_ci_high": pl.Float64,
        "corr_sharpe_ac_n_eras": pl.Int64,
        "std_corr": pl.Float64,
        "max_drawdown": pl.Float64,
        "deflated_sharpe": pl.Float64,
        "fnc": pl.Float64,
        "mmc": pl.Float64,
        "mmc_sharpe_ac": pl.Float64,
        "bmc": pl.Float64,
        "cwmm": pl.Float64,
        "mean_payout": pl.Float64,
        "cagr_1y": pl.Float64,
        "gain_to_pain_ratio": pl.Float64,
        "kelly_fraction": pl.Float64,
        "mmc_down": pl.Float64,
        "mmc_down_reason": pl.String,
        "turnover_mean": pl.Float64,
        "n_eras": pl.Int64,
        "rank_scalar": pl.Float64,
        "cvar5": pl.Float64,
        "burn_rate": pl.Float64,
        "max_feature_exposure": pl.Float64,
        "max_feature_exposure_reason": pl.String,
        "has_bmc": pl.Boolean,
        "has_horizon": pl.Boolean,
        "has_perturb": pl.Boolean,
        "has_regime": pl.Boolean,
        "tier": pl.Int64,
        "run_dir": pl.String,
    }
)

# Single predicate for every chart / candidate-selection path: rows that carry
# validation metrics. Trained and benchmark rows are evaluable (reference
# curves chart alongside candidates, spec §8); only diagnostic rows are
# excluded — full (in-sample metrics) and partial (train-only cross-check)
# are never ranked (Task 10).
EVALUABLE_ROWS: pl.Expr = ~pl.col("source").is_in(["full", "partial"])


@dataclass(frozen=True)
class DashboardMetricSpec:
    """Display and ranking contract for one unified leaderboard metric."""

    name: str
    label: str
    higher_is_better: bool
    window: str = "standardized meta-overlap"


DASHBOARD_METRICS: tuple[DashboardMetricSpec, ...] = (
    DashboardMetricSpec("cagr_1y", "Profitability (CAGR 1Y)", True),
    DashboardMetricSpec("mmc", "MMC", True),
    DashboardMetricSpec("corr", "CORR", True),
    DashboardMetricSpec("corr_sharpe_ac", "CORR Sharpe (AC)", True),
    DashboardMetricSpec("fnc", "FNC", True),
    DashboardMetricSpec("bmc", "BMC", True),
    DashboardMetricSpec("deflated_sharpe", "Deflated Sharpe", True),
    DashboardMetricSpec("mmc_down", "Downside MMC", True),
    DashboardMetricSpec("cwmm", "CWMM", True),
    DashboardMetricSpec("gain_to_pain_ratio", "Gain-to-Pain Ratio", True),
    DashboardMetricSpec("mean_payout", "Mean Payout", True),
    DashboardMetricSpec("std_corr", "CORR Volatility", False),
    DashboardMetricSpec("max_drawdown", "Max Drawdown", False),
    DashboardMetricSpec("turnover_mean", "Mean Turnover", False),
)
DEFAULT_RANK_METRIC = "mmc"
_DASHBOARD_METRIC_BY_NAME = {spec.name: spec for spec in DASHBOARD_METRICS}


def _dashboard_metric_spec(metric: str) -> DashboardMetricSpec:
    try:
        return _DASHBOARD_METRIC_BY_NAME[metric]
    except KeyError as exc:
        raise ValueError(
            f"metric={metric!r} not in {sorted(_DASHBOARD_METRIC_BY_NAME)}"
        ) from exc


def dashboard_cohort(row: Mapping[str, Any]) -> str:
    """Derive a display cohort from the unified source and benchmark tier."""
    source = row.get("source")
    if source in ("trained", "trained_legacy"):
        return "trained"
    if source == "full":
        return "full"
    if source == "partial":
        return "partial"
    if source == "benchmark":
        tier = row.get("tier")
        try:
            if tier is not None and int(tier) <= 2:
                return "heuristic"
        except (TypeError, ValueError):
            pass
        return "benchmark"
    return "unknown"


def _finite_metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _dashboard_sort_key(
    row: Mapping[str, Any], metric: str, higher_is_better: bool
) -> tuple[bool, float, str]:
    value = _finite_metric_value(row, metric)
    if value is None:
        return True, 0.0, str(row.get("model_id") or "")
    return (
        False,
        (-value if higher_is_better else value),
        str(row.get("model_id") or ""),
    )


def rank_leaderboard(
    leaderboard: pl.DataFrame, metric: str = DEFAULT_RANK_METRIC
) -> pl.DataFrame:
    """Decorate and deterministically rank all evaluable leaderboard rows.

    Diagnostic scopes (full: in-sample metrics; partial: train-only
    cross-check) are never ranked — they render at the bottom with ``rank``
    None and their own cohort stamp.
    """
    spec = _dashboard_metric_spec(metric)
    rows = leaderboard.to_dicts()
    evaluable = [
        row for row in rows if dashboard_cohort(row) not in ("full", "partial")
    ]
    ranked_rows = [
        row for row in evaluable if _finite_metric_value(row, metric) is not None
    ]
    unranked_rows = [
        row for row in evaluable if _finite_metric_value(row, metric) is None
    ]
    ranked_rows.sort(
        key=lambda row: _dashboard_sort_key(row, metric, spec.higher_is_better)
    )
    unranked_rows.sort(key=lambda row: str(row.get("model_id") or ""))
    ranks = {id(row): index for index, row in enumerate(ranked_rows, start=1)}
    decorated: list[dict[str, Any]] = []
    for row in [*ranked_rows, *unranked_rows]:
        item = dict(row)
        item["cohort"] = dashboard_cohort(row)
        item["rank"] = ranks.get(id(row))
        decorated.append(item)
    for row in rows:
        if dashboard_cohort(row) in ("full", "partial"):
            item = dict(row)
            item["cohort"] = dashboard_cohort(row)
            item["rank"] = None
            decorated.append(item)
    if not decorated:
        return leaderboard.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("rank"),
            pl.lit(None, dtype=pl.String).alias("cohort"),
        )
    return pl.DataFrame(decorated, strict=False)


def rank_map_by_metric(leaderboard: pl.DataFrame) -> dict[str, dict[str, int]]:
    """Return deterministic ranks for every available metric and model."""
    result: dict[str, dict[str, int]] = {}
    for spec in DASHBOARD_METRICS:
        ranked = rank_leaderboard(leaderboard, metric=spec.name)
        metric_ranks: dict[str, int] = {}
        for row in ranked.to_dicts():
            model_id = row.get("model_id")
            rank = row.get("rank")
            if model_id is not None and rank is not None:
                metric_ranks[str(model_id)] = int(rank)
        for model_id in sorted(metric_ranks):
            result.setdefault(model_id, {})[spec.name] = metric_ranks[model_id]
    return result


def _strict_beats_by_metric(leaderboard: pl.DataFrame) -> dict[str, list[str]]:
    """Return trained IDs whose full-precision metric strictly beats its benchmark."""
    rows = leaderboard.to_dicts()
    result: dict[str, list[str]] = {}
    for spec in DASHBOARD_METRICS:
        benchmarks = [
            row
            for row in rows
            if dashboard_cohort(row) == "benchmark"
            and _finite_metric_value(row, spec.name) is not None
        ]
        if not benchmarks:
            result[spec.name] = []
            continue
        benchmark = min(
            benchmarks,
            key=lambda row: _dashboard_sort_key(row, spec.name, spec.higher_is_better),
        )
        benchmark_value = _finite_metric_value(benchmark, spec.name)
        assert benchmark_value is not None
        winners = []
        for row in rows:
            value = _finite_metric_value(row, spec.name)
            if dashboard_cohort(row) != "trained" or value is None:
                continue
            beats = (
                value > benchmark_value
                if spec.higher_is_better
                else value < benchmark_value
            )
            if beats:
                winners.append(str(row.get("model_id") or ""))
        result[spec.name] = sorted(winners)
    return result


def _best_cohort_row(
    rows: Sequence[Mapping[str, Any]], cohort: str, metric: str, higher_is_better: bool
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("cohort") == cohort and _finite_metric_value(row, metric) is not None
    ]
    if not candidates:
        return None
    return dict(
        sorted(
            candidates,
            key=lambda row: _dashboard_sort_key(row, metric, higher_is_better),
        )[0]
    )


def _advantage_record(
    trained: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
    metric: str,
    higher_is_better: bool,
    baseline_name: str,
) -> dict[str, Any]:
    if trained is None:
        return {
            "absolute_edge": None,
            "relative_percent": None,
            "reason": "no_trained_model",
        }
    if baseline is None:
        return {
            "absolute_edge": None,
            "relative_percent": None,
            "reason": f"no_{baseline_name}_baseline",
        }
    trained_value = _finite_metric_value(trained, metric)
    baseline_value = _finite_metric_value(baseline, metric)
    if trained_value is None or baseline_value is None:
        return {
            "absolute_edge": None,
            "relative_percent": None,
            "reason": "metric_unavailable",
        }
    edge = (
        trained_value - baseline_value
        if higher_is_better
        else baseline_value - trained_value
    )
    relative = None if baseline_value == 0.0 else edge / abs(baseline_value) * 100.0
    return {
        "absolute_edge": float(edge),
        "relative_percent": None if relative is None else float(relative),
        "reason": None if relative is not None else "baseline_zero",
    }


def compute_ml_advantage(
    leaderboard: pl.DataFrame, metric: str = DEFAULT_RANK_METRIC
) -> dict[str, Any]:
    """Compare the strongest trained row with the strongest baseline cohorts."""
    spec = _dashboard_metric_spec(metric)
    ranked = rank_leaderboard(leaderboard, metric=metric)
    rows = ranked.to_dicts()
    best = {
        cohort: _best_cohort_row(rows, cohort, metric, spec.higher_is_better)
        for cohort in ("trained", "heuristic", "benchmark")
    }

    def summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "model_id": row.get("model_id"),
            "value": _finite_metric_value(row, metric),
            "cohort": row.get("cohort"),
        }

    return {
        "metric": metric,
        "trained": summary(best["trained"]),
        "heuristic": summary(best["heuristic"]),
        "benchmark": summary(best["benchmark"]),
        "vs_heuristic": _advantage_record(
            best["trained"],
            best["heuristic"],
            metric,
            spec.higher_is_better,
            "heuristic",
        ),
        "vs_benchmark": _advantage_record(
            best["trained"],
            best["benchmark"],
            metric,
            spec.higher_is_better,
            "benchmark",
        ),
    }


_CORE_DISPLAY_FIELDS = (
    "corr",
    "corr_ci_low",
    "corr_ci_high",
    "corr_n_eras",
    "mmc",
    "mmc_ci_low",
    "mmc_ci_high",
    "mmc_n_eras",
    "corr_sharpe_ac",
    "corr_sharpe_ac_ci_low",
    "corr_sharpe_ac_ci_high",
    "corr_sharpe_ac_n_eras",
    "cagr_1y",
    "gain_to_pain_ratio",
    "max_drawdown",
    "n_eras",
)
_ALL_SCORECARD_FIELDS = (
    "mean_payout",
    "mean_payout_ci_low",
    "mean_payout_ci_high",
    "mean_payout_n_eras",
    "corr",
    "corr_ci_low",
    "corr_ci_high",
    "corr_n_eras",
    "mmc",
    "mmc_ci_low",
    "mmc_ci_high",
    "mmc_n_eras",
    "corr_sharpe_ac",
    "corr_sharpe_ac_ci_low",
    "corr_sharpe_ac_ci_high",
    "corr_sharpe_ac_n_eras",
    "fnc",
    "fnc_ci_low",
    "fnc_ci_high",
    "fnc_n_eras",
    "bmc",
    "bmc_ci_low",
    "bmc_ci_high",
    "bmc_n_eras",
    "cwmm",
    "cwmm_ci_low",
    "cwmm_ci_high",
    "cwmm_n_eras",
    "rank_scalar",
    "deflated_sharpe",
    "cvar5",
    "burn_rate",
    "max_drawdown",
    "std_corr",
    "max_feature_exposure",
    "cagr_1y",
    "gain_to_pain_ratio",
    "kelly_fraction",
    "mmc_down",
    "mmc_down_n_eras",
    "mmc_down_reason",
    "turnover_mean",
    "turnover_std",
    "turnover_reason",
    "sim_portfolio_cagr",
    "sim_portfolio_mdd",
    "sim_capital_utilization",
    "horizon_target_name",
    "horizon_n_eras",
    "horizon_model_sharpe_20",
    "horizon_model_sharpe_60",
    "horizon_model_decay",
    "horizon_benchmark_decay",
    "horizon_relative_divergence",
    "horizon_reason",
    "perturb_alpha",
    "perturb_n_eras",
    "perturb_ceiling_stability",
    "perturb_manifold_stability",
    "perturb_gap",
    "perturb_effective_perturb_frac",
    "regime_count",
    "regime_min_n_eras",
    "regime_max_n_eras",
    "regime_corr_json",
    "regime_reason",
    "bmc_reason",
    "cwmm_reason",
)
_DETAIL_SCORECARD_FIELDS = tuple(
    field
    for field in _ALL_SCORECARD_FIELDS
    if field not in {spec.name for spec in DASHBOARD_METRICS} and field != "n_eras"
)
_DETAIL_PROVENANCE_FIELDS = (
    "seed",
    "pipeline_device",
    "oof_device",
    "timestamp",
)
_ROW_FIELDS = (
    "model_id",
    "source",
    "run_name",
    "display_name",
    "lifecycle_stage",
    "current_full_status",
    "stale",
    "cohort",
    "family",
    "backend",
    "preset",
    "feature_set",
    "feature_subset",
    "targets",
    "neutralization_proportion",
    "oof_device",
    "status",
    "champion",
    "values",
    "ci",
    "robustness",
)
_ROW_VALUE_SCALE = 1_000_000


def _scaled_row_value(value: Any) -> int | None:
    numeric = _finite_metric_value({"value": value}, "value")
    return None if numeric is None else int(round(numeric * _ROW_VALUE_SCALE))


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _compact_json_value(value: Any, *, digits: int = 6) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _compact_json_value(value[key], digits=digits)
            for key in sorted(value)
        }
    if isinstance(value, list):
        return [_compact_json_value(item, digits=digits) for item in value]
    if isinstance(value, float):
        return round(value, digits) if np.isfinite(value) else None
    return value


def _sparse_payload_values(values: Sequence[Any]) -> list[list[Any]]:
    return [[index, value] for index, value in enumerate(values) if value is not None]


def _detail_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    config = manifest.get("config") or {}
    data = config.get("data") or {}
    model = config.get("model") or {}
    risk = config.get("risk") or {}
    run = config.get("run") or {}
    return {
        "backend": model.get("backend"),
        "preset": model.get("preset"),
        "feature_set": data.get("feature_set"),
        "feature_subset": data.get("feature_subset"),
        "targets": data.get("targets"),
        "neutralization_proportion": risk.get("neutralization_proportion"),
        "seed": run.get("seed"),
        "pipeline_device": manifest.get("pipeline_device"),
        "oof_device": manifest.get("oof_device"),
        "timestamp": next(
            (
                manifest.get(key)
                for key in ("timestamp", "created_at", "completed_at", "promoted_at")
                if manifest.get(key) is not None
            ),
            None,
        ),
    }


def load_model_detail(row: Mapping[str, Any]) -> dict[str, Any]:
    """Load compact immutable evidence for one leaderboard row."""
    model_id = str(row.get("model_id") or "?")
    source = str(row.get("source") or "unknown")
    cohort = dashboard_cohort(row)
    detail: dict[str, Any] = {
        "model_id": model_id,
        "source": source,
        "cohort": cohort,
        "scorecard": {},
        "provenance": {},
        "evidence_ref": None,
        "reason": None,
    }
    run_dir = row.get("run_dir")
    payload: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    if source in ("trained", "trained_legacy") and run_dir:
        run_path = Path(str(run_dir)) / "run.json"
        payload = _read_json_mapping(run_path)
        detail["evidence_ref"] = f"registry/{model_id}/run.json"
        if payload is None:
            detail["reason"] = "run_metadata_unavailable"
        else:
            manifest = (
                payload.get("manifest")
                if isinstance(payload.get("manifest"), dict)
                else {}
            )
            scorecard = payload.get("scorecard")
            if isinstance(scorecard, dict):
                detail["scorecard"] = _compact_json_value(
                    {
                        key: scorecard.get(key)
                        for key in _ALL_SCORECARD_FIELDS
                        if key in scorecard
                    }
                )
            else:
                detail["reason"] = "scorecard_unavailable"
    elif source == "full" and run_dir:
        manifest = _read_json_mapping(Path(str(run_dir)) / "export.json")
        family = str(row.get("family") or row.get("run_name") or "unknown")
        promoted_from = (manifest or {}).get("promoted_from_run_id")
        if promoted_from:
            detail["evidence_ref"] = (
                f"experiments/{family}/exports/full/{promoted_from}/export.json"
            )
        if manifest is None:
            detail["reason"] = "full_manifest_unavailable"
        else:
            detail["scorecard"] = {}
    elif source == "partial" and run_dir:
        # Cross-check scorecard (spec §8 + §12 #15): the slot's own
        # scorecard.json — flat-mapped onto the same detail key convention as
        # trained rows so the cross-check metrics render on the family detail.
        slot_dir = Path(str(run_dir))
        family = str(row.get("family") or row.get("run_name") or "unknown")
        detail["evidence_ref"] = (
            f"experiments/{family}/exports/partial/{slot_dir.name}/scorecard.json"
        )
        cells = _partial_cross_check_cells(slot_dir)
        if cells:
            detail["scorecard"] = _compact_json_value(cells)
        else:
            detail["reason"] = "scorecard_unavailable"
    else:
        detail["scorecard"] = _compact_json_value(
            {
                key: row.get(key)
                for key in _DETAIL_SCORECARD_FIELDS
                if row.get(key) is not None
            }
        )
        if source == "benchmark":
            detail["reason"] = "benchmark_csv_projection"
        elif source != "full":
            detail["reason"] = "source_metadata_unavailable"
    if manifest is not None:
        detail["provenance"] = _compact_json_value(_detail_provenance(manifest))
    return detail


def build_tournament_payload(
    leaderboard: pl.DataFrame,
    *,
    champion_id: str | None = None,
    evaluation_eras: Sequence[str] = (),
    suite_version: str = "v2",
    data_version: str = "v5.3",
) -> dict[str, Any]:
    """Build the deterministic, renderer-neutral Model Tournament payload."""
    ranked = rank_leaderboard(leaderboard, metric=DEFAULT_RANK_METRIC)
    rank_map = rank_map_by_metric(leaderboard)
    rows: list[dict[str, Any]] = []
    model_ids: list[str] = []
    details: list[Any] = []
    rank_values: list[list[int | None]] = []
    landscape: list[dict[str, Any]] = []
    for row in ranked.to_dicts():
        model_id = str(row.get("model_id") or "?")
        cohort = str(row.get("cohort") or dashboard_cohort(row))
        champion = model_id == champion_id and row.get("status") in (
            "CHAMPION",
            "CAPITAL READY",
        )
        item: dict[str, Any] = {
            "model_id": model_id,
            "source": row.get("source"),
            "run_name": row.get("run_name"),
            "display_name": row.get("display_name"),
            "lifecycle_stage": row.get("lifecycle_stage"),
            "current_full_status": row.get("current_full_status"),
            "stale": row.get("stale"),
            "cohort": cohort,
            "family": row.get("family"),
            "backend": row.get("backend"),
            "preset": row.get("preset"),
            "feature_set": row.get("feature_set"),
            "feature_subset": row.get("feature_subset"),
            "targets": row.get("targets"),
            "neutralization_proportion": row.get("neutralization_proportion"),
            "oof_device": row.get("oof_device"),
            "status": row.get("status"),
            "champion": champion,
            "values": [
                _scaled_row_value(row.get(spec.name)) for spec in DASHBOARD_METRICS
            ],
            "ci": [
                _scaled_row_value(row.get("corr_sharpe_ac_ci_low")),
                _scaled_row_value(row.get("corr_sharpe_ac_ci_high")),
                row.get("n_eras"),
            ],
            "robustness": {
                "has_bmc": row.get("has_bmc"),
                "has_horizon": row.get("has_horizon"),
                "has_perturb": row.get("has_perturb"),
                "has_regime": row.get("has_regime"),
            },
        }
        item = _compact_json_value(item)
        rows.append(item)
        model_ids.append(model_id)
        detail = load_model_detail(row)
        details.append(
            [
                _sparse_payload_values(
                    _compact_json_value(
                        [
                            detail["scorecard"].get(key)
                            for key in _DETAIL_SCORECARD_FIELDS
                        ],
                        digits=4,
                    )
                ),
                _sparse_payload_values(
                    _compact_json_value(
                        [
                            detail["provenance"].get(key)
                            for key in _DETAIL_PROVENANCE_FIELDS
                        ],
                        digits=4,
                    )
                ),
                detail["evidence_ref"] if row.get("source") == "full" else None,
                detail["reason"],
            ]
        )
        rank_values.append(
            [rank_map.get(model_id, {}).get(spec.name) for spec in DASHBOARD_METRICS]
        )
        corr = _finite_metric_value(row, "corr")
        mmc = _finite_metric_value(row, "mmc")
        if corr is not None and mmc is not None and cohort not in ("full", "partial"):
            landscape.append(
                {
                    "model_id": model_id,
                    "cohort": cohort,
                    "corr": corr,
                    "mmc": mmc,
                    "champion": champion,
                }
            )

    eras = sorted(
        {str(era) for era in evaluation_eras},
        key=lambda era: int(era) if era.isdigit() else era,
    )
    window = {
        "start": eras[0] if eras else None,
        "end": eras[-1] if eras else None,
        "n_eras": len(eras),
    }
    return {
        "meta": {
            "suite_version": suite_version,
            "data_version": data_version,
            "evaluation_window": window,
            "offline_only": True,
            "default_rank_metric": DEFAULT_RANK_METRIC,
            "metric_window": "standardized meta-overlap",
        },
        "metric_specs": [
            {
                "name": spec.name,
                "label": spec.label,
                "higher_is_better": spec.higher_is_better,
                "direction": "higher" if spec.higher_is_better else "lower",
            }
            for spec in DASHBOARD_METRICS
        ],
        "cohorts": ["all", "trained", "heuristic", "benchmark"],
        "model_ids": model_ids,
        "row_fields": list(_ROW_FIELDS),
        "rows": [[row.get(field) for field in _ROW_FIELDS] for row in rows],
        "details": details,
        "metric_fields": [spec.name for spec in DASHBOARD_METRICS],
        "row_value_scale": _ROW_VALUE_SCALE,
        "strict_beats": _strict_beats_by_metric(leaderboard),
        "ci_fields": ["corr_sharpe_ac_ci_low", "corr_sharpe_ac_ci_high", "n_eras"],
        "scorecard_fields": list(_DETAIL_SCORECARD_FIELDS),
        "provenance_fields": list(_DETAIL_PROVENANCE_FIELDS),
        "rank_values": rank_values,
        "advantage": compute_ml_advantage(leaderboard),
        "landscape": sorted(landscape, key=lambda point: point["model_id"]),
        "rank_movement": {"available": False, "reason": "no_comparable_prior_snapshot"},
    }


def resolve_benchmark_path(
    benchmark_path: Path | None | bool = None,
    reports_dir: Path | None = None,
) -> Path | None:
    """Resolve the benchmark scorecard CSV via the fallback chain.

    Chain: given path (if it exists) -> full hierarchy CSV -> smoke CSV ->
    None. ``benchmark_path=False`` is an explicit directive to disable
    benchmark loading entirely (test isolation).
    """
    if benchmark_path is False:
        return None
    if benchmark_path is not None:
        given = Path(benchmark_path)
        if given.exists():
            return given
    reports = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    for candidate in (
        reports / "benchmark_hierarchy_scorecard.csv",
        reports / "benchmark_hierarchy_scorecard_smoke.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def load_benchmark_frame(benchmark_path: Path) -> pl.DataFrame:
    """Normalize a benchmark scorecard CSV into unified-schema rows.

    Mirrors the legacy ``dashboard_app.load_benchmarks`` column semantics but
    carries the full scorecard mapping (fnc, deflated_sharpe, capital cells,
    CIs). A missing file, or an empty CSV, returns the empty schema frame.
    """
    path = Path(benchmark_path)
    if not path.exists():
        return pl.DataFrame(schema=UNIFIED_SCHEMA)
    df = pl.read_csv(path)
    if df.height == 0:
        return pl.DataFrame(schema=UNIFIED_SCHEMA)

    rows: list[dict] = []
    for row in df.to_dicts():
        tier_value = row.get("tier")
        rows.append(
            {
                "model_id": row.get("model_id"),
                "source": "benchmark",
                "family": None,
                "training_scope": None,
                "has_full_version": False,
                "run_name": row.get("strategy_group")
                or (
                    f"tier{int(tier_value)}" if tier_value is not None else "benchmark"
                ),
                "backend": "benchmark",
                "preset": "benchmark",
                "feature_set": "all",
                "feature_subset": None,
                "n_targets": 1,
                "targets": row.get("horizon_target_name") or "target",
                "neutralization_proportion": None,
                "oof_device": None,
                "corr": row.get("corr"),
                "corr_ci_low": row.get("corr_ci_low"),
                "corr_ci_high": row.get("corr_ci_high"),
                "corr_n_eras": row.get("corr_n_eras"),
                "corr_sharpe_ac": row.get("corr_sharpe_ac"),
                "corr_sharpe_ac_ci_low": row.get("corr_sharpe_ac_ci_low"),
                "corr_sharpe_ac_ci_high": row.get("corr_sharpe_ac_ci_high"),
                "corr_sharpe_ac_n_eras": row.get("corr_sharpe_ac_n_eras"),
                "std_corr": row.get("std_corr"),
                "max_drawdown": row.get("max_drawdown"),
                "deflated_sharpe": row.get("deflated_sharpe"),
                "fnc": row.get("fnc"),
                "mmc": row.get("mmc"),
                "mmc_sharpe_ac": row.get("mmc_sharpe_ac"),
                "bmc": row.get("bmc"),
                "cwmm": row.get("cwmm"),
                "mean_payout": row.get("mean_payout"),
                "cagr_1y": row.get("cagr_1y"),
                "gain_to_pain_ratio": row.get("gain_to_pain_ratio"),
                "kelly_fraction": row.get("kelly_fraction"),
                "mmc_down": row.get("mmc_down"),
                "mmc_down_reason": row.get("mmc_down_reason"),
                "turnover_mean": row.get("turnover_mean"),
                "n_eras": row.get("n_eras"),
                "rank_scalar": row.get("rank_scalar"),
                "cvar5": row.get("cvar5"),
                "burn_rate": row.get("burn_rate"),
                "max_feature_exposure": row.get("max_feature_exposure"),
                # Benchmark cells keep their exposure values (a designed mixed
                # population — null models are legitimately exposed); the
                # definition is unranked, recorded explicitly.
                "max_feature_exposure_reason": "unranked_predictions",
                "has_bmc": row.get("bmc") is not None,
                "has_horizon": row.get("horizon_model_sharpe_20") is not None,
                "has_perturb": row.get("perturb_ceiling_stability") is not None,
                "has_regime": row.get("regime_count") is not None,
                "tier": tier_value,
                "run_dir": str(path),
            }
        )
    return pl.DataFrame(rows, schema=UNIFIED_SCHEMA, strict=False)


def _registry_row(payload: dict[str, Any], run_file: Path) -> dict[str, Any]:
    """Build one unified row from a run.json record (legacy or experiments).

    ``has_full_version`` / family facts are patched in after the experiments
    scan; a scorecard value of 0.0 is real and must not fall through to the
    legacy train-OOF ``metrics``.
    """
    metrics = payload.get("metrics") or {}
    manifest = payload.get("manifest") or {}
    cfg = manifest.get("config") or {}
    data_cfg = cfg.get("data") or {}
    model_cfg = cfg.get("model") or {}
    run_cfg = cfg.get("run") or {}
    risk_cfg = cfg.get("risk") or {}

    scorecard = payload.get("scorecard") or {}
    sc_corr = scorecard.get("corr")
    sc_sharpe = scorecard.get("corr_sharpe_ac")
    sc_std = scorecard.get("std_corr")
    sc_dd = scorecard.get("max_drawdown")
    return {
        "model_id": payload.get("run_id") or run_file.parent.name,
        "source": "trained" if scorecard else "trained_legacy",
        "run_name": run_cfg.get("name", "unknown"),
        "family": run_cfg.get("name", "unknown"),
        "training_scope": "research",
        "has_full_version": False,
        "backend": model_cfg.get("backend", "unknown"),
        "preset": model_cfg.get("preset", "unknown"),
        "feature_set": data_cfg.get("feature_set", "unknown"),
        "feature_subset": data_cfg.get("feature_subset"),
        "n_targets": len(data_cfg.get("targets", [])),
        "targets": ", ".join(data_cfg.get("targets", [])),
        "neutralization_proportion": risk_cfg.get("neutralization_proportion"),
        "oof_device": manifest.get("oof_device"),
        "corr": sc_corr if sc_corr is not None else metrics.get("mean"),
        "corr_ci_low": scorecard.get("corr_ci_low"),
        "corr_ci_high": scorecard.get("corr_ci_high"),
        "corr_n_eras": scorecard.get("corr_n_eras"),
        "corr_sharpe_ac": (
            sc_sharpe if sc_sharpe is not None else metrics.get("sharpe")
        ),
        "corr_sharpe_ac_ci_low": scorecard.get("corr_sharpe_ac_ci_low"),
        "corr_sharpe_ac_ci_high": scorecard.get("corr_sharpe_ac_ci_high"),
        "corr_sharpe_ac_n_eras": scorecard.get("corr_sharpe_ac_n_eras"),
        "std_corr": sc_std if sc_std is not None else metrics.get("std"),
        "max_drawdown": (
            sc_dd if sc_dd is not None else metrics.get("max_drawdown")
        ),
        "deflated_sharpe": scorecard.get("deflated_sharpe"),
        "fnc": scorecard.get("fnc"),
        "mmc": scorecard.get("mmc"),
        "mmc_sharpe_ac": scorecard.get("mmc_sharpe_ac"),
        "bmc": scorecard.get("bmc"),
        "cwmm": scorecard.get("cwmm"),
        "mean_payout": scorecard.get("mean_payout"),
        "cagr_1y": scorecard.get("cagr_1y"),
        "gain_to_pain_ratio": scorecard.get("gain_to_pain_ratio"),
        "kelly_fraction": scorecard.get("kelly_fraction"),
        "mmc_down": scorecard.get("mmc_down"),
        "mmc_down_reason": scorecard.get("mmc_down_reason"),
        "turnover_mean": scorecard.get("turnover_mean"),
        "n_eras": scorecard.get("n_eras"),
        "rank_scalar": scorecard.get("rank_scalar"),
        "cvar5": scorecard.get("cvar5"),
        "burn_rate": scorecard.get("burn_rate"),
        # Exposure definition boundary (SEV-1 #14): post-fix runs
        # (scorecard_prediction_scale=percentile_rank) measure exposure
        # on the ranked (0,1) vector Numerai receives; legacy rows
        # measured ~machine-epsilon on unranked neutralized preds and
        # must not be compared with them — null + documented reason.
        "max_feature_exposure": (
            scorecard.get("max_feature_exposure")
            if manifest.get("scorecard_prediction_scale") == "percentile_rank"
            else None
        ),
        "max_feature_exposure_reason": (
            None
            if manifest.get("scorecard_prediction_scale") == "percentile_rank"
            else "pre_rank_fix_definition"
        ),
        "has_bmc": scorecard.get("bmc") is not None,
        "has_horizon": scorecard.get("horizon_model_sharpe_20") is not None,
        "has_perturb": scorecard.get("perturb_ceiling_stability") is not None,
        "has_regime": scorecard.get("regime_count") is not None,
        "tier": None,
        "run_dir": str(run_file.parent),
    }


def _partial_cross_check_cells(slot_dir: Path) -> dict[str, Any]:
    """Map a partial slot's cross-check ``scorecard.json`` onto metric cells.

    Reads the versioned record written by ``promote`` (schema v3: a nested
    ``scorecard`` block of MetricScorecard-shaped cells) and flattens exactly
    the fields the cross-check defines — corr/mmc/corr_sharpe_ac cells (value,
    ci bounds, n_eras), the fnc scalar, n_eras, deflated_sharpe, max_drawdown,
    burn_rate — onto ``_ALL_SCORECARD_FIELDS``-style flat keys. An unreadable
    or malformed record yields an empty mapping (the row stays all-None
    diagnostic); slot validity already requires ``scorecard.json`` to exist.
    """
    payload = _read_json_mapping(slot_dir / "scorecard.json")
    if payload is None:
        return {}
    scorecard = payload.get("scorecard")
    if not isinstance(scorecard, dict):
        return {}
    cells: dict[str, Any] = {}
    for metric in ("corr", "mmc", "corr_sharpe_ac"):
        cell = scorecard.get(metric)
        if isinstance(cell, dict):
            for key in ("value", "ci_low", "ci_high", "n_eras"):
                if cell.get(key) is not None:
                    cells[f"{metric}_{key}" if key != "value" else metric] = cell[key]
    for scalar in ("fnc", "n_eras", "deflated_sharpe", "max_drawdown", "burn_rate"):
        if scorecard.get(scalar) is not None:
            cells[scalar] = scorecard[scalar]
    return cells


def _export_row(
    version: lifecycle.ExportVersion, family: str, info: dict[str, Any]
) -> dict[str, Any]:
    """One unified row per VALID export slot (full or partial — diagnostic)."""
    row = dict.fromkeys(UNIFIED_SCHEMA.names())  # all metric cells null
    cfg_data = (version.config.get("data") or {}) if version.config else {}
    cfg_model = (version.config.get("model") or {}) if version.config else {}
    targets = cfg_data.get("targets") or []
    row.update(
        {
            "model_id": f"{family}::{version.scope}::{version.run_id}",
            "source": version.scope,
            "run_name": family,
            "family": family,
            "display_name": info["display_name"],
            "lifecycle_stage": info["lifecycle_stage"],
            "current_full_status": info["current_full_status"],
            "stale": info["stale"],
            "training_scope": version.scope,
            "has_full_version": False,
            "backend": cfg_model.get("backend"),
            "preset": cfg_model.get("preset"),
            "feature_set": cfg_data.get("feature_set"),
            "feature_subset": cfg_data.get("feature_subset"),
            "n_targets": len(targets) if targets else None,
            "targets": ", ".join(targets) if targets else None,
            "run_dir": str(version.slot_dir),
        }
    )
    if version.scope == "partial":
        # Cross-check metrics (spec §8 + §12 #15): partial rows carry their
        # post-fit scorecard on the family detail but stay diagnostic-only.
        # Only fields with a unified column land on the row (e.g. the mmc
        # value, not its CI cells — those render via the detail).
        row.update(
            {
                key: value
                for key, value in _partial_cross_check_cells(version.slot_dir).items()
                if key in UNIFIED_SCHEMA.names()
            }
        )
    return row


def _scan_experiment_families(
    experiments_root: Path,
    legacy_run_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], set[str], list[dict[str, Any]]]:
    """Iterate ``experiments/<family>/`` for exports + family metadata.

    Returns ``(family_info, promoted_families, export_rows)``: per-family
    display facts (display_name, lifecycle stage from ``derive_stage``, stale
    flag), the set of families with a genuine (non-rehearsal) full version,
    and one unified row per VALID export slot (``family::<scope>::<run_id>``).
    Exports whose ``run_id`` has no run record — neither in the legacy registry
    nor ``experiments/<family>/runs/<run_id>/run.json`` — warn (dangling
    lineage) but still render.
    """
    family_info: dict[str, dict[str, Any]] = {}
    promoted_families: set[str] = set()
    export_rows: list[dict[str, Any]] = []
    root = Path(experiments_root)
    if not root.is_dir():
        return family_info, promoted_families, export_rows
    for family_dir in sorted(root.iterdir()):
        if not family_dir.is_dir() or not paths.SLUG_RE.fullmatch(family_dir.name):
            continue
        family = family_dir.name
        meta_payload = _read_json_mapping(family_dir / "meta.json") or {}
        display_name = (
            str(meta_payload.get("display_name") or family)
            if isinstance(meta_payload, dict)
            else family
        )
        staked = lifecycle.load_staked_record(family_dir / "meta.json")
        stage, full_status = lifecycle.derive_stage(family, staked)
        stale = bool(
            staked is not None and staked.status == "active" and stage != "staked"
        )
        info = {
            "display_name": display_name,
            "lifecycle_stage": stage,
            "current_full_status": full_status,
            "stale": stale,
        }
        family_info[family] = info
        full_versions = [
            v for v in lifecycle.scan_valid_exports(family, "full") if not v.rehearsal
        ]
        partial_versions = [
            v
            for v in lifecycle.scan_valid_exports(family, "partial")
            if not v.rehearsal
        ]
        if full_versions:
            promoted_families.add(family)
        for version in [*full_versions, *partial_versions]:
            if (
                version.run_id not in legacy_run_ids
                and not (family_dir / "runs" / version.run_id / "run.json").is_file()
            ):
                logger.warning(
                    "nmr.dashboard: family %s %s export %s lineage dangling "
                    "(run_id %s has no run record)",
                    family,
                    version.scope,
                    version.run_id[:8],
                    version.run_id,
                )
            export_rows.append(_export_row(version, family, info))
    return family_info, promoted_families, export_rows


def load_unified_leaderboard(
    registry_dir: Path,
    benchmark_path: Path | None | bool = None,
    reports_dir: Path | None = None,
    models_dir: Path | None = None,
) -> pl.DataFrame:
    """Load run records and (optionally) benchmark rows into one frame.

    Task-10 bridge scan: run records come from BOTH the legacy registry
    (``<registry_dir>/<run_id>/run.json`` — the pre-Task-11 recording side)
    and the experiments layout (``experiments/<family>/runs/<run_id>/run.json``);
    exports come from the experiments layout via ``nmr.lifecycle`` — one
    ``family::full::<run_id>`` row per VALID full slot and one
    ``family::partial::<run_id>`` per VALID partial slot (diagnostic-only,
    never ranked). ``models_dir`` overrides the experiments root (test
    isolation); it defaults to ``paths.EXPERIMENTS_ROOT``. Corrupt ``run.json``
    files are skipped. ``benchmark_path=False`` disables benchmark loading
    (registry-only); otherwise a missing path falls through the resolution
    chain.
    """
    rows: list[dict] = []
    registry = Path(registry_dir)
    experiments_root = (
        Path(models_dir) if models_dir is not None else paths.EXPERIMENTS_ROOT
    )

    # 1) legacy registry run records (pre-Task-11 recording side)
    legacy_run_ids: set[str] = set()
    for run_file in sorted(registry.glob("*/run.json")):
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        rows.append(_registry_row(payload, run_file))
        legacy_run_ids.add(str(payload.get("run_id") or run_file.parent.name))

    # 2) experiments layout: family metadata + export slots (full/partial)
    family_info, promoted_families, export_rows = _scan_experiment_families(
        experiments_root, legacy_run_ids
    )
    rows.extend(export_rows)

    # 3) experiments run records (Task-11 side; deduped against legacy)
    seen_run_ids = set(legacy_run_ids)
    if experiments_root.is_dir():
        for family_dir in sorted(experiments_root.iterdir()):
            if not family_dir.is_dir() or not paths.SLUG_RE.fullmatch(family_dir.name):
                continue
            runs_dir = family_dir / "runs"
            if not runs_dir.is_dir():
                continue
            for run_file in sorted(runs_dir.glob("*/run.json")):
                run_id = run_file.parent.name
                if run_id in seen_run_ids:
                    continue
                try:
                    payload = json.loads(run_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if not isinstance(payload, dict):
                    continue
                rows.append(_registry_row(payload, run_file))
                seen_run_ids.add(run_id)

    # 4) family facts on trained rows (display_name / lifecycle / stale)
    for row in rows:
        if row["source"] not in ("trained", "trained_legacy"):
            continue
        info = family_info.get(str(row.get("family"))) or {}
        row["display_name"] = info.get("display_name") or row.get("family")
        row["lifecycle_stage"] = info.get("lifecycle_stage")
        row["current_full_status"] = info.get("current_full_status")
        row["stale"] = bool(info.get("stale"))
        row["has_full_version"] = str(row.get("family")) in promoted_families

    resolved = resolve_benchmark_path(benchmark_path, reports_dir=reports_dir)
    if resolved is not None:
        rows.extend(load_benchmark_frame(resolved).to_dicts())

    if not rows:
        return pl.DataFrame(schema=UNIFIED_SCHEMA)
    return pl.DataFrame(rows, schema=UNIFIED_SCHEMA, strict=False)


_GATE_THRESHOLD_ATTRS = {
    "corr": "corr_min",
    "corr_sharpe_ac": "corr_sharpe_ac_min",
    "fnc": "fnc_min",
    "gain_to_pain_ratio": "gain_to_pain_min",
    "cagr_1y": "cagr_min",
}
# A6 (audit SEV-2 #4): deflated_sharpe is display-only — no search history
# exists at gate time to bind deflation to, so gating on it was false
# assurance. The dashboard's CAPITAL READY badge must agree with the benchmark
# gate, which no longer checks it.
_GATE_FIELDS = (
    "corr",
    "corr_sharpe_ac",
    "fnc",
    "gain_to_pain_ratio",
    "cagr_1y",
    "turnover_mean",
)
_STATUS_SCHEMA = pl.Schema(
    {
        "model_id": pl.String,
        "status": pl.String,
        **{f"gate_{f}": pl.Boolean for f in _GATE_FIELDS},
    }
)


def read_champion_pointer(champion_path: Path) -> str | None:
    """Opaque champion pointer; missing or corrupt file -> None."""
    path = Path(champion_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    return run_id if isinstance(run_id, str) else None


def _gate_receipt(field: str, row: dict, gate) -> bool | None:
    value = row.get(field)
    if value is None:
        return None
    measured = float(value)
    if field == "turnover_mean":
        return measured <= float(gate.turnover_max)
    if field == "cagr_1y":
        return measured > float(gate.cagr_min)  # strict, mirrors assert_tier4_gate
    return measured >= float(getattr(gate, _GATE_THRESHOLD_ATTRS[field]))


def evaluate_gate_status(
    leaderboard: pl.DataFrame,
    gate_config_path: Path,
    champion_path: Path,
) -> pl.DataFrame:
    """Project each row against the tier-4 gate (read-only, never enforces).

    Status ladder: full rows are stamped ``FULL`` (in-sample metrics, never
    gated); benchmark rows are exempt (``GATE HURDLE`` for the gate
    file's reference column, ``BENCHMARK`` otherwise); registry rows are
    ``CHAMPION`` (champion.json pointer), ``CAPITAL READY`` (all hard
    hurdles), or ``RESEARCH``. Per-field receipts mirror
    ``assert_tier4_gate``: >= for most fields, strict > for cagr_1y, turnover
    exempt when None.
    """
    from nmr.benchmark import load_benchmark_file

    file_cfg = load_benchmark_file(gate_config_path)
    gate = file_cfg.gate
    if gate is None:
        raise ValueError(f"gate config {gate_config_path} has no gate section")
    reference_column = file_cfg.reference_column
    champion_id = read_champion_pointer(champion_path)

    rows: list[dict] = []
    for row in leaderboard.to_dicts():
        model_id = row["model_id"]
        if row["source"] == "full":
            status = "FULL"
        elif row["source"] == "partial":
            status = "PARTIAL"  # train-only cross-check — never gated
        elif row["source"] == "benchmark":
            status = "GATE HURDLE" if model_id == reference_column else "BENCHMARK"
        elif (
            all(
                _gate_receipt(f, row, gate) is True
                for f in _GATE_FIELDS
                if f != "turnover_mean"
            )
            # turnover is exempt only when None; a measured violation blocks the gate
            and _gate_receipt("turnover_mean", row, gate) is not False
        ):
            status = "CHAMPION" if model_id == champion_id else "CAPITAL READY"
        else:
            status = "RESEARCH"
        rows.append(
            {
                "model_id": model_id,
                "status": status,
                **{f"gate_{f}": _gate_receipt(f, row, gate) for f in _GATE_FIELDS},
            }
        )
    if not rows:
        return pl.DataFrame(schema=_STATUS_SCHEMA)
    return pl.DataFrame(rows, schema=_STATUS_SCHEMA, strict=False)


_CAPITAL_SCALAR_CELLS = ("cagr_1y", "gain_to_pain_ratio", "kelly_fraction")


def _has_stored_capital_block(row: dict) -> bool:
    return all(row.get(c) is not None for c in _CAPITAL_SCALAR_CELLS)


def _load_shared_lookups(
    data_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]] | None:
    """Load the 86-era meta-overlap targets + meta lookups once.

    Returns ``(targets_86, meta, meta_eras)`` or None when either data asset
    is missing.
    """
    data = Path(data_dir)
    targets_path = data / "validation.parquet"
    meta_path = data / "meta_model.parquet"
    if not (targets_path.exists() and meta_path.exists()):
        return None
    targets = pl.read_parquet(targets_path, columns=["era", "id", "target"])
    meta = pl.read_parquet(meta_path, columns=["era", "id", "numerai_meta_model"])
    meta_eras = sorted(meta.get_column("era").unique().to_list(), key=int)
    targets_86 = targets.filter(pl.col("era").is_in(meta_eras))
    return targets_86, meta, meta_eras


def _per_era_metrics(
    preds_path: Path,
    targets_86: pl.DataFrame,
    meta: pl.DataFrame,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Per-era CORR, MMC, and meta-CORR for one stored predictions file.

    Joins on [era, id] against the shared lookups — the meta inner join
    restricts to the standard 86-era overlap window.
    """
    preds = pl.read_parquet(preds_path, columns=["era", "id", "prediction"])
    joined = preds.join(targets_86, on=["era", "id"], how="inner").join(
        meta, on=["era", "id"], how="inner"
    )
    engine = EvaluationEngine()
    corr = engine.per_era_corr(joined, pred_col="prediction", target_col="target")
    mmc = engine.per_era_mmc(
        joined,
        pred_col="prediction",
        meta_col="numerai_meta_model",
        target_col="target",
    )
    meta_corr = engine.per_era_corr(
        joined, pred_col="numerai_meta_model", target_col="target"
    )
    return corr, mmc, meta_corr


@dataclass(frozen=True)
class _V2Lookups:
    targets: pl.DataFrame
    target_20_col: str
    target_60_col: str
    meta: pl.DataFrame
    meta_eras: list[str]
    benchmarks: pl.DataFrame


def _resolve_horizon_targets(schema_cols: Sequence[str]) -> tuple[str, str]:
    """Resolve the 20D/60D target columns with fallback chains (decision #10)."""
    target_20 = next(
        (
            c
            for c in ("target_ender_20", "target_cyrusd_20", "target_20", "target")
            if c in schema_cols
        ),
        "target",
    )
    target_60 = next(
        (
            c
            for c in ("target_ender_60", "target_cyrusd_60", "target_60", "target")
            if c in schema_cols
        ),
        target_20,
    )
    return target_20, target_60


def _load_v2_lookups(data_dir: Path, tier4_columns: Sequence[str]) -> _V2Lookups | None:
    """Single-pass v2 lookups: deduped targets, meta, benchmarks on meta eras.

    Returns None when validation.parquet or meta_model.parquet is missing;
    benchmarks are optional (empty frame when the file is absent). Reads the
    tier-4 reference columns that exist in the benchmark parquet; a requested
    column that the file lacks is skipped with a warning (schema drift).
    """
    data = Path(data_dir)
    targets_path = data / "validation.parquet"
    meta_path = data / "meta_model.parquet"
    bench_path = data / "validation_benchmark_models.parquet"
    if not (targets_path.exists() and meta_path.exists()):
        return None
    schema_cols = pl.read_parquet_schema(targets_path).names()
    target_20, target_60 = _resolve_horizon_targets(schema_cols)
    # decision #18: deduped — both horizons may resolve to "target"
    target_cols = list(dict.fromkeys(["era", "id", "target", target_20, target_60]))
    targets = pl.read_parquet(targets_path, columns=target_cols)
    meta = pl.read_parquet(meta_path, columns=["era", "id", "numerai_meta_model"])
    meta_eras = sorted_era_labels(meta.get_column("era").unique().to_list())
    targets_86 = targets.filter(pl.col("era").is_in(meta_eras))
    present_columns: list[str] = []
    if bench_path.exists():
        bench_schema = pl.read_parquet_schema(bench_path).names()
        missing = [c for c in tier4_columns if c not in bench_schema]
        if missing:
            logger.warning(
                "nmr.dashboard: benchmark parquet %s lacks tier-4 columns %s; "
                "those reference slices are skipped",
                bench_path,
                missing,
            )
        present_columns = [c for c in tier4_columns if c in bench_schema]
    benchmarks = pl.DataFrame(
        schema={
            "era": pl.String,
            "id": pl.String,
            **{c: pl.Float64 for c in present_columns},
        }
    )
    if present_columns:
        benchmarks = pl.read_parquet(
            bench_path, columns=["era", "id", *present_columns]
        ).filter(pl.col("era").is_in(meta_eras))
    return _V2Lookups(
        targets=targets_86,
        target_20_col=target_20,
        target_60_col=target_60,
        meta=meta,
        meta_eras=meta_eras,
        benchmarks=benchmarks,
    )


def reconcile_capital_metrics(
    leaderboard: pl.DataFrame,
    data_dir: Path,
) -> pl.DataFrame:
    """Fill missing capital cells by recomputing from stored parquets.

    Stored-first: rows whose scorecard carries all three scalar capital cells
    are trusted verbatim (including a stored ``mmc_down=None`` with reason).
    Everything else for trained/trained_legacy rows is recomputed via the
    oracle-parity evaluation/payout paths. Registry files are never written.
    """
    rows = leaderboard.to_dicts()
    needs_recompute = [
        row
        for row in rows
        if row["source"] in ("trained", "trained_legacy")
        and not _has_stored_capital_block(row)
    ]
    if not needs_recompute:
        return leaderboard

    lookups = _load_shared_lookups(data_dir)
    if lookups is None:
        logger.warning(
            "nmr.dashboard: v5.3 targets/meta_model missing at %s; "
            "capital cells left None",
            data_dir,
        )
        return leaderboard
    targets_86, meta, _ = lookups

    for row in rows:
        if not (
            row["source"] in ("trained", "trained_legacy")
            and not _has_stored_capital_block(row)
        ):
            continue
        preds_path = Path(row["run_dir"]) / "validation_preds.parquet"
        if not preds_path.exists():
            logger.warning(
                "nmr.dashboard: %s has no validation_preds.parquet; "
                "capital cells left None",
                row["model_id"],
            )
            continue
        corr, mmc, meta_corr = _per_era_metrics(preds_path, targets_86, meta)
        series = payout_series(corr, mmc)
        row["cagr_1y"] = annual_compounded_return(series.clipped)
        row["gain_to_pain_ratio"] = gain_to_pain_ratio(series.clipped)
        row["kelly_fraction"] = kelly_fraction(series.raw)
        downside = downside_era_indices(meta_corr)
        if len(downside) >= MMC_DOWN_MIN_ERAS:
            row["mmc_down"] = float(np.mean([mmc[e] for e in downside]))
            row["mmc_down_reason"] = None
        else:
            row["mmc_down"] = None
            row["mmc_down_reason"] = "insufficient_downside_eras"

    return pl.DataFrame(rows, schema=UNIFIED_SCHEMA, strict=False)


def _family_display_name(family: str) -> str:
    """meta.json display_name for a family slug; falls back to the slug."""
    if not paths.SLUG_RE.fullmatch(family):
        return family
    meta = _read_json_mapping(paths.experiment_dir(family) / "meta.json")
    if isinstance(meta, dict) and meta.get("display_name"):
        return str(meta["display_name"])
    return family


def _series_label(registry_dir: Path, run_id: str) -> str:
    run_file = Path(registry_dir) / run_id / "run.json"
    if not run_file.exists():
        # Task-11 bridge: experiments runs live under experiments/<family>/runs/
        run_file = next(
            iter(paths.EXPERIMENTS_ROOT.glob(f"*/runs/{run_id}/run.json")),
            run_file,
        )
    name = "unknown"
    if run_file.exists():
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
        if isinstance(payload, dict):
            manifest = payload.get("manifest") or {}
            name = (manifest.get("config") or {}).get("run", {}).get("name", "unknown")
    display = _family_display_name(name) if name != "unknown" else name
    return f"{display} · {run_id[:8]}"


_METRIC_NAMES = ("payout", "corr20", "mmc20", "corr60", "mmc60", "bmc", "cwmm")


def _cumulative_from_standard(
    standard: Sequence[float | None], *, payout: bool
) -> list[float | None]:
    result: list[float | None] = []
    accumulator = 1.0 if payout else 0.0
    available = True
    for value in standard:
        if not available or value is None or not np.isfinite(value):
            result.append(None)
            available = False
            continue
        accumulator = accumulator * (1.0 + value) if payout else accumulator + value
        result.append(float(accumulator))
    return result


def _align_era_values(
    axis: Sequence[str], values: Mapping[str, float]
) -> list[float | None]:
    return [
        (
            float(value)
            if (value := values.get(era)) is not None and np.isfinite(value)
            else None
        )
        for era in axis
    ]


def extract_multimetric_timeseries(
    registry_dir: Path,
    data_dir: Path,
    run_ids: Sequence[str],
    include_tier4_ref: bool = True,
    tier4_columns: Sequence[str] = ("v53_lgbm_ender60", "v53_lgbm_ender20"),
) -> dict[str, Any]:
    """7-metric per-era trajectories over the standardized meta window.

    Payout is anchored to main_target="target" (decision #19); correlation
    metrics use cumsum, payout uses cumprod (decision #9); the tier-4 BMC is
    short-circuited to zeros (decision #11); an absent benchmark frame or a
    model sharing no era with the meta window leaves its unavailable bmc/cwmm
    slices as null and skips its payout slice with warnings (decision #23). Every tier-4
    reference column (first = the gated capital line, the BMC benchmark for
    other models) is rendered as a reference curve. Never raises on missing
    assets — returns the empty payload.
    """
    lookups = _load_v2_lookups(data_dir, tier4_columns)
    if lookups is None:
        logger.warning(
            "nmr.dashboard: data assets missing at %s; empty timeseries", data_dir
        )
        return {"eras": [], "meta_downside_mask": [], "metrics": {}, "drawdowns": {}}

    axis = lookups.meta_eras
    engine = EvaluationEngine()
    meta_joined = lookups.meta.join(lookups.targets, on=["era", "id"], how="inner")
    meta_corr = engine.per_era_corr(
        meta_joined, pred_col="numerai_meta_model", target_col="target"
    )
    mask = [bool(meta_corr[era] < 0.0) for era in axis]

    metrics: dict[str, dict] = {name: {} for name in _METRIC_NAMES}
    drawdowns: dict[str, list[float | None]] = {}

    ref_set = set(tier4_columns)
    primary_ref = tier4_columns[0] if tier4_columns else None
    ids = [mid for mid in sorted(set(run_ids)) if mid not in ref_set]
    if include_tier4_ref and lookups.benchmarks.height > 0:
        ids.extend([c for c in tier4_columns if c in lookups.benchmarks.columns])

    for model_id in ids:
        if model_id in ref_set:
            preds = lookups.benchmarks.select(
                ["era", "id", pl.col(model_id).alias("prediction")]
            )
            label = model_id
        else:
            preds_path = Path(registry_dir) / model_id / "validation_preds.parquet"
            if not preds_path.exists():
                logger.warning("nmr.dashboard: skipping missing preds %s", preds_path)
                continue
            preds = pl.read_parquet(preds_path, columns=["era", "id", "prediction"])
            label = _series_label(registry_dir, model_id)

        joined = preds.join(lookups.targets, on=["era", "id"], how="inner").join(
            lookups.meta, on=["era", "id"], how="inner"
        )
        corr_t = engine.per_era_corr(joined, pred_col="prediction", target_col="target")
        mmc_t = engine.per_era_mmc(
            joined,
            pred_col="prediction",
            meta_col="numerai_meta_model",
            target_col="target",
        )
        # decision #23: a stale run whose preds share no era with the meta
        # window must not abort the report — skip its payout slice (drawdowns
        # derive from payout wealth) while the other metric slices still render
        if set(corr_t) & set(mmc_t):
            pay = payout_series(corr_t, mmc_t)
            clipped_by_era = dict(zip(pay.eras, pay.clipped))
            standard = _align_era_values(axis, clipped_by_era)
            metrics["payout"][model_id] = {
                "standard": standard,
                "cumulative": _cumulative_from_standard(standard, payout=True),
                "label": label,
            }
            wealth = metrics["payout"][model_id]["cumulative"]
            peak = float("-inf")
            drawdowns[model_id] = []
            for value in wealth:
                if value is None:
                    drawdowns[model_id].append(None)
                    continue
                peak = max(peak, value)
                drawdowns[model_id].append(value / peak - 1.0 if peak > 0.0 else 0.0)
        else:
            logger.warning(
                "nmr.dashboard: %s shares no eras with the meta window; "
                "payout slice skipped",
                model_id,
            )

        horizon_metrics = (
            ("corr20", lookups.target_20_col, "corr"),
            ("corr60", lookups.target_60_col, "corr"),
            ("mmc20", lookups.target_20_col, "mmc"),
            ("mmc60", lookups.target_60_col, "mmc"),
        )
        for name, target_col, kind in horizon_metrics:
            # resolved target columns are guaranteed present in joined by construction
            if kind == "corr":
                per = engine.per_era_corr(
                    joined, pred_col="prediction", target_col=target_col
                )
            else:
                per = engine.per_era_mmc(
                    joined,
                    pred_col="prediction",
                    meta_col="numerai_meta_model",
                    target_col=target_col,
                )
            aligned = _align_era_values(axis, per)
            metrics[name][model_id] = {
                "standard": aligned,
                "cumulative": _cumulative_from_standard(aligned, payout=False),
                "label": label,
            }

        if model_id in ref_set:
            observed_eras = set(joined.get_column("era").to_list())
            zeros = [0.0 if era in observed_eras else None for era in axis]
            metrics["bmc"][model_id] = {
                "standard": zeros,
                "cumulative": zeros,
                "label": label,
            }
        elif primary_ref is not None:
            joined_b = joined.join(lookups.benchmarks, on=["era", "id"], how="inner")
            if lookups.benchmarks.height > 0 and joined_b.height > 0:
                # reporting path relaxes the evaluation vacuity gate (real meta
                # window satisfies 20 anyway); absent eras remain unavailable
                per_bmc = engine.per_era_bmc(
                    joined_b,
                    pred_col="prediction",
                    benchmark_col=primary_ref,
                    target_col="target",
                    min_overlap_eras=1,
                )
                aligned = _align_era_values(axis, per_bmc)
                metrics["bmc"][model_id] = {
                    "standard": aligned,
                    "cumulative": _cumulative_from_standard(aligned, payout=False),
                    "label": label,
                }
            else:
                if lookups.benchmarks.height == 0:
                    logger.warning(
                        "nmr.dashboard: benchmark models absent at %s; bmc unavailable",
                        data_dir,
                    )
                else:
                    logger.warning(
                        "nmr.dashboard: %s shares no eras with the benchmark "
                        "window; bmc unavailable",
                        model_id,
                    )
                unavailable = [None for _ in axis]
                metrics["bmc"][model_id] = {
                    "standard": unavailable,
                    "cumulative": unavailable,
                    "label": label,
                }
        else:
            # no primary reference configured — BMC is unavailable
            metrics["bmc"][model_id] = {
                "standard": [None for _ in axis],
                "cumulative": [None for _ in axis],
                "label": label,
            }

        if joined.height > 0:
            # reporting path relaxes the evaluation vacuity gate (real meta window
            # satisfies 20 anyway); absent eras remain unavailable
            per_cwmm = engine.per_era_cwmm(
                joined,
                pred_col="prediction",
                meta_col="numerai_meta_model",
                min_overlap_eras=1,
            )
            aligned = _align_era_values(axis, per_cwmm)
            metrics["cwmm"][model_id] = {
                "standard": aligned,
                "cumulative": _cumulative_from_standard(aligned, payout=False),
                "label": label,
            }
        else:
            # zero-overlap model: CWMM is unavailable
            metrics["cwmm"][model_id] = {
                "standard": [None for _ in axis],
                "cumulative": [None for _ in axis],
                "label": label,
            }

    return {
        "eras": axis,
        "meta_downside_mask": mask,
        "metrics": metrics,
        "drawdowns": drawdowns,
    }


def extract_pairwise_similarity_matrix(
    registry_dir: Path,
    data_dir: Path,
    run_ids: Sequence[str],
    include_tier4_ref: bool = True,
    tier4_column: str = "v53_lgbm_ender60",
) -> tuple[list[str], list[str], list[list[float]], dict[str, Any]]:
    """Pairwise rank-gaussian pooled-Pearson similarity over the meta window.

    Single multi-way inner join across all candidates (decision #12), global
    intersection (not pairwise-complete); the tier-4 candidate is read from
    the benchmark parquet, never a registry dir (decision #20); degenerate
    columns guarded (decision #15); matrix clamped to [-1, 1] (decision #22).
    Returns (labels, run_ids, matrix, stress_stats) with stress_stats =
    {"mean_delta": mean off-diagonal (rho_stress - rho_normal) | None,
     "n_pairs": int} (decision #26).
    """
    lookups = _load_v2_lookups(data_dir, (tier4_column,))
    if lookups is None:
        logger.warning(
            "nmr.dashboard: data assets missing at %s; empty similarity", data_dir
        )
        return [], [], [], {"mean_delta": None, "n_pairs": 0}

    axis = lookups.meta_eras
    frames: list[pl.DataFrame] = []
    ids_used: list[str] = []
    labels: list[str] = []
    for model_id in sorted(set(run_ids)):
        if model_id == tier4_column:
            continue
        preds_path = Path(registry_dir) / model_id / "validation_preds.parquet"
        if not preds_path.exists():
            logger.warning("nmr.dashboard: skipping missing preds %s", preds_path)
            continue
        frames.append(
            pl.read_parquet(preds_path, columns=["era", "id", "prediction"]).rename(
                {"prediction": model_id}
            )
        )
        ids_used.append(model_id)
        labels.append(_series_label(registry_dir, model_id))
    if include_tier4_ref and lookups.benchmarks.height > 0:
        frames.append(lookups.benchmarks)
        ids_used.append(tier4_column)
        labels.append(tier4_column)
    if not frames:
        return [], [], [], {"mean_delta": None, "n_pairs": 0}

    aligned = frames[0]
    for frame in frames[1:]:
        aligned = aligned.join(frame, on=["era", "id"], how="inner")
    gauss = Ensembler.rank_normalize(aligned, pred_cols=ids_used, era_col="era")

    columns: list[np.ndarray] = []
    for model_id in ids_used:
        # polars 1.41 Series.to_numpy has no dtype kwarg; cast numpy-side
        arr = gauss.get_column(model_id).to_numpy().astype(np.float64, copy=False)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if np.std(arr) <= 0.0:
            arr = np.zeros_like(arr)
        columns.append(arr)
    stacked = np.vstack(columns)
    # decision #15/#22: zero-variance columns contribute zeros; a lone
    # degenerate candidate (0-d/scalar corrcoef) keeps a valid 1x1 identity
    matrix = np.atleast_2d(np.clip(np.corrcoef(stacked), -1.0, 1.0))
    matrix = np.nan_to_num(matrix, nan=0.0)
    np.fill_diagonal(matrix, 1.0)

    meta_corr = EvaluationEngine().per_era_corr(
        lookups.meta.join(lookups.targets, on=["era", "id"], how="inner"),
        pred_col="numerai_meta_model",
        target_col="target",
    )
    stress_eras = {era for era in axis if meta_corr.get(era, 0.0) < 0.0}
    era_arr = gauss.get_column("era").to_list()
    stress_idx = np.asarray([era in stress_eras for era in era_arr])

    def _mean_offdiag(mat: np.ndarray) -> float | None:
        mat = np.atleast_2d(mat)
        if mat.shape[0] < 2:
            return None
        upper = [
            mat[i, j] for i in range(mat.shape[0]) for j in range(i + 1, mat.shape[0])
        ]
        return float(np.mean(upper)) if upper else None

    mean_delta = None
    if stress_idx.sum() >= 5:
        # degenerate columns inside the stress/normal subsets produce NaN
        # correlations — neutralize them (0) before the off-diagonal mean so
        # mean_delta is always finite (0-d scalar corrcoef kept 2-d too)
        rho_stress = _mean_offdiag(
            np.clip(
                np.nan_to_num(
                    np.atleast_2d(np.corrcoef(stacked[:, stress_idx])), nan=0.0
                ),
                -1.0,
                1.0,
            )
        )
        rho_normal = _mean_offdiag(
            np.clip(
                np.nan_to_num(
                    np.atleast_2d(np.corrcoef(stacked[:, ~stress_idx])), nan=0.0
                ),
                -1.0,
                1.0,
            )
        )
        if rho_stress is not None and rho_normal is not None:
            mean_delta = rho_stress - rho_normal

    n_pairs = len(ids_used) * (len(ids_used) - 1) // 2
    return (
        labels,
        ids_used,
        matrix.tolist(),
        {"mean_delta": mean_delta, "n_pairs": n_pairs},
    )
