"""Pure analytical engine for the executive performance dashboard.

Registry scans, benchmark reconciliation, gate projection, capital-cell
recompute, and payout timeseries extraction. Plotly/Streamlit-free; every
function here is covered by tests/test_dashboard.py.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nmr.config import REPO_ROOT
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
    "UNIFIED_SCHEMA",
    "evaluate_gate_status",
    "extract_payout_timeseries",
    "load_benchmark_frame",
    "load_unified_leaderboard",
    "read_champion_pointer",
    "reconcile_capital_metrics",
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
        "model_id": pl.String, "source": pl.String, "run_name": pl.String,
        "backend": pl.String, "preset": pl.String, "feature_set": pl.String,
        "feature_subset": pl.String, "n_targets": pl.Int64, "targets": pl.String,
        "neutralization_proportion": pl.Float64, "oof_device": pl.String,
        "corr": pl.Float64, "corr_ci_low": pl.Float64, "corr_ci_high": pl.Float64,
        "corr_n_eras": pl.Int64,
        "corr_sharpe_ac": pl.Float64, "corr_sharpe_ac_ci_low": pl.Float64,
        "corr_sharpe_ac_ci_high": pl.Float64, "corr_sharpe_ac_n_eras": pl.Int64,
        "std_corr": pl.Float64, "max_drawdown": pl.Float64,
        "deflated_sharpe": pl.Float64, "fnc": pl.Float64, "mmc": pl.Float64,
        "mmc_sharpe_ac": pl.Float64, "bmc": pl.Float64, "cwmm": pl.Float64,
        "mean_payout": pl.Float64,
        "cagr_1y": pl.Float64, "gain_to_pain_ratio": pl.Float64,
        "kelly_fraction": pl.Float64, "mmc_down": pl.Float64,
        "mmc_down_reason": pl.String, "turnover_mean": pl.Float64,
        "n_eras": pl.Int64, "rank_scalar": pl.Float64, "cvar5": pl.Float64,
        "burn_rate": pl.Float64, "max_feature_exposure": pl.Float64,
        "has_bmc": pl.Boolean, "has_horizon": pl.Boolean,
        "has_perturb": pl.Boolean, "has_regime": pl.Boolean,
        "tier": pl.Int64, "run_dir": pl.String,
    }
)


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
                "run_name": row.get("strategy_group")
                or (
                    f"tier{int(tier_value)}"
                    if tier_value is not None
                    else "benchmark"
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
                "std_corr": row.get("std_corr", 0.0),
                "max_drawdown": row.get("max_drawdown", 0.0),
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
                "has_bmc": row.get("bmc") is not None,
                "has_horizon": row.get("horizon_model_sharpe_20") is not None,
                "has_perturb": row.get("perturb_ceiling_stability") is not None,
                "has_regime": row.get("regime_count") is not None,
                "tier": tier_value,
                "run_dir": str(path),
            }
        )
    return pl.DataFrame(rows, schema=UNIFIED_SCHEMA, strict=False)


def load_unified_leaderboard(
    registry_dir: Path,
    benchmark_path: Path | None | bool = None,
    reports_dir: Path | None = None,
) -> pl.DataFrame:
    """Load registry runs and (optionally) benchmark rows into one frame.

    Explicit-None discipline: a scorecard value of 0.0 is real and must not
    fall through to the legacy train-OOF ``metrics``. Corrupt ``run.json``
    files are skipped. ``benchmark_path=False`` disables benchmark loading
    (registry-only); otherwise a missing path falls through the resolution
    chain.
    """
    rows: list[dict] = []
    registry = Path(registry_dir)
    for run_file in sorted(registry.glob("*/run.json")):
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
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
        rows.append(
            {
                "model_id": payload.get("run_id") or run_file.parent.name,
                "source": "trained" if scorecard else "trained_legacy",
                "run_name": run_cfg.get("name", "unknown"),
                "backend": model_cfg.get("backend", "unknown"),
                "preset": model_cfg.get("preset", "unknown"),
                "feature_set": data_cfg.get("feature_set", "unknown"),
                "feature_subset": data_cfg.get("feature_subset"),
                "n_targets": len(data_cfg.get("targets", [])),
                "targets": ", ".join(data_cfg.get("targets", [])),
                "neutralization_proportion": risk_cfg.get("neutralization_proportion"),
                "oof_device": manifest.get("oof_device"),
                "corr": float(sc_corr if sc_corr is not None else metrics.get("mean", 0.0)),
                "corr_ci_low": scorecard.get("corr_ci_low"),
                "corr_ci_high": scorecard.get("corr_ci_high"),
                "corr_n_eras": scorecard.get("corr_n_eras"),
                "corr_sharpe_ac": float(
                    sc_sharpe if sc_sharpe is not None else metrics.get("sharpe", 0.0)
                ),
                "corr_sharpe_ac_ci_low": scorecard.get("corr_sharpe_ac_ci_low"),
                "corr_sharpe_ac_ci_high": scorecard.get("corr_sharpe_ac_ci_high"),
                "corr_sharpe_ac_n_eras": scorecard.get("corr_sharpe_ac_n_eras"),
                "std_corr": float(sc_std if sc_std is not None else metrics.get("std", 0.0)),
                "max_drawdown": float(
                    sc_dd if sc_dd is not None else metrics.get("max_drawdown", 0.0)
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
                "max_feature_exposure": scorecard.get("max_feature_exposure"),
                "has_bmc": scorecard.get("bmc") is not None,
                "has_horizon": scorecard.get("horizon_model_sharpe_20") is not None,
                "has_perturb": scorecard.get("perturb_ceiling_stability") is not None,
                "has_regime": scorecard.get("regime_count") is not None,
                "tier": None,
                "run_dir": str(run_file.parent),
            }
        )

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
    "deflated_sharpe": "deflated_sharpe_min",
    "gain_to_pain_ratio": "gain_to_pain_min",
    "cagr_1y": "cagr_min",
}
_GATE_FIELDS = ("corr", "corr_sharpe_ac", "fnc", "deflated_sharpe",
                "gain_to_pain_ratio", "cagr_1y", "turnover_mean")
_STATUS_SCHEMA = pl.Schema(
    {"model_id": pl.String, "status": pl.String,
     **{f"gate_{f}": pl.Boolean for f in _GATE_FIELDS}}
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

    Status ladder: benchmark rows are exempt (``GATE HURDLE`` for the gate
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
        if row["source"] == "benchmark":
            status = "GATE HURDLE" if model_id == reference_column else "BENCHMARK"
        elif champion_id is not None and model_id == champion_id:
            status = "CHAMPION"
        elif (
            all(
                _gate_receipt(f, row, gate) is True
                for f in _GATE_FIELDS
                if f != "turnover_mean"
            )
            # turnover is exempt only when None; a measured violation blocks the gate
            and _gate_receipt("turnover_mean", row, gate) is not False
        ):
            status = "CAPITAL READY"
        else:
            status = "RESEARCH"
        rows.append(
            {"model_id": model_id, "status": status,
             **{f"gate_{f}": _gate_receipt(f, row, gate) for f in _GATE_FIELDS}}
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
    meta = pl.read_parquet(
        meta_path, columns=["era", "id", "numerai_meta_model"]
    )
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
    joined = (
        preds.join(targets_86, on=["era", "id"], how="inner")
        .join(meta, on=["era", "id"], how="inner")
    )
    engine = EvaluationEngine()
    corr = engine.per_era_corr(joined, pred_col="prediction", target_col="target")
    mmc = engine.per_era_mmc(
        joined, pred_col="prediction",
        meta_col="numerai_meta_model", target_col="target",
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
        (c for c in ("target_ender_20", "target_cyrusd_20", "target_20", "target")
         if c in schema_cols),
        "target",
    )
    target_60 = next(
        (c for c in ("target_ender_60", "target_cyrusd_60", "target_60", "target")
         if c in schema_cols),
        target_20,
    )
    return target_20, target_60


def _load_v2_lookups(data_dir: Path, tier4_column: str) -> _V2Lookups | None:
    """Single-pass v2 lookups: deduped targets, meta, benchmarks on meta eras.

    Returns None when validation.parquet or meta_model.parquet is missing;
    benchmarks are optional (empty frame when the file is absent).
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
    benchmarks = pl.DataFrame(
        schema={"era": pl.String, "id": pl.String, tier4_column: pl.Float64}
    )
    if bench_path.exists():
        benchmarks = pl.read_parquet(
            bench_path, columns=["era", "id", tier4_column]
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
        row for row in rows
        if row["source"] in ("trained", "trained_legacy")
        and not _has_stored_capital_block(row)
    ]
    if not needs_recompute:
        return leaderboard

    lookups = _load_shared_lookups(data_dir)
    if lookups is None:
        logger.warning(
            "nmr.dashboard: v5.3 targets/meta_model missing at %s; "
            "capital cells left None", data_dir,
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
                "capital cells left None", row["model_id"],
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


def _series_label(registry_dir: Path, run_id: str) -> str:
    run_file = Path(registry_dir) / run_id / "run.json"
    name = "unknown"
    if run_file.exists():
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
        if isinstance(payload, dict):
            manifest = payload.get("manifest") or {}
            name = (manifest.get("config") or {}).get("run", {}).get("name", "unknown")
    return f"{name} · {run_id[:8]}"


def _series_from_metrics(
    corr: dict[str, float], mmc: dict[str, float], axis_eras: list[str], label: str
) -> dict:
    """Aligned wealth/drawdown arrays over ``axis_eras`` (fail-loud on gaps)."""
    missing = set(axis_eras) - set(corr) - set(mmc)
    if set(corr) != set(axis_eras) or set(mmc) != set(axis_eras) or missing:
        raise ValueError(
            f"series {label!r} does not cover the full era axis "
            f"(missing {sorted(missing, key=int)[:5]}...)"
        )
    # Order dicts by the axis explicitly: wealth compounding and drawdown
    # watermarks are sequence-sensitive (defense in depth — payout_series
    # sorts numerically, but this keeps the invariant local).
    ordered_corr = {era: float(corr[era]) for era in axis_eras}
    ordered_mmc = {era: float(mmc[era]) for era in axis_eras}
    pay = payout_series(ordered_corr, ordered_mmc)
    wealth = np.cumprod(1.0 + pay.clipped)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return {
        "label": label,
        "cumulative_wealth": [float(v) for v in wealth],
        "drawdown": [float(v) for v in drawdown],
        "cagr": float(annual_compounded_return(pay.clipped)),
        "mdd": float(np.min(drawdown)),
    }


def extract_payout_timeseries(
    registry_dir: Path,
    data_dir: Path,
    run_ids: Sequence[str],
    include_tier4_ref: bool = True,
    tier4_column: str = "v53_lgbm_ender60",
) -> dict[str, Any]:
    """Per-era payout/wealth/drawdown series over the standard 86-era window.

    Numeric era ordering throughout (``sorted_era_labels``); arrays aligned
    to the shared axis; deterministic key order (sorted run ids).
    """
    lookups = _load_shared_lookups(data_dir)
    if lookups is None:
        logger.warning(
            "nmr.dashboard: data assets missing at %s; "
            "returning empty timeseries", data_dir,
        )
        return {"eras": [], "meta_downside_mask": [], "series": {}}
    targets_86, meta, meta_eras = lookups
    axis = sorted_era_labels(meta_eras)

    # meta-model downside mask (strict CORR_meta < 0), computed once
    meta_only = meta.filter(pl.col("era").is_in(axis))
    meta_joined = meta_only.join(targets_86, on=["era", "id"], how="inner")
    meta_corr = EvaluationEngine().per_era_corr(
        meta_joined, pred_col="numerai_meta_model", target_col="target"
    )
    mask = [bool(meta_corr[e] < 0.0) for e in axis]

    series: dict[str, dict] = {}
    for run_id in sorted(set(run_ids)):
        preds_path = Path(registry_dir) / run_id / "validation_preds.parquet"
        if not preds_path.exists():
            logger.warning("nmr.dashboard: skipping missing preds %s", preds_path)
            continue
        corr, mmc, _ = _per_era_metrics(preds_path, targets_86, meta)
        series[run_id] = _series_from_metrics(
            corr, mmc, axis, _series_label(registry_dir, run_id)
        )

    if include_tier4_ref:
        bench_path = Path(data_dir) / "validation_benchmark_models.parquet"
        if bench_path.exists():
            bench = pl.read_parquet(
                bench_path, columns=["era", "id", tier4_column]
            ).filter(pl.col("era").is_in(axis))
            ref_joined = (
                bench.join(targets_86, on=["era", "id"], how="inner")
                .join(meta, on=["era", "id"], how="inner")
            )
            engine = EvaluationEngine()
            ref_corr = engine.per_era_corr(
                ref_joined, pred_col=tier4_column, target_col="target"
            )
            ref_mmc = engine.per_era_mmc(
                ref_joined, pred_col=tier4_column,
                meta_col="numerai_meta_model", target_col="target",
            )
            series[tier4_column] = _series_from_metrics(
                ref_corr, ref_mmc, axis, tier4_column
            )
        else:
            logger.warning("nmr.dashboard: %s missing; tier-4 curve omitted", bench_path)

    return {"eras": axis, "meta_downside_mask": mask, "series": series}
