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
from nmr.ensemble import Ensembler
from nmr.evaluation import EvaluationEngine, downside_era_indices, sorted_era_labels
from nmr.families import DEFAULT_MODELS_DIR, scan_full_versions
from nmr.payout import (
    annual_compounded_return,
    gain_to_pain_ratio,
    kelly_fraction,
    payout_series,
)
from nmr.scorecard import MMC_DOWN_MIN_ERAS

logger = logging.getLogger("nmr.dashboard")

__all__ = [
    "EVALUABLE_ROWS",
    "UNIFIED_SCHEMA",
    "evaluate_gate_status",
    "extract_multimetric_timeseries",
    "extract_pairwise_similarity_matrix",
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
        "family": pl.String, "training_scope": pl.String, "has_full_version": pl.Boolean,
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

# Single predicate for every chart / candidate-selection path: rows that carry
# validation metrics. Source-based (never null) so benchmark rows (null
# training_scope) stay visible in charts; full rows (in-sample metrics) are
# excluded everywhere.
EVALUABLE_ROWS: pl.Expr = pl.col("source") != "full"


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
    models_dir: Path | None = None,
) -> pl.DataFrame:
    """Load registry runs and (optionally) benchmark rows into one frame.

    Explicit-None discipline: a scorecard value of 0.0 is real and must not
    fall through to the legacy train-OOF ``metrics``. Corrupt ``run.json``
    files are skipped. ``benchmark_path=False`` disables benchmark loading
    (registry-only); otherwise a missing path falls through the resolution
    chain.
    """
    rows: list[dict] = []
    full_versions = scan_full_versions(
        Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    )
    promoted_families = set(full_versions)
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
                "family": run_cfg.get("name", "unknown"),
                "training_scope": "research",
                "has_full_version": run_cfg.get("name", "unknown") in promoted_families,
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

    for family in sorted(full_versions):
        version = full_versions[family]
        if not (Path(registry_dir) / version.promoted_from_run_id).is_dir():
            logger.warning(
                "nmr.dashboard: full version %s lineage dangling "
                "(promoted_from_run_id %s not in registry)",
                family, version.promoted_from_run_id,
            )
        full_row = dict.fromkeys(UNIFIED_SCHEMA.names())  # all metric cells null
        cfg_data = (version.config.get("data") or {}) if version.config else {}
        cfg_model = (version.config.get("model") or {}) if version.config else {}
        targets = cfg_data.get("targets") or []
        full_row.update(
            {
                "model_id": f"{family}::full",
                "source": "full",
                "run_name": family,
                "family": family,
                "training_scope": "full",
                "has_full_version": False,
                "backend": cfg_model.get("backend"),
                "preset": cfg_model.get("preset"),
                "feature_set": cfg_data.get("feature_set"),
                "feature_subset": cfg_data.get("feature_subset"),
                "n_targets": len(targets) if targets else None,
                "targets": ", ".join(targets) if targets else None,
                "run_dir": str(version.manifest_path.parent),
            }
        )
        rows.append(full_row)

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
        elif row["source"] == "benchmark":
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
        try:
            benchmarks = pl.read_parquet(
                bench_path, columns=["era", "id", tier4_column]
            ).filter(pl.col("era").is_in(meta_eras))
        except pl.exceptions.ColumnNotFoundError:
            # schema drift: the parquet exists but lacks the tier-4 column —
            # degrade to the empty benchmark frame instead of raising
            logger.warning(
                "nmr.dashboard: benchmark parquet %s lacks tier-4 column %s; "
                "benchmarks empty", bench_path, tier4_column,
            )
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


_METRIC_NAMES = ("payout", "corr20", "mmc20", "corr60", "mmc60", "bmc", "cwmm")


def _cumulative_from_standard(standard: list[float], *, payout: bool) -> list[float]:
    values = np.asarray(standard, dtype=float)
    if payout:
        return [float(v) for v in np.cumprod(1.0 + values)]
    return [float(v) for v in np.cumsum(values)]


def extract_multimetric_timeseries(
    registry_dir: Path,
    data_dir: Path,
    run_ids: Sequence[str],
    include_tier4_ref: bool = True,
    tier4_column: str = "v53_lgbm_ender60",
) -> dict[str, Any]:
    """7-metric per-era trajectories over the standardized meta window.

    Payout is anchored to main_target="target" (decision #19); correlation
    metrics use cumsum, payout uses cumprod (decision #9); the tier-4 BMC is
    short-circuited to zeros (decision #11); an absent benchmark frame or a
    model sharing no era with the meta window zero-fills its bmc/cwmm slices
    and skips its payout slice with warnings (decision #23). Never raises on
    missing assets — returns the empty payload.
    """
    lookups = _load_v2_lookups(data_dir, tier4_column)
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
    drawdowns: dict[str, list[float]] = {}

    ids = [mid for mid in sorted(set(run_ids)) if mid != tier4_column]
    if include_tier4_ref and lookups.benchmarks.height > 0:
        ids.append(tier4_column)

    for model_id in ids:
        if model_id == tier4_column:
            preds = lookups.benchmarks.select(
                ["era", "id", pl.col(tier4_column).alias("prediction")]
            )
            label = tier4_column
        else:
            preds_path = Path(registry_dir) / model_id / "validation_preds.parquet"
            if not preds_path.exists():
                logger.warning(
                    "nmr.dashboard: skipping missing preds %s", preds_path
                )
                continue
            preds = pl.read_parquet(preds_path, columns=["era", "id", "prediction"])
            label = _series_label(registry_dir, model_id)

        joined = (
            preds.join(lookups.targets, on=["era", "id"], how="inner")
            .join(lookups.meta, on=["era", "id"], how="inner")
        )
        corr_t = engine.per_era_corr(joined, pred_col="prediction", target_col="target")
        mmc_t = engine.per_era_mmc(
            joined, pred_col="prediction",
            meta_col="numerai_meta_model", target_col="target",
        )
        # decision #23: a stale run whose preds share no era with the meta
        # window must not abort the report — skip its payout slice (drawdowns
        # derive from payout wealth) while the other metric slices still render
        if set(corr_t) & set(mmc_t):
            pay = payout_series(corr_t, mmc_t)
            clipped_by_era = dict(zip(pay.eras, pay.clipped))
            standard = [float(clipped_by_era.get(era, 0.0)) for era in axis]
            metrics["payout"][model_id] = {
                "standard": standard,
                "cumulative": _cumulative_from_standard(standard, payout=True),
                "label": label,
            }
            wealth = np.asarray(metrics["payout"][model_id]["cumulative"], dtype=float)
            peak = np.maximum.accumulate(wealth)
            drawdowns[model_id] = [float(v) for v in wealth / peak - 1.0]
        else:
            logger.warning(
                "nmr.dashboard: %s shares no eras with the meta window; "
                "payout slice skipped", model_id,
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
                    joined, pred_col="prediction",
                    meta_col="numerai_meta_model", target_col=target_col,
                )
            aligned = [float(per.get(era, 0.0)) for era in axis]
            metrics[name][model_id] = {
                "standard": aligned,
                "cumulative": _cumulative_from_standard(aligned, payout=False),
                "label": label,
            }

        if model_id == tier4_column:
            zeros = [0.0 for _ in axis]
            metrics["bmc"][model_id] = {
                "standard": zeros, "cumulative": zeros, "label": label,
            }
        elif lookups.benchmarks.height > 0 and joined.height > 0:
            joined_b = joined.join(lookups.benchmarks, on=["era", "id"], how="inner")
            # reporting path relaxes the evaluation vacuity gate (real meta
            # window satisfies 20 anyway); alignment below zero-fills missing eras
            per_bmc = engine.per_era_bmc(
                joined_b, pred_col="prediction",
                benchmark_col=tier4_column, target_col="target",
                min_overlap_eras=1,
            )
            aligned = [float(per_bmc.get(era, 0.0)) for era in axis]
            metrics["bmc"][model_id] = {
                "standard": aligned,
                "cumulative": _cumulative_from_standard(aligned, payout=False),
                "label": label,
            }
        else:
            # decision #23: an absent benchmark frame (or a model with no era
            # overlap with it) zero-fills the BMC slice instead of raising
            if lookups.benchmarks.height == 0:
                logger.warning(
                    "nmr.dashboard: benchmark models absent at %s; bmc zeroed",
                    data_dir,
                )
            else:
                logger.warning(
                    "nmr.dashboard: %s shares no eras with the benchmark "
                    "window; bmc zeroed", model_id,
                )
            zeros = [0.0 for _ in axis]
            metrics["bmc"][model_id] = {
                "standard": zeros, "cumulative": zeros, "label": label,
            }

        if joined.height > 0:
            # reporting path relaxes the evaluation vacuity gate (real meta window
            # satisfies 20 anyway); alignment below zero-fills missing eras
            per_cwmm = engine.per_era_cwmm(
                joined, pred_col="prediction", meta_col="numerai_meta_model",
                min_overlap_eras=1,
            )
            aligned = [float(per_cwmm.get(era, 0.0)) for era in axis]
            metrics["cwmm"][model_id] = {
                "standard": aligned,
                "cumulative": _cumulative_from_standard(aligned, payout=False),
                "label": label,
            }
        else:
            # zero-overlap model: cwmm renders zero-filled (decision #23)
            zeros = [0.0 for _ in axis]
            metrics["cwmm"][model_id] = {
                "standard": zeros, "cumulative": zeros, "label": label,
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
    lookups = _load_v2_lookups(data_dir, tier4_column)
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
        pred_col="numerai_meta_model", target_col="target",
    )
    stress_eras = {era for era in axis if meta_corr.get(era, 0.0) < 0.0}
    era_arr = gauss.get_column("era").to_list()
    stress_idx = np.asarray([era in stress_eras for era in era_arr])

    def _mean_offdiag(mat: np.ndarray) -> float | None:
        mat = np.atleast_2d(mat)
        if mat.shape[0] < 2:
            return None
        upper = [mat[i, j] for i in range(mat.shape[0]) for j in range(i + 1, mat.shape[0])]
        return float(np.mean(upper)) if upper else None

    mean_delta = None
    if stress_idx.sum() >= 5:
        # degenerate columns inside the stress/normal subsets produce NaN
        # correlations — neutralize them (0) before the off-diagonal mean so
        # mean_delta is always finite (0-d scalar corrcoef kept 2-d too)
        rho_stress = _mean_offdiag(
            np.clip(np.nan_to_num(np.atleast_2d(np.corrcoef(stacked[:, stress_idx])), nan=0.0), -1.0, 1.0)
        )
        rho_normal = _mean_offdiag(
            np.clip(np.nan_to_num(np.atleast_2d(np.corrcoef(stacked[:, ~stress_idx])), nan=0.0), -1.0, 1.0)
        )
        if rho_stress is not None and rho_normal is not None:
            mean_delta = rho_stress - rho_normal

    n_pairs = len(ids_used) * (len(ids_used) - 1) // 2
    return labels, ids_used, matrix.tolist(), {"mean_delta": mean_delta, "n_pairs": n_pairs}
