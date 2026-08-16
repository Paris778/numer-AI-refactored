"""Pure analytical engine for the executive performance dashboard.

Registry scans, benchmark reconciliation, gate projection, capital-cell
recompute, and payout timeseries extraction. Plotly/Streamlit-free; every
function here is covered by tests/test_dashboard.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

from nmr.config import REPO_ROOT

logger = logging.getLogger("nmr.dashboard")

__all__ = [
    "UNIFIED_SCHEMA",
    "load_benchmark_frame",
    "load_unified_leaderboard",
    "resolve_benchmark_path",
]

REPORTS_DIR = REPO_ROOT / "artifacts" / "reports"
LEGACY_BENCHMARK_PATH = REPO_ROOT / "artifacts" / "benchmark_scores.csv"
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
    legacy_path: Path | None = None,
) -> Path | None:
    """Resolve the benchmark scorecard CSV via the fallback chain.

    Chain: given path (if it exists) -> full hierarchy CSV -> smoke CSV ->
    legacy CSV -> None. ``benchmark_path=False`` is an explicit directive to
    disable benchmark loading entirely (test isolation).
    """
    if benchmark_path is False:
        return None
    if benchmark_path is not None:
        given = Path(benchmark_path)
        if given.exists():
            return given
    reports = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    legacy = Path(legacy_path) if legacy_path is not None else LEGACY_BENCHMARK_PATH
    for candidate in (
        reports / "benchmark_hierarchy_scorecard.csv",
        reports / "benchmark_hierarchy_scorecard_smoke.csv",
        legacy,
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
