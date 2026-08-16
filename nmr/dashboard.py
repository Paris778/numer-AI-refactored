# ruff: noqa: F822
# __all__ documents the module's final public surface; the remaining names are
# implemented in later dashboard tasks (Task 2+ of the executive-dashboard plan).
"""Pure analytical engine for the executive performance dashboard.

Registry scans, benchmark reconciliation, gate projection, capital-cell
recompute, and payout timeseries extraction. Plotly/Streamlit-free; every
function here is covered by tests/test_dashboard.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from nmr.config import REPO_ROOT

logger = logging.getLogger("nmr.dashboard")

__all__ = [
    "UNIFIED_SCHEMA",
    "evaluate_gate_status",
    "extract_payout_timeseries",
    "load_benchmark_frame",
    "load_unified_leaderboard",
    "reconcile_capital_metrics",
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
