"""Pure data-shaping helpers for the Streamlit+Plotly dashboard (Task 2).

Thin control plane only: column selection, rename, join, and boolean-flag
extraction. No metric formulas, no transforms, no registry writes. The only
computation in the app (``fleet_summary``) lives in ``nmr/meta.py`` and is
consumed by the render layer in Task 3.

Mirrors the column semantics of ``generate_dashboard._load_registry_runs`` /
``_load_benchmarks`` (which returned pandas DataFrames) but returns Polars
DataFrames and adds scorecard-CI / robustness columns.

Critical semantic: explicit ``None`` checks — a legitimate scorecard value of
``0.0`` must NOT fall through to the legacy ``metrics`` fallback. See the same
trap documented in ``generate_dashboard.py`` lines 43-50.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

_LEADERBOARD_SCHEMA = pl.Schema(
    {
        "model_id": pl.String,
        "source": pl.String,
        "run_name": pl.String,
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
        "corr_sharpe_ac": pl.Float64,
        "corr_sharpe_ac_ci_low": pl.Float64,
        "corr_sharpe_ac_ci_high": pl.Float64,
        "max_drawdown": pl.Float64,
        "std_corr": pl.Float64,
        "deflated_sharpe": pl.Float64,
        "max_feature_exposure": pl.Float64,
        "has_bmc": pl.Boolean,
        "has_horizon": pl.Boolean,
        "has_perturb": pl.Boolean,
        "has_regime": pl.Boolean,
        "run_dir": pl.String,
    }
)

_CAMPAIGN_SCHEMA = pl.Schema(
    {
        "campaign_id": pl.String,
        "name": pl.String,
        "config_path": pl.String,
        "run_id": pl.String,
        "status": pl.String,
        "error": pl.String,
    }
)

# Scorecard cells that drive the has_* robustness flags. A flag is True when
# the cell is present in the scorecard block (present means not None).
_ROBUSTNESS_CELLS = {
    "has_bmc": "bmc",
    "has_horizon": "horizon_model_sharpe_20",
    "has_perturb": "perturb_ceiling_stability",
    "has_regime": "regime_count",
}

_EMPTY_LEADERBOARD = pl.DataFrame(schema=_LEADERBOARD_SCHEMA)
_EMPTY_CAMPAIGNS = pl.DataFrame(schema=_CAMPAIGN_SCHEMA)


def load_registry_frame(registry_dir: Path) -> pl.DataFrame:
    """Load all registry runs into a leaderboard frame.

    Runs with a scorecard block are ``source="trained"`` and read their
    metrics from the scorecard cells (including CI bounds and robustness
    cells). Runs without a scorecard are ``source="trained_legacy"`` and fall
    back to the train-OOF ``metrics`` for corr / sharpe / drawdown — mirroring
    ``generate_dashboard._load_registry_runs``. Explicit ``None`` checks mean a
    legitimate scorecard ``0.0`` never falls through to the legacy metric.
    """
    rows: list[dict] = []
    for run_file in sorted(registry_dir.glob("*/run.json")):
        payload = json.loads(run_file.read_text(encoding="utf-8"))
        metrics = payload.get("metrics") or {}
        manifest = payload.get("manifest") or {}
        cfg = manifest.get("config") or {}
        data_cfg = cfg.get("data") or {}
        model_cfg = cfg.get("model") or {}
        run_cfg = cfg.get("run") or {}
        risk_cfg = cfg.get("risk") or {}

        # Explicit None checks: a legitimate scorecard 0.0 must NOT fall
        # through to the legacy OOF metric (same trap as generate_dashboard).
        scorecard = payload.get("scorecard") or {}
        sc_corr = scorecard.get("corr")
        sc_sharpe = scorecard.get("corr_sharpe_ac")
        sc_std = scorecard.get("std_corr")
        sc_dd = scorecard.get("max_drawdown")

        flags = {
            flag: scorecard.get(cell) is not None
            for flag, cell in _ROBUSTNESS_CELLS.items()
        }
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
                "corr_sharpe_ac": float(
                    sc_sharpe if sc_sharpe is not None else metrics.get("sharpe", 0.0)
                ),
                "corr_sharpe_ac_ci_low": scorecard.get("corr_sharpe_ac_ci_low"),
                "corr_sharpe_ac_ci_high": scorecard.get("corr_sharpe_ac_ci_high"),
                "max_drawdown": float(
                    sc_dd if sc_dd is not None else metrics.get("max_drawdown", 0.0)
                ),
                "std_corr": float(sc_std if sc_std is not None else metrics.get("std", 0.0)),
                "deflated_sharpe": scorecard.get("deflated_sharpe"),
                "max_feature_exposure": scorecard.get("max_feature_exposure"),
                **flags,
                "run_dir": str(run_file.parent),
            }
        )
    if not rows:
        return _EMPTY_LEADERBOARD
    return pl.DataFrame(rows, schema=_LEADERBOARD_SCHEMA, strict=False)


def load_benchmarks(path: Path) -> pl.DataFrame:
    """Normalize the benchmark CSV to the leaderboard schema.

    Mirrors ``generate_dashboard._load_benchmarks`` column semantics:
    ``strategy_group`` -> ``run_name``, ``horizon_target_name`` -> ``targets``,
    ``corr`` / ``corr_sharpe_ac`` / ``std_corr`` / ``max_drawdown`` keep their
    names, everything else is ``source="benchmark"``. A missing file yields an
    empty frame.
    """
    if not path.exists():
        return _EMPTY_LEADERBOARD

    df = pl.read_csv(path)
    if df.height == 0:
        return _EMPTY_LEADERBOARD

    rows = [
        {
            "model_id": row["model_id"],
            "source": "benchmark",
            "run_name": row.get("strategy_group", "benchmark"),
            "backend": "benchmark",
            "preset": "benchmark",
            "feature_set": "all",
            "feature_subset": None,
            "n_targets": 1,
            "targets": row.get("horizon_target_name", "cyrusd"),
            "neutralization_proportion": None,
            "oof_device": None,
            "corr": row.get("corr"),
            "corr_ci_low": None,
            "corr_ci_high": None,
            "corr_sharpe_ac": row.get("corr_sharpe_ac"),
            "corr_sharpe_ac_ci_low": None,
            "corr_sharpe_ac_ci_high": None,
            "max_drawdown": row.get("max_drawdown", 0.0),
            "std_corr": row.get("std_corr", 0.0),
            "deflated_sharpe": None,
            "max_feature_exposure": None,
            "has_bmc": False,
            "has_horizon": False,
            "has_perturb": False,
            "has_regime": False,
            "run_dir": str(path),
        }
        for row in df.to_dicts()
    ]
    return pl.DataFrame(rows, schema=_LEADERBOARD_SCHEMA, strict=False)


def merge_leaderboard(registry: pl.DataFrame, benchmarks: pl.DataFrame) -> pl.DataFrame:
    """Row-concat registry runs and benchmark rows into one leaderboard frame."""
    frames = [f for f in (registry, benchmarks) if f is not None and f.height > 0]
    if not frames:
        return _EMPTY_LEADERBOARD
    return pl.concat(frames, how="vertical")


def load_campaigns(campaigns_dir: Path) -> pl.DataFrame:
    """Flatten each campaign ``*.json`` log to one row per run.

    Each run row carries ``campaign_id, name, config_path, run_id, status,
    error``. A missing directory yields an empty frame.
    """
    if not campaigns_dir.is_dir():
        return _EMPTY_CAMPAIGNS

    rows: list[dict] = []
    for log_file in sorted(campaigns_dir.glob("*.json")):
        payload = json.loads(log_file.read_text(encoding="utf-8"))
        campaign_id = payload.get("campaign_id") or log_file.stem
        name = payload.get("name")
        for run in payload.get("runs", []):
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "name": name,
                    "config_path": run.get("config_path"),
                    "run_id": run.get("run_id"),
                    "status": run.get("status"),
                    "error": run.get("error"),
                }
            )
    if not rows:
        return _EMPTY_CAMPAIGNS
    return pl.DataFrame(rows, schema=_CAMPAIGN_SCHEMA, strict=False)


def robustness_matrix(registry: pl.DataFrame) -> pl.DataFrame:
    """Project the robustness cells of trained runs (numeric casts for heatmap)."""
    columns = [
        "model_id",
        "has_bmc",
        "has_horizon",
        "has_perturb",
        "has_regime",
        "max_feature_exposure",
        "std_corr",
        "max_drawdown",
    ]
    casts = {
        "has_bmc": pl.Boolean,
        "has_horizon": pl.Boolean,
        "has_perturb": pl.Boolean,
        "has_regime": pl.Boolean,
        "max_feature_exposure": pl.Float64,
        "std_corr": pl.Float64,
        "max_drawdown": pl.Float64,
    }
    frame = registry.select(columns)
    return frame.cast(casts)


def champion_run_id(registry_dir: Path) -> str | None:
    """Read the champion pointer; missing or corrupt champion.json -> None."""
    champion_path = registry_dir / "champion.json"
    if not champion_path.exists():
        return None
    try:
        payload = json.loads(champion_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    run_id = payload.get("run_id")
    return run_id if isinstance(run_id, str) else None


def main() -> int:
    """Entry point — the render layer lands in Task 3."""
    print("dashboard_app: data-shaping helpers only; rendering lands in Task 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
