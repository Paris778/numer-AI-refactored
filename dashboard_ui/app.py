"""Streamlit host for the shared vanilla Model Tournament renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st
import streamlit.components.v1 as components

from dashboard_ui.report import build_dashboard_html
from nmr import paths
from nmr.config import REPO_ROOT
from nmr.dashboard import (
    EVALUABLE_ROWS,
    UNIFIED_SCHEMA,
    load_benchmark_frame,
    load_unified_leaderboard,
    read_champion_pointer,
)

_LEADERBOARD_SCHEMA = UNIFIED_SCHEMA
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
_ROBUSTNESS_CELLS = {
    "has_bmc": "bmc",
    "has_horizon": "horizon_model_sharpe_20",
    "has_perturb": "perturb_ceiling_stability",
    "has_regime": "regime_count",
}
_DEFAULT_REGISTRY_DIR = paths.EXPERIMENTS_ROOT
_DEFAULT_BENCHMARK_PATH = (
    REPO_ROOT / "artifacts" / "reports" / "benchmark_hierarchy_scorecard.csv"
)
_DEFAULT_DATA_DIR = REPO_ROOT / "data" / "v5.3"

# Environment-configurable roots (deployment: point the dashboard at a
# different experiments root / benchmark report / data version without
# editing code). Defaults match the offline report compiler.
_ENV_REGISTRY = "NMR_DASH_REGISTRY_DIR"
_ENV_BENCHMARK = "NMR_DASH_BENCHMARK_PATH"
_ENV_DATA = "NMR_DASH_DATA_DIR"


def _env_path(env_name: str, default: Path) -> Path:
    import os

    value = os.environ.get(env_name)
    return Path(value) if value else default


def load_registry_frame(
    registry_dir: Path, models_dir: Path | None = None
) -> pl.DataFrame:
    """Compatibility projection of registry rows onto the unified schema."""
    frame = load_unified_leaderboard(
        Path(registry_dir), benchmark_path=False, models_dir=models_dir
    )
    return frame.filter(
        pl.col("source").is_in(["trained", "trained_legacy", "full", "partial"])
    )


def load_benchmarks(path: Path) -> pl.DataFrame:
    """Compatibility loader for benchmark rows."""
    return load_benchmark_frame(Path(path))


def merge_leaderboard(registry: pl.DataFrame, benchmarks: pl.DataFrame) -> pl.DataFrame:
    """Concatenate registry and benchmark frames without changing their rows."""
    frames = [
        frame for frame in (registry, benchmarks) if frame is not None and frame.height
    ]
    return (
        pl.concat(frames, how="vertical")
        if frames
        else pl.DataFrame(schema=UNIFIED_SCHEMA)
    )


def load_campaigns(campaigns_dir: Path) -> pl.DataFrame:
    """Flatten campaign logs for callers of the former helper API."""
    rows: list[dict] = []
    directory = Path(campaigns_dir)
    if not directory.is_dir():
        return pl.DataFrame(schema=_CAMPAIGN_SCHEMA)
    for log_file in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(log_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        campaign_id = payload.get("campaign_id") or log_file.stem
        for run in payload.get("runs") or []:
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "name": payload.get("name"),
                    "config_path": run.get("config_path"),
                    "run_id": run.get("run_id"),
                    "status": run.get("status"),
                    "error": run.get("error"),
                }
            )
    return (
        pl.DataFrame(rows, schema=_CAMPAIGN_SCHEMA, strict=False)
        if rows
        else pl.DataFrame(schema=_CAMPAIGN_SCHEMA)
    )


def robustness_matrix(registry: pl.DataFrame) -> pl.DataFrame:
    """Project robustness fields from rows that can be compared out of sample."""
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
    frame = (
        registry.filter(EVALUABLE_ROWS) if "source" in registry.columns else registry
    )
    return frame.select(columns)


def champion_run_id(registry_dir: Path) -> str | None:
    """Read the opaque champion pointer without writing registry state."""
    return read_champion_pointer(Path(registry_dir) / "champion.json")


def _bar_label(source: str, run_name: str, model_id: str | None, display_name=None) -> str:
    """Format the former chart label helper for compatibility callers."""
    identifier = model_id or "?"
    label = display_name or run_name
    return (
        f"{label} · {identifier}"
        if source == "benchmark"
        else f"{label} · {identifier[:8]}"
    )


def _shaped_leaderboard_pdf(
    leaderboard: pl.DataFrame, champion: str | None
) -> pd.DataFrame:
    """Return a deterministic pandas projection for legacy consumers."""
    frame = leaderboard.with_columns(pl.col("source").eq("full").alias("_is_full"))
    pdf = frame.sort(
        ["_is_full", "corr_sharpe_ac"],
        descending=[True, True],
        nulls_last=[False, True],
    ).to_pandas()
    pdf["champion"] = pdf["model_id"] == champion
    pdf["ci_plus"] = pdf["corr_sharpe_ac_ci_high"] - pdf["corr_sharpe_ac"]
    pdf["ci_minus"] = pdf["corr_sharpe_ac"] - pdf["corr_sharpe_ac_ci_low"]
    pdf["label"] = [
        _bar_label(source, run_name, model_id, display_name)
        for source, run_name, model_id, display_name in zip(
            pdf["source"],
            pdf["run_name"],
            pdf["model_id"],
            pdf.get("display_name", [None] * len(pdf)),
        )
    ]
    return pdf.drop(columns=["_is_full"])


def render_tournament() -> None:
    """Embed the exact HTML document generated by the offline report compiler.

    Every rerun re-reads the local artifacts (the host keeps no long-lived
    cache); the fingerprint-based refresh lives on the typed data service and
    is exposed here as an explicit sidebar action for programmatic callers.
    """
    registry_dir = _env_path(_ENV_REGISTRY, _DEFAULT_REGISTRY_DIR)
    benchmark_path = _env_path(_ENV_BENCHMARK, _DEFAULT_BENCHMARK_PATH)
    data_dir = _env_path(_ENV_DATA, _DEFAULT_DATA_DIR)

    from dashboard_ui.service import DashboardDataService

    # Retain ONE service instance for the session: the sidebar Refresh action
    # clears ITS caches (the refreshable layer for every dashboard host). The
    # rendered page is rebuilt fresh on each rerun via the offline compiler,
    # so the visible report is always current after a rerun.
    if "dash_service" not in st.session_state:
        st.session_state["dash_service"] = DashboardDataService(
            registry_dir=registry_dir,
            benchmark_path=benchmark_path,
            data_dir=data_dir,
        )
    service = st.session_state["dash_service"]

    with st.sidebar:
        st.markdown("### Model Tournament")
        st.caption("Read-only dashboard over local artifacts. Never calls Numerai.")
        if st.button("Refresh data", use_container_width=True):
            service.refresh()
            st.rerun()
        st.markdown("#### Sources")
        st.code(
            f"registry: {registry_dir}\n"
            f"benchmark: {benchmark_path}\n"
            f"data: {data_dir}",
            language="text",
        )
        st.markdown("#### Snapshot")
        st.caption(f"source fingerprint: {service.compute_source_fingerprint()[:12]}…")
        st.caption("caches: leaderboard + timeseries + full-history")

    html_text = build_dashboard_html(
        registry_dir=registry_dir,
        benchmark_path=benchmark_path,
        data_dir=data_dir,
    )
    components.html(html_text, height=3000, scrolling=True)


def main() -> None:
    """Streamlit entry point; the browser UI has one renderer."""
    st.set_page_config(page_title="Numerai Model Tournament", layout="wide")
    render_tournament()


if __name__ == "__main__":
    main()
