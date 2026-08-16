"""Pure data-shaping helpers + thin Streamlit render layer for the dashboard.

Thin control plane only: column selection, rename, join, boolean-flag
extraction, and rendering frames through Streamlit/Plotly. No metric
formulas, no transforms, no registry writes. The only computation in the app
(``fleet_summary``) lives in ``nmr/meta.py`` and is consumed by the render
layer below.

Data loading delegates to the shared engine ``nmr.dashboard``
(``load_unified_leaderboard`` / ``load_benchmark_frame``); this module
projects the engine's unified frame down to ``_LEADERBOARD_SCHEMA`` for the
Streamlit views and adds the render layer.

Critical semantic: explicit ``None`` checks — a legitimate scorecard value of
``0.0`` must NOT fall through to the legacy ``metrics`` fallback. That
None-discipline lives in ``nmr.dashboard.load_unified_leaderboard``.

Render layer: all ``st.*`` calls live inside ``main()`` or the five view
functions, so importing this module is side-effect free (streamlit/plotly
import headless; no server is launched).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st

from nmr.config import REPO_ROOT
from nmr.dashboard import (
    load_benchmark_frame,
    load_unified_leaderboard,
    resolve_benchmark_path,
)
from nmr.meta import fleet_summary

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
    """Load all registry runs into a leaderboard frame (engine delegation).

    Projects the engine's unified frame down to ``_LEADERBOARD_SCHEMA`` for
    the Streamlit views; parsing and None-discipline live in
    ``nmr.dashboard.load_unified_leaderboard``.
    """
    frame = load_unified_leaderboard(registry_dir, benchmark_path=False)
    trained = frame.filter(pl.col("source").is_in(["trained", "trained_legacy"]))
    if trained.height == 0:
        return _EMPTY_LEADERBOARD
    return trained.select(_LEADERBOARD_SCHEMA.names())


def load_benchmarks(path: Path) -> pl.DataFrame:
    """Normalize the benchmark CSV to the leaderboard schema (engine delegation)."""
    frame = load_benchmark_frame(path)
    if frame.height == 0:
        return _EMPTY_LEADERBOARD
    return frame.select(_LEADERBOARD_SCHEMA.names())


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
        try:
            payload = json.loads(log_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # corrupt campaign log degrades gracefully (skipped)
        campaign_id = payload.get("campaign_id") or log_file.stem
        name = payload.get("name")
        for run in payload.get("runs") or []:
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


# ---------------------------------------------------------------------------
# Streamlit render layer — thin, read-only (Task 3)
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY_DIR = REPO_ROOT / "artifacts" / "registry"
_DEFAULT_BENCHMARK_PATH = (
    REPO_ROOT / "artifacts" / "reports" / "benchmark_hierarchy_scorecard.csv"
)
_DEFAULT_CAMPAIGNS_DIR = REPO_ROOT / "artifacts" / "campaigns"

_BAR_METRIC = "corr_sharpe_ac"


def _read_run_payload(run_dir: Path) -> dict | None:
    """Read a registry ``run.json``; ``None`` when ``run_dir`` is not a run dir.

    Benchmark rows carry the benchmark CSV path as ``run_dir``, so no
    ``run.json`` exists there — callers fall back to the leaderboard row.
    Corrupt JSON degrades to ``None`` (same precedent as
    :func:`champion_run_id`).
    """
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return None
    try:
        payload = json.loads(run_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_registry_entries(registry_dir: Path) -> list[dict]:
    """Raw registry run payloads for ``nmr.meta.fleet_summary`` (read-only).

    Mirrors ``RunRegistry.list`` content but avoids the ``RunRegistry``
    constructor, which ``mkdir``s the root — the dashboard is strictly
    read-only. Order is irrelevant: ``fleet_summary`` re-sorts deterministically
    by metric desc / run_id tiebreak.
    """
    entries: list[dict] = []
    for run_file in sorted(registry_dir.glob("*/run.json")):
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and payload.get("run_id"):
            entries.append(payload)
    return entries


def _bar_label(source: str, run_name: str, model_id: str | None) -> str:
    """Unique, readable per-bar key for the leaderboard chart.

    Registry runs share ``run_name`` (their config name) across reruns of the
    same config, so ``run_name`` alone cannot key the bars — the real registry
    holds two ``first-competitive-lgbm-small`` runs that would draw overlapping
    bars at one x-tick. Trained runs key as ``run_name · short run_id``; the
    readable name stays in the label and in the hover data. Benchmark rows have
    no ``run_id`` — their ``model_id`` is the strategy name, already unique per
    row.
    """
    model_id = model_id or "?"
    if source == "benchmark":
        return f"{run_name} · {model_id}"
    return f"{run_name} · {model_id[:8]}"


def _shaped_leaderboard_pdf(
    leaderboard: pl.DataFrame, champion: str | None
) -> pd.DataFrame:
    """Sort + champion flag + CI error-bar deltas + unique bar labels (pure).

    Pure frame-shaping for :func:`render_leaderboard`, isolated so tests can
    exercise the unique-label logic without a Streamlit runtime. CI bounds are
    absolute (``corr_sharpe_ac_ci_low``/``_ci_high``); Plotly error bars need
    per-bar magnitudes, so the deltas are derived here. Rows without CI bounds
    get NaN deltas, which drop the error bars.
    """
    pdf = leaderboard.sort(_BAR_METRIC, descending=True, nulls_last=True).to_pandas()
    pdf["champion"] = pdf["model_id"] == champion
    pdf["ci_plus"] = pdf["corr_sharpe_ac_ci_high"] - pdf["corr_sharpe_ac"]
    pdf["ci_minus"] = pdf["corr_sharpe_ac"] - pdf["corr_sharpe_ac_ci_low"]
    pdf["label"] = [
        _bar_label(source, run_name, model_id)
        for source, run_name, model_id in zip(
            pdf["source"], pdf["run_name"], pdf["model_id"]
        )
    ]
    return pdf


def render_leaderboard(leaderboard: pl.DataFrame, champion: str | None) -> None:
    """Bar chart of ``corr_sharpe_ac`` with CI error bars + sortable dataframe.

    Bars key on a unique ``label`` (``run_name · short run_id`` for trained
    runs, ``run_name · model_id`` for benchmarks) so reruns of one config never
    overlap; the readable name stays in the label and hover data. The champion
    run (when present in the frame) is hatched via ``pattern_shape``.
    """
    if leaderboard.height == 0:
        st.info("No runs to display — train one with `train_first_model.py`.")
        return
    pdf = _shaped_leaderboard_pdf(leaderboard, champion)
    fig = px.bar(
        pdf,
        x="label",
        y=_BAR_METRIC,
        color="source",
        error_y="ci_plus",
        error_y_minus="ci_minus",
        pattern_shape="champion",
        pattern_shape_map={True: "/", False: ""},
        hover_data=["run_name", "model_id", "backend", "preset", "corr", "max_drawdown"],
        title="CORR Sharpe (auto-correlated)",
    )
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig)
    st.dataframe(pdf.drop(columns=["champion", "ci_plus", "ci_minus", "label"]))


def _render_run_manifest(manifest: dict) -> None:
    cfg = manifest.get("config") or {}
    data_cfg = cfg.get("data") or {}
    model_cfg = cfg.get("model") or {}
    risk_cfg = cfg.get("risk") or {}
    summary = {
        "oof_device": manifest.get("oof_device"),
        "weights": manifest.get("weights"),
        "feature_set": data_cfg.get("feature_set"),
        "feature_subset": data_cfg.get("feature_subset"),
        "targets": data_cfg.get("targets"),
        "backend": model_cfg.get("backend"),
        "preset": model_cfg.get("preset"),
        "neutralization_proportion": risk_cfg.get("neutralization_proportion"),
    }
    st.write("Summary", summary)
    st.json(manifest)


def render_run_detail(leaderboard: pl.DataFrame) -> None:
    """Per-run expander: scorecard cells, robustness flags, manifest summary.

    Trained runs re-read their raw ``run.json`` so every scorecard cell —
    value, CI bounds, and ``*_n_eras`` — is shown, not just the leaderboard
    projection. Benchmark rows (no ``run.json``) fall back to the leaderboard
    row.
    """
    for row in leaderboard.sort(_BAR_METRIC, descending=True, nulls_last=True).to_dicts():
        model_id = row["model_id"] or "?"  # null-safe: model_id may be None (benchmark rows)
        label = f"{model_id[:16]}… — {row['run_name']} ({row['source']})"
        with st.expander(label):
            payload = _read_run_payload(Path(row["run_dir"]))
            if payload is None:
                st.caption("Benchmark row / missing run.json — leaderboard row only.")
                st.dataframe(pl.DataFrame([row], strict=False))
                continue
            scorecard = payload.get("scorecard") or {}
            if scorecard:
                st.subheader("Scorecard")
                st.dataframe(
                    pl.DataFrame(
                        {
                            "cell": sorted(scorecard),
                            "value": [scorecard[key] for key in sorted(scorecard)],
                        },
                        strict=False,
                    )
                )
            st.subheader("Robustness")
            st.caption(
                "Coverage — "
                + ", ".join(f"{flag}={bool(row[flag])}" for flag in _ROBUSTNESS_CELLS)
            )
            st.subheader("Manifest")
            _render_run_manifest(payload.get("manifest") or {})


def render_fleet(
    registry_entries: Sequence[dict],
    *,
    n_trials: int,
    dsr_confidence: float,
) -> None:
    """Fleet table via ``nmr.meta.fleet_summary`` + neutralization scatter."""
    if not registry_entries:
        st.info("No registry entries to analyze.")
        return
    summary = fleet_summary(
        registry_entries, n_trials=n_trials, dsr_confidence=dsr_confidence
    )
    st.dataframe(summary)
    fig = px.scatter(
        summary.to_pandas(),
        x="neutralization_proportion",
        y="metric",
        color="preset",
        hover_data=["run_id", "oof_device", "dsr_pass"],
        title="Neutralization proportion vs CORR Sharpe",
    )
    st.plotly_chart(fig)


def render_campaigns(campaigns: pl.DataFrame) -> None:
    """Campaign browser — the flattened ``load_campaigns`` table."""
    if campaigns.height == 0:
        st.info("No campaign logs found under `artifacts/campaigns`.")
        return
    st.dataframe(campaigns)


def render_robustness_matrix(registry: pl.DataFrame) -> None:
    """Plotly heatmap over ``robustness_matrix`` (booleans shown as 0/1)."""
    if registry.height == 0:
        st.info("No trained runs in the registry.")
        return
    matrix = robustness_matrix(registry)
    numeric = matrix.with_columns(pl.col(flag).cast(pl.Int8) for flag in _ROBUSTNESS_CELLS)
    pdf = numeric.to_pandas().set_index("model_id").astype(float)
    fig = px.imshow(
        pdf,
        x=pdf.columns,
        y=pdf.index,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Robustness matrix",
    )
    st.plotly_chart(fig)
    st.caption(
        "Boolean cells (has_*) shown as 0/1; numeric cells "
        "(max_feature_exposure, std_corr, max_drawdown) shown raw."
    )
    st.dataframe(matrix)


def main() -> None:
    """Streamlit entry point — thin, read-only dashboard render."""
    st.set_page_config(page_title="nmr Dashboard", layout="wide")
    st.title("Numerai Model Dashboard")

    registry_dir = _DEFAULT_REGISTRY_DIR
    benchmark_path = (
        resolve_benchmark_path(_DEFAULT_BENCHMARK_PATH) or _DEFAULT_BENCHMARK_PATH
    )
    campaigns_dir = _DEFAULT_CAMPAIGNS_DIR

    registry = load_registry_frame(registry_dir)
    leaderboard = merge_leaderboard(registry, load_benchmarks(benchmark_path))
    champion = champion_run_id(registry_dir)

    st.sidebar.header("Filters")
    backend_options = sorted(leaderboard.get_column("backend").unique().to_list())
    preset_options = sorted(leaderboard.get_column("preset").unique().to_list())
    source_options = sorted(leaderboard.get_column("source").unique().to_list())
    selected_backends = st.sidebar.multiselect(
        "Backend", backend_options, default=backend_options
    )
    selected_presets = st.sidebar.multiselect(
        "Preset", preset_options, default=preset_options
    )
    selected_sources = st.sidebar.multiselect(
        "Source", source_options, default=source_options
    )
    n_trials = st.sidebar.number_input(
        "Fleet DSR n_trials", min_value=1, value=1, step=1
    )
    st.sidebar.caption(
        "n_trials is recorded as policy context; the stored DSR was computed "
        "with n_trials=1 at scorecard time."
    )
    dsr_confidence = st.sidebar.number_input(
        "DSR confidence", min_value=0.01, max_value=0.99, value=0.95, step=0.01
    )

    filtered = leaderboard
    if selected_backends:
        filtered = filtered.filter(pl.col("backend").is_in(selected_backends))
    if selected_presets:
        filtered = filtered.filter(pl.col("preset").is_in(selected_presets))
    if selected_sources:
        filtered = filtered.filter(pl.col("source").is_in(selected_sources))

    st.header("Leaderboard")
    render_leaderboard(filtered, champion)

    st.header("Run detail")
    render_run_detail(filtered)

    st.header("Fleet analysis")
    render_fleet(
        _load_registry_entries(registry_dir),
        n_trials=int(n_trials),
        dsr_confidence=float(dsr_confidence),
    )

    st.header("Campaigns")
    render_campaigns(load_campaigns(campaigns_dir))

    st.header("Robustness matrix")
    render_robustness_matrix(registry)


if __name__ == "__main__":
    main()
