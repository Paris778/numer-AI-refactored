# dashboard_ui/charts.py
"""Plotly figure builders and the multimetric JS-controller block for the executive dashboard.

Thin presentation layer: consumes clean frames/dicts from ``nmr.dashboard``
and returns configured ``plotly.graph_objects.Figure`` instances (or, for the
multimetric chart, a self-contained HTML block with an embedded vanilla-JS
controller). No metric math, no file I/O, no registry access.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl

_STATIC_DIR = Path(__file__).parent / "static"


def _read_asset(name: str) -> str:
    """Read a static asset once (cached at import). Content is static."""
    return (_STATIC_DIR / name).read_text(encoding="utf-8")


_APP_JS = _read_asset("app.js")

_HURDLE_COLOR = "#f85149"
_HURDLE_ANNOTATION = "tier-4 hurdle"
_DOWNSIDE_FILL = "rgba(248, 81, 73, 0.10)"


def build_leaderboard_bar_chart(
    df: pl.DataFrame, *, hurdle_sharpe: float
) -> go.Figure:
    """Horizontal Sharpe bars, best on top, with asymmetric CIs + hurdle line."""
    fig = go.Figure()
    if df.height == 0:
        fig.add_annotation(text="No models recorded yet", showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig
    for row in df.sort("corr_sharpe_ac", descending=False, nulls_last=True).to_dicts():
        value = row["corr_sharpe_ac"]
        error_x = None
        if value is not None and row["corr_sharpe_ac_ci_low"] is not None \
                and row["corr_sharpe_ac_ci_high"] is not None:
            error_x = {
                "type": "data",
                "symmetric": False,
                "array": [row["corr_sharpe_ac_ci_high"] - value],
                "arrayminus": [value - row["corr_sharpe_ac_ci_low"]],
            }
        escaped_label = html.escape(row["label"])
        cagr = row.get("cagr_1y")
        mdd = row.get("max_drawdown")
        dsr = row.get("deflated_sharpe")
        has_hover_fields = (
            "cagr_1y" in row and "max_drawdown" in row and "deflated_sharpe" in row
        )
        if has_hover_fields:
            customdata = [[cagr, mdd, dsr]]
            hovertemplate = (
                f"<b>{escaped_label}</b><br>Sharpe (AC): %{{x:.3f}}<br>"
                f"Ann. Return: %{{customdata[0]:.2%}}<br>"
                f"Max DD: %{{customdata[1]:.2%}}<br>"
                f"DSR: %{{customdata[2]:.3f}}<extra></extra>"
            )
        else:
            customdata = None
            hovertemplate = (
                f"{escaped_label}<br>Sharpe (AC): %{{x:.3f}}<extra></extra>"
            )
        fig.add_trace(
            go.Bar(
                name=row["label"],
                x=[value],
                y=[row["label"]],
                orientation="h",
                error_x=error_x,
                marker_pattern_shape="/" if row["champion"] else "",
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )
    fig.add_vline(
        x=hurdle_sharpe,
        line_dash="dash",
        line_color=_HURDLE_COLOR,
        annotation_text=f"{_HURDLE_ANNOTATION} {hurdle_sharpe:.2f}",
    )
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="CORR Sharpe (autocorrelation-adjusted)",
    )
    return fig


def build_drawdown_chart(payload: dict[str, Any]) -> go.Figure:
    """Underwater payout drawdown curves (v2 payload: drawdowns + eras)."""
    fig = go.Figure()
    eras = payload.get("eras") or []
    if not eras:
        fig.add_annotation(
            text="Timeseries data unavailable without local v5.3 assets",
            showarrow=False,
        )
        fig.update_layout(template="plotly_dark")
        return fig
    drawdowns = payload.get("drawdowns") or {}
    payout_metric = (payload.get("metrics") or {}).get("payout", {})
    for model_id in sorted(drawdowns):
        label = payout_metric.get(model_id, {}).get("label", model_id)
        fig.add_trace(
            go.Scatter(
                name=label,
                x=eras,
                y=drawdowns[model_id],
                mode="lines",
                fill="tozeroy",
                fillcolor=_DOWNSIDE_FILL,
                hovertemplate="%{y:.2%}<extra>" + html.escape(label) + "</extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Era",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h"),
    )
    return fig


def build_similarity_matrix_chart(
    labels: list[str], matrix: list[list[float]]
) -> go.Figure:
    """Pairwise similarity heatmap with the top-ranked row/col highlighted."""
    fig = go.Figure()
    if not matrix:
        fig.add_annotation(
            text="Similarity matrix unavailable without local v5.3 assets",
            showarrow=False,
        )
        fig.update_layout(template="plotly_dark")
        return fig
    text = [
        [
            f"<b>{v:.3f}</b>" if (i == 0 or j == 0) else f"{v:.3f}"
            for j, v in enumerate(row)
        ]
        for i, row in enumerate(matrix)
    ]
    fig.add_trace(
        go.Heatmap(
            z=matrix, x=labels, y=labels, colorscale="RdBu_r", zmid=0.5,
            text=text, texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
    return fig


def multimetric_chart_html(payload: dict[str, Any]) -> str:
    """Interactive 7-metric trajectory chart: data-node payload + static JS controller.

    No plotly ``updatemenus`` (state collision); two JS state variables drive
    ``Plotly.react``. The payload is serialized into a JSON script node that
    ``static/app.js`` reads via ``JSON.parse``; ``</`` is escaped so a hostile
    label can never close the script tag. Deterministic: fixed template +
    sorted-key JSON + static asset.
    """
    if not payload.get("eras"):
        return (
            '<div id="multimetric-chart" class="chart-box">'
            "<p>Timeseries data unavailable without local v5.3 assets</p></div>"
        )
    payload_json = json.dumps(payload, sort_keys=True, allow_nan=False).replace("</", "<\\/")
    return (
        '<div id="multimetric-chart" class="chart-box"></div>\n'
        '<script type="application/json" id="dashboard-multimetric-data">'
        f"{payload_json}</script>\n"
        f"<script>\n{_APP_JS}</script>"
    )


_METRIC_NAMES = ("payout", "corr20", "mmc20", "corr60", "mmc60", "bmc", "cwmm")
_ZERO_SPAN_EPS = 1e-12


def global_y_range(*series: Sequence[float]) -> tuple[float, float]:
    """Global min/max across all series (shared axis); (0.0, 1.0) when empty."""
    values = [v for s in series for v in s]
    if not values:
        return (0.0, 1.0)
    return (float(min(values)), float(max(values)))


def _resolve_range(
    values: Sequence[float], y_min: float | None, y_max: float | None
) -> tuple[float, float]:
    """Resolve the y range, expanding a degenerate flat span so scaling never divides by zero."""
    lo, hi = global_y_range(values) if y_min is None or y_max is None else (y_min, y_max)
    if abs(hi - lo) < _ZERO_SPAN_EPS:
        lo -= 1.0
        hi += 1.0
    return lo, hi


def data_to_svg_path(
    values: Sequence[float],
    *,
    width: float,
    height: float,
    y_min: float | None = None,
    y_max: float | None = None,
    pad: tuple[float, float, float, float] = (20.0, 20.0, 20.0, 40.0),
) -> str:
    """Map a series to an SVG polyline path (y axis inverted, top-left origin).

    ``pad`` order is (top, right, bottom, left). Empty input returns ``""``.
    """
    if not values:
        return ""
    lo, hi = _resolve_range(values, y_min, y_max)
    span = hi - lo
    pad_top, pad_right, pad_bottom, pad_left = pad
    inner_w = width - pad_left - pad_right
    inner_h = height - pad_top - pad_bottom
    denom = max(1, len(values) - 1)
    points = []
    for i, v in enumerate(values):
        x = pad_left + (i / denom) * inner_w
        y = pad_top + (1.0 - (v - lo) / span) * inner_h
        points.append(f"{x:.1f},{y:.1f}")
    return "M " + " L ".join(points)


def svg_area_path(
    values: Sequence[float],
    *,
    width: float,
    height: float,
    y_min: float | None = None,
    y_max: float | None = None,
    y_baseline: float = 0.0,
    pad: tuple[float, float, float, float] = (20.0, 20.0, 20.0, 40.0),
) -> str:
    """Closed SVG polygon: line path + baseline anchors (``L xN,yBase L x0,yBase Z``)."""
    if not values:
        return ""
    lo, hi = _resolve_range(values, y_min, y_max)
    span = hi - lo
    line = data_to_svg_path(values, width=width, height=height, y_min=lo, y_max=hi, pad=pad)
    pad_top, pad_right, pad_bottom, pad_left = pad
    inner_h = height - pad_top - pad_bottom
    inner_w = width - pad_left - pad_right
    y_base = pad_top + (1.0 - (y_baseline - lo) / span) * inner_h
    denom = max(1, len(values) - 1)
    x0 = pad_left
    x_n = pad_left + ((len(values) - 1) / denom) * inner_w
    return f"{line} L {x_n:.1f},{y_base:.1f} L {x0:.1f},{y_base:.1f} Z"


def cumulative_series(standard: Sequence[float], *, payout: bool) -> list[float]:
    """cumprod(1+r) for payout, cumsum(rho) for correlations (spec decision #9)."""
    values = np.asarray(standard, dtype=float)
    if payout:
        return [float(v) for v in np.cumprod(1.0 + values)]
    return [float(v) for v in np.cumsum(values)]


def drawdown_series(cumulative: Sequence[float]) -> list[float]:
    """wealth/peak - 1 (peak = running maximum)."""
    wealth = np.asarray(cumulative, dtype=float)
    peak = np.maximum.accumulate(wealth)
    return [float(v) for v in wealth / peak - 1.0]


def build_dashboard_payload(
    *,
    eras: Sequence[str],
    meta_downside_mask: Sequence[bool],
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    leaderboard_bars: pl.DataFrame,
    similarity_labels: Sequence[str],
    similarity_matrix: Sequence[Sequence[float]],
    hurdle_sharpe: float,
    ensemble_sharpe: float | None,
) -> dict[str, Any]:
    """Shape engine output into the standard-only, metric-first vanilla contract.

    ``metrics`` mirrors ``nmr.dashboard.extract_multimetric_timeseries``
    (metric-first); cumulative/drawdown are derived client-side, so only the
    ``standard`` arrays and labels are carried. ``leaderboard_bars`` must be a
    frame with columns ``label, corr_sharpe_ac, corr_sharpe_ac_ci_low,
    corr_sharpe_ac_ci_high, cagr_1y, max_drawdown, deflated_sharpe, champion``.
    """
    shaped_metrics: dict[str, Any] = {}
    for metric, models in metrics.items():
        shaped_metrics[metric] = {
            model_id: {"standard": list(series["standard"]), "label": series["label"]}
            for model_id, series in models.items()
        }
    rows = [
        {
            "label": row["label"],
            "sharpe": row["corr_sharpe_ac"],
            "ci_low": row["corr_sharpe_ac_ci_low"],
            "ci_high": row["corr_sharpe_ac_ci_high"],
            "cagr_1y": row.get("cagr_1y"),
            "max_drawdown": row.get("max_drawdown"),
            "deflated_sharpe": row.get("deflated_sharpe"),
            "champion": row["champion"],
        }
        for row in leaderboard_bars.to_dicts()
    ]
    return {
        "eras": list(eras),
        "meta_downside_mask": [bool(m) for m in meta_downside_mask],
        "metrics": shaped_metrics,
        "leaderboard": rows,
        "similarity": {
            "labels": list(similarity_labels),
            "matrix": [list(r) for r in similarity_matrix],
        },
        "hurdle_sharpe": float(hurdle_sharpe),
        "ensemble_sharpe": ensemble_sharpe,
    }


__all__ = [
    "build_dashboard_payload",
    "cumulative_series",
    "data_to_svg_path",
    "drawdown_series",
    "global_y_range",
    "svg_area_path",
]
