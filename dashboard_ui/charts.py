# dashboard_charts.py
"""Plotly figure builders and the multimetric JS-controller block for the executive dashboard.

Thin presentation layer: consumes clean frames/dicts from ``nmr.dashboard``
and returns configured ``plotly.graph_objects.Figure`` instances (or, for the
multimetric chart, a self-contained HTML block with an embedded vanilla-JS
controller). No metric math, no file I/O, no registry access.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

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
    payload_json = json.dumps(payload, sort_keys=True).replace("</", "<\\/")
    return (
        '<div id="multimetric-chart" class="chart-box"></div>\n'
        '<script type="application/json" id="dashboard-multimetric-data">'
        f"{payload_json}</script>\n"
        f"<script>{_APP_JS}</script>"
    )
