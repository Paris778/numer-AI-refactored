# dashboard_charts.py
"""Plotly figure builders for the executive dashboard (presentation only).

Thin presentation layer: consumes clean frames/dicts from ``nmr.dashboard``
and returns configured ``plotly.graph_objects.Figure`` instances. No metric
math, no file I/O, no registry access.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import polars as pl

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
        fig.add_trace(
            go.Bar(
                name=row["label"],
                x=[value],
                y=[row["label"]],
                orientation="h",
                error_x=error_x,
                marker_pattern_shape="/" if row["champion"] else "",
                hovertemplate=(
                    f"{row['label']}<br>Sharpe (AC): %{{x:.3f}}<extra></extra>"
                ),
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


def _downside_spans(eras: list[str], mask: list[bool]) -> list[tuple[str, str]]:
    spans: list[tuple[str, str]] = []
    start: int | None = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        if not flag and start is not None:
            spans.append((eras[start], eras[index - 1]))
            start = None
    if start is not None:
        spans.append((eras[start], eras[-1]))
    return spans


def build_cumulative_wealth_chart(payload: dict[str, Any]) -> go.Figure:
    """Cumulative wealth curves with shaded meta-model drawdown eras."""
    eras = payload["eras"]
    fig = go.Figure()
    if not payload.get("eras"):
        fig.add_annotation(
            text="Timeseries data unavailable without local v5.3 assets",
            showarrow=False,
        )
        fig.update_layout(template="plotly_dark")
        return fig
    for series in payload["series"].values():
        fig.add_trace(
            go.Scatter(
                name=series["label"],
                x=eras,
                y=series["cumulative_wealth"],
                mode="lines",
                hovertemplate="%{y:.4f}<extra>" + series["label"] + "</extra>",
            )
        )
    for x0, x1 in _downside_spans(eras, payload["meta_downside_mask"]):
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=_DOWNSIDE_FILL,
            line_width=0, layer="below",
        )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Era",
        yaxis_title="Cumulative wealth (1.0 stake)",
        legend=dict(orientation="h"),
    )
    return fig


def build_drawdown_chart(payload: dict[str, Any]) -> go.Figure:
    """Underwater drawdown curves, filled red to zero."""
    eras = payload["eras"]
    fig = go.Figure()
    if not payload.get("eras"):
        fig.add_annotation(
            text="Timeseries data unavailable without local v5.3 assets",
            showarrow=False,
        )
        fig.update_layout(template="plotly_dark")
        return fig
    for series in payload["series"].values():
        fig.add_trace(
            go.Scatter(
                name=series["label"],
                x=eras,
                y=series["drawdown"],
                mode="lines",
                fill="tozeroy",
                fillcolor=_DOWNSIDE_FILL,
                hovertemplate="%{y:.2%}<extra>" + series["label"] + "</extra>",
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
