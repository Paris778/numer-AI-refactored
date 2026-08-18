"""Pure geometry + payload builders for the vanilla executive dashboard.

Presentation math only: SVG coordinate scaling, series transforms, and the
JSON data contract for ``static/app.js``. No metric formulas, no file I/O, no
registry access; ``nmr.dashboard`` stays the analytical engine. All geometry
here is mirrored client-side by ``static/app.js`` and covered by
``tests/test_dashboard_ui.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

__all__ = [
    "build_dashboard_payload",
    "cumulative_series",
    "data_to_svg_path",
    "drawdown_series",
    "global_y_range",
    "svg_area_path",
]

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
