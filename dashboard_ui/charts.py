"""Pure geometry + payload builders for the vanilla executive dashboard.

Presentation math only: SVG coordinate scaling, series transforms, and the
JSON data contract for ``static/app.js``. No metric formulas, no file I/O, no
registry access; ``nmr.dashboard`` stays the analytical engine. All geometry
here is mirrored client-side by ``static/app.js`` and covered by
``tests/test_dashboard_ui.py``.
"""

from __future__ import annotations

import base64
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

__all__ = [
    "build_dashboard_payload",
    "compact_timeseries_payload",
    "cumulative_series",
    "data_to_svg_path",
    "drawdown_series",
    "global_y_range",
    "svg_area_path",
]

_ZERO_SPAN_EPS = 1e-12
_PAYLOAD_ROUND = 6
_SERIES_SCALE = 1_000_000
_SERIES_MISSING = -(2**31)
_SERIES_MAX = 2**31 - 1


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _pack_series(values: Sequence[Any]) -> str:
    packed = bytearray()
    for value in values:
        numeric = _finite_float(value)
        if numeric is None:
            scaled = _SERIES_MISSING
        else:
            scaled = int(round(numeric * _SERIES_SCALE))
            if scaled <= _SERIES_MISSING or scaled > _SERIES_MAX:
                raise ValueError("timeseries value is outside the packed integer range")
        packed.extend(struct.pack("<i", scaled))
    return base64.b64encode(packed).decode("ascii")


def _round6(value: Any) -> Any:
    """Round payload floats to 6 decimals (display precision is 4) — keeps the
    data node honest while fitting the report artifact budget."""
    if isinstance(value, (float, np.floating)):
        numeric = _finite_float(value)
        return None if numeric is None else round(numeric, _PAYLOAD_ROUND)
    return value


def compact_timeseries_payload(
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Encode metric-first series without repeating model IDs and labels."""
    model_ids = sorted({model_id for series in metrics.values() for model_id in series})
    labels = [
        next(
            (
                series[model_id].get("label")
                for series in metrics.values()
                if model_id in series
            ),
            model_id,
        )
        for model_id in model_ids
    ]
    return {
        "model_ids": model_ids,
        "labels": labels,
        "scale": _SERIES_SCALE,
        "series_encoding": "int32-le-base64-v1",
        "series": {
            metric: [
                _pack_series(metrics[metric].get(model_id, {}).get("standard") or [])
                for model_id in model_ids
            ]
            for metric in sorted(metrics)
        },
    }


def global_y_range(*series: Sequence[float]) -> tuple[float, float]:
    """Global min/max across all series (shared axis); (0.0, 1.0) when empty."""
    values = [
        numeric
        for s in series
        if s
        for v in s
        if (numeric := _finite_float(v)) is not None
    ]
    if not values:
        return (0.0, 1.0)
    return (float(min(values)), float(max(values)))


def _resolve_range(
    values: Sequence[float], y_min: float | None, y_max: float | None
) -> tuple[float, float]:
    """Resolve the y range, expanding a degenerate flat span so scaling never divides by zero."""
    lo, hi = (
        global_y_range(values) if y_min is None or y_max is None else (y_min, y_max)
    )
    if abs(hi - lo) < _ZERO_SPAN_EPS:
        lo -= 1.0
        hi += 1.0
    return lo, hi


def data_to_svg_path(
    values: Sequence[float | None],
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
    commands: list[str] = []
    open_segment = False
    for i, value in enumerate(values):
        numeric = _finite_float(value)
        if numeric is None:
            open_segment = False
            continue
        x = pad_left + (i / denom) * inner_w
        y = pad_top + (1.0 - (numeric - lo) / span) * inner_h
        commands.append(f"{'L' if open_segment else 'M'} {x:.1f},{y:.1f}")
        open_segment = True
    return " ".join(commands)


def svg_area_path(
    values: Sequence[float | None],
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
    pad_top, pad_right, pad_bottom, pad_left = pad
    inner_h = height - pad_top - pad_bottom
    inner_w = width - pad_left - pad_right
    y_base = pad_top + (1.0 - (y_baseline - lo) / span) * inner_h
    denom = max(1, len(values) - 1)
    parts: list[str] = []
    segment: list[tuple[int, float]] = []
    for index, value in enumerate(values):
        numeric = _finite_float(value)
        if numeric is None:
            if segment:
                parts.append(
                    _area_segment(
                        segment,
                        y_min=lo,
                        y_max=hi,
                        y_base=y_base,
                        inner_w=inner_w,
                        inner_h=inner_h,
                        pad_left=pad_left,
                        pad_top=pad_top,
                        denom=denom,
                    )
                )
                segment = []
            continue
        segment.append((index, numeric))
    if segment:
        parts.append(
            _area_segment(
                segment,
                y_min=lo,
                y_max=hi,
                y_base=y_base,
                inner_w=inner_w,
                inner_h=inner_h,
                pad_left=pad_left,
                pad_top=pad_top,
                denom=denom,
            )
        )
    return " ".join(parts)


def _area_segment(
    segment: Sequence[tuple[int, float]],
    *,
    y_min: float,
    y_max: float,
    y_base: float,
    inner_w: float,
    inner_h: float,
    pad_left: float,
    pad_top: float,
    denom: int,
) -> str:
    points = []
    for index, value in segment:
        x = pad_left + (index / denom) * inner_w
        y = pad_top + (1.0 - (value - y_min) / (y_max - y_min)) * inner_h
        points.append(f"{x:.1f},{y:.1f}")
    first_x = pad_left + (segment[0][0] / denom) * inner_w
    last_x = pad_left + (segment[-1][0] / denom) * inner_w
    return (
        "M "
        + " L ".join(points)
        + f" L {last_x:.1f},{y_base:.1f} L {first_x:.1f},{y_base:.1f} Z"
    )


def cumulative_series(
    standard: Sequence[float | None], *, payout: bool
) -> list[float | None]:
    """cumprod(1+r) for payout, cumsum(rho) for correlations (spec decision #9)."""
    result: list[float | None] = []
    accumulator = 1.0 if payout else 0.0
    available = True
    for value in standard:
        numeric = _finite_float(value)
        if not available or numeric is None:
            result.append(None)
            available = False
            continue
        accumulator = accumulator * (1.0 + numeric) if payout else accumulator + numeric
        result.append(float(accumulator))
    return result


def drawdown_series(cumulative: Sequence[float | None]) -> list[float | None]:
    """wealth/peak - 1 (peak = running maximum); 0.0 when peak <= 0 (degenerate)."""
    result: list[float | None] = []
    peak = float("-inf")
    available = True
    for value in cumulative:
        numeric = _finite_float(value)
        if not available or numeric is None:
            result.append(None)
            available = False
            continue
        peak = max(peak, numeric)
        result.append(float(numeric / peak - 1.0) if peak > 0.0 else 0.0)
    return result


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
            model_id: {
                "standard": [_round6(v) for v in series["standard"]],
                "label": series["label"],
            }
            for model_id, series in models.items()
        }
    rows = [
        {
            "label": row["label"],
            "sharpe": _round6(row["corr_sharpe_ac"]),
            "ci_low": _round6(row["corr_sharpe_ac_ci_low"]),
            "ci_high": _round6(row["corr_sharpe_ac_ci_high"]),
            "cagr_1y": _round6(row.get("cagr_1y")),
            "max_drawdown": _round6(row.get("max_drawdown")),
            "deflated_sharpe": _round6(row.get("deflated_sharpe")),
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
        "hurdle_sharpe": _round6(float(hurdle_sharpe)),
        "ensemble_sharpe": _round6(ensemble_sharpe),
    }
