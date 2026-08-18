"""Presentation layer for the executive dashboard.

All front-end code (vanilla SVG/HTML/JS report compiler, the Streamlit app,
and static assets) lives here. Pure engine logic stays in ``nmr/``; this
package only consumes ``nmr.dashboard``.
"""

from dashboard_ui.charts import (
    build_dashboard_payload,
    cumulative_series,
    data_to_svg_path,
    drawdown_series,
    global_y_range,
    svg_area_path,
)

__all__ = [
    "build_dashboard_payload",
    "cumulative_series",
    "data_to_svg_path",
    "drawdown_series",
    "global_y_range",
    "svg_area_path",
]
