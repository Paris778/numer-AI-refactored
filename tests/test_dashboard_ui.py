"""Presentation-layer tests for the vanilla dashboard (geometry, payload)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import nmr.dashboard as nmr_dashboard
from dashboard_ui import charts


def test_data_to_svg_path_basic_polyline() -> None:
    path = charts.data_to_svg_path(
        [0.0, 1.0], width=100.0, height=100.0,
        y_min=0.0, y_max=1.0, pad=(0.0, 0.0, 0.0, 0.0),
    )
    assert path == "M 0.0,100.0 L 100.0,0.0"


def test_data_to_svg_path_inverts_y() -> None:
    # larger value -> smaller SVG y (returns rise on screen)
    path = charts.data_to_svg_path(
        [1.0, 2.0], width=100.0, height=100.0,
        y_min=0.0, y_max=2.0, pad=(0.0, 0.0, 0.0, 0.0),
    )
    assert path == "M 0.0,50.0 L 100.0,0.0"


def test_data_to_svg_path_zero_span_guard() -> None:
    path = charts.data_to_svg_path(
        [0.5, 0.5], width=100.0, height=100.0, pad=(10.0, 10.0, 10.0, 10.0),
    )
    assert "NaN" not in path and "Inf" not in path
    ys = [pt.split(",")[1] for pt in path.split(" L ")]
    assert ys[0] == ys[1]


def test_data_to_svg_path_single_point() -> None:
    path = charts.data_to_svg_path(
        [0.5], width=100.0, height=100.0,
        y_min=0.0, y_max=1.0, pad=(0.0, 0.0, 0.0, 0.0),
    )
    assert path == "M 0.0,50.0"


def test_data_to_svg_path_empty_input() -> None:
    assert charts.data_to_svg_path([], width=100.0, height=100.0) == ""


def test_svg_area_path_closes_to_baseline() -> None:
    path = charts.svg_area_path(
        [0.0, -0.1], width=100.0, height=100.0,
        y_min=-0.1, y_max=0.0, y_baseline=0.0, pad=(0.0, 0.0, 0.0, 0.0),
    )
    assert path.endswith(" Z")
    assert " L 100.0,0.0 L 0.0,0.0 Z" in path


def test_svg_area_path_empty_input() -> None:
    assert charts.svg_area_path([], width=100.0, height=100.0) == ""


def test_cumulative_series_parity_with_engine() -> None:
    standard = [0.01, -0.02, 0.03]
    assert charts.cumulative_series(standard, payout=True) == \
        nmr_dashboard._cumulative_from_standard(standard, payout=True)
    assert charts.cumulative_series(standard, payout=False) == \
        nmr_dashboard._cumulative_from_standard(standard, payout=False)


def test_drawdown_series_peak_trough() -> None:
    cum = [1.0, 0.9, 1.05]
    dd = charts.drawdown_series(cum)
    assert dd == pytest.approx([0.0, -0.1, 0.0])


def test_global_y_range_across_series() -> None:
    assert charts.global_y_range([1.0, 2.0], [3.0, 0.5]) == (0.5, 3.0)
    assert charts.global_y_range() == (0.0, 1.0)
    assert charts.global_y_range([], []) == (0.0, 1.0)


def test_build_dashboard_payload_metric_first_standard_only() -> None:
    engine_metrics = {
        "payout": {"a" * 64: {"standard": [0.01, 0.02],
                              "cumulative": [1.01, 1.0302], "label": "r1"}},
        "corr20": {"a" * 64: {"standard": [0.1, 0.2],
                              "cumulative": [0.1, 0.3], "label": "r1"}},
    }
    bars = pl.DataFrame([
        {"label": "r1 · abc", "corr_sharpe_ac": 0.8, "corr_sharpe_ac_ci_low": 0.6,
         "corr_sharpe_ac_ci_high": 1.0, "cagr_1y": 0.5, "max_drawdown": 0.1,
         "deflated_sharpe": 0.97, "champion": True},
    ])
    payload = charts.build_dashboard_payload(
        eras=["0001", "0002"], meta_downside_mask=[False, True],
        metrics=engine_metrics, leaderboard_bars=bars,
        similarity_labels=["a"], similarity_matrix=[[1.0]],
        hurdle_sharpe=0.78, ensemble_sharpe=1.2,
    )
    assert set(payload["metrics"]) == {"payout", "corr20"}   # metric-first keys
    entry = payload["metrics"]["payout"]["a" * 64]
    assert entry["standard"] == [0.01, 0.02]
    assert "cumulative" not in entry                         # standard-only
    assert entry["label"] == "r1"
    assert payload["eras"] == ["0001", "0002"]
    assert payload["meta_downside_mask"] == [False, True]
    assert payload["leaderboard"][0]["sharpe"] == 0.8
    assert payload["leaderboard"][0]["champion"] is True
    assert payload["similarity"] == {"labels": ["a"], "matrix": [[1.0]]}
    assert payload["hurdle_sharpe"] == 0.78
    assert payload["ensemble_sharpe"] == 1.2


def test_style_css_design_tokens() -> None:
    from dashboard_ui import report
    css = (Path(report._STATIC_DIR) / "style.css").read_text(encoding="utf-8")
    for token in ("--bg: #0d1117", "--surface: #161b22", "--border: #30363d",
                  "--text: #c9d1d9", "--accent: #58a6ff", "--danger: #f85149",
                  "--success: #3fb950", "--gold: #d29922"):
        assert token in css
    for selector in (".badge.champion", ".badge.ready", ".badge.research",
                     ".badge.hurdle", ".badge.benchmark", ".badge.full",
                     ".gate-fail", ".grid-line", ".crosshair", ".tooltip"):
        assert selector in css


def test_app_js_contains_renderer_functions() -> None:
    from dashboard_ui import report
    js = (Path(report._STATIC_DIR) / "app.js").read_text(encoding="utf-8")
    for fn in ("dataToSvgPath", "svgAreaPath", "cumulativeSeries",
               "drawdownSeries", "globalYRange", "renderTimeseries",
               "renderLeaderboard", "renderSimilarity", "renderDrawdown"):
        assert fn in js
    assert "</script" not in js   # inlined into a <script> node — must never close it


def test_layout_html_has_compiler_placeholders() -> None:
    from dashboard_ui import report
    layout = (Path(report._STATIC_DIR) / "layout.html").read_text(encoding="utf-8")
    for ph in ("{{ INLINE_STYLE }}", "{{ N_ERAS }}", "{{ DATA_VERSION }}",
               "{{ KPI_CARDS }}", "{{ METRIC_CONTROLS }}", "{{ TIMESERIES_SVG }}",
               "{{ LEADERBOARD_SVG }}", "{{ DIVERSIFICATION_SECTION }}",
               "{{ DECISION_TABLE }}", "{{ DRAWDOWN_SVG }}",
               "{{ AUDIT_ACCORDION }}", "{{ INLINE_DATA_SCRIPT }}"):
        assert ph in layout
