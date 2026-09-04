"""Presentation-layer tests for the vanilla dashboard (geometry, payload, compiler)."""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

import polars as pl
import pytest

import nmr.dashboard as nmr_dashboard
from dashboard_ui import charts, report


def test_data_to_svg_path_basic_polyline() -> None:
    path = charts.data_to_svg_path(
        [0.0, 1.0],
        width=100.0,
        height=100.0,
        y_min=0.0,
        y_max=1.0,
        pad=(0.0, 0.0, 0.0, 0.0),
    )
    assert path == "M 0.0,100.0 L 100.0,0.0"


def test_data_to_svg_path_inverts_y() -> None:
    # larger value -> smaller SVG y (returns rise on screen)
    path = charts.data_to_svg_path(
        [1.0, 2.0],
        width=100.0,
        height=100.0,
        y_min=0.0,
        y_max=2.0,
        pad=(0.0, 0.0, 0.0, 0.0),
    )
    assert path == "M 0.0,50.0 L 100.0,0.0"


def test_data_to_svg_path_zero_span_guard() -> None:
    path = charts.data_to_svg_path(
        [0.5, 0.5],
        width=100.0,
        height=100.0,
        pad=(10.0, 10.0, 10.0, 10.0),
    )
    assert "NaN" not in path and "Inf" not in path
    ys = [pt.split(",")[1] for pt in path.split(" L ")]
    assert ys[0] == ys[1]


def test_data_to_svg_path_single_point() -> None:
    path = charts.data_to_svg_path(
        [0.5],
        width=100.0,
        height=100.0,
        y_min=0.0,
        y_max=1.0,
        pad=(0.0, 0.0, 0.0, 0.0),
    )
    assert path == "M 0.0,50.0"


def test_data_to_svg_path_empty_input() -> None:
    assert charts.data_to_svg_path([], width=100.0, height=100.0) == ""


def test_chart_geometry_and_transport_preserve_missing_values() -> None:
    path = charts.data_to_svg_path(
        [0.0, None, 1.0],
        width=100.0,
        height=100.0,
        y_min=0.0,
        y_max=1.0,
        pad=(0.0, 0.0, 0.0, 0.0),
    )
    assert path == "M 0.0,100.0 M 100.0,0.0"
    assert charts.cumulative_series([0.1, None, 0.1], payout=True) == [1.1, None, None]
    assert charts.drawdown_series([1.0, None, 1.1]) == [0.0, None, None]

    compact = charts.compact_timeseries_payload(
        {
            "payout": {
                "model": {
                    "standard": [0.01, None, float("nan"), -0.02],
                    "label": "model",
                }
            }
        }
    )
    assert compact["series_encoding"] == "int32-le-base64-v1"
    encoded = base64.b64decode(compact["series"]["payout"][0])
    assert struct.unpack("<iiii", encoded) == (10_000, -(2**31), -(2**31), -20_000)


def test_svg_area_path_closes_to_baseline() -> None:
    path = charts.svg_area_path(
        [0.0, -0.1],
        width=100.0,
        height=100.0,
        y_min=-0.1,
        y_max=0.0,
        y_baseline=0.0,
        pad=(0.0, 0.0, 0.0, 0.0),
    )
    assert path.endswith(" Z")
    assert " L 100.0,0.0 L 0.0,0.0 Z" in path


def test_svg_area_path_empty_input() -> None:
    assert charts.svg_area_path([], width=100.0, height=100.0) == ""


def test_cumulative_series_parity_with_engine() -> None:
    standard = [0.01, -0.02, 0.03]
    assert charts.cumulative_series(
        standard, payout=True
    ) == nmr_dashboard._cumulative_from_standard(standard, payout=True)
    assert charts.cumulative_series(
        standard, payout=False
    ) == nmr_dashboard._cumulative_from_standard(standard, payout=False)


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
        "payout": {
            "a"
            * 64: {
                "standard": [0.01, 0.02],
                "cumulative": [1.01, 1.0302],
                "label": "r1",
            }
        },
        "corr20": {
            "a" * 64: {"standard": [0.1, 0.2], "cumulative": [0.1, 0.3], "label": "r1"}
        },
    }
    bars = pl.DataFrame(
        [
            {
                "label": "r1 · abc",
                "corr_sharpe_ac": 0.8,
                "corr_sharpe_ac_ci_low": 0.6,
                "corr_sharpe_ac_ci_high": 1.0,
                "mean_payout": 0.125,
                "cagr_1y": 0.5,
                "max_drawdown": 0.1,
                "deflated_sharpe": 0.97,
                "champion": True,
            },
        ]
    )
    payload = charts.build_dashboard_payload(
        eras=["0001", "0002"],
        meta_downside_mask=[False, True],
        metrics=engine_metrics,
        leaderboard_bars=bars,
        similarity_labels=["a"],
        similarity_matrix=[[1.0]],
        hurdle_sharpe=0.78,
        ensemble_sharpe=1.2,
    )
    assert set(payload["metrics"]) == {"payout", "corr20"}  # metric-first keys
    entry = payload["metrics"]["payout"]["a" * 64]
    assert entry["standard"] == [0.01, 0.02]
    assert "cumulative" not in entry  # standard-only
    assert entry["label"] == "r1"
    assert payload["eras"] == ["0001", "0002"]
    assert payload["meta_downside_mask"] == [False, True]
    assert payload["leaderboard"][0]["sharpe"] == 0.8
    assert payload["leaderboard"][0]["mean_payout"] == 0.125
    assert "cagr_1y" not in payload["leaderboard"][0]
    assert payload["leaderboard"][0]["champion"] is True
    assert payload["similarity"] == {"labels": ["a"], "matrix": [[1.0]]}
    assert payload["hurdle_sharpe"] == 0.78
    assert payload["ensemble_sharpe"] == 1.2


def test_style_css_design_tokens() -> None:
    from dashboard_ui import report

    css = (Path(report._STATIC_DIR) / "style.css").read_text(encoding="utf-8")
    for token in (
        "--bg-deep: #0a0a0d",
        "--bg-panel: #151519",
        "--border: #2b2b35",
        "--text-hi: #f4f3ef",
        "--gold: #f5b921",
        "--coral: #ff7a7a",
        "--mint: #7edaa3",
        "--text-dim: #62626d",
    ):
        assert token in css
    for selector in (
        ".model-row.champion-row",
        ".advantage-strip",
        ".landscape-point",
        ".profile-bar",
        ".crosshair",
        ".tooltip",
        ".model-drawer",
    ):
        assert selector in css


def test_app_js_contains_renderer_functions() -> None:
    from dashboard_ui import report

    js = (Path(report._STATIC_DIR) / "app.js").read_text(encoding="utf-8")
    for fn in (
        "dataToSvgPath",
        "svgAreaPath",
        "cumulativeSeries",
        "drawdownSeries",
        "globalYRange",
        "renderTimeseries",
        "renderLeaderboard",
        "renderSimilarity",
        "renderDrawdown",
    ):
        assert fn in js
    assert "</script" not in js  # inlined into a <script> node — must never close it


def test_static_renderer_sources_are_encoding_safe_and_have_valid_row_markup() -> None:
    from dashboard_ui import report

    js = report._read_asset("app.js")
    min_js = report._read_asset("app.min.js")
    layout = report._read_asset("layout.html")
    for text in (js, min_js, layout):
        assert not any(
            marker in text for marker in ("\u00d4", "\u252c", "\u00e5\u00c6", "\u00c7")
        )
    assert 'data-model-id=\\"" + esc(row.model_id) + "\\" tabindex=\\"0\\">' in js
    assert 'default_rank_metric || "mean_payout"' in js
    assert 'default_rank_metric||"mean_payout"' in min_js
    assert "policyContext" in js
    assert "metric-unavailable" in js
    assert "Max DD (proxy)" in js
    assert "RANKED: " in js


def test_static_renderer_uses_per_era_payout_proxy_column() -> None:
    from dashboard_ui import report

    js = report._read_asset("app.js")
    min_js = report._read_asset("app.min.js")

    assert 'headerCell("mean_payout"' in js
    assert "PAYOUT / ERA" in js
    assert "CAGR 1Y" not in js
    assert "cagr_1y" not in js
    assert 'metricValue(row, "mean_payout")' in js
    assert "mean_payout" in min_js
    assert "cagr_1y" not in min_js
    assert "not annual return" in min_js
    assert "Best Payout / Era (proxy)" in report._kpi_cards_html(_kpis_for_test())
    assert "column_tooltips" in js
    assert "metric-info" in js
    assert "metric-tooltip" in js
    assert "aria-expanded" in js


def test_table_headers_have_metric_tooltips() -> None:
    from dashboard_ui import report

    rows = pl.DataFrame(
        [
            {
                "model_id": "a" * 64,
                "source": "trained",
                "run_name": "sample-run",
                "status": "RESEARCH",
                "mean_payout": 0.05,
                "corr_sharpe_ac": 0.5,
            }
        ]
    )

    html_text = report._table_html(rows, champion=None)

    for field in (
        "status",
        "model",
        "mean_payout",
        "corr_sharpe_ac",
        "sharpe_ci",
        "max_drawdown",
        "gain_to_pain_ratio",
        "mmc_down",
        "deflated_sharpe",
    ):
        assert f'data-metric="{field}"' in html_text
    assert "Policy-clipped mean payout per eligible scored era" in html_text
    assert "Higher is better" in html_text
    assert "Lower is better" in html_text


def test_renderer_supports_pointer_tooltips_axes_benchmark_rows_and_medals() -> None:
    from dashboard_ui import report

    js = report._read_asset("app.js")
    css = report._read_asset("style.css")
    layout = report._read_asset("layout.html")
    for token in (
        "pointermove",
        "pointerdown",
        "showChartTooltip",
        "attachTimeseriesTooltip",
        "Evaluation era",
        "landscape-tooltip",
        "profile-tooltip",
        "similarity-tooltip",
        "drawdown-tooltip",
        "timeseries-legend",
        "hover-guide",
        "matrix-alpha",
        "data-sim-row",
        "rankCell",
        "decodeSeries",
        "metrics.model_ids",
        "chart-hit-area",
        "drawdown-legend",
        "landscape-legend",
        "aria-pressed",
        "trapDrawerFocus",
        "strictlyBeats",
        "compareLowerMetric",
        "Most robust",
        "strict_beats",
        "aria-label",
        "event.preventDefault",
        "tooltipLine",
        "tooltip-series",
        "--tooltip-color",
    ):
        assert token in js or token in layout
    for token in (
        ".benchmark-row",
        ".rank-number",
        ".rank-1",
        ".rank-2",
        ".rank-3",
        ".chart-tooltip",
        ".timeseries-legend",
        ".legend-item",
        ".hover-guide",
        ".similarity td.diagonal",
        "@keyframes pointPulse",
    ):
        assert token in css
    assert 'role="dialog"' in layout
    assert 'aria-modal="true"' in layout


def test_alpha_chart_has_explicit_title_and_axis_mounts() -> None:
    from dashboard_ui import report

    layout = report._read_asset("layout.html")
    chart_fragment = report._TS_CHART_HTML
    js = report._read_asset("app.js")
    for token in (
        "Per-era performance trajectory",
        "Evaluation era",
        "timeseries-y-axis",
    ):
        assert token in layout or token in chart_fragment or token in js


def test_build_html_exposes_tournament_shell_contract() -> None:
    from dashboard_ui import report

    html_text = report._build_html(**_build_html_kwargs())
    assert "Model Tournament" in html_text
    assert "OFFLINE EVALUATION" in html_text
    assert "ML ADVANTAGE" in html_text
    assert "leaderboard-table" in html_text
    assert "model-drawer" in html_text


def test_layout_html_has_compiler_placeholders() -> None:
    from dashboard_ui import report

    layout = (Path(report._STATIC_DIR) / "layout.html").read_text(encoding="utf-8")
    for ph in (
        "{{ INLINE_STYLE }}",
        "{{ N_ERAS }}",
        "{{ DATA_VERSION }}",
        "{{ METRIC_CONTROLS }}",
        "{{ TIMESERIES_SVG }}",
        "{{ DIVERSIFICATION_SECTION }}",
        "{{ DRAWDOWN_SVG }}",
        "{{ INLINE_DATA_SCRIPT }}",
    ):
        assert ph in layout


def _kpis_for_test() -> dict:
    return {
        "champion_label": "champ · abc12345",
        "champion_detail": "Active",
        "top_contender_label": "top · def67890",
        "top_contender_payout": 0.12,
        "top_contender_sharpe": 0.9,
        "hurdle_sharpe": 0.78,
        "gap": 0.12,
        "fleet_best_payout": 0.15,
        "worst_drawdown": -0.2,
        "capital_ready_count": 1,
        "fleet_count": 3,
        "data_version": "v5.3",
        "n_eras": 86,
    }


def _payload_for_test() -> dict:
    return {
        "eras": ["0001", "0002"],
        "meta_downside_mask": [False, True],
        "metrics": {
            "payout": {"a" * 64: {"standard": [0.01, -0.02], "label": "r · abc12345"}},
        },
        "leaderboard": [
            {
                "label": "r · abc12345",
                "sharpe": 0.8,
                "ci_low": 0.6,
                "ci_high": 1.0,
                "cagr_1y": 0.5,
                "max_drawdown": 0.1,
                "deflated_sharpe": 0.97,
                "champion": True,
            }
        ],
        "similarity": {"labels": ["r"], "matrix": [[1.0]]},
        "hurdle_sharpe": 0.78,
        "ensemble_sharpe": 1.2,
    }


def _build_html_kwargs() -> dict:
    return dict(
        kpis=_kpis_for_test(),
        table_html="<table><tbody><tr><td>x</td></tr></tbody></table>",
        diversification_html='<p>BADGE MODERATE OVERLAP</p><div id="similarity-host"></div><p>ENSEMBLE CARD 1.200</p>',
        accordion_html="<details><summary>s</summary><pre>j</pre></details>",
        payload=_payload_for_test(),
    )


def test_build_html_sections_and_data_node() -> None:
    html_text = report._build_html(**_build_html_kwargs())
    for section in (
        "ALPHA GENERATION",
        "SIGNAL DIVERSIFICATION",
        "CAPITAL DRAWDOWN",
        "BADGE MODERATE OVERLAP",
        "ENSEMBLE CARD",
    ):
        assert section in html_text
    assert html_text.count('id="dashboard-data"') == 1
    assert "<script src" not in html_text
    assert 'id="metric-select"' in html_text
    assert 'id="timeseries-svg"' in html_text
    assert 'id="leaderboard-table"' in html_text
    assert 'id="drawdown-svg"' in html_text
    assert 'id="similarity-host"' in html_text
    assert html_text.count("--bg-deep:#0a0a0d") == 1  # minified style inlined once
    assert "Evidence boundary" in html_text


def test_build_html_escapes_hostile_strings() -> None:
    payload = _payload_for_test()
    payload["metrics"]["payout"]["x" * 64] = {
        "standard": [0.0],
        "label": "<script>alert(1)</script>",
    }
    html_text = report._build_html(
        **{
            **_build_html_kwargs(),
            "kpis": {
                **_kpis_for_test(),
                "champion_label": '"><img src=x onerror=alert(2)>',
            },
            "payload": payload,
        }
    )
    assert '"><img src=x' not in html_text
    # The hostile label's closing tag must be neutralized inside the JSON node;
    # the former KPI markup is no longer server-rendered.
    start = html_text.index('id="dashboard-data"')
    end = html_text.index("</script>", start)
    node = html_text[start:end]
    assert "</script" not in node
    assert "<\\/script>" in node


def test_build_html_deterministic_across_calls() -> None:
    a = report._build_html(**_build_html_kwargs())
    b = report._build_html(**_build_html_kwargs())
    assert a == b


def test_build_html_empty_payload_placeholder_message() -> None:
    html_text = report._build_html(
        **{
            **_build_html_kwargs(),
            "payload": {**_payload_for_test(), "eras": [], "metrics": {}},
        }
    )
    assert "Timeseries data unavailable without local v5.3 assets" in html_text


def test_build_html_rejects_non_finite_payload() -> None:
    # fail loud: NaN/Inf must never serialize into the data node (browser
    # JSON.parse would throw on them at runtime)
    kwargs = _build_html_kwargs()
    payload = dict(kwargs["payload"])
    payload["metrics"]["payout"]["a" * 64]["standard"] = [0.01, float("nan")]
    with pytest.raises(ValueError):
        report._build_html(
            kpis=kwargs["kpis"],
            table_html=kwargs["table_html"],
            diversification_html=kwargs["diversification_html"],
            accordion_html=kwargs["accordion_html"],
            payload=payload,
        )


def test_asset_resolution_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert "--bg-deep: #0a0a0d" in report._read_asset("style.css")
    assert "dataToSvgPath" in report._read_asset("app.js")
    assert "{{ INLINE_DATA_SCRIPT }}" in report._read_asset("layout.html")


def test_table_html_gate_fail_tint() -> None:
    rows = pl.DataFrame(
        [
            {
                "model_id": "a" * 64,
                "source": "trained",
                "run_name": "sample-run",
                "corr_sharpe_ac": 0.5,
                "corr_sharpe_ac_ci_low": None,
                "corr_sharpe_ac_ci_high": None,
                "cagr_1y": 0.1,
                "max_drawdown": 0.2,
                "gain_to_pain_ratio": 1.0,
                "mmc_down": 0.02,
                "deflated_sharpe": 0.5,
                "gate_corr_sharpe_ac": False,
                "gate_cagr_1y": True,
                "gate_gain_to_pain_ratio": True,
                "gate_deflated_sharpe": True,
                "status": "RESEARCH",
            }
        ]
    )
    html_text = report._table_html(rows, champion=None)
    assert 'class="num gate-fail"' in html_text
    assert "badge research" in html_text


def test_bar_label_prefers_display_name() -> None:
    row = {
        "model_id": "a" * 64,
        "run_name": "slug-name",
        "display_name": "Fancy Label",
        "source": "trained",
    }
    assert report._bar_label(row) == "Fancy Label · " + "a" * 8
    benchmark = {
        "model_id": "ref",
        "run_name": "ref",
        "display_name": "Tier-4 Ref",
        "source": "benchmark",
    }
    assert report._bar_label(benchmark) == "Tier-4 Ref · ref"


def test_row_html_escapes_display_name() -> None:
    row = {
        "model_id": "a" * 64,
        "run_name": "slug-name",
        "display_name": '<script>alert(1)</script> & "quoted"',
        "source": "trained",
        "status": "RESEARCH",
        "corr_sharpe_ac": 0.5,
        "corr_sharpe_ac_ci_low": None,
        "corr_sharpe_ac_ci_high": None,
        "cagr_1y": None,
        "max_drawdown": None,
        "gain_to_pain_ratio": None,
        "mmc_down": None,
        "deflated_sharpe": None,
        "gate_cagr_1y": None,
        "gate_corr_sharpe_ac": None,
        "gate_gain_to_pain_ratio": None,
        "gate_deflated_sharpe": None,
        "has_full_version": False,
    }
    html_out = report._row_html(row)
    assert "<script>" not in html_out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out
    assert "&quot;quoted&quot;" in html_out


def test_row_html_renders_lifecycle_badge() -> None:
    base = {
        "model_id": "a" * 64,
        "run_name": "slug-name",
        "display_name": "Slug Name",
        "source": "trained",
        "status": "RESEARCH",
        "corr_sharpe_ac": 0.5,
        "corr_sharpe_ac_ci_low": None,
        "corr_sharpe_ac_ci_high": None,
        "cagr_1y": None,
        "max_drawdown": None,
        "gain_to_pain_ratio": None,
        "mmc_down": None,
        "deflated_sharpe": None,
        "gate_cagr_1y": None,
        "gate_corr_sharpe_ac": None,
        "gate_gain_to_pain_ratio": None,
        "gate_deflated_sharpe": None,
        "has_full_version": False,
    }
    degraded = dict(
        base, lifecycle_stage="degraded", current_full_status="degraded", stale=True
    )
    html_out = report._row_html(degraded)
    assert "DEGRADED" in html_out
    assert "STALE" in html_out
    staked = dict(
        base, lifecycle_stage="staked", current_full_status="full", stale=False
    )
    staked_html = report._row_html(staked)
    assert "STAKED" in staked_html
    assert "STALE" not in staked_html


def test_row_html_renders_new_badge() -> None:
    row = {
        "model_id": "a" * 64,
        "run_name": "latest-run",
        "source": "trained",
        "status": "RESEARCH",
        "corr_sharpe_ac": 0.5,
        "corr_sharpe_ac_ci_low": None,
        "corr_sharpe_ac_ci_high": None,
        "max_drawdown": None,
        "gain_to_pain_ratio": None,
        "mmc_down": None,
        "deflated_sharpe": None,
        "gate_corr_sharpe_ac": None,
        "gate_gain_to_pain_ratio": None,
        "gate_deflated_sharpe": None,
        "has_full_version": False,
        "is_new": True,
    }
    html_out = report._row_html(row)
    assert 'class="new-badge"' in html_out
    assert ">NEW</span>" in html_out


def test_app_js_renders_lifecycle_badge_and_display_name() -> None:
    js = report._read_asset("app.js")
    assert "lifecycle_stage" in js
    assert "current_full_status" in js
    assert "display_name" in js
    assert "is_new" in js
    assert "newBadge" in js
    assert "STALE" in js
    assert "DEGRADED" in js


def test_renderer_has_type_and_tier_badges() -> None:
    """The renderer emits colored type badges (from the engine's type_label)
    and tier badges, plus the rich model title used in drawer/profile."""
    js = report._read_asset("app.js")
    css = report._read_asset("style.css")
    min_js = report._read_asset("app.min.js")
    for token in (
        "type_label",
        "type_labels",
        "tier_label",
        "typeBadge",
        "tierBadge",
        "typeLabel",
        "type-badge",
        "tier-badge",
        "modelTitle",
        "description",
        "drawer-description",
        "About this model",
        # cohort tags render from the stack: benchmark (short "Bench") / heuristic.
        'benchmark: "Bench"',
        'heuristic: "Heuristic"',
    ):
        assert token in js
    for token in (
        ".type-badge",
        ".type-badge + .type-badge",
        ".tier-badge",
        ".type-null",
        ".type-ridge",
        ".type-benchmark",
        ".type-heuristic",
        ".type-ensemble",
        ".type-lgbm",
        ".type-xgb",
        ".type-catboost",
        ".type-trained",
        ".tier-0",
        ".tier-1",
        ".tier-2",
        ".tier-3",
        ".tier-4",
        ".type-cell",
        ".drawer-description",
        ".new-badge",
    ):
        assert token in css
    # minified renderer must stay encoding-safe and carry the badge classes.
    assert "type-badge" in min_js
    assert "tier-badge" in min_js


def test_badge_color_layers_are_distinct() -> None:
    """Three visual layers avoid collision: sequential heat tiers, neutral
    category types (benchmark fill / heuristic ghost), categorical arches."""
    css = report._read_asset("style.css")
    # Tier = sequential heat scale (distinct hexes, apex at tier 4).
    for token in ("#94A3B8", "#38BDF8", "#34D399", "#A78BFA", "#F59E0B"):
        assert token in css
    # Category types are neutral: benchmark has a fill, heuristic is ghost.
    assert ".type-benchmark" in css and ".type-heuristic" in css
    assert "background: rgba(226, 232, 240, 0.10)" in css
    assert ".type-heuristic { color: #64748B" in css
    assert "background: transparent" in css
    # Architecture palette is categorical (teal LGBM, orange XGB, indigo ridge).
    for token in ("#2DD4BF", "#FB923C", "#818CF8", "#F472B6", "#6B7280"):
        assert token in css


def test_rank_and_return_are_color_coded() -> None:
    """Olympic rank colors + podium row accents + return heat scale."""
    js = report._read_asset("app.js")
    css = report._read_asset("style.css")
    min_js = report._read_asset("app.min.js")
    for token in (
        "rankClass",
        "rank-1",
        "rank-2",
        "rank-3",
        "rank-default",
        "row-rank-1",
        "tier4MaxReturn",
        "returnClass",
        "metric-return",
        "return-neg",
        "return-tier4-peak",
        "return-alpha-breakthrough",
    ):
        assert token in js
    for token in (
        ".rank-1",
        ".rank-2",
        ".rank-3",
        ".rank-default",
        ".row-rank-1",
        ".row-rank-2",
        ".row-rank-3",
        ".metric-return",
        ".return-neg",
        ".return-low",
        ".return-mid",
        ".return-tier4-peak",
        ".return-alpha-breakthrough",
    ):
        assert token in css
    assert "rank-1" in min_js
    assert "return-tier4-peak" in min_js


def test_leaderboard_columns_are_sortable() -> None:
    """Column headers carry sort metadata; the renderer has the sort cycle,
    comparator, and indicator plumbing."""
    js = report._read_asset("app.js")
    css = report._read_asset("style.css")
    min_js = report._read_asset("app.min.js")
    for token in (
        "data-sort",
        "cycleColumnSort",
        "applyColumnSort",
        "compareColumn",
        "columnSortValue",
        "bestFirstDir",
        "sortArrow",
        "headerCell",
        "th.sortable",
        "state.sort",
        "aria-sort",
    ):
        assert token in js
    for token in ("th.sortable", "th.sortable.sorted", "sort-ind", "cursor: pointer"):
        assert token in css
    assert "data-sort" in min_js


def test_motion_polish_has_reduced_motion_guard() -> None:
    """Subtle motion polish (row entrance, medal glow, orb pulse, drawer slide)
    — all gated behind prefers-reduced-motion."""
    css = report._read_asset("style.css")
    for token in (
        "@keyframes rowIn",
        "@keyframes medalGlow",
        "@keyframes orbPulse",
        "@keyframes drawerPanelIn",
        "@keyframes drawerBackdropIn",
        "prefers-reduced-motion",
        "animation: rowIn",
        "animation: orbPulse",
        "animation: drawerPanelIn",
    ):
        assert token in css
    assert "animation: none !important" in css


def _registry_entry(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "metrics": {"mean": 0.1, "std": 0.2, "sharpe": 0.5, "max_drawdown": 0.05},
        "manifest": {
            "oof_device": "cpu",
            "config": {
                "data": {
                    "feature_set": "small",
                    "feature_subset": None,
                    "targets": ["target"],
                },
                "model": {"backend": "lightgbm", "preset": "fast"},
                "evaluation": {
                    "main_target": "target",
                    "payout_policy": "classic_atomic_ender60_r1343_v1",
                },
                "risk": {"neutralization_proportion": 1.0},
                "run": {"name": "sample-run"},
            },
        },
        "scorecard": {
            "payout_policy_id": "classic_atomic_ender60_r1343_v1",
            "scoring_target": "target_ender_60",
            "scoring_horizon": "60D",
            "corr": 0.12,
            "corr_ci_low": 0.05,
            "corr_ci_high": 0.19,
            "corr_n_eras": 30,
            "corr_sharpe_ac": 0.87,
            "corr_sharpe_ac_ci_low": 0.6,
            "corr_sharpe_ac_ci_high": 1.0,
            "max_drawdown": 0.1,
            "std_corr": 0.2,
            "deflated_sharpe": 0.97,
            "max_feature_exposure": 0.3,
            "bmc": 0.02,
            "fnc": 0.05,
            "n_eras": 30,
            "cagr_1y": 1.5,
            "gain_to_pain_ratio": 2.0,
            "kelly_fraction": 0.4,
            "mmc_down": 0.01,
            "mmc_down_reason": None,
        },
    }


def _write_registry(tmp_path: Path, entries: list[dict]) -> None:
    for entry in entries:
        run_dir = tmp_path / entry["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps(entry), encoding="utf-8")


def test_generate_dashboard_end_to_end_synthetic(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("a" * 64)])
    out = report.generate_dashboard(
        registry_dir=tmp_path,
        benchmark_path=False,
        output_path=tmp_path / "dashboard.html",
        open_browser=False,
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "sample-run" in text
    # the synthetic fixture genuinely clears the real tier-4 gate -> CAPITAL READY badge
    assert "CAPITAL READY" in text
    for section in ("ALPHA GENERATION", "SIGNAL DIVERSIFICATION", "CAPITAL DRAWDOWN"):
        assert section in text
    assert 'id="dashboard-data"' in text
    assert "plotly" not in text.lower()


def test_generate_dashboard_artifact_contract(tmp_path: Path) -> None:
    _write_registry(tmp_path, [_registry_entry("b" * 64)])
    out = report.generate_dashboard(
        registry_dir=tmp_path,
        benchmark_path=False,
        output_path=tmp_path / "dashboard.html",
        open_browser=False,
    )
    assert out.stat().st_size < report.MAX_ARTIFACT_BYTES
    text = out.read_text(encoding="utf-8")
    assert "plotly" not in text.lower()
    assert "<script src=" not in text
    assert 'id="dashboard-data"' in text


def test_generate_dashboard_empty_registry_compiles(tmp_path: Path) -> None:
    out = report.generate_dashboard(
        registry_dir=tmp_path,
        benchmark_path=False,
        output_path=tmp_path / "dashboard.html",
        open_browser=False,
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "ALPHA GENERATION" in text
    assert 'id="dashboard-data"' in text


def test_generate_dashboard_isolates_custom_registry_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nmr import paths

    isolated = tmp_path / "isolated"
    global_root = tmp_path / "global"
    isolated_id = "a" * 64
    global_id = "b" * 64
    _write_registry(isolated, [_registry_entry(isolated_id)])
    global_run = global_root / "global-family" / "runs" / global_id
    global_run.mkdir(parents=True)
    (global_run / "run.json").write_text(
        json.dumps(_registry_entry(global_id)), encoding="utf-8"
    )
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", global_root)

    text = report.build_dashboard_html(
        registry_dir=isolated,
        benchmark_path=False,
    )

    assert isolated_id in text
    assert global_id not in text


def test_technical_entries_summary_only(tmp_path: Path) -> None:
    # < 112 KiB budget: the audit accordion must carry config summaries, not
    # full run.json dumps (~25 KB per run; 29 runs = ~715 KB measured)
    _write_registry(tmp_path, [_registry_entry("c" * 64)])
    entries = report._technical_entries(tmp_path)
    assert len(entries) == 1
    assert "backend" in entries[0]["json_text"]
    assert '"scorecard"' not in entries[0]["json_text"]
    assert '"metrics"' not in entries[0]["json_text"]
    assert len(entries[0]["json_text"]) < 2048


def test_artifact_contract_real_scale_payload(tmp_path: Path) -> None:
    # real-scale guard: ~86-era meta window, 4 models x 7 metrics, 10 leaderboard
    # rows, 29 accordion summaries — pins the artifact budget against growth
    # (data node grows ~326 B/era, accordion ~465 B/run)
    eras = [f"{1100 + i}" for i in range(86)]
    metrics: dict[str, dict] = {}
    for metric in ("payout", "corr20", "mmc20", "corr60", "mmc60", "bmc", "cwmm"):
        metrics[metric] = {
            f"model{i}": {
                "standard": [0.001 * (j % 7) for j in range(86)],
                "label": f"model{i} · abc12345",
            }
            for i in range(4)
        }
    payload = {
        "eras": eras,
        "meta_downside_mask": [j % 5 == 0 for j in range(86)],
        "metrics": metrics,
        "leaderboard": [
            {
                "label": f"model{i} · abc12345",
                "sharpe": 0.8 - 0.05 * i,
                "ci_low": 0.6 - 0.05 * i,
                "ci_high": 1.0 - 0.05 * i,
                "cagr_1y": 0.5,
                "max_drawdown": 0.1,
                "deflated_sharpe": 0.97,
                "champion": i == 0,
            }
            for i in range(10)
        ],
        "similarity": {
            "labels": [f"m{i}" for i in range(6)],
            "matrix": [[1.0 if i == j else 0.5 for j in range(6)] for i in range(6)],
        },
        "hurdle_sharpe": 0.78,
        "ensemble_sharpe": 1.2,
    }
    _write_registry(tmp_path, [_registry_entry(f"{i:064d}") for i in range(29)])
    accordion = report._accordion_html(report._technical_entries(tmp_path))
    html_text = report._build_html(
        kpis=_kpis_for_test(),
        table_html="<table><tbody><tr><td>x</td></tr></tbody></table>",
        diversification_html="<p>BADGE</p>",
        accordion_html=accordion,
        payload=payload,
    )
    assert len(html_text.encode("utf-8")) < report.MAX_ARTIFACT_BYTES


@pytest.mark.skipif(
    not (Path("artifacts/registry").is_dir() and Path("data/v5.3").is_dir()),
    reason="real registry/v5.3 data absent; skipped in CI",
)
def test_real_artifact_respects_size_budget(tmp_path: Path) -> None:
    out = report.generate_dashboard(
        registry_dir=Path("artifacts/registry"),
        benchmark_path=False,
        output_path=tmp_path / "real-dashboard.html",
        open_browser=False,
    )
    assert out.stat().st_size < report.MAX_ARTIFACT_BYTES
