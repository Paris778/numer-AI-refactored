"""Presentation-layer tests for the vanilla dashboard (geometry, payload, compiler)."""

from __future__ import annotations

import json
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
    assert "RANKED: " in js


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
        "top_contender_sharpe": 0.9,
        "hurdle_sharpe": 0.78,
        "gap": 0.12,
        "fleet_best_cagr": 0.15,
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
                "risk": {"neutralization_proportion": 1.0},
                "run": {"name": "sample-run"},
            },
        },
        "scorecard": {
            "corr": 0.12,
            "corr_ci_low": 0.05,
            "corr_ci_high": 0.19,
            "corr_n_eras": 30,
            "corr_sharpe_ac": 0.8,
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
    size_kb = out.stat().st_size / 1024
    assert size_kb < 100, f"bundle too large: {size_kb:.2f} KB"
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


def test_technical_entries_summary_only(tmp_path: Path) -> None:
    # < 100 KB gate: the audit accordion must carry config summaries, not
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
    # rows, 29 accordion summaries — pins the < 100 KB gate against growth
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
    size_kb = len(html_text.encode("utf-8")) / 1024
    assert size_kb < 100, f"real-scale artifact too large: {size_kb:.2f} KB"
