"""Compile the executive HTML performance report from the shared engine.

Thin control plane only: data comes from ``nmr.dashboard``, payload/geometry
from ``dashboard_ui.charts``, raw assets from ``dashboard_ui.static``. No
metric math here. The output is a single self-contained HTML file (vanilla
CSS + JS, no Plotly, no CDN, < 112 KiB) that runs offline from ``file://``.
"""

from __future__ import annotations

import html
import json
import logging
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from dashboard_ui import charts
from nmr.benchmark import load_benchmark_file
from nmr.config import REPO_ROOT
from nmr.dashboard import (
    _DASHBOARD_COLUMN_TOOLTIPS,
    DEFAULT_DATA_DIR,
    DEFAULT_GATE_PATH,
    DEFAULT_RANK_METRIC,
    DEFAULT_REGISTRY_DIR,
    EVALUABLE_ROWS,
    build_tournament_payload,
    evaluate_gate_status,
    extract_multimetric_timeseries,
    extract_pairwise_similarity_matrix,
    load_unified_leaderboard,
    read_champion_pointer,
    reconcile_capital_metrics,
)

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_ARTIFACT_BYTES = 112 * 1024


def _read_asset(name: str) -> str:
    """Read a static asset once (cached at import). Content is static."""
    return (_STATIC_DIR / name).read_text(encoding="utf-8")


_STYLE_CSS = _read_asset("style.min.css")
_APP_JS = _read_asset("app.min.js")
_LAYOUT_HTML = _read_asset("layout.html")

_METRIC_CONTROLS_HTML = (
    '<div class="controls">'
    '<select id="metric-select">'
    '<option value="payout">Net Payout Return</option>'
    '<option value="corr20">CORR (20D)</option>'
    '<option value="mmc20">MMC (20D)</option>'
    '<option value="corr60">CORR (60D)</option>'
    '<option value="mmc60">MMC (60D)</option>'
    '<option value="bmc">BMC</option>'
    '<option value="cwmm">CWMM</option>'
    "</select>"
    '<button id="view-standard" class="active">Standard View</button>'
    '<button id="view-cumulative">Cumulative View</button>'
    '<span id="axis-label" class="axis-label"></span>'
    "</div>"
)
_TS_CHART_HTML = (
    '<div class="chart-box interactive-chart">'
    '<div class="chart-titlebar"><div><span class="chart-kicker">ERA LENS</span>'
    "<strong>Per-era performance trajectory</strong></div>"
    '<span id="timeseries-axis-label" class="axis-label"></span></div>'
    '<svg id="timeseries-svg" viewBox="0 0 800 320" role="img" '
    'aria-label="Per-era model performance trajectory"></svg>'
    '<div id="timeseries-legend" class="timeseries-legend" aria-label="Model colour legend"></div>'
    '<div id="timeseries-tooltip" class="chart-tooltip tooltip" hidden></div></div>'
)
_LB_CHART_HTML = '<div class="chart-box"><svg id="leaderboard-svg" viewBox="0 0 800 420"></svg></div>'
_DD_CHART_HTML = (
    '<div class="chart-box interactive-chart">'
    '<div class="chart-titlebar"><div><span class="chart-kicker">RISK LENS</span>'
    "<strong>Underwater trajectory</strong></div>"
    '<span id="drawdown-axis-label" class="axis-label">Peak-to-trough loss</span></div>'
    '<svg id="drawdown-svg" viewBox="0 0 800 240" role="img" '
    'aria-label="Model payout drawdown chart"></svg>'
    '<div id="drawdown-legend" class="timeseries-legend" aria-label="Drawdown model legend"></div>'
    '<div id="drawdown-tooltip" class="chart-tooltip tooltip" hidden></div></div>'
)
_EMPTY_TS_HTML = '<div class="chart-box"><p>Timeseries data unavailable without local v5.3 assets</p></div>'


def _fmt(value, *, pct: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:  # NaN
        return "—"
    if number == float("inf"):
        return "∞"
    if number == float("-inf"):
        return "-∞"
    if pct:
        return f"{number:.2%}"
    return f"{number:.4f}"


def _table_header(field: str, label: str) -> str:
    tooltip = _DASHBOARD_COLUMN_TOOLTIPS[field]
    return (
        f'<th class="metric-header" data-metric="{html.escape(field)}">'
        f'<span class="header-label">{html.escape(label)}</span>'
        f'<button type="button" class="metric-info" aria-expanded="false" '
        f'aria-label="Explain {html.escape(label)}">'
        '<span aria-hidden="true">i</span>'
        f'<span class="metric-tooltip" role="tooltip">{html.escape(tooltip)}</span>'
        "</button></th>"
    )


def _bar_label(row: dict) -> str:
    model_id = row["model_id"] or "?"
    label = row.get("display_name") or row["run_name"]
    if row["source"] == "benchmark":
        return f"{label} · {model_id}"
    return f"{label} · {model_id[:8]}"


def _bar_input(leaderboard: pl.DataFrame, champion: str | None) -> pl.DataFrame:
    evaluable = leaderboard.filter(EVALUABLE_ROWS)
    top = evaluable.sort(DEFAULT_RANK_METRIC, descending=True, nulls_last=True).head(10)
    return pl.DataFrame(
        [
            {
                "label": _bar_label(row),
                "corr_sharpe_ac": row["corr_sharpe_ac"],
                "corr_sharpe_ac_ci_low": row["corr_sharpe_ac_ci_low"],
                "corr_sharpe_ac_ci_high": row["corr_sharpe_ac_ci_high"],
                "champion": row["model_id"] == champion,
                "mean_payout": row.get("mean_payout"),
                "max_drawdown": row.get("max_drawdown"),
                "deflated_sharpe": row.get("deflated_sharpe"),
            }
            for row in top.to_dicts()
        ]
    )


def _kpi_cards(
    leaderboard: pl.DataFrame, champion: str | None, hurdle_sharpe: float
) -> dict:
    fleet = leaderboard.filter(pl.col("source").is_in(["trained", "trained_legacy"]))
    top = fleet.sort(DEFAULT_RANK_METRIC, descending=True, nulls_last=True).head(1)
    top_row = top.row(0, named=True) if top.height else None
    payout_values = [
        row.get("mean_payout")
        for row in fleet.to_dicts()
        if row.get("mean_payout") is not None
    ]
    champion_row = None
    if champion is not None:
        champ_frame = leaderboard.filter(pl.col("model_id") == champion)
        if champ_frame.height:
            champion_row = champ_frame.row(0, named=True)
        else:
            logger.warning(
                "dashboard_ui.report: champion %s not found in leaderboard; "
                "treating as none designated",
                champion,
            )
    return {
        "champion_label": (
            "None Designated" if champion_row is None else _bar_label(champion_row)
        ),
        "champion_detail": "(Unallocated)" if champion_row is None else "Active",
        "top_contender_label": _bar_label(top_row) if top_row else "—",
        "top_contender_payout": top_row["mean_payout"] if top_row else None,
        "top_contender_sharpe": top_row["corr_sharpe_ac"] if top_row else None,
        "hurdle_sharpe": hurdle_sharpe,
        "gap": (
            (top_row["corr_sharpe_ac"] - hurdle_sharpe)
            if top_row and top_row["corr_sharpe_ac"] is not None
            else None
        ),
        "fleet_best_payout": max(payout_values) if payout_values else None,
        "worst_drawdown": min(
            [
                row["max_drawdown"]
                for row in fleet.to_dicts()
                if row["max_drawdown"] is not None
            ],
            default=None,
        ),
        "capital_ready_count": fleet.join(
            leaderboard.select(["model_id", "status"]), on="model_id", how="left"
        )
        .filter(pl.col("status") == "CAPITAL READY")
        .height,
        "fleet_count": fleet.height,
        "data_version": "v5.3",
        "n_eras": (
            leaderboard.get_column("n_eras").drop_nulls().max()
            if leaderboard.height
            else None
        ),
    }


def _table_rows(leaderboard: pl.DataFrame, champion: str | None) -> list[dict]:
    rows = leaderboard.to_dicts()
    champion_rows = [
        r for r in rows if champion is not None and r["model_id"] == champion
    ]
    full_rows = sorted(
        [r for r in rows if r["source"] == "full"],
        key=lambda r: (str(r["run_name"] or ""), str(r["model_id"])),
    )
    partial_rows = sorted(
        [r for r in rows if r["source"] == "partial"],
        key=lambda r: (str(r["run_name"] or ""), str(r["model_id"])),
    )
    fleet_rows = sorted(
        [
            r
            for r in rows
            if r["source"] in ("trained", "trained_legacy")
            and r["model_id"] != champion
        ],
        key=lambda r: (
            -(
                r.get("mean_payout")
                if r.get("mean_payout") is not None
                else float("-inf")
            ),
            r["model_id"],
        ),
    )
    bench_rows = sorted(
        [r for r in rows if r["source"] == "benchmark"],
        key=lambda r: ((r["tier"] if r["tier"] is not None else 99), r["model_id"]),
    )
    if full_rows or partial_rows:
        groups: list[dict] = []
        if full_rows:
            groups.append({"_group_header": "Promoted Full Versions"})
            groups.extend(full_rows)
        if partial_rows:
            groups.append({"_group_header": "Train-Only Exports"})
            groups.extend(partial_rows)
        return champion_rows + groups + fleet_rows + bench_rows
    return champion_rows + fleet_rows + bench_rows


_STATUS_BADGE = {
    "CHAMPION": "champion",
    "CAPITAL READY": "ready",
    "RESEARCH": "research",
    "GATE HURDLE": "hurdle",
    "BENCHMARK": "benchmark",
    "FULL": "full",
    "PARTIAL": "partial",
}


def _status_badge(status: str) -> str:
    cls = _STATUS_BADGE.get(status, "research")
    return f'<span class="badge {cls}">{html.escape(status)}</span>'


def _td_gate(value_str: str, gate_pass: bool | None) -> str:
    if gate_pass is False:
        return f'<td class="num gate-fail">{value_str}</td>'
    return f'<td class="num">{value_str}</td>'


def _lifecycle_badge(row: dict) -> str:
    """Family lifecycle badge; DEGRADED surfaced under current_full_status,
    STALE flagged on broken active staked records (spec §5/§8)."""
    stage = row.get("lifecycle_stage")
    if not stage:
        return ""
    degraded = stage == "degraded" or row.get("current_full_status") == "degraded"
    cls = "lifecycle" + (" degraded" if degraded else f" {stage}")
    text = "DEGRADED" if degraded else str(stage).upper()
    if row.get("stale"):
        text += " · STALE"
    return f' <span class="badge {cls}">{html.escape(text)}</span>'


def _new_badge(row: dict) -> str:
    if not row.get("is_new"):
        return ""
    return '<span class="new-badge" title="Latest trained model">NEW</span>'


def _row_html(row: dict) -> str:
    if row.get("_group_header"):
        return (
            '<tr class="group-header"><td colspan="9">'
            f"{html.escape(row['_group_header'])}</td></tr>"
        )
    status = _status_badge(row.get("status", "RESEARCH"))
    sharpe = _fmt(row.get("corr_sharpe_ac"))
    ci = "—"
    if (
        row.get("corr_sharpe_ac_ci_low") is not None
        and row.get("corr_sharpe_ac_ci_high") is not None
    ):
        ci = f"[{_fmt(row['corr_sharpe_ac_ci_low'])}–{_fmt(row['corr_sharpe_ac_ci_high'])}]"
    model_label = html.escape(_bar_label(row))
    model_label += _new_badge(row)
    if row.get("has_full_version") and not row.get("lifecycle_stage"):
        model_label += ' <span class="badge full">FULL</span>'
    model_label += _lifecycle_badge(row)
    return (
        "<tr>"
        f"<td>{status}</td>"
        f"<td>{model_label}</td>"
        f'<td class="num">{_fmt(row.get("mean_payout"), pct=True)}</td>'
        f"{_td_gate(sharpe, row.get('gate_corr_sharpe_ac'))}"
        f'<td class="num">{ci}</td>'
        f"<td class=\"num\">{_fmt(row.get('max_drawdown'), pct=True)}</td>"
        f"{_td_gate(_fmt(row.get('gain_to_pain_ratio')), row.get('gate_gain_to_pain_ratio'))}"
        f"<td class=\"num\">{_fmt(row.get('mmc_down'))}</td>"
        f"{_td_gate(_fmt(row.get('deflated_sharpe')), row.get('gate_deflated_sharpe'))}"
        "</tr>"
    )


def _technical_entries(registry_dir: Path) -> list[dict]:
    """Per-run config summaries for the audit accordion (bounded size).

    Full ``run.json`` dumps (~25 KB per run) blow the < 112 KiB artifact budget
    (measured: 29 runs = ~715 KB), so the accordion carries the curated config
    summary only; the immutable full payload lives in the registry. Legacy
    one-level layout first (test fixtures); the experiments layout
    (``*/runs/*/run.json`` — where records live since Task 11) when empty.
    """
    entries = []
    run_files = sorted(registry_dir.glob("*/run.json"))
    if not run_files:
        run_files = sorted(registry_dir.glob("*/runs/*/run.json"))
    for run_file in run_files:
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        manifest = payload.get("manifest") or {}
        cfg = manifest.get("config") or {}
        run_cfg = cfg.get("run") or {}
        summary = {
            "backend": (cfg.get("model") or {}).get("backend"),
            "preset": (cfg.get("model") or {}).get("preset"),
            "feature_set": (cfg.get("data") or {}).get("feature_set"),
            "feature_subset": (cfg.get("data") or {}).get("feature_subset"),
            "neutralization_proportion": (cfg.get("risk") or {}).get(
                "neutralization_proportion"
            ),
            "seed": run_cfg.get("seed"),
            "device": manifest.get("oof_device"),
            "targets": (cfg.get("data") or {}).get("targets"),
        }
        entries.append(
            {
                "label": f"{run_cfg.get('name', 'unknown')} · "
                f"{str(payload.get('run_id') or run_file.parent.name)[:8]}",
                "summary": summary,
                "json_text": json.dumps(summary, indent=2, sort_keys=True),
            }
        )
    return entries


def _diversification_stats(matrix: list[list[float]]) -> dict:
    """Max/mean off-diagonal overlap + badge tier (decision #16)."""
    n = len(matrix)
    off = [matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    mean = float(np.mean(off)) if off else None
    maximum = float(max(off)) if off else None
    if mean is None:
        badge = "—"
    elif mean < 0.65:
        badge = "EXCELLENT DIVERSIFICATION"
    elif mean <= 0.85:
        badge = "MODERATE OVERLAP"
    else:
        badge = "HIGH REDUNDANCY"
    return {"mean_overlap": mean, "max_overlap": maximum, "badge": badge}


def _ensemble_sharpe(payout_metric: dict) -> float | None:
    """Equal-weighted blended Sharpe from per-era payout series (decision #17).

    SR_blended = mean(mu) / sqrt(w^T Sigma w), w uniform; None when fewer
    than 3 usable series (decision #27) or zero variance.
    """
    series = [
        np.asarray(v["standard"], dtype=float)
        for v in payout_metric.values()
        if v.get("standard")
    ]
    if len(series) < 3:
        return None
    stacked = np.vstack(series)
    mu = np.mean(stacked, axis=1)
    weights = np.full(len(mu), 1.0 / len(mu))
    variance = float(weights @ np.cov(stacked) @ weights)
    if variance <= 0.0 or not np.isfinite(variance):
        return None
    return float(np.mean(mu) / np.sqrt(variance))


def _badge_html(stats: dict, stress: dict) -> str:
    delta = stress.get("mean_delta")
    delta_text = "—" if delta is None else f"{delta:+.3f}"
    mean_text = "—" if stats["mean_overlap"] is None else f"{stats['mean_overlap']:.3f}"
    max_text = "—" if stats["max_overlap"] is None else f"{stats['max_overlap']:.3f}"
    return (
        f'<p class="badge-line"><b>{html.escape(stats["badge"])}</b> · '
        f"Mean Overlap {mean_text} · Max Overlap {max_text} · "
        f"Stress-Regime Δρ {delta_text}</p>"
    )


def _ensemble_card_html(value: float | None) -> str:
    text = "—" if value is None else f"{value:.3f}"
    return (
        '<div class="kpi"><div class="label">Equal-Weight Ensemble Sharpe '
        f'(top-3, heuristic)</div><div class="value">{text}</div></div>'
    )


def _kpi_cards_html(kpis: dict) -> str:
    return (
        '<div class="kpi"><div class="label">Active Champion</div>'
        f'<div class="value">{html.escape(kpis["champion_label"])}</div>'
        f'<div>{html.escape(kpis["champion_detail"])}</div></div>'
        '<div class="kpi"><div class="label">Top Research Contender</div>'
        f'<div class="value">{html.escape(kpis["top_contender_label"])}</div>'
        f'<div>Payout / Era {_fmt(kpis.get("top_contender_payout"), pct=True)} · '
        f'Sharpe {_fmt(kpis["top_contender_sharpe"])} vs hurdle {_fmt(kpis["hurdle_sharpe"])}</div></div>'
        '<div class="kpi"><div class="label">Best Payout / Era (proxy)</div>'
        f'<div class="value">{_fmt(kpis["fleet_best_payout"], pct=True)}</div></div>'
        '<div class="kpi"><div class="label">Worst Proxy Drawdown</div>'
        f'<div class="value">{_fmt(kpis["worst_drawdown"], pct=True)}</div></div>'
        '<div class="kpi"><div class="label">Capital Readiness</div>'
        f'<div class="value">{kpis["capital_ready_count"]} / {kpis["fleet_count"]}</div></div>'
    )


def _table_html(leaderboard: pl.DataFrame, champion: str | None) -> str:
    rows_html = "".join(_row_html(row) for row in _table_rows(leaderboard, champion))
    headers = "".join(
        (
            _table_header("status", "Status"),
            _table_header("model", "Model"),
            _table_header("mean_payout", "Payout / Era (proxy)"),
            _table_header("corr_sharpe_ac", "Sharpe (AC)"),
            _table_header("sharpe_ci", "Sharpe CI"),
            _table_header("max_drawdown", "Max DD (proxy)"),
            _table_header("gain_to_pain_ratio", "Gain-to-Pain"),
            _table_header("mmc_down", "Downside"),
            _table_header("deflated_sharpe", "Confidence (DSR)"),
        )
    )
    return (
        f"<table><thead><tr>{headers}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def _accordion_html(technical_entries: list[dict]) -> str:
    accordion = ""
    for entry in technical_entries:
        accordion += (
            "<details><summary>"
            f"{html.escape(entry['label'])} — technical &amp; audit</summary>"
            f"<pre>{html.escape(entry['json_text'])}</pre></details>"
        )
    return accordion


def _diversification_html(badge_html: str, ensemble_card_html: str) -> str:
    return (
        badge_html
        + '<div class="chart-box interactive-chart similarity-frame">'
        + '<div class="chart-titlebar"><div><span class="chart-kicker">CORRELATION</span>'
        + "<strong>Signal similarity</strong></div>"
        + '<span class="axis-label">hover a cell for pair detail</span></div>'
        + '<div id="similarity-host"></div>'
        + '<div id="similarity-tooltip" class="chart-tooltip tooltip" hidden></div></div>'
        + ensemble_card_html
    )


def _build_html(
    *,
    kpis: dict,
    table_html: str,
    diversification_html: str,
    accordion_html: str,
    payload: dict[str, Any],
    suite_version: str = "v2",
    as_of_era: str | None = None,
) -> str:
    """Assemble the full HTML document from the layout template + payload.

    Deterministic: fixed template + sorted-key JSON + static assets. The
    data-node substitution runs LAST so payload text can never be re-processed
    by a later placeholder replacement.
    """
    payload_json = json.dumps(
        payload, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).replace("</", "<\\/")
    ts_html = _TS_CHART_HTML if payload.get("eras") else _EMPTY_TS_HTML
    replacements = [
        ("{{ INLINE_STYLE }}", _STYLE_CSS),
        ("{{ SUITE_VERSION }}", html.escape(suite_version)),
        ("{{ AS_OF_ERA }}", html.escape(as_of_era or "—")),
        ("{{ N_ERAS }}", str(kpis["n_eras"]) if kpis["n_eras"] is not None else "—"),
        ("{{ DATA_VERSION }}", html.escape(kpis["data_version"])),
        ("{{ KPI_CARDS }}", _kpi_cards_html(kpis)),
        ("{{ METRIC_CONTROLS }}", _METRIC_CONTROLS_HTML),
        ("{{ TIMESERIES_SVG }}", ts_html),
        ("{{ LEADERBOARD_SVG }}", _LB_CHART_HTML),
        ("{{ DIVERSIFICATION_SECTION }}", diversification_html),
        ("{{ DECISION_TABLE }}", table_html),
        ("{{ DRAWDOWN_SVG }}", _DD_CHART_HTML),
        ("{{ AUDIT_ACCORDION }}", accordion_html),
        (
            "{{ INLINE_DATA_SCRIPT }}",
            '<script type="application/json" id="dashboard-data">'
            f"{payload_json}</script>\n<script>\n{_APP_JS}</script>",
        ),
    ]
    html_text = _LAYOUT_HTML
    for key, value in replacements:
        html_text = html_text.replace(key, value)
    return html_text


def build_dashboard_html(
    *,
    registry_dir: Path | None = None,
    benchmark_path: Path | None | bool = None,
    data_dir: Path | None = None,
    gate_path: Path | None = None,
) -> str:
    """Build the deterministic dashboard document without writing it.

    ``data_dir`` / ``gate_path`` override the repo-default data version dir
    and tier-4 gate YAML when a custom registry root is supplied (2026-08-26
    review, SECONDARY 5): the report must never silently read the repo
    defaults through a caller-supplied root — the roots propagate from the
    caller.
    """
    registry_dir = (
        Path(registry_dir) if registry_dir is not None else DEFAULT_REGISTRY_DIR
    )
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    gate_path = Path(gate_path) if gate_path is not None else DEFAULT_GATE_PATH

    leaderboard = load_unified_leaderboard(
        registry_dir,
        benchmark_path=benchmark_path,
        models_dir=registry_dir,
    )
    leaderboard = reconcile_capital_metrics(leaderboard, data_dir)
    statuses = evaluate_gate_status(
        leaderboard, gate_path, registry_dir / "champion.json"
    )
    leaderboard = leaderboard.join(statuses, on="model_id", how="left")

    gate_cfg = load_benchmark_file(gate_path)
    assert gate_cfg.reference_column is not None
    # All official tier-4 reference columns render as curves; the gated
    # capital line (reference_column) comes first and owns the BMC benchmark.
    tier4_columns = [str(gate_cfg.reference_column), *gate_cfg.reference_columns]
    tier4_column = str(gate_cfg.reference_column)
    hurdle_sharpe = float(gate_cfg.gate.corr_sharpe_ac_min)

    champion = read_champion_pointer(registry_dir / "champion.json")
    fleet = leaderboard.filter(pl.col("source").is_in(["trained", "trained_legacy"]))
    top3_ids = (
        fleet.sort(DEFAULT_RANK_METRIC, descending=True, nulls_last=True)
        .head(3)
        .get_column("model_id")
        .to_list()
    )
    engine_payload = extract_multimetric_timeseries(
        registry_dir,
        data_dir,
        run_ids=top3_ids,
        include_tier4_ref=True,
        tier4_columns=tier4_columns,
    )
    top5_ids = (
        fleet.sort(DEFAULT_RANK_METRIC, descending=True, nulls_last=True)
        .head(5)
        .get_column("model_id")
        .to_list()
    )
    labels, _sim_ids, matrix, stress = extract_pairwise_similarity_matrix(
        registry_dir,
        data_dir,
        run_ids=top5_ids,
        include_tier4_ref=True,
        tier4_column=tier4_column,
    )
    stats = _diversification_stats(matrix)
    payout_metric = (engine_payload.get("metrics") or {}).get("payout") or {}
    top3_payout = {mid: payout_metric[mid] for mid in top3_ids if mid in payout_metric}
    ensemble_value = _ensemble_sharpe(top3_payout)

    payload = charts.build_dashboard_payload(
        eras=engine_payload.get("eras") or [],
        meta_downside_mask=engine_payload.get("meta_downside_mask") or [],
        metrics=engine_payload.get("metrics") or {},
        leaderboard_bars=_bar_input(leaderboard, champion),
        similarity_labels=labels,
        similarity_matrix=matrix,
        hurdle_sharpe=hurdle_sharpe,
        ensemble_sharpe=ensemble_value,
    )
    payload.update(
        build_tournament_payload(
            leaderboard,
            champion_id=champion,
            evaluation_eras=engine_payload.get("eras") or [],
        )
    )
    compact_metrics = charts.compact_timeseries_payload(payload.get("metrics") or {})
    series_model_ids = compact_metrics["model_ids"]
    compact_metrics["model_indices"] = [
        payload["model_ids"].index(model_id)
        for model_id in series_model_ids
        if model_id in payload["model_ids"]
    ]
    payload["metrics"] = compact_metrics
    for key in (
        "leaderboard",
        "hurdle_sharpe",
        "ensemble_sharpe",
        "landscape",
        "meta_downside_mask",
        "ci_fields",
        "cohorts",
        "rank_movement",
        "advantage",
    ):
        payload.pop(key, None)

    return _build_html(
        kpis=_kpi_cards(leaderboard, champion, hurdle_sharpe),
        table_html=_table_html(leaderboard, champion),
        diversification_html=_diversification_html(
            _badge_html(stats, stress), _ensemble_card_html(ensemble_value)
        ),
        accordion_html=_accordion_html(_technical_entries(registry_dir)),
        payload=payload,
        suite_version=str(payload["meta"]["suite_version"]),
        as_of_era=payload["meta"]["evaluation_window"]["end"],
    )


def generate_dashboard(
    *,
    registry_dir: Path | None = None,
    benchmark_path: Path | None | bool = None,
    data_dir: Path | None = None,
    gate_path: Path | None = None,
    output_path: Path | None = None,
    open_browser: bool = True,
) -> Path:
    """Build the dashboard HTML report and write it to disk."""
    output_path = (
        Path(output_path)
        if output_path is not None
        else REPO_ROOT / "artifacts" / "dashboard.html"
    )
    html_text = build_dashboard_html(
        registry_dir=registry_dir,
        benchmark_path=benchmark_path,
        data_dir=data_dir,
        gate_path=gate_path,
    )
    artifact_bytes = html_text.encode("utf-8")
    artifact_size = len(artifact_bytes)
    if artifact_size >= MAX_ARTIFACT_BYTES:
        raise ValueError(
            f"dashboard artifact exceeds {MAX_ARTIFACT_BYTES // 1024} KiB budget: "
            f"{artifact_size} bytes"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(artifact_bytes)
    if open_browser:
        webbrowser.open(output_path.as_uri())
    return output_path


def main() -> int:
    output = generate_dashboard()
    print(f"Dashboard written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
