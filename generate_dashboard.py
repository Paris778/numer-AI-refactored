"""Compile the executive HTML performance report from the shared engine.

Thin control plane only: data comes from ``nmr.dashboard``, figures from
``dashboard_charts``, HTML from the template below. No metric math here.
"""

from __future__ import annotations

import html
import json
import logging
import webbrowser
from pathlib import Path

import plotly.io as pio
import polars as pl
from plotly.offline import get_plotlyjs

import dashboard_charts as charts
from nmr.benchmark import load_benchmark_file
from nmr.config import REPO_ROOT
from nmr.dashboard import (
    DEFAULT_DATA_DIR,
    DEFAULT_GATE_PATH,
    DEFAULT_REGISTRY_DIR,
    evaluate_gate_status,
    extract_payout_timeseries,
    load_unified_leaderboard,
    reconcile_capital_metrics,
)

logger = logging.getLogger(__name__)


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
    if pct:
        return f"{number:.2%}"
    return f"{number:.4f}"


def _bar_label(row: dict) -> str:
    model_id = row["model_id"] or "?"
    if row["source"] == "benchmark":
        return f"{row['run_name']} · {model_id}"
    return f"{row['run_name']} · {model_id[:8]}"


def _bar_input(leaderboard: pl.DataFrame, champion: str | None) -> pl.DataFrame:
    top = leaderboard.sort("corr_sharpe_ac", descending=True, nulls_last=True).head(10)
    return pl.DataFrame(
        [
            {
                "label": _bar_label(row),
                "corr_sharpe_ac": row["corr_sharpe_ac"],
                "corr_sharpe_ac_ci_low": row["corr_sharpe_ac_ci_low"],
                "corr_sharpe_ac_ci_high": row["corr_sharpe_ac_ci_high"],
                "champion": row["model_id"] == champion,
                "cagr_1y": row.get("cagr_1y"),
                "max_drawdown": row.get("max_drawdown"),
                "deflated_sharpe": row.get("deflated_sharpe"),
            }
            for row in top.to_dicts()
        ]
    )


def _champion_id(registry_dir: Path) -> str | None:
    champion_path = registry_dir / "champion.json"
    if not champion_path.exists():
        return None
    try:
        payload = json.loads(champion_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    return run_id if isinstance(run_id, str) else None


def _kpi_cards(leaderboard: pl.DataFrame, champion: str | None,
               hurdle_sharpe: float) -> dict:
    fleet = leaderboard.filter(pl.col("source").is_in(["trained", "trained_legacy"]))
    top = fleet.sort("corr_sharpe_ac", descending=True, nulls_last=True).head(1)
    top_row = top.row(0, named=True) if top.height else None
    cagr_values = [
        row["cagr_1y"] for row in fleet.to_dicts()
        if row["cagr_1y"] is not None
    ]
    champion_row = None
    if champion is not None:
        champ_frame = leaderboard.filter(pl.col("model_id") == champion)
        if champ_frame.height:
            champion_row = champ_frame.row(0, named=True)
        else:
            logger.warning(
                "generate_dashboard: champion %s not found in leaderboard; "
                "treating as none designated", champion,
            )
    return {
        "champion_label": "None Designated" if champion_row is None
                          else _bar_label(champion_row),
        "champion_detail": "(Unallocated)" if champion_row is None else "Active",
        "top_contender_label": _bar_label(top_row) if top_row else "—",
        "top_contender_sharpe": top_row["corr_sharpe_ac"] if top_row else None,
        "hurdle_sharpe": hurdle_sharpe,
        "gap": (top_row["corr_sharpe_ac"] - hurdle_sharpe)
               if top_row and top_row["corr_sharpe_ac"] is not None else None,
        "fleet_best_cagr": max(cagr_values) if cagr_values else None,
        "worst_drawdown": min(
            [row["max_drawdown"] for row in fleet.to_dicts()
             if row["max_drawdown"] is not None],
            default=None,
        ),
        "capital_ready_count": fleet.join(
            leaderboard.select(["model_id", "status"]), on="model_id", how="left"
        ).filter(pl.col("status") == "CAPITAL READY").height,
        "fleet_count": fleet.height,
        "data_version": "v5.3",
        "n_eras": leaderboard.get_column("n_eras").drop_nulls().max()
                  if leaderboard.height else None,
    }


def _table_rows(leaderboard: pl.DataFrame, champion: str | None) -> list[dict]:
    rows = leaderboard.to_dicts()
    champion_rows = [r for r in rows if champion is not None and r["model_id"] == champion]
    fleet_rows = sorted(
        [r for r in rows
         if r["source"] in ("trained", "trained_legacy") and r["model_id"] != champion],
        key=lambda r: (-(r["corr_sharpe_ac"] if r["corr_sharpe_ac"] is not None
                        else float("-inf")), r["model_id"]),
    )
    bench_rows = sorted(
        [r for r in rows if r["source"] == "benchmark"],
        key=lambda r: ((r["tier"] if r["tier"] is not None else 99), r["model_id"]),
    )
    return champion_rows + fleet_rows + bench_rows


_STATUS_BADGE = {
    "CHAMPION": "champion",
    "CAPITAL READY": "ready",
    "RESEARCH": "research",
    "GATE HURDLE": "hurdle",
    "BENCHMARK": "benchmark",
}


def _status_badge(status: str) -> str:
    cls = _STATUS_BADGE.get(status, "research")
    return f'<span class="badge {cls}">{html.escape(status)}</span>'


def _td_gate(value_str: str, gate_pass: bool | None) -> str:
    if gate_pass is False:
        return f'<td class="num gate-fail">{value_str}</td>'
    return f'<td class="num">{value_str}</td>'


def _row_html(row: dict) -> str:
    status = _status_badge(row.get("status", "RESEARCH"))
    sharpe = _fmt(row.get("corr_sharpe_ac"))
    ci = "—"
    if row.get("corr_sharpe_ac_ci_low") is not None and row.get("corr_sharpe_ac_ci_high") is not None:
        ci = f"[{_fmt(row['corr_sharpe_ac_ci_low'])}–{_fmt(row['corr_sharpe_ac_ci_high'])}]"
    return (
        "<tr>"
        f"<td>{status}</td>"
        f"<td>{html.escape(_bar_label(row))}</td>"
        f"{_td_gate(_fmt(row.get('cagr_1y'), pct=True), row.get('gate_cagr_1y'))}"
        f"{_td_gate(sharpe, row.get('gate_corr_sharpe_ac'))}"
        f"<td class=\"num\">{ci}</td>"
        f"<td class=\"num\">{_fmt(row.get('max_drawdown'), pct=True)}</td>"
        f"{_td_gate(_fmt(row.get('gain_to_pain_ratio')), row.get('gate_gain_to_pain_ratio'))}"
        f"<td class=\"num\">{_fmt(row.get('mmc_down'))}</td>"
        f"{_td_gate(_fmt(row.get('deflated_sharpe')), row.get('gate_deflated_sharpe'))}"
        "</tr>"
    )


def _technical_entries(registry_dir: Path) -> list[dict]:
    entries = []
    for run_file in sorted(registry_dir.glob("*/run.json")):
        try:
            payload = json.loads(run_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        manifest = payload.get("manifest") or {}
        cfg = manifest.get("config") or {}
        run_cfg = cfg.get("run") or {}
        entries.append(
            {
                "label": f"{run_cfg.get('name', 'unknown')} · "
                         f"{str(payload.get('run_id') or run_file.parent.name)[:8]}",
                "summary": {
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
                },
                "json_text": json.dumps(payload, indent=2, sort_keys=True),
            }
        )
    return entries


def _build_html(leaderboard: pl.DataFrame, champion: str | None, kpis: dict,
                figures: dict, registry_dir: Path,
                technical_entries: list[dict]) -> str:
    """Assemble the full HTML document (single plotly engine in <head>)."""
    engine_js = get_plotlyjs()
    figure_html = {
        name: pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=name)
        for name, fig in figures.items()
    }
    rows_html = "".join(_row_html(row) for row in _table_rows(leaderboard, champion))
    accordion = ""
    for entry in technical_entries:
        accordion += (
            "<details><summary>"
            f"{html.escape(entry['label'])} — technical &amp; audit</summary>"
            f"<pre>{html.escape(entry['json_text'])}</pre></details>"
        )
    return (
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NumerAI Executive Performance Report</title>
<style>
  body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, 'Segoe UI', sans-serif;"""
        f""" margin: 0; padding: 1.5rem; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));"""
        f""" gap: 1rem; margin: 1rem 0 2rem; }}
  .kpi {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem; }}
  .kpi .label {{ font-size: 0.8rem; color: #8b949e; text-transform: uppercase; }}
  .kpi .value {{ font-size: 1.4rem; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; background: #161b22; border: 1px solid #30363d; }}
  th, td {{ padding: 0.6rem 0.8rem; border-bottom: 1px solid #30363d; text-align: left; }}
  th {{ background: #21262d; font-size: 0.8rem; text-transform: uppercase; }}
  .num {{ font-variant-numeric: tabular-nums; text-align: right; }}
  .gate-fail {{ color: #f85149; font-weight: 500; }}
  .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px;"""
        f""" font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }}
  .badge.champion {{ background: rgba(137, 87, 229, 0.2); color: #a371f7; border: 1px solid #8957e5; }}
  .badge.ready {{ background: rgba(46, 160, 67, 0.2); color: #3fb950; border: 1px solid #2ea043; }}
  .badge.research {{ background: rgba(110, 118, 129, 0.2); color: #8b949e; border: 1px solid #30363d; }}
  .badge.hurdle {{ background: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid #da3633; }}
  .badge.benchmark {{ background: rgba(137, 87, 229, 0.12); color: #a371f7; border: 1px solid #30363d; }}
  details {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;"""
        f""" padding: 0.5rem 1rem; margin: 0.5rem 0; }}
  summary {{ cursor: pointer; }}
  pre {{ white-space: pre-wrap; font-size: 0.75rem; }}
  h1, h2 {{ color: #e6edf3; }}
</style>
<!-- plotly-engine-embed -->
<script>{engine_js}</script>
</head>
<body>
<h1>🏆 NumerAI Executive Performance Report</h1>
<p>Evaluation window: {kpis['n_eras']} overlap eras · data version {kpis['data_version']}</p>
<div class="kpis">
  <div class="kpi"><div class="label">Active Champion</div><div class="value">{html.escape(kpis['champion_label'])}"""
        f"""</div><div>{html.escape(kpis['champion_detail'])}</div></div>
  <div class="kpi"><div class="label">Top Research Contender</div>"""
        f"""<div class="value">{html.escape(kpis['top_contender_label'])}</div><div>Sharpe"""
        f""" {_fmt(kpis['top_contender_sharpe'])} vs hurdle {_fmt(kpis['hurdle_sharpe'])}</div></div>
  <div class="kpi"><div class="label">Fleet Best Return (CAGR)</div>"""
        f"""<div class="value">{_fmt(kpis['fleet_best_cagr'], pct=True)}</div></div>
  <div class="kpi"><div class="label">Worst Fleet Drawdown</div>"""
        f"""<div class="value">{_fmt(kpis['worst_drawdown'], pct=True)}</div></div>
  <div class="kpi"><div class="label">Capital Readiness</div>"""
        f"""<div class="value">{kpis['capital_ready_count']} / {kpis['fleet_count']}</div></div>
</div>
<h2>1. Cumulative Wealth &amp; Downside Protection</h2>
{figure_html['wealth']}
<h2>2. Risk-Adjusted Return Leaderboard</h2>
{figure_html['leaderboard']}
<h2>3. Executive Allocation &amp; Risk Decision Table</h2>
<table>
<thead><tr><th>Status</th><th>Model</th><th>Ann. Return</th><th>Sharpe (AC)</th><th>Sharpe CI</th><th>Max DD</th>"""
        f"""<th>Gain-to-Pain</th><th>Downside</th><th>Confidence (DSR)</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
<h2>4. Underwater Drawdown</h2>
{figure_html['drawdown']}
<h2>Technical &amp; Audit Metadata</h2>
{accordion}
</body>
</html>"""
    )


def generate_dashboard(
    *,
    registry_dir: Path | None = None,
    benchmark_path: Path | None | bool = None,
    output_path: Path | None = None,
    open_browser: bool = True,
) -> Path:
    """Build the executive HTML report and write it to disk."""
    registry_dir = Path(registry_dir) if registry_dir is not None else DEFAULT_REGISTRY_DIR
    output_path = Path(output_path) if output_path is not None else REPO_ROOT / "artifacts" / "dashboard.html"

    leaderboard = load_unified_leaderboard(registry_dir, benchmark_path=benchmark_path)
    leaderboard = reconcile_capital_metrics(leaderboard, registry_dir, DEFAULT_DATA_DIR)
    statuses = evaluate_gate_status(leaderboard, DEFAULT_GATE_PATH, registry_dir / "champion.json")
    leaderboard = leaderboard.join(statuses, on="model_id", how="left")

    gate_cfg = load_benchmark_file(DEFAULT_GATE_PATH)
    assert gate_cfg.reference_column is not None
    hurdle_sharpe = float(gate_cfg.gate.corr_sharpe_ac_min)

    champion = _champion_id(registry_dir)
    fleet = leaderboard.filter(pl.col("source").is_in(["trained", "trained_legacy"]))
    top_ids = fleet.sort("corr_sharpe_ac", descending=True, nulls_last=True).head(3)
    timeseries = extract_payout_timeseries(
        registry_dir, DEFAULT_DATA_DIR,
        run_ids=top_ids.get_column("model_id").to_list(),
        include_tier4_ref=True,
        tier4_column=str(gate_cfg.reference_column),
    )
    figures = {
        "leaderboard": charts.build_leaderboard_bar_chart(
            _bar_input(leaderboard, champion), hurdle_sharpe=hurdle_sharpe
        ),
        "wealth": charts.build_cumulative_wealth_chart(timeseries),
        "drawdown": charts.build_drawdown_chart(timeseries),
    }
    html_text = _build_html(
        leaderboard=leaderboard, champion=champion,
        kpis=_kpi_cards(leaderboard, champion, hurdle_sharpe),
        figures=figures, registry_dir=registry_dir,
        technical_entries=_technical_entries(registry_dir),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    if open_browser:
        webbrowser.open(output_path.as_uri())
    return output_path


def main() -> int:
    output = generate_dashboard()
    print(f"Dashboard written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
