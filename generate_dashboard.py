"""Generate an HTML leaderboard dashboard from trained runs and benchmarks."""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

import pandas as pd
import polars as pl

from nmr.config import REPO_ROOT


def _load_registry_runs(registry_dir: Path) -> pd.DataFrame:
    """Load all runs from the registry into a tidy DataFrame."""
    rows: list[dict] = []
    for run_file in sorted(registry_dir.glob("*/run.json")):
        payload = json.loads(run_file.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        manifest = payload.get("manifest", {})
        cfg = manifest.get("config", {})
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        run_cfg = cfg.get("run", {})

        rows.append(
            {
                "model_id": payload.get("run_id", run_file.parent.name),
                "source": "trained",
                "run_name": run_cfg.get("name", "unknown"),
                "feature_set": data_cfg.get("feature_set", "unknown"),
                "backend": model_cfg.get("backend", "unknown"),
                "preset": model_cfg.get("preset", "unknown"),
                "n_targets": len(data_cfg.get("targets", [])),
                "targets": ", ".join(data_cfg.get("targets", [])),
                "mean": float(metrics.get("mean", 0.0)),
                "std": float(metrics.get("std", 0.0)),
                "sharpe": float(metrics.get("sharpe", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "artifact_path": payload.get("artifact_path"),
                "run_dir": str(run_file.parent),
            }
        )
    return pd.DataFrame(rows)


def _load_benchmarks(path: Path) -> pd.DataFrame:
    """Load benchmark_scores.csv and normalize columns to the dashboard schema."""
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    rename = {
        "model_id": "model_id",
        "corr": "mean",
        "corr_sharpe_ac": "sharpe",
    }
    df = df.rename(columns=rename)
    df["source"] = "benchmark"
    df["run_name"] = df.get("strategy_group", "benchmark")
    df["feature_set"] = "all"
    df["backend"] = "benchmark"
    df["preset"] = "benchmark"
    df["n_targets"] = 1
    df["targets"] = df.get("horizon_target_name", "cyrusd")
    df["std"] = df.get("std_corr", 0.0)
    df["max_drawdown"] = df.get("max_drawdown", 0.0)
    df["artifact_path"] = None
    df["run_dir"] = str(path)

    keep = [
        "model_id",
        "source",
        "run_name",
        "feature_set",
        "backend",
        "preset",
        "n_targets",
        "targets",
        "mean",
        "std",
        "sharpe",
        "max_drawdown",
        "artifact_path",
        "run_dir",
    ]
    return df[[col for col in keep if col in df.columns]].copy()


def _rank_models(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by Sharpe and add a rank column."""
    df = df.copy()
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def _format_value(value: float | None, fmt: str = ".5f") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:{fmt}}"


def _build_html(df: pd.DataFrame, benchmark_path: Path, registry_dir: Path) -> str:
    """Render the leaderboard as a self-contained HTML page."""
    rows_html = ""
    for _, row in df.iterrows():
        badge_class = "trained" if row["source"] == "trained" else "benchmark"
        rows_html += f"""
        <tr>
          <td class="rank">{int(row['rank'])}</td>
          <td class="model-id" title="{row['model_id']}">{row['model_id'][:16]}</td>
          <td><span class="badge {badge_class}">{row['source']}</span></td>
          <td>{row['run_name']}</td>
          <td>{row['feature_set']}</td>
          <td>{row['backend']}</td>
          <td>{row['preset']}</td>
          <td>{int(row['n_targets'])}</td>
          <td class="num">{_format_value(row['mean'])}</td>
          <td class="num">{_format_value(row['std'])}</td>
          <td class="num sharpe">{_format_value(row['sharpe'])}</td>
          <td class="num">{_format_value(row['max_drawdown'])}</td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NumerAI Model Dashboard</title>
  <style>
    :root {{
      --bg: #0d1117;
      --surface: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --muted: #8b949e;
      --accent: #58a6ff;
      --trained: #238636;
      --benchmark: #8957e5;
      --danger: #f85149;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    header {{
      padding: 2rem 1.5rem 1rem;
      border-bottom: 1px solid var(--border);
    }}
    header h1 {{ margin: 0 0 0.5rem; font-size: 1.75rem; }}
    header p {{ margin: 0; color: var(--muted); }}
    main {{ padding: 1.5rem; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }}
    .stat-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
    }}
    .stat-card .label {{ font-size: 0.8rem; color: var(--muted); text-transform: uppercase; }}
    .stat-card .value {{ font-size: 1.5rem; font-weight: 600; margin-top: 0.25rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{
      background: #21262d;
      color: var(--muted);
      font-weight: 600;
      font-size: 0.8rem;
      text-transform: uppercase;
      position: sticky;
      top: 0;
    }}
    tr:hover {{ background: rgba(88, 166, 255, 0.08); }}
    .rank {{ font-weight: 700; color: var(--accent); }}
    .num {{ font-variant-numeric: tabular-nums; text-align: right; }}
    .sharpe {{ color: #3fb950; font-weight: 600; }}
    .model-id {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    .badge {{
      display: inline-block;
      padding: 0.15rem 0.5rem;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
    }}
    .badge.trained {{ background: rgba(35, 134, 54, 0.2); color: #3fb950; }}
    .badge.benchmark {{ background: rgba(137, 87, 229, 0.2); color: #a371f7; }}
    footer {{
      padding: 1.5rem;
      color: var(--muted);
      font-size: 0.85rem;
      border-top: 1px solid var(--border);
    }}
  </style>
</head>
<body>
  <header>
    <h1>🏆 NumerAI Model Dashboard</h1>
    <p>Trained runs from {registry_dir} plus benchmark models from {benchmark_path.name}</p>
  </header>
  <main>
    <div class="stats">
      <div class="stat-card">
        <div class="label">Total Models</div>
        <div class="value">{len(df)}</div>
      </div>
      <div class="stat-card">
        <div class="label">Trained</div>
        <div class="value">{len(df[df['source'] == 'trained'])}</div>
      </div>
      <div class="stat-card">
        <div class="label">Benchmarks</div>
        <div class="value">{len(df[df['source'] == 'benchmark'])}</div>
      </div>
      <div class="stat-card">
        <div class="label">Best Sharpe</div>
        <div class="value">{_format_value(df['sharpe'].max())}</div>
      </div>
    </div>
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Model ID</th>
          <th>Source</th>
          <th>Name</th>
          <th>Features</th>
          <th>Backend</th>
          <th>Preset</th>
          <th>Targets</th>
          <th class="num">Mean CORR</th>
          <th class="num">Std</th>
          <th class="num">Sharpe</th>
          <th class="num">Max DD</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </main>
  <footer>
    Generated from repository root: {REPO_ROOT}<br>
    Registry: {registry_dir} | Benchmarks: {benchmark_path}
  </footer>
</body>
</html>"""


def generate_dashboard(
    *,
    registry_dir: Path | None = None,
    benchmark_path: Path | None = None,
    output_path: Path | None = None,
    open_browser: bool = True,
) -> Path:
    """Build the HTML dashboard and write it to disk."""
    registry_dir = registry_dir or REPO_ROOT / "artifacts" / "registry"
    benchmark_path = benchmark_path or REPO_ROOT / "artifacts" / "benchmark_scores.csv"
    output_path = output_path or REPO_ROOT / "artifacts" / "dashboard.html"

    trained = _load_registry_runs(registry_dir)
    benchmarks = _load_benchmarks(benchmark_path)
    combined = pd.concat([trained, benchmarks], ignore_index=True)
    ranked = _rank_models(combined)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _build_html(ranked, benchmark_path, registry_dir), encoding="utf-8"
    )

    if open_browser:
        webbrowser.open(output_path.as_uri())

    return output_path


def main() -> int:
    output = generate_dashboard()
    print(f"Dashboard written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
