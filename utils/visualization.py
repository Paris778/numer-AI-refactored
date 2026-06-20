"""Visualization utilities aligned to utils.metrics canonical output keys."""

from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display

# ---------------------------------------------------------------------------
#    Global Configuration & Goals
# ---------------------------------------------------------------------------

GOALS = {
    "CORR": 0.022,
    "MMC": 0.01,
    "ESTIMATED_RETURN_PCT": 25.0,
    "SHARPE": 1.0,
    "RAPS": 0.10,
    "PAYOUT_SHARPE": 1.25,
    "MAX_DRAWDOWN": -0.10,
    "WIN_RATE": 0.85,
    "FNC": 0.010,
    "PAYOUT_SORTINO": 1.0,
    "FEATURE_EXPOSURE_P95": 0.10,
    "MAX_FEATURE_EXPOSURE": 0.15,
    "BENCHMARK_CORR_ABS": 0.10,
}

METRIC_NOTES = {
    "Estimated Return": "Approximate payout percent using 0.75× CORR + 2.25× MMC, capped at +/-5%",
    "Mean MMC (BMC Proxy)": "Uniqueness vs benchmark with the current 2.25× payout weight proxy",
    "RAPS": "Risk-adjusted payout proxy with drawdown and tail-risk penalties",
    "Mean CORR": "Official Numerai correlation score; strong models are often 0.01-0.03",
    "Sharpe Ratio": "Risk-adjusted return consistency",
    "Payout Sharpe": "Sharpe ratio of payout proxy (0.75× CORR + 2.25× MMC)",
    "MMC Volatility": "Standard deviation of era-by-era MMC",
    "Max Drawdown": "Worst sustained CORR cumulative loss",
    "Mean FNC": "CORR after removing linear feature exposure",
    "Win Rate": "Fraction of eras with positive CORR",
    "P95 Max Feat Exposure": "95th percentile of strongest single-feature correlation",
    "|Benchmark Corr|": "Average absolute similarity vs benchmark predictions",
    "Max Feature Exposure": "Absolute peak single-feature correlation across eras",
    "SNR (Mean/Std)": "Signal-to-Noise Ratio. Higher means flatter stable performance.",
    "Std Dev (σ)": "Direct smoothness measure. Lower means less oscillation.",
    "Autocorrelation": "Temporal consistency. Higher means eras perform similarly consecutively.",
}

STATUS_STYLES = {
    "🟣 ABOVE TARGET !!!": "background-color:#6A0DAD; color:#FFFFFF; font-weight:bold",
    "🟢 MET": "background-color:#1a3d2b; color:#00FF7F; font-weight:bold",
    "🟡 CLOSE": "background-color:#3d3500; color:#F1C40F; font-weight:bold",
    "🟠 LAGGING": "background-color:#3d2000; color:#F39C12; font-weight:bold",
    "🔴 BELOW": "background-color:#3d0a0a; color:#E74C3C; font-weight:bold",
    "🔴 EXCEEDED": "background-color:#3d0a0a; color:#E74C3C; font-weight:bold",
    "—": "",
}

DARK_THEME = {
    "figure_face": "#0f1115",
    "axes_face": "#171a21",
    "grid": "#2a2f3a",
    "text": "#e6e8ee",
    "muted": "#b7bfcc",
    "spine": "#5b6575",
}

# ---------------------------------------------------------------------------
#    Internal Helpers
# ---------------------------------------------------------------------------


def _get_metric(
    metrics: dict[str, Any],
    key: str,
    *,
    required: bool = False,
) -> float | None:
    value = metrics.get(key)
    if value is None or isinstance(value, str) or pd.isna(value):
        if required:
            raise KeyError(
                f"Required metric '{key}' is missing or invalid. "
                "Pass the unmodified output of utils.metrics.calculate_metrics."
            )
        return None
    return float(value)


def _pct_to_color(pct: float) -> str:
    anchors = [
        (-1.0, "#000000"),
        (0.0, "#C41D0A"),
        (0.5, "#F1930F"),
        (0.75, "#ADFF2F"),
        (1.0, "#2ECC71"),
        (1.0001, "#AF2AC9"),
        (1.25, "#852598"),
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pct_grad", [c for _, c in anchors], N=256
    )
    return mcolors.to_hex(cmap(np.clip(pct / 110.0, 0.0, 1.0)))


def _status(raw: float | None, goal: float | None, hib: bool) -> str:
    if raw is None or goal is None:
        return "—"
    delta = raw - goal if hib else goal - raw
    if goal == 0:
        return (
            "🟢 MET"
            if delta == 0
            else ("🟣 ABOVE TARGET !!!" if delta > 0 else "🔴 BELOW")
        )

    margin = delta / abs(goal)
    if margin >= 0.25:
        return "🟣 ABOVE TARGET !!!"
    if delta >= 0:
        return "🟢 MET"

    shortfall = abs(margin)
    if shortfall <= 0.25:
        return "🟡 CLOSE"
    if shortfall <= 0.50:
        return "🟠 LAGGING"
    return "🔴 BELOW" if hib else "🔴 EXCEEDED"


def _fmt_val(val: float | None, label: str, signed: bool = True) -> str:
    if val is None:
        return "—"
    if "Return" in label:
        return f"{val:+.2f}%"
    if "Win Rate" in label:
        return f"{val:.1%}" if val <= 1 else f"{val:.1f}%"
    dec = 5 if any(x in label for x in ["CORR", "MMC", "FNC", "Std", "Drawdown"]) else 4
    return f"{val:+.{dec}f}" if signed else f"{val:.{dec}f}"


def _set_dark_plot_theme() -> None:
    sns.set_theme(
        style="darkgrid",
        rc={
            "figure.facecolor": DARK_THEME["figure_face"],
            "axes.facecolor": DARK_THEME["axes_face"],
            "axes.edgecolor": DARK_THEME["spine"],
            "axes.labelcolor": DARK_THEME["text"],
            "axes.titlecolor": DARK_THEME["text"],
            "xtick.color": DARK_THEME["text"],
            "ytick.color": DARK_THEME["text"],
            "grid.color": DARK_THEME["grid"],
            "text.color": DARK_THEME["text"],
            "savefig.facecolor": DARK_THEME["figure_face"],
            "savefig.edgecolor": DARK_THEME["figure_face"],
        },
    )


def _apply_dark_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(DARK_THEME["axes_face"])
    ax.tick_params(colors=DARK_THEME["text"])
    ax.xaxis.label.set_color(DARK_THEME["text"])
    ax.yaxis.label.set_color(DARK_THEME["text"])
    ax.title.set_color(DARK_THEME["text"])
    ax.grid(color=DARK_THEME["grid"], alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color(DARK_THEME["spine"])


def _style_legend(ax: plt.Axes) -> None:
    legend = ax.get_legend()
    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor("#1f2430")
        frame.set_edgecolor(DARK_THEME["spine"])
        frame.set_alpha(0.95)
        for txt in legend.get_texts():
            txt.set_color(DARK_THEME["text"])


# ---------------------------------------------------------------------------
#    Public API
# ---------------------------------------------------------------------------


def _goal_gap_pct(
    value: float | None, goal: float | None, higher_is_better: bool
) -> float | None:
    """Signed percent gap from goal. Positive means better-than-goal."""
    if value is None or goal is None or goal == 0:
        return None
    if higher_is_better:
        return ((value - goal) / abs(goal)) * 100.0
    return ((goal - value) / abs(goal)) * 100.0


def _goal_progress_pct(
    value: float | None,
    goal: float | None,
    higher_is_better: bool,
    *,
    clip_low: float = -10.0,
    clip_high: float = 200.0,
) -> float | None:
    """
    Goal attainment as percent where 100 = at goal.
    Handles positive and negative goals robustly.
    """
    if value is None or goal is None or goal == 0:
        return None

    if higher_is_better:
        if goal > 0:
            raw = (value / goal) * 100.0
        else:
            # For negative goals (example: drawdown threshold), goal is met at 100.
            # Better-than-goal (less negative / more positive) rises above 100.
            raw = (1.0 + (value - goal) / abs(goal)) * 100.0
    else:
        if goal > 0:
            denom = value if value > 0 else np.nan
            raw = (goal / denom) * 100.0 if np.isfinite(denom) else 0.0
        else:
            denom = abs(value) if value != 0 else np.nan
            raw = (abs(goal) / denom) * 100.0 if np.isfinite(denom) else 0.0

    return float(np.clip(raw, clip_low, clip_high))


def display_metrics_table(metrics: dict[str, Any], extended: bool = False) -> None:
    """Institutional-grade dashboard for tournament quality, risk, and objective attainment."""
    _set_dark_plot_theme()
    print(
        "=" * 90
        + "\n"
        + "KEY PERFORMANCE METRICS vs GOALS".center(90)
        + "\n"
        + "=" * 90
    )

    # Fallback-aware key selection to stay robust with alias variations.
    def _first_metric(*keys: str) -> float | None:
        for k in keys:
            v = _get_metric(metrics, k, required=False)
            if v is not None:
                return v
        return None

    # World-class panel: objective, robustness, uniqueness, and quality.
    metric_spec = [
        {
            "label": "Estimated Return",
            "category": "Payout",
            "keys": ["9_Annualized_Return_PCT"],
            "goal_key": "ESTIMATED_RETURN_PCT",
            "higher_is_better": True,
            "weight": 0.20,
            "core": True,
        },
        {
            "label": "RAPS",
            "category": "Payout",
            "keys": ["1_RAPS"],
            "goal_key": "RAPS",
            "higher_is_better": True,
            "weight": 0.20,
            "core": True,
        },
        {
            "label": "Mean CORR",
            "category": "Payout",
            "keys": ["3_Mean_CORR20V2"],
            "goal_key": "CORR",
            "higher_is_better": True,
            "weight": 0.15,
            "core": True,
        },
        {
            "label": "Mean MMC (BMC Proxy)",
            "category": "Payout",
            "keys": ["4_Mean_BMC20", "4_Mean_MMC20"],
            "goal_key": "MMC",
            "higher_is_better": True,
            "weight": 0.15,
            "core": True,
        },
        {
            "label": "Sharpe Ratio",
            "category": "Risk",
            "keys": ["5_Sharpe_CORR"],
            "goal_key": "SHARPE",
            "higher_is_better": True,
            "weight": 0.10,
            "core": True,
        },
        {
            "label": "Payout Sharpe",
            "category": "Risk",
            "keys": ["2_Sharpe_Payout"],
            "goal_key": "PAYOUT_SHARPE",
            "higher_is_better": True,
            "weight": 0.07,
            "core": False,
        },
        {
            "label": "Max Drawdown",
            "category": "Risk",
            "keys": ["7_Max_Drawdown_CORR"],
            "goal_key": "MAX_DRAWDOWN",
            "higher_is_better": True,
            "weight": 0.05,
            "core": False,
        },
        {
            "label": "Win Rate",
            "category": "Quality",
            "keys": ["8_Win_Rate"],
            "goal_key": "WIN_RATE",
            "higher_is_better": True,
            "weight": 0.03,
            "core": False,
        },
        {
            "label": "Mean FNC",
            "category": "Quality",
            "keys": ["6_Mean_FNC"],
            "goal_key": "FNC",
            "higher_is_better": True,
            "weight": 0.03,
            "core": False,
        },
        {
            "label": "|Benchmark Corr|",
            "category": "Uniqueness",
            "keys": ["10_Benchmark_Corr"],
            "goal_key": "BENCHMARK_CORR_ABS",
            "higher_is_better": False,
            "weight": 0.01,
            "core": False,
        },
        {
            "label": "MMC Volatility",
            "category": "Risk",
            "keys": ["11_BMC_Volatility", "11_MMC_Volatility"],
            "goal_key": None,
            "higher_is_better": False,
            "weight": 0.0,
            "core": False,
        },
    ]

    if extended:
        metric_spec.extend(
            [
                {
                    "label": "P95 Max Feat Exposure",
                    "category": "Uniqueness",
                    "keys": ["9_Feature_Exposure_P95"],
                    "goal_key": "FEATURE_EXPOSURE_P95",
                    "higher_is_better": False,
                    "weight": 0.0,
                    "core": False,
                },
                {
                    "label": "Max Feature Exposure",
                    "category": "Uniqueness",
                    "keys": ["12_Max_Feature_Exposure"],
                    "goal_key": "MAX_FEATURE_EXPOSURE",
                    "higher_is_better": False,
                    "weight": 0.0,
                    "core": False,
                },
                {
                    "label": "SNR (Mean/Std)",
                    "category": "Smoothness",
                    "keys": ["13_MMC_SNR"],
                    "goal_key": None,
                    "higher_is_better": True,
                    "weight": 0.0,
                    "core": False,
                },
                {
                    "label": "Std Dev (sigma)",
                    "category": "Smoothness",
                    "keys": ["14_MMC_Std"],
                    "goal_key": None,
                    "higher_is_better": False,
                    "weight": 0.0,
                    "core": False,
                },
                {
                    "label": "Autocorrelation",
                    "category": "Smoothness",
                    "keys": ["15_MMC_Autocorr"],
                    "goal_key": None,
                    "higher_is_better": True,
                    "weight": 0.0,
                    "core": False,
                },
            ]
        )

    rows: list[dict[str, Any]] = []
    for spec in metric_spec:
        value = _first_metric(*spec["keys"])
        goal = GOALS.get(spec["goal_key"]) if spec["goal_key"] else None
        gap_pct = _goal_gap_pct(value, goal, spec["higher_is_better"])
        progress_pct = _goal_progress_pct(value, goal, spec["higher_is_better"])
        status = _status(value, goal, spec["higher_is_better"])
        rows.append(
            {
                "label": spec["label"],
                "category": spec["category"],
                "value": value,
                "goal": goal,
                "gap_pct": gap_pct,
                "progress_pct": progress_pct,
                "status": status,
                "weight": float(spec["weight"]),
                "core": bool(spec["core"]),
            }
        )

    core_rows = [r for r in rows if r["core"] and r["progress_pct"] is not None]
    if core_rows:
        total_w = sum(r["weight"] for r in core_rows) or 1.0
        weighted_progress = (
            sum(r["progress_pct"] * r["weight"] for r in core_rows) / total_w
        )
        score_0_100 = float(np.clip(weighted_progress, 0.0, 150.0) / 1.5)
    else:
        weighted_progress = 0.0
        score_0_100 = 0.0

    if weighted_progress >= 110:
        regime = "OUTPERFORMING"
    elif weighted_progress >= 95:
        regime = "ON TRACK"
    elif weighted_progress >= 80:
        regime = "WATCHLIST"
    else:
        regime = "HIGH RISK"

    # --- Charts ---
    chart_rows = [r for r in rows if r["core"]]
    names = [r["label"] for r in chart_rows]
    progress = [
        float(r["progress_pct"] if r["progress_pct"] is not None else 0.0)
        for r in chart_rows
    ]
    gaps = [
        float(r["gap_pct"] if r["gap_pct"] is not None else 0.0) for r in chart_rows
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(20, 8.5), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )
    fig.patch.set_facecolor(DARK_THEME["figure_face"])
    y = np.arange(len(names))

    # 1) Bullet-style attainment chart (readable replacement for signed gap bars)
    for yi in y:
        ax1.barh(
            yi,
            10,
            left=-10,
            height=0.72,
            color="#341313",
            alpha=0.45,
            edgecolor="none",
        )
        ax1.barh(
            yi, 70, left=0, height=0.72, color="#4a1f1f", alpha=0.45, edgecolor="none"
        )
        ax1.barh(
            yi, 25, left=70, height=0.72, color="#5a4b16", alpha=0.45, edgecolor="none"
        )
        ax1.barh(
            yi, 10, left=95, height=0.72, color="#1d4d33", alpha=0.50, edgecolor="none"
        )
        ax1.barh(
            yi, 45, left=105, height=0.72, color="#3a2a5a", alpha=0.45, edgecolor="none"
        )

    bar_colors = [_pct_to_color(p) for p in progress]
    ax1.barh(
        y,
        [min(150.0, p) for p in progress],
        height=0.34,
        color=bar_colors,
        alpha=0.95,
        edgecolor="#000000",
    )
    ax1.axvline(
        100,
        color="#7CFC00",
        linestyle="--",
        linewidth=1.4,
        alpha=0.9,
        label="Goal = 100%",
    )

    for yi, p in zip(y, progress):
        x_txt = p + 2.0 if p >= 0 else p - 1.5
        x_txt = min(151.0, max(-9.5, x_txt))
        txt_align = "left" if p >= 0 else "right"
        ax1.text(
            x_txt,
            yi,
            f"{p:,.0f}%",
            va="center",
            ha=txt_align,
            color=DARK_THEME["text"],
            fontsize=11,
            fontweight="bold",
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontweight="bold", fontsize=12)
    ax1.set_xlim(-10, 160)
    ax1.set_xlabel("Goal Attainment (%)", fontsize=13, fontweight="bold")
    ax1.set_title(
        "Bullet View: Metric Attainment vs Goal", fontweight="bold", fontsize=15
    )
    ax1.tick_params(axis="x", labelsize=11)
    _apply_dark_axes(ax1)
    _style_legend(ax1)

    # 2) Goal gap chart in the same metric order as the left panel.
    y_gap = np.arange(len(names))
    ax2.barh(
        y_gap,
        gaps,
        color=["#C41D0A" if g < 0 else "#2ECC71" for g in gaps],
        alpha=0.9,
        edgecolor="#000000",
    )
    ax2.axvline(0, color="white", linestyle="--", alpha=0.7)
    ax2.set_yticks(y_gap)
    ax2.set_yticklabels(names, fontweight="bold", fontsize=12)
    ax2.set_title("Goal Gap (%)", fontweight="bold", fontsize=15)
    ax2.set_xlabel("Positive = Above Goal", fontsize=13, fontweight="bold")
    ax2.tick_params(axis="x", labelsize=11)
    _apply_dark_axes(ax2)

    fig.suptitle(
        f"Composite Score: {score_0_100:.1f}/100   |   Regime: {regime}   |   Weighted Progress: {weighted_progress:.1f}%",
        fontsize=14,
        fontweight="bold",
        color=DARK_THEME["text"],
        y=1.01,
    )

    plt.tight_layout()
    plt.show()

    # --- Rich HTML Table ---
    # Keep Estimated Return first, then sort remaining rows by category and severity.
    def _severity_key(r: dict[str, Any]) -> float:
        g = r["gap_pct"]
        return 1e9 if g is None else g

    estimated_return_rows = [r for r in rows if r["label"] == "Estimated Return"]
    remaining_rows = [r for r in rows if r["label"] != "Estimated Return"]
    ordered_rows = estimated_return_rows + sorted(
        remaining_rows, key=lambda r: (r["category"], _severity_key(r))
    )

    rows_html = []
    for r in ordered_rows:
        gap_str = "—" if r["gap_pct"] is None else f"{r['gap_pct']:+.1f}%"
        prog_str = "—" if r["progress_pct"] is None else f"{r['progress_pct']:.1f}%"
        rows_html.append(
            f"""
            <tr>
                <td style='color:#9aa4b2; font-style:italic; border:1px solid #333; padding:6px'>{r['category']}</td>
                <td style='border:1px solid #333; padding:6px'>{r['label']}</td>
                <td style='border:1px solid #333; padding:6px; font-weight:bold'>{_fmt_val(r['value'], r['label'])}</td>
                <td style='border:1px solid #333; padding:6px'>{_fmt_val(r['goal'], r['label'], True)}</td>
                <td style='border:1px solid #333; padding:6px; color:{"#2ECC71" if (r["gap_pct"] or -1) >= 0 else "#E74C3C"}'>{gap_str}</td>
                <td style='border:1px solid #333; padding:6px'>{prog_str}</td>
                <td style='border:1px solid #333; padding:6px; {STATUS_STYLES.get(r["status"], "")}'>{r['status']}</td>
                <td style='border:1px solid #333; padding:6px; font-size:11px'>{METRIC_NOTES.get(r['label'], '')}</td>
            </tr>
            """
        )

    # Headline diagnostics
    misses = [r for r in rows if r["gap_pct"] is not None and r["gap_pct"] < 0]
    wins = [r for r in rows if r["gap_pct"] is not None and r["gap_pct"] > 0]
    top_miss = (
        sorted(misses, key=lambda r: r["gap_pct"])[0]["label"] if misses else "None"
    )
    top_win = (
        sorted(wins, key=lambda r: r["gap_pct"], reverse=True)[0]["label"]
        if wins
        else "None"
    )

    display(
        HTML(
            f"""
        <div style='margin:6px 0 10px 0; padding:10px; border:1px solid #2d3440; background:#131722; color:#d0d7e3; font-family:monospace'>
            <b>Portfolio Diagnostic</b><br>
            Composite Score: <b>{score_0_100:.1f}/100</b> |
            Regime: <b>{regime}</b> |
            Best Relative Metric: <b>{top_win}</b> |
            Largest Shortfall: <b>{top_miss}</b>
        </div>
        <table style='border-collapse:collapse; width:100%; font-family:monospace; background:#11151d; color:#d0d7e3'>
            <caption style='color:#f1f4f9; font-weight:bold; padding:10px; text-align:left'>
                CORE TOURNAMENT METRICS - Institutional Dashboard
            </caption>
            <thead style='background:#1a2233; color:#f1f4f9'>
                <tr>
                    <th style='padding:6px; border:1px solid #333'>Category</th>
                    <th style='padding:6px; border:1px solid #333'>Metric</th>
                    <th style='padding:6px; border:1px solid #333'>Value</th>
                    <th style='padding:6px; border:1px solid #333'>Goal</th>
                    <th style='padding:6px; border:1px solid #333'>Gap %</th>
                    <th style='padding:6px; border:1px solid #333'>Progress %</th>
                    <th style='padding:6px; border:1px solid #333'>Status</th>
                    <th style='padding:6px; border:1px solid #333'>Interpretation</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
        """
        )
    )


def _parse_era_axis(index: pd.Index) -> np.ndarray:
    """Parse mixed-format era labels into numeric x-axis values."""
    parsed_values: list[int] = []
    for position, value in enumerate(index):
        if isinstance(value, (int, np.integer)):
            parsed_values.append(int(value))
            continue

        text = str(value)
        digits = "".join(character for character in text if character.isdigit())
        if digits:
            parsed_values.append(int(digits))
            continue

        return np.arange(len(index), dtype=np.int64)

    return np.asarray(parsed_values, dtype=np.int64)


def plot_metric_over_time(
    per_era_df: pd.DataFrame,
    metric: str = "BMC20",
    title: str | None = None,
) -> None:
    """Institutional time-series dashboard with compact graph-native diagnostics."""
    _set_dark_plot_theme()
    if metric not in per_era_df.columns:
        raise KeyError(
            f"Metric column '{metric}' is missing from per_era_df. "
            f"Available columns: {list(per_era_df.columns)}"
        )

    s = per_era_df[metric].dropna()
    if s.empty:
        return

    # Era alignment
    x_vals = _parse_era_axis(s.index)

    cum = s.cumsum()
    peak = cum.cummax()
    dd = peak - cum

    # Stats Calculations
    mean, std = float(s.mean()), float(s.std(ddof=0))
    metric_snr = mean / std if std != 0 else 0
    win_rate = float((s > 0).mean())
    autocorr = float(s.autocorr(lag=1)) if len(s) > 1 else 0
    max_dd = float(dd.max())
    max_dd_signed = -max_dd
    ulcer = float(np.sqrt(np.mean(np.square(dd.to_numpy(dtype=float)))))

    fig = plt.figure(figsize=(20, 10.5))
    fig.patch.set_facecolor(DARK_THEME["figure_face"])
    gs = fig.add_gridspec(2, 2, width_ratios=[5.4, 2.2], hspace=0.26, wspace=0.16)
    ax_m = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0], sharex=ax_m)
    ax_kpi = fig.add_subplot(gs[:, 1])

    ax_m.plot(x_vals, s, color="#58A6FF", lw=2.4, label=f"Per-Era {metric}")
    ax_m.axhline(0, color="#700909", ls="--", alpha=0.5, lw=1.75)
    ax_m.set_title(title or f"{metric} Over Time", fontweight="bold", fontsize=16)
    ax_m.set_ylabel(metric, fontsize=12, fontweight="bold")
    ax_m.tick_params(axis="both", labelsize=11)
    _apply_dark_axes(ax_m)
    _style_legend(ax_m)

    ax_c.plot(x_vals, cum, color="#2ECC71", lw=2.5, label="Cumulative")
    ax_c.plot(x_vals, peak, color="#95A5A6", ls=":", lw=1.8, alpha=0.9, label="Peak")
    ax_c.fill_between(
        x_vals,
        cum,
        peak,
        where=dd > 0,
        color="#E74C3C",
        alpha=0.18,
        label="Drawdown",
    )
    ax_c.set_title(f"Cumulative {metric} Path", fontweight="bold", fontsize=16)
    ax_c.set_xlabel("Era", fontsize=12, fontweight="bold")
    ax_c.set_ylabel("Cumulative", fontsize=12, fontweight="bold")
    ax_c.tick_params(axis="both", labelsize=11)
    ax_c.axhline(0, color="#700909", ls="--", alpha=0.5, lw=1.75)
    _apply_dark_axes(ax_c)
    _style_legend(ax_c)

    ax_kpi.axis("off")
    ax_kpi.set_facecolor(DARK_THEME["axes_face"])

    ax_kpi.text(
        0.02,
        0.98,
        "Signal Dashboard",
        va="top",
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=DARK_THEME["text"],
    )

    kpi_cards = [
        ("SNR (Mean/Std)", f"{metric_snr:.3f}"),
        ("Mean", f"{mean:+.5f}"),
        ("Std Dev", f"{std:.5f}"),
        ("Win Rate", f"{win_rate:.1%}"),
        ("Autocorr (lag1)", f"{autocorr:+.3f}"),
    ]

    start_y = 0.90
    card_step = 0.145
    for i, (kpi_label, kpi_val) in enumerate(kpi_cards):
        y_pos = start_y - i * card_step
        ax_kpi.text(
            0.03,
            y_pos,
            f"{kpi_label}\n{kpi_val}",
            va="top",
            ha="left",
            fontsize=12,
            fontfamily="monospace",
            linespacing=1.35,
            color=DARK_THEME["text"],
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="#1f2430",
                edgecolor=DARK_THEME["spine"],
                linewidth=1.1,
            ),
        )

    footer_text = (
        "Risk Footer\n"
        f"Max Drawdown: {max_dd_signed:+.4f}\n"
        f"Ulcer Index : {ulcer:.4f}"
    )
    ax_kpi.text(
        0.03,
        0.05,
        footer_text,
        va="bottom",
        ha="left",
        fontsize=11,
        fontfamily="monospace",
        linespacing=1.35,
        color=DARK_THEME["text"],
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#1b2230",
            edgecolor=DARK_THEME["spine"],
            linewidth=1.1,
        ),
    )

    fig.suptitle(
        f"{metric} Time-Series Review | Eras: {len(s)} | Terminal Cumulative: {float(cum.iloc[-1]):+.4f}",
        fontsize=15,
        fontweight="bold",
        color=DARK_THEME["text"],
        y=0.99,
    )

    plt.tight_layout()
    plt.show()
