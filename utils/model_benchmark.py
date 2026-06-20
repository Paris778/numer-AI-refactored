from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = None
    display = None


_DEFAULT_HISTORY_PATH = (
    Path(__file__).resolve().parents[1] / "submissions" / "model_metrics_history_v1.csv"
)

_IDENTITY_COLUMNS = [
    "run_id",
    "model_name",
]

_MISC_COLUMNS = [
    "notebook_name",
    "timestamp_utc",
    "history_path",
]

_METADATA_COLUMNS = [*_IDENTITY_COLUMNS, *_MISC_COLUMNS]

_PRIORITY_METRIC_COLUMNS = [
    # Match the objective order from display_metrics_table.
    "9_Annualized_Return_PCT",
    "4_Mean_BMC20",
    "3_Mean_CORR20V2",
    "1_RAPS",
    "5_Sharpe_CORR",
    "2_Sharpe_Payout",
    "7_Max_Drawdown_CORR",
    "8_Win_Rate",
    "6_Mean_FNC",
    "10_Benchmark_Corr",
    "11_BMC_Volatility",
    "4_Mean_MMC20",
    "11_MMC_Volatility",
]

MODEL_METRICS_DF = pd.DataFrame(columns=_METADATA_COLUMNS)

_WINNER_MESSAGE = (
    "🎉🏆 NEW #1 MODEL! Your latest run now leads on annualized return, "
    "BMC, and CORR. 🚀✨"
)


def _round_sigfig(value: float, sigfigs: int = 5) -> float:
    """Round a finite float to the requested significant figures."""
    return float(f"{value:.{sigfigs}g}")


def _coerce_metric_value(value: Any) -> float:
    """Normalize metric values to numeric form for stable persistence."""
    if value is None or isinstance(value, str) or pd.isna(value):
        return float("nan")

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")

    if not np.isfinite(numeric):
        return float("nan")

    return _round_sigfig(numeric, sigfigs=5)


def _extract_metric_payload(metrics: dict[str, Any]) -> dict[str, float]:
    """Capture all raw metric keys from calculate_metrics output."""
    payload: dict[str, float] = {}
    for key, value in metrics.items():
        key_str = str(key)
        if key_str in _METADATA_COLUMNS:
            continue
        payload[key_str] = _coerce_metric_value(value)
    return payload


def _resolve_history_path(path: str | Path | None = None) -> Path:
    target = Path(path) if path is not None else _DEFAULT_HISTORY_PATH
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _metric_columns_from_df(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in _METADATA_COLUMNS]


def _order_metric_columns(metric_columns: list[str]) -> list[str]:
    """Order metrics by dashboard priority, then append any extras alphabetically."""
    seen: set[str] = set()
    ordered: list[str] = []

    metric_set = set(metric_columns)
    for column in _PRIORITY_METRIC_COLUMNS:
        if column in metric_set and column not in seen:
            ordered.append(column)
            seen.add(column)

    for column in sorted(metric_columns):
        if column not in seen:
            ordered.append(column)
            seen.add(column)

    return ordered


def _history_columns(metric_columns: list[str]) -> list[str]:
    ordered_metrics = _order_metric_columns(metric_columns)
    return [*_IDENTITY_COLUMNS, *ordered_metrics, *_MISC_COLUMNS]


def _comparison_columns(metric_columns: list[str], bmc_key: str) -> list[str]:
    """Order leaderboard metrics by comparison priority without duplicate aliases."""
    primary_columns = ["9_Annualized_Return_PCT", bmc_key, "3_Mean_CORR20V2"]
    excluded_columns = {"rank", "is_current", *primary_columns}

    # Hide the duplicate alias when BMC is present and used for ranking/display.
    if bmc_key == "4_Mean_BMC20":
        excluded_columns.add("4_Mean_MMC20")

    remaining_columns = [
        column
        for column in _order_metric_columns(metric_columns)
        if column not in excluded_columns
    ]
    return [
        *[column for column in primary_columns if column in metric_columns],
        *remaining_columns,
    ]


def _resolve_metric_columns(
    existing_df: pd.DataFrame,
    incoming_metrics: dict[str, float] | None = None,
) -> list[str]:
    existing_columns = set(_metric_columns_from_df(existing_df))
    incoming_columns = set(incoming_metrics.keys()) if incoming_metrics else set()
    return sorted(existing_columns | incoming_columns)


def _normalize_df(df: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    data = df.copy()
    ordered_columns = _history_columns(metric_columns)
    for column in ordered_columns:
        if column not in data.columns:
            data[column] = np.nan
    data = data.drop_duplicates(subset=["model_name"], keep="last")
    return data[ordered_columns]


def _ensure_history_headers(history_path: Path, metric_columns: list[str]) -> None:
    """Initialize an empty CSV with headers when missing or zero-byte."""
    if history_path.exists() and history_path.stat().st_size > 0:
        return
    pd.DataFrame(columns=_history_columns(metric_columns)).to_csv(
        history_path, index=False
    )


def _has_metric_changes(
    existing_row: pd.Series,
    new_row: dict[str, Any],
    metric_columns: list[str],
) -> bool:
    for column in metric_columns:
        existing_raw = existing_row.get(column, np.nan)
        new_raw = new_row.get(column, np.nan)
        existing_value = _coerce_metric_value(existing_raw)
        new_value = _coerce_metric_value(new_raw)

        # Treat NaN/NaN as unchanged for optional diagnostics.
        if np.isnan(existing_value) and np.isnan(new_value):
            continue
        if not np.isclose(
            existing_value, new_value, rtol=0.0, atol=0.0, equal_nan=True
        ):
            return True
    return False


def _build_result_row(
    row: pd.Series | dict[str, Any],
    status: str,
    message: str,
) -> pd.Series:
    result = pd.Series(row).copy()
    result["status"] = status
    result["message"] = message
    return result


def _leaderboard_label(column: str) -> str:
    prefix, _, remainder = column.partition("_")
    label = remainder if prefix.isdigit() and remainder else column
    return label.replace("_", " ")


def _format_leaderboard_value(column: str, value: Any) -> tuple[str, str]:
    if pd.isna(value):
        return '<span class="lb-muted">-</span>', "lb-empty"

    if column == "rank":
        rank = int(value)
        top_rank_class = f" lb-top-{rank}" if rank in {1, 2, 3} else ""
        return (
            f'<span class="lb-rank-badge{top_rank_class}">#{rank}</span>',
            "lb-rank",
        )

    if column == "run_id":
        return f'<span class="lb-run-id">{escape(str(value))}</span>', "lb-meta"

    if column == "timestamp_utc":
        return (
            f'<span class="lb-timestamp">{escape(str(value))}</span>',
            "lb-meta",
        )

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return escape(str(value)), ""

    if not np.isfinite(numeric):
        return '<span class="lb-muted">-</span>', "lb-empty"

    sign_class = "lb-neutral"
    if numeric > 0:
        sign_class = "lb-positive"
    elif numeric < 0:
        sign_class = "lb-negative"

    suffix = "%" if column == "9_Annualized_Return_PCT" else ""
    return f"{numeric:+.5g}{suffix}", sign_class


def _build_leaderboard_html(leaderboard: pd.DataFrame, is_top_performer: bool) -> str:
    display_columns = [
        column for column in leaderboard.columns if column != "is_current"
    ]
    current_rows = leaderboard[leaderboard["is_current"]]
    current_rank_text = ""
    if not current_rows.empty:
        current_rank_text = f" Current run rank: #{int(current_rows.iloc[0]['rank'])}."

    header_html = "".join(
        f"<th>{escape(_leaderboard_label(column))}</th>" for column in display_columns
    )

    body_rows: list[str] = []
    for _, row in leaderboard.iterrows():
        is_current = bool(row.get("is_current", False))
        row_class = "lb-current-row" if is_current else ""
        current_attr = "true" if is_current else "false"
        cells: list[str] = []

        for column in display_columns:
            cell_html, cell_class = _format_leaderboard_value(column, row[column])

            if column == "model_name":
                badges: list[str] = []
                rank = int(row["rank"])
                if rank <= 3:
                    badges.append(
                        f'<span class="lb-mini-badge lb-top-badge lb-top-{rank}">TOP {rank}</span>'
                    )
                if is_current:
                    badges.append(
                        '<span class="lb-mini-badge lb-current-badge">CURRENT RUN</span>'
                    )
                cell_html = (
                    f'<span class="lb-model-name">{escape(str(row[column]))}</span>'
                    f"{''.join(badges)}"
                )
                cell_class = "lb-model-cell"

            class_attr = f' class="{cell_class}"' if cell_class else ""
            cells.append(f"<td{class_attr}>{cell_html}</td>")

        body_rows.append(
            f"<tr class=\"{row_class}\" data-current=\"{current_attr}\">{''.join(cells)}</tr>"
        )

    banner_html = ""
    if is_top_performer:
        banner_html = (
            '<div class="lb-winner-banner">' f"{escape(_WINNER_MESSAGE)}" "</div>"
        )

    return (
        '<div class="numerai-leaderboard">'
        "<style>"
        ".numerai-leaderboard{font-family:Segoe UI,Helvetica,Arial,sans-serif;"
        "background:linear-gradient(180deg,#0f1723 0%,#0a1018 100%);"
        "color:#e8eef7;border:1px solid #253246;border-radius:16px;padding:16px;"
        "box-shadow:0 18px 48px rgba(0,0,0,.28);overflow:auto;margin:10px 0;}"
        ".lb-winner-banner{padding:12px 14px;border-radius:12px;"
        "background:linear-gradient(90deg,#136f63,#1b9aaa,#e9c46a);"
        "color:#081018;font-weight:800;font-size:15px;margin-bottom:12px;"
        "text-align:center;letter-spacing:.02em;}"
        ".lb-caption{color:#9fb0c6;font-size:12px;margin:0 0 12px 0;}"
        ".lb-table{width:100%;border-collapse:separate;border-spacing:0;}"
        ".lb-table th{position:sticky;top:0;background:#142033;color:#dbe7f5;"
        "font-size:11px;text-transform:uppercase;letter-spacing:.06em;padding:10px 12px;"
        "border-bottom:1px solid #2a3a52;text-align:left;white-space:nowrap;}"
        ".lb-table td{padding:10px 12px;border-bottom:1px solid #1d2939;"
        "white-space:nowrap;vertical-align:middle;}"
        ".lb-table tr:hover td{background:#101a29;}"
        ".lb-current-row td{background:#1d2c3f;}"
        ".lb-current-row:hover td{background:#23364d;}"
        ".lb-rank-badge,.lb-mini-badge{display:inline-flex;align-items:center;"
        "border-radius:999px;font-weight:700;letter-spacing:.02em;}"
        ".lb-rank-badge{padding:4px 8px;background:#253246;color:#dbe7f5;}"
        ".lb-mini-badge{padding:3px 8px;font-size:11px;margin-left:8px;}"
        ".lb-top-1{background:#e9c46a;color:#241700;}"
        ".lb-top-2{background:#b8c4d6;color:#102030;}"
        ".lb-top-3{background:#c08b5c;color:#1c1207;}"
        ".lb-current-badge{background:#2a9d8f;color:#041411;}"
        ".lb-model-name{font-weight:700;color:#f5f7fb;}"
        ".lb-run-id{font-family:Consolas,'Courier New',monospace;font-size:12px;color:#9bd1ff;}"
        ".lb-timestamp,.lb-meta,.lb-muted{color:#90a4bb;}"
        ".lb-positive{color:#74d39f;font-weight:700;}"
        ".lb-negative{color:#ff8f8f;font-weight:700;}"
        ".lb-neutral{color:#d6dde8;}"
        "</style>"
        f"{banner_html}"
        '<p class="lb-caption">'
        "Ranked by Annualized Return, then BMC/MMC proxy, then CORR."
        f"{escape(current_rank_text)}"
        "</p>"
        '<table class="lb-table">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def load_model_metrics_history(path: str | Path | None = None) -> pd.DataFrame:
    global MODEL_METRICS_DF
    history_path = _resolve_history_path(path)
    if history_path.exists() and history_path.stat().st_size > 0:
        loaded = pd.read_csv(history_path)
        metric_columns = _resolve_metric_columns(loaded)
        MODEL_METRICS_DF = _normalize_df(loaded, metric_columns)
    else:
        MODEL_METRICS_DF = pd.DataFrame(columns=_history_columns([]))
        _ensure_history_headers(history_path, [])
    return MODEL_METRICS_DF.copy()


def get_model_metrics_history(path: str | Path | None = None) -> pd.DataFrame:
    if MODEL_METRICS_DF.empty and len(MODEL_METRICS_DF.columns) == len(
        _METADATA_COLUMNS
    ):
        load_model_metrics_history(path)
    return MODEL_METRICS_DF.copy()


def record_model_metrics(
    metrics: dict[str, Any],
    model_name: str,
    notebook_name: str | None = None,
    path: str | Path | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Persist all calculate_metrics output keys with dynamic schema evolution."""
    global MODEL_METRICS_DF

    history_path = _resolve_history_path(path)
    history_df = load_model_metrics_history(history_path)

    metric_payload = _extract_metric_payload(metrics)
    metric_columns = _resolve_metric_columns(history_df, metric_payload)
    _ensure_history_headers(history_path, metric_columns)
    history_df = _normalize_df(history_df, metric_columns)

    row: dict[str, Any] = {
        "run_id": str(uuid4())[:8],
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "notebook_name": notebook_name or "",
        "history_path": str(history_path),
    }
    for column in metric_columns:
        row[column] = metric_payload.get(column, np.nan)

    row_df = _normalize_df(pd.DataFrame([row]), metric_columns)
    row = row_df.iloc[0].to_dict()

    existing_rows = history_df[history_df["model_name"] == model_name]
    if not existing_rows.empty:
        latest_existing = existing_rows.iloc[-1]
        if not force:
            return history_df.copy(), _build_result_row(
                latest_existing,
                "skipped",
                "Model already exists. Pass force=True to overwrite.",
            )
        if not _has_metric_changes(latest_existing, row, metric_columns):
            return history_df.copy(), _build_result_row(
                latest_existing,
                "skipped",
                "Model exists but metrics are unchanged. Nothing overwritten.",
            )
        history_df = history_df[history_df["model_name"] != model_name].copy()

    MODEL_METRICS_DF = pd.concat([history_df, row_df], ignore_index=True)
    MODEL_METRICS_DF = _normalize_df(MODEL_METRICS_DF, metric_columns)
    MODEL_METRICS_DF.to_csv(history_path, index=False)

    status = "inserted"
    message = "Inserted new model metrics."
    if not existing_rows.empty:
        status = "overwritten"
        message = "Overwrote existing model metrics (force=True and metrics changed)."

    return MODEL_METRICS_DF.copy(), _build_result_row(row, status, message)


def compare_top_models_with_current(
    current_run_id: str,
    top_n: int = 3,
    path: str | Path | None = None,
    show_message: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Return top leaderboard rows and whether the current run is #1."""
    history = get_model_metrics_history(path)
    if history.empty:
        return history, False

    return_key = "9_Annualized_Return_PCT"
    mmc_key = "4_Mean_BMC20" if "4_Mean_BMC20" in history.columns else "4_Mean_MMC20"
    if return_key not in history.columns:
        raise KeyError(
            "Unable to rank models: missing '9_Annualized_Return_PCT' in history."
        )
    if mmc_key not in history.columns:
        raise KeyError(
            "Unable to rank models: missing '4_Mean_BMC20' or '4_Mean_MMC20' in history."
        )
    if "3_Mean_CORR20V2" not in history.columns:
        raise KeyError("Unable to rank models: missing '3_Mean_CORR20V2' in history.")

    ranking = history.sort_values(
        by=[return_key, mmc_key, "3_Mean_CORR20V2"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1

    top = ranking.head(top_n).copy()
    current_rows = ranking[ranking["run_id"] == current_run_id].copy()

    if current_rows.empty:
        current_included = top
    else:
        current_row = current_rows.iloc[[0]]
        if (top["run_id"] == current_run_id).any():
            current_included = top
        else:
            current_included = pd.concat([top, current_row], ignore_index=True)

    current_included = current_included.drop_duplicates(subset=["run_id"]).copy()
    current_included["is_current"] = current_included["run_id"] == current_run_id
    current_included = current_included.sort_values(by=["rank"]).reset_index(drop=True)

    comparison_metric_columns = _comparison_columns(
        _metric_columns_from_df(current_included), mmc_key
    )
    leaderboard_columns = [
        "rank",
        "run_id",
        "model_name",
        *comparison_metric_columns,
        "timestamp_utc",
        "is_current",
    ]
    leaderboard = current_included[
        [col for col in leaderboard_columns if col in current_included.columns]
    ].copy()

    is_top_performer = bool(
        not ranking.empty
        and ranking.iloc[0]["run_id"] == current_run_id
        and len(ranking) > 1
    )

    if show_message:
        if HTML is not None and display is not None:
            display(HTML(_build_leaderboard_html(leaderboard, is_top_performer)))
        elif is_top_performer:
            print(_WINNER_MESSAGE)

    return leaderboard, is_top_performer
