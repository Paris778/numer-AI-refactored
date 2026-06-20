"""
Numerai Performance Analytics Engine.
Optimized for v5.2 'Supermassive' Data and Era-Based Validation.

This module provides production-grade scoring utilities for the Numerai Tournament,
incorporating official payout formulas, risk diagnostics, and era-based validation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numerai_tools.scoring import (
    correlation_contribution,
    feature_neutral_corr,
    max_feature_correlation,
    numerai_corr,
    sharpe_ratio,
)

# --- Production Constants ---
EPS = 1e-6
PAYOUT_FACTOR_ESTIMATE = (
    0.1  # This in reality varies each round , but here we are using this estimate
)
PAYOUT_CAP = 0.05  # Official ±5% per-round cap
DEFAULT_CORR_WEIGHT = 0.75  # Official CORR weight for payout
DEFAULT_MMC_WEIGHT = 2.25  # Official MMC weight for payout
DEFAULT_MAX_FILTERED_INDEX_RATIO = 0.2

#######################################################################
#######################################################################
#######################################################################


def _validate_inputs(df: pd.DataFrame, required_cols: set, name: str):
    """Strict input validation to prevent silent downstream failures."""
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Critical failure: {name} is missing columns: {missing}")


def _coerce_optional_int(value):
    if value is None:
        return None
    return int(value)


def _safe_divide(numerator: float, denominator: float) -> float:
    """Zero-division protected division for financial ratios."""
    return float(numerator / (denominator + EPS))


def _max_drawdown(series: pd.Series) -> float:
    """Calculates maximum peak-to-trough drawdown from an equity curve."""
    values = np.asarray(series, dtype=np.float64)
    cumulative = np.cumsum(values)
    running_peak = np.maximum.accumulate(cumulative)
    return float(np.max(running_peak - cumulative, initial=0.0))


def _conditional_value_at_risk(series: pd.Series, alpha: float = 0.05) -> float:
    """Measures the expected loss in the worst alpha% of cases (Tail Risk)."""
    values = np.asarray(series, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    var_level = float(np.quantile(values, alpha))
    tail = values[values <= var_level]
    return float(tail.mean()) if tail.size > 0 else var_level


#######################################################################
#######################################################################
#######################################################################


def estimate_era_returns(
    corr_series: pd.Series,
    mmc_series: pd.Series,
    payout_factor: float = PAYOUT_FACTOR_ESTIMATE,
    c_weight: float = DEFAULT_CORR_WEIGHT,
    m_weight: float = DEFAULT_MMC_WEIGHT,
) -> pd.Series:
    """
    Calculates the capped percentage return for each era individually.
    Formula: payout = clip(payout_factor * (0.75*CORR + 2.25*MMC), -0.05, 0.05)
    """
    raw_payout = payout_factor * ((corr_series * c_weight) + (mmc_series * m_weight))
    # Apply official ±5% cap
    capped_returns = np.clip(raw_payout, -PAYOUT_CAP, PAYOUT_CAP)
    return capped_returns * 100.0


def calculate_custom_raps(
    payout_series: pd.Series,
    lambda_penalty: float = 0.2,
    lambda_tail: float = 0.5,
) -> float:
    """
    Risk-Adjusted Payout Score (RAPS).
    RAPS = Sharpe(Payout) - (lambda * MaxDrawdown) - (lambda_tail * |CVaR|)
    """
    mean_payout = float(payout_series.mean())
    payout_vol = float(payout_series.std(ddof=0))

    drawdown = _max_drawdown(payout_series)
    cvar = abs(min(_conditional_value_at_risk(payout_series), 0.0))

    return (
        _safe_divide(mean_payout, payout_vol)
        - (lambda_penalty * drawdown)
        - (lambda_tail * cvar)
    )


def calculate_metrics(
    df_validation: pd.DataFrame,
    benchmarks: pd.DataFrame,
    features: List[str],
    target_col: str = "target_ender20",  # Primary tournament payout target (2026+)
    benchmark_col: str = "v52_lgbm_ender20",
    fast_mode: bool = False,
    **kwargs,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Principal Engineer Refactored Metrics Engine.
    Computes the 'Golden Eleven' evaluation panel for Numerai model performance.
    """
    # 1. Data Integrity & Alignment
    _validate_inputs(df_validation, {"era", "prediction", target_col}, "df_validation")
    _validate_inputs(benchmarks, {benchmark_col}, "benchmarks")
    _validate_inputs(df_validation, set(features), "df_validation(feature set)")

    # Optional passthrough knobs for strict alignment experiments with numerai_tools.
    # Keep fast_mode as the primary argument, while honoring legacy alias fast_metrics.
    fast_mode = bool(kwargs.get("fast_metrics", fast_mode))
    top_bottom = _coerce_optional_int(kwargs.get("top_bottom"))
    target_pow15 = bool(kwargs.get("target_pow15", True))
    max_filtered_index_ratio = float(
        kwargs.get("max_filtered_index_ratio", DEFAULT_MAX_FILTERED_INDEX_RATIO)
    )
    corr_weight = float(kwargs.get("corr_weight", DEFAULT_CORR_WEIGHT))
    mmc_weight = float(kwargs.get("mmc_weight", DEFAULT_MMC_WEIGHT))

    # Align on ID index to ensure row-wise correspondence
    df_val = (
        df_validation.set_index("id")
        if "id" in df_validation.columns
        else df_validation
    )
    bench = benchmarks.set_index("id") if "id" in benchmarks.columns else benchmarks

    # Efficient inner join
    eval_df = df_val.join(bench[[benchmark_col]], how="inner").dropna(
        subset=["prediction", benchmark_col, target_col]
    )

    if eval_df.empty:
        raise ValueError(
            "No rows remain after aligning df_validation with benchmarks and dropping "
            "NaNs for prediction/benchmark/target. Check id overlap and nulls."
        )

    # 2. Era-Wise Calculation Loop
    records = []
    for era, frame in eval_df.groupby("era", sort=False):
        era_stats = {
            "era": era,
            "CORR20V2": numerai_corr(
                frame[["prediction"]],
                frame[target_col],
                max_filtered_index_ratio=max_filtered_index_ratio,
                top_bottom=top_bottom,
                target_pow15=target_pow15,
            ).iloc[0],
            # Using benchmark_col as the neutralizer yields benchmark contribution
            # semantics (BMC-style), not canonical SWMM-based MMC.
            "BMC20": correlation_contribution(
                frame[["prediction"]],
                frame[benchmark_col],
                frame[target_col],
                top_bottom=top_bottom,
            ).iloc[0],
            "BENCHMARK_CORR": frame["prediction"].corr(
                frame[benchmark_col], method="pearson"
            ),
        }

        if not fast_mode:
            era_stats["FNC"] = feature_neutral_corr(
                frame[["prediction"]],
                frame[features],
                frame[target_col],
                top_bottom=top_bottom,
            ).iloc[0]
            era_stats["MAX_FE"] = max_feature_correlation(
                frame["prediction"],
                frame[features],
                top_bottom=top_bottom,
            )[1]

        # Backward-compatible alias for existing downstream consumers.
        era_stats["MMC20"] = era_stats["BMC20"]

        records.append(era_stats)

    # 3. Post-Processing & Compounding
    per_era_df = pd.DataFrame(records).set_index("era").sort_index()

    # Vectorized Return Proxies
    per_era_df["PAYOUT_PROXY"] = (per_era_df["CORR20V2"] * corr_weight) + (
        per_era_df["BMC20"] * mmc_weight
    )

    per_era_df["ESTIMATED_RETURN_PCT"] = estimate_era_returns(
        per_era_df["CORR20V2"],
        per_era_df["BMC20"],
        c_weight=corr_weight,
        m_weight=mmc_weight,
    )

    # Compounding Logic (Annualized over 52 eras/year)
    ret_decimal = per_era_df["ESTIMATED_RETURN_PCT"] / 100.0
    compounded_growth = np.prod(1.0 + ret_decimal)
    ann_factor = 52.0 / len(per_era_df) if not per_era_df.empty else 0

    # 4. Final Metrics Panel
    metrics = {
        "1_RAPS": calculate_custom_raps(per_era_df["PAYOUT_PROXY"]),
        "2_Sharpe_Payout": float(sharpe_ratio(per_era_df["PAYOUT_PROXY"])),
        "3_Mean_CORR20V2": float(per_era_df["CORR20V2"].mean()),
        "4_Mean_BMC20": float(per_era_df["BMC20"].mean()),
        # Backward-compatible alias retained intentionally.
        "4_Mean_MMC20": float(per_era_df["MMC20"].mean()),
        "5_Sharpe_CORR": float(sharpe_ratio(per_era_df["CORR20V2"])),
        "6_Mean_FNC": (
            float(per_era_df["FNC"].mean()) if "FNC" in per_era_df else np.nan
        ),
        "7_Max_Drawdown_CORR": -_max_drawdown(per_era_df["CORR20V2"]),
        "8_Win_Rate": float((per_era_df["CORR20V2"] > 0).mean()),
        "9_Annualized_Return_PCT": float((compounded_growth**ann_factor - 1.0) * 100.0),
        "10_Benchmark_Corr": float(per_era_df["BENCHMARK_CORR"].mean()),
        "11_BMC_Volatility": float(per_era_df["BMC20"].std(ddof=0)),
        # Backward-compatible alias retained intentionally.
        "11_MMC_Volatility": float(per_era_df["MMC20"].std(ddof=0)),
    }

    return metrics, per_era_df
