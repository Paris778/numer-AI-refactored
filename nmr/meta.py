"""Cross-run meta-analysis: paired era comparison and promotion decisions.

Decision layer on top of ``nmr.inference`` and ``nmr.evaluation``. All
statistics reuse the repo's seeded block-bootstrap machinery; nothing here
mutates the registry.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from nmr.evaluation import MIN_OVERLAP_ERAS, NonVacuityError
from nmr.inference import Horizon, block_bootstrap_ci, resolve_block_len

__all__ = ["PairedResult", "fleet_summary", "paired_era_comparison", "promotion_verdict"]


@dataclass(frozen=True)
class PairedResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    n_eras: int
    device_mismatch: bool
    alpha: float
    n_boot: int
    block_len: int


def paired_era_comparison(
    oof_a: pl.DataFrame,
    oof_b: pl.DataFrame,
    *,
    metric_fn: Callable[[pl.DataFrame], dict[str, float]],
    era_col: str = "era",
    horizon: Horizon = "20D",
    n_boot: int = 1000,
    seed: int,
    alpha: float = 0.05,
    min_overlap_eras: int = MIN_OVERLAP_ERAS,
    block_len: int | None = None,
    device_a: str | None = None,
    device_b: str | None = None,
) -> PairedResult:
    """Compare two runs on per-era metric differences via block bootstrap.

    ``metric_fn`` maps an OOF frame to ``{era: metric}`` (e.g. a closure over
    ``EvaluationEngine().per_era_corr`` with explicit pred/target/era columns).
    Positive ``mean_diff`` means A is better. Both frames must contain the
    ``era_col`` column; a missing one raises ``ValueError`` naming the
    offending frame(s). Eras are intersected on the
    numeric era index; fewer than ``min_overlap_eras`` overlapping eras raises
    :class:`NonVacuityError`. A device mismatch is reported (GPU vs CPU OOF
    values are not comparable — see AGENTS.md operational hazards), never
    silently corrected.
    """
    missing_frames = [
        name
        for name, frame in (("oof_a", oof_a), ("oof_b", oof_b))
        if era_col not in frame.columns
    ]
    if missing_frames:
        raise ValueError(
            f"era_col {era_col!r} missing from column set of: "
            + ", ".join(missing_frames)
        )
    per_era_a = metric_fn(oof_a)
    per_era_b = metric_fn(oof_b)
    overlap = sorted(set(per_era_a) & set(per_era_b), key=int)
    if len(overlap) < min_overlap_eras:
        raise NonVacuityError(
            f"paired overlap {len(overlap)} eras < MIN_OVERLAP_ERAS "
            f"{min_overlap_eras}"
        )
    diffs = np.asarray(
        [float(per_era_a[era]) - float(per_era_b[era]) for era in overlap],
        dtype=float,
    )
    blen = (
        block_len
        if block_len is not None
        else resolve_block_len(int(diffs.size), horizon)
    )
    ci = block_bootstrap_ci(
        diffs,
        lambda arr: float(np.mean(arr)),
        block_len=blen,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )
    return PairedResult(
        mean_diff=float(np.mean(diffs)),
        ci_low=ci.lo,
        ci_high=ci.hi,
        n_eras=int(diffs.size),
        device_mismatch=(
            device_a is not None
            and device_b is not None
            and device_a != device_b
        ),
        alpha=float(alpha),
        n_boot=int(n_boot),
        block_len=int(blen),
    )


# Directions aligned with RunRegistry._SCORECARD_METRIC_DIRECTION (parity-tested
# in test_meta.py). True = higher-is-better.
_VERDICT_DIRECTIONS: dict[str, bool] = {
    "corr": True,
    "mmc": True,
    "fnc": True,
    "corr_sharpe_ac": True,
    "deflated_sharpe": True,
    "std_corr": False,
    "max_drawdown": False,
}


def promotion_verdict(
    candidate: dict,
    champion: dict | None,
    *,
    metric: str = "corr_sharpe_ac",
    alpha: float = 0.05,
) -> Literal["promote", "hold", "caution"]:
    """Significance-aware promotion decision on registry entries.

    Compares CI-bearing scorecard cells: candidate ``ci_low > champion
    ci_high`` (higher-is-better) -> ``"promote"``; the mirror -> ``"hold"``.
    With no champion, or a champion lacking the scorecard metric (an
    unmeasurable champion is treated like no champion), -> ``"promote"``.
    Missing CIs on either side, or any CI overlap, -> ``"caution"``.
    This is an advisory verdict only — it never writes the registry.
    """
    if metric not in _VERDICT_DIRECTIONS:
        raise ValueError(
            f"metric={metric!r} not in {sorted(_VERDICT_DIRECTIONS)}"
        )
    higher_is_better = _VERDICT_DIRECTIONS[metric]

    def _cell(entry: dict) -> tuple[float | None, float | None, float | None]:
        scorecard = entry.get("scorecard") or {}
        value = scorecard.get(metric)
        if value is None:
            return None, None, None
        return (
            float(value),
            scorecard.get(f"{metric}_ci_low"),
            scorecard.get(f"{metric}_ci_high"),
        )

    cand_value, cand_lo, cand_hi = _cell(candidate)
    if cand_value is None:
        raise ValueError(
            f"candidate run lacks scorecard metric {metric!r}; "
            "cannot issue a significance-aware verdict"
        )
    if champion is None:
        return "promote"
    champ_value, champ_lo, champ_hi = _cell(champion)
    if champ_value is None:
        return "promote"
    if None in (cand_lo, cand_hi, champ_lo, champ_hi):
        return "caution"

    if higher_is_better:
        if cand_lo > champ_hi:
            return "promote"
        if cand_hi < champ_lo:
            return "hold"
    else:
        if cand_hi < champ_lo:
            return "promote"
        if cand_lo > champ_hi:
            return "hold"
    return "caution"


def fleet_summary(
    runs: Sequence[dict],
    *,
    metric: str = "corr_sharpe_ac",
    n_trials: int,
    dsr_confidence: float = 0.95,
) -> pl.DataFrame:
    """Flatten registry entries into a per-run fleet table.

    Per-run cells: the requested scorecard metric (value + CI + n_eras), the
    stored ``deflated_sharpe`` with a pass/fail flag against
    ``dsr_confidence``, max feature exposure, ``oof_device``, and grouping
    attributes from the manifest config (preset, feature_set, feature_subset,
    neutralization_proportion) plus robustness presence flags (bmc, horizon,
    perturbation, regime). ``n_trials`` and ``dsr_confidence`` are recorded as
    policy context columns; the stored DSR itself was computed with
    ``n_trials=1`` at scorecard time — campaign-aware DSR requires era-level
    recompute via :func:`paired_era_comparison` tooling and is out of scope
    here. Runs without a scorecard are flagged (legacy), never silently
    dropped. Deterministic: sorted by metric desc, run_id tiebreak.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if not (0.0 < dsr_confidence < 1.0):
        raise ValueError("dsr_confidence must satisfy 0 < dsr_confidence < 1")

    rows: list[dict] = []
    for entry in runs:
        run_id = entry["run_id"]
        manifest = entry.get("manifest") or {}
        config = manifest.get("config") or {}
        data_cfg = config.get("data") or {}
        model_cfg = config.get("model") or {}
        risk_cfg = config.get("risk") or {}
        scorecard = entry.get("scorecard") or {}
        metric_value = scorecard.get(metric)
        rows.append(
            {
                "run_id": run_id,
                "metric": float(metric_value) if metric_value is not None else None,
                "metric_ci_low": scorecard.get(f"{metric}_ci_low"),
                "metric_ci_high": scorecard.get(f"{metric}_ci_high"),
                "metric_n_eras": scorecard.get(f"{metric}_n_eras"),
                "deflated_sharpe": scorecard.get("deflated_sharpe"),
                "dsr_pass": bool(
                    scorecard.get("deflated_sharpe") is not None
                    and float(scorecard["deflated_sharpe"]) >= dsr_confidence
                ),
                "max_feature_exposure": scorecard.get("max_feature_exposure"),
                "oof_device": manifest.get("oof_device"),
                "preset": model_cfg.get("preset"),
                "feature_set": data_cfg.get("feature_set"),
                "feature_subset": data_cfg.get("feature_subset"),
                "neutralization_proportion": risk_cfg.get("neutralization_proportion"),
                "has_bmc": scorecard.get("bmc") is not None,
                "has_horizon": scorecard.get("horizon_model_sharpe_20") is not None,
                "has_perturb": scorecard.get("perturb_ceiling_stability") is not None,
                "has_regime": scorecard.get("regime_count") is not None,
                "policy_n_trials": n_trials,
                "policy_dsr_confidence": dsr_confidence,
            }
        )
    frame = pl.DataFrame(rows)
    if frame.height > 0:
        frame = frame.sort(
            ["metric", "run_id"], descending=[True, False], nulls_last=True
        )
    return frame
