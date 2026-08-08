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

__all__ = ["PairedResult", "paired_era_comparison", "promotion_verdict"]


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
    ci_high`` (higher-is-better) -> ``"promote"``; the mirror -> ``"hold"``;
    any overlap, missing CI, or missing champion scorecard -> ``"caution"``.
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
