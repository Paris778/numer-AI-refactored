"""Cross-run meta-analysis: paired era comparison and promotion decisions.

Decision layer on top of ``nmr.inference`` and ``nmr.evaluation``. All
statistics reuse the repo's seeded block-bootstrap machinery; nothing here
mutates the registry.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from nmr.evaluation import MIN_OVERLAP_ERAS, NonVacuityError
from nmr.inference import Horizon, block_bootstrap_ci, resolve_block_len

__all__ = ["PairedResult", "paired_era_comparison"]


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
