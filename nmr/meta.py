"""Cross-run meta-analysis: paired era comparison and promotion decisions.

Decision layer on top of ``nmr.inference`` and ``nmr.evaluation``. All
statistics reuse the repo's seeded block-bootstrap machinery; nothing here
mutates the registry.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from nmr.config import DataConfig
from nmr.evaluation import MIN_OVERLAP_ERAS, NonVacuityError
from nmr.inference import Horizon, block_bootstrap_ci, resolve_block_len

__all__ = [
    "PairedResult",
    "campaign_evidence",
    "fleet_summary",
    "paired_era_comparison",
    "promotion_verdict",
    "CampaignEvidence",
]


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

@dataclass(frozen=True)
class CampaignEvidence:
    """Per-variant validation evidence plus pairwise screen verdicts."""

    variants: pl.DataFrame
    pairwise: pl.DataFrame


def _identity_era_ic(df: pl.DataFrame) -> dict[str, float]:
    return dict(zip(df.get_column("era"), df.get_column("ic")))


def campaign_evidence(
    campaign_log_path: str | Path,
    registry_root: str | Path,
    *,
    data: DataConfig,
    main_target: str = "target",
    fne_reference_set: str = "medium",
    n_boot: int = 200,
    seed: int = 0,
    min_overlap_eras: int = MIN_OVERLAP_ERAS,
) -> CampaignEvidence:
    """Assemble validation evidence for every recorded run of a campaign.

    Reads the campaign log (``artifacts/campaigns/<name>.json``) and the
    per-run registry payloads. For each recorded run: validation mean IC with
    the run scorecard's 95% block-bootstrap CI, IC Sharpe and max drawdown
    (scorecard), feature count and backend/device (manifest), and FNE at 100%
    neutralization against ``fne_reference_set`` (medium by default) with its
    own bootstrap CI over per-era residual ICs (:func:`neutralized_ic_series`).

    ``pairwise`` reports block-bootstrap mean differences on the validation IC
    series for the screen-defining pairs (v2 vs v3, v2 vs v4, v3 vs v4) per
    backend — a positive diff means the first variant is better. Runs whose
    status is not ``recorded`` or whose artifacts are missing are collected as
    ``error`` rows (fail loud, never silently dropped).
    """
    from nmr.analysis import neutralized_ic_series
    from nmr.data import IngestionAgent
    from nmr.features import _per_era_pearson

    log_path = Path(campaign_log_path)
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if not runs:
        raise ValueError(f"{log_path}: campaign log contains no runs")

    agent = IngestionAgent(data)
    targets = agent.scan(
        "validation", columns=["era", "id", main_target]
    ).collect()
    medium_cols = agent.features(fne_reference_set)
    val_medium = agent.scan(
        "validation", columns=["era", "id", *medium_cols, main_target]
    ).collect()

    variant_rows: list[dict[str, object]] = []
    ic_frames: dict[str, pl.DataFrame] = {}
    for entry in runs:
        label = Path(str(entry["config_path"])).stem
        run_id = entry.get("run_id")
        if entry.get("status") != "recorded" or run_id is None:
            err_msg = entry.get("error")
            variant_rows.append(
                {"variant": label, "status": str(entry.get("status")),
                 "error": err_msg if err_msg else f"status={entry.get('status')!r}"}
            )
            continue
        run_json = Path(registry_root) / run_id / "run.json"
        if not run_json.exists():
            variant_rows.append(
                {"variant": label, "status": "recorded",
                 "error": f"run.json missing for {run_id}"}
            )
            continue
        meta = json.loads(run_json.read_text(encoding="utf-8"))
        scorecard = meta.get("scorecard") or {}
        manifest = meta.get("manifest") or {}
        config = manifest.get("config") or {}
        preds_path = Path(registry_root) / run_id / "validation_preds.parquet"
        if not preds_path.exists():
            variant_rows.append(
                {"variant": label, "status": "recorded",
                 "error": f"validation_preds.parquet missing for {run_id}"}
            )
            continue
        preds = pl.read_parquet(preds_path)

        joined = preds.join(targets, on=["era", "id"], how="inner")
        corrs, degenerate = _per_era_pearson(
            joined, ["prediction"], main_target, "era"
        )
        series = pl.DataFrame(
            [
                {"era": era, "ic": float(vec[0])}
                for era, vec in corrs.items()
                if era not in degenerate
            ],
            schema={"era": pl.Utf8, "ic": pl.Float64},
        ).sort("era")
        ic_frames[label] = series
        ics = series["ic"].to_numpy()
        n_eras = int(ics.size)
        ic_ci_lo = ic_ci_hi = None
        if n_eras >= 2:
            ci = block_bootstrap_ci(
                ics, np.mean,
                block_len=resolve_block_len(n_eras, "20D"),
                n_boot=n_boot, seed=seed,
            )
            ic_ci_lo, ic_ci_hi = ci.lo, ci.hi

        fne_joined = preds.join(val_medium, on=["era", "id"], how="inner")
        fne_series = neutralized_ic_series(
            fne_joined.partition_by("era", maintain_order=True),
            ["prediction"], medium_cols, main_target,
            proportion=1.0,
        )
        fne_ics = fne_series["ic"].to_numpy()
        fne_ci_lo = fne_ci_hi = None
        if int(fne_ics.size) >= 2:
            fci = block_bootstrap_ci(
                fne_ics, np.mean,
                block_len=resolve_block_len(int(fne_ics.size), "20D"),
                n_boot=n_boot, seed=seed,
            )
            fne_ci_lo, fne_ci_hi = fci.lo, fci.hi

        variant_rows.append(
            {
                "variant": label,
                "status": "recorded",
                "backend": (config.get("model") or {}).get("backend"),
                "device": (config.get("model") or {}).get("device", "auto"),
                "n_features": len(manifest.get("feature_cols") or []),
                "mean_ic": scorecard.get("corr"),
                "ic_ci_lo": scorecard.get("corr_ci_low", ic_ci_lo),
                "ic_ci_hi": scorecard.get("corr_ci_high", ic_ci_hi),
                "ic_sharpe": scorecard.get("corr_sharpe_ac"),
                "max_drawdown": scorecard.get("max_drawdown"),
                "n_eras": scorecard.get("n_eras", n_eras),
                "fne100": float(np.mean(fne_ics)) if fne_ics.size else None,
                "fne100_ci_lo": fne_ci_lo,
                "fne100_ci_hi": fne_ci_hi,
            }
        )

    variants = pl.DataFrame(variant_rows)

    pairwise_rows: list[dict[str, object]] = []
    if ic_frames:
        # pair keys derive from the config-filename prefix (e.g. 'lgbm_v2'),
        # never from the backend name, so renames cannot silently drop pairs.
        prefixes: dict[str, list[str]] = {}
        for label in ic_frames:
            if "_v" in label:
                prefixes.setdefault(label.rsplit("_v", 1)[0], []).append(label)
        for prefix, labels in sorted(prefixes.items()):
            backend = next(
                (r["backend"] for r in variant_rows
                 if r.get("variant") == f"{prefix}_v2"),
                prefix,
            )
            for pair in (("v2", "v3"), ("v2", "v4"), ("v3", "v4")):
                key_a = f"{prefix}_{pair[0]}"
                key_b = f"{prefix}_{pair[1]}"
                if key_a not in labels or key_b not in labels:
                    continue
                try:
                    res = paired_era_comparison(
                        ic_frames[key_a], ic_frames[key_b],
                        metric_fn=_identity_era_ic,
                        horizon="20D",
                        n_boot=n_boot,
                        seed=seed,
                        min_overlap_eras=min_overlap_eras,
                        device_a=str(
                            next(
                                (r["device"] for r in variant_rows
                                 if r.get("variant") == key_a), None
                            )
                        ),
                        device_b=str(
                            next(
                                (r["device"] for r in variant_rows
                                 if r.get("variant") == key_b), None
                            )
                        ),
                    )
                except NonVacuityError as exc:
                    pairwise_rows.append(
                        {
                            "pair": f"{key_a} vs {key_b}",
                            "backend": backend,
                            "mean_diff": None,
                            "ci_low": None,
                            "ci_high": None,
                            "n_eras": 0,
                            "error": str(exc),
                        }
                    )
                    continue
                pairwise_rows.append(
                    {
                        "pair": f"{key_a} vs {key_b}",
                        "backend": backend,
                        "mean_diff": res.mean_diff,
                        "ci_low": res.ci_low,
                        "ci_high": res.ci_high,
                        "n_eras": res.n_eras,
                        "error": None,
                    }
                )
    pairwise = pl.DataFrame(pairwise_rows)
    return CampaignEvidence(variants=variants, pairwise=pairwise)
