"""Promotion writer: train the full version (train+validation) and publish it.

The money-path terminus of the audit remediation. ``promote_full_version``
takes a registry run, re-trains per-target full-history models on
**train+validation** (the research run trained on train only), rebuilds the
exact deploy closure shared with the runner
(``nmr.runner._build_deploy_pipeline`` — never a second copy), and publishes
one immutable slot per promoted run at
``artifacts/models/<family>/full/<run_id>/`` plus the atomic ``current.json``
pointer (families.py D2 layout).

The full-version manifest is families.py-compatible and additionally records
the promotion verdict block: ``tier4_gate_passed``, ``tier4_receipts``,
``override_used``, and ``config_normalizations``. A rehearsal artifact is
byte-indistinguishable from a real promotion: if the run fails the tier-4 gate
and ``--override-gate`` was used, the artifact's own manifest says
``tier4_gate_passed: false``. ``override_gate`` covers tier-4 *performance*
only — never contract *validity* (the (0,1) submission contract is enforced by
the D5 acceptance path against ``numerai_tools`` on real live data).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nmr._atomicio import atomic_write_text
from nmr.benchmark import Tier4GateConfig, load_benchmark_file
from nmr.config import ExperimentConfig, config_from_dict
from nmr.data import IngestionAgent
from nmr.families import (
    CURRENT_POINTER_NAME,
    DEFAULT_MODELS_DIR,
    FULL_DIR_NAME,
    FULL_MANIFEST_NAME,
    validate_family_name,
)
from nmr.models import ModelOrchestrator
from nmr.runner import ExperimentRunner, _build_deploy_pipeline, _serialize_predict_artifact

logger = logging.getLogger("nmr.promote")

__all__ = [
    "PromotionResult",
    "RehearsalResult",
    "promote_full_version",
    "rehearse_promotion",
    "resolve_champion_run_id",
]

_RID_RE = re.compile(r"^[0-9a-f]{64}$")
RAM_ESTIMATE_FILENAME = "full_version_ram_estimate.json"
# RAM guard derivation (reviewed 2026-08-18): set to the documented WORST CASE
# the machine has already sustained — the recorded solo full-universe fit peak
# of ~40-45 GiB commit (AGENTS.md operational hazards; 3,555 features on 2.12M
# train rows). The medium (780-feature) full-version fit measured ~30 GiB by
# the three-point curve (artifacts/reports/ram_curve.json), so 45 GiB leaves
# ~15 GiB headroom over medium while still refusing any job approaching the
# full-universe ceiling. 63.7 GiB total machine RAM.
_RAM_GUARD_BYTES = 45 * 2**30
# Working-set thrash guard: refuse when the extrapolated combined working set
# exceeds this fraction of physical RAM (leaving headroom for the OS and
# anything else resident — a job at commit-limit-minus-1 GiB that is largely
# resident still thrashes at 1.1 iters/s).
_RAM_WS_FRACTION = 0.85
# Default tier-4 gate config (the line in the sand).
_TIER4_GATE_YAML = Path(__file__).resolve().parent.parent / "configs" / "benchmarks" / "tier4_gate.yaml"

# Hard gate fields: (scorecard field, gate threshold attr, strict needs >).
# deflated_sharpe is display-only (A6: no search history to bind deflation to
# at gate time); turnover is structurally unavailable on v5.3 — both recorded
# in receipts, neither a hard failure.
_HARD_GATE_FIELDS = (
    ("corr", "corr_min", False),
    ("corr_sharpe_ac", "corr_sharpe_ac_min", False),
    ("fnc", "fnc_min", False),
    ("gain_to_pain_ratio", "gain_to_pain_min", False),
    ("cagr_1y", "cagr_min", True),
)


@dataclass(frozen=True)
class PromotionResult:
    """Outcome of a full-version promotion (or rehearsal)."""

    artifact_path: Path
    manifest_path: Path
    run_id: str
    family: str
    tier4_gate_passed: bool
    override_used: bool
    measured_peak_bytes: int | None = None
    measured_peak_commit_bytes: int | None = None


@dataclass(frozen=True)
class RehearsalResult:
    """Outcome of the D7 Stage-1 truncated-window rehearsal."""

    artifact_path: Path
    ram_estimate_path: Path
    acceptance_passed: bool
    measured_peak_bytes: int | None
    measured_peak_commit_bytes: int | None
    train_validation_rows: int


def _normalize_stored_config(
    config_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Deep-copy a stored run config, neutralizing pre-schema-change fields.

    Every change is recorded in ``config_normalizations`` so the promoted
    manifest is explicit about how it differs from the stored research config.
    The run_id is NEVER recomputed from the normalized config — it belongs to
    the research run and is referenced via ``promoted_from_run_id``.
    """
    normalized = copy.deepcopy(config_dict)
    normalizations: list[dict[str, Any]] = []
    split = normalized.setdefault("split", {})
    embargo = split.get("embargo_eras")
    if embargo not in (None, 0):
        split["embargo_eras"] = 0
        normalizations.append(
            {"field": "split.embargo_eras", "from": embargo, "to": 0}
        )
    data = normalized.setdefault("data", {})
    if "horizon" not in data:
        data["horizon"] = "20D"
        normalizations.append({"field": "data.horizon", "from": None, "to": "20D"})
    return normalized, normalizations


def _evaluate_gate(
    scorecard: dict[str, Any], gate: Tier4GateConfig
) -> tuple[bool, dict[str, Any]]:
    """Evaluate the stored run scorecard against the tier-4 hard thresholds.

    Missing evidence for a hard field is a failure (a candidate without the
    full gate metrics cannot be promoted on faith). ``deflated_sharpe`` is
    recorded display-only; ``turnover_mean`` n/a on v5.3.
    """
    receipts: dict[str, Any] = {}
    violations: list[str] = []
    for field, threshold_attr, strict in _HARD_GATE_FIELDS:
        threshold = float(getattr(gate, threshold_attr))
        measured = scorecard.get(field)
        if measured is None:
            passed = False
            violations.append(f"{field}: missing measured value")
        elif strict:
            passed = float(measured) > threshold
            if not passed:
                violations.append(
                    f"{field}: observed={measured:.8f}, need > {threshold:.8f}"
                )
        else:
            passed = float(measured) >= threshold
            if not passed:
                violations.append(
                    f"{field}: observed={measured:.8f}, need >= {threshold:.8f}"
                )
        receipts[field] = {"threshold": threshold, "measured": measured, "passed": passed}
    receipts["turnover_mean"] = {
        "threshold": float(gate.turnover_max),
        "measured": scorecard.get("turnover_mean"),
        "passed": None,
    }
    receipts["deflated_sharpe"] = {
        "threshold": float(gate.deflated_sharpe_min),
        "measured": scorecard.get("deflated_sharpe"),
        "passed": None,
    }
    return not violations, receipts


def _load_registry_run(registry_dir: Path, run_id: str) -> dict[str, Any]:
    if not _RID_RE.fullmatch(run_id):
        raise ValueError(f"run_id={run_id!r} is not a 64-char lowercase hex string")
    run_json = Path(registry_dir) / run_id / "run.json"
    if not run_json.is_file():
        raise FileNotFoundError(
            f"Run {run_id!r} does not exist in registry {registry_dir}"
        )
    try:
        payload = json.loads(run_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"corrupt run.json for {run_id!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"run.json for {run_id!r} is not a mapping")
    return payload


def _supplemental_identity_check(
    config: ExperimentConfig, manifest: dict[str, Any]
) -> None:
    """The promoted artifact must resolve the SAME supplemental feature set as
    the research run (content identity of derived_feature_sets.json)."""
    stored_sha = manifest.get("supplemental_feature_sets_sha256")
    supp = config.data.supplemental_feature_sets
    if stored_sha is None and supp is None:
        return
    if stored_sha is None or supp is None:
        raise ValueError(
            "supplemental feature-set identity mismatch: stored run "
            f"sha256={stored_sha!r} vs config path={supp!r}"
        )
    actual = ExperimentRunner._supplemental_fingerprint(supp)
    if actual != stored_sha:
        raise ValueError(
            "supplemental feature-set identity mismatch: resolved "
            f"{supp} sha256={actual[:12]}... != stored {stored_sha[:12]}..."
        )


def _scan_len(path: Path) -> int:
    return pl.scan_parquet(path).select(pl.len()).collect().item()


def _ram_guard(config: ExperimentConfig, models_dir: Path) -> None:
    """Enforce the measured dual-metric guard (D7 rehearsal/curve).

    Two metrics guard two different failure modes (review directive 2026-08-18):

    | Metric        | Failure it prevents          | Compared against        |
    |---------------|------------------------------|-------------------------|
    | peak commit   | hard allocation failure (OOM)| commit limit (RAM+page) |
    | peak WS       | thrash (1.1 iters/s collapse)| physical RAM            |

    Extrapolation uses the FITTED CURVE (``artifacts/reports/ram_curve.json``,
    ``peak = a + b*rows`` — intercept + slope from the three measured points),
    never a through-origin single-point scaling: the curve's intercept captures
    histograms + fixed overhead, and forcing it through zero inflates the slope
    by ~40% (measured 2026-08-18). The child terms extrapolate by the fitted
    line at the CURRENT row count (validation grows weekly); the parent terms
    are the curve's measured fixed parent medians. Refuses when: combined
    commit exceeds the ``_RAM_GUARD_BYTES`` ceiling, combined commit exceeds
    the machine's commit limit, or combined working set approaches physical
    RAM (``_RAM_WS_FRACTION``). Falls back to the single-point rehearsal
    estimate (through-origin, logged as a weak extrapolation) only when no
    curve exists. Missing/incomplete data ⇒ SKIP WITH A WARNING — a guard that
    silently approves on the wrong metric is worse than no guard.
    """
    curve_path = Path(models_dir).parent / "reports" / "ram_curve.json"
    estimate_path = Path(models_dir).parent / "reports" / RAM_ESTIMATE_FILENAME
    if curve_path.is_file():
        try:
            curve = json.loads(curve_path.read_text(encoding="utf-8"))
            fit_commit = curve["fit"]
            fit_ws = curve["fit_ws"]
            parent_commit = (
                float(np.median([p["parent_commit_gib"] for p in curve["points"]]))
                * 2**30
            )
            parent_ws = (
                float(np.median([p["parent_ws_gib"] for p in curve["points"]]))
                * 2**30
            )
            source = "curve"
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "[promote] unreadable RAM curve %s (%s); falling back to the estimate",
                curve_path,
                exc,
            )
            curve = None
        if curve is not None:
            current_rows = _scan_len(config.data.path("train.parquet")) + _scan_len(
                config.data.path("validation.parquet")
            )
            child_commit = (
                (fit_commit["intercept_gib"] + fit_commit["slope_gib_per_row"] * current_rows)
                * 2**30
            )
            child_ws = (
                (fit_ws["intercept_gib"] + fit_ws["slope_gib_per_row"] * current_rows)
                * 2**30
            )
            combined_commit = child_commit + parent_commit
            combined_ws = child_ws + parent_ws
            _raise_if_over_guard(
                combined_commit, combined_ws, current_rows, source=source
            )
            return
    if estimate_path.is_file():
        try:
            est = json.loads(estimate_path.read_text(encoding="utf-8"))
            child_commit = est.get("peak_commit_bytes")
            child_ws = est.get("peak_bytes")
            parent_commit = est.get("parent_peak_commit_bytes")
            parent_ws = est.get("parent_peak_bytes")
            rows = int(est["train_validation_rows"])
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "[promote] unreadable RAM estimate %s (%s); skipping the guard",
                estimate_path,
                exc,
            )
            return
        if child_commit is None or child_ws is None:
            logger.warning(
                "[promote] RAM estimate %s lacks dual-metric data; skipping the "
                "guard (the guard gates on commit AND working set)",
                estimate_path,
            )
            return
        current_rows = _scan_len(config.data.path("train.parquet")) + _scan_len(
            config.data.path("validation.parquet")
        )
        scale = current_rows / max(1, rows)
        combined_commit = (child_commit or 0) * scale + (parent_commit or 0)
        combined_ws = (child_ws or 0) * scale + (parent_ws or 0)
        logger.warning(
            "[promote] no RAM curve; single-point estimate extrapolation "
            "(through-origin) is a WEAK upper bound"
        )
        _raise_if_over_guard(combined_commit, combined_ws, current_rows, source="estimate")
        return
    logger.warning(
        "[promote] no RAM curve or estimate at %s; skipping the guard "
        "(run the truncated rehearsal first for a full-history run)",
        estimate_path,
    )


def _raise_if_over_guard(
    combined_commit: float, combined_ws: float, current_rows: int, *, source: str
) -> None:
    """Raise when either guard is breached (shared by curve and estimate paths)."""
    from nmr.models import _machine_memory_limits

    physical, commit_limit = _machine_memory_limits()
    logger.info(
        "[promote] extrapolated full-version combined commit ~%.1f GiB / ws ~%.1f "
        "GiB on %d rows (%s) | physical %.1f GiB commit_limit %.1f GiB",
        combined_commit / 2**30,
        combined_ws / 2**30,
        current_rows,
        source,
        (physical or 0) / 2**30,
        (commit_limit or 0) / 2**30,
    )
    if combined_commit > _RAM_GUARD_BYTES:
        raise RuntimeError(
            f"extrapolated combined commit {combined_commit / 2**30:.1f} GiB "
            f"exceeds the {_RAM_GUARD_BYTES / 2**30:.0f} GiB guard ({source}); "
            "confirmation point / guard raise with written justification "
            "required before a full-history run"
        )
    if commit_limit is not None and combined_commit > commit_limit:
        raise RuntimeError(
            f"extrapolated combined commit {combined_commit / 2**30:.1f} GiB "
            f"exceeds the machine commit limit {commit_limit / 2**30:.1f} GiB"
        )
    if physical is not None and combined_ws > _RAM_WS_FRACTION * physical:
        raise RuntimeError(
            f"extrapolated combined working set {combined_ws / 2**30:.1f} GiB "
            f"approaches physical RAM {physical / 2**30:.1f} GiB "
            f"({_RAM_WS_FRACTION:.0%} threshold) — the job would thrash "
            "(the documented 1.1 iters/s collapse)"
        )


def _full_history_frame(
    config: ExperimentConfig,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    orchestrator: ModelOrchestrator,
) -> pl.DataFrame:
    """Load train+validation with only needed columns.

    When the fit will spawn a subprocess (medium/full scale), the parent frame
    stays lightweight (era column only — the child re-reads from disk via the
    spawn spec); otherwise the full column set is loaded for the in-process
    fit.
    """
    cols = ["era", "id", *feature_cols, *target_cols]
    train_path = config.data.path("train.parquet")
    val_path = config.data.path("validation.parquet")
    for path, name in ((train_path, "train.parquet"), (val_path, "validation.parquet")):
        if not path.is_file():
            raise FileNotFoundError(
                f"full-version training requires {path} ({name} missing)"
            )
    agent = IngestionAgent(config.data)

    def _light(extra_cols: Sequence[str]) -> pl.DataFrame:
        return pl.concat(
            [
                pl.scan_parquet(train_path).select(["era", *extra_cols]).collect(),
                pl.scan_parquet(val_path).select(["era", *extra_cols]).collect(),
            ]
        )

    if orchestrator._should_spawn_full_history(_light([]), feature_cols):
        # Spawn path: keep the parent frame lightweight (era + target columns
        # only — target columns are required by train_full_history's
        # null-target filter; the spawned child re-reads the full column set
        # from disk via the spec).
        logger.info(
            "[promote] full-history fit will spawn a subprocess; "
            "parent frame kept lightweight (%d rows)",
            _light([]).height,
        )
        return _light(list(target_cols))
    return pl.concat(
        [
            agent.load("train", columns=cols),
            agent.load("validation", columns=cols),
        ]
    )


def resolve_champion_run_id(registry_dir: Path) -> str:
    """Read the atomic ``champion.json`` pointer's run_id."""
    champion_path = Path(registry_dir) / "champion.json"
    if not champion_path.is_file():
        raise FileNotFoundError(f"no champion: {champion_path} missing")
    try:
        payload = json.loads(champion_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"corrupt champion.json: {exc}") from exc
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    if not isinstance(run_id, str) or not _RID_RE.fullmatch(run_id):
        raise ValueError(f"champion.json has no valid run_id: {payload!r}")
    return run_id


def promote_full_version(
    run_id: str,
    family: str,
    *,
    models_dir: Path | None = None,
    registry_dir: Path | None = None,
    override_gate: bool = False,
    force: bool = False,
    data_dir: Path | None = None,
    rehearsal: bool = False,
) -> PromotionResult:
    """Train the full version (train+validation) for ``run_id`` and publish it.

    Enforces, in order: registry-run existence, the tier-4 promotion gate
    (refused without ``override_gate``; the verdict is recorded in the
    manifest either way), supplemental feature-set identity, and the measured
    RAM guard. Writes the immutable slot ``full/<run_id>/`` and repoints
    ``current.json`` atomically — **except for rehearsals** (``rehearsal=True``
    writes the slot with ``rehearsal: true`` + training provenance but never
    touches ``current.json``, so a rehearsal can never be read as the family's
    current full version). ``force`` permits overwriting an existing slot /
    repointing away from an existing current version. ``data_dir`` is the
    rehearsal override — train/validation are read from this directory
    instead of the stored config's, and the substitution is recorded in
    ``config_normalizations``.
    """
    models_dir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    registry_dir = Path(registry_dir) if registry_dir is not None else (
        DEFAULT_MODELS_DIR.parent / "registry"
    )
    validate_family_name(family)
    payload = _load_registry_run(registry_dir, run_id)
    manifest = payload.get("manifest") or {}
    stored_config = manifest.get("config")
    if not isinstance(stored_config, dict):
        raise ValueError(f"run {run_id!r} manifest has no config dict")
    scorecard = payload.get("scorecard") or {}
    if not scorecard:
        raise ValueError(
            f"run {run_id!r} has no validation scorecard; a promotion requires "
            "gate evidence"
        )

    gate = load_benchmark_file(_TIER4_GATE_YAML).gate
    if gate is None:
        raise ValueError(f"no gate in {_TIER4_GATE_YAML}")
    gate_passed, receipts = _evaluate_gate(scorecard, gate)
    if not gate_passed and not override_gate:
        violations = [
            f"{field}: {receipts[field]['measured']!r} vs "
            f"{receipts[field]['threshold']!r}"
            for field, _, _ in _HARD_GATE_FIELDS
            if receipts[field]["passed"] is False
        ]
        raise ValueError(
            f"run {run_id!r} fails the tier-4 promotion gate: "
            + "; ".join(violations)
            + "; pass override_gate=True to promote/rehearse anyway "
            "(recorded as tier4_gate_passed: false in the manifest)"
        )

    normalized, normalizations = _normalize_stored_config(stored_config)
    if data_dir is not None:
        original_data_dir = normalized.get("data", {}).get("data_dir")
        normalized["data"]["data_dir"] = str(Path(data_dir))
        normalizations.append(
            {
                "field": "data.data_dir",
                "from": original_data_dir,
                "to": str(Path(data_dir)),
            }
        )
    config = config_from_dict(normalized)
    _supplemental_identity_check(config, manifest)
    _ram_guard(config, models_dir)

    full_dir = models_dir / family / FULL_DIR_NAME
    slot_dir = full_dir / run_id
    if slot_dir.exists() and not force:
        raise ValueError(
            f"slot {slot_dir} already exists; pass force=True to overwrite "
            "(slots are immutable by design — prefer a new run)"
        )
    pointer = full_dir / CURRENT_POINTER_NAME
    if pointer.is_file():
        try:
            current = json.loads(pointer.read_text(encoding="utf-8"))
            current_id = current.get("run_id") if isinstance(current, dict) else None
        except (json.JSONDecodeError, OSError):
            current_id = None
        if current_id != run_id and not force:
            raise ValueError(
                f"current.json for family {family!r} points to {current_id!r}; "
                "repointing requires force=True"
            )

    slot_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = list(manifest.get("feature_cols") or [])
    if not feature_cols:
        raise ValueError(f"run {run_id!r} manifest has no feature_cols")
    target_cols = list(config.data.targets)
    weights = list(manifest.get("weights") or [])
    if len(weights) != len(target_cols):
        raise ValueError(
            f"run {run_id!r} weights ({len(weights)}) do not match targets "
            f"({len(target_cols)})"
        )
    proportion = float(config.risk.neutralization_proportion)

    orchestrator = ModelOrchestrator(config.model, seed=config.run.seed)
    frame = _full_history_frame(config, feature_cols, target_cols, orchestrator)
    # Training provenance: actual rows + era range the artifact was fit on —
    # first-class in the manifest (not buried in normalizations), so a
    # rehearsal (~68k rows on a truncated window) can never be mistaken for a
    # genuine full version (6.85M rows, full era range) at a glance.
    training_rows = frame.height
    era_series = frame.get_column("era").cast(pl.Int32)
    training_era_range = [int(era_series.min()), int(era_series.max())]
    predict_fn, model_meta = _build_deploy_pipeline(
        orchestrator=orchestrator,
        train_df=frame,
        feature_cols=feature_cols,
        target_cols=target_cols,
        weights=weights,
        proportion=proportion,
        data=config.data,
    )
    del frame
    artifact_path = slot_dir / "predict.pkl"
    _serialize_predict_artifact(
        predict_fn=predict_fn,
        model_meta=model_meta,
        artifact_path=artifact_path,
    )

    promoted_at = datetime.now(UTC).isoformat()
    slot_manifest = {
        "family": family,
        "training_scope": "full",
        "promoted_from_run_id": run_id,
        "promoted_at": promoted_at,
        "artifact_path": "predict.pkl",
        "config": json.loads(
            json.dumps(dataclasses.asdict(config), default=str)
        ),
        "tier4_gate_passed": bool(gate_passed),
        "tier4_receipts": receipts,
        "override_used": bool(override_gate),
        "config_normalizations": normalizations,
        # First-class rehearsal discriminator + training provenance (review
        # directive 2026-08-18): an artifact whose manifest overstates its own
        # training scope is the one thing we agreed never to ship.
        "rehearsal": bool(rehearsal),
        "training_rows": training_rows,
        "training_era_range": training_era_range,
    }
    manifest_path = slot_dir / FULL_MANIFEST_NAME
    atomic_write_text(manifest_path, json.dumps(slot_manifest, sort_keys=True, indent=2))
    if not rehearsal:
        atomic_write_text(
            pointer,
            json.dumps({"run_id": run_id, "promoted_at": promoted_at}, sort_keys=True),
        )
    logger.info(
        "[promote] %s published: %s (gate_passed=%s, override=%s, rows=%d)",
        "rehearsal" if rehearsal else "full version",
        artifact_path,
        gate_passed,
        override_gate,
        training_rows,
    )
    return PromotionResult(
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        run_id=run_id,
        family=family,
        tier4_gate_passed=bool(gate_passed),
        override_used=bool(override_gate),
        measured_peak_bytes=orchestrator.last_full_history_peak_bytes,
        measured_peak_commit_bytes=orchestrator.last_full_history_peak_commit_bytes,
    )


def _build_truncated_data(
    stored_config: dict[str, Any],
    rehearsal_root: Path,
    *,
    train_eras: int,
    validation_eras: int,
) -> tuple[Path, int]:
    """Build an on-disk truncated data dir (same schema, real features.json).

    Returns ``(truncated_dir, total_rows)``. The truncated dir mirrors the
    stored config's version layout so ``DataConfig(data_dir=rehearsal_root)``
    resolves the same paths against the subset files.
    """
    version = stored_config.get("data", {}).get("version", "v5.3")
    source_data_dir = Path(
        stored_config.get("data", {}).get(
            "data_dir", DEFAULT_MODELS_DIR.parent.parent / "data"
        )
    )
    truncated_dir = rehearsal_root / version
    truncated_dir.mkdir(parents=True, exist_ok=True)
    src_train = source_data_dir / version / "train.parquet"
    src_val = source_data_dir / version / "validation.parquet"
    src_features = source_data_dir / version / "features.json"
    for path, name in (
        (src_train, "train.parquet"),
        (src_val, "validation.parquet"),
        (src_features, "features.json"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"rehearsal requires {path} (data assets missing?)")
    train_eras_list = (
        pl.scan_parquet(src_train)
        .select("era")
        .unique()
        .collect()
        .get_column("era")
        .cast(pl.Int32)
        .sort()
        .to_list()
    )
    val_eras_list = (
        pl.scan_parquet(src_val)
        .select("era")
        .unique()
        .collect()
        .get_column("era")
        .cast(pl.Int32)
        .sort()
        .to_list()
    )
    if len(train_eras_list) < train_eras or len(val_eras_list) < validation_eras:
        raise ValueError(
            f"data has {len(train_eras_list)} train / {len(val_eras_list)} "
            f"validation eras; rehearsal needs {train_eras}/{validation_eras}"
        )
    selected_train = [f"{e:04d}" for e in train_eras_list[-train_eras:]]
    selected_val = [f"{e:04d}" for e in val_eras_list[:validation_eras]]
    (
        pl.scan_parquet(src_train)
        .filter(pl.col("era").cast(pl.Int32).is_in([int(e) for e in selected_train]))
        .collect()
        .write_parquet(truncated_dir / "train.parquet")
    )
    (
        pl.scan_parquet(src_val)
        .filter(pl.col("era").cast(pl.Int32).is_in([int(e) for e in selected_val]))
        .collect()
        .write_parquet(truncated_dir / "validation.parquet")
    )
    shutil.copyfile(src_features, truncated_dir / "features.json")
    rows = _scan_len(truncated_dir / "train.parquet") + _scan_len(
        truncated_dir / "validation.parquet"
    )
    return truncated_dir, rows


def measure_full_history_peak(
    stored_config: dict[str, Any],
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    weights: Sequence[float],
    *,
    data_dir: Path,
    seed: int = 42,
) -> tuple[int | None, int | None, int | None, int | None, int]:
    """Fit the full-history worker on ``data_dir`` and return the measured peaks.

    Returns ``(child_ws, child_commit, parent_ws, parent_commit, rows)`` —
    both the spawned worker's peak working set AND peak commit charge (the
    gate's quantity), plus the calling (parent) process's lifetime peaks
    (the fit must hold as combined commit against the machine). Runs the REAL
    promotion path (normalized config, spawn forced by
    ``NMR_FULL_HISTORY_SPAWN_MIN_BYTES``, train+validation via
    ``include_validation=True``) but publishes nothing — used by the RAM-curve
    measurement (measured, never estimated). The caller must restore the env
    override.
    """
    normalized, _ = _normalize_stored_config(stored_config)
    normalized["data"]["data_dir"] = str(Path(data_dir))
    config = config_from_dict(normalized)
    orchestrator = ModelOrchestrator(config.model, seed=seed)
    frame = _full_history_frame(config, feature_cols, target_cols, orchestrator)
    rows = frame.height
    orchestrator.train_full_history(
        frame,
        feature_cols=feature_cols,
        target_col=target_cols[0],
        era_col="era",
        data=config.data,
        include_validation=True,
    )
    from nmr.models import _peak_memory_counters

    parent_ws, parent_commit = _peak_memory_counters()
    return (
        orchestrator.last_full_history_peak_bytes,
        orchestrator.last_full_history_peak_commit_bytes,
        parent_ws,
        parent_commit,
        rows,
    )


def rehearse_promotion(
    run_id: str,
    family: str,
    *,
    models_dir: Path | None = None,
    registry_dir: Path | None = None,
    rehearsal_data_root: Path | None = None,
    train_eras: int = 6,
    validation_eras: int = 6,
    live_features_path: Path | None = None,
    live_benchmark_path: Path | None = None,
) -> RehearsalResult:
    """D7 Stage-1 rehearsal: prove the promotion writer end-to-end in minutes.

    Builds an on-disk truncated data dir (last ``train_eras`` train eras +
    first ``validation_eras`` validation eras, same schema, real
    ``features.json``), forces the fresh-process full-history path via
    ``NMR_FULL_HISTORY_SPAWN_MIN_BYTES``, promotes with ``override_gate=True``
    (the leading family fails tier-4 by design), measures the worker's peak
    RSS, extrapolates full-scale RAM into
    ``artifacts/reports/full_version_ram_estimate.json`` (the promotion RAM
    guard reads it), and validates the artifact's RAW contract output against
    the official validator on the real local ``live.parquet`` — the Phase D
    acceptance criterion (decision 10), which is NOT overridable.
    """
    models_dir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    registry_dir = Path(registry_dir) if registry_dir is not None else (
        DEFAULT_MODELS_DIR.parent / "registry"
    )
    payload = _load_registry_run(registry_dir, run_id)
    stored_config = (payload.get("manifest") or {}).get("config")
    if not isinstance(stored_config, dict):
        raise ValueError(f"run {run_id!r} manifest has no config dict")
    version = stored_config.get("data", {}).get("version", "v5.3")
    source_data_dir = Path(
        stored_config.get("data", {}).get("data_dir", DEFAULT_MODELS_DIR.parent.parent / "data")
    )

    rehearsal_root = (
        Path(rehearsal_data_root)
        if rehearsal_data_root is not None
        else DEFAULT_MODELS_DIR.parent / "cache" / "rehearsal_data"
    )
    truncated_dir, rows = _build_truncated_data(
        stored_config,
        rehearsal_root,
        train_eras=train_eras,
        validation_eras=validation_eras,
    )
    old_threshold = os.environ.get("NMR_FULL_HISTORY_SPAWN_MIN_BYTES")
    os.environ["NMR_FULL_HISTORY_SPAWN_MIN_BYTES"] = "1"  # force the spawn path
    try:
        result = promote_full_version(
            run_id,
            family,
            models_dir=models_dir,
            registry_dir=registry_dir,
            override_gate=True,
            data_dir=rehearsal_root,
            force=True,  # rehearsal regenerates its scratch slot on re-runs
            rehearsal=True,
        )
    finally:
        if old_threshold is None:
            os.environ.pop("NMR_FULL_HISTORY_SPAWN_MIN_BYTES", None)
        else:
            os.environ["NMR_FULL_HISTORY_SPAWN_MIN_BYTES"] = old_threshold

    # A rehearsal is never the family's current full version: the writer does
    # not repoint current.json, and any stale pointer is removed (review
    # directive 2026-08-18 — an artifact trained on the truncated subset must
    # not be readable as the deployed full version at a glance).
    pointer = Path(models_dir) / family / FULL_DIR_NAME / CURRENT_POINTER_NAME
    if pointer.exists():
        pointer.unlink()
        logger.info("[rehearse] removed stale current.json pointer (rehearsal is not a full version)")

    estimate_path = Path(models_dir).parent / "reports" / RAM_ESTIMATE_FILENAME
    estimate_path.parent.mkdir(parents=True, exist_ok=True)
    from nmr.models import _peak_memory_counters

    parent_ws, parent_commit = _peak_memory_counters()
    estimate_path.write_text(
        json.dumps(
            {
                "peak_bytes": result.measured_peak_bytes,
                "peak_commit_bytes": result.measured_peak_commit_bytes,
                "parent_peak_bytes": parent_ws,
                "parent_peak_commit_bytes": parent_commit,
                "train_validation_rows": rows,
                "measured_at": datetime.now(UTC).isoformat(),
                "note": (
                    "measured on the truncated rehearsal; the promotion RAM "
                    "guard gates on COMMIT (child extrapolated linearly by row "
                    "count + fixed parent commit)"
                ),
            },
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "[rehearse] child ws=%.1f MiB commit=%.1f MiB | parent ws=%.1f MiB "
        "commit=%.1f MiB on %d rows; estimate written to %s",
        (result.measured_peak_bytes or 0) / 2**20,
        (result.measured_peak_commit_bytes or 0) / 2**20,
        (parent_ws or 0) / 2**20,
        (parent_commit or 0) / 2**20,
        rows,
        estimate_path,
    )

    # Phase D acceptance criterion (decision 10): validate the RAW contract
    # output on the real local live.parquet via the official validator.
    live_features_path = Path(live_features_path) if live_features_path else (
        source_data_dir / version / "live.parquet"
    )
    live_benchmark_path = Path(live_benchmark_path) if live_benchmark_path else (
        source_data_dir / version / "live_benchmark_models.parquet"
    )
    feature_cols = list((payload.get("manifest") or {}).get("feature_cols") or [])
    if not feature_cols:
        raise ValueError(f"run {run_id!r} manifest has no feature_cols")
    live_df = pl.read_parquet(live_features_path).select(["era", "id", *feature_cols])
    live_pd = live_df.to_pandas().set_index("id")
    bench_pd = None
    if live_benchmark_path.is_file():
        schema = pl.read_parquet_schema(live_benchmark_path)
        bench_cols = [c for c in schema if c not in {"era", "id"}]
        if bench_cols:
            bench_pd = pl.read_parquet(
                live_benchmark_path, columns=["era", "id", bench_cols[0]]
            ).to_pandas()
    from nmr.submission import accept_promoted_artifact

    try:
        accept_promoted_artifact(
            result.artifact_path,
            live_features=live_pd,
            live_benchmark_models=bench_pd,
        )
        acceptance_passed = True
    except ValueError as exc:
        logger.error("[rehearse] acceptance FAILED: %s", exc)
        raise
    return RehearsalResult(
        artifact_path=result.artifact_path,
        ram_estimate_path=estimate_path,
        acceptance_passed=acceptance_passed,
        measured_peak_bytes=result.measured_peak_bytes,
        measured_peak_commit_bytes=result.measured_peak_commit_bytes,
        train_validation_rows=rows,
    )
