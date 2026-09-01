"""Lifecycle: export validity, stage derivation (a total function), ordering.

Derives the six lifecycle stages from filesystem state. Importing this module
must not trigger heavy loads; ``valid_export`` calls ``load_predict`` (hash-
verified) per the spec's validity predicate — trusted-source rule applies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nmr import paths

logger = logging.getLogger("nmr.lifecycle")

SCOPES = ("partial", "full")
LIFECYCLE_STAGES = (
    "uninitialized",
    "research",
    "partial",
    "degraded",
    "full",
    "staked",
)

# Badge precedence: staked > full > degraded > partial > research > uninitialized.
_PRECEDENCE = {stage: i for i, stage in enumerate(LIFECYCLE_STAGES)}

_STAGE_ORDER_SCORES: dict[str, int] = {
    "uninitialized": 0,
    "research": 1,
    "partial": 2,
    "degraded": 3,
    "full": 4,
    "staked": 5,
}


@dataclass(frozen=True)
class StakedRecord:
    run_id: str
    scope: str
    numerai_model_id: str | None
    staked_at: str | None
    status: str


@dataclass(frozen=True)
class ExportVersion:
    family: str
    scope: str
    run_id: str
    slot_dir: Path
    training_scope: str
    promoted_at: str | None
    training_rows: int | None
    training_era_range: tuple[int, int] | None
    config: dict[str, Any]
    tier4_gate_passed: bool | None
    rehearsal: bool
    override_used: bool = False
    acceptance_passed: bool = False
    activation_eligible: bool = False


def activation_eligible(version: ExportVersion) -> bool:
    """Whether a structurally valid full export may be made current/staked."""
    return bool(
        version.scope == "full"
        and version.tier4_gate_passed is True
        and not version.override_used
        and not version.rehearsal
        and version.acceptance_passed
        and version.activation_eligible
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _verified_acceptance(
    payload: dict[str, Any], artifact_path: Path, artifact_sha256: str
) -> bool:
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("passed") is not True:
        return False
    if acceptance.get("artifact_sha256") != artifact_sha256:
        return False
    feature_cols = payload.get("feature_cols")
    if not isinstance(feature_cols, list) or not feature_cols:
        return False
    if acceptance.get("feature_cols_sha256") != _sha256_json(feature_cols):
        return False
    config = payload.get("config")
    if not isinstance(config, dict):
        return False
    data = config.get("data")
    if not isinstance(data, dict):
        return False
    version = data.get("version")
    data_dir = data.get("data_dir")
    if not isinstance(version, str) or not isinstance(data_dir, str):
        return False
    if acceptance.get("data_version") != version:
        return False
    version_dir = Path(data_dir) / version
    live_path = version_dir / "live.parquet"
    benchmark_path = version_dir / "live_benchmark_models.parquet"
    if not live_path.is_file() or not benchmark_path.is_file():
        return False
    if acceptance.get("live_features_sha256") != _sha256_of(live_path):
        return False
    if acceptance.get("live_benchmark_models_sha256") != _sha256_of(benchmark_path):
        return False
    try:
        import polars as pl

        live = pl.read_parquet(live_path, columns=["id"])
        benchmark_schema = pl.read_parquet_schema(benchmark_path)
    except Exception:  # noqa: BLE001 - malformed evidence is ineligible
        return False
    benchmark_cols = [c for c in benchmark_schema if c not in {"era", "id"}]
    if not benchmark_cols:
        return False
    live_ids = live.get_column("id").cast(pl.String).to_list()
    evidence_matches = bool(
        acceptance.get("n_rows") == live.height
        and acceptance.get("live_ids_sha256") == _sha256_json(live_ids)
        and acceptance.get("benchmark_columns") == benchmark_cols
        and artifact_path.is_file()
    )
    if not evidence_matches:
        return False
    try:
        live_frame = pl.read_parquet(live_path).select(["era", "id", *feature_cols])
        live_features = live_frame.to_pandas().set_index("id")
        benchmark_frame = pl.read_parquet(
            benchmark_path,
            columns=["era", "id", benchmark_cols[0]],
        ).to_pandas()
        from nmr.submission import accept_promoted_artifact

        accept_promoted_artifact(
            artifact_path,
            live_features=live_features,
            live_benchmark_models=benchmark_frame,
        )
    except Exception:  # noqa: BLE001 - failed replay is never activatable
        return False
    return True


def _verified_run_identity(export: dict[str, Any], run_payload: dict[str, Any]) -> bool:
    run_manifest = run_payload.get("manifest")
    scorecard = run_payload.get("scorecard")
    if not isinstance(run_manifest, dict) or not isinstance(scorecard, dict):
        return False
    canonical_scorecard = {
        key: value
        for key, value in scorecard.items()
        if not key.startswith(("timing_", "quality_metric"))
    }
    if export.get("scorecard_sha256") != _sha256_json(canonical_scorecard):
        return False
    for field in ("data_fingerprint", "promotion_data_fingerprint"):
        value = run_manifest.get(field)
        if not isinstance(value, str) or export.get(field) != value:
            return False

    config = run_manifest.get("config")
    if not isinstance(config, dict):
        return False
    data = config.get("data") or {}
    evaluation = config.get("evaluation") or {}
    targets = data.get("targets")
    pred_cols = run_manifest.get("pred_cols")
    weights = run_manifest.get("weights")
    if not isinstance(targets, list) or not isinstance(pred_cols, list):
        return False
    if not isinstance(weights, list) or len(weights) != len(targets):
        return False
    expected_preds = [f"pred_{target}" for target in targets]
    if pred_cols != expected_preds:
        return False
    try:
        numeric_weights = [float(weight) for weight in weights]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(weight) for weight in numeric_weights):
        return False
    components = [
        {"target": target, "prediction_col": pred, "weight": weight}
        for target, pred, weight in zip(targets, pred_cols, numeric_weights)
    ]
    if export.get("components") != components:
        return False
    if export.get("components_sha256") != _sha256_json(components):
        return False

    policy_id = evaluation.get("payout_policy")
    try:
        from nmr.payout import resolve_payout_policy

        policy = resolve_payout_policy(policy_id)
    except (TypeError, ValueError):
        return False
    expected_target = policy.target or evaluation.get("main_target")
    expected_horizon = policy.scoring_horizon or data.get("horizon")
    expected_policy = {
        "payout_policy_id": policy.policy_id,
        "scoring_target": expected_target,
        "scoring_horizon": expected_horizon,
    }
    for field, expected in expected_policy.items():
        if export.get(field) != expected or scorecard.get(field) != expected:
            return False
    receipts = export.get("tier4_receipts")
    if not isinstance(receipts, dict):
        return False
    for field, expected in expected_policy.items():
        receipt = receipts.get(field)
        if not isinstance(receipt, dict):
            return False
        if receipt.get("measured") != expected or receipt.get("passed") is not True:
            return False
    for field in (
        "corr",
        "corr_sharpe_ac",
        "fnc",
        "gain_to_pain_ratio",
    ):
        receipt = receipts.get(field)
        if not isinstance(receipt, dict) or receipt.get("measured") != scorecard.get(
            field
        ):
            return False
    return True


def load_staked_record(meta_path: Path) -> StakedRecord | None:
    """Read ``meta.json``'s staked record; None when absent or malformed."""
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    staked = payload.get("staked") if isinstance(payload, dict) else None
    if not isinstance(staked, dict) or not isinstance(staked.get("run_id"), str):
        return None
    return StakedRecord(
        run_id=staked["run_id"],
        scope=staked.get("scope", "full"),
        numerai_model_id=staked.get("numerai_model_id"),
        staked_at=staked.get("staked_at"),
        status=str(staked.get("status", "")),
    )


def _read_export_json(slot_dir: Path) -> dict[str, Any] | None:
    p = slot_dir / "export.json"
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_run_record(family: str, run_id: str) -> dict[str, Any] | None:
    """Read ``experiments/<family>/runs/<run_id>/run.json``; None when absent
    or malformed (identity binding: an export is valid only when its run
    record exists and agrees — a slot without a run record is an orphan)."""
    p = paths.run_json_path(family, run_id)
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def valid_export(family: str, scope: str, run_id: str) -> ExportVersion | None:
    """Validate one export slot; None on any violation. Identity binding: slot
    dir run_id == manifest.promoted_from_run_id == family slug match, AND the
    run record is present and agrees (``run.json`` payload ``run_id`` == slot
    run_id, and its manifest config ``run.name``, when present, == family). An
    export without a run record is an orphan — invalid, never render-valid."""
    if scope not in SCOPES:
        return None
    slot_dir = paths.export_dir(family, scope, run_id)
    payload = _read_export_json(slot_dir)
    if payload is None:
        return None
    if payload.get("family") != family:
        return None
    if payload.get("promoted_from_run_id") != run_id:
        return None
    training_scope = payload.get("training_scope")
    if training_scope != scope:
        return None
    # Run-record identity binding (2026-08-26 review, BLOCKING 2): the slot is
    # valid only when its immutable run record exists and agrees on identity.
    run_payload = _read_run_record(family, run_id)
    if run_payload is None:
        return None
    if run_payload.get("run_id") != run_id:
        return None
    run_name = (
        ((run_payload.get("manifest") or {}).get("config") or {}).get("run") or {}
    ).get("name")
    if run_name is not None and run_name != family:
        return None
    pkl = slot_dir / "predict.pkl"
    sibling = slot_dir / "predict.pkl.manifest.json"
    if not pkl.is_file() or not sibling.is_file():
        return None
    try:
        manifest = json.loads(sibling.read_text(encoding="utf-8"))
        expected = manifest.get("sha256")
    except (json.JSONDecodeError, OSError):
        return None
    artifact_sha256 = _sha256_of(pkl)
    if not isinstance(expected, str) or expected != artifact_sha256:
        return None
    # Verified loadability (trusted-source: only repo artifacts reach here).
    from nmr.deployment import load_predict

    try:
        load_predict(pkl)
    except Exception:  # noqa: BLE001 - any failure => invalid
        return None
    if scope == "partial" and not (slot_dir / "scorecard.json").is_file():
        return None
    era_range = payload.get("training_era_range")
    # Malformed numeric metadata invalidates the slot (never aborts a scan —
    # the caller's conversion exceptions are contained here).
    try:
        training_rows = None
        if "training_rows" in payload and payload["training_rows"] is not None:
            training_rows = int(payload["training_rows"])
        parsed_era_range = None
        if isinstance(era_range, list) and len(era_range) == 2:
            parsed_era_range = (int(era_range[0]), int(era_range[1]))
    except (TypeError, ValueError):
        return None
    return ExportVersion(
        family=family,
        scope=scope,
        run_id=run_id,
        slot_dir=slot_dir,
        training_scope=str(training_scope),
        promoted_at=payload.get("promoted_at"),
        training_rows=training_rows,
        training_era_range=parsed_era_range,
        config=payload.get("config") if isinstance(payload.get("config"), dict) else {},
        tier4_gate_passed=payload.get("tier4_gate_passed"),
        rehearsal=bool(payload.get("rehearsal", False)),
        override_used=bool(payload.get("override_used", False)),
        acceptance_passed=bool((payload.get("acceptance") or {}).get("passed", False)),
        activation_eligible=bool(
            payload.get("activation_eligible", False)
            and _verified_run_identity(payload, run_payload)
            and _verified_acceptance(payload, pkl, artifact_sha256)
        ),
    )


def _sha256_of(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_valid_exports(family: str, scope: str) -> list[ExportVersion]:
    """All VALID exports of one scope for a family, sorted (deterministic)."""
    if scope not in SCOPES:
        return []
    base = paths.experiment_dir(family) / "exports" / scope
    if not base.is_dir():
        return []
    found = []
    for entry in sorted(base.iterdir()):
        # Non-hex names are never run_ids — skip them (paths.export_dir would
        # refuse; the total scan must survive a stray directory).
        if not paths.RID_RE.fullmatch(entry.name):
            continue
        if entry.is_dir() and not entry.name.startswith(".tmp-"):
            version = valid_export(family, scope, entry.name)
            if version is not None:
                found.append(version)
    return sort_exports(found)


def sort_exports(exports: list[ExportVersion]) -> list[ExportVersion]:
    """ISO-8601 ``promoted_at`` descending; unparseable sorts last; tie-break
    ``run_id`` ascending."""

    def key(e: ExportVersion):
        try:
            if e.promoted_at is None:
                raise ValueError("missing")
            from datetime import datetime

            ts = datetime.fromisoformat(e.promoted_at)
            return (0, -ts.timestamp(), e.run_id)
        except (ValueError, TypeError):
            return (1, 0.0, e.run_id)

    return sorted(exports, key=key)


def current_full_status(family: str) -> Literal["full", "degraded", "none"]:
    """'full' when the pointer resolves to a valid full slot; 'degraded' when
    valid full slots exist but the pointer is missing/dangling; else 'none'.

    A syntactically valid pointer whose run_id is not a 64-hex string is
    treated as corrupt/invalid (never resolved, never crashes the total scan —
    2026-08-29 re-review SECONDARY finding)."""
    pointer = paths.current_pointer_path(family)
    activatable_full = [
        version
        for version in scan_valid_exports(family, "full")
        if activation_eligible(version)
    ]
    if not activatable_full:
        return "none"
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        run_id = payload.get("run_id") if isinstance(payload, dict) else None
    except (json.JSONDecodeError, OSError):
        run_id = None
    if (
        isinstance(run_id, str)
        and paths.RID_RE.fullmatch(run_id) is not None
        and (version := valid_export(family, "full", run_id)) is not None
        and activation_eligible(version)
    ):
        return "full"
    return "degraded"


def _has_runs(family: str) -> bool:
    runs_dir = paths.experiment_dir(family) / "runs"
    return runs_dir.is_dir() and any(runs_dir.iterdir())


def derive_stage(family: str, staked: StakedRecord | None) -> tuple[str, str]:
    """Total function over filesystem state -> (lifecycle_stage, current_full_status)."""
    status = current_full_status(family)
    stage_score = 0
    if _has_runs(family):
        stage_score = _STAGE_ORDER_SCORES["research"]
    if any(scan_valid_exports(family, "partial")):
        stage_score = max(stage_score, _STAGE_ORDER_SCORES["partial"])
    if status in ("full", "degraded"):
        stage_score = max(stage_score, _STAGE_ORDER_SCORES[status])
    if (
        staked is not None
        and staked.status == "active"
        and staked.scope == "full"
        and paths.RID_RE.fullmatch(staked.run_id) is not None
        and (version := valid_export(family, "full", staked.run_id)) is not None
        and activation_eligible(version)
    ):
        stage_score = _STAGE_ORDER_SCORES["staked"]
    stage = next(s for s, v in _STAGE_ORDER_SCORES.items() if v == stage_score)
    return stage, status
