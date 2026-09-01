"""Tests for lifecycle: export validity, total stage derivation, ordering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from nmr import lifecycle, paths
from nmr.deployment import serialize_predict
from nmr.lifecycle import ExportVersion, StakedRecord


@pytest.fixture(autouse=True)
def _isolated_experiments_root(tmp_path, monkeypatch) -> None:
    """Route all experiment paths under tmp_path; never touch the real repo root."""
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")


def _write_export(
    slug,
    scope,
    run_id,
    *,
    training_scope=None,
    sha=None,
    meta_json=None,
    write_run_record: bool = True,
    activatable: bool = True,
):
    slot = paths.export_dir(slug, scope, run_id)
    slot.mkdir(parents=True, exist_ok=True)

    # Write a real deploy artifact: valid_export()'s validity predicate calls
    # load_predict(), which hash-verifies and cloudpickle-loads predict.pkl.
    def dummy_predict(live_features, live_benchmark_models=None):
        import pandas as pd

        return pd.DataFrame({"prediction": [0.25, 0.75]}, index=live_features.index)

    serialize_predict(dummy_predict, path=slot / "predict.pkl", feature_names=["f1"])
    if sha is not None:
        (slot / "predict.pkl.manifest.json").write_text(json.dumps({"sha256": sha}))
    ts = training_scope or scope
    data_root = paths.EXPERIMENTS_ROOT.parent / "data" / slug
    version_dir = data_root / "vtest"
    version_dir.mkdir(parents=True, exist_ok=True)
    live_ids = ["live_0", "live_1"]
    pl.DataFrame({"era": ["1", "1"], "id": live_ids, "f1": [0.1, 0.2]}).write_parquet(
        version_dir / "live.parquet"
    )
    pl.DataFrame(
        {"era": ["1", "1"], "id": live_ids, "benchmark": [0.2, 0.3]}
    ).write_parquet(version_dir / "live_benchmark_models.parquet")

    def file_sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def json_sha(value: object) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    artifact_sha = file_sha(slot / "predict.pkl")
    policy_id = "classic_legacy_075_225_clip005_v1"
    scorecard = {
        "payout_policy_id": policy_id,
        "scoring_target": "target",
        "scoring_horizon": "20D",
        "corr": 0.05,
        "corr_sharpe_ac": 1.0,
        "fnc": 0.03,
        "gain_to_pain_ratio": 2.0,
        "cagr_1y": 0.1,
    }
    components = [{"target": "target", "prediction_col": "pred_target", "weight": 1.0}]
    data_fingerprint = "d" * 64
    promotion_data_fingerprint = "e" * 64
    receipts = {
        field: {"measured": scorecard[field], "passed": True}
        for field in (
            "corr",
            "corr_sharpe_ac",
            "fnc",
            "gain_to_pain_ratio",
            "cagr_1y",
        )
    }
    receipts.update(
        {
            "payout_policy_id": {
                "measured": policy_id,
                "passed": True,
            },
            "scoring_target": {"measured": "target", "passed": True},
            "scoring_horizon": {"measured": "20D", "passed": True},
        }
    )
    (slot / "export.json").write_text(
        json.dumps(
            {
                "family": slug,
                "training_scope": ts,
                "promoted_from_run_id": run_id,
                "promoted_at": "2026-08-26T10:00:00+00:00",
                "config": {
                    "data": {
                        "version": "vtest",
                        "data_dir": str(data_root),
                        "targets": ["target"],
                        "horizon": "20D",
                    },
                    "evaluation": {
                        "main_target": "target",
                        "payout_policy": policy_id,
                    },
                },
                "feature_cols": ["f1"],
                "components": components,
                "components_sha256": json_sha(components),
                "scorecard_sha256": json_sha(scorecard),
                "data_fingerprint": data_fingerprint,
                "promotion_data_fingerprint": promotion_data_fingerprint,
                "payout_policy_id": policy_id,
                "scoring_target": "target",
                "scoring_horizon": "20D",
                "tier4_receipts": receipts,
                "tier4_gate_passed": activatable,
                "override_used": False,
                "rehearsal": False,
                "activation_eligible": activatable,
                "acceptance": {
                    "passed": activatable,
                    "artifact_sha256": artifact_sha,
                    "live_features_sha256": file_sha(version_dir / "live.parquet"),
                    "live_benchmark_models_sha256": file_sha(
                        version_dir / "live_benchmark_models.parquet"
                    ),
                    "live_ids_sha256": json_sha(live_ids),
                    "feature_cols_sha256": json_sha(["f1"]),
                    "benchmark_columns": ["benchmark"],
                    "data_version": "vtest",
                    "n_rows": 2,
                },
            }
        )
    )
    if scope == "partial":
        (slot / "scorecard.json").write_text(json.dumps({"schema_version": 3}))
    if meta_json is not None:
        paths.experiment_dir(slug).joinpath("meta.json").write_text(
            json.dumps(meta_json)
        )
    # Identity binding (2026-08-26 review, BLOCKING 2): valid_export requires
    # the run record to exist and agree — write it unless the test is
    # deliberately constructing an orphan.
    if write_run_record:
        run_dir = paths.run_dir(slug, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "manifest": {
                        "config": {
                            "run": {"name": slug},
                            "data": {
                                "targets": ["target"],
                                "horizon": "20D",
                            },
                            "evaluation": {
                                "main_target": "target",
                                "payout_policy": policy_id,
                            },
                        },
                        "pred_cols": ["pred_target"],
                        "weights": [1.0],
                        "data_fingerprint": data_fingerprint,
                        "promotion_data_fingerprint": promotion_data_fingerprint,
                    },
                    "scorecard": scorecard,
                }
            ),
            encoding="utf-8",
        )
    return slot


def test_stage_derivation_total():
    # uninitialized: dir exists, no run.json
    paths.experiment_dir("fam1").mkdir(parents=True, exist_ok=True)
    assert lifecycle.derive_stage("fam1", None) == ("uninitialized", "none")
    # research: run.json present, no exports
    paths.run_json_path("fam1", "a" * 64).parent.mkdir(parents=True, exist_ok=True)
    paths.run_json_path("fam1", "a" * 64).write_text("{}")
    assert lifecycle.derive_stage("fam1", None) == ("research", "none")
    # partial: valid partial export, no full
    _write_export("fam1", "partial", "b" * 64)
    assert lifecycle.derive_stage("fam1", None) == ("partial", "none")
    # degraded: valid full export, dangling pointer
    _write_export("fam1", "full", "c" * 64)
    assert lifecycle.derive_stage("fam1", None) == ("degraded", "degraded")
    # full: pointer at valid slot
    paths.current_pointer_path("fam1").write_text(json.dumps({"run_id": "c" * 64}))
    assert lifecycle.derive_stage("fam1", None) == ("full", "full")
    # staked: active stake on valid full
    staked = StakedRecord(
        run_id="c" * 64,
        scope="full",
        numerai_model_id="m1",
        staked_at="2026-08-26T11:00:00+00:00",
        status="active",
    )
    assert lifecycle.derive_stage("fam1", staked) == ("staked", "full")


def test_staked_stale_when_export_invalid():
    staked = StakedRecord(
        run_id="dead" * 16,
        scope="full",
        numerai_model_id="m1",
        staked_at="2026-08-26T11:00:00+00:00",
        status="active",
    )
    assert lifecycle.derive_stage("fam1", staked)[0] != "staked"


def test_tampered_acceptance_receipt_is_not_activation_eligible() -> None:
    slot = _write_export("tampered", "full", "a" * 64)
    payload = json.loads((slot / "export.json").read_text(encoding="utf-8"))
    payload["acceptance"]["artifact_sha256"] = "0" * 64
    (slot / "export.json").write_text(json.dumps(payload), encoding="utf-8")
    paths.current_pointer_path("tampered").write_text(
        json.dumps({"run_id": "a" * 64}), encoding="utf-8"
    )

    version = lifecycle.valid_export("tampered", "full", "a" * 64)
    assert version is not None
    assert lifecycle.activation_eligible(version) is False
    assert lifecycle.current_full_status("tampered") == "none"
    assert lifecycle.derive_stage("tampered", None)[0] == "research"


def test_export_identity_binding():
    # manifest promoted_from_run_id != slot dir run_id -> invalid
    slot = _write_export("fam2", "full", "a" * 64)
    (slot / "export.json").write_text(
        json.dumps(
            {
                "family": "fam2",
                "training_scope": "full",
                "promoted_from_run_id": "b" * 64,
                "promoted_at": "x",
                "config": {},
            }
        )
    )
    assert lifecycle.valid_export("fam2", "full", "a" * 64) is None


def test_sort_exports_deterministic():
    e1 = ExportVersion(
        family="f",
        scope="partial",
        run_id="b" * 64,
        slot_dir=Path("."),
        training_scope="partial",
        promoted_at="2026-08-26T10:00:00+00:00",
        training_rows=None,
        training_era_range=None,
        config={},
        tier4_gate_passed=None,
        rehearsal=False,
    )
    e2 = ExportVersion(
        family="f",
        scope="partial",
        run_id="a" * 64,
        slot_dir=Path("."),
        training_scope="partial",
        promoted_at="2026-08-26T10:00:00+00:00",
        training_rows=None,
        training_era_range=None,
        config={},
        tier4_gate_passed=None,
        rehearsal=False,
    )
    e3 = ExportVersion(
        family="f",
        scope="partial",
        run_id="c" * 64,
        slot_dir=Path("."),
        training_scope="partial",
        promoted_at="2026-08-25T10:00:00+00:00",
        training_rows=None,
        training_era_range=None,
        config={},
        tier4_gate_passed=None,
        rehearsal=False,
    )
    got = lifecycle.sort_exports([e1, e2, e3])
    assert [e.run_id for e in got] == ["a" * 64, "b" * 64, "c" * 64]


def test_orphan_export_without_run_record_invalid():
    """BLOCKING 2: an export without its run record is an orphan — invalid,
    never render-valid; the run record must exist AND agree."""
    slot = _write_export("orphan-fam", "full", "a" * 64, write_run_record=False)
    assert lifecycle.valid_export("orphan-fam", "full", "a" * 64) is None
    assert lifecycle.scan_valid_exports("orphan-fam", "full") == []
    # With the run record present, the same slot is valid.
    _write_export("orphan-fam", "full", "a" * 64, write_run_record=True)
    assert lifecycle.valid_export("orphan-fam", "full", "a" * 64) is not None
    assert slot is not None


def test_run_record_identity_mismatch_invalid():
    """BLOCKING 2: run.json whose run_id disagrees with the slot run_id, or
    whose manifest run.name disagrees with the family, invalidates the slot."""
    slot = _write_export("fam-mismatch", "full", "a" * 64)
    run_dir = paths.run_dir("fam-mismatch", "a" * 64)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "b" * 64,
                "manifest": {"config": {"run": {"name": "fam-mismatch"}}},
            }
        ),
        encoding="utf-8",
    )
    assert lifecycle.valid_export("fam-mismatch", "full", "a" * 64) is None
    # run.name disagrees with the family slug.
    (run_dir / "run.json").write_text(
        json.dumps(
            {"run_id": "a" * 64, "manifest": {"config": {"run": {"name": "other-fam"}}}}
        ),
        encoding="utf-8",
    )
    assert lifecycle.valid_export("fam-mismatch", "full", "a" * 64) is None
    # run.name absent is tolerated; agreement restores validity.
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "a" * 64, "manifest": {"config": {"run": {}}}}),
        encoding="utf-8",
    )
    assert lifecycle.valid_export("fam-mismatch", "full", "a" * 64) is not None
    assert slot is not None


def test_malformed_staked_run_id_is_not_staked_not_a_crash():
    """2026-08-29 re-review (fixer-flagged residual): a meta.json staked
    run_id that is not 64-hex must not let ``paths.export_dir``'s ValueError
    escape ``derive_stage`` — the stake is treated as corrupt/absent (same as
    ``load_staked_record`` returning None), the underlying stage shows, and the
    total function never crashes."""
    _write_export("fam-badstake", "full", "a" * 64)
    bad = StakedRecord(
        run_id="../bad",
        scope="full",
        numerai_model_id="m1",
        staked_at="2026-08-26T11:00:00+00:00",
        status="active",
    )
    # No crash; the corrupt stake does not lift the stage (pointer missing ->
    # degraded; a valid staked run_id would lift to staked only when the
    # pointer resolves).
    assert lifecycle.derive_stage("fam-badstake", bad) == ("degraded", "degraded")

    # A valid full pointer + a malformed stake: still never 'staked'.
    paths.current_pointer_path("fam-badstake").write_text(
        json.dumps({"run_id": "a" * 64}), encoding="utf-8"
    )
    assert lifecycle.derive_stage("fam-badstake", bad) == ("full", "full")

    # A well-formed active stake on the resolved slot lifts to 'staked'.
    good = StakedRecord(
        run_id="a" * 64,
        scope="full",
        numerai_model_id="m1",
        staked_at="2026-08-26T11:00:00+00:00",
        status="active",
    )
    assert lifecycle.derive_stage("fam-badstake", good) == ("staked", "full")


def test_scan_survives_malformed_numeric_metadata():
    """SECONDARY 3: a slot with NaN (or non-numeric) training_rows is INVALID
    and scan_valid_exports still returns the other valid slots — one
    malformed export never aborts the total scan."""
    _write_export("fam-scan", "full", "a" * 64)
    bad = _write_export("fam-scan", "full", "b" * 64)
    payload = json.loads((bad / "export.json").read_text(encoding="utf-8"))
    payload["training_rows"] = float("nan")  # serializes as NaN JSON literal
    (bad / "export.json").write_text(json.dumps(payload), encoding="utf-8")
    assert lifecycle.valid_export("fam-scan", "full", "b" * 64) is None
    found = lifecycle.scan_valid_exports("fam-scan", "full")
    assert [v.run_id for v in found] == ["a" * 64]

    # Non-numeric string training_rows is malformed too — invalid, scan intact.
    payload["training_rows"] = "not-a-number"
    (bad / "export.json").write_text(json.dumps(payload), encoding="utf-8")
    assert lifecycle.valid_export("fam-scan", "full", "b" * 64) is None
    assert [v.run_id for v in lifecycle.scan_valid_exports("fam-scan", "full")] == [
        "a" * 64
    ]

    # Malformed era range likewise invalidates only that slot.
    payload["training_rows"] = 100
    payload["training_era_range"] = [float("nan"), 10]
    (bad / "export.json").write_text(json.dumps(payload), encoding="utf-8")
    assert lifecycle.valid_export("fam-scan", "full", "b" * 64) is None
    assert [v.run_id for v in lifecycle.scan_valid_exports("fam-scan", "full")] == [
        "a" * 64
    ]


def test_malformed_pointer_run_id_is_degraded_not_a_crash():
    """SECONDARY (2026-08-29 re-review): a syntactically valid ``current.json``
    whose run_id is not a 64-hex string (``"../bad"``) must not let the
    ValueError escape ``valid_export``'s path validation — the pointer is
    treated as invalid (same as corrupt/missing): 'degraded' when valid full
    slots exist, 'none' otherwise, and ``derive_stage`` stays total."""
    _write_export("fam-badptr", "full", "a" * 64)
    paths.current_pointer_path("fam-badptr").write_text(
        json.dumps({"run_id": "../bad"}), encoding="utf-8"
    )
    assert lifecycle.current_full_status("fam-badptr") == "degraded"
    assert lifecycle.derive_stage("fam-badptr", None) == ("degraded", "degraded")

    # A non-string run_id in the pointer is equally invalid.
    paths.current_pointer_path("fam-badptr").write_text(
        json.dumps({"run_id": 123}), encoding="utf-8"
    )
    assert lifecycle.current_full_status("fam-badptr") == "degraded"

    # Without any valid full slot the status is 'none' regardless of pointer.
    paths.experiment_dir("fam-noslot").mkdir(parents=True)
    no_slot_pointer = paths.current_pointer_path("fam-noslot")
    no_slot_pointer.parent.mkdir(parents=True, exist_ok=True)
    no_slot_pointer.write_text(json.dumps({"run_id": "../bad"}), encoding="utf-8")
    assert lifecycle.current_full_status("fam-noslot") == "none"
    assert lifecycle.derive_stage("fam-noslot", None) == ("uninitialized", "none")

    # A well-formed pointer still resolves to 'full'.
    paths.current_pointer_path("fam-badptr").write_text(
        json.dumps({"run_id": "a" * 64}), encoding="utf-8"
    )
    assert lifecycle.current_full_status("fam-badptr") == "full"
