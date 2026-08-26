"""Unit tests for nmr.families — read-only full-version discovery.

Layout under test: ``experiments/<family>/exports/full/<run_id>/export.json``
(one immutable slot per promoted run) + atomic ``current.json`` pointer, the
experiment layout owned by ``nmr.paths`` / ``nmr.lifecycle``. Resolution is
pointer-driven and fails loud on a missing/dangling pointer; ``nmr.families``
is a thin compatibility wrapper over ``nmr.lifecycle``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import nmr.families as fam
from nmr import lifecycle, paths
from nmr.deployment import serialize_predict


@pytest.fixture(autouse=True)
def _isolated_experiments_root(tmp_path, monkeypatch) -> None:
    """Route all experiment paths under tmp_path; never touch the real repo root."""
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")


def _write_export(
    slug: str,
    scope: str,
    run_id: str,
    *,
    family: str | None = None,
    training_scope: str | None = None,
    promoted_from_run_id: str | None = None,
    promoted_at: str = "2026-08-26T10:00:00+00:00",
    rehearsal: bool = False,
    training_rows: int | None = None,
    training_era_range: tuple[int, int] | None = None,
    body: str | None = None,
    write_pointer: bool = False,
) -> Path:
    """Write a REAL valid export slot under the (monkeypatched) experiments root.

    ``serialize_predict`` writes the sha256 sibling manifest, so
    ``lifecycle.valid_export``'s hash-verified loadability predicate holds.
    The ``current.json`` pointer is optional — callers control it explicitly.
    """
    slot = paths.export_dir(slug, scope, run_id)
    slot.mkdir(parents=True, exist_ok=True)

    def dummy_predict(live_features, live_benchmark_models=None):
        return live_features

    serialize_predict(dummy_predict, path=slot / "predict.pkl", feature_names=["f1"])
    if body is None:
        body = json.dumps(
            {
                "family": family if family is not None else slug,
                "training_scope": training_scope or scope,
                "promoted_from_run_id": (
                    promoted_from_run_id
                    if promoted_from_run_id is not None
                    else run_id
                ),
                "promoted_at": promoted_at,
                "artifact_path": "predict.pkl",
                "config": {"run": {"name": slug}},
                "rehearsal": rehearsal,
                "training_rows": (
                    training_rows if training_rows is not None else 6_853_308
                ),
                "training_era_range": list(
                    training_era_range if training_era_range is not None else [1, 1231]
                ),
                "tier4_gate_passed": True,
            }
        )
    (slot / "export.json").write_text(body, encoding="utf-8")
    if write_pointer:
        paths.current_pointer_path(slug).write_text(
            json.dumps({"run_id": run_id, "promoted_at": promoted_at}),
            encoding="utf-8",
        )
    return slot


def test_full_version_is_lifecycle_export_version() -> None:
    assert fam.FullVersion is lifecycle.ExportVersion


def test_full_manifest_path_resolves_via_pointer(tmp_path: Path) -> None:
    _write_export("brb1-xgb-v6", "full", "a" * 64, write_pointer=True)
    assert fam.full_manifest_path(tmp_path, "brb1-xgb-v6") == (
        paths.export_json_path("brb1-xgb-v6", "full", "a" * 64)
    )


@pytest.mark.parametrize("bad", ["../evil", "a/b", "a b", "a:b", "", "ModelA", "A"])
def test_full_manifest_path_rejects_invalid_family(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        fam.full_manifest_path(tmp_path, bad)


@pytest.mark.parametrize("bad", ["../evil", "a/b", "a b", "a:b", "", "ModelA", "A"])
def test_load_full_version_rejects_invalid_family(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        fam.load_full_version(tmp_path, bad)


def test_full_manifest_path_missing_pointer_fails_loud(tmp_path: Path) -> None:
    """No mtime-based slot guessing: an absent pointer is an explicit absence."""
    _write_export("brb1-xgb-v6", "full", "a" * 64)  # no current.json -> degraded
    with pytest.raises(FileNotFoundError, match="available slots"):
        fam.full_manifest_path(tmp_path, "brb1-xgb-v6")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_full_manifest_path_dangling_pointer_fails_loud(tmp_path: Path) -> None:
    _write_export("brb1-xgb-v6", "full", "b" * 64, write_pointer=True)
    paths.current_pointer_path("brb1-xgb-v6").write_text(
        json.dumps({"run_id": "c" * 64}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        fam.full_manifest_path(tmp_path, "brb1-xgb-v6")


def test_full_manifest_path_corrupt_pointer_fails_loud(tmp_path: Path) -> None:
    _write_export("brb1-xgb-v6", "full", "b" * 64, write_pointer=True)
    paths.current_pointer_path("brb1-xgb-v6").write_text("{not json", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        fam.full_manifest_path(tmp_path, "brb1-xgb-v6")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_available_slots_lists_all_slot_dirs(tmp_path: Path) -> None:
    _write_export("brb1-xgb-v6", "full", "b" * 64)
    _write_export("brb1-xgb-v6", "full", "a" * 64)
    _write_export("brb1-xgb-v6", "full", "c" * 64)
    # A bare slot dir (no record) is still listed — it is diagnostics, not
    # a validity claim.
    paths.export_dir("brb1-xgb-v6", "full", "d" * 64).mkdir(parents=True)
    # Staging dirs are never available slots.
    (paths.export_dir("brb1-xgb-v6", "full", "x").parent / ".tmp-zzz").mkdir()
    assert fam.available_slots(tmp_path, "brb1-xgb-v6") == [
        "a" * 64, "b" * 64, "c" * 64, "d" * 64,
    ]
    assert fam.available_slots(tmp_path, "nope") == []


def test_load_full_version_happy_path(tmp_path: Path) -> None:
    _write_export("brb1-xgb-v6", "full", "a" * 64, write_pointer=True)
    v = fam.load_full_version(tmp_path, "brb1-xgb-v6")
    assert v is not None
    assert v.family == "brb1-xgb-v6"
    assert v.scope == "full"
    assert v.training_scope == "full"
    assert v.run_id == "a" * 64
    assert v.slot_dir == paths.export_dir("brb1-xgb-v6", "full", "a" * 64)
    assert v.config == {"run": {"name": "brb1-xgb-v6"}}
    assert v.rehearsal is False
    assert v.training_rows == 6_853_308
    assert v.training_era_range == (1, 1231)


def test_load_full_version_missing_pointer_returns_none(tmp_path: Path) -> None:
    _write_export("brb1-xgb-v6", "full", "a" * 64)  # valid export, no pointer
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_missing_family_returns_none(tmp_path: Path) -> None:
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_corrupt_json_returns_none(tmp_path: Path) -> None:
    _write_export("brb1-xgb-v6", "full", "a" * 64, body="{not json")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_family_mismatch_returns_none(tmp_path: Path) -> None:
    _write_export("brb1-xgb-v6", "full", "a" * 64, family="other-family")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_wrong_scope_returns_none(tmp_path: Path) -> None:
    _write_export("brb1-xgb-v6", "full", "a" * 64, training_scope="research")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_missing_run_id_returns_none(tmp_path: Path) -> None:
    _write_export("brb1-xgb-v6", "full", "a" * 64, promoted_from_run_id="")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_rejects_hollow_slot(tmp_path: Path) -> None:
    slot = _write_export("brb1-xgb-v6", "full", "a" * 64)
    (slot / "predict.pkl").unlink()
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_rejects_tampered_artifact(tmp_path: Path) -> None:
    slot = _write_export("brb1-xgb-v6", "full", "a" * 64)
    (slot / "predict.pkl").write_bytes(b"tampered")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_scan_full_versions_over_experiments(tmp_path: Path) -> None:
    _write_export("fam1", "full", "a" * 64)  # no pointer — degraded, still scanned
    versions = fam.scan_full_versions(tmp_path / "experiments")
    assert set(versions) == {"fam1"}
    assert versions["fam1"].run_id == "a" * 64


def test_full_version_requires_pointer(tmp_path: Path) -> None:
    _write_export("fam2", "full", "b" * 64)  # no current.json
    assert fam.load_full_version(tmp_path / "experiments", "fam2") is None
    paths.current_pointer_path("fam2").write_text(json.dumps({"run_id": "b" * 64}))
    assert fam.load_full_version(tmp_path / "experiments", "fam2") is not None


def test_scan_full_versions_only_valid(tmp_path: Path) -> None:
    _write_export("brb1-xgb-v6", "full", "a" * 64, write_pointer=True)
    _write_export("brb1-lgbm-v6", "full", "b" * 64, write_pointer=True)
    # invalid: corrupt record; mixed-case dir; a bare dir with no export
    _write_export("brb1-xgb-v5", "full", "c" * 64, body="{not json")
    (paths.EXPERIMENTS_ROOT / "Brb1-Xgb-V4" / "exports" / "full" / ("d" * 64)).mkdir(
        parents=True
    )
    (paths.EXPERIMENTS_ROOT / "lonely").mkdir()
    found = fam.scan_full_versions(tmp_path)
    assert set(found) == {"brb1-xgb-v6", "brb1-lgbm-v6"}


def test_scan_full_versions_picks_newest_valid_when_pointer_absent(tmp_path: Path) -> None:
    _write_export("fam3", "full", "a" * 64, promoted_at="2026-08-25T10:00:00+00:00")
    _write_export("fam3", "full", "b" * 64, promoted_at="2026-08-26T10:00:00+00:00")
    found = fam.scan_full_versions(tmp_path)
    assert set(found) == {"fam3"}
    assert found["fam3"].run_id == "b" * 64  # newest by promoted_at


def test_scan_full_versions_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert fam.scan_full_versions(tmp_path / "nope") == {}


def test_family_has_full_version(tmp_path: Path) -> None:
    _write_export("brb1-xgb-v6", "full", "a" * 64, write_pointer=True)
    assert fam.family_has_full_version(tmp_path, "brb1-xgb-v6") is True
    assert fam.family_has_full_version(tmp_path, "brb1-lgbm-v6") is False


def test_validate_family_name_delegates_to_paths() -> None:
    assert fam.validate_family_name("brb1-xgb-v6") == "brb1-xgb-v6"
    with pytest.raises(ValueError):
        fam.validate_family_name("ModelA")


def test_rehearsal_never_reads_as_full_version(tmp_path: Path) -> None:
    """Review directive 2026-08-18: a rehearsal artifact (rehearsal: true) must
    never be read as a genuine full version — excluded from
    scans/has_full_version even when a pointer exists; the flag + training
    provenance are surfaced on direct load."""
    _write_export(
        "brb1-lgbm-v6",
        "full",
        "a" * 64,
        rehearsal=True,
        training_rows=68_096,
        training_era_range=(3, 14),
        write_pointer=True,
    )
    version = fam.load_full_version(tmp_path, "brb1-lgbm-v6")
    assert version is not None
    assert version.rehearsal is True
    assert version.training_rows == 68_096
    assert version.training_era_range == (3, 14)

    # Scans and has_full_version exclude rehearsals entirely.
    assert fam.scan_full_versions(tmp_path) == {}
    assert fam.family_has_full_version(tmp_path, "brb1-lgbm-v6") is False

    # A genuine full version alongside is still discovered.
    _write_export("brb1-xgb-v6", "full", "b" * 64, write_pointer=True)
    found = fam.scan_full_versions(tmp_path)
    assert set(found) == {"brb1-xgb-v6"}
    assert found["brb1-xgb-v6"].rehearsal is False
