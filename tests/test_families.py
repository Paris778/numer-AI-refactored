"""Unit tests for nmr.families — read-only full-version discovery.

Layout under test: ``models_dir/<family>/full/<run_id>/manifest.json``
(one immutable slot per promoted run) + atomic ``current.json`` pointer.
Resolution is pointer-driven and fails loud on a missing/dangling pointer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import nmr.families as fam


def _write_full_manifest(
    models_dir: Path,
    family: str,
    *,
    run_id: str = "a" * 64,
    family_name: str | None = None,
    training_scope: str = "full",
    promoted_from_run_id: str | None = None,
    artifact_path: str | None = "predict.pkl",
    body: str | None = None,
    write_pointer: bool = True,
    rehearsal: bool = False,
) -> Path:
    slot_dir = models_dir / family / "full" / run_id
    slot_dir.mkdir(parents=True, exist_ok=True)
    # Only write artifacts for safe relative paths — never write outside the
    # tmp models dir (absolute / ../ / empty artifact_paths are validation
    # fixtures and must not touch the real filesystem).
    if artifact_path is not None and artifact_path:
        candidate = Path(artifact_path)
        if (
            not candidate.is_absolute()
            and not candidate.root
            and not candidate.drive
            and not re.match(r"^[A-Za-z]:[\\/]", str(candidate))
            and ".." not in candidate.parts
        ):
            (slot_dir / candidate).write_text("weights", encoding="utf-8")
    manifest = body
    if manifest is None:
        manifest = json.dumps(
            {
                "family": family_name if family_name is not None else family,
                "training_scope": training_scope,
                "promoted_from_run_id": (
                    promoted_from_run_id
                    if promoted_from_run_id is not None
                    else "a" * 64
                ),
                "promoted_at": "2026-08-17T12:00:00Z",
                "artifact_path": artifact_path,
                "config": {"run": {"name": family}},
                "rehearsal": rehearsal,
                "training_rows": 68_096 if rehearsal else 6_853_308,
                "training_era_range": [3, 14] if rehearsal else [1, 1231],
            }
        )
    path = slot_dir / "manifest.json"
    path.write_text(manifest, encoding="utf-8")
    if write_pointer:
        (models_dir / family / "full" / fam.CURRENT_POINTER_NAME).write_text(
            json.dumps({"run_id": run_id, "promoted_at": "2026-08-17T12:00:00Z"}),
            encoding="utf-8",
        )
    return path


def test_full_manifest_path_resolves_via_pointer(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6")
    assert fam.full_manifest_path(tmp_path, "brb1-xgb-v6") == (
        tmp_path / "brb1-xgb-v6" / "full" / ("a" * 64) / "manifest.json"
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
    _write_full_manifest(tmp_path, "brb1-xgb-v6", write_pointer=False)
    with pytest.raises(FileNotFoundError, match="available slots"):
        fam.full_manifest_path(tmp_path, "brb1-xgb-v6")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_full_manifest_path_dangling_pointer_fails_loud(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6", run_id="b" * 64)
    pointer = tmp_path / "brb1-xgb-v6" / "full" / fam.CURRENT_POINTER_NAME
    pointer.write_text(json.dumps({"run_id": "c" * 64}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing slot"):
        fam.full_manifest_path(tmp_path, "brb1-xgb-v6")


def test_full_manifest_path_corrupt_pointer_fails_loud(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6", run_id="b" * 64)
    pointer = tmp_path / "brb1-xgb-v6" / "full" / fam.CURRENT_POINTER_NAME
    pointer.write_text("{not json", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="corrupt"):
        fam.full_manifest_path(tmp_path, "brb1-xgb-v6")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_available_slots_sorted_and_manifest_only(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6", run_id="b" * 64)
    _write_full_manifest(tmp_path, "brb1-xgb-v6", run_id="a" * 64)
    _write_full_manifest(tmp_path, "brb1-xgb-v6", run_id="c" * 64)
    # A slot dir without a manifest is not an available slot.
    (tmp_path / "brb1-xgb-v6" / "full" / ("d" * 64)).mkdir()
    assert fam.available_slots(tmp_path, "brb1-xgb-v6") == ["a" * 64, "b" * 64, "c" * 64]
    assert fam.available_slots(tmp_path, "nope") == []


def test_load_full_version_happy_path(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6")
    v = fam.load_full_version(tmp_path, "brb1-xgb-v6")
    assert v is not None
    assert v.family == "brb1-xgb-v6"
    assert v.promoted_from_run_id == "a" * 64
    assert v.manifest_path == (
        tmp_path / "brb1-xgb-v6" / "full" / ("a" * 64) / "manifest.json"
    )
    assert v.config == {"run": {"name": "brb1-xgb-v6"}}


def test_load_full_version_missing_pointer_returns_none(tmp_path: Path) -> None:
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_corrupt_json_returns_none(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6", body="{not json")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_family_mismatch_returns_none(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6", family_name="other-family")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_wrong_scope_returns_none(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6", training_scope="research")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_missing_run_id_returns_none(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6", promoted_from_run_id="")
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


@pytest.mark.parametrize(
    "bad_artifact",
    ["", "../predict.pkl", "C:\\abs\\predict.pkl", "C:/abs/predict.pkl", "/abs/predict.pkl"],
)
def test_load_full_version_rejects_invalid_artifact_path(
    tmp_path: Path, bad_artifact: str
) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6", artifact_path=bad_artifact)
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_load_full_version_rejects_hollow_promotion(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6", artifact_path="predict.pkl")
    (tmp_path / "brb1-xgb-v6" / "full" / ("a" * 64) / "predict.pkl").unlink()
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_scan_full_versions_only_valid(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6")
    _write_full_manifest(tmp_path, "brb1-lgbm-v6")
    # invalid: corrupt json, mixed-case dir, slots without a pointer
    _write_full_manifest(tmp_path, "brb1-xgb-v5", body="{not json")
    _write_full_manifest(tmp_path, "brb1-xgb-v3", write_pointer=False)
    mixed = tmp_path / "Brb1-Xgb-V4" / "full" / ("b" * 64)
    mixed.mkdir(parents=True)
    (mixed / "manifest.json").write_text(
        json.dumps(
            {
                "family": "Brb1-Xgb-V4",
                "training_scope": "full",
                "promoted_from_run_id": "b" * 64,
                "artifact_path": "predict.pkl",
            }
        ),
        encoding="utf-8",
    )
    (mixed / "predict.pkl").write_text("x", encoding="utf-8")
    (tmp_path / "lonely" / "full").mkdir(parents=True)
    found = fam.scan_full_versions(tmp_path)
    assert set(found) == {"brb1-xgb-v6", "brb1-lgbm-v6"}


def test_scan_full_versions_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert fam.scan_full_versions(tmp_path / "nope") == {}


def test_family_has_full_version(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6")
    assert fam.family_has_full_version(tmp_path, "brb1-xgb-v6") is True
    assert fam.family_has_full_version(tmp_path, "brb1-lgbm-v6") is False


def test_rehearsal_never_reads_as_full_version(tmp_path: Path) -> None:
    """Review directive 2026-08-18: a rehearsal artifact (rehearsal: true in
    the manifest) must never be read as a genuine full version — excluded from
    scans/has_full_version even when a pointer exists; the flag + training
    provenance are surfaced on direct load."""
    _write_full_manifest(tmp_path, "brb1-lgbm-v6", rehearsal=True)
    version = fam.load_full_version(tmp_path, "brb1-lgbm-v6")
    assert version is not None
    assert version.rehearsal is True
    assert version.training_rows == 68_096
    assert version.training_era_range == (3, 14)

    # Scans and has_full_version exclude rehearsals entirely.
    assert fam.scan_full_versions(tmp_path) == {}
    assert fam.family_has_full_version(tmp_path, "brb1-lgbm-v6") is False

    # A genuine full version alongside is still discovered.
    _write_full_manifest(tmp_path, "brb1-xgb-v6")
    found = fam.scan_full_versions(tmp_path)
    assert set(found) == {"brb1-xgb-v6"}
    assert found["brb1-xgb-v6"].rehearsal is False
