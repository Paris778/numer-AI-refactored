"""Unit tests for nmr.families — the read-only model-family / full-version layer."""

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
    family_name: str | None = None,
    training_scope: str = "full",
    promoted_from_run_id: str = "a" * 64,
    artifact_path: str | None = "predict.pkl",
    body: str | None = None,
) -> Path:
    full_dir = models_dir / family / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    # Only write artifacts for safe relative paths — never write outside the
    # tmp models dir (absolute / ../ / empty artifact_paths are validation
    # fixtures and must not touch the real filesystem).
    if artifact_path is not None and artifact_path:
        candidate = Path(artifact_path)
        # Portable relative-path guard: is_absolute() alone is platform-
        # specific (C:\... is not absolute on POSIX; /abs/... is not absolute
        # on Windows). Reject root/drive + drive-letter forms on both
        # platforms so fixtures never write outside the tmp models dir.
        if (
            not candidate.is_absolute()
            and not candidate.root
            and not candidate.drive
            and not re.match(r"^[A-Za-z]:[\\/]", str(candidate))
            and ".." not in candidate.parts
        ):
            (full_dir / candidate).write_text("weights", encoding="utf-8")
    manifest = body
    if manifest is None:
        manifest = json.dumps(
            {
                "family": family_name if family_name is not None else family,
                "training_scope": training_scope,
                "promoted_from_run_id": promoted_from_run_id,
                "promoted_at": "2026-08-17T12:00:00Z",
                "artifact_path": artifact_path,
                "config": {"run": {"name": family}},
            }
        )
    path = full_dir / "manifest.json"
    path.write_text(manifest, encoding="utf-8")
    return path


def test_full_manifest_path_resolves(tmp_path: Path) -> None:
    assert fam.full_manifest_path(tmp_path, "brb1-xgb-v6") == (
        tmp_path / "brb1-xgb-v6" / "full" / "manifest.json"
    )


@pytest.mark.parametrize("bad", ["../evil", "a/b", "a b", "a:b", "", "ModelA", "A"])
def test_full_manifest_path_rejects_invalid_family(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        fam.full_manifest_path(tmp_path, bad)


@pytest.mark.parametrize("bad", ["../evil", "a/b", "a b", "a:b", "", "ModelA", "A"])
def test_load_full_version_rejects_invalid_family(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        fam.load_full_version(tmp_path, bad)


def test_load_full_version_happy_path(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6")
    v = fam.load_full_version(tmp_path, "brb1-xgb-v6")
    assert v is not None
    assert v.family == "brb1-xgb-v6"
    assert v.promoted_from_run_id == "a" * 64
    assert v.manifest_path == tmp_path / "brb1-xgb-v6" / "full" / "manifest.json"
    assert v.config == {"run": {"name": "brb1-xgb-v6"}}


def test_load_full_version_missing_manifest_returns_none(tmp_path: Path) -> None:
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
    (tmp_path / "brb1-xgb-v6" / "full" / "predict.pkl").unlink()
    assert fam.load_full_version(tmp_path, "brb1-xgb-v6") is None


def test_scan_full_versions_only_valid(tmp_path: Path) -> None:
    _write_full_manifest(tmp_path, "brb1-xgb-v6")
    _write_full_manifest(tmp_path, "brb1-lgbm-v6")
    # invalid: corrupt json, mixed-case dir, dir with no manifest
    _write_full_manifest(tmp_path, "brb1-xgb-v5", body="{not json")
    mixed = tmp_path / "Brb1-Xgb-V4" / "full"
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
