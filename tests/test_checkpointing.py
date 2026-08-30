"""OOF, deploy, and validation checkpoint contracts from ARCHITECTURE.md section G."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from nmr._oof import (
    checkpoint_manifest,
    ensure_no_torn_tree,
    fitting_code_sha256,
    train_multi_target_oof,
    verify_checkpoint_manifest,
    write_bytes_atomic,
    write_frame_atomic,
)
from nmr.config import ModelConfig, SplitConfig
from nmr.models import ModelOrchestrator
from nmr.splitter import PurgedEraSplitter


def _synthetic_train(*, n_eras: int = 16, rows_per_era: int = 6) -> pl.DataFrame:
    rows: list[dict[str, float | str]] = []
    for era_num in range(1, n_eras + 1):
        for row_num in range(rows_per_era):
            f1 = float((era_num * 3 + row_num) % 11) / 10.0
            f2 = float((era_num * 5 - row_num * 2) % 13) / 10.0
            f3 = float((era_num + row_num * 7) % 17) / 10.0
            target = 0.45 * f1 - 0.25 * f2 + 0.15 * f3 + (era_num / 100.0)
            target_ender = -0.20 * f1 + 0.35 * f2 - 0.10 * f3 + (era_num / 200.0)
            rows.append(
                {
                    "id": f"{era_num}_{row_num}",
                    "era": str(era_num),
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "target": target,
                    "target_ender_20": target_ender,
                }
            )
    return pl.DataFrame(rows)


def _tiny_model_params(**extra: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": 1,
        "max_depth": 1,
        "min_child_weight": 1,
    }
    params.update(extra)
    return params


def _modeler() -> ModelOrchestrator:
    return ModelOrchestrator(
        ModelConfig(backend="lightgbm", preset="fast", params=_tiny_model_params()),
        seed=7,
    )


def _splitter() -> PurgedEraSplitter:
    return PurgedEraSplitter(
        SplitConfig(scheme="walk_forward", n_folds=3, purge_eras=1)
    )


def _run(ckpt: Path | None, train: pl.DataFrame) -> pl.DataFrame:
    return train_multi_target_oof(
        _modeler(),
        train,
        feature_cols=["f1", "f2"],
        splitter=_splitter(),
        targets=["target", "target_ender_20"],
        checkpoint_dir=ckpt,
    )


def test_all_loaded_resume_equals_fresh_and_fits_nothing(tmp_path, caplog):
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    fresh = _run(ckpt, train)
    caplog.clear()
    with caplog.at_level("INFO", logger="nmr.models"):
        resumed = _run(ckpt, train)
    assert fresh.equals(resumed)
    assert "loaded from checkpoint" in caplog.text
    assert "trained in" not in caplog.text


def test_mixed_resume_within_target_is_bit_for_bit(tmp_path, caplog):
    """The only case that can break determinism: fold loaded + fold refit."""
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    fresh = _run(ckpt, train)
    parts = sorted((ckpt / "target").glob("fold_*.parquet"))
    assert len(parts) >= 2
    parts[0].unlink()  # delete exactly ONE fold within one target
    caplog.clear()
    with caplog.at_level("INFO", logger="nmr.models"):
        resumed = _run(ckpt, train)
    assert fresh.equals(resumed)
    assert "loaded from checkpoint" in caplog.text
    assert "trained in" in caplog.text  # the refit actually happened


def test_partial_target_resume_refits_only_missing_target(tmp_path):
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    fresh = _run(ckpt, train)
    shutil.rmtree(ckpt / "target_ender_20")  # rmtree, NOT unlink (directory)
    resumed = _run(ckpt, train)
    assert fresh.equals(resumed)


def test_code_mismatch_raises(tmp_path):
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    _run(ckpt, train)
    # The manifest is PER-TARGET (2026-08-26 review SECONDARY 1): it sits
    # next to its own fold parquets under <ckpt>/<target>/.
    manifest_path = ckpt / "target" / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["code_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="code_sha256"):
        _run(ckpt, train)


def test_device_mismatch_raises(tmp_path):
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    _run(ckpt, train)
    manifest_path = ckpt / "target" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["device"] = "totally_different_device"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="device"):
        _run(ckpt, train)


def test_parts_without_manifest_raise(tmp_path):
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    _run(ckpt, train)
    (ckpt / "target" / "manifest.json").unlink()  # simulate a torn tree
    with pytest.raises(ValueError, match="no manifest.json"):
        _run(ckpt, train)


def test_corrupt_checkpoint_raises(tmp_path):
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    _run(ckpt, train)
    parts = sorted((ckpt / "target").glob("fold_*.parquet"))
    parts[0].write_bytes(b"garbage")
    with pytest.raises(ValueError, match="corrupt OOF checkpoint"):
        _run(ckpt, train)


def test_checkpoint_tree_contains_no_temp_files(tmp_path):
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    _run(ckpt, train)
    all_files = sorted(p.name for p in ckpt.rglob("*") if p.is_file())
    for name in all_files:
        assert name == "manifest.json" or (
            name.startswith("fold_") and name.endswith(".parquet")
        ), f"unexpected file in checkpoint tree: {name}"


# --- Direct unit tests for the extracted shared helpers (Task A) ---


def test_fitting_code_sha256_is_stable_and_feeds_manifest():
    digest = fitting_code_sha256()
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert digest == fitting_code_sha256()  # deterministic across calls
    assert digest == checkpoint_manifest("cpu")["code_sha256"]


def test_checkpoint_manifest_roundtrip():
    assert checkpoint_manifest("cpu") == {
        "code_sha256": fitting_code_sha256(),
        "device": "cpu",
    }


def test_checkpoint_manifest_carries_rebuild_identity_fields():
    """Spec §3.1: the checkpoint manifest gains data_fingerprint + environment
    when the caller (the runner) provides them — the same values run.json
    persists, so resume refuses on data-snapshot or environment drift."""
    manifest = checkpoint_manifest(
        "cpu", data_fingerprint="d" * 64, environment="numpy==1.0"
    )
    assert manifest == {
        "code_sha256": fitting_code_sha256(),
        "device": "cpu",
        "data_fingerprint": "d" * 64,
        "environment": "numpy==1.0",
    }


def test_checkpoint_manifest_carries_fit_identity_fields():
    """2026-08-26 review SECONDARY 1: the manifest records the fit identity —
    target_col, the canonical feature-list fingerprint, and the
    splitter-geometry fingerprint — when the caller provides them."""
    from nmr._oof import feature_list_fingerprint, splitter_geometry_fingerprint

    train = _synthetic_train()
    feature_fp = feature_list_fingerprint(["f2", "f1"])  # order-independent
    splitter_fp = splitter_geometry_fingerprint(_splitter(), train["era"].to_list())
    manifest = checkpoint_manifest(
        "cpu",
        target_col="target",
        feature_fingerprint=feature_fp,
        splitter_fingerprint=splitter_fp,
    )
    assert manifest["target_col"] == "target"
    assert manifest["feature_fingerprint"] == feature_fp
    assert manifest["splitter_fingerprint"] == splitter_fp
    assert len(manifest["feature_fingerprint"]) == 64
    assert len(manifest["splitter_fingerprint"]) == 64
    # Order independence: the same feature set in a different order hashes
    # identically.
    assert feature_list_fingerprint(["f1", "f2"]) == feature_fp


def test_verify_checkpoint_manifest_data_fingerprint_mismatch_raises(tmp_path):
    """Spec §3.1 rebuild-refusal: a checkpoint manifest recording a different
    data snapshot must refuse resume with delete-to-refit guidance."""
    manifest_path = tmp_path / "manifest.json"
    manifest = checkpoint_manifest(
        "cpu", data_fingerprint="d" * 64, environment="numpy==1.0"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="data_fingerprint"):
        verify_checkpoint_manifest(
            manifest_path,
            "cpu",
            data_fingerprint="e" * 64,
            environment="numpy==1.0",
        )


def test_verify_checkpoint_manifest_environment_mismatch_raises(tmp_path):
    """Spec §3.1 rebuild-refusal: a checkpoint manifest recording a different
    dependency environment must refuse resume."""
    manifest_path = tmp_path / "manifest.json"
    manifest = checkpoint_manifest(
        "cpu", data_fingerprint="d" * 64, environment="numpy==1.0"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="environment"):
        verify_checkpoint_manifest(
            manifest_path,
            "cpu",
            data_fingerprint="d" * 64,
            environment="numpy==2.0",
        )


def test_verify_checkpoint_manifest_missing_identity_fields_refuses(tmp_path):
    """A legacy code+device-only manifest (no data_fingerprint/environment)
    must refuse when the caller guards those terms — never resume vacuously."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(checkpoint_manifest("cpu")), encoding="utf-8")
    with pytest.raises(ValueError, match="data_fingerprint"):
        verify_checkpoint_manifest(
            manifest_path, "cpu", data_fingerprint="d" * 64, environment="env"
        )


def test_verify_checkpoint_manifest_accepts_matching_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(checkpoint_manifest("cpu")), encoding="utf-8")
    verify_checkpoint_manifest(manifest_path, "cpu")  # known device, exact match
    verify_checkpoint_manifest(manifest_path, None)  # unresolved device, valid schema


def test_verify_checkpoint_manifest_code_mismatch_raises(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = checkpoint_manifest("cpu")
    manifest["code_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="code_sha256"):
        verify_checkpoint_manifest(manifest_path, "cpu")


def test_verify_checkpoint_manifest_device_mismatch_raises(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = checkpoint_manifest("cpu")
    manifest["device"] = "gpu"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="device"):
        verify_checkpoint_manifest(manifest_path, "cpu")


def test_verify_checkpoint_manifest_rejects_unknown_device_when_unresolved(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = checkpoint_manifest("not_a_real_device")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="device"):
        verify_checkpoint_manifest(manifest_path, None)


def test_ensure_no_torn_tree_raises_with_parts(tmp_path):
    root = tmp_path / "ckpt"
    (root / "target").mkdir(parents=True)
    (root / "target" / "fold_01.parquet").write_bytes(b"not really parquet")
    with pytest.raises(ValueError, match="no manifest.json"):
        ensure_no_torn_tree(root / "manifest.json")
    ensure_no_torn_tree(tmp_path / "nonexistent" / "manifest.json")  # empty is fine


def test_write_frame_and_bytes_atomic_leave_no_temp_files(tmp_path):
    frame = pl.DataFrame({"x": [1, 2, 3]})
    frame_path = tmp_path / "frame.parquet"
    write_frame_atomic(frame, frame_path)
    assert pl.read_parquet(frame_path).equals(frame)

    payload = b"\x00\x01\x02checkpoint-payload"
    bytes_path = tmp_path / "payload.bin"
    write_bytes_atomic(payload, bytes_path)
    assert bytes_path.read_bytes() == payload

    leftovers = [
        p.name
        for p in tmp_path.iterdir()
        if ".tmp." in p.name or p.name.endswith(".part")
    ]
    assert not leftovers, f"atomic write left temp files behind: {leftovers}"


# --- Fit-identity guards (2026-08-26 review, SECONDARY 1) -----------------


def _write_fit_identity_manifest(tmp_path: Path) -> Path:
    from nmr._oof import feature_list_fingerprint, splitter_geometry_fingerprint

    train = _synthetic_train()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            checkpoint_manifest(
                "cpu",
                target_col="target",
                feature_fingerprint=feature_list_fingerprint(["f1", "f2"]),
                splitter_fingerprint=splitter_geometry_fingerprint(
                    _splitter(), train["era"].to_list()
                ),
            )
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_verify_checkpoint_manifest_target_mismatch_raises(tmp_path):
    manifest_path = _write_fit_identity_manifest(tmp_path)
    with pytest.raises(ValueError, match="target mismatch"):
        verify_checkpoint_manifest(manifest_path, "cpu", target_col="target_ender_20")
    verify_checkpoint_manifest(manifest_path, "cpu", target_col="target")  # agrees


def test_verify_checkpoint_manifest_feature_mismatch_raises(tmp_path):
    from nmr._oof import feature_list_fingerprint

    manifest_path = _write_fit_identity_manifest(tmp_path)
    with pytest.raises(ValueError, match="feature-list mismatch"):
        verify_checkpoint_manifest(
            manifest_path, "cpu", feature_fingerprint=feature_list_fingerprint(["f1"])
        )
    verify_checkpoint_manifest(
        manifest_path, "cpu", feature_fingerprint=feature_list_fingerprint(["f1", "f2"])
    )


def test_verify_checkpoint_manifest_splitter_mismatch_raises(tmp_path):
    from nmr._oof import splitter_geometry_fingerprint

    manifest_path = _write_fit_identity_manifest(tmp_path)
    other_splitter = PurgedEraSplitter(
        SplitConfig(scheme="walk_forward", n_folds=4, purge_eras=1)
    )
    train = _synthetic_train()
    with pytest.raises(ValueError, match="splitter-geometry mismatch"):
        verify_checkpoint_manifest(
            manifest_path,
            "cpu",
            splitter_fingerprint=splitter_geometry_fingerprint(
                other_splitter, train["era"].to_list()
            ),
        )
    verify_checkpoint_manifest(
        manifest_path,
        "cpu",
        splitter_fingerprint=splitter_geometry_fingerprint(
            _splitter(), train["era"].to_list()
        ),
    )


def test_verify_checkpoint_manifest_missing_fit_identity_fields_refuses(tmp_path):
    """A legacy manifest without the fit-identity fields must refuse when the
    caller guards them — never resume vacuously."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(checkpoint_manifest("cpu")), encoding="utf-8")
    with pytest.raises(ValueError, match="target mismatch"):
        verify_checkpoint_manifest(manifest_path, "cpu", target_col="target")


def test_copied_target_dir_refused_for_different_target(tmp_path):
    """2026-08-26 review SECONDARY 1: copying a target's checkpoint dir onto
    another target name and fitting a DIFFERENT target with the SAME
    code/data/device identity is REFUSED — the copied manifest records the
    source target, and the target-mismatch guard rejects resume instead of
    silently reusing another target's folds."""
    import shutil

    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    _run(ckpt, train)  # fits targets ["target", "target_ender_20"]

    # Copy target's fold dir (with its manifest) onto the OTHER target name.
    shutil.rmtree(ckpt / "target_ender_20")
    shutil.copytree(ckpt / "target", ckpt / "target_ender_20")
    with pytest.raises(ValueError, match="target mismatch"):
        _run(ckpt, train)

    # The un-tampered target still resumes cleanly (code/data/device identity
    # intact) — only the copied dir is refused.
    shutil.rmtree(ckpt / "target_ender_20")
    _run(ckpt, train)
