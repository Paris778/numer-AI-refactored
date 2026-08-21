"""OOF fold-checkpointing & resume contracts (spec 2026-08-20-oof-checkpoint-resume v2)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from nmr._oof import train_multi_target_oof
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
        _modeler(), train, feature_cols=["f1", "f2"],
        splitter=_splitter(), targets=["target", "target_ender_20"],
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
    manifest_path = ckpt / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["code_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="code_sha256"):
        _run(ckpt, train)


def test_device_mismatch_raises(tmp_path):
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    _run(ckpt, train)
    manifest_path = ckpt / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["device"] = "totally_different_device"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="device"):
        _run(ckpt, train)


def test_parts_without_manifest_raise(tmp_path):
    train = _synthetic_train()
    ckpt = tmp_path / "ckpt"
    _run(ckpt, train)
    (ckpt / "manifest.json").unlink()  # simulate a torn tree
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
