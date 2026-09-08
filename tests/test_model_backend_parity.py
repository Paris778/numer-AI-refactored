from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nmr._oof import train_multi_target_oof
from nmr.config import DataConfig, ModelConfig, SplitConfig
from nmr.deployment import load_predict
from nmr.ensemble import Ensembler
from nmr.model_backend_protocol import canonical_json_bytes
from nmr.models import ModelOrchestrator
from nmr.runner import _build_deploy_pipeline, _serialize_predict_artifact
from nmr.splitter import PurgedEraSplitter

_EXPECTED_BASELINES: dict[str, dict[str, str]] = {
    "lightgbm": {
        "resolved_params_hash": "9d0e07a343322ffb09c6d3ffb85939d8db428a1da0649ca3ae77c79053c9c7e6",
        "oof_key_hash": "8f22ff05387ff6eedecdb9e67ebfdf03fb36c078801160642964a559e9fb2a13",
        "oof_prediction_hash": "7f7d8bcc1e7d13b7adacc5d4a2013dac19eefdc764eefe2916273177fe61ec75",
        "weights_hash": "df93da06592c5e581d527f27ee99816e1f0d01677d6693d386cf06cdf1b2448d",
        "checkpoint_manifest_hash": "ff767be2e067f1e328d93b34e0029e8de6799ab0a11c6f4fa4cd8c224a8edfae",
        "deploy_prediction_hash": "f1454b1f97d2f3457f82978b3c93db2ec42e9c8aef40323e4f1d83e8ffd53d58",
    },
    "xgboost": {
        "resolved_params_hash": "dab866c939037f6fe7a8e8793916a3f3cd8ad6bec184d4b0f7cc473b792c2a9a",
        "oof_key_hash": "8f22ff05387ff6eedecdb9e67ebfdf03fb36c078801160642964a559e9fb2a13",
        "oof_prediction_hash": "a5ea308805beeb57175cba350673b3b9b957e74a458e00a886d923d8464bbe16",
        "weights_hash": "9b9f14817a52831b6c325ab4a88bae9ea06233f4dd71d7cec3c4a8654abb8715",
        "checkpoint_manifest_hash": "ff767be2e067f1e328d93b34e0029e8de6799ab0a11c6f4fa4cd8c224a8edfae",
        "deploy_prediction_hash": "b44d924c0e32e19cc9c8fe5b44c4040dcd6b79ba12860f51279e131b9033c6f1",
    },
    "catboost": {
        "resolved_params_hash": "ada4649bd39e89981dd47c0ea925229460837231a35fa8fc3db441720c5574f0",
        "oof_key_hash": "8f22ff05387ff6eedecdb9e67ebfdf03fb36c078801160642964a559e9fb2a13",
        "oof_prediction_hash": "46f279d206ad149dacfd7d286ec13595e58be292e7ea1830fcb7a63866407008",
        "weights_hash": "f34ead4a43491a55b75823ddd90258a280de9d5ee96b291e12feabe09e0bdb4e",
        "checkpoint_manifest_hash": "ff767be2e067f1e328d93b34e0029e8de6799ab0a11c6f4fa4cd8c224a8edfae",
        "deploy_prediction_hash": "7c57f1ce1348fb0485a62a7e6a023676f16330aaf68d5d9993139ea1e8aca480",
    },
}

_MANIFEST_STABLE_KEYS = (
    "device",
    "data_fingerprint",
    "environment",
    "target_col",
    "feature_fingerprint",
    "splitter_fingerprint",
)


def _training_frame(*, n_eras: int = 12, rows_per_era: int = 5) -> pl.DataFrame:
    rows: list[dict[str, float | str]] = []
    for era_num in range(1, n_eras + 1):
        for row_num in range(rows_per_era):
            f1 = float((era_num * 3 + row_num) % 11) / 10.0
            f2 = float((era_num * 5 - row_num * 2) % 13) / 10.0
            f3 = float((era_num + row_num * 7) % 17) / 10.0
            rows.append(
                {
                    "id": f"{era_num}_{row_num}",
                    "era": str(era_num),
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "target": 0.45 * f1 - 0.25 * f2 + 0.15 * f3 + (era_num / 100.0),
                    "target_alt": -0.20 * f1
                    + 0.35 * f2
                    - 0.10 * f3
                    + (era_num / 200.0),
                }
            )
    return pl.DataFrame(rows)


def _live_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "era": ["21", "21", "21", "22", "22", "22"],
            "f1": [0.0, 0.2, 0.4, 0.3, 0.5, 0.7],
            "f2": [0.1, 0.0, 0.4, 0.2, 0.1, 0.3],
            "f3": [0.2, 0.6, 0.1, 0.8, 0.4, 0.0],
        },
        index=[f"live_{idx}" for idx in range(6)],
    )


def _splitter() -> PurgedEraSplitter:
    return PurgedEraSplitter(
        SplitConfig(scheme="walk_forward", n_folds=3, purge_eras=1)
    )


def _model_config(backend: str) -> ModelConfig:
    return ModelConfig(
        backend=backend,
        preset="fast",
        device="cpu",
        params={
            "n_estimators": 8,
            "learning_rate": 0.05,
            "max_depth": 3,
            "num_leaves": 7,
            "colsample_bytree": 0.4,
            "min_data_in_leaf": 2,
        },
    )


def _hash_array(values: np.ndarray | tuple[float, ...]) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _hash_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _checkpoint_stable_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: manifest[key] for key in _MANIFEST_STABLE_KEYS}


def _compute_backend_baseline(backend: str, tmp_path: Path) -> dict[str, str]:
    train_df = _training_frame()
    feature_cols = ["f1", "f2", "f3"]
    splitter = _splitter()
    seed = 17
    config = _model_config(backend)

    resolved = ModelOrchestrator(config, seed=seed)._resolved_params(
        use_gpu=False,
        n_features=len(feature_cols),
    )

    target_oof = (
        ModelOrchestrator(config, seed=seed)
        .train_cross_validation(
            train_df,
            feature_cols=feature_cols,
            target_col="target",
            splitter=splitter,
            era_col="era",
        )
        .oof.sort(["era", "id"])
    )
    target_alt_oof = (
        ModelOrchestrator(config, seed=seed)
        .train_cross_validation(
            train_df,
            feature_cols=feature_cols,
            target_col="target_alt",
            splitter=splitter,
            era_col="era",
        )
        .oof.sort(["era", "id"])
    )

    weights_frame = (
        target_oof.rename({"prediction": "pred_target"})
        .join(
            target_alt_oof.rename({"prediction": "pred_target_alt"}),
            on=["id", "era"],
            how="inner",
        )
        .join(train_df.select(["id", "era", "target"]), on=["id", "era"], how="inner")
        .sort(["era", "id"])
    )
    weights = Ensembler().learn_weights(
        weights_frame,
        pred_cols=["pred_target", "pred_target_alt"],
        target_col="target",
    )

    checkpoint_dir = tmp_path / backend / "oof_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_multi_target_oof(
        ModelOrchestrator(config, seed=seed),
        train_df,
        feature_cols=feature_cols,
        splitter=splitter,
        targets=["target"],
        checkpoint_dir=checkpoint_dir,
        data_fingerprint="f" * 64,
        environment="deps==1",
    )
    manifest = json.loads(
        (checkpoint_dir / "target" / "manifest.json").read_text(encoding="utf-8")
    )

    deploy_dir = tmp_path / backend / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    predict_fn, model_meta = _build_deploy_pipeline(
        orchestrator=ModelOrchestrator(config, seed=seed),
        train_df=train_df,
        feature_cols=feature_cols,
        target_cols=["target", "target_alt"],
        weights=weights,
        proportion=0.25,
        data=DataConfig(version="vtest", data_dir=tmp_path / "data"),
        fit_device="cpu",
    )
    artifact = _serialize_predict_artifact(
        predict_fn=predict_fn,
        model_meta=model_meta,
        artifact_path=deploy_dir / "predict.pkl",
    )
    deployed = load_predict(artifact.path)(_live_features())

    return {
        "resolved_params_hash": _hash_json(resolved),
        "oof_key_hash": _hash_json(target_oof.select(["id", "era"]).rows(named=True)),
        "oof_prediction_hash": _hash_array(
            target_oof.get_column("prediction").to_numpy()
        ),
        "weights_hash": _hash_array(weights),
        "checkpoint_manifest_hash": _hash_json(_checkpoint_stable_manifest(manifest)),
        "deploy_prediction_hash": _hash_array(deployed["prediction"].to_numpy()),
    }


@pytest.mark.parametrize("backend", ["lightgbm", "xgboost", "catboost"])
def test_tree_backend_baseline_parity_hashes(backend: str, tmp_path: Path) -> None:
    actual = _compute_backend_baseline(backend, tmp_path)
    assert actual == _EXPECTED_BASELINES[backend], (
        f"{backend} baseline changed:\nexpected={_EXPECTED_BASELINES[backend]}\n"
        f"actual={actual}"
    )
