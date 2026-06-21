"""Tests for deployment artifact serialization and integrity checks."""

from __future__ import annotations

import json
import pickle

import pandas as pd
import pandas.testing as pdt
import pytest

from nmr.deployment import load_predict, serialize_predict


def test_serialize_load_roundtrip_is_exact_and_manifest_has_provenance(
    tmp_path,
) -> None:
    bias = 0.125

    def make_predict(scale: float):
        def predict(
            live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame = None
        ) -> pd.DataFrame:
            values = (live_features["f1"] * scale) + bias
            return pd.DataFrame({"prediction": values}, index=live_features.index)

        return predict

    predict_fn = make_predict(0.5)
    artifact = serialize_predict(
        predict_fn,
        path=tmp_path / "predict.pkl",
        feature_names=["f1", "f2"],
    )
    loaded_predict = load_predict(artifact.path)
    live_features = pd.DataFrame(
        {"f1": [0.0, 1.0, 2.0], "f2": [5.0, 6.0, 7.0]},
        index=["id1", "id2", "id3"],
    )

    pdt.assert_frame_equal(predict_fn(live_features), loaded_predict(live_features))
    assert artifact.manifest["feature_names"] == ["f1", "f2"]
    assert artifact.manifest["sha256"]
    assert artifact.manifest["created_at"]
    assert artifact.manifest["environment"]["python_version"]
    assert artifact.manifest["environment"]["packages"]["cloudpickle"]


def test_load_predict_raises_on_hash_mismatch(tmp_path) -> None:
    def predict(
        live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame = None
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {"prediction": live_features["f1"]}, index=live_features.index
        )

    artifact = serialize_predict(
        predict,
        path=tmp_path / "predict.pkl",
        feature_names=["f1"],
    )
    tampered = bytearray(artifact.path.read_bytes())
    tampered[-1] = (tampered[-1] + 1) % 256
    artifact.path.write_bytes(bytes(tampered))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_predict(artifact.path)


def test_cloudpickle_captures_closure_that_plain_pickle_cannot(tmp_path) -> None:
    offset = 0.75

    def make_predict():
        def predict(
            live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame = None
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {"prediction": live_features["f1"] + offset},
                index=live_features.index,
            )

        return predict

    predict_fn = make_predict()
    with pytest.raises(Exception):
        pickle.dumps(predict_fn)

    artifact = serialize_predict(
        predict_fn,
        path=tmp_path / "closure.pkl",
        feature_names=["f1"],
    )
    loaded_predict = load_predict(artifact.path)
    live_features = pd.DataFrame({"f1": [1.0, 2.0]}, index=["a", "b"])
    pdt.assert_frame_equal(predict_fn(live_features), loaded_predict(live_features))


def test_manifest_written_next_to_artifact(tmp_path) -> None:
    def predict(
        live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame = None
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {"prediction": live_features["f1"]}, index=live_features.index
        )

    artifact = serialize_predict(
        predict,
        path=tmp_path / "artifact.pkl",
        feature_names=["f1"],
    )
    manifest_path = tmp_path / "artifact.pkl.manifest.json"

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sha256"] == artifact.manifest["sha256"]
    assert manifest["feature_names"] == ["f1"]
