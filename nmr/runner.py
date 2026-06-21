"""Deterministic experiment orchestration over existing slice components."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from nmr.config import ExperimentConfig, SplitConfig, set_global_seeds
from nmr.data import IngestionAgent
from nmr.deployment import DeploymentArtifact, serialize_predict
from nmr.ensemble import Ensembler
from nmr.evaluation import EvaluationEngine, MetricSummary
from nmr.models import ModelOrchestrator
from nmr.risk import NeutralizationEngine
from nmr.splitter import PurgedEraSplitter

__all__ = ["RunResult", "ExperimentRunner"]


@dataclass(frozen=True)
class RunResult:
    run_id: str
    oof: pl.DataFrame
    metrics: MetricSummary
    artifact: DeploymentArtifact | None
    manifest: dict[str, Any]


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self._config = config
        self._run_id = self._compute_run_id(config)

    def run(self, *, deploy: bool = False) -> RunResult:
        set_global_seeds(self._config.run.seed)

        agent = IngestionAgent(self._config.data)
        feature_cols = agent.features(self._config.data.feature_set)
        main_target = self._config.evaluation.main_target
        target_cols = list(dict.fromkeys([*self._config.data.targets, main_target]))
        train_df = agent.load(
            "train",
            columns=["era", "id", *feature_cols, *target_cols],
        )

        splitter = PurgedEraSplitter(self._config.split)
        model_orchestrator = ModelOrchestrator(
            self._config.model, seed=self._config.run.seed
        )

        cv_oof = self._train_multi_target_oof(
            train_df,
            feature_cols=feature_cols,
            splitter=splitter,
            model_orchestrator=model_orchestrator,
        )

        joined = train_df.select(["id", "era", main_target, *feature_cols]).join(
            cv_oof,
            on=["id", "era"],
            how="inner",
        )
        pred_cols = [col for col in cv_oof.columns if col.startswith("pred_")]

        ensembler = Ensembler()
        weights = ensembler.learn_weights(
            joined.select(["era", *pred_cols, main_target]),
            pred_cols=pred_cols,
            target_col=main_target,
            era_col="era",
            method="ridge",
        )
        blended = ensembler.blend(
            joined,
            pred_cols=pred_cols,
            weights=weights,
            era_col="era",
            out_col="prediction",
        )

        neutralized = NeutralizationEngine().neutralize(
            blended,
            pred_col="prediction",
            feature_cols=feature_cols,
            era_col="era",
            proportion=1.0,
        )

        evaluator = EvaluationEngine(self._config.evaluation.backend)
        per_era_corr = evaluator.per_era_corr(
            neutralized,
            pred_col="prediction",
            target_col=main_target,
            era_col="era",
        )
        metrics = evaluator.summarize(per_era_corr)
        oof = neutralized.select(["id", "era", "prediction"]).sort(["era", "id"])

        artifact = None
        if deploy:
            artifact = self._serialize_predict_artifact(
                model=model_orchestrator,
                train_df=train_df,
                feature_cols=feature_cols,
                target_col=main_target,
                splitter=splitter,
            )

        manifest = {
            "run_id": self._run_id,
            "config": _to_jsonable(dataclasses.asdict(self._config)),
            "data_version": self._config.data.version,
            "seed": self._config.run.seed,
            "feature_cols": list(feature_cols),
            "pred_cols": pred_cols,
            "weights": list(weights),
            "metrics": dataclasses.asdict(metrics),
            "code_fingerprint": self._code_fingerprint(),
            "environment": self._environment_fingerprint(),
        }

        return RunResult(
            run_id=self._run_id,
            oof=oof,
            metrics=metrics,
            artifact=artifact,
            manifest=manifest,
        )

    def _train_multi_target_oof(
        self,
        train_df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        splitter: PurgedEraSplitter,
        model_orchestrator: ModelOrchestrator,
    ) -> pl.DataFrame:
        stacked: pl.DataFrame | None = None
        for target in self._config.data.targets:
            cv_result = model_orchestrator.train_cross_validation(
                train_df,
                feature_cols=feature_cols,
                target_col=target,
                splitter=splitter,
                era_col="era",
            )
            target_oof = cv_result.oof.rename({"prediction": f"pred_{target}"})
            if stacked is None:
                stacked = target_oof
            else:
                stacked = stacked.join(target_oof, on=["id", "era"], how="inner")

        assert stacked is not None
        return stacked

    def _serialize_predict_artifact(
        self,
        *,
        model: ModelOrchestrator,
        train_df: pl.DataFrame,
        feature_cols: Sequence[str],
        target_col: str,
        splitter: PurgedEraSplitter,
    ) -> DeploymentArtifact:
        del splitter
        anchor_splitter = PurgedEraSplitter(
            SplitConfig(
                scheme="anchor",
                purge_eras=self._config.split.purge_eras,
                embargo_eras=self._config.split.embargo_eras,
                n_folds=1,
            )
        )
        anchor_model, _ = model.train_anchor_fold(
            train_df,
            feature_cols=feature_cols,
            target_col=target_col,
            splitter=anchor_splitter,
            era_col="era",
        )
        ordered_features = list(feature_cols)

        def predict(
            live_features: pd.DataFrame,
            live_benchmark_models: pd.DataFrame = None,
        ) -> pd.DataFrame:
            del live_benchmark_models
            frame = live_features.loc[:, ordered_features]
            values = anchor_model.predict(frame)
            return pd.DataFrame({"prediction": values}, index=live_features.index)

        artifact_path = (
            self._config.run.artifacts_dir / "runs" / self._run_id / "predict.pkl"
        )
        return serialize_predict(
            predict,
            path=artifact_path,
            feature_names=ordered_features,
            models=[self._config.model.backend, self._config.model.preset],
        )

    @staticmethod
    def _compute_run_id(config: ExperimentConfig) -> str:
        config_payload = _to_jsonable(dataclasses.asdict(config))
        _strip_path_dependent_fields(config_payload)
        payload = {
            "config": config_payload,
            "data_version": config.data.version,
            "code_fingerprint": ExperimentRunner._code_fingerprint(),
            "environment": ExperimentRunner._environment_fingerprint(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _code_fingerprint() -> str:
        package_dir = Path(__file__).resolve().parent
        digest = hashlib.sha256()
        for path in sorted(package_dir.glob("*.py")):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _environment_fingerprint() -> dict[str, Any]:
        return {
            "python_version": platform.python_version(),
            "packages": {
                name: _package_version(name)
                for name in ["numpy", "polars", "pandas", "lightgbm", "xgboost"]
            },
        }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _strip_path_dependent_fields(config_payload: dict[str, Any]) -> None:
    data_section = config_payload.get("data")
    if isinstance(data_section, dict):
        data_section.pop("data_dir", None)

    run_section = config_payload.get("run")
    if isinstance(run_section, dict):
        run_section.pop("artifacts_dir", None)
