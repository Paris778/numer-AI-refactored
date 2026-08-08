"""Deterministic experiment orchestration over existing slice components."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd
import polars as pl

from nmr import _transforms
from nmr._transforms import (
    neutralize_array,
    rank_gaussianize,
    rank_gaussianize_unit_variance,
)
from nmr.config import ExperimentConfig, set_global_seeds
from nmr.data import IngestionAgent
from nmr.deployment import DeploymentArtifact, serialize_predict
from nmr.ensemble import Ensembler
from nmr.evaluation import EvaluationEngine, MetricSummary
from nmr.models import ModelOrchestrator
from nmr.risk import NeutralizationEngine
from nmr.scorecard import MetricScorecard, evaluate_model
from nmr.splitter import PurgedEraSplitter

logger = logging.getLogger("nmr.runner")

__all__ = ["RunResult", "ExperimentRunner"]


@dataclass(frozen=True)
class RunResult:
    run_id: str
    oof: pl.DataFrame
    metrics: MetricSummary
    artifact: DeploymentArtifact | None
    manifest: dict[str, Any]
    scorecard: MetricScorecard | None = None
    validation_predictions: pl.DataFrame | None = None


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self._config = config
        self._run_id = self._compute_run_id(config)

    def run(self, *, deploy: bool = False) -> RunResult:
        requested_metrics = set(self._config.evaluation.metrics)
        if "mmc" in requested_metrics and not self._config.evaluation.validation_scorecard:
            raise ValueError(
                "evaluation.metrics includes 'mmc' but the validation scorecard stage "
                "is disabled (evaluation.validation_scorecard=false). MMC requires the "
                "meta model, which covers validation eras only."
            )
        logger.info("[run] starting experiment run_id=%s", self._run_id)
        set_global_seeds(self._config.run.seed)

        agent = IngestionAgent(self._config.data)
        feature_cols = agent.features(self._config.data.resolved_feature_set)
        main_target = self._config.evaluation.main_target
        target_cols = list(dict.fromkeys([*self._config.data.targets, main_target]))
        logger.info(
            "[run] loading train data: features=%d targets=%s",
            len(feature_cols),
            target_cols,
        )
        train_df = agent.load(
            "train",
            columns=["era", "id", *feature_cols, *target_cols],
        )
        logger.info(
            "[run] train data loaded: rows=%d cols=%d",
            train_df.height,
            len(train_df.columns),
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
        logger.info("[run] blending %d target predictions", len(pred_cols))

        ensembler = Ensembler()
        folds = splitter.split(train_df.get_column("era").to_list())
        if len(folds) < 2:
            logger.warning(
                "[run] n_folds < 2; falling back to uniform ensemble weights"
            )
            weights = tuple(1.0 / len(pred_cols) for _ in pred_cols)
            weight_learning_eras: list[str] = []
        else:
            weight_learning_eras = [
                era for fold in folds[:-1] for era in fold.val_eras
            ]
            weight_df = joined.filter(pl.col("era").is_in(weight_learning_eras))
            weights = ensembler.learn_weights(
                weight_df.select(["era", *pred_cols, main_target]),
                pred_cols=pred_cols,
                target_col=main_target,
                era_col="era",
                method=self._config.ensemble.method,
            )
        scoring_eras = (
            [era for fold in folds for era in fold.val_eras]
            if len(folds) < 2
            else list(folds[-1].val_eras)
        )
        logger.info("[run] ensemble weights: %s (learned on %d eras, scored on %d)",
                    dict(zip(pred_cols, weights)), len(weight_learning_eras), len(scoring_eras))
        blended = ensembler.blend(
            joined,
            pred_cols=pred_cols,
            weights=weights,
            era_col="era",
            out_col="prediction",
        )

        logger.info(
            "[run] neutralizing predictions with %d features", len(feature_cols)
        )
        neutralization_proportion = self._config.risk.neutralization_proportion
        neutralized = NeutralizationEngine(
            max_cache_bytes=self._config.risk.cache_max_bytes
        ).neutralize(
            blended,
            pred_col="prediction",
            feature_cols=feature_cols,
            era_col="era",
            proportion=neutralization_proportion,
        )

        logger.info("[run] evaluating OOF predictions")
        evaluator = EvaluationEngine(self._config.evaluation.backend)
        per_era_all = evaluator.per_era_corr(
            neutralized,
            pred_col="prediction",
            target_col=main_target,
            era_col="era",
        )
        per_era_corr = {
            era: value for era, value in per_era_all.items() if era in set(scoring_eras)
        }
        metrics = evaluator.summarize(per_era_corr)
        logger.info(
            "[run] metrics: mean=%.5f std=%.5f sharpe=%.5f max_drawdown=%.5f",
            metrics.mean,
            metrics.std,
            metrics.sharpe,
            metrics.max_drawdown,
        )

        summary_metrics: dict[str, float] = {
            "corr": metrics.mean,
            "sharpe": metrics.sharpe,
        }
        if "fnc" in set(self._config.evaluation.metrics):
            fnc_by_era = evaluator.per_era_fnc(
                neutralized.filter(pl.col("era").is_in(set(scoring_eras))),
                pred_col="prediction",
                feature_cols=feature_cols,
                target_col=main_target,
                era_col="era",
            )
            summary_metrics["fnc"] = evaluator.summarize(fnc_by_era).mean
        oof = neutralized.select(["id", "era", "prediction"]).sort(["era", "id"])

        pipeline = None
        if deploy or self._config.evaluation.validation_scorecard:
            pipeline = self._build_deploy_pipeline(
                orchestrator=model_orchestrator,
                train_df=train_df,
                feature_cols=feature_cols,
                target_cols=target_cols,
                weights=weights,
                proportion=neutralization_proportion,
            )

        scorecard = None
        validation_predictions = None
        validation_purge = None
        if self._config.evaluation.validation_scorecard:
            scorecard, validation_predictions, validation_purge = (
                self._run_validation_stage(
                    predict_fn=pipeline[0], feature_cols=feature_cols
                )
            )
            logger.info("[run] validation scorecard ready: corr_sharpe_ac=%.5f",
                        scorecard.corr_sharpe_ac.value)

        artifact = None
        if deploy:
            logger.info("[run] serializing deploy artifact")
            assert pipeline is not None  # built once above when deploy=True
            artifact = self._serialize_predict_artifact(
                predict_fn=pipeline[0],
                model_meta=pipeline[1],
                artifact_path=(
                    self._config.run.artifacts_dir / "runs" / self._run_id / "predict.pkl"
                ),
            )
            logger.info("[run] artifact written to %s", artifact.path)

        manifest = {
            "run_id": self._run_id,
            "config": _to_jsonable(dataclasses.asdict(self._config)),
            "data_version": self._config.data.version,
            "seed": self._config.run.seed,
            "feature_cols": list(feature_cols),
            "pred_cols": pred_cols,
            "weights": list(weights),
            "weight_learning_eras": weight_learning_eras,
            "scoring_eras": scoring_eras,
            "summary_metrics": summary_metrics,
            "pipeline_device": "cpu",
            "oof_device": model_orchestrator.resolved_device,
            "metrics": dataclasses.asdict(metrics),
            "code_fingerprint": self._code_fingerprint(),
            "environment": self._environment_fingerprint(self._config.model.backend),
            "validation_purge_dropped_first_eras": validation_purge,
        }

        return RunResult(
            run_id=self._run_id,
            oof=oof,
            metrics=metrics,
            artifact=artifact,
            manifest=manifest,
            scorecard=scorecard,
            validation_predictions=validation_predictions,
        )

    def _train_multi_target_oof(
        self,
        train_df: pl.DataFrame,
        *,
        feature_cols: Sequence[str],
        splitter: PurgedEraSplitter,
        model_orchestrator: ModelOrchestrator,
    ) -> pl.DataFrame:
        targets = self._config.data.targets
        logger.info(
            "[train_multi_target_oof] training %d target(s): %s", len(targets), targets
        )
        stacked: pl.DataFrame | None = None
        for idx, target in enumerate(targets, start=1):
            logger.info(
                "[train_multi_target_oof] target %d/%d: %s", idx, len(targets), target
            )
            t0 = time.time()
            cv_result = model_orchestrator.train_cross_validation(
                train_df,
                feature_cols=feature_cols,
                target_col=target,
                splitter=splitter,
                era_col="era",
            )
            logger.info(
                "[train_multi_target_oof] target %d/%d complete in %.1fs (oof rows=%d)",
                idx,
                len(targets),
                time.time() - t0,
                cv_result.oof.height,
            )
            target_oof = cv_result.oof.rename({"prediction": f"pred_{target}"})
            if stacked is None:
                stacked = target_oof
            else:
                stacked = stacked.join(target_oof, on=["id", "era"], how="inner")

        assert stacked is not None
        logger.info("[train_multi_target_oof] stacked OOF shape: %s", stacked.shape)
        return stacked

    def _run_validation_stage(
        self, *, predict_fn, feature_cols: Sequence[str]
    ) -> tuple[MetricScorecard, pl.DataFrame, int]:
        data = self._config.data
        agent = IngestionAgent(data)
        target_cols = list(
            dict.fromkeys([*self._config.data.targets, self._config.evaluation.main_target])
        )
        val_df = agent.load(
            "validation", columns=["era", "id", *feature_cols, *target_cols]
        )
        meta_path = data.path("meta_model.parquet")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"validation_scorecard=true requires {meta_path}; disable the "
                "validation stage or provide the meta model"
            )
        meta_model = pl.read_parquet(meta_path).select(["era", "id", "numerai_meta_model"])

        bench_path = data.path("validation_benchmark_models.parquet")
        benchmarks = (
            pl.read_parquet(bench_path) if bench_path.exists() else None
        )
        if benchmarks is None:
            logger.warning("[validation] benchmark models missing; BMC/horizon disabled")

        purge = self._config.split.purge_eras
        all_eras = sorted({int(e) for e in val_df.get_column("era").unique().to_list()})
        if purge > 0:
            keep = {str(e) for e in all_eras[purge:]}
            val_df = val_df.filter(pl.col("era").is_in(keep))
        logger.info(
            "[validation] dropping first %d validation eras (20D-target overlap); "
            "%d eras scored", purge, val_df.select(pl.col("era").n_unique()).item()
        )

        features_pd = val_df.select(["id", "era", *feature_cols]).to_pandas().set_index("id")
        prediction_frame = predict_fn(features_pd)
        preds = val_df.select(["era", "id"]).with_columns(
            pl.Series("prediction", prediction_frame["prediction"].to_numpy())
        )
        scorecard = evaluate_model(
            preds,
            meta_model=meta_model,
            benchmarks=benchmarks,
            features=val_df.select(["era", "id", *feature_cols]),
            targets=val_df.select(["era", "id", *target_cols]),
            n_trials=1,
            seed=self._config.run.seed,
            horizon="20D",
            main_target=self._config.evaluation.main_target,
            benchmark_col=(
                # First non-join column (same convention as benchmark_runner) —
                # never positional index 2, which assumes column order.
                next(
                    (
                        col
                        for col in benchmarks.columns
                        if col not in {"era", "id"}
                    ),
                    None,
                )
                if benchmarks is not None
                else None
            ),
            backend=self._config.evaluation.backend,
            model_id=self._run_id,
        )
        return scorecard, preds, purge

    def _build_deploy_pipeline(
        self,
        *,
        orchestrator: ModelOrchestrator,
        train_df: pl.DataFrame,
        feature_cols: Sequence[str],
        target_cols: Sequence[str],
        weights: Sequence[float],
        proportion: float,
    ) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], dict[str, object]]:
        """Train per-target full-history models ONCE and return (predict, model_meta).

        The closure's code path references only numpy/pandas plus the shared
        transform helpers; cloudpickle.register_pickle_by_value(nmr._transforms)
        embeds those helpers by value so the artifact loads without `nmr`.
        """
        logger.info("[build_deploy_pipeline] training full-history models (CPU-only)")
        trained: dict[str, object] = {}
        for target in target_cols:
            trained[target] = orchestrator.train_full_history(
                train_df,
                feature_cols=feature_cols,
                target_col=target,
                era_col="era",
            )
        ordered_features = list(feature_cols)
        target_order = list(target_cols)
        weight_array = np.asarray(list(weights), dtype=float)

        def predict(
            live_features: pd.DataFrame,
            live_benchmark_models: pd.DataFrame = None,
        ) -> pd.DataFrame:
            del live_benchmark_models
            frame = live_features.loc[:, ordered_features]
            components = [
                np.asarray(trained[t].predict(frame), dtype=float)
                for t in target_order
            ]
            design = np.column_stack(components)
            if "era" in live_features.columns:
                era_values = live_features["era"].astype(str).to_numpy()
            else:
                era_values = np.full(len(live_features), "1")
            feature_matrix = frame.to_numpy(dtype=float)
            blended = np.empty(len(live_features), dtype=float)
            for era in np.unique(era_values):
                mask = era_values == era
                block = design[mask]
                normalized = np.column_stack(
                    [
                        rank_gaussianize_unit_variance(block[:, i])
                        for i in range(block.shape[1])
                    ]
                )
                combined = rank_gaussianize(normalized.dot(weight_array))
                blended[mask] = neutralize_array(
                    combined, feature_matrix[mask], proportion
                )
            return pd.DataFrame({"prediction": blended}, index=live_features.index)

        meta = {
            "targets": target_order,
            "weights": [float(w) for w in weights],
            "proportion": float(proportion),
            "geometry": "all_eras",
            "device": "cpu",
            "feature_names": ordered_features,
        }
        return predict, meta

    def _serialize_predict_artifact(
        self,
        *,
        predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
        model_meta: dict[str, object],
        artifact_path: Path,
    ) -> DeploymentArtifact:
        """Serialize the prebuilt pipeline closure. Does NOT retrain models."""
        cloudpickle.register_pickle_by_value(_transforms)
        return serialize_predict(
            predict_fn,
            path=artifact_path,
            feature_names=list(model_meta["feature_names"]),
            models=model_meta,
        )

    @staticmethod
    def _compute_run_id(config: ExperimentConfig) -> str:
        config_payload = _to_jsonable(dataclasses.asdict(config))
        _strip_path_dependent_fields(config_payload)
        payload = {
            "config": config_payload,
            "data_version": config.data.version,
            "code_fingerprint": ExperimentRunner._code_fingerprint(),
            "environment": ExperimentRunner._environment_fingerprint(config.model.backend),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def compute_run_id(config: ExperimentConfig) -> str:
        """Public accessor for the canonical run id (used by campaign tooling)."""
        return ExperimentRunner._compute_run_id(config)

    @staticmethod
    def _code_fingerprint() -> str:
        package_dir = Path(__file__).resolve().parent
        digest = hashlib.sha256()
        for path in sorted(package_dir.glob("*.py")):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _environment_fingerprint(backend: str | None = None) -> dict[str, Any]:
        """Package-version fingerprint for the canonical run id.

        Config-aware: the ``catboost`` package version is included only for
        ``backend == "catboost"`` configs, so catboost-backend runs flag
        catboost version drift while lightgbm/xgboost fingerprints stay
        byte-identical to the legacy shape (no existing run_id changes).
        """
        packages = {
            name: _package_version(name)
            for name in ["numpy", "polars", "pandas", "lightgbm", "xgboost"]
        }
        if backend == "catboost":
            packages["catboost"] = _package_version("catboost")
        return {
            "python_version": platform.python_version(),
            "packages": packages,
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
