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

from nmr import _transforms, paths
from nmr._oof import (
    checkpoint_manifest,
    ensure_no_torn_tree,
    train_multi_target_oof,
    verify_checkpoint_manifest,
    write_bytes_atomic,
    write_frame_atomic,
)
from nmr._transforms import (
    neutralize_array,
    rank_gaussianize,
    rank_gaussianize_unit_variance,
    tie_kept_rank,
)
from nmr.config import (
    DataConfig,
    ExperimentConfig,
    enforce_purge_horizon_law,
    set_global_seeds,
)
from nmr.data import IngestionAgent
from nmr.deployment import DeploymentArtifact, serialize_predict
from nmr.ensemble import Ensembler
from nmr.evaluation import EvaluationEngine, MetricSummary
from nmr.models import ModelOrchestrator
from nmr.risk import NeutralizationEngine
from nmr.scorecard import MetricScorecard, evaluate_model
from nmr.splitter import PurgedEraSplitter

logger = logging.getLogger("nmr.runner")

_VAL_PREDICT_ERA_BATCH = 40  # eras per validation predict chunk (bounds peak RAM)


def _era_batch_frames(val_df: pl.DataFrame, batch_eras: int) -> list[pl.DataFrame]:
    """Split ``val_df`` into era-batches in the frame's era appearance order.

    The single boundary computation for the batched validation predict paths
    (spec 2026-08-23-checkpoint-coverage-extension): eras chunk in their
    appearance order and each batch is the filter of ``val_df`` to that era
    slice, so concatenating per-batch results in batch order reproduces
    ``val_df``'s row order. Both ``_predict_in_era_batches`` and the
    checkpoint-aware ``_predict_validation_era_batches`` derive their batches
    here — the two paths can never drift apart.
    """
    eras = list(dict.fromkeys(val_df.get_column("era").to_list()))
    batches: list[pl.DataFrame] = []
    for start in range(0, len(eras), batch_eras):
        batch_era_set = set(eras[start : start + batch_eras])
        batches.append(val_df.filter(pl.col("era").is_in(batch_era_set)))
    return batches


def _predict_era_batch(
    batch: pl.DataFrame,
    feature_cols: Sequence[str],
    predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> pl.DataFrame:
    """Predict one era-batch; return its ``(era, id, prediction)`` frame.

    Shared per-batch computation for the batched validation predict paths:
    feature columns are coerced to a single Float32 block first (``nmr.models.
    coerce_float32_features``) so the backend's float32 conversion is a
    zero-copy view; the deploy closure is stateless across calls and the
    era-partitioned neutralization inside it is order-independent (cache keyed
    per era), so batching is bit-identical to a single full-frame predict.
    """
    from nmr.models import coerce_float32_features

    feats = coerce_float32_features(batch, feature_cols)
    features_pd = (
        pl.concat([batch.select(["id", "era"]), feats], how="horizontal")
        .to_pandas()
        .set_index("id")
    )
    prediction_frame = predict_fn(features_pd)
    return batch.select(["era", "id"]).with_columns(
        pl.Series("prediction", prediction_frame["prediction"].to_numpy())
    )


def _predict_in_era_batches(
    val_df: pl.DataFrame,
    feature_cols: Sequence[str],
    predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
    batch_eras: int,
) -> pl.DataFrame:
    """Predict validation eras in era-batches to bound peak memory.

    Batches follow the frame's era appearance order, so the output row order
    equals ``val_df``'s (see ``_era_batch_frames``). Batching is bit-identical
    to a single full-frame predict (see ``_predict_era_batch``).
    """
    if val_df.is_empty():
        return val_df.select(["era", "id"]).with_columns(
            pl.Series("prediction", [], dtype=pl.Float64)
        )
    chunks: list[pl.DataFrame] = []
    for batch in _era_batch_frames(val_df, batch_eras):
        chunks.append(_predict_era_batch(batch, feature_cols, predict_fn))
    return pl.concat(chunks)


def _predict_validation_era_batches(
    val_df: pl.DataFrame,
    feature_cols: Sequence[str],
    predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
    batch_eras: int,
    *,
    validation_checkpoint_dir: Path | None = None,
    checkpoint_device: str | None = None,
    data_fingerprint: str | None = None,
    environment: str | None = None,
) -> pl.DataFrame:
    """Checkpoint-aware variant of ``_predict_in_era_batches``.

    Batch boundaries are identical (shared ``_era_batch_frames``), so with
    ``validation_checkpoint_dir=None`` the output is bit-identical to
    ``_predict_in_era_batches``. With a directory, each batch persists to
    ``validation_checkpoints/preds_batch_NN.parquet`` (atomic write) and a
    resume loads present batches instead of predicting (spec 2026-08-23-
    checkpoint-coverage-extension). The root ``manifest.json`` follows the
    OOF/deploy identity discipline: verified at entry (code exact-compare
    always; device exact-compare when ``checkpoint_device`` is known, schema
    check otherwise; the rebuild-identity terms — ``data_fingerprint`` and
    ``environment``, spec §3.1 — exact-compared when provided) and
    verified/initialized again at the first computed batch — initializing
    requires a known device (a predict never resolves one), so a None device
    there raises loudly; ``run()`` passes the orchestrator's post-fit
    ``resolved_device``. The final ``evaluate_model`` scorecard call is NOT
    checkpointed (single call, no clean granularity).
    """
    if val_df.is_empty():
        return val_df.select(["era", "id"]).with_columns(
            pl.Series("prediction", [], dtype=pl.Float64)
        )
    manifest_path = (
        validation_checkpoint_dir / "manifest.json"
        if validation_checkpoint_dir is not None
        else None
    )
    if manifest_path is not None:
        if manifest_path.exists():
            verify_checkpoint_manifest(
                manifest_path,
                checkpoint_device,
                checkpoint_kind="validation_checkpoints",
                data_fingerprint=data_fingerprint,
                environment=environment,
            )
        else:
            ensure_no_torn_tree(
                manifest_path,
                checkpoint_kind="validation_checkpoints",
                part_glob="preds_batch_*.parquet",
            )
    manifest_checked = False
    chunks: list[pl.DataFrame] = []
    for batch_index, batch in enumerate(_era_batch_frames(val_df, batch_eras)):
        batch_path = (
            validation_checkpoint_dir / f"preds_batch_{batch_index:02d}.parquet"
            if validation_checkpoint_dir is not None
            else None
        )
        if batch_path is not None and batch_path.exists():
            try:
                chunk = pl.read_parquet(batch_path)
            except Exception as exc:
                raise ValueError(
                    f"corrupt validation checkpoint {batch_path}: {exc}"
                ) from exc
            logger.info(
                "[validation] batch %02d: loaded validation checkpoint %s",
                batch_index,
                batch_path,
            )
            chunks.append(chunk)
            continue
        if manifest_path is not None and not manifest_checked:
            # PINNED DECISION (mirrors the OOF/deploy paths): the manifest is
            # verified / written at the first computed batch — an earlier
            # write would record a vacuous device, and a predict never
            # resolves one, so a None checkpoint_device here raises loudly
            # (the deploy fits that precede the stage resolve it normally).
            if manifest_path.exists():
                verify_checkpoint_manifest(
                    manifest_path,
                    checkpoint_device,
                    checkpoint_kind="validation_checkpoints",
                    data_fingerprint=data_fingerprint,
                    environment=environment,
                )
            else:
                if checkpoint_device is None:
                    raise ValueError(
                        "validation_checkpoints requires a resolved fit device "
                        "to initialize its manifest, but checkpoint_device is "
                        "None (the deploy fits resolve it before the "
                        "validation stage). Delete the validation_checkpoints "
                        "directory to force a full repredict."
                    )
                write_bytes_atomic(
                    json.dumps(
                        checkpoint_manifest(
                            checkpoint_device,
                            data_fingerprint=data_fingerprint,
                            environment=environment,
                        ),
                        sort_keys=True,
                    ).encode("utf-8"),
                    manifest_path,
                )
            manifest_checked = True
        chunk = _predict_era_batch(batch, feature_cols, predict_fn)
        if batch_path is not None:
            write_frame_atomic(chunk, batch_path)
            logger.info(
                "[validation] batch %02d: predicted and wrote validation "
                "checkpoint %s",
                batch_index,
                batch_path,
            )
        chunks.append(chunk)
    return pl.concat(chunks)

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
        # Rebuild identity (spec §3.1): persist the EXACT data fingerprint that
        # enters the run_id hash. Computed once here so the manifest field can
        # never drift from the run-id data term (a re-computation at run() time
        # could differ if the data files change between construction and run).
        self._data_fingerprint = _data_fingerprint(config)
        # The portable environment identity (pinned package versions, no paths
        # or timestamps) — computed once so run.json and every checkpoint
        # manifest record the same value (spec §3.1).
        self._environment = _portable_environment()
        self._run_id = self._compute_run_id(
            config, data_fingerprint=self._data_fingerprint
        )

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
        # Experiment layout (spec §3): every runner output lives under
        # experiments/<slug>/runs/<run_id>/; the slug is the validated run name.
        slug = paths.validate_slug(self._config.run.name)
        run_dir = paths.run_dir(slug, self._run_id)

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
        # Leakage law (AGENTS.md §4): data-aware purge/horizon floor — enforced
        # now that the era count is known (real-data regime only; small
        # synthetic datasets are governed by the splitter's geometry).
        enforce_purge_horizon_law(
            len(set(train_df.get_column("era").to_list())), self._config
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
            checkpoint_dir=(run_dir / "oof_checkpoints"),
            data_fingerprint=self._data_fingerprint,
            environment=self._environment,
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
            cache_dir=(
                paths.shared_cache_dir(self._config.run.artifacts_dir)
                / "neutralization"
            ),
            max_cache_bytes=self._config.risk.cache_max_bytes,
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
            pipeline = _build_deploy_pipeline(
                orchestrator=model_orchestrator,
                train_df=train_df,
                feature_cols=feature_cols,
                target_cols=target_cols,
                weights=weights,
                proportion=neutralization_proportion,
                data=self._config.data,
                deploy_checkpoint_dir=(run_dir / "deploy_checkpoints"),
                data_fingerprint=self._data_fingerprint,
                environment=self._environment,
            )

        scorecard = None
        validation_predictions = None
        validation_purge = None
        if self._config.evaluation.validation_scorecard:
            scorecard, validation_predictions, validation_purge = (
                self._run_validation_stage(
                    predict_fn=pipeline[0],
                    feature_cols=feature_cols,
                    validation_checkpoint_dir=(run_dir / "validation_checkpoints"),
                    checkpoint_device=(
                        str(model_orchestrator.resolved_device)
                        if model_orchestrator.resolved_device is not None
                        else None
                    ),
                    data_fingerprint=self._data_fingerprint,
                    environment=self._environment,
                )
            )
            logger.info("[run] validation scorecard ready: corr_sharpe_ac=%.5f",
                        scorecard.corr_sharpe_ac.value)

        artifact = None
        if deploy:
            logger.info("[run] serializing deploy artifact")
            assert pipeline is not None  # built once above when deploy=True
            artifact = _serialize_predict_artifact(
                predict_fn=pipeline[0],
                model_meta=pipeline[1],
                artifact_path=(run_dir / "predict.pkl"),
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
            # Rebuild identity (spec §3.1): data_fingerprint is the SAME value
            # hashed into the run_id (computed once in __init__); code
            # fingerprint/environment are the portable helpers (no paths);
            # pipeline_device is the config knob, oof_device the actual fit
            # device (post-fit resolved_device, config fallback).
            "data_fingerprint": self._data_fingerprint,
            "code_fingerprint": _compute_code_fingerprint(),
            "environment": self._environment,
            "pipeline_device": str(self._config.model.device),
            "oof_device": str(
                model_orchestrator.resolved_device or self._config.model.device
            ),
            # Present for every run whose validation/deploy closure ends with
            # the per-era (0,1) tie_kept_rank step. Absent in pre-fix legacy
            # rows: their max_feature_exposure was measured on unranked
            # predictions (~machine epsilon) and is not comparable to post-fix
            # values (dashboard/meta null it with a documented reason).
            "scorecard_prediction_scale": "percentile_rank",
            "metrics": dataclasses.asdict(metrics),
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
        checkpoint_dir: Path | None = None,
        data_fingerprint: str | None = None,
        environment: str | None = None,
    ) -> pl.DataFrame:
        """Delegate to the shared OOF implementation (C10, audit SEV-2 #5).

        The runner and research.py once each carried a copy of the
        leakage-critical OOF construction; the single source now lives in
        ``nmr._oof.train_multi_target_oof`` and this method only adds
        run-scoped logging. The run-scoped ``checkpoint_dir`` routes the
        shared helper to its checkpoint-aware path (fold parts persisted and
        replayed on resume — spec 2026-08-20-oof-checkpoint-resume); the
        rebuild-identity terms (spec §3.1) are recorded in the checkpoint
        manifest so a data-snapshot or environment drift refuses resume.
        """
        targets = self._config.data.targets
        logger.info(
            "[train_multi_target_oof] training %d target(s): %s", len(targets), targets
        )
        t0 = time.time()
        stacked = train_multi_target_oof(
            model_orchestrator,
            train_df,
            feature_cols=feature_cols,
            splitter=splitter,
            targets=targets,
            checkpoint_dir=checkpoint_dir,
            data_fingerprint=data_fingerprint,
            environment=environment,
        )
        logger.info(
            "[train_multi_target_oof] stacked OOF complete in %.1fs (shape: %s)",
            time.time() - t0,
            stacked.shape,
        )
        return stacked

    def _run_validation_stage(
        self,
        *,
        predict_fn,
        feature_cols: Sequence[str],
        validation_checkpoint_dir: Path | None = None,
        checkpoint_device: str | None = None,
        data_fingerprint: str | None = None,
        environment: str | None = None,
    ) -> tuple[MetricScorecard, pl.DataFrame, int]:
        data = self._config.data
        agent = IngestionAgent(data)
        # Config targets + main target + every horizon target column present in
        # the validation schema (benchmark_runner convention). Horizon stability
        # (and BMC) can only run when the scorecard receives the target_*_20/60
        # pairs; loading only config targets silently disabled the flagship
        # diagnostic on every runner-produced scorecard.
        schema_cols = list(pl.read_parquet_schema(data.path("validation.parquet")).keys())
        schema_target_cols = [
            c for c in schema_cols if c == "target" or c.startswith("target_")
        ]
        target_cols = list(
            dict.fromkeys(
                [*self._config.data.targets, self._config.evaluation.main_target, *schema_target_cols]
            )
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
            # Compare on the NUMERIC era index: the era column is zero-padded
            # ("0583"), so str(int) strings would match nothing below 1000 and
            # silently truncate the window to eras >= 1000 (regression: the
            # validation stage scored only 232 of 649 eras).
            val_df = val_df.filter(
                pl.col("era").cast(pl.Int32).is_in(all_eras[purge:])
            )
        logger.info(
            "[validation] dropping first %d validation eras (20D-target overlap); "
            "%d eras scored", purge, val_df.select(pl.col("era").n_unique()).item()
        )

        preds = _predict_validation_era_batches(
            val_df,
            feature_cols,
            predict_fn,
            _VAL_PREDICT_ERA_BATCH,
            validation_checkpoint_dir=validation_checkpoint_dir,
            checkpoint_device=checkpoint_device,
            data_fingerprint=data_fingerprint,
            environment=environment,
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

    @staticmethod
    def _compute_run_id(
        config: ExperimentConfig, data_fingerprint: str | None = None
    ) -> str:
        config_payload = _to_jsonable(dataclasses.asdict(config))
        _strip_path_dependent_fields(config_payload)
        supp_path = config.data.supplemental_feature_sets
        if data_fingerprint is None:
            # Public compute_run_id() path: derive the data snapshot here.
            data_fingerprint = _data_fingerprint(config)
        payload: dict[str, Any] = {
            "config": config_payload,
            "data_version": config.data.version,
            # B1 (audit SEV-1 #3): the data term enters run identity as a
            # snapshot fingerprint (era range, row counts, schema, features.json
            # content) — not the literal version string. Same config + same
            # data snapshot + same code + same env ⇒ same run_id, enforced on
            # the data term. See _data_fingerprint for detection limits.
            "data_fingerprint": data_fingerprint,
            "code_fingerprint": ExperimentRunner._code_fingerprint(),
            "environment": ExperimentRunner._environment_fingerprint(config.model.backend),
        }
        # Content identity for derived feature sets: the absolute path is
        # stripped above (never hashed), while the resolved file's SHA256 is
        # included so editing the file changes run identity (ARCHITECTURE.md
        # §P). The key is present only when a supplemental set is configured,
        # so configs without one keep their legacy run_ids byte-identical.
        if supp_path is not None:
            payload["supplemental_feature_sets_sha256"] = (
                ExperimentRunner._supplemental_fingerprint(supp_path)
            )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _supplemental_fingerprint(path: Path) -> str:
        """SHA256 of a resolved supplemental feature-sets file, CRLF-normalized.

        Line endings are normalized (CRLF -> LF) before hashing so Windows and
        POSIX checkouts of the same derived-sets file produce the same run_id —
        the same normalization ``_code_fingerprint`` applies to source files.
        """
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()

    @staticmethod
    def compute_run_id(config: ExperimentConfig) -> str:
        """Public accessor for the canonical run id (used by campaign tooling)."""
        return ExperimentRunner._compute_run_id(config)

    @staticmethod
    def _code_fingerprint(package_dir: Path | None = None) -> str:
        """SHA256 over sorted ``nmr/*.py`` names+contents.

        Delegates to the module-level :func:`_compute_code_fingerprint` — the
        run manifest and the run-id code term always share one implementation.
        Line endings are normalized (CRLF -> LF) before hashing so a Windows
        checkout (autocrlf) and a POSIX checkout of the same commit produce
        the same fingerprint — otherwise run_ids diverge across machines.
        """
        return _compute_code_fingerprint(package_dir)

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
            for name in ["numpy", "polars", "pandas", "lightgbm", "xgboost", "optuna"]
        }
        if backend == "catboost":
            packages["catboost"] = _package_version("catboost")
        return {
            "python_version": platform.python_version(),
            "packages": packages,
        }


def _compute_code_fingerprint(package_dir: Path | None = None) -> str:
    """Portable SHA-256 over the nmr package sources (no paths, no timestamps).

    Canonical rebuild-identity code fingerprint (spec §3.1): sorted
    ``nmr/*.py`` file names + contents, with line endings normalized (CRLF ->
    LF) so a Windows checkout (autocrlf) and a POSIX checkout of the same
    commit hash identically. The run-id code term
    (``ExperimentRunner._code_fingerprint``) and the persisted manifest field
    both delegate here, so they can never drift apart. Note this is the
    full-package identity — distinct from ``nmr/_oof.fitting_code_sha256``,
    which covers only the fitting-code subset for checkpoint manifests.
    """
    package_dir = package_dir or Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _portable_environment() -> str:
    """Normalized dependency identity — pinned package names + versions only.

    Rebuild identity (spec §3.1): a sorted ``name==version`` list over the
    requirements.txt pins (numpy, polars, lightgbm, xgboost, catboost, scipy,
    numerai-tools, cloudpickle), with no paths or timestamps. Unlike the
    run-id environment term (``ExperimentRunner._environment_fingerprint`` —
    config-aware and dict-shaped for byte-compat with legacy run_ids), this is
    the human-verifiable manifest record.
    """
    names = (
        "numpy",
        "polars",
        "lightgbm",
        "xgboost",
        "catboost",
        "scipy",
        "numerai-tools",
        "cloudpickle",
    )
    parts = [f"{name}=={_package_version(name)}" for name in names]
    return ",".join(sorted(parts))


def _build_deploy_pipeline(
    *,
    orchestrator: ModelOrchestrator,
    train_df: pl.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    weights: Sequence[float],
    proportion: float,
    data: DataConfig,
    deploy_checkpoint_dir: Path | None = None,
    include_validation: bool = False,
    data_fingerprint: str | None = None,
    environment: str | None = None,
) -> tuple[Callable[[pd.DataFrame], pd.DataFrame], dict[str, object]]:
    """Train per-target full-history models ONCE and return (predict, model_meta).

    Module-level so the promotion writer (nmr/promote.py) shares the exact
    closure construction with the runner — no second copy of the riskiest
    code. The closure's code path references only numpy/pandas plus the shared
    transform helpers; cloudpickle.register_pickle_by_value(nmr._transforms)
    embeds those helpers by value so the artifact loads without `nmr`.

    ``include_validation`` (promotion writer only): forwarded to
    ``train_full_history`` so a spawned full-history fit re-reads
    train+validation when the promotion scope is ``"full"`` and train only
    when ``"train_only"`` (fit-phase isolation). The runner passes the
    default False — its deploy artifact trains on the train frame it passes.

    With ``deploy_checkpoint_dir`` set (runner only; the promotion writer
    passes None), each fitted model is persisted with cloudpickle to
    ``<deploy_checkpoint_dir>/<target>.pkl`` (atomic write) and replayed on
    resume instead of refit — per-target deploy-fit checkpoints (spec
    2026-08-23-checkpoint-coverage-extension). The checkpoint root carries a
    ``manifest.json`` with the same code+device identity discipline as the
    OOF checkpoints, plus the rebuild-identity terms (``data_fingerprint``
    and ``environment``, spec §3.1) when the caller provides them; the
    recorded device is the orchestrator's post-fit ``resolved_device`` at the
    FIRST fitted target (deploy fits are CPU-only).
    """
    logger.info("[build_deploy_pipeline] training full-history models (CPU-only)")
    trained: dict[str, object] = {}
    manifest_path = (
        deploy_checkpoint_dir / "manifest.json"
        if deploy_checkpoint_dir is not None
        else None
    )
    manifest_written = False
    if manifest_path is not None:
        if manifest_path.exists():
            # current_device=None: the orchestrator's resolved_device at this
            # point belongs to the CV stage — a different identity. The deploy
            # manifest records the deploy fit's device (CPU-only), which is
            # exact-checked post-fit at the first fitted target below; the
            # schema check here still rejects unknown stored devices.
            verify_checkpoint_manifest(
                manifest_path,
                None,
                checkpoint_kind="deploy_checkpoints",
                data_fingerprint=data_fingerprint,
                environment=environment,
            )
        else:
            # PINNED DECISION (mirrors the OOF path): the manifest is written
            # at the FIRST fitted target, never here — resolved_device is None
            # until a fit completes, and an early write would record "None",
            # making the device guard pass vacuously on resume.
            ensure_no_torn_tree(
                manifest_path,
                checkpoint_kind="deploy_checkpoints",
                part_glob="*.pkl",
            )
    for target in target_cols:
        pkl_path = (
            deploy_checkpoint_dir / f"{target}.pkl"
            if deploy_checkpoint_dir is not None
            else None
        )
        if pkl_path is not None and pkl_path.exists():
            try:
                trained[target] = cloudpickle.loads(pkl_path.read_bytes())
            except Exception as exc:
                raise ValueError(
                    f"corrupt deploy checkpoint {pkl_path}: {exc}"
                ) from exc
            logger.info(
                "[build_deploy_pipeline] %s: loaded deploy checkpoint %s",
                target,
                pkl_path,
            )
            continue
        trained[target] = orchestrator.train_full_history(
            train_df,
            feature_cols=feature_cols,
            target_col=target,
            era_col="era",
            data=data,
            include_validation=include_validation,
        )
        if pkl_path is not None:
            if not manifest_written:
                resolved_device = str(orchestrator.resolved_device)
                if manifest_path is not None and manifest_path.exists():
                    verify_checkpoint_manifest(
                        manifest_path,
                        resolved_device,
                        checkpoint_kind="deploy_checkpoints",
                        data_fingerprint=data_fingerprint,
                        environment=environment,
                    )
                else:
                    write_bytes_atomic(
                        json.dumps(
                            checkpoint_manifest(
                                resolved_device,
                                data_fingerprint=data_fingerprint,
                                environment=environment,
                            ),
                            sort_keys=True,
                        ).encode("utf-8"),
                        manifest_path,
                    )
                manifest_written = True
            write_bytes_atomic(cloudpickle.dumps(trained[target]), pkl_path)
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
            # Final (0,1) percentile-rank step per era, AFTER neutralization —
            # the canonical neutralize -> rank order (Numerai's own notebook).
            # Required by the submission contract: raw output must be in (0,1)
            # (numerai_tools validate_values hard-asserts it); the deploy
            # artifact is consumed verbatim by Model Uploads. Rank-invariant
            # metrics are unaffected; max_feature_exposure legitimately moves
            # off machine-epsilon (the submitted vector is not feature-neutral).
            blended[mask] = tie_kept_rank(
                neutralize_array(combined, feature_matrix[mask], proportion)
            )
        return pd.DataFrame({"prediction": blended}, index=live_features.index)

    meta = {
        "targets": target_order,
        "weights": [float(w) for w in weights],
        "proportion": float(proportion),
        "geometry": "all_eras",
        "device": "cpu",
        "feature_names": ordered_features,
        "output_range": "tie_kept_rank (0,1) exclusive, per era",
    }
    return predict, meta


def _serialize_predict_artifact(
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


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


_DATA_FINGERPRINT_CACHE_NAME = "data_fingerprint.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parquet_snapshot(name: str, path: Path) -> dict[str, Any]:
    """Deterministic per-file snapshot: footer schema, row count, era stats.

    Row count comes from the parquet footer metadata; era min/max/count reads
    only the era column (fast). Byte size is deliberately excluded (a
    re-download with different compression must not change identity).
    """
    schema = pl.scan_parquet(path).collect_schema().names()
    row_count = pl.scan_parquet(path).select(pl.len()).collect().item()
    era_stats = (
        pl.scan_parquet(path)
        .select(
            pl.col("era").min().alias("era_min"),
            pl.col("era").max().alias("era_max"),
            pl.len().alias("era_count"),
        )
        .collect()
        .row(0)
    )
    return {
        "name": name,
        "schema": list(schema),
        "row_count": int(row_count),
        "era_min": era_stats[0],
        "era_max": era_stats[1],
        "era_count": int(era_stats[2]),
    }


def _data_fingerprint(config: ExperimentConfig) -> str:
    """SHA256 over a deterministic snapshot of the data-defining files.

    Per-file records for the files the run learns from and is scored on:
    ``{name, footer schema, footer row count, era min, era max, era count}``
    for train/validation (plus meta/benchmarks when
    ``evaluation.validation_scorecard`` consumes them — config-aware), and
    ``features.json`` by content SHA256.

    **Detection limits (documented):** restated feature values within an
    unchanged schema/row-count/era-stats are NOT detected; local edits are
    caught by the mtime-based cache invalidation. This is a snapshot marker,
    not a full content hash (a 28 GB+ parquet hash per run is disproportionate).

    Cached under ``artifacts/cache/`` keyed on (path, mtime, size) so repeated
    run_id computation is cheap. **Fail-loud** on missing data files: run_id
    requires the data snapshot (``run_campaign.py --dry-run`` needs the data
    present).
    """
    files = [
        ("train.parquet", True),
        ("validation.parquet", True),
        ("meta_model.parquet", config.evaluation.validation_scorecard),
        (
            "validation_benchmark_models.parquet",
            config.evaluation.validation_scorecard,
        ),
    ]
    records: list[dict[str, Any]] = []
    cache_keys: list[str] = []
    for name, required in files:
        path = config.data.path(name)
        if required and not path.is_file():
            raise ValueError(f"run_id requires the data fingerprint; {path} missing")
        if not path.is_file():
            continue
        stat = path.stat()
        cache_keys.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")
        records.append(_parquet_snapshot(name, path))
    features_path = config.data.data_dir / config.data.version / "features.json"
    if not features_path.is_file():
        raise ValueError(
            f"run_id requires the data fingerprint; {features_path} missing"
        )
    stat = features_path.stat()
    cache_keys.append(f"features.json:{stat.st_mtime_ns}:{stat.st_size}")
    records.append({"name": "features.json", "sha256": _sha256_file(features_path)})

    cache_path = config.run.artifacts_dir / "cache" / _DATA_FINGERPRINT_CACHE_NAME
    key = hashlib.sha256("\n".join(cache_keys).encode("utf-8")).hexdigest()
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        cached = {}
    if cached.get("key") == key:
        return str(cached["fingerprint"])
    fingerprint = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"key": key, "fingerprint": fingerprint}, sort_keys=True),
        encoding="utf-8",
    )
    return fingerprint


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
        data_section.pop("supplemental_feature_sets", None)

    run_section = config_payload.get("run")
    if isinstance(run_section, dict):
        run_section.pop("artifacts_dir", None)
