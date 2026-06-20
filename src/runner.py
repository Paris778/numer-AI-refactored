from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from src.data import IngestionAgent
from src.deployment import AdversarialStressTester, DeploymentHarness, StressTestResult
from src.evaluation import EvaluationEngine, EvaluationSummary, FastFailGateResult
from src.features import PurgedEraSplitter
from src.models import CrossValidationResult, ModelOrchestrator
from src.risk import NeutralizationEngine


@dataclass(frozen=True)
class PromotionRunResult:
    evaluation_summary: EvaluationSummary
    gate_result: FastFailGateResult
    parity_eras: tuple[str, ...]
    parity_custom_summary: EvaluationSummary
    parity_official_summary: EvaluationSummary
    stress_result: StressTestResult
    artifact_dir: Path
    smoke_test_passed: bool
    cross_validation_result: CrossValidationResult
    log_lines: tuple[str, ...]


class PromotionRunner:
    def __init__(
        self,
        *,
        agent: IngestionAgent,
        dataset_name: str,
        feature_subset: str,
        target_column: str,
        splitter: PurgedEraSplitter,
        orchestrator: ModelOrchestrator,
        custom_evaluation_engine: EvaluationEngine,
        official_evaluation_engine: EvaluationEngine,
        neutralization_engine: NeutralizationEngine,
        stress_tester: AdversarialStressTester,
        deployment_harness: DeploymentHarness,
        artifact_dir: Path,
        config_metadata: dict[str, Any],
        gate_kwargs: dict[str, Any] | None = None,
        requirements_path: Path | None = None,
        era_slice: Sequence[str] | None = None,
        parity_era_count: int = 5,
        stress_noise_std_ratio: float = 0.05,
        random_state: int = 42,
        id_col: str = "id",
        era_col: str = "era",
    ) -> None:
        if custom_evaluation_engine.backend != "custom":
            raise ValueError("custom_evaluation_engine must use backend='custom'")
        if official_evaluation_engine.backend != "official":
            raise ValueError("official_evaluation_engine must use backend='official'")

        self.agent = agent
        self.dataset_name = dataset_name
        self.feature_subset = feature_subset
        self.target_column = target_column
        self.splitter = splitter
        self.orchestrator = orchestrator
        self.custom_evaluation_engine = custom_evaluation_engine
        self.official_evaluation_engine = official_evaluation_engine
        self.neutralization_engine = neutralization_engine
        self.stress_tester = stress_tester
        self.deployment_harness = deployment_harness
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        self.config_metadata = dict(config_metadata)
        self.gate_kwargs = dict(gate_kwargs or {})
        self.requirements_path = (
            Path(requirements_path).expanduser().resolve()
            if requirements_path is not None
            else None
        )
        self.era_slice = tuple(era_slice) if era_slice is not None else None
        self.parity_era_count = parity_era_count
        self.stress_noise_std_ratio = stress_noise_std_ratio
        self.random_state = random_state
        self.id_col = id_col
        self.era_col = era_col

    def run(self) -> PromotionRunResult:
        logs: list[str] = []
        feature_names = self.agent.get_feature_names(self.feature_subset)

        logs.append("[1/8] Ingest & Split")
        working_frame = self._load_working_frame(feature_names)
        unique_eras = working_frame.get_column(self.era_col).unique().sort().to_list()
        logs.append(
            f"Loaded dataset='{self.dataset_name}' subset='{self.feature_subset}' rows={working_frame.height} eras={len(unique_eras)}"
        )

        logs.append("[2/8] Train")
        cv_result = self.orchestrator.train_cross_validation(
            working_frame,
            self.splitter,
            id_col=self.id_col,
            era_col=self.era_col,
        )
        logs.append(
            f"Cross-validation trained models={cv_result.model_count} backend={cv_result.backend} fit_seconds={cv_result.total_fit_seconds:.3f}"
        )

        evaluation_frame = working_frame.with_columns(
            pl.Series(
                self.custom_evaluation_engine.prediction_col,
                cv_result.oof_predictions,
                dtype=pl.Float64,
            )
        )

        logs.append("[3/8] Evaluate (Custom)")
        evaluation_summary = self.custom_evaluation_engine.summarize(
            self.custom_evaluation_engine.evaluate_eras(
                evaluation_frame,
                feature_columns=feature_names,
                neutralization_engine=self.neutralization_engine,
                neutralization_subset_name=self.feature_subset,
            )
        )
        logs.append(
            f"Custom evaluation mean_corr={evaluation_summary.mean_corr:.6f} mean_fnc={self._format_optional(evaluation_summary.mean_fnc)} max_feature_exposure={self._format_optional(evaluation_summary.max_feature_exposure)}"
        )

        logs.append("[4/8] Gate")
        gate_result = self.custom_evaluation_engine.fast_fail_gate(
            evaluation_summary,
            **self.gate_kwargs,
        )
        if not gate_result.passed:
            raise ValueError(
                "Fast-fail gate rejected candidate: " + "; ".join(gate_result.failures)
            )
        logs.append("Fast-fail gate passed")

        logs.append("[5/8] Oracle Parity (Live Slice)")
        parity_eras, parity_custom_summary, parity_official_summary = (
            self._run_parity_gate(
                evaluation_frame,
                feature_names,
            )
        )
        logs.append(
            "Oracle parity passed for eras="
            + ",".join(parity_eras)
            + f" | mean_corr={parity_custom_summary.mean_corr:.6f} mean_fnc={self._format_optional(parity_custom_summary.mean_fnc)}"
        )

        logs.append("[6/8] Stress Test (v1)")
        stress_result = self.stress_tester.evaluate_noise_resilience(
            working_frame,
            cv_result.models,
            feature_names,
            noise_std_ratio=self.stress_noise_std_ratio,
            neutralization_engine=self.neutralization_engine,
            neutralization_subset_name=self.feature_subset,
        )
        if not stress_result.passed:
            raise ValueError(
                f"Stress test failed: degradation_pct={stress_result.degradation_pct:.3f}"
            )
        logs.append(
            f"Stress test passed degradation_pct={stress_result.degradation_pct:.3f}"
        )

        logs.append("[7/8] Serialize")
        artifact_dir = self.deployment_harness.serialize_candidate(
            self.artifact_dir,
            cv_result.models,
            feature_names,
            self.config_metadata,
            requirements_path=self.requirements_path,
        )
        manifest = json.loads(
            (artifact_dir / "manifest.json").read_text(encoding="utf-8")
        )
        logs.append(f"Wrote artifact bundle to {artifact_dir}")
        logs.append(
            "Recorded SHA-256 hashes for model files: "
            + ", ".join(
                f"{file_name}={file_hash}"
                for file_name, file_hash in sorted(manifest["model_hashes"].items())
            )
        )

        logs.append("[8/8] Smoke Test")
        control_live = working_frame.select([self.id_col, self.era_col, *feature_names])
        pre_serialization_predictions = self.orchestrator.predict_ensemble(
            control_live,
            models=cv_result.models,
        )
        post_serialization = self.deployment_harness.predict_live(
            control_live.select([self.id_col, *feature_names]),
            artifact_dir,
        )
        post_serialization_predictions = post_serialization.get_column(
            "prediction"
        ).to_numpy()
        smoke_test_passed = np.array_equal(
            pre_serialization_predictions,
            post_serialization_predictions,
        )
        if not smoke_test_passed:
            raise ValueError("Post-reload smoke test predictions do not match exactly")
        logs.append("Post-reload smoke test passed with exact prediction match")

        return PromotionRunResult(
            evaluation_summary=evaluation_summary,
            gate_result=gate_result,
            parity_eras=parity_eras,
            parity_custom_summary=parity_custom_summary,
            parity_official_summary=parity_official_summary,
            stress_result=stress_result,
            artifact_dir=artifact_dir,
            smoke_test_passed=smoke_test_passed,
            cross_validation_result=cv_result,
            log_lines=tuple(logs),
        )

    def _load_working_frame(self, feature_names: Sequence[str]) -> pl.DataFrame:
        lazy_frame = self.agent.scan_dataset(
            self.dataset_name,
            feature_subset=self.feature_subset,
            include_metadata=True,
            include_targets=False,
            extra_columns=[self.target_column],
        )
        if self.era_slice is not None:
            lazy_frame = lazy_frame.filter(pl.col(self.era_col).is_in(self.era_slice))

        return lazy_frame.select(
            [self.id_col, self.era_col, self.target_column, *feature_names]
        ).collect()

    def _run_parity_gate(
        self,
        evaluation_frame: pl.DataFrame,
        feature_names: Sequence[str],
    ) -> tuple[tuple[str, ...], EvaluationSummary, EvaluationSummary]:
        rng = np.random.default_rng(self.random_state)
        unique_eras = np.asarray(
            evaluation_frame.get_column(self.era_col).unique().sort().to_list(),
            dtype=str,
        )
        if unique_eras.size == 0:
            raise ValueError("No eras available for parity gate")
        sample_size = min(self.parity_era_count, unique_eras.size)
        selected_eras = tuple(
            sorted(rng.choice(unique_eras, size=sample_size, replace=False).tolist())
        )
        parity_frame = evaluation_frame.filter(
            pl.col(self.era_col).is_in(selected_eras)
        )

        custom_summary = self.custom_evaluation_engine.summarize(
            self.custom_evaluation_engine.evaluate_eras(
                parity_frame,
                feature_columns=feature_names,
                neutralization_engine=self.neutralization_engine,
                neutralization_subset_name=self.feature_subset,
            )
        )
        official_summary = self.official_evaluation_engine.summarize(
            self.official_evaluation_engine.evaluate_eras(
                parity_frame,
                feature_columns=feature_names,
            )
        )

        self._assert_optional_allclose(
            custom_summary.mean_corr,
            official_summary.mean_corr,
            name="mean_corr",
        )
        self._assert_optional_allclose(
            custom_summary.mean_fnc,
            official_summary.mean_fnc,
            name="mean_fnc",
        )
        self._assert_optional_allclose(
            custom_summary.max_feature_exposure,
            official_summary.max_feature_exposure,
            name="max_feature_exposure",
        )
        return selected_eras, custom_summary, official_summary

    @staticmethod
    def _assert_optional_allclose(
        left: float | None,
        right: float | None,
        *,
        name: str,
    ) -> None:
        if left is None and right is None:
            return
        if left is None or right is None:
            raise ValueError(f"Parity mismatch for {name}: {left!r} vs {right!r}")
        if not np.allclose(left, right, rtol=1e-8, atol=1e-8):
            raise ValueError(
                f"Parity mismatch for {name}: custom={left:.12f} official={right:.12f}"
            )

    @staticmethod
    def _format_optional(value: float | None) -> str:
        return "None" if value is None else f"{value:.6f}"
