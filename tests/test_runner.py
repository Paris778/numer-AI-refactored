import json
from pathlib import Path

import polars as pl

from src.data import IngestionAgent
from src.deployment import AdversarialStressTester, DeploymentHarness
from src.evaluation import EvaluationEngine
from src.features import PurgedEraSplitter
from src.models import ModelOrchestrator
from src.protocols import FeaturePipeline, IdentityTransformer
from src.risk import NeutralizationEngine
from src.runner import PromotionRunner


def test_promotion_runner_executes_full_dag(tmp_path: Path) -> None:
    rows_per_era = 8
    eras = [f"{era:04d}" for era in range(1, 11) for _ in range(rows_per_era)]
    row_index = pl.Series("row_index", list(range(len(eras))), dtype=pl.Float64)
    toy_frame = pl.DataFrame(
        {
            "id": [f"toy_{index:04d}" for index in range(len(eras))],
            "era": eras,
            "feature_alpha": row_index.sin() * 0.5 + 0.5,
            "feature_beta": (row_index / 3.0).cos() * 0.5 + 0.5,
            "feature_gamma": ((row_index % 5) / 5.0),
        }
    ).with_columns(
        (
            0.6 * pl.col("feature_alpha")
            - 0.3 * pl.col("feature_beta")
            + 0.2 * pl.col("feature_gamma")
        ).alias("target")
    )

    data_root = tmp_path / "data" / "v5.2"
    data_root.mkdir(parents=True)
    toy_frame.write_parquet(data_root / "train.parquet")
    toy_frame.select(["id", "era"]).with_columns(
        (0.55 * pl.col("id").cum_count().cast(pl.Float64)).alias("v52_lgbm_ender20")
    ).write_parquet(data_root / "train_benchmark_models.parquet")
    (data_root / "features.json").write_text(
        json.dumps(
            {
                "feature_sets": {
                    "small": ["feature_alpha", "feature_beta", "feature_gamma"]
                }
            }
        ),
        encoding="utf-8",
    )
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("polars\nlightgbm\nnumpy\nscipy\n", encoding="utf-8")

    agent = IngestionAgent(
        data_root,
        dataset_files={
            "train": "train.parquet",
            "train_benchmark_models": "train_benchmark_models.parquet",
        },
    )
    splitter = PurgedEraSplitter(n_splits=3, purge_buffer=1)
    orchestrator = ModelOrchestrator(
        feature_names=agent.get_feature_names("small"),
        target_columns=["target"],
        model_library="lightgbm",
        prefer_gpu=False,
        early_stopping_rounds=10,
        model_params={
            "n_estimators": 80,
            "learning_rate": 0.1,
            "num_leaves": 15,
            "min_child_samples": 5,
        },
    )
    custom_engine = EvaluationEngine(
        era_col="era",
        prediction_col="prediction",
        target_col="target",
        id_col="id",
        backend="custom",
    )
    official_engine = EvaluationEngine(
        era_col="era",
        prediction_col="prediction",
        target_col="target",
        id_col="id",
        backend="official",
    )
    neutralization_engine = NeutralizationEngine(cache_root=tmp_path / "cache")
    stress_tester = AdversarialStressTester(
        custom_engine,
        degradation_threshold=0.95,
        random_state=7,
    )
    deployment_harness = DeploymentHarness()

    runner = PromotionRunner(
        agent=agent,
        dataset_name="train",
        feature_subset="small",
        target_column="target",
        splitter=splitter,
        orchestrator=orchestrator,
        custom_evaluation_engine=custom_engine,
        official_evaluation_engine=official_engine,
        neutralization_engine=neutralization_engine,
        stress_tester=stress_tester,
        deployment_harness=deployment_harness,
        artifact_dir=tmp_path / "artifacts" / "bundle",
        config_metadata={"target_column": "target", "mode": "cv"},
        feature_pipeline=FeaturePipeline([IdentityTransformer()]),
        gate_kwargs={
            "min_mean_corr": -1.0,
            "min_sharpe_corr": -10.0,
            "max_drawdown_corr": 10.0,
        },
        requirements_path=requirements_path,
        benchmark_dataset_name="train_benchmark_models",
        benchmark_prediction_col="v52_lgbm_ender20",
        benchmark_mmc_col="v52_lgbm_ender20",
        parity_era_count=5,
        stress_noise_std_ratio=0.05,
    )

    result = runner.run()

    assert result.smoke_test_passed is True
    assert result.payload_path.exists()
    assert len(result.parity_eras) == 5
    assert result.evaluation_summary.mean_mmc is not None
    assert any("Oracle parity passed" in line for line in result.log_lines)
    assert any("mean_mmc=" in line for line in result.log_lines)
    assert any("Post-reload smoke test passed" in line for line in result.log_lines)
