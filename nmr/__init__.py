"""nmr — Numerai V2 quantitative research framework.

The `nmr` package is the single tested system boundary. Notebooks and scripts
are a thin control plane; all logic lives here and is covered by `tests/`.
"""

from __future__ import annotations

from .benchmark import (
    NULL_BASELINES,
    TUTORIAL_NOTEBOOK_TO_MODEL_ID,
    BenchmarkSuite,
    assert_notebook_prediction_contract,
    assert_null_floor,
    assert_slice1_monotone,
    canonical_scorecards_bytes,
    discover_tutorial_notebooks,
    extract_oos_predictions,
    ingest_tutorial_prediction,
    ingest_tutorial_prediction_batch,
    scorecards_sha256,
)
from .config import ExperimentConfig, load_config, set_global_seeds
from .data import IngestionAgent
from .deployment import DeploymentArtifact, load_predict, serialize_predict
from .ensemble import Ensembler
from .evaluation import (
    MIN_OVERLAP_ERAS,
    EvaluationEngine,
    MetricSummary,
    NonVacuityError,
)
from .inference import (
    BootstrapCI,
    SeriesStats,
    ac_adjusted_sharpe,
    block_bootstrap_ci,
    deflated_sharpe,
    era_series_stats,
    resolve_bandwidth,
    resolve_block_len,
)
from .models import CVResult, ModelOrchestrator
from .payout import (
    PayoutResult,
    PayoutSeries,
    burn_rate,
    calmar,
    cvar,
    max_burn_streak,
    max_drawdown,
    payout_report,
    payout_series,
    sortino,
    time_to_recovery,
)
from .registry import RunRegistry
from .research import (
    HyperparameterSweep,
    NeutralizationFrontier,
    SweepResult,
    feature_exposure_report,
    neutralization_frontier,
)
from .risk import NeutralizationEngine
from .robustness import (
    HorizonStabilityResult,
    PerturbationResult,
    RegimeCorr,
    adversarial_perturbation,
    regime_conditioned_corr,
    time_horizon_stability,
)
from .runner import ExperimentRunner, RunResult
from .scorecard import MetricCell, MetricScorecard, evaluate_model
from .splitter import Fold, PurgedEraSplitter
from .submission import build_submission, validate_submission, write_submission

__all__ = [
    "ExperimentConfig",
    "load_config",
    "set_global_seeds",
    "NULL_BASELINES",
    "TUTORIAL_NOTEBOOK_TO_MODEL_ID",
    "BenchmarkSuite",
    "discover_tutorial_notebooks",
    "assert_notebook_prediction_contract",
    "extract_oos_predictions",
    "ingest_tutorial_prediction",
    "ingest_tutorial_prediction_batch",
    "assert_null_floor",
    "assert_slice1_monotone",
    "canonical_scorecards_bytes",
    "scorecards_sha256",
    "IngestionAgent",
    "DeploymentArtifact",
    "serialize_predict",
    "load_predict",
    "Ensembler",
    "CVResult",
    "ModelOrchestrator",
    "MetricSummary",
    "EvaluationEngine",
    "MIN_OVERLAP_ERAS",
    "NonVacuityError",
    "SeriesStats",
    "BootstrapCI",
    "era_series_stats",
    "resolve_block_len",
    "resolve_bandwidth",
    "block_bootstrap_ci",
    "ac_adjusted_sharpe",
    "deflated_sharpe",
    "PayoutSeries",
    "PayoutResult",
    "payout_series",
    "payout_report",
    "burn_rate",
    "cvar",
    "sortino",
    "max_drawdown",
    "calmar",
    "max_burn_streak",
    "time_to_recovery",
    "NeutralizationEngine",
    "PerturbationResult",
    "HorizonStabilityResult",
    "RegimeCorr",
    "adversarial_perturbation",
    "time_horizon_stability",
    "regime_conditioned_corr",
    "MetricCell",
    "MetricScorecard",
    "evaluate_model",
    "RunResult",
    "ExperimentRunner",
    "RunRegistry",
    "SweepResult",
    "HyperparameterSweep",
    "NeutralizationFrontier",
    "neutralization_frontier",
    "feature_exposure_report",
    "Fold",
    "PurgedEraSplitter",
    "build_submission",
    "validate_submission",
    "write_submission",
    "__version__",
]
__version__ = "0.1.0"
