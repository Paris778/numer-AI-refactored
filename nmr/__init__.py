"""nmr — Numerai V2 quantitative research framework.

The `nmr` package is the single tested system boundary. Notebooks and scripts
are a thin control plane; all logic lives here and is covered by `tests/`.
"""

from __future__ import annotations

from .config import ExperimentConfig, load_config, set_global_seeds
from .data import IngestionAgent
from .deployment import DeploymentArtifact, load_predict, serialize_predict
from .ensemble import Ensembler
from .evaluation import EvaluationEngine, MetricSummary
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
from .registry import RunRegistry
from .research import (
    HyperparameterSweep,
    NeutralizationFrontier,
    SweepResult,
    feature_exposure_report,
    neutralization_frontier,
)
from .risk import NeutralizationEngine
from .runner import ExperimentRunner, RunResult
from .splitter import Fold, PurgedEraSplitter
from .submission import build_submission, validate_submission, write_submission

__all__ = [
    "ExperimentConfig",
    "load_config",
    "set_global_seeds",
    "IngestionAgent",
    "DeploymentArtifact",
    "serialize_predict",
    "load_predict",
    "Ensembler",
    "CVResult",
    "ModelOrchestrator",
    "MetricSummary",
    "EvaluationEngine",
    "SeriesStats",
    "BootstrapCI",
    "era_series_stats",
    "resolve_block_len",
    "resolve_bandwidth",
    "block_bootstrap_ci",
    "ac_adjusted_sharpe",
    "deflated_sharpe",
    "NeutralizationEngine",
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
