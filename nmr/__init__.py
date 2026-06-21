"""nmr — Numerai V2 quantitative research framework.

The `nmr` package is the single tested system boundary. Notebooks and scripts
are a thin control plane; all logic lives here and is covered by `tests/`.
"""

from __future__ import annotations

from .config import ExperimentConfig, load_config, set_global_seeds
from .data import IngestionAgent
from .evaluation import EvaluationEngine, MetricSummary
from .models import CVResult, ModelOrchestrator
from .risk import NeutralizationEngine
from .splitter import Fold, PurgedEraSplitter

__all__ = [
    "ExperimentConfig",
    "load_config",
    "set_global_seeds",
    "IngestionAgent",
    "CVResult",
    "ModelOrchestrator",
    "MetricSummary",
    "EvaluationEngine",
    "NeutralizationEngine",
    "Fold",
    "PurgedEraSplitter",
    "__version__",
]
__version__ = "0.1.0"
