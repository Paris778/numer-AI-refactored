# Active Design Records

This area holds detailed design contracts that are still active and too narrow
for the repository-wide architecture document. It is not a task backlog.

| Design | Status | Current owner |
| --- | --- | --- |
| [`Vanilla dashboard`](specs/2026-08-18-vanilla-dashboard-design.md) | Active | `nmr/dashboard.py`, `dashboard_ui/`; architecture section W |
| [`Benchmark fleet`](specs/2026-08-19-benchmark-fleet-design.md) | Active | `nmr/benchmark_fleet.py`; architecture benchmark-fleet section |
| [`Model lifecycle and experiments`](specs/2026-08-26-model-lifecycle-experiments-design.md) | Active | `nmr/paths.py`, `nmr/lifecycle.py`, `nmr/experiment_store.py`, `nmr/registry.py`; architecture sections X-Z |
| [`Model-development OS`](specs/2026-09-07-model-os-design.md) | Active | `nmr/predictions.py`, `nmr/features.py` (`derive_feature_sets`); architecture prediction-artifact section |
| [`Extensible model backend registry`](specs/2026-09-08-model-backend-registry-design.md) | Active | `nmr/model_backend_protocol.py`, `nmr/model_backend_registry.py`, `nmr/model_backend_lightgbm.py`, `nmr/model_backend_xgboost.py`, `nmr/model_backend_catboost.py`, `nmr/model_backend_ridge.py`, `nmr/models.py`, `nmr/opt.py`, `nmr/promote.py`; architecture backend-modeling/HPO/promotion sections |

When a design is fully integrated, durable contracts move to their named owner,
its provenance is condensed under [`docs/99-archive/`](../99-archive/README.md),
and its implementation plan is deleted. Archived designs never override current
code, tests, or the core documentation hierarchy.