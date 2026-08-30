# Completed Design Records, August 2026

> **Status:** Condensed historical provenance. These records cannot override
> current code, tests, or the active documentation owners.

The repository previously retained separate design specifications and verbose
implementation plans for each delivery. Completed plans copied code, commands,
test counts, and task checklists that became stale after integration. They were
removed during the 2026-08-30 knowledge-base cleanup after current contracts
were verified in [`ARCHITECTURE.md`](../../ARCHITECTURE.md),
[`AGENTS.md`](../../AGENTS.md), and the nearest tests.

| Date | Workstream | Durable outcome |
| --- | --- | --- |
| 2026-08-13 | Train-only FDR screen | Feature screening and train-only evidence became tested research behavior. |
| 2026-08-14 | Campaign DSR trial tracking | The design remained draft/pending review; trial-lineage behavior was implemented later and is documented by the current campaign/meta contracts. |
| 2026-08-14 | Dynamic column-sampling floor | Backend feature-sampling constraints moved into model configuration. |
| 2026-08-15 | Benchmark hierarchy | The tiered hierarchy, gates, and deterministic scorecards became benchmark and evaluation contracts. |
| 2026-08-15 | Evaluation suite v2.5 | Capital-readiness metrics and strict failure behavior entered the evaluation specification. |
| 2026-08-15 | Dependency pinning | Exact direct dependency pins and clean-room verification became contributor policy. |
| 2026-08-16 to 2026-08-18 | Dashboard iterations | Plotly and parallel renderers were superseded by the single-renderer Model Tournament; see [`dashboard-history-2026-08.md`](dashboard-history-2026-08.md). |
| 2026-08-17 | Model-family markers | Full-version discovery evolved into the current experiment lifecycle and export validity model. |
| 2026-08-19 | Coverage, mutation, oracle, and promotion depth | CI gates and deep correctness tests became scripts, workflows, and test-suite contracts. |
| 2026-08-20 to 2026-08-23 | Checkpoint coverage | Identity-bound OOF, deploy, and validation checkpoints became runner contracts. |

Three design specifications remain active because they still provide detailed
contracts used by current work:

- [`vanilla dashboard`](../superpowers/specs/2026-08-18-vanilla-dashboard-design.md)
- [`benchmark fleet`](../superpowers/specs/2026-08-19-benchmark-fleet-design.md)
- [`model lifecycle and experiments`](../superpowers/specs/2026-08-26-model-lifecycle-experiments-design.md)