# NumerAI Docs Meta-Guide

This document is the docs-library index for humans and LLM agents. For repository-wide routing to code, tests, workflows, and documentation owners, start with [`CODEBASE.md`](../CODEBASE.md).

Goal: learn the Numerai Classic tournament from scratch with minimal noise and a deterministic reading path.

## 1) Fast Start For Agents

Read in this order:

1. [`01-canon/NUMERAI-CANON-DOCS-README.md`](01-canon/NUMERAI-CANON-DOCS-README.md)
2. [`02-strategy/strategy-bible.md`](02-strategy/strategy-bible.md)
3. [`02-strategy/model-lifecycle.md`](02-strategy/model-lifecycle.md)
4. [`03-reference/numerapi.md`](03-reference/numerapi.md) and [`03-reference/numerai-tools.md`](03-reference/numerai-tools.md)
5. [`04-research/research-program.md`](04-research/research-program.md) (optional but high value)
6. [`06-evaluation/evaluation-suite-bible.md`](06-evaluation/evaluation-suite-bible.md) (how this repo judges a model)

If you only have 15 minutes, read the canon index and the evaluation bible.

## 2) Canonical Domain Source

The [Numerai canon index](01-canon/NUMERAI-CANON-DOCS-README.md) is the sole maintained source for official domain rules. It routes to the topic page for data, models, scoring, submissions, model uploads, staking, live scoring, and API behavior. This guide deliberately does not restate those laws.

## 3) Repository-Specific Evaluation

Validation, purge, metric, scorecard, and benchmark behavior are repository contracts. Use [`ARCHITECTURE.md`](../ARCHITECTURE.md) for implementation details and [`06-evaluation/evaluation-suite-bible.md`](06-evaluation/evaluation-suite-bible.md) for the evaluation protocol. The canon explains Numerai; these references explain how this repository implements and judges it.

## 4) Importance Ranking (Tiered)

### Tier T0: Must-read canonical truth

- [`01-canon/NUMERAI-CANON-DOCS-README.md`](01-canon/NUMERAI-CANON-DOCS-README.md) — canonical index and reading order

### Tier T1: Core strategy and execution intuition

- `02-strategy/strategy-bible.md`
- `01-canon/03-models.md`
- `05-notebooks/1_hello_numerai.ipynb`
- `05-notebooks/2_feature_neutralization.ipynb`
- `05-notebooks/3_target_ensemble.ipynb`
- `05-notebooks/example-model-sunshine.ipynb` (community example: multi-target + 25% neutralization + model upload)

### Tier T2: High-value context and heuristics

- `02-strategy/community-wisdom.md`
- `02-strategy/why-it-works.md`
- `04-research/research-program.md`

### Tier T3: Implementation reference

- `03-reference/numerapi.md`
- `03-reference/numerai-tools.md`
- `02-strategy/target-ensembling-math.md`
- `02-strategy/model-lifecycle.md` (repo-internal: how a model moves research → partial → full → staked over `experiments/`)
- `04-research/advanced-ideas.md` (merged with former `neural-networks.md`, 2026-08-06)
- `04-research/State-of-the-Art Deep Learning for Obfuscated, Non-Stationary Tabular Regression.md`
- [`superpowers/README.md`](superpowers/README.md) — active, narrowly scoped design records

### Tier T4: Archive / non-essential for modeling core

- [`99-archive/README.md`](99-archive/README.md) — archive index and authority boundary

## 5) Full File Map

| New Path | Tier | Purpose | Source |
|---|---|---|---|
| `01-canon/NUMERAI-CANON-DOCS-README.md` | T0 | Canonical domain index and reading order | this build |
| `01-canon/00-overview.md` | T0 | Tournament overview | `Overview.txt` |
| `01-canon/01-faq.md` | T0 | FAQ and scope boundaries | `FAQ.txt` |
| `01-canon/02-data.md` | T0 | Data structure and target semantics | `Data.txt` |
| `01-canon/03-models.md` | T1 | Official model and benchmark guidance | `Models.txt` |
| `01-canon/04-submissions.md` | T0 | Submission lifecycle and automation paths | `Submissions.txt` |
| `01-canon/05-model-uploads.md` | T0 | Hosted prediction contract and runtime | `Model-Uploads.txt` |
| `01-canon/06-staking-legacy.md` | T0 | Legacy NMR staking and payout mechanics | `Staking.txt` |
| `01-canon/07-staking-atomic.md` | T0 | Atomic staking and allocation mechanics | `Staking-Atomic.txt` |
| `01-canon/08-grandmasters.md` | T2 | Grandmasters, seasons, and titles | `Grandmasters.txt` |
| `01-canon/09-scoring-live.md` | T0 | Live scoring and leaderboard behavior | `Scoring-Live.txt` |
| `01-canon/10-scoring-reference.md` | T0 | Statistical and scoring definitions | `Scoring-Definitions.txt` |
| `01-canon/11-api-and-mcp.md` | T1 | API and MCP access | `API/` |
| `02-strategy/strategy-bible.md` | T1 | Consolidated tactical bible | `bible.md` + `Golden Bible.txt` |
| `02-strategy/community-wisdom.md` | T2 | Community heuristics and caveats | `community_notes.md` |
| `02-strategy/why-it-works.md` | T2 | System-level architecture and intuition | `Pipeline Grand Scheme.txt` + `NumerAI Architecture Explained.txt` |
| `02-strategy/target-ensembling-math.md` | T3 | Meta-learning and stacking notes | `Gemini-Ensemble-Meta-Learning.txt` |
| `02-strategy/model-lifecycle.md` | T3 | Model lifecycle + experiment-layout workflow: six stages, promotion, naming, git rules, operations | this build |
| `03-reference/numerapi.md` | T3 | NumerAPI practical reference | `numerapi_reference.md` + `API/` docs |
| `03-reference/numerai-tools.md` | T3 | numerai_tools scoring/ref utility map | `numerai_tools_reference.md` |
| `04-research/research-program.md` | T2 | Main advanced research playbook | `llm_reports/perplexity_deep_research.md` |
| `04-research/advanced-ideas.md` | T3 | Research ideas: tree-level upgrades + NN directions (merged with former `neural-networks.md`, 2026-08-06) | `llm_reports/perplexity_deep_research_ideas.md` + `perplexity_deep_research_NN.md` |
| `04-research/State-of-the-Art Deep Learning for Obfuscated, Non-Stationary Tabular Regression.md` | T3 | Deep tabular-DL survey: TabFM/ICL, TabR, neutralization-aware objectives, CPCV blueprint | deep-research survey (references in-file) |
| `04-research/state-of-the-art-boosting.md` | T3 | GBDT state-of-the-art survey (XGBoost/LightGBM/CatBoost): peer-reviewed, high-prestige sources from top institutions, ~last 3 years | research survey (references in-file) |
| `04-research/state-of-the-art-beyond-boosting.md` | T3 | Practitioner-grade survey of methods competitive beyond GBDT for low-signal large tabular regression (boosting, linear-regularized ensembles, kernel approximations, monotonic NNs, hybrid/meta-learners) | research survey (references in-file) |
| `04-research/pre-modelling-dataset-feature-study-2026-08.md` | T1 | **Golden pre-modelling reference (single source of truth)**: dataset refresh record, era/target/feature diagnostics (§1–6), feature-campaign evidence (§7: 12 cells × 2 backends, full 649-era validation window, CIs + FNE), decision log (§8), methodology & reproduction (§9), file/artifact map (§10). Regenerate via `analyze_dataset.py` + `render_dataset_report.py --campaign-log <log>` after each `refresh_data.py` | generated from `artifacts/reports/dataset_analysis/` dumps + campaign evidence parquets |
| `05-notebooks/*` | T1 | Executable onboarding and examples | `onboarding_notebooks/` + `community_models_and_notebooks/` |
| `05-notebooks/community_notebooks/analysis_and_tips.ipynb` | T2 | Community notebook: data analysis and modeling tips | `community_models_and_notebooks/` |
| `05-notebooks/community_notebooks/numerai-example-model-sunshine (1).ipynb` | T1 | Community example model notebook ("Numerai Example Model Sunshine", Kaggle re-download) | `community_models_and_notebooks/` |
| `06-evaluation/evaluation-suite-bible.md` | T0 | Evaluation suite spec of record (metrics, math, build slices E1–E6) | this build |
| `06-evaluation/benchmark-line-in-the-sand.md` | T0 | Active benchmark hierarchy and capital-gate reference | this build |
| `superpowers/README.md` | T3 | Index of active detailed design contracts | this build |
| [`superpowers/specs/2026-08-18-vanilla-dashboard-design.md`](superpowers/specs/2026-08-18-vanilla-dashboard-design.md) | T3 | Active renderer and dashboard presentation contract | approved design |
| [`superpowers/specs/2026-08-19-benchmark-fleet-design.md`](superpowers/specs/2026-08-19-benchmark-fleet-design.md) | T3 | Active untiered benchmark-fleet contract | approved design |
| [`superpowers/specs/2026-08-26-model-lifecycle-experiments-design.md`](superpowers/specs/2026-08-26-model-lifecycle-experiments-design.md) | T3 | Active experiment-layout and lifecycle contract | approved design |
| `99-archive/README.md` | T4 | Archive index and authority boundary | this build |
| `99-archive/dashboard-history-2026-08.md` | T4 | Condensed dashboard architecture provenance | retired delivery records |
| `99-archive/design-records-2026-08.md` | T4 | Condensed completed-design chronology | retired specs and plans |
| `99-archive/super-research.prompt.md` | T4 | Historical research context | archived prompt |

## 6) Provenance And Merge Policy

- Canon docs in `01-canon` should be treated as highest authority.
- `02-strategy/strategy-bible.md` intentionally deduplicates overlapping guidance from two source bibles.
- `03-reference/numerapi.md` is a practical consolidated surface, not a full generated API spec.
- `04-research` is useful but speculative. Do not treat as protocol truth.
- `superpowers/` contains only active design records whose detailed contracts
	have not yet been fully absorbed by an owner document. Once integrated,
	durable facts move to the owner, provenance is condensed under `99-archive/`,
	and completed implementation plans are deleted.
- `99-archive/` is provenance only and cannot override an active owner. Its
	`raw-source/` subtree is immutable and intentionally excluded from link
	maintenance.
- The `Source` column in section 5 names the original inputs. Surviving raw originals are preserved unmodified in `99-archive/raw-source/` (normalized lowercase names, e.g. `Golden Bible.txt` -> `golden-bible.txt`); the remaining sources (canon `*.txt`, `Scoring-*.txt`, `numerai_tools_reference.md`, `llm_reports/`, notebook collections) are not archived as separate files. The 2026-08-06 trim removed two superseded files (`04-research/the-state-of-the-art.md`, `99-archive/general-ml-cookbook.md`) and merged `04-research/neural-networks.md` into `04-research/advanced-ideas.md`.

## 7) Minimal Traversal Recipes

Subsets of the §1 fast-start order, for focused goals:

- **Scoring comprehension:** `01-canon/NUMERAI-CANON-DOCS-README.md` → `01-canon/10-scoring-reference.md` → `01-canon/09-scoring-live.md` → `01-canon/06-staking-legacy.md`
- **Data-to-submission lifecycle:** `01-canon/02-data.md` → `01-canon/03-models.md` → `01-canon/04-submissions.md` → `01-canon/05-model-uploads.md` → `03-reference/numerapi.md`
- **Robust modeling intuition:** `02-strategy/strategy-bible.md` → `02-strategy/community-wisdom.md` → `02-strategy/why-it-works.md` → `04-research/research-program.md`
- **Model promotion & lifecycle ops:** `02-strategy/model-lifecycle.md` → `06-evaluation/evaluation-suite-bible.md` (how this repo judges a model) → [`ARCHITECTURE.md`](../ARCHITECTURE.md) §N/§X–§Z (schemas)

## 8) Scope Boundary

This reorganization is intentionally Classic-tournament-first. Signals/Crypto details are retained only where they clarify shared API mechanics or staking thresholds.
