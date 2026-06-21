# NumerAI Docs Meta-Guide

This document is the primary entrypoint for both humans and LLM agents.

Goal: learn the Numerai Classic tournament from scratch with minimal noise and a deterministic reading path.

## 1) Fast Start For Agents

Read in this order:

1. `01-canon/overview.md`
2. `01-canon/data.md`
3. `01-canon/scoring/00-definitions.md`
4. `01-canon/scoring/01-correlation.md`
5. `01-canon/scoring/02-mmc-bmc.md`
6. `01-canon/scoring/03-fnc.md`
7. `01-canon/submissions.md`
8. `01-canon/staking.md`
9. `02-strategy/strategy-bible.md`
10. `01-canon/models.md`
11. `02-strategy/community-wisdom.md`
12. `03-reference/numerapi.md` and `03-reference/numerai-tools.md`
13. `04-research/research-program.md` (optional but high value)

If you only have 15 minutes, read sections 2 and 3 below, then read items 1-8 above.

## 2) Canonical Laws (Ground Truth)

### Data and identity laws

- Data is obfuscated tabular equity data; it is intentionally not tradable outside Numerai.
- Each row is stock-at-time; `id` is unique per stock per era.
- You cannot track a single stock across eras using `id`.
- `era` is the temporal unit; historical eras are weekly snapshots.
- Features are mostly binned values and may contain missing values.
- Main target is stock-specific residual return (beta-neutralized) and is never NaN.
- Auxiliary targets can be NaN and vary by neutralization scheme and horizon.

### Temporal and validation laws

- Random row-level CV is invalid.
- Targets are forward-looking and overlapping, so temporal leakage is easy.
- Use era-grouped, walk-forward validation with purge/embargo.
- Per-era scoring first, then aggregate. Flattened metrics are misleading.
- Optimize risk-adjusted stability (era-wise Sharpe), not only mean correlation.

### Scoring laws

- CORR is Numerai-specific rank correlation:
  - rank predictions (ties kept), gaussianize, apply power 1.5, then Pearson with transformed target.
  - tails matter more than the center.
- MMC is the covariance of your predictions with the target after tie-kept ranking, gaussianizing, and orthogonalizing them against the stake-weighted meta model (SWMM). It rewards unique, target-aligned signal, not consensus.
- FNC measures performance after neutralizing predictions to features.
- Rank order dominates; absolute prediction scale is secondary.

### Ensembling laws

- Ensemble by rank/percentile domain, not raw regression magnitude.
- Multi-target and multi-seed ensembling generally improves stability and uniqueness.
- Neutralization is a risk tool: reduces fragile linear feature exposure.

### Submission and round laws

- Submissions are predictions in [0, 1] (0 = lowest return, 0.5 = average, 1 = highest).
- A new round opens each day Tuesday through Saturday; rounds overlap (up to ~25 active at once) and resolve roughly one month later (~260 rounds/year).
- Only the latest valid submission in-window is selected for scoring.
- Late submissions can be scored but carry zero at-risk stake (no payout, no meta-model or payout-factor impact).

### Staking and economics laws

- NMR stake is optional but required for rewards/burn and influence on meta-model weighting.
- Per-round payout/burn is clipped at +/-5%. Canonical formula:
  `payout = stake * clip(payout_factor * (0.5*corr + 2*mmc), -0.05, 0.05)` (updated weighting: `0.75*corr + 2.25*mmc`).
- `payout_factor = min(1, stake_threshold / total_at_risk)`. Thresholds: Numerai 72000, Signals 36000, Crypto 10000.
- Negative outcomes burn stake irreversibly (sent to a null address, removed from supply; not taken by Numerai).
- Releasing a stake takes ~1 month (one scoring round).

Exact figures live in `01-canon/staking.md`; treat that file as authoritative if numbers ever drift.

## 3) Critical Caveat: Purge / Embargo Convention

You will see two conventions:

- Minimum horizon-based logic: 4 eras for 20D targets, 16 eras for 60D targets.
- Numerai benchmark walk-forward convention: 8 eras purge for 20D, 16 eras for 60D.

Interpretation for this repository:

- Treat 8/16 as the operational benchmark convention.
- Treat 4/16 as the theoretical minimum.

## 4) Importance Ranking (Tiered)

### Tier T0: Must-read canonical truth

- `01-canon/overview.md`
- `01-canon/data.md`
- `01-canon/scoring/00-definitions.md`
- `01-canon/scoring/01-correlation.md`
- `01-canon/scoring/02-mmc-bmc.md`
- `01-canon/scoring/03-fnc.md`
- `01-canon/submissions.md`
- `01-canon/staking.md`

### Tier T1: Core strategy and execution intuition

- `02-strategy/strategy-bible.md`
- `01-canon/models.md`
- `05-notebooks/1_hello_numerai.ipynb`
- `05-notebooks/2_feature_neutralization.ipynb`
- `05-notebooks/3_target_ensemble.ipynb`

### Tier T2: High-value context and heuristics

- `02-strategy/community-wisdom.md`
- `02-strategy/why-it-works.md`
- `04-research/research-program.md`

### Tier T3: Implementation reference

- `03-reference/numerapi.md`
- `03-reference/numerai-tools.md`
- `02-strategy/target-ensembling-math.md`
- `04-research/advanced-ideas.md`
- `04-research/neural-networks.md`

### Tier T4: Archive / non-essential for modeling core

- `99-archive/bounties.md`
- `99-archive/grandmasters-seasons.md`
- `99-archive/general-ml-cookbook.md`
- `99-archive/super-research.prompt.md`

## 5) Full File Map

| New Path | Tier | Purpose | Source |
|---|---|---|---|
| `01-canon/overview.md` | T0 | Tournament overview | `Overview.txt` |
| `01-canon/data.md` | T0 | Data structure and target semantics | `Data.txt` |
| `01-canon/models.md` | T1 | Official model and benchmark guidance | `Models.txt` |
| `01-canon/submissions.md` | T0 | Submission lifecycle and automation paths | `Submissions.txt` |
| `01-canon/staking.md` | T0 | NMR staking rules and payout mechanics | `Staking.txt` |
| `01-canon/scoring/00-definitions.md` | T0 | Statistical and scoring definitions | `Scoring-Definitions.txt` |
| `01-canon/scoring/01-correlation.md` | T0 | CORR definition and rationale | `Scoring-Correlation.txt` |
| `01-canon/scoring/02-mmc-bmc.md` | T0 | MMC/BMC definitions and calculation | `Scoring-Meta-Model-Contribution.txt` |
| `01-canon/scoring/03-fnc.md` | T0 | FNC definition and calculation | `Scoring-Feature-Neutral-Correlation.txt` |
| `02-strategy/strategy-bible.md` | T1 | Consolidated tactical bible | `bible.md` + `Golden Bible.txt` |
| `02-strategy/community-wisdom.md` | T2 | Community heuristics and caveats | `community_notes.md` |
| `02-strategy/why-it-works.md` | T2 | System-level architecture and intuition | `Pipeline Grand Scheme.txt` + `NumerAI Architecture Explained.txt` |
| `02-strategy/target-ensembling-math.md` | T3 | Meta-learning and stacking notes | `Gemini-Ensemble-Meta-Learning.txt` |
| `03-reference/numerapi.md` | T3 | NumerAPI practical reference | `numerapi_reference.md` + `API/` docs |
| `03-reference/numerai-tools.md` | T3 | numerai_tools scoring/ref utility map | `numerai_tools_reference.md` |
| `04-research/research-program.md` | T2 | Main advanced research playbook | `llm_reports/perplexity_deep_research.md` |
| `04-research/advanced-ideas.md` | T3 | Experimental ideas backlog | `llm_reports/perplexity_deep_research_ideas.md` |
| `04-research/neural-networks.md` | T3 | NN-specific advanced exploration | `llm_reports/perplexity_deep_research_NN.md` |
| `05-notebooks/*` | T1 | Executable onboarding and examples | `onboarding_notebooks/` + `community_models_and_notebooks/` |
| `99-archive/*` | T4 | Peripheral or low-priority context | archive sources |

## 6) Provenance And Merge Policy

- Canon docs in `01-canon` should be treated as highest authority.
- `02-strategy/strategy-bible.md` intentionally deduplicates overlapping guidance from two source bibles.
- `03-reference/numerapi.md` is a practical consolidated surface, not a full generated API spec.
- `04-research` is useful but speculative. Do not treat as protocol truth.
- The `Source` column in section 5 names the original files. Every original is preserved unmodified in `99-archive/raw-source/` (with normalized lowercase names, e.g. `Golden Bible.txt` -> `golden-bible.txt`). Nothing was deleted; low-value material was relocated, not destroyed.

## 7) Minimal Traversal Recipes

### A) Scoring comprehension only

Read:

1. `01-canon/scoring/00-definitions.md`
2. `01-canon/scoring/01-correlation.md`
3. `01-canon/scoring/02-mmc-bmc.md`
4. `01-canon/scoring/03-fnc.md`
5. `01-canon/staking.md`

### B) Data-to-submission full lifecycle

Read:

1. `01-canon/data.md`
2. `01-canon/models.md`
3. `01-canon/submissions.md`
4. `01-canon/staking.md`
5. `03-reference/numerapi.md`

### C) Robust modeling intuition

Read:

1. `02-strategy/strategy-bible.md`
2. `02-strategy/community-wisdom.md`
3. `02-strategy/why-it-works.md`
4. `04-research/research-program.md`

## 8) Scope Boundary

This reorganization is intentionally Classic-tournament-first. Signals/Crypto details are retained only where they clarify shared API mechanics or staking thresholds.
