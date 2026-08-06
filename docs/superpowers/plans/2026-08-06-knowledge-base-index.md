# Knowledge Base Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add task-oriented references to the `docs/` knowledge base across the four golden docs (`AGENTS.md`, `ARCHITECTURE.md`, `README.md`, `CONTRIBUTING.md`) so future coding agents can find the right domain knowledge fast.

**Architecture:** Pure documentation change. `docs/DOCS_README.md` stays the single master map (tiers, file map, reading recipes — SSOT); the golden docs add *pointers* (relative links + `§N` text references), never copies. AGENTS.md gets a task→knowledge table; ARCHITECTURE.md gets in-place canonical-source pointers on its formula sections; README.md and CONTRIBUTING.md get light enrichment.

**Tech Stack:** Markdown, relative links. No code, no tests. Verification = byte budget, link resolution, duplication grep, `pytest -q` regression guard.

## Global Constraints

- **SSOT / no duplication:** never copy a table, formula, or command from `docs/` into a golden doc — links and `§N` references only.
- **AGENTS.md hard budget:** must stay ≤ 32 768 bytes after Task 1 (currently ~17.7 KB).
- **Links:** every added relative path must exist; use file-level links (`docs/01-canon/staking.md`) and text `§N` references. Do NOT invent `#anchor` suffixes unless the exact heading text is verified.
- **Scope:** only the four golden docs may change (plus this plan/spec in `docs/superpowers/`). No edits to `docs/DOCS_README.md` or any `docs/` file.
- **Git:** do NOT run `git commit` unless the user explicitly confirms. Each task's commit step requires asking the user first.
- **Verification gate:** `pytest -q` must stay green (docs-only change; run once in Task 5 as a regression guard).

---

### Task 0: Rename `docs/README.md` → `docs/DOCS_README.md` (done 2026-08-06)

**Files:**
- Rename: `docs/README.md` → `docs/DOCS_README.md` (avoids confusion with the top-level `README.md`)
- Modify: `AGENTS.md` (§4 + §6 toolkit row), `README.md` (tree comment + knowledge-base header), `docs/06-evaluation/evaluation-suite-bible.md` (`../README.md` ×2), and this plan/spec — replace every `docs/README.md` / `../README.md` reference with the new name
- `CONTRIBUTING.md` is handled by Task 4's single step-3 replacement

- [x] **Step 1: Rename the file**

```bash
mv docs/README.md docs/DOCS_README.md
```

- [x] **Step 2: Update all references**

Replace `docs/README.md` → `docs/DOCS_README.md` in `AGENTS.md`, `README.md`, this plan, and the design spec; replace `../README.md` → `../DOCS_README.md` in `docs/06-evaluation/evaluation-suite-bible.md` (×2).

- [x] **Step 3: Verify no stale references remain**

Run: `grep -rn "docs/README\.md\|\.\./README\.md" AGENTS.md README.md CONTRIBUTING.md ARCHITECTURE.md docs/06-evaluation/evaluation-suite-bible.md`
Expected: no matches.

---

### Task 1: AGENTS.md — add "Knowledge base map (docs/)" section

**Files:**
- Modify: `AGENTS.md` (insert new subsection at the end of §6 Agent Toolkit, between the toolkit table and the `<verification_gates>` block)

**Interfaces:**
- Consumes: the §6 Agent Toolkit table's final row (`| Understand how models are judged | \`docs/06-evaluation/evaluation-suite-bible.md\` (evaluation spec of record) |`).
- Produces: a task→knowledge table that Tasks 2–4 link into; also referenced by the coverage matrix in Task 5.

- [ ] **Step 1: Read AGENTS.md and locate the insertion point**

Run: `Read AGENTS.md`. Find the line:
```
| Understand how models are judged | `docs/06-evaluation/evaluation-suite-bible.md` (evaluation spec of record) |
```
Confirm the following lines are:
```
(blank line)
---
(blank line)
<verification_gates>
```

- [ ] **Step 2: Apply the edit**

Use `Edit` on `AGENTS.md`. `old_string`:
```
| Understand how models are judged | `docs/06-evaluation/evaluation-suite-bible.md` (evaluation spec of record) |

---

<verification_gates>
```
`new_string`:
```
| Understand how models are judged | `docs/06-evaluation/evaluation-suite-bible.md` (evaluation spec of record) |

### Knowledge base map (docs/)

The `docs/` tree is a curated Numerai domain library; `docs/DOCS_README.md` is its master map (importance tiers, per-file table, reading recipes). Task-oriented pointers into it:

| When you... | Read first |
|---|---|
| Touch CORR / MMC / FNC / BMC metric formulas | `docs/01-canon/scoring/00-definitions.md` → `docs/01-canon/scoring/01-correlation.md` / `02-mmc-bmc.md` / `03-fnc.md` |
| Change neutralization | `docs/01-canon/models.md` (official `neutralize()` code) + `docs/05-notebooks/2_feature_neutralization.ipynb` |
| Change ensembling | `docs/02-strategy/target-ensembling-math.md` + `docs/05-notebooks/3_target_ensemble.ipynb` |
| Change the payout proxy | `docs/01-canon/staking.md` (0.75/2.25 weights, ±5% clip, stake thresholds) |
| Change model presets / params | `docs/01-canon/models.md` (benchmark walk-forward: 8-era purge for 20D, 16 for 60D; standard/deep params) |
| Touch submission or deployment | `docs/01-canon/submissions.md` + `docs/02-strategy/strategy-bible.md` §8 (deployment contract) |
| Change benchmark gates | `docs/06-evaluation/benchmark-line-in-the-sand.md` (null floor + S11 ladder) |
| Change evaluation semantics | `docs/06-evaluation/evaluation-suite-bible.md` (evaluation spec of record) |
| Use `numerapi` / `numerai_tools` | `docs/03-reference/numerapi.md` + `docs/03-reference/numerai-tools.md` |
| Plan research work | `docs/04-research/research-program.md`, `docs/04-research/advanced-ideas.md`, `docs/04-research/neural-networks.md` |
| Seek domain intuition | `docs/02-strategy/strategy-bible.md` + `docs/02-strategy/why-it-works.md` |

Start with the agent reading order in `docs/DOCS_README.md` §1; the 15-minute version is §2–§3.

---

<verification_gates>
```

- [ ] **Step 3: Verify byte budget and rendering**

Run: `wc -c AGENTS.md`
Expected: a value ≤ 32768 (should be ~19–20 KB).

- [ ] **Step 4: Verify the new links resolve**

Run:
```bash
ls docs/01-canon/scoring/00-definitions.md docs/01-canon/scoring/01-correlation.md docs/01-canon/scoring/02-mmc-bmc.md docs/01-canon/scoring/03-fnc.md docs/01-canon/models.md docs/01-canon/staking.md docs/01-canon/submissions.md docs/02-strategy/strategy-bible.md docs/02-strategy/target-ensembling-math.md docs/02-strategy/why-it-works.md docs/03-reference/numerapi.md docs/03-reference/numerai-tools.md docs/04-research/research-program.md docs/04-research/advanced-ideas.md docs/04-research/neural-networks.md docs/05-notebooks/2_feature_neutralization.ipynb docs/05-notebooks/3_target_ensemble.ipynb docs/06-evaluation/benchmark-line-in-the-sand.md docs/06-evaluation/evaluation-suite-bible.md docs/DOCS_README.md
```
Expected: all listed files exist (no `ls: cannot access` lines).

- [ ] **Step 5: Commit (requires user confirmation)**

```bash
git add AGENTS.md
git commit -m "docs: add task-oriented knowledge base map to AGENTS.md"
```
Ask the user for explicit confirmation before running these commands.

---

### Task 2: ARCHITECTURE.md — add canonical-source pointers

**Files:**
- Modify: `ARCHITECTURE.md` — one short line at the end of each of these sections: §2C (line ~96), §2D (line ~110), §2E (line ~126), §2G (line ~154), §2J (line ~189), §2K (line ~197), §2M (line ~217)

**Interfaces:**
- Consumes: existing section prose (read the file first to anchor each edit).
- Produces: in-place pointers mapping each formula section to its canonical `docs/` source; used by the coverage check in Task 5.

- [ ] **Step 1: Read ARCHITECTURE.md**

Run: `Read ARCHITECTURE.md` (lines 90–220) and locate the exact ending lines of sections §2C, §2D, §2E, §2G, §2J, §2K, §2M.

- [ ] **Step 2: Apply the seven edits (one `Edit` per section, sequential — re-read before each)**

For each edit below, use `Edit` with the given `old_string` → `new_string`. The `old_string` is the final line of that section's prose (verify against the file; if the file differs, anchor on the section's last unique sentence).

1. **§2C** — after the `embargo_eras` sentence. `old_string`:
```
`embargo_eras` is accepted but **structurally inert** (see [`AGENTS.md`](AGENTS.md#8-critical-operational-hazards)).
```
`new_string`:
```
`embargo_eras` is accepted but **structurally inert** (see [`AGENTS.md`](AGENTS.md#8-critical-operational-hazards)). Purge/embargo convention (8/16 operational vs 4/16 minimum): [docs/DOCS_README.md](docs/DOCS_README.md) §3; official benchmark walk-forward table (156-era blocks): [docs/01-canon/models.md](docs/01-canon/models.md).
```

2. **§2D** — after the transforms table. `old_string`:
```
| `power_1_5(v)` | `sign(v) · |v|^1.5` |
```
`new_string`:
```
| `power_1_5(v)` | `sign(v) · |v|^1.5` |

All transforms follow the canonical definitions in [docs/01-canon/scoring/00-definitions.md](docs/01-canon/scoring/00-definitions.md).
```

3. **§2E** — after the `summarize` sentence. `old_string`:
```
`summarize(per_era) -> MetricSummary(mean, std, sharpe, max_drawdown)` — std ddof=0, `sharpe = mean/std` (0 if std=0), drawdown on cumulative sum. Degenerate eras (<2 rows, zero variance, non-finite) short-circuit to score 0.0 after `clean_frame()` null/finite filtering.
```
`new_string`:
```
`summarize(per_era) -> MetricSummary(mean, std, sharpe, max_drawdown)` — std ddof=0, `sharpe = mean/std` (0 if std=0), drawdown on cumulative sum. Degenerate eras (<2 rows, zero variance, non-finite) short-circuit to score 0.0 after `clean_frame()` null/finite filtering.

Metric definitions follow the canonical tournament spec: CORR → [docs/01-canon/scoring/01-correlation.md](docs/01-canon/scoring/01-correlation.md); MMC/BMC → [docs/01-canon/scoring/02-mmc-bmc.md](docs/01-canon/scoring/02-mmc-bmc.md); FNC → [docs/01-canon/scoring/03-fnc.md](docs/01-canon/scoring/03-fnc.md). The repo's judging rules are the evaluation spec of record: [docs/06-evaluation/evaluation-suite-bible.md](docs/06-evaluation/evaluation-suite-bible.md).
```

4. **§2G** — after the preset/backend paragraph. `old_string`:
```
LightGBM adds `objective="regression"`, `random_state=seed`, `n_jobs=1`, `deterministic=True`, `force_col_wise=True`. XGBoost translates `num_leaves→max_leaves`, `min_data_in_leaf→min_child_weight`, adds `reg:squarederror` + `seed`. `ModelConfig.params` overrides presets key-by-key.
```
`new_string`:
```
LightGBM adds `objective="regression"`, `random_state=seed`, `n_jobs=1`, `deterministic=True`, `force_col_wise=True`. XGBoost translates `num_leaves→max_leaves`, `min_data_in_leaf→min_child_weight`, adds `reg:squarederror` + `seed`. `ModelConfig.params` overrides presets key-by-key.

Presets mirror Numerai's published benchmark params and walk-forward purge convention in [docs/01-canon/models.md](docs/01-canon/models.md).
```

5. **§2J** — after the diagnostics sentence. `old_string`:
```
`payout_report(...) -> PayoutResult` bundles all of these plus `BootstrapCI` on mean payout, deflated Sharpe, and AC-adjusted MMC Sharpe.
```
`new_string`:
```
`payout_report(...) -> PayoutResult` bundles all of these plus `BootstrapCI` on mean payout, deflated Sharpe, and AC-adjusted MMC Sharpe.

Payout weights, ±5% clip, and stake thresholds follow [docs/01-canon/staking.md](docs/01-canon/staking.md).
```

6. **§2K** — after the `to_frame()` sentence. `old_string`:
```
**Timing columns are excluded from canonical hashing** (§M).
```
`new_string`:
```
**Timing columns are excluded from canonical hashing** (§M).

The evaluation spec of record (metrics, gates, build slices E1–E6) is [docs/06-evaluation/evaluation-suite-bible.md](docs/06-evaluation/evaluation-suite-bible.md).
```

7. **§2M** — after the Output bullet. `old_string`:
```
- **Output** — `scorecards_to_frame` / `write_scorecards_csv` (column inventory = `MetricScorecard.to_frame()` §K).
```
`new_string`:
```
- **Output** — `scorecards_to_frame` / `write_scorecards_csv` (column inventory = `MetricScorecard.to_frame()` §K).

Benchmark ladder rationale (null floor, S11 rungs, hard gates): [docs/06-evaluation/benchmark-line-in-the-sand.md](docs/06-evaluation/benchmark-line-in-the-sand.md).
```

- [ ] **Step 3: Verify all seven edits landed**

Run:
```bash
grep -n "docs/01-canon\|docs/06-evaluation\|docs/README" ARCHITECTURE.md
```
Expected: at least 9 matching lines (transforms, evaluation ×2, presets, payout, scorecard, benchmark, splitter ×2).

- [ ] **Step 4: Commit (requires user confirmation)**

```bash
git add ARCHITECTURE.md
git commit -m "docs: link ARCHITECTURE formulas to canonical docs sources"
```
Ask the user for explicit confirmation before running these commands.

---

### Task 3: README.md — enrich "Domain knowledge base"

**Files:**
- Modify: `README.md` lines 152–161 (the `### Domain knowledge base` section)

**Interfaces:**
- Consumes: existing "Domain knowledge base" bullets.
- Produces: richer directory summaries + explicit SSOT pointer to `docs/DOCS_README.md`; used by the coverage check in Task 5.

- [ ] **Step 1: Read README.md**

Run: `Read README.md` (lines 144–166) to anchor the exact section text.

- [ ] **Step 2: Apply the edit**

`old_string`:
```
- [docs/01-canon/](docs/01-canon/overview.md) — canonical tournament truth: data, scoring (CORR/MMC/BMC/FNC), submissions, staking
- [docs/02-strategy/](docs/02-strategy/strategy-bible.md) — strategy bible, community wisdom, target-ensembling math
- [docs/03-reference/](docs/03-reference/numerai-tools.md) — `numerapi` and `numerai-tools` API references
- [docs/04-research/](docs/04-research/research-program.md) — research program, advanced ideas
- [docs/05-notebooks/](docs/05-notebooks/) — onboarding notebooks (hello-numerai, neutralization, target ensembles)
- [docs/06-evaluation/](docs/06-evaluation/evaluation-suite-bible.md) — **the evaluation spec of record**: how this repo judges a model
```
`new_string`:
```
- [docs/01-canon/](docs/01-canon/overview.md) — canonical tournament truth: data, scoring (CORR/MMC/BMC/FNC), submissions, staking
- [docs/02-strategy/](docs/02-strategy/strategy-bible.md) — strategy bible, community wisdom, target-ensembling math
- [docs/03-reference/](docs/03-reference/numerai-tools.md) — `numerapi` and `numerai-tools` API references
- [docs/04-research/](docs/04-research/research-program.md) — research program, advanced ideas, neural-network / tabular-DL notes
- [docs/05-notebooks/](docs/05-notebooks/) — onboarding notebooks (hello-numerai, neutralization, target ensembles, sunshine example)
- [docs/06-evaluation/](docs/06-evaluation/evaluation-suite-bible.md) — **the evaluation spec of record**: how this repo judges a model; the benchmark null-floor / S11 ladder is [benchmark-line-in-the-sand.md](docs/06-evaluation/benchmark-line-in-the-sand.md)
- [docs/99-archive/](docs/99-archive/) — archived, low-priority reference (bounties, general ML cookbook, grandmaster seasons); raw source originals preserved unmodified under `docs/99-archive/raw-source/`

The authoritative map — importance tiers, per-file table, and reading recipes — lives in [docs/DOCS_README.md](docs/DOCS_README.md); the bullets above are a directory summary, not the map itself.
```

- [ ] **Step 3: Verify links resolve**

Run:
```bash
ls docs/01-canon/overview.md docs/02-strategy/strategy-bible.md docs/03-reference/numerai-tools.md docs/04-research/research-program.md docs/05-notebooks/ docs/06-evaluation/evaluation-suite-bible.md docs/06-evaluation/benchmark-line-in-the-sand.md docs/99-archive/ docs/99-archive/raw-source/ docs/DOCS_README.md
```
Expected: all paths exist.

- [ ] **Step 4: Commit (requires user confirmation)**

```bash
git add README.md
git commit -m "docs: expand README domain knowledge base summary"
```
Ask the user for explicit confirmation before running these commands.

---

### Task 4: CONTRIBUTING.md — enrich "Before you start" step 3

**Files:**
- Modify: `CONTRIBUTING.md` line 15 (step 3 under "Before you start")

**Interfaces:**
- Consumes: existing step-3 sentence.
- Produces: a domain reading path for human contributors; used by the coverage check in Task 5.

- [ ] **Step 1: Read CONTRIBUTING.md**

Run: `Read CONTRIBUTING.md` (lines 11–25) to anchor the exact step-3 text.

- [ ] **Step 2: Apply the edit**

`old_string`:
```
3. Skim [docs/DOCS_README.md](docs/DOCS_README.md) — the Numerai domain knowledge base (canonical laws, scoring, purge/embargo conventions).
```
`new_string`:
```
3. Skim [docs/DOCS_README.md](docs/DOCS_README.md) — the Numerai domain knowledge base (canonical laws, scoring, purge/embargo conventions, tiered reading paths). For domain intuition, [docs/02-strategy/strategy-bible.md](docs/02-strategy/strategy-bible.md) and the [docs/05-notebooks/](docs/05-notebooks/) tutorials; before touching any metric or evaluation code, read [docs/06-evaluation/evaluation-suite-bible.md](docs/06-evaluation/evaluation-suite-bible.md) — the evaluation spec of record.
```

- [ ] **Step 3: Verify links resolve**

Run:
```bash
ls docs/DOCS_README.md docs/02-strategy/strategy-bible.md docs/05-notebooks/ docs/06-evaluation/evaluation-suite-bible.md
```
Expected: all paths exist.

- [ ] **Step 4: Commit (requires user confirmation)**

```bash
git add CONTRIBUTING.md
git commit -m "docs: add domain reading path to CONTRIBUTING setup steps"
```
Ask the user for explicit confirmation before running these commands.

---

### Task 5: Cross-document verification

**Files:**
- None (verification only).

**Interfaces:**
- Consumes: the outputs of Tasks 1–4.
- Produces: evidence the change is complete, SSOT-clean, and non-breaking.

- [ ] **Step 1: AGENTS.md byte budget**

Run: `wc -c AGENTS.md`
Expected: ≤ 32768.

- [ ] **Step 2: Coverage matrix — every docs/ file reachable**

Run:
```bash
grep -l "docs/01-canon" AGENTS.md ARCHITECTURE.md README.md CONTRIBUTING.md
grep -l "docs/02-strategy" AGENTS.md ARCHITECTURE.md README.md CONTRIBUTING.md
grep -l "docs/03-reference" AGENTS.md ARCHITECTURE.md README.md CONTRIBUTING.md
grep -l "docs/04-research" AGENTS.md ARCHITECTURE.md README.md CONTRIBUTING.md
grep -l "docs/05-notebooks" AGENTS.md ARCHITECTURE.md README.md CONTRIBUTING.md
grep -l "docs/06-evaluation" AGENTS.md ARCHITECTURE.md README.md CONTRIBUTING.md
grep -l "docs/99-archive" AGENTS.md ARCHITECTURE.md README.md CONTRIBUTING.md
```
Expected: each grep returns ≥ 1 file. (Note: `docs/04-research` also covers the two long-named research files via directory reference in the AGENTS.md research row; `docs/99-archive/raw-source` provenance is covered by the README raw-source note and `docs/DOCS_README.md` §6.)

- [ ] **Step 3: No duplication of master-map content**

Run:
```bash
grep -rn "T0" README.md CONTRIBUTING.md ARCHITECTURE.md AGENTS.md | grep -i "tier" | head
```
Expected: no `T0–T4` tier table copied into the golden docs (the tiered map stays in `docs/DOCS_README.md` only). Also verify no payout formula (`0.75`) or preset table was copied: `grep -c "0.75" AGENTS.md README.md CONTRIBUTING.md` — matches only where the value already existed before this change (AGENTS.md may legitimately contain none; do not add any new occurrences beyond existing ones).

- [ ] **Step 4: git diff scope**

Run: `git status --short`
Expected: only `AGENTS.md`, `ARCHITECTURE.md`, `README.md`, `CONTRIBUTING.md` modified, plus the untracked `docs/superpowers/specs/2026-08-06-knowledge-base-index-design.md` and `docs/superpowers/plans/2026-08-06-knowledge-base-index.md`. Nothing else.

- [ ] **Step 5: Regression guard**

Run: `.\.venv\Scripts\python -m pytest -q` (from repo root)
Expected: 301 passed (0 failures) — docs-only change, this is a no-op guard.

- [ ] **Step 6: Commit (requires user confirmation)**

```bash
git add AGENTS.md ARCHITECTURE.md README.md CONTRIBUTING.md
git commit -m "docs: add knowledge-base index across golden docs"
```
Ask the user for explicit confirmation before running these commands.
