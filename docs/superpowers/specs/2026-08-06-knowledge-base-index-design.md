# Design: Knowledge Base Index across the Four Golden Docs

- **Date:** 2026-08-06
- **Status:** Approved by user (via brainstorming skill)
- **Scope:** Documentation only — no code, no test changes.

## 1. Context & Goal

`docs/` contains a curated Numerai knowledge base (~40 files) spanning canonical tournament rules, strategy, library references, research notes, teaching notebooks, and the evaluation spec of record. `docs/DOCS_README.md` is already the master map (tiers T0–T4, full file map, reading recipes, purge/embargo conventions).

**Problem:** The four top-level "golden bible" docs (`AGENTS.md`, `ARCHITECTURE.md`, `README.md`, `CONTRIBUTING.md`) barely reference this knowledge base. `AGENTS.md` points only at `docs/DOCS_README.md` and `docs/06-evaluation/evaluation-suite-bible.md`; `ARCHITECTURE.md` references none of it (formulas sit without canonical sources); `CONTRIBUTING.md` has one line; `README.md` has a short per-directory list.

**Goal:** Add references for the contents of every `docs/` file across the four golden docs, chosen per doc-hygiene (SSOT) so future coding agents get a task-oriented map into the knowledge base, without duplicating `docs/DOCS_README.md`.

## 2. SSOT Principles (non-negotiable)

- `docs/DOCS_README.md` **owns** the master file map, tiers, and reading paths. The golden docs never reproduce that map — they cross-reference it and add task orientation.
- One fact, one home: no formula, table, or convention from `docs/` may be copied into the golden docs. All additions are links (relative markdown links + section anchors).
- `AGENTS.md` has a hard ≤ 32 KB budget (currently ~17.7 KB).
- Every file under `docs/` must be reachable from at least one golden doc (coverage guarantee, §5).

## 3. Per-Doc Change Plan

### 3.1 `AGENTS.md` — new "Knowledge Base" section (agent map)

Add a compact task→knowledge table in §6 (Agent Toolkit), extending the existing row "Understand tournament rules & scoring". One new subsection, roughly:

```
### Knowledge base map (docs/)

The master map, tiers, and reading recipes live in docs/DOCS_README.md. Task-oriented pointers:

| When you... | Read first |
|---|---|
| Touch CORR/MMC/FNC/BMC formulas | docs/01-canon/scoring/00-definitions.md → 01-correlation.md / 02-mmc-bmc.md / 03-fnc.md |
| Change neutralization | docs/01-canon/models.md (official neutralize() code) + docs/05-notebooks/2_feature_neutralization.ipynb |
| Change ensembling | docs/02-strategy/target-ensembling-math.md + docs/05-notebooks/3_target_ensemble.ipynb |
| Change payout proxy | docs/01-canon/staking.md (0.75/2.25 weights, ±5% clip, thresholds) |
| Change model presets/params | docs/01-canon/models.md (benchmark walk-forward 8/16 purge, standard/deep params) |
| Touch submission / deployment | docs/01-canon/submissions.md + docs/02-strategy/strategy-bible.md §8 (deployment contract) |
| Change benchmark gates | docs/06-evaluation/benchmark-line-in-the-sand.md |
| Change evaluation semantics | docs/06-evaluation/evaluation-suite-bible.md (spec of record) |
| Use numerapi / numerai_tools | docs/03-reference/numerapi.md + docs/03-reference/numerai-tools.md |
| Plan research | docs/04-research/research-program.md, advanced-ideas.md, neural-networks.md |
| Seek domain intuition | docs/02-strategy/strategy-bible.md + docs/02-strategy/why-it-works.md |
```

Also add one line: the fast-start agent reading order lives in `docs/DOCS_README.md` §1. Keep total AGENTS.md growth < ~2 KB.

### 3.2 `ARCHITECTURE.md` — canonical-source pointers (in place)

Add one short line per formula-bearing section, pointing at the canonical `docs/` source. No formula duplication:

- §2C splitter purge → `docs/DOCS_README.md` §3 (purge/embargo convention) + `docs/01-canon/models.md` (benchmark walk-forward table)
- §2D transforms → `docs/01-canon/scoring/00-definitions.md`
- §2E CORR → `docs/01-canon/scoring/01-correlation.md`; MMC/BMC → `02-mmc-bmc.md`; FNC → `03-fnc.md`
- §2G presets → `docs/01-canon/models.md` (standard/deep params)
- §2J payout → `docs/01-canon/staking.md` (weights/clip/thresholds)
- §2K scorecard → `docs/06-evaluation/evaluation-suite-bible.md` (spec of record)
- §2M benchmark → `docs/06-evaluation/benchmark-line-in-the-sand.md` (S11 ladder / null floor)

### 3.3 `README.md` — enrich "Domain knowledge base"

- Add `docs/06-evaluation/benchmark-line-in-the-sand.md` to the 06-evaluation bullet.
- Add a `docs/99-archive/` bullet (archived/low-priority reference; raw sources preserved unmodified).
- Add an explicit sentence: the full map, tiers, and reading recipes live in `docs/DOCS_README.md` (SSOT).
- Keep the existing per-directory bullets; do not turn this into a full file map.

### 3.4 `CONTRIBUTING.md` — enrich "Before you start" step 3

- Point to `docs/DOCS_README.md` reading paths (already present) and add: skim `docs/02-strategy/strategy-bible.md` + `docs/05-notebooks/` tutorials for domain intuition; read `docs/06-evaluation/evaluation-suite-bible.md` before touching metrics (spec of record).

## 4. Non-Goals

- No restructuring of `docs/` itself; `docs/DOCS_README.md` content stays as-is.
- No new standalone index file.
- No content duplication into golden docs.
- No code or test changes.

## 5. Coverage Matrix (every docs file reachable)

| docs/ area | Referenced from |
|---|---|
| `01-canon/*` (9 files) | AGENTS.md §6 map + ARCHITECTURE.md in-place pointers |
| `02-strategy/*` (4 files) | AGENTS.md §6 map + README.md + CONTRIBUTING.md step 3 |
| `03-reference/*` (2 files) | AGENTS.md §6 map + README.md |
| `04-research/*` (5 files incl. long-named) | AGENTS.md §6 map + README.md |
| `05-notebooks/*` (4 files) | AGENTS.md §6 map + README.md + CONTRIBUTING.md step 3 |
| `06-evaluation/*` (2 files) | AGENTS.md + ARCHITECTURE.md + README.md |
| `99-archive/*` (4 non-raw + raw-source provenance) | README.md archive bullet; raw sources via `docs/DOCS_README.md` §6 provenance |

## 6. Verification (before sign-off)

1. `wc -c AGENTS.md` ≤ 32768 bytes.
2. Grep the four golden docs for any table/command/formula duplicated from `docs/` — none may exist (links only).
3. Every added relative link resolves (file exists; section anchor exists).
4. `git diff --stat` shows exactly the four golden docs modified (plus this spec).
5. `pytest -q` full suite still green (docs-only change, run as regression guard).

## 7. Risks & Notes

- AGENTS.md budget: additions must stay lean; if the table grows past ~1 KB, trim descriptions, don't expand.
- Anchor drift: relative links with `#section` anchors may break if docs/ files are later restructured — keep anchors short and stable (`docs/DOCS_README.md#2-canonical-laws-ground-truth` style).
- The design doc and its sibling plan (`docs/superpowers/plans/`) are process artifacts and are excluded from the golden-doc coverage requirement.
