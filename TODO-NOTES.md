# TODO / Session Notes (LLM agent state map)

> Record for future LLM agents. Last updated: 2026-08-13 (golden report finalised).
> **The authoritative record is now `docs/04-research/pre-modelling-dataset-feature-study-2026-08.md`** — the
> golden pre-modelling document (single source of truth). Read it before starting any
> model-design work. This file is a short state map only.

## Current state

- **DONE — Benchmark & Evidence Rebuild (v5.3):** 12-cell feature campaign (6 subsets × 2
  backends) on the corrected 649-era validation window (583..1231). 10/12 cells recorded;
  `lgbm_v1`/`xgb_v1` (full 3,555 features) are **hardware-infeasible** on this 63.7 GiB
  machine — final, do not retry (failure modes: §8 of the report).
- **DONE — Evidence:** `artifacts/reports/dataset_analysis/campaign_{variants,pairwise}.parquet`
  (full-window headline metrics, 636 eras; scorecard 86-era kept as secondary columns).
  Do not delete — regeneration is a ~30-50 min `campaign_evidence` run.
- **DONE — Report:** `docs/04-research/pre-modelling-dataset-feature-study-2026-08.md` rendered with §0 exec
  summary, §7 campaign tables, §8 decision log, §9 methodology, §10 file/artifact map.
  Render command in §10 / DOCS_README.
- **DONE — Docs pointers:** AGENTS.md knowledge-map row + DOCS_README T1 row updated.
- **DONE — Tests:** full suite green, 580 collected (docs-hygiene guard enforces the count).
- **OPEN — Commit:** the whole changeset is uncommitted. No git mutations without explicit
  user instruction. When approved: `git add` modified files + new configs/docs +
  `TODO-NOTES.md`, one descriptive commit.
- **OPEN — Screen defaults:** the v3-vs-v2 gate could not fire (v2≡v3≡v4 structurally).
  Screen defaults unchanged; revisiting requires explicit user go-ahead (evidence: report §7.2/§8).

## Standing flags / rules

- No git mutations without explicit user instruction.
- `./.venv/Scripts/pip` is a shim into legacy `../numer-AI/.venv` — always `./.venv/Scripts/python -m pip`.
- Long jobs: `nohup ... > log 2>&1 &`; poll logs. Test count: 580.
- GPU: `model.device` (`auto|gpu|cpu`); xgboost 3.x needs `device="cuda"` + `tree_method="hist"`.
- Never run two full-universe (3,555-feature) jobs concurrently; keep the machine idle for
  full-universe fits (63.7 GiB RAM ceiling — see AGENTS.md hazards).
- Registry holds 21 runs (11 old + 10 new campaign cells); champion is empty by design
  (purged 2026-08-10, no promotion without human decision).

---------------------------------------

Add numerAI MCP for agents 

Purge old benchamrk results 

Do new benchmark models (simple and trivial) 

Update libraries (lgbm etc) 

For a personal project, this is above-average and very respectable; for a team or production system, it still has too much research-code entropy and too little platform discipline. (maybe give to a powerful model to restructure etc, do better modules etc.)


Deliberately deferred — consciously ignored

   • test_analysis.py monolith and duplicated MetricScorecard builders — works fine; split only when editing that subsystem.
     Churn without functional gain.
   • Private cross-module imports (meta.py:480 et al.) — promote during the next features.py refactor, not standalone.
   • Dead placeholders (book_correlation, redundancy metrics) — E6-deferred by plan; add TODO markers if touched.
   • Duplicate _train_multi_target_oof — consolidate when one of the two copies next changes.
   • _gpu axis semantics, opt.py broad except, era_col plumbing — edge or deliberate today; revisit on first contact, not
     proactively.
   • Warning volume, O(n²) set rebuild, open_browser default — cosmetic; fold into future edits.
   • evaluation-bible v5.2 facts, canon target-name staleness, skill command nits, docs/superpowers mapping — batch into the
     next docs-hygiene pass.