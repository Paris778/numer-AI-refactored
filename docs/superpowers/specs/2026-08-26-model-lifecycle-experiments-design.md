# Design Spec: Model Lifecycle & Self-Contained Experiment Layout

> Status: APPROVED (user disposition 2026-08-26). Replaces the three global per-model pools (`artifacts/runs/`, `artifacts/registry/`, `artifacts/models/`) with self-contained per-experiment directories at repo root `experiments/`, introduces a human display-name naming layer over the machine slug, and defines the four-stage model lifecycle (research → partial → full → staked) with a new train-only (partial) promotion scope and a local cross-check scoring step.
## 1. Mission

Two problems motivate this change:

1. **Illegibility.** A model is known by `brb1-xgb-v6 · 2610a99d` — a cryptic slug plus a 64-hex hash, with three near-identical runs per family differing only by hash. There is no human-facing name, no visible statement of how far a model has progressed (research-only? promoted? deployed? earning?), and no per-experiment record of what was done and why.
2. **Fragility + scatter.** Per-model state is spread across three global pools with no versioning and no per-experiment documentation. The 2026-08-26 registry-loss incident (a junction/worktree deletion followed into `artifacts/registry/` and wiped all 29 run dirs — not in git, not recoverable) proved the cost. The registry was already empty at that point; this spec rebuilds the layout so the small, human-relevant record is git-versioned and the per-experiment story is browsable in one place.

The change rides the clean slate: no legacy runs to migrate — the new layout is the baseline.

## 2. Decisions

| # | Question | Decision |
|---|---|---|
| 1 | Experiment home | **Repo root `experiments/<family-slug>/`.** Browsable beside `configs/`/`docs/`; the small metadata record is committed to git (decision 9). Slug matches the existing family rule `^[a-z0-9_-]+$`. |
| 2 | Layout | Fixed per-experiment structure (see §2.1). All run/export output lives under it; nothing per-model lives in `artifacts/` anymore. |
| 3 | Pool absorption | `artifacts/runs/<run_id>/` → `experiments/<slug>/runs/<run_id>/` (incl. `oof_checkpoints/`, `deploy_checkpoints/`, `validation_checkpoints/`); `artifacts/registry/<run_id>/run.json` → `experiments/<slug>/runs/<run_id>/run.json`; `artifacts/models/<family>/full/<run_id>/` → `experiments/<slug>/exports/full/<run_id>/` (new `exports/partial/` for the train-only scope). Global `champion.json` → `experiments/champion.json` (same `{run_id, promoted_at}` schema). `nmr/registry.py`, `nmr/runner.py`, `nmr/promote.py`, `nmr/families.py`, `nmr/dashboard.py`, `nmr/meta.py` and the CLIs retarget accordingly. |
| 4 | Naming | **Display name layered over machine slug.** Slug = experiment dir + `run.name`, template `<theme>-<backend>-<vN>` going forward (e.g. `ender-xgb-v1`). Human `display_name` (e.g. `"Ender XGB v1 · medium"`) lives in `experiments/<slug>/meta.json`, surfaces in dashboard labels, manifests, and docs; the hash appears only in tooltips/URIs. Lifecycle is a **badge** (`· partial` / `· full` / `· staked`), never a name mutation — one family can hold both partial and full exports. `display_name` is deliberately **not** a run-config field: run-id determinism (config + data fingerprint hashes) is untouched. |
| 5 | Lifecycle stages | Four stages, **derived from filesystem state** (never a mutable counter): `research` (runs exist, no exports) → `partial` (a valid `exports/partial/<run_id>/manifest.json` exists) → `full` (a valid `exports/full/<run_id>/manifest.json` exists AND `exports/full/current.json` points at a valid slot) → `staked` (manual `meta.json` field — external platform state cannot be derived). Badge precedence: `staked` > `full` > `partial` > `research`. |
| 6 | Partial promotion scope | `promote_full_version` gains `scope: "full" \| "train_only"` (single writer, one code path; the scope controls fit data, target dir, and pointer handling). `train_only`: fits on `train.parquet` only (no `validation.parquet` read — `_full_history_frame`'s validation requirement is scoped out), publishes to `exports/partial/<run_id>/`, never touches `current.json`. Manifest records `training_scope: "train_only"` + existing provenance (`training_rows`, `training_era_range`). Tier-4 gate applies to both scopes (same `override_gate` path, `tier4_gate_passed` recorded). RAM guard applies to both scopes (train-only ≈ full in memory). |
| 7 | Local cross-check scoring | New step after a partial promotion: run the published artifact's predict on `validation.parquet`, score era-grouped with `numerai_tools` (corr, mmc, fnc, sharpe), write `exports/partial/<run_id>/scorecard.json` — the reference the user compares against Numerai platform diagnostics (same artifact, same eras: local vs platform). |
| 8 | Dashboard lifecycle | Unified schema gains `display_name` + `lifecycle_stage`; rows show the lifecycle badge; the existing `family::full` detail rows are joined by `family::partial`; the payload carries the two new fields through the vanilla renderer. |
| 9 | Git versioning | **Commit the small record:** `README.md`, `config.yaml`, `meta.json`, `runs/<run_id>/run.json`, all `manifest.json`, `scorecard.json`. **Ignore the heavy:** `validation_preds.parquet`, `predict.pkl`, checkpoint trees. `.gitignore` gains `experiments/` rules mirroring the `artifacts/` pattern (keep the tree via `.gitkeep`). This makes the loss class of 2026-08-26 impossible for the record. |
| 10 | Config schema | No new hashed fields. `run.name` unchanged (family slug). `run.artifacts_dir` semantics become the **experiment root** (`experiments/<slug>`); exact derivation (constant vs config) is an implementation detail for the plan. |
| 11 | Documentation | New `docs/02-strategy/model-lifecycle.md` — the lifecycle, the workflow (research → partial cross-check → full → stake), the naming convention, the experiment layout, and "how we operate" — plus same-commit SSOT updates: `AGENTS.md` (toolkit table, operational hazards incl. the junction-deletion hazard), `ARCHITECTURE.md` (artifact/registry schema sections), `CONTRIBUTING.md`, `README.md` (project tree). |

### 2.1 Experiment layout (normative)

```
experiments/<slug>/                          # e.g. ender-xgb-v1
  README.md          # human record: what was done, decisions, results
  config.yaml        # canonical experiment config (copy of the executed config)
  meta.json          # {display_name, staked: {status, numerai_model_id, staked_at}}
  runs/<run_id>/
    run.json                                 # scorecard + provenance (was artifacts/registry)
    validation_preds.parquet
    oof_checkpoints/  deploy_checkpoints/  validation_checkpoints/   # resumed-run state
  exports/
    partial/<run_id>/
      predict.pkl                            # train-only artifact
      manifest.json                          # training_scope: "train_only"
      scorecard.json                         # local validation cross-check reference
    full/<run_id>/
      predict.pkl                            # full-history artifact
      manifest.json                          # training_scope: "full"
    full/current.json                        # active full-version pointer (was models/<family>)
experiments/champion.json                   # global best-run pointer (was registry/champion.json)
```

## 3. Tests

1. **Layout round-trip** (`tests/test_experiment_layout.py`): a synthetic run writes/reads `experiments/<slug>/runs/<run_id>/run.json` + preds; scan discovers runs and derives `lifecycle_stage` at each stage (research → partial → full → staked) incl. badge precedence and the `current.json` requirement for `full`.
2. **Partial promotion scope** (`tests/test_promote.py`): `scope="train_only"` fits on train eras only (assert `training_era_range` = train range, `training_rows` = train rows), writes `exports/partial/<run_id>/`, does not create `current.json`, manifest says `training_scope: "train_only"`; gate + override behavior identical to full.
3. **Cross-check scorecard** (`tests/test_promote.py`): the partial artifact's `scorecard.json` equals a direct `numerai_tools` scoring of the artifact's predictions on the same validation window.
4. **Display-name propagation** (`tests/test_dashboard.py`): `meta.json` display_name reaches the unified payload row + label; lifecycle badge correct per stage.
5. **Determinism**: existing determinism/hash tests stay green — run-id must be unchanged by the layout move (paths excluded from hashes).
6. **Retarget**: every test referencing `artifacts/registry`, `artifacts/runs`, `artifacts/models` (registry, families, promote, dashboard, meta, runner, submission) moves to the `experiments/` layout.

## 4. Files

- Modify: `nmr/registry.py`, `nmr/runner.py`, `nmr/promote.py` (scope param + cross-check scoring), `nmr/families.py` (scan `experiments/`, partial versions), `nmr/dashboard.py` (display_name + lifecycle_stage), `nmr/meta.py`, `nmr/submission.py` (path resolution), CLIs (`promote_model.py`, `rehearse_promotion.py`, `generate_dashboard.py`, `run_campaign.py`), `.gitignore`, `configs/example.yaml` (annotated layout), `configs/benchmarks/tier4_gate.yaml` (only if paths are referenced)
- New: `experiments/` scaffold (`.gitkeep`), `docs/02-strategy/model-lifecycle.md`
- Tests: new `tests/test_experiment_layout.py`; retargeted existing path-dependent tests
- Docs (same commit, SSOT): `AGENTS.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `README.md`

## 5. Out of Scope

- Migrating legacy runs (registry was wiped by the clean slate — by design).
- Renaming existing families (none remain; the convention applies going forward).
- Automating the Numerai upload/stake itself — the upload is a manual act; the stake is recorded in `meta.json` post-hoc.
- Benchmark/fleet/campaign storage (stays in `artifacts/reports/` + `artifacts/campaigns/` — cross-experiment by nature).
- Re-creating the promising models from the 2026-08-26 roster (a separate campaign; the roster snapshot lives in `artifacts/dashboard.html` + `/tmp/model_roster.csv`).
- Changing run-id hashing or metric formulas.
