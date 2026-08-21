# Cold-Start Handoff: Untiered Benchmark Fleet (2026-08-19)

> Purpose: a self-contained continuation document for a fresh LLM agent (or human) resuming the "Untiered Benchmark Fleet" work in `C:/dev/numer-AI-refactored`. Read this first, then the spec and plan it points to.

## 1. Initial Goal

The user asked to extend the repo's benchmark system with a set of benchmark models spanning silly heuristics to competitive community models, recreating:

1. The three Numerai tutorial notebooks (`docs/05-notebooks/1_hello_numerai.ipynb`, `2_feature_neutralization.ipynb`, `3_target_ensemble.ipynb`) — small AND deep versions.
2. All community notebooks under `docs/05-notebooks/community_notebooks/` (3 model scripts: `example_model.py`, `example_model_advanced.py`, `example_model_sunshine.py`) — shallow AND deep versions; v4/v4.1 sources re-fit on the repo's v5.3 data.
3. A comprehensive audit list of baselines (constant 0.5, rolling previous-era target mean, simple ridge) and the Finance Arena model series v0.2, v0.3, v0.4, v0.5, v0.6.0, v1.5.0, v1.5.1 (notebook sources are read-only in the legacy repo `C:/dev/numer-AI/` under `models/version_0/` and `models/version_1/`).

## 2. What Was Decided (Design Rationale)

- **Integration target**: extend the benchmark system (config + `nmr/` code), NOT standalone `ExperimentRunner` registry runs.
- **Tier assignment (user's explicit directive)**: the new models are **untiered** — they live in a new config layer `configs/benchmarks/fleet/` and are scored through the identical evaluation pipeline; their measured performance places them against the existing 5-tier ladder indirectly (report-only `placement` column). Rationale: the hard gates/monotonicity of the 5-tier "line in the sand" stay untouched, and new models can be added without re-semanticizing tiers.
- **Roster**: only the listed models + missing variants. Cells that already existed were NOT duplicated (`null_constant_05`, `linear_ridge_*`, `canon_hello_numerai`). Legacy extras (`finance_arena_v0_1.ipynb`, `simple_lgbm_shallow.ipynb`) excluded.
- **Fidelity policy**: architecture/params/targets/neutralization proportions are faithful to the source notebooks; ALL processing goes through framework machinery — purged trimmed-train fit → predict validation (exact 8-era purge), per-era rank-Gaussianization (`Ensembler`), neutralization ONLY via the oracle-parity `NeutralizationEngine`. Where notebooks conflict with invariants (4-era embargoes, hand-rolled neutralize, multi-seed retraining, CV loops), the framework wins and the deviation is documented per cell.
- **"small/deep" and "shallow/deep" definitions**: shallow/small = notebook default params (2k trees, lr 0.01, depth 5); deep = the notebooks' commented "recommended" params (30k trees / lr 0.001 / depth 10 for tutorials; 20k / 0.001 / 6 / 64 leaves for community scripts).
- **SNNR auxiliary targets**: pinned from the legacy CSV `C:/dev/numer-AI/exploratory_notebooks/outputs/snnr_weights_vs_correlation_v5.2.csv` (17 targets, all present in v5.3; weights pinned into the v1.5.1 config). No runtime SNNR computation.
- **v1.5.1 selection bias**: the search cell's candidate selection uses validation (as the notebook did) — the scorecard row carries `selection_bias: true`; never compare it naively against unbiased cells.
- **v4→v5.3 target mapping** (one flagged assumption): `nomi_v4_60 → target_ender_60`, `jerome_v4_60 → target_ender_60`, `nomi_v4_20 → target_ender_20`, `jerome_v4_20 → target_jeremy_20`; identical-name 20D targets map verbatim. One-line config edits if wrong.
- **Branch**: work was done on `main` with the user's explicit consent (no worktree; data/v5.3 and the shared `.venv` live only in this checkout).

Authoritative documents:
- **Spec (SSOT for semantics)**: `docs/superpowers/specs/2026-08-19-benchmark-fleet-design.md`
- **Plan (SSOT for implementation, tasks 1–13, code-verbatim, synced with all reviewed deviations)**: `docs/superpowers/plans/2026-08-19-benchmark-fleet.md`

## 3. What Was Implemented (All Done and Reviewed)

Workflow used: Superpowers **subagent-driven development** — fresh coder subagent per task, TDD (write failing test → implement → green → commit), independent task reviewer per task (spec compliance + quality verdicts), fix loop, final whole-branch review. Progress ledger (recovery map): `.superpowers/sdd/2026-08-19-benchmark-fleet/progress.md` (git-ignored workspace; also contains per-task deferred minors).

Task → deliverable → commit:

| Task | Deliverable | Commit |
|---|---|---|
| — | Spec + plan | `7bcc1a8`, `d1c93b9` |
| 1 | Gate helpers `tier4_gate_verdict` + `tier_max_corrs`; `assert_tier4_gate`/`assert_hierarchy_monotone`/`gate_report_frame` refactored onto shared rows (hard-gate behavior unchanged) | `8a860aa` |
| 2 | `nmr/benchmark_fleet.py` schema: `FleetCellConfig`, `FleetFileConfig`, `load_fleet_config`, `load_fleet_suite_config`, `VALID_FLEET_*` closed sets | `4c68cef` |
| 3 | `generate_lagged_target_predictions` (trailing-train target mean, leak-safe) | `8885801` |
| 4 | `generate_fleet_lightgbm_predictions` + `_select_riskiest_features` (risk-50 neutralizer selection via `feature_stability_screen`) | `5628029` |
| 5 | `generate_fleet_xgb_predictions` (weighted multi-target rank blend, tail-holdout early stopping via `construct_tree_model(extra_params=...)`) | `70532ce` |
| 6 | `generate_mlp_predictions` (sklearn MLPRegressor, `_standardize_feature_block`, closed param set) | `c6adf53` |
| 7 | `generate_ridge_stack_predictions` fixed mode + `_stack_partitions` (horizon-aware 8/16-era internal purge) | `f147722` |
| 8 | `_ridge_stack_search` (v1.5.1: quality filter, alpha grids, Sharpe pruning, nonneg-ridge + LGBM meta candidates, decorr × neutralization sweeps, validation-based selection) | `641f987` |
| 9 | `BenchmarkFleet` runner, `FleetResult`, `fleet_placement`, `fleet_frame`, `write_fleet_csv`; `canonical_scorecards_bytes(fleet_scorecards=...)` + collision guard | `05832ee` |
| 10 | 4 fleet config YAMLs, 19 cells (`configs/benchmarks/fleet/`) | `2a44f82` |
| 11 | Runner CLI `--fleet-configs` / `--fleet-output` / `--no-fleet` | `74756d2` |
| 12 | SSOT docs (line-in-the-sand fleet section, hierarchy-spec amendment, ARCHITECTURE.md, AGENTS.md) + fixed the 3 pre-existing `test_docs_hygiene` failures | `60a2a16` |
| — | Plan synced with reviewed deviations | `b51fe60` |
| Final review fix | **C1**: xgb/mlp cells silently ignored their `neutralization` config → wired `NeutralizationEngine` through both generators + runner branches + 3 oracle tests | `b584791` |

**Final review verdict**: one Critical found and fixed (`b584791`), scoped re-review clean; residuals parked with rulings in the ledger. Fleet code ends at `b584791`.

**Commits after the fleet work (NOT part of this plan — made outside this session, presumably the user's other work)**: `804f08a` (dataless-CI campaign test stub + `scripts/ci_repro.Dockerfile` + CONTRIBUTING gate note), `0bded5c` (lint fixes for the community example scripts + CI coverage floors re-pinned to measured dataless numbers 90.3/97.8/92.6). HEAD at the time of writing this doc = `59a893e` (this doc's own commit). Note: `AGENTS.md`'s claimed suite count is **988** (from earlier mutation-gate work, commit `cb4ea6d` lineage) — newer than this plan's measured 979; reconcile by re-running the suite before relying on either number.

## 4. Current Verified State (as of handoff)

- `git status` clean; branch `main`.
- Full test suite at `b584791`: **979 passed / 0 failures** (`./.venv/Scripts/python -m pytest -q`).
- `./.venv/Scripts/python -m ruff check .` at HEAD `0bded5c`: **All checks passed** (the 17 pre-existing community-notebook lint errors were fixed by `0bded5c`).
- All per-task reviewer verdicts: spec ✅ + quality approved (minors parked in ledger).
- **NOT verified on real data**: Task 13 (verification gates) is incomplete. The fast-mode smoke was attempted 3 times and never finished (details below). No `artifacts/reports/benchmark_fleet_scorecard.csv` exists yet — the 19 fleet cells have never been scored on real v5.3 data, no placements/gate-verdicts measured, no anchors re-pinned, and the full multi-hour run has never been launched.
- pytest at HEAD `0bded5c` has NOT been re-run (the two post-plan commits may have changed the count). Re-run at resume.

## 5. What Remains (Continuation Protocol)

1. **Verify state**: `git log --oneline -15`, `git status`, then `./.venv/Scripts/python -m pytest -q` (expect green; report the actual count; if `test_docs_test_count_matches_suite` fails, update the stale count claim in AGENTS.md/CONTRIBUTING.md to the measured number).
2. **Real-data smoke** (fast mode): check memory headroom first (see §6.3), then launch exactly ONE runner:
   `nohup ./.venv/Scripts/python -u benchmark_runner.py --fast-mode > artifacts/reports/fleet_smoke_run4.log 2>&1 &`
   Verify it's alive with `wmic process where "name='python.exe'" get ProcessId,CommandLine | grep benchmark_runner` (never trust Git Bash `ps` for Windows processes — see §6.1).
   Expected: hierarchy (13 cells, ~50–60 min — `canon_neutralized_50`'s 657-era neutralization alone takes ~20 min) then the 19 fleet cells (community medium cells do expensive riskiest-50 screens + full neutralization; total smoke ≈ 1–3 h). Success = log ends with "All hard gates passed" + exit 0.
3. **Verify smoke outputs**: `artifacts/reports/benchmark_hierarchy_scorecard_smoke.csv`, `benchmark_gate_report_smoke.csv`, and `benchmark_fleet_scorecard.csv` with **19 data rows**; every fleet row has `placement`, `selection_bias`, and 7 `gate_*` verdict columns; `fa_v151_ridge_ensemble` must show `selection_bias: true`. Spot-check placements make sense vs the tier rungs.
4. **Full fleet run** (multi-hour — tens of CPU-hours; the 20k/30k-tree deep cells dominate):
   `nohup ./.venv/Scripts/python -u benchmark_runner.py > artifacts/reports/fleet_full_run.log 2>&1 &`
   Poll with `tail -f` (markers: `[fleet] <benchmark_id> (kind=...)`). This is the run that produces the real measured scorecards.
5. **Record results**: measured CORR + placement per cell; optionally re-pin fleet `anchors` in the configs (evidence-driven, hierarchy-spec decision #2 procedure) as a follow-up commit.
6. **End-of-session gate**: `./.venv/Scripts/python -m ruff check .` + `./.venv/Scripts/python -m pytest -q` on the final state; report truthfully (skips included — real-data tests skip if `data/v5.3` parquets are missing).
7. **SDD finish**: when everything is green, delete the plan workspace `.superpowers/sdd/2026-08-19-benchmark-fleet/` (git history is the record) and run the `finishing-a-development-branch` skill.

Do NOT re-implement tasks 1–12: they are complete and reviewed (ledger + git log are the evidence).

## 6. Operational Lessons Learned (Read Before Relaunching Anything)

1. **Git Bash `ps` does NOT show Windows python processes.** A "lost" kimi background task can mean the tracker lost it while the process lives on. Always use `wmic process where "name='python.exe'" get ProcessId,CommandLine` (or `tasklist`) to determine whether a runner is actually alive. This mistake once caused TWO concurrent runners on the same log/outputs (and likely contributed to the OOM in §6.3); kill duplicates with `taskkill //PID <pid> //F`.
2. **Use `nohup ... -u` for long jobs** (`-u` = unbuffered log lines). The kimi background-task tracker can drop tasks; nohup keeps the process independent of the session. Give each attempt a NEW log file (`run2.log`, `run3.log`, ...) — `>` truncation from a second process destroys the first's log view (interleaved offsets made a finished-looking log that wasn't).
3. **Memory ceiling is real on this box** (63.7 GiB physical; commit limit measured at 111.7 GiB at one point, 148.8 GiB per AGENTS.md's hazard — page-file growth makes it variable, so measure live before launching). `canon_sunshine_ensemble` (4× medium LightGBM fits) does `to_pandas()` allocations of ~12.8 GB and died once with `pyarrow.lib.ArrowMemoryError: malloc of size 12813964800 failed` when the box was under external memory pressure. Check headroom first:
   `powershell -NoProfile -Command 'Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory,TotalVirtualMemorySize,FreeVirtualMemory | Format-List'`
   (quote with single quotes in Git Bash — `$_` gets mangled otherwise). If free physical is well above ~25 GB and commit headroom above ~30 GB, launch; otherwise wait for an idle machine. Run ONE benchmark job at a time, ever.
4. **The exact 8-era purge gap**: `train_validation_purged_split` requires `min(val_eras) - max(trimmed_train) - 1 == 8` exactly. Synthetic fixtures must derive val eras as `max(train_eras) + 1, +2` (for 8-era purge) — hardcoding caused three fixture bugs during the run.
5. **polars ≥ 1.41 removed `Expr.nulls_last()`/`Expr.desc()`** — use `df.sort(by=[...], descending=[...], nulls_last=[...])` kwargs.
6. **"Config knob that lies" is a Critical bug class in this repo** (see the `embargo_eras` SEV-3 precedent in AGENTS.md): every `FleetCellConfig.neutralization` value MUST be wired through its generator (xgb/mlp were fixed in `b584791` — keep that invariant when adding new kinds).
7. **AGENTS.md size budget**: 32,719 B / 32,768 B as of HEAD `59a893e` → **49 B headroom** — any new AGENTS.md content needs a trim elsewhere first. (An earlier session measured 32,517 B; the file grew via unrelated mutation-gate commits — do not trust stale size figures.)

## 7. Parked Findings (All Ruled Defer — Do Not Block)

From the final review (full list with rulings in the ledger):
- Spec's `BenchmarkFleet.from_config_dir` not implemented; module-level `load_fleet_suite_config` + CLI achieves the intent (interface-shape deviation).
- Search-mode subtests for the quality filter / grid bounds absent (config-driven deterministic code; exercised end-to-end via the runner test + real-data smoke).
- `priority_hints` contains `target_teager_20` which is not among the specialists — verbatim from the legacy notebook (faithful-to-source).
- Ledger minors per task (error-message cosmetics, `fleet_frame` KeyError corner when placements are empty, zero-weights ZeroDivisionError, broad `except TypeError` on `Ridge(positive=True)`, scaler stats include non-finite-target rows, etc.) — triage again only if a future change touches those lines.

## 8. Post-Handoff Developments (2026-08-20, second session)

A parallel session reviewed this state and added the following verified findings + queued work. Treat this section as more recent than everything above:

- **Runner coupling (confirmed)**: `benchmark_runner.py::main()` always runs the full tier hierarchy before the fleet, and the fleet's placement rungs come only from that live run (`rungs = tier_max_corrs(result.scorecards, result.tier_of)`). There is `--no-fleet` but no inverse, and `--fleet-configs` takes a directory only — no cell-level selection. Running even the zero-fit `silly_target_lag_mean` today forces a full hierarchy first.
- **Thread caps (absent)**: `nmr/models.py::_resolved_params` sets `n_jobs: 1` for LightGBM/XGBoost fits, but NOTHING in `nmr/` caps polars' thread pool or OpenMP/BLAS (`POLARS_MAX_THREADS`/`OMP_NUM_THREADS` appear nowhere). Polars reads/joins and scipy linear algebra run across all cores — that is the real CPU contention when the campaign runs. A thread-cap change needs a deliberate design point (env-at-process-start vs config knob).
- **Queued work** (in priority order): (1) checkpointing/resume inside a run — **DONE**: fold-granularity incremental persistence + skip-on-resume with a resumed-run-equals-uninterrupted-run determinism test (`ca6883c` core, `6f17bd4` runner wiring); spec: `docs/superpowers/specs/2026-08-20-oof-checkpoint-resume-design.md`; (2) runner composability (`--only-fleet`, rungs sourced from the last hierarchy scorecard CSV, `--fleet-ids` cell selection); (3) thread caps; (4) the mutmut upstream issue. Fleet cell EXECUTION stays deferred until the campaign frees the CPU.

## 9. Key Files

- Spec: `docs/superpowers/specs/2026-08-19-benchmark-fleet-design.md` (roster tables, decisions log, risks)
- Plan: `docs/superpowers/plans/2026-08-19-benchmark-fleet.md` (13 tasks, code-verbatim)
- Code: `nmr/benchmark_fleet.py` (schema + 5 generators + runner), `nmr/benchmark.py` (gate helpers, canonical bytes), `nmr/models.py` (`construct_tree_model(extra_params=...)`), `benchmark_runner.py` (CLI)
- Configs: `configs/benchmarks/fleet/*.yaml` (19 cells)
- Tests: `tests/test_benchmark_fleet.py` (47 tests), `tests/test_benchmark_gates.py`, `tests/test_benchmark_hierarchy.py`
- Ledger: `.superpowers/sdd/2026-08-19-benchmark-fleet/progress.md`
- Smoke logs so far: `artifacts/reports/fleet_smoke_run.log` (interleaved, ignore), `fleet_smoke_run2.log` (OOM evidence), `fleet_smoke_run3.log` (killed mid-tier-1)
- Legacy sources (read-only, never modify): `C:/dev/numer-AI/models/version_0/v0.*/finance_arena_v0*.ipynb`, `.../version_1/v1.5/fa_v1.5.*.ipynb`, `C:/dev/numer-AI/exploratory_notebooks/outputs/snnr_weights_vs_correlation_v5.2.csv`

## 10. Environment Quick Facts

- Windows, Git Bash at `C:\Program Files\Git\bin\bash.exe`; repo root `C:/dev/numer-AI-refactored`; branch `main`.
- Python: `./.venv/Scripts/python` (NEVER `./.venv/Scripts/pip` — the shim points into the legacy repo).
- Data: `data/v5.3/` (train/validation/meta_model/benchmarks parquets + features.json). Real-data tests and the smoke need it.
- Commands: `./.venv/Scripts/python -m pytest -q`, `./.venv/Scripts/python -m ruff check .`.
