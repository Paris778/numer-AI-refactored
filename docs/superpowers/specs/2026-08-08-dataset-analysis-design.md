# Dataset Refresh & Analysis — Design Spec

- **Date:** 2026-08-08
- **Status:** Approved (3-section design review) — pending spec review
- **Repo:** `C:/dev/numer-AI-refactored` (`nmr`)
- **Consumers of the deliverable:** a team of Distinguished Principal Engineers **and** LLM agents — the report must be machine-parse-friendly (dense tables, explicit schemas, no chart dependence)

## 1. Context & Goals

The `nmr` framework consumes Numerai v5.2 tournament data (`data/v5.2/*.parquet`). Two gaps exist:

1. **Data staleness.** `data/refresh_data.ipynb` was the only refresh path; it hardcodes the legacy path `C:\dev\numer-AI\data`, predates this repo, and has no integrity checks. `live.parquet` is stale by ~7 weeks (round 1294 vs current).
2. **No dataset intelligence.** There is no comprehensive, reproducible analysis of the data, targets, and features to guide model design.

**Goal 1:** A scriptable, round-aware data refresh (`refresh_data.py` + pure logic in `nmr/refresh.py`) that refreshes `live.parquet` when the tournament round advances, re-downloads weekly-expanding files, and **alerts promptly when a newer data version (v5.3+, v6.x) is available**.

**Goal 2:** A PhD-level, deterministic analysis of the full 2,748-feature universe across train+validation (eras 0001–1208), producing machine-readable dumps (`artifacts/reports/dataset_analysis/`) and an LLM-optimized Markdown report (`docs/04-research/dataset-analysis-YYYY-MM.md`).

**Non-goals:** model training, feature campaigns, promotion decisions, charts in the critical path, notebook deliverable.

## 2. Deliverables & Artifact Layout

| Artifact | Path | Kind |
|---|---|---|
| Refresh logic | `nmr/refresh.py` | tested module |
| Refresh CLI | `refresh_data.py` | thin control plane |
| Analysis logic | `nmr/analysis.py` | tested module |
| Analysis CLI | `analyze_dataset.py` | thin control plane |
| Renderer CLI | `render_dataset_report.py` | thin control plane |
| Data dumps | `artifacts/reports/dataset_analysis/` | machine-generated |
| Report | `docs/04-research/dataset-analysis-YYYY-MM.md` | committed deliverable |
| Spec | `docs/superpowers/specs/2026-08-08-dataset-analysis-design.md` | this file |

Repo invariants honored: all business logic in `nmr/` (tested); root scripts are argument parsing/wiring/printing only; no new third-party dependencies (stdlib + Polars + NumPy/SciPy + existing `numerapi`); canonical outputs carry no wall-clock fields; atomic writes via `nmr/_atomicio.py`.

---

## 3. Workstream A: Data Refresh

### 3.1 `nmr/refresh.py` — pure, tested (no I/O, no numerapi, no polars)

| Symbol | Contract |
|---|---|
| `CURRENT_DATA_VERSION = "v5.2"` | Module constant. Drift-guarded by a test asserting it equals `load_config("configs/first_model.yaml").data.version`; **`pytest.skip` if the config file is absent** (partial checkout). |
| `_parse_version(v: str) -> tuple[int, int]` | Strict regex `^v(\d+)\.(\d+)$` → `(major, minor)` ints. Any other format **raises `ValueError`** (incl. `v5.2.1`, `v5`, `5.2`, `vX.2`). Fail loud — never silently mishandle a version. |
| `detect_newer_version(available: Sequence[str], current: str) -> str \| None` | Parses all; returns the numerically-greatest version **strictly exceeding** `current` (so `v5.10` > `v5.3`, `v6.0` > `v5.2`); `None` if none. A malformed entry in `available` raises. |
| `needs_live_refresh(current_round: int, last_recorded: int \| None, live_exists: bool) -> bool` | `not live_exists or last_recorded is None or last_recorded != current_round`. Handles both stale and ahead-of-remote markers (reconcile on mismatch). |
| `build_era_manifest(era_ranges: Mapping[str, tuple[str \| None, str \| None]], round_id: int, today: str) -> list[dict]` | Pure — era ranges are **inputs**, computed by the script layer. Emits rows with exactly the existing CSV columns `date, dataset, start_era, end_era, round_id` (`round_id` set only for `live`, `None` for train/validation). **Validation rule:** raises `ValueError` if any **non-live** dataset's range is `(None, None)` (empty/zero-era parquet). **Live's `(None, None)` is explicitly valid** — live rounds are unlabeled; the script serializes it as `"X"` (existing CSV convention). Deterministic: same inputs ⇒ same output. |
| `classify_refresh_plan(round_advanced: bool, existing: set[str], live_only: bool) -> dict[str, str]` | Per-file decision (`"refresh"` / `"ensure"` / `"skip"`) from the policy table §3.3 and the three file tuples. |
| `STATIC_FILES`, `LIVE_FRESH_FILES`, `EXPANDING_FILES` | Module-level tuples (no magic values), see §3.3. |

**Exports** (`nmr/__init__.py` `__all__`): `detect_newer_version`, `needs_live_refresh`, `build_era_manifest`, `classify_refresh_plan`. `_parse_version` stays private.

### 3.2 `refresh_data.py` — thin control plane

The only place numerapi and file I/O live:

1. Parse CLI flags (argparse).
2. `NumerAPI()` (public datasets — no credentials). `get_current_round()` returns `int | None` (verified in installed numerapi 2.22.0); **`None` → abort with clear message, exit 1**.
3. `list_datasets()` → version alert (exit matrix §3.4). The script **intersects the policy tuples with the actual version listing** and only downloads files that exist in the API.
4. Downloads: write to `*.part` temp **in the target directory** → `pl.scan_parquet(tmp).collect_schema()` integrity check → read `min`/`max` era → `os.replace` (atomic swap; a truncated download never replaces a good file; on failure delete temp and exit 1).
5. **`features.json` structural check** after download: reuse `nmr/features.resolve_feature_sets(tmp_path)` (raises on malformed/empty `feature_sets`) **and** assert `raw["targets"]` is a non-empty list. Catches Numerai schema drift at the earliest point, before the analysis pipeline crashes downstream.
6. **Stale `*.part` cleanup at startup** — orphaned temp files from a crashed/killed run are deleted before re-downloading (never mistaken for valid files).
7. Rebuild + atomically write `data/numerai_era_data.csv` via `nmr/_atomicio.atomic_write_text`. On any failure before the write, the old CSV is untouched (no partial writes). **CSV serialization matches the legacy format exactly** (verified against the existing file): `round_id` non-`None` → float (`1294.0`), `None` → empty string; live era `None` → `"X"`.
8. Post-refresh summary: per-file status, era ranges, and whether `validation.parquet`'s era range actually expanded vs the previous manifest.

**`numerai_era_data.csv` (refresh ledger)**: records per-dataset last-refresh `date`, `start_era`, `end_era`, and `round_id` (live only). Read back by `refresh_data.py` for round tracking; consumed by the report's provenance section. Schema documented in ARCHITECTURE.md.

### 3.3 File refresh policy (corrected classification)

| File | Trigger |
|---|---|
| `train.parquet`, `train_benchmark_models.parquet`, `features.json` | **Missing only** (truly static) → `STATIC_FILES` |
| `live.parquet`, `live_benchmark_models.parquet`, `live_example_preds.parquet` (+ `.csv` if listed) | **Round advanced or missing** → `LIVE_FRESH_FILES` |
| `validation.parquet`, `validation_benchmark_models.parquet`, `validation_example_preds.parquet` (+ `.csv`), `meta_model.parquet` | **Round advanced (weekly expansion) or missing** → `EXPANDING_FILES` |

`--live-only` skips `EXPANDING_FILES` with a loud staleness warning (validation staleness → `NonVacuityError` risk in scorecard joins per the known operational hazard). Three tuples mirror the three trigger categories exactly.

### 3.4 CLI & exit-code matrix (one behavior per mode)

| Mode | Newer version found | Files need refresh | Action / exit |
|---|---|---|---|
| default | yes | — | `[WARNING] vX.Y available` banner → proceed with v5.2 refresh → **exit 0** |
| default | no | — | normal refresh → exit 0 |
| `--check-only` | yes | — | no writes; **exit 3** |
| `--check-only` | no | yes | no writes; report what would download; **exit 3** |
| `--check-only` | no | no | "everything current"; **exit 0** |
| `--strict` | yes | — | abort before any download; **exit 3** (CI gate) |
| `--dry-run` | yes | — | print plan **including the `[WARNING]` banner**; no writes; exit 0 |
| any network / parse / integrity failure | — | — | loud message; exit 1 |
| `--live-only` | — | — | skip expanding files (staleness warning), else as above |
| `--version vX.Y` | — | — | override target data version (default `CURRENT_DATA_VERSION`) |

### 3.5 Neutralization cache note (documented, not implemented)

`NeutralizationEngine` cache keys are content-addressed (feature set + eras): stale entries are inert, and a data-version change changes the keys, so old entries only waste disk. No explicit invalidation needed. `artifacts/cache/` hygiene is safe per AGENTS.md.

### 3.6 TDD plan — `tests/test_refresh.py` + `tests/test_refresh_script.py`

1. `_parse_version`: `v5.2`→(5,2); `v5.10`→(5,10); malformed set (`v5.2.1`, `v5`, `5.2`, `vX.2`, `""`) → `ValueError`.
2. `detect_newer_version`: `[]`→None; `["v5.2"]`→None; `["v5.3"]`→`v5.3`; **`["v5.10"]`→`v5.10` (multi-digit regression)**; `["v6.0"]`→`v6.0`; mixed lists (`["v4.9","v5.3","v5.2"]`→`v5.3`); malformed entry → raise.
3. `needs_live_refresh` full truth table incl. ahead-of-remote marker.
4. `build_era_manifest`: exact columns/values; live `round_id` set vs None; empty range → `ValueError`; deterministic (two calls identical).
5. `classify_refresh_plan`: every file × {round advanced, missing, live-only} → correct decision; `--live-only` skips expanding with reason.
6. Script wiring (mocked `NumerAPI` + `tmp_path`): expected download set; atomic CSV content; mocked download failure → no CSV write + exit 1; `None` round → exit 1; default-mode newer version → exit 0 + warning; `--check-only` newer version → exit 3; `--dry-run` → no writes.
7. Drift guard: `CURRENT_DATA_VERSION == load_config(...).data.version` (skip if config absent).
8. `build_era_manifest({"live": (None, None)}, round_id=1300, ...)` → **valid row** (no raise); non-live `(None, None)` → `ValueError`.
9. CSV round-trip (script layer): manifest → CSV → `pandas.read_csv` → identical records; `round_id` float/empty and live `"X"` serialization match the legacy format.
10. Script `features.json` validation: malformed/empty `feature_sets` or empty `targets` → raise (fail loudly), no CSV write.

---

## 4. Workstream B: Analysis Module (`nmr/analysis.py`)

### 4.1 Function surface

Typed, deterministic, era-aware, tested. No I/O (frames in, frames/dicts out), no wall-clock, **no stochastic ops** (regimes are percentile-based, never clustered). Reuses `resolve_feature_sets` and `feature_stability_screen` from `nmr/features.py`.

| Function | Computes |
|---|---|
| `describe_splits(splits: Mapping[str, pl.DataFrame]) -> dict[str, SplitStats]` | Per split: n_rows, n_eras, era range, rows-per-era min/median/max/mean/std, n_unique_ids. Takes **collected** frames (the script collects before calling — no I/O inside the module) |
| `era_structure(frame, era_col) -> pl.DataFrame` | Per-era row/id counts + era-index gaps (non-consecutive era labels flagged) |
| `target_profile(frame, target_cols, era_col) -> pl.DataFrame` | Per target: n_eras_present (staggered availability), per-era mean drift (era-mean std), pooled mean/std/skew/kurtosis, min/max, zero-variance era count |
| `target_correlation_matrix(frame, target_cols, era_col) -> pl.DataFrame` | Per-era Spearman of target pairs, equal-era-weighted mean → long-form with `n_eras` per pair |
| `feature_ic_screen(frame, feature_cols, targets, era_col) -> pl.DataFrame` | Aggregated per-feature stats per reference target (wraps `feature_stability_screen`): mean_corr, corr_std, decay_slope, cross_regime_variance, n_eras, stable |
| `feature_ic_by_era(frame, feature_cols, target_col, era_col) -> pl.DataFrame` | **Per-era per-feature IC long-form** `(era, feature, ic, degenerate)` — `degenerate` flags zero-variance / <2-row / non-finite eras per the screen convention (0.0 IC). The input to `regime_analysis`. Built on the extracted `_per_era_pearson` helper (Phase 0). Per-feature zero-variance-era counts derive from this flag (moved out of `feature_summary`) |
| `feature_summary(chunks: Iterable[pl.DataFrame], feature_cols, era_col) -> pl.DataFrame` | Per-feature pooled mean/std/skew/kurtosis/min/max/missing-rate via **Welford + Terriberry**; caller drives chunking (era-sorted ascending). Zero-variance-era counting lives in the IC path, not here |
| `feature_correlation_structure(chunks: Iterable[pl.DataFrame], feature_cols, era_col) -> FeatureCorrResult` | Per-era rank-gaussianize → per-era corr matrix → equal-weighted average. `FeatureCorrResult(matrix: np.ndarray float32 N×N, top_pairs: pl.DataFrame, summary: dict)`. Caller-driven era chunks (same contract as `feature_summary`). |
| `within_set_redundancy(result: FeatureCorrResult, sets) -> pl.DataFrame` | Per feature set: mean pairwise \|corr\|, median, max — indexes the full matrix |
| `cross_set_membership(sets) -> pl.DataFrame` | Set sizes, overlap counts, empirical subset relations (computed, not assumed — real v5.2: `medium ⊆ all` holds, while `small` is a curated 42-feature set that is a subset of `all` but not of `medium`) |
| `regime_analysis(ic_by_era: pl.DataFrame) -> dict` | Per-era mean feature IC; quartile (25/50/75) + decile regime bands; crash/hot era lists; IC persistence (adjacent-era IC-vector rank corr); rolling IC volatility |
| `benchmark_era_corr(...) -> dict` | Per-era CORR of each benchmark model + meta-model vs reference target → achievable floors/ceilings |

`FeatureCorrResult` lives in `nmr/analysis.py`; tests import it directly (`from nmr.analysis import FeatureCorrResult`).

**Exports** (`nmr/__init__.py` `__all__`): `describe_splits`, `era_structure`, `target_profile`, `target_correlation_matrix`, `feature_ic_screen`, `feature_ic_by_era`, `feature_summary`, `feature_correlation_structure`, `within_set_redundancy`, `cross_set_membership`, `regime_analysis`, `benchmark_era_corr`.

### 4.2 Defined semantics

- **Era weighting:** equal era weight everywhere (simple mean of per-era values/matrices) — matches the repo's per-era-first aggregation convention. Size-weighted pooling explicitly rejected and documented.
- **NaN / staggered targets:**
  - `target_profile`: per target, **non-finite rows are dropped before computing moments** (mean/std/skew/kurtosis); `missing_rate = n_nan / n_total` reported; `n_eras_present` and `zero_variance_era_count` computed on cleaned data. Matches the "main target never NaN, auxiliary targets can be" reality.
  - `target_correlation_matrix`: an era with <2 valid rows or zero variance for either target in a pair is **skipped for that pair**; `n_eras` records the count; mean over observed eras only. Never 0.0, never NaN-poisoned.
  - IC functions (`feature_ic_screen`, `feature_ic_by_era`): degenerate eras (zero variance, all-NaN target) → **0.0** IC with `degenerate=True`, consistent with the existing `feature_stability_screen` convention.
- **Streaming moments:** Welford (mean/variance) + Terriberry extension (M3/M4 → skewness/kurtosis); min/max and missing counts tracked per chunk. Same chunk order + same NumPy build ⇒ bit-identical; cross-platform bit-identity for higher moments is **not** guaranteed — chunked-vs-single-pass is a tight `allclose` test.
- **Split provenance:** functions operate on the split(s) the caller passes. The script analyzes **train ∪ validation concatenated** (global era order 0001–1208) for feature/target/regime stats; the concatenated `decay_slope` includes the train→validation transition, which is **desirable** (exposes the distribution gap) and documented. `benchmark_era_corr` uses validation benchmarks (better coverage) + train benchmarks within their overlap; eras with <2 valid benchmark rows are **silently absent** (not failed), `n_eras` reflects actual overlap.
- **Regime criteria (named constants, no magic values):** `REGIME_LOW_PCT = 10.0`, `REGIME_HIGH_PCT = 90.0` (crash = bottom decile of per-era mean IC; hot = top decile), quartile bands 25/50/75, `IC_VOL_WINDOW = 20` for rolling volatility.

### 4.3 Data flow (`analyze_dataset.py`, thin) & dumps

`resolve_feature_sets(features.json)` + `IngestionAgent` → analysis functions → `artifacts/reports/dataset_analysis/`:

| Dump | Content |
|---|---|
| `overview.json` | splits, era ranges, row counts, features.json summary (sets + 41 targets) |
| `era_structure.parquet` | per-era rows/ids, gaps |
| `targets.json` | per-target profile (availability, drift, moments) |
| `target_corr.parquet` | long-form target-pair correlations with n_eras |
| `feature_summary.parquet` | per-feature moments/missingness |
| `feature_ic_screen.parquet` | feature × reference-target screen (mean_corr, decay, regime variance, stable) |
| `feature_ic_by_era.parquet` | per-era per-feature IC (regime input) |
| `feature_corr_medium.parquet` | medium (780) full matrix, long-form |
| `feature_corr_all_summary.json` | all (2748): top-100 pairs + redundancy summary (full 30 MB float32 matrix kept in-memory; persisted on `--full-all-matrix`) |
| `set_membership.parquet` | set sizes/overlap/subset relations |
| `regimes.json` + `era_signal.parquet` | regime bands, crash/hot eras, per-era metrics, IC persistence |
| `benchmarks.json` | benchmark/meta-model per-era corr |
| `manifest.json` | data version, era ranges, feature counts, refresh date; `generated_at` informational only — **never hashed, never fed into `run_id`/`canonical_scorecards_bytes()`** |

**Reference targets:** resolved at runtime from actual target columns — primary 20D (`target`) + primary 60D by default; `--all-targets` extends the screen to all 41 (cheap: the screen is vectorized per era, ~1000 rows × 2748 features per partition).

**Memory/compute discipline (Design A — single collected frame):** the script collects the concatenated train∪validation frame **once** (≈13 GB float32 at `all`, within the 32 GB budget; freed after the screen calls), partitions it by era, and passes era-chunks to the chunked functions. `feature_ic_screen` wraps the existing frame-taking screen and receives that single frame directly. `feature_summary` / `feature_correlation_structure` consume the era-partitioned chunks — this is **chunked accumulation from the already-collected frame**, not streaming from disk; it bounds per-call working memory to ~one era (≈11 MB) plus accumulators. Budget ~16–32 GB RAM, minutes-to-1h runtime. `feature_correlation_structure(all)` dominates: rank-gaussianize ≈ O(14B) ops (~2–3 min) + per-era corr accumulation ≈ 18 TFLOP (~3–6 min) → ~10–20 min; `medium` ~1–2 min. `--max-eras` for quick iteration.

### 4.4 TDD plan — `tests/test_analysis.py`

1. `describe_splits`/`era_structure`: shapes; empty & single-era frames; gap detection.
2. `target_profile`: moments vs numpy hand-computation; zero-variance column; staggered availability; **all-NaN target era → non-finite rows dropped, missing_rate reported**.
3. `target_correlation_matrix`: hand-computed Spearman on 2×2; symmetry; **partially-NaN target pairs → skipped with n_eras**; determinism.
4. `feature_ic_screen`: multi-target output shape + target labeling (screen itself already tested).
5. `feature_ic_by_era`: hand-computed per-era IC on synthetic frame; **`degenerate` flag on zero-variance / <2-row eras (0.0 IC)**; **single-source-of-truth vs `_per_era_pearson`**.
6. `feature_summary`: moments vs numpy reference; constant column; missing-rate; **two chunked runs bit-identical; chunked vs single-pass tight-allclose**; no zero-variance-era column (moved to IC path).
7. `feature_correlation_structure`: era-averaged matrix vs hand-computed on 5-feature synthetic; symmetry; **unequal-size eras → equal-weight average**; **zero-variance era → 0.0 row/col**.
8. `within_set_redundancy` / `cross_set_membership`: counts; **real-data test — `small ⊆ medium ⊆ all` asserted from `features.json`** (JSON only, cheap for CI).
9. `regime_analysis`: percentile thresholds; crash/hot lists; persistence on constructed rank-stable series; determinism (no stochastic ops).
10. `benchmark_era_corr`: known corr on synthetic benchmark frame; era-overlap filtering (<2 rows → absent).

---

## 5. Workstream C: Renderer & Report

### 5.1 `render_dataset_report.py` (thin)

Reads the dumps → renders `docs/04-research/dataset-analysis-YYYY-MM.md` (YYYY-MM = latest data-refresh date from the manifest, encoding freshness in the filename). Pure string formatting from dump data — same dumps ⇒ byte-identical report. Validates dumps exist/non-empty and the manifest data version equals `CURRENT_DATA_VERSION` (fail loudly otherwise). No charts.

### 5.2 Report structure (LLM-agent-optimized)

| § | Content |
|---|---|
| 0 | Front matter: title, data version, era ranges, feature/target counts, refresh date, artifact manifest path |
| 1 | Dataset overview & schema — splits, columns, feature sets, obfuscation semantics |
| 2 | Era structure — era range, rows-per-era stats, gaps, train/validation boundary, transition note |
| 3 | Targets — availability (n_eras, era windows), distributions, drift, 20D/60D families, target-target correlation highlights |
| 4 | Features — moments/missingness; IC + stability vs reference targets; set membership; per-set redundancy; top pairs; ranked tables |
| 5 | Regimes & signal dynamics — per-era mean IC, quartile/decile bands, crash/hot eras, IC persistence, rolling volatility, train→validation shift |
| 6 | Benchmarks & meta-model — per-era corr, achievable floors/ceilings, benchmark ladder |
| 7 | Modeling implications — prose guidance (purged era CV, target selection, stability-based feature selection, ensembling/neutralization, leakage warnings) with injected numbers |

**LLM-friendliness rules:** every table is preceded by a one-line **schema block** (columns + units); fixed precision in tables, full precision in dumps; every section ends with a **"Key takeaways"** bullet block of citable facts; each section header cites its backing artifact path.

**Renderer test (`tests/test_report_render.py`):** synthetic dump fixture → byte-identical output across two runs; valid Markdown structure (headers, well-formed tables); numbers formatted.

---

## 6. Docs Updates (SSOT, same commit)

| File | Addition |
|---|---|
| `README.md` | "Refreshing data" subsection (command, flags, version alert) + pointer to the dataset analysis report (README owns data-asset requirements) |
| `ARCHITECTURE.md` | `numerai_era_data.csv` schema; §3 dependency graph adds `nmr/refresh.py`, `nmr/analysis.py`, three root scripts; §P analysis-function spec |
| `AGENTS.md` | toolkit table rows: refresh / analyze / render scripts (keep within 32 KB budget) |
| `docs/DOCS_README.md` | register the new report in the master docs map |
| `CONTRIBUTING.md` | no change (verification commands unchanged) |

---

## 7. Verification Gates

1. Full `pytest -q` — existing 413 tests + new refresh/analysis/render tests, all green.
2. Real-data smoke: `analyze_dataset.py --max-eras 5 --features small` (pipeline sanity, minutes).
3. Refresh smoke: `refresh_data.py --dry-run` (no network writes).
4. Real refresh run once (live.parquet ~20 MB + weekly expanding files incl. validation ~4 GB; `--live-only` is the documented cheap escape hatch).
5. Full production run: `analyze_dataset.py` (full universe) + `render_dataset_report.py` → commit the report. Timings reported truthfully.

## 8. Implementation Phases

- **Phase 0** — extract `_per_era_pearson` in `nmr/features.py` (micro-commit; screen behavior unchanged, existing screen tests stay green; add single-source-of-truth test).
- **Phase 1** — `nmr/refresh.py` + `refresh_data.py` + tests + README/ARCH docs.
- **Phase 2** — `nmr/analysis.py` + tests.
  - **Phase-boundary gate:** run `analyze_dataset.py --max-eras 5 --features small`; assert all dumps exist, have the expected columns, and `manifest.json` parses. Catches Phase 2 integration issues before renderer work begins.
- **Phase 3** — `analyze_dataset.py` + `render_dataset_report.py` + renderer test.
- **Phase 4** — real-data full run → report + docs registration.
- **Phase 5** — full verification gates.

## 9. Risks & Open Items

- **Validation re-download cost** (~4 GB weekly) — `--live-only` escape hatch; default remains correct-by-construction.
- **`feature_correlation_structure(all)` runtime** (~10–20 min) — acceptable; `--max-eras` for iteration.
- **Full 41-target screen** — cheap per cost model but gated behind `--all-targets`.
- **Benchmark parquet gaps** (no rows in first ~30 train eras) — overlap-filtered, `n_eras` reports actual coverage.
- **numerapi API drift** (version format, dataset listing) — malformed version strings raise loudly rather than being silently mishandled; policy tuples are intersected with the live listing.
- **LLM report readability** — no charts by design; dense tables + schema blocks are the contract; dumps carry full precision.
- **Phase-0 refactor risk** — `_per_era_pearson` extraction must not change `feature_stability_screen` output; existing screen tests are the guard.
