# Model Tournament Dashboard Delivery Summary

**Date:** 2026-08-25
**Status:** Implemented; final verification is blocked by two pre-existing real-data parity failures.

## Task Summary & Assumptions

The dashboard is now a leaderboard-first **Model Tournament** for offline validation evidence. The first viewport emphasizes the master leaderboard and the ML ADVANTAGE comparison, with model selection opening an evidence dossier. The UI is explicitly labelled `OFFLINE EVALUATION · Suite v2` and `No live / production performance included`.

The dashboard uses the evaluation suite's existing `v5.3` scorecard fields without changing scorecard schemas or canonical serialization.

- Default rank metric: `mmc`, matching the requested priority order and active scorecard availability.
- Profitability: `cagr_1y`, the stored or reconciled annual compounded return. No fallback to `mean_payout` is used when CAGR is unavailable.
- Trained cohort: `source` is `trained` or `trained_legacy`.
- Heuristic cohort: benchmark tiers 0–2, covering null, Ridge, and shallow-tree baselines.
- Benchmark cohort: benchmark tiers 3–4, covering canonical/community and official tier-4 references.
- Full-version rows: lineage metadata only; they have no comparable out-of-sample rank.
- Champion: the registry pointer and a passing capital gate are both required. Rank #1 alone is not champion.
- RAPS and win-rate do not exist in the active scorecard. RAPS was not recreated; MMC is the selected available default rank metric. Win-rate is omitted rather than inferred.
- Rank movement is disabled until a comparable prior snapshot exists. Registry mtimes do not create trend claims.

## Affected Files

- `nmr/dashboard.py`: direction-aware metric registry, cohort derivation, deterministic ranking, rank maps, ML ADVANTAGE calculations, compact detail dossiers, and renderer-neutral tournament payload.
- `nmr/__init__.py`: public dashboard API exports.
- `dashboard_ui/charts.py`: compact columnar timeseries payload helper.
- `dashboard_ui/report.py`: deterministic HTML builder and file compiler; generated production assets are inlined.
- `dashboard_ui/app.py`: thin Streamlit host embedding the same HTML renderer.
- `dashboard_app.py`: thin Streamlit entry wrapper.
- `dashboard_ui/static/layout.html`: leaderboard-first Model Tournament shell and model drawer mount.
- `dashboard_ui/static/app.js`: client-side rank/search/cohort/shortcut state, leaderboard, Advantage strip, landscape, profile, dossier, and secondary charts.
- `dashboard_ui/static/style.css`: reference-inspired dark research workspace with gold/mint/coral evidence states and responsive drawer/table styling.
- `dashboard_ui/static/app.min.js`, `dashboard_ui/static/style.min.css`: generated minified production assets used by the self-contained report.
- `dashboard_ui/components.py`, `dashboard_ui/charts_components.py`, `dashboard_ui/tables.py`: removed superseded parallel rendering system.
- `tests/test_dashboard.py`, `tests/test_dashboard_ui.py`, `tests/test_scripts.py`: engine, payload, renderer, XSS, host, lazy-import lint, and missing-reference contracts updated.
- `tests/test_dashboard_charts_components.py`, `tests/test_dashboard_tables.py`: removed tests for the retired rendering system.
- `AGENTS.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `README.md`: updated architecture, test-count, host, and governance records.
- `docs/superpowers/specs/2026-08-18-vanilla-dashboard-design.md`, `docs/superpowers/plans/2026-08-18-vanilla-dashboard.md`: reconciled active budget/transport ownership and marked the original plan as historical.
- `DASHBOARD_DELIVERY_REPORT.md`, `SESSION_COMPLETION_REPORT.md`, and superseded 2026-08-16/18 dashboard plans/specs: marked obsolete dashboard claims as historical and pointed readers to this summary.

## Architectural Approach

`nmr/dashboard.py` remains the analytical and tested boundary. It reads registry, benchmark, and manifest inputs and produces deterministic ranking/cohort/detail data. `dashboard_ui` performs presentation shaping only.

There is now one visual renderer. `dashboard_ui/report.py` builds the self-contained offline document, and `dashboard_ui/app.py` embeds that same document with `st.components.v1.html`. The former native Streamlit/Altair rendering path and the unfinished three-format component classes were retired; `service.py` remains available as a read-only compatibility/data facade for existing explainers and service callers, but it is not a second renderer.

The wire payload is compact and deterministic:

- leaderboard rows use a shared field schema and aligned arrays;
- ranks use aligned arrays across the metric registry;
- scorecard details keep all displayed fields but reuse row-carried metric values;
- per-era series use shared model indexes and deterministic signed-32-bit base64 transport with an explicit missing sentinel; strict benchmark winners are computed in full precision by Python and transported as model IDs;
- filesystem paths, wall-clock generation values, and timing fields are excluded from the payload.

## Implementation

The report offers:

- search, direction-aware rank switching, cohort tabs, and shortcuts;
- explicit higher/lower direction text for every ranking metric;
- rank, type marker, selected metric, CORR, MMC, CORR Sharpe with CI, CAGR/gain-to-pain, drawdown, and era count;
- an ML ADVANTAGE strip comparing the best trained, heuristic, and benchmark rows;
- CORR × MMC landscape scatter and selected-model horizontal profile;
- the existing multimetric timeseries, similarity, and drawdown evidence as secondary views;
- numeric axes and identity legends for landscape, timeseries, and drawdown charts;
- chart hover popups include color-matched swatches for each model/value line;
- keyboard-accessible row selection and an escape/backdrop-close model drawer;
- keyboard-accessible landscape marks with true-scale dossier evidence;
- full scorecard evidence, provenance, gate state, and immutable relative evidence references in the dossier.

Missing and non-finite values remain unavailable through transport, render as gaps or `—`, and are excluded from chart ranges. The dashboard never writes registry, champion, parquet, cache, or deployment state.

## Tests

Added coverage for metric directions, default ranking, cohort semantics, deterministic null/tie ordering, null-rank behavior, rank maps, ML ADVANTAGE edge cases, compact payloads, missing-series gaps, missing-reference gaps, sub-micro rank precision, artifact size, champion requirements, XSS-safe embedding, encoding safety, valid row markup, pointer tooltip/axis/medal contracts, and the single Streamlit renderer contract. The current collection is 1,099 tests after retiring 19 tests belonging to the removed component/table renderer and adding eight dashboard regression contracts.

## Execution Verification

- Baseline before implementation: `ruff check .` failed on pre-existing dashboard lint findings; the full suite reported 1,105 passed and 4 failed.
- Focused dashboard/compiler suites after the final polish: **32 presentation tests passed**; the full suite result is recorded below.
- The two failures are unchanged real-data fixture mismatches in `tests/test_dashboard.py`:
  - stored CORR versus recomputed CORR (`0.0141614` stored versus `0.0149663` recomputed);
  - stored/reconciled CAGR parity (`1.0220556` versus `0.8804255`).
- Final repository-wide Ruff check: **passed**.
- `node --check dashboard_ui/static/app.js` and `node --check dashboard_ui/static/app.min.js`: passed.
- Real `artifacts/dashboard.html`: generated successfully at **110,031 bytes** (**107.45 KiB**), within the named 112 KiB budget.
- The final report build SHA-256 is `E1C492E7BD38CDC85D94EE418AA8FE9E7A6DC03DE78E458E2FD7BB9E09EC1BD1`.
- The generated artifact contains no `plotly`, `altair`, or workspace absolute path markers.
- Streamlit smoke: started successfully on port 8502 and `/_stcore/health` returned `ok`; the process was terminated after verification.

## Browser UX Review

The built-in browser review found and fixed four concrete defects:

- raw Unicode punctuation rendered as mojibake in the report and minified asset;
- row click handlers were broken by an unterminated `data-model-id` attribute;
- missing turnover values received misleading numeric ranks;
- negative ML Advantage against the benchmark used the same positive styling.
- full model hashes made chart hover cards unnecessarily wide.

The review also clarified the selected metric as `RANKED: <metric>` in the active leaderboard header, shortened the mobile search placeholder, tightened the mobile toolbar, and changed chart tooltips to use model name plus an eight-character ID while preserving full hashes in the dossier. Verified interactions include desktop and mobile rendering, rank switching, lower-is-better ordering, search, shortcut empty states, keyboard dossier open/close with focus restoration, scorecard rendering, and secondary chart population. At a 375px mobile viewport, the document client width is 360px because of the scrollbar, with no horizontal overflow and an internally scrollable leaderboard.

The visual pass then matched the supplied reference direction without importing its portfolio semantics: near-black canvas, compact left navigation, rounded dark panels, tabular metrics, warm gold primary accent, mint/coral result states, active section navigation, and live pointer tooltips across the timeseries, landscape, profile, similarity, and drawdown charts. The alpha chart exposes y-axis metric ticks, x-axis era ticks, an explicit `Evaluation era` title, and a separated color legend with compact model identities. Benchmark rows use a distinct background/stripe, the correlation matrix uses colored off-diagonal cells with black self-similarity squares, and the first three trained models receive gold, silver, and bronze `1`, `2`, and `3` markers with their normal leaderboard ranks shown beneath.

## Risks & Follow-Ups

The full functional gate completed with **1,097 passed and 2 failed** out of 1,099 collected tests. The two real-data parity failures predate this dashboard work and must be resolved before claiming a green functional gate. They indicate stale or inconsistent stored real-data scorecard/payout fixtures and should be investigated separately rather than masked by dashboard code.

The minified production assets are generated from `dashboard_ui/static/app.js` and `style.css`; regenerate both when their readable sources change. A future build helper could formalize this mechanical step without adding a runtime dependency.

The Streamlit shell still requires the installed Streamlit runtime for manual launch verification. Browser automation is not part of the Python test dependencies, so interactive behavior is validated against the generated artifact during review rather than in CI. No live-performance monitoring or deployment telemetry has been added; this dashboard remains an offline evaluation surface.
