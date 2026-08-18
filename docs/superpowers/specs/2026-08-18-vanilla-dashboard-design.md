# Pure Vanilla HTML/CSS/SVG Executive Dashboard — Design Spec

> **Date:** 2026-08-18 · **Status:** Approved for implementation
> **Supersedes:** the Plotly embedding portions of `2026-08-16-executive-dashboard-design.md` and `2026-08-16-executive-dashboard-v2-design.md` (historical records, unchanged). The `nmr/dashboard.py` engine contract is preserved verbatim; only the presentation layer is replaced.

---

## 1. Mission

Completely eliminate Plotly (the Python package, the `get_plotlyjs()` bundle embed, and every legacy charting wrapper) from the repository. Replace the executive dashboard's presentation layer with an isolated, zero-dependency Vanilla HTML/CSS/SVG front-end that compiles into `artifacts/dashboard.html` at **< 100 KB** — double-clickable offline, deterministic, and with all chart geometry math covered by pytest.

## 2. Non-Negotiable Invariants

1. **Zero external visual dependencies.** No `plotly` import anywhere (code *and* tests); no CDN `<script src>` tags; no npm/Node toolchains; no bundled charting libraries. `plotly==6.6.0` is removed from `requirements.txt` in the same commit that stops importing it.
2. **Double-clickable `file://` portability.** `artifacts/dashboard.html` runs standalone: single file, inline CSS + JS, no network requests, no CORS restrictions, no server.
3. **Front-end isolation.** All raw web assets (HTML scaffold, CSS, JS) live in `dashboard_ui/static/`; presentation logic lives in `dashboard_ui/`. `nmr/` never imports plotly/streamlit (unchanged) and `nmr/dashboard.py` is **untouched** by this work.
4. **Tested boundary.** Every metric/formula stays in `nmr/` (unchanged). Chart geometry math has a pure Python reference implementation in `dashboard_ui/charts.py` that pytest asserts against; `app.js` mirrors the same algorithms client-side (the repo has no JS test runner — decision #7).
5. **Deterministic artifacts.** No wall-clock timestamps, no absolute paths, sorted-key JSON, fixed templates and static assets ⇒ byte-identical output for identical registry/data state.
6. **Hard gates.** Every commit passes `ruff check .` + `pytest -q`. Test count claims in `AGENTS.md` (two places) and `CONTRIBUTING.md` are updated in the same commit that changes the collected count (`tests/test_docs_hygiene.py` enforces this).

## 3. Architecture & Module Topology

```
dashboard_ui/                       # presentation package (home unchanged)
├── __init__.py
├── charts.py                       # pure geometry + payload builders (Plotly builders deleted)
│     data_to_svg_path(...)         #   reference: X,Y scaling -> SVG path string (tested)
│     cumulative_series(...)        #   reference: cumsum / cumprod(1+r) (tested)
│     drawdown_series(...)          #   reference: wealth/peak - 1 (tested)
│     build_dashboard_payload(...)  #   merges engine output into the JSON data contract
├── report.py                       # compiler: layout.html template + style.css + app.js -> dashboard.html
├── app.py                          # Streamlit, native st.* charts only (no plotly)
└── static/
    ├── layout.html                 # semantic HTML scaffold with {{ PLACEHOLDER }} slots
    ├── style.css                   # full design system (tokens, badges, grid, tooltip)
    └── app.js                      # vanilla controller: dataToSvgPath + view toggle + tooltip

generate_dashboard.py               # unchanged thin wrapper (all logic in dashboard_ui.report)
dashboard_app.py                    # unchanged thin wrapper (all logic in dashboard_ui.app)

tests/
├── test_dashboard.py               # engine tests ONLY (nmr.dashboard) — plotly assertions removed
└── test_dashboard_ui.py            # NEW: presentation tests (geometry, payload, compiler, artifact)
```

**Decisions**

- #1 **Front-end home:** extend the existing `dashboard_ui/` package. The just-merged isolation refactor already placed all presentation code there; a parallel `templates/dashboard/` would split the front-end across two homes. The spec's `templates/dashboard/` alternative is rejected in favor of `dashboard_ui/static/` as the raw-assets location.
- #2 **Streamlit survives, natively.** The interactive research view stays (documented tool), rewritten without Plotly: `st.bar_chart`, `st.scatter_chart`, and a pandas `Styler.background_gradient` heatmap in `st.dataframe`. `streamlit` remains a pinned user-granted dependency; `plotly` is removed.
- #3 **`nmr/dashboard.py` untouched.** It already emits plotly-free raw data (eras, metrics, drawdowns, matrix, gate status, KPIs). Presentation geometry is not an engine concern.
- #4 **Test split:** `tests/test_dashboard.py` shrinks to engine-only tests; new `tests/test_dashboard_ui.py` owns all presentation-layer tests (geometry reference math, payload contract, HTML compiler, artifact contract). Net count change synced to docs in the same commit.

## 4. Data Payload Contract (`dashboard-data`)

`build_dashboard_payload(...)` in `charts.py` consumes the existing `nmr.dashboard` outputs and emits one JSON object serialized into `<script id="dashboard-data" type="application/json">`:

```json
{
  "eras": ["1133", ..., "1218"],
  "meta_downside_mask": [false, ...],
  "metrics": {
    "<run_id>": {
      "label": "name · abc12345",
      "payout": {"standard": [r_1, ...]},
      "corr20":  {"standard": [rho_1, ...]},
      "mmc20":   {"standard": [...]}, "corr60": {...}, "mmc60": {...},
      "bmc": {...}, "cwmm": {...}
    }
  },
  "leaderboard": [{"label": "...", "sharpe": 0.9, "ci_low": 0.7, "ci_high": 1.1,
                    "cagr_1y": 0.12, "max_drawdown": 0.2, "deflated_sharpe": 0.97,
                    "champion": false}, ...],
  "similarity": {"labels": [...], "matrix": [[1.0, ...]]},
  "hurdle_sharpe": 0.78,
  "ensemble_sharpe": 1.234 | null
}
```

**Rules (all inherited from the v2 contract, unchanged):**

- **Standard arrays only.** Cumulative series and drawdowns are *derived client-side* (decision #7) — this halves the payload and is what keeps the total artifact under the **< 100 KB** hard gate.
- **Derivation rules:** payout cumulative = `cumprod(1 + r_t)`; correlation-family cumulative = `cumsum(rho_t)`; drawdown = `wealth / peak - 1` (peak = `np.maximum.accumulate`). The Python reference (`cumulative_series`, `drawdown_series`) is asserted against `nmr.dashboard._cumulative_from_standard` in tests (parity, decision #8).
- Serialization: `sort_keys=True`, `allow_nan=False` (fail loud on non-finite), `</` → `<\/` escaping. No wall-clock fields, no absolute paths.
- Empty registry / missing v5.3 assets degrade to the existing empty-payload shape; charts render the "Timeseries data unavailable without local v5.3 assets" placeholder (never raise).
- XSS: labels escaped end-to-end (`html.escape` server-side for server-rendered slots; the JSON node is `</`-escaped; `app.js` has an `esc()` helper for tooltip text).

## 5. Front-End Engine

### 5.1 `dashboard_ui/static/style.css`

Full design system per spec §3.1: dark palette tokens (`--bg: #0d1117`, `--surface: #161b22`, `--border: #30363d`, `--text: #c9d1d9`, `--accent: #58a6ff`, `--danger: #f85149`, `--success: #3fb950`, `--gold: #d29922`), CSS grid KPI cards, responsive table overflow wrapper, pill badges (`.champion`, `.ready`, `.research`, `.hurdle`, `.benchmark`, `.full`, `.gate-fail` tint), tooltip styling, SVG chart container rules. Grows from the current 22-line file into the complete token/component layer.

### 5.2 `dashboard_ui/static/app.js` (thin controller, ~150 lines)

- Reads `#dashboard-data` via `JSON.parse`.
- `dataToSvgPath(values, width, height, yMin, yMax)` — ~20-line coordinate-scaling helper producing an SVG `<path d="M... L...">` string; mirrors the tested Python reference `data_to_svg_path` (decision #7).
- **Multi-metric timeseries:** native `<select>` (payout/corr20/mmc20/corr60/mmc60/bmc/cwmm) + Standard/Cumulative toggle buttons; on change derives the cumulative series (cumsum / cumprod), regenerates paths via `dataToSvgPath`, swaps `<path>` elements, updates the axis label and tickformat legend (per metric × view: e.g. payout standard = "Per-Era Net Return" / percent; payout cumulative = "Cumulative Wealth (1.0 Stake)"; corr-family `.4f`). Meta-model drawdown eras render as background `<rect>` spans from `meta_downside_mask`.
- **Sharpe leaderboard:** horizontal SVG bars sorted ascending, asymmetric CI whiskers (`<line>` per bar), dashed vertical hurdle line at `hurdle_sharpe`, champion hatch/stripe marker.
- **Similarity matrix:** a native `<table>` of `K×K` cells with `rgba(88, 166, 255, α)` background opacity from correlation magnitude; row/col 0 (top contender) highlighted; diversification badge + equal-weight blended Sharpe card (values computed in Python, rendered server-side).
- **Underwater drawdown:** SVG area path (downside fill) derived client-side from payout standard.
- **Tooltip:** one floating div on `mousemove` / `mouseleave` over SVG data points.
- Scoped to its own DOM root (`#dashboard-root`), never `document`-level globals beyond the IIFE (decision #24 pattern from v2, retained).

### 5.3 `dashboard_ui/static/layout.html` + `dashboard_ui/report.py`

`layout.html` holds the semantic scaffold with the spec's placeholders: `{{ KPI_CARDS }}`, `{{ METRIC_CONTROLS }}`, `{{ TIMESERIES_SVG }}`, `{{ LEADERBOARD_SVG }}`, `{{ DIVERSIFICATION_SECTION }}`, `{{ DECISION_TABLE }}`, `{{ DRAWDOWN_SVG }}`, `{{ AUDIT_ACCORDION }}`, `{{ INLINE_DATA_SCRIPT }}`.

`report.py` compiles: formats the template, inlines `style.css` + `app.js`, injects the `dashboard-data` node, and server-renders the deterministic slots — KPI cards, decision table rows (badges, `gate-fail` tinting, `FULL` chips, group headers: Champion → Promoted Full → Fleet → Benchmark), diversification badge + ensemble card, technical accordion. All existing pure helpers (`_kpi_cards`, `_table_rows`, `_row_html`, `_technical_entries`, `_diversification_stats`, `_ensemble_sharpe`) are preserved as-is. The five report sections and the technical accordion are preserved 1:1 from the current output.

## 6. Streamlit Rewrite (`dashboard_ui/app.py`)

Only the three render functions change; every pure shaping helper is untouched:

- `render_leaderboard` → `st.bar_chart` (horizontal) for evaluable rows + `st.dataframe` carrying Sharpe, CI bounds, CAGR, Max DD, DSR (CI rendered as columns — native charts have no error bars).
- `render_fleet` → `st.scatter_chart` (neutralization proportion vs metric).
- `render_robustness_matrix` → `st.dataframe` of the numeric matrix styled with `Styler.background_gradient(cmap="RdYlGn")`; `pl.DataFrame.to_pandas()` nulls render as blank cells (missing metrics never error the formatter).

Imports drop `plotly.express`; `streamlit` stays. `tests/test_scripts.py` adds a guard that the module source contains no `plotly` reference.

## 7. Testing Plan

### `tests/test_dashboard.py` (engine — plotly assertions removed)

Unchanged engine tests: payload extraction, gate status, capital reconcile, similarity matrix, determinism, missing-asset degradation. Deleted: every `from plotly.colors import ...` import, `build_leaderboard_bar_chart`/`build_drawdown_chart`/`build_similarity_matrix_chart` figure-object assertions, `multimetric_chart_html` block tests, `Plotly.newPlot(` counts, `plotly-engine-embed` marker, "size unbounded" assertions, `_build_html` figure-injection tests (moved/replaced below).

### `tests/test_dashboard_ui.py` (new — presentation)

1. **Geometry reference math:** `data_to_svg_path` produces exact path strings for known inputs (0, 1, 2-point edge cases; min==max degenerate range); `cumulative_series` parity with `nmr.dashboard._cumulative_from_standard`; `drawdown_series` peak-trough correctness; determinism across calls.
2. **Payload contract:** `build_dashboard_payload` emits the exact schema (standard-only arrays, labels, leaderboard, similarity, hurdle, ensemble); sorted-key JSON round-trips through the data node.
3. **HTML compiler:** `_build_html`/`generate_dashboard()` output contains the five section headings, `id="dashboard-data"`, `class="badge"` pills, `gate-fail` tinting, group headers; escapes hostile strings (`<script>`, `"><img onerror>`); byte-deterministic across two calls.
4. **Artifact contract (hard gate):** `generate_dashboard()` writes the file; `size < 100 KB`; zero occurrences of `plotly` (case-insensitive); no `<script src=`; `id="dashboard-data"` present; report still compiles with an empty registry and with missing v5.3 assets.

### `tests/test_scripts.py`

Streamlit pure-helper tests unchanged; add the no-plotly-import guard.

## 8. Docs & Hygiene (same-commit)

- `requirements.txt`: remove `plotly==6.6.0`.
- `AGENTS.md`: dependency-exception line drops Plotly (keeps Streamlit, re-pointed at the native `dashboard_ui/app.py`); executive-dashboard toolkit row re-pointed at the new spec; test-count claims (two places) updated to the new collected count.
- `ARCHITECTURE.md`: §W and the module table rows for `dashboard_ui/charts.py` / `report.py` / `app.py` — vanilla SVG rendering, native Streamlit, `< 100 KB` budget, `tests/test_dashboard_ui.py`.
- `CONTRIBUTING.md`: test-count claim updated.
- Old specs (`2026-08-16-executive-dashboard*.md`) remain as historical records; this spec supersedes their presentation decisions.

## 9. Verification Gates (each task commit + final)

```powershell
# 1. Lint gate
.\.venv\Scripts\python -m ruff check .
# 2. Functional gate (count must match the doc claims)
.\.venv\Scripts\python -m pytest -q
# 3. Compiler execution
.\.venv\Scripts\python generate_dashboard.py
# 4. Artifact contract
.\.venv\Scripts\python -c "
from pathlib import Path
p = Path('artifacts/dashboard.html')
assert p.exists()
size_kb = p.stat().st_size / 1024
assert size_kb < 100, f'Bundle size too large: {size_kb:.2f} KB'
text = p.read_text(encoding='utf-8')
assert 'plotly' not in text.lower()
assert '<script src=' not in text
assert 'id=\"dashboard-data\"' in text
print(f'SUCCESS: {size_kb:.2f} KB, pure HTML/CSS/JS')
"
```

## 10. Implementation Order (for the writing-plans phase)

1. `dashboard_ui/charts.py` — delete Plotly builders; add `data_to_svg_path`, `cumulative_series`, `drawdown_series`, `build_dashboard_payload`; re-export in `dashboard_ui/__init__.py`.
2. `dashboard_ui/static/style.css` — full design system.
3. `dashboard_ui/static/app.js` — controller + `dataToSvgPath` + tooltip.
4. `dashboard_ui/static/layout.html` + `dashboard_ui/report.py` — compiler rewrite; `dashboard-data` node; drop `plotly.io`/`get_plotlyjs`.
5. `dashboard_ui/app.py` — native Streamlit rewrite; drop `plotly.express`.
6. `tests/test_dashboard_ui.py` (new) + prune `tests/test_dashboard.py` + `tests/test_scripts.py` guard.
7. `requirements.txt`, `AGENTS.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md` — same-commit sync; regenerate `artifacts/dashboard.html`.
