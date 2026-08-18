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
│     data_to_svg_path(...)         #   reference: X/Y scaling -> SVG path string (tested)
│     svg_area_path(...)            #   reference: closed baseline polygon for fills (tested)
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
    "payout": {"<run_id>": {"standard": [r_1, ...], "label": "name · abc12345"}},
    "corr20":  {"<run_id>": {"standard": [rho_1, ...], "label": "..."}},
    "mmc20":   {"<run_id>": {"standard": [...]}}, "corr60": {...},
    "mmc60":   {...}, "bmc": {...}, "cwmm": {...}
  },
  "leaderboard": [{"label": "...", "sharpe": 0.9, "ci_low": 0.7, "ci_high": 1.1,
                    "cagr_1y": 0.12, "max_drawdown": 0.2, "deflated_sharpe": 0.97,
                    "champion": false}, ...],
  "similarity": {"labels": [...], "matrix": [[1.0, ...]]},
  "hurdle_sharpe": 0.78,
  "ensemble_sharpe": 1.234 | null
}
```

**Rules:**

- **Metric-first hierarchy.** `metrics` is keyed `metrics[metric_name][run_id]`, **identical to `nmr.dashboard.extract_multimetric_timeseries` output — `build_dashboard_payload` performs no key regrouping**, so `app.js` indexes `payload.metrics[currentMetric][model_id]` (this is also what the v2 controller already does; the review's initial model-first sketch was inverted and is rejected).
- **Standard arrays only.** Cumulative series and drawdowns are *derived client-side* (decision #7) — this halves the payload and is what keeps the total artifact under the **< 100 KB** hard gate.
- **Derivation rules:** payout cumulative = `cumprod(1 + r_t)`; correlation-family cumulative = `cumsum(rho_t)`; drawdown = `wealth / peak - 1` (peak = `np.maximum.accumulate`). The Python reference (`cumulative_series`, `drawdown_series`) is asserted against `nmr.dashboard._cumulative_from_standard` in tests (parity, decision #8).
- Serialization: `sort_keys=True`, `allow_nan=False` (fail loud on non-finite), `</` → `<\/` escaping. No wall-clock fields, no absolute paths.
- Empty registry / missing v5.3 assets degrade to the existing empty-payload shape; charts render the "Timeseries data unavailable without local v5.3 assets" placeholder (never raise).
- XSS: labels escaped end-to-end (`html.escape` server-side for server-rendered slots; the JSON node is `</`-escaped; `app.js` has an `esc()` helper for tooltip text).

## 5. Front-End Engine

### 5.1 `dashboard_ui/static/style.css`

Full design system per spec §3.1: dark palette tokens (`--bg: #0d1117`, `--surface: #161b22`, `--border: #30363d`, `--text: #c9d1d9`, `--accent: #58a6ff`, `--danger: #f85149`, `--success: #3fb950`, `--gold: #d29922`), CSS grid KPI cards, responsive table overflow wrapper, pill badges (`.champion`, `.ready`, `.research`, `.hurdle`, `.benchmark`, `.full`, `.gate-fail` tint), tooltip styling, SVG chart container rules, `.grid-line` and `.crosshair` stroke rules. Grows from the current 22-line file into the complete token/component layer.

### 5.2 `dashboard_ui/static/app.js` (thin controller, ~200 lines)

Reads `#dashboard-data` via `JSON.parse`. All geometry mirrors the tested Python reference in `charts.py` (decision #7).

**Coordinate scaling (`dataToSvgPath`, mirrored in Python as `data_to_svg_path`).** SVG places (0,0) at the **top-left**, so the y-axis is inverted. For values indexed `i = 0..N-1` over an inner plot area of `W × H` (viewport minus padding `pad = (top, right, bottom, left)`), with `y_min`/`y_max` the series range:

$$x_i = \text{pad}_{\text{left}} + \frac{i}{N-1} \cdot (W - \text{pad}_{\text{left}} - \text{pad}_{\text{right}})$$

$$y_i = \text{pad}_{\text{top}} + \left(1 - \frac{v_i - y_{\min}}{y_{\max} - y_{\min}}\right) \cdot (H - \text{pad}_{\text{top}} - \text{pad}_{\text{bottom}})$$

- **Degenerate-range guard:** when `abs(y_max - y_min) < 1e-12` (flat series — zero correlation, null baseline), expand the range (`y_min -= 1; y_max += 1`) in **both** the Python reference and `app.js`, so the divisor never zeroes and no `NaN`/`Infinity` path strings are emitted.
- **Empty input** → empty string (`""`), never a malformed `d` attribute.
- **Global y-range per metric:** `y_min`/`y_max` are computed across **all active series of the selected metric** (both views), not per line — toggling metrics or models never misaligns the shared axis.

**`svg_area_path` (Python) / area closure (JS).** Filled series (drawdown, return areas) render as closed polygons: the line path `M x0,y0 L x1,y1 … L xN,yN` then closes to the baseline anchor — `L xN, y_base → L x0, y_base → Z` — so the SVG fill covers exactly the region between the curve and the baseline (drawdown baseline `y_base` = the 0-axis).

**Multi-metric timeseries:** native `<select>` (payout/corr20/mmc20/corr60/mmc60/bmc/cwmm) + Standard/Cumulative toggle buttons; on change, derives the cumulative series (cumsum / cumprod), recomputes the global y-range, regenerates paths via `dataToSvgPath`, swaps `<path>` elements, and updates the axis label + tickformat legend (per metric × view: e.g. payout standard = "Per-Era Net Return" / percent; payout cumulative = "Cumulative Wealth (1.0 Stake)"; corr-family `.4f`). Meta-model drawdown eras render as background `<rect>` spans from `meta_downside_mask`.

**Grid lines:** three dashed horizontal `<line class="grid-line">`s at `y_max`, the zero/midpoint, and `y_min`, each with a numeric text label, regenerated on every view change.

**Sharpe leaderboard:** horizontal SVG bars sorted ascending, asymmetric CI whiskers (`<line>` per bar), dashed vertical hurdle line at `hurdle_sharpe`, champion hatch/stripe marker.

**Similarity matrix:** a native `<table>` of `K×K` cells with `rgba(88, 166, 255, α)` background opacity from correlation magnitude; row/col 0 (top contender) highlighted; diversification badge + equal-weight blended Sharpe card (values computed in Python, rendered server-side).

**Underwater drawdown:** SVG area path (downside fill, closed to baseline) derived client-side from payout standard.

**Tooltip + crosshair:** one floating div on `mousemove` / `mouseleave` over SVG data points. Era index from the pointer via proportional mapping `t = round((x_mouse − pad_left) / Δx)` with an explicit bounds check `0 ≤ t < N_eras` (out-of-range → no tooltip). A vertical dashed `<line class="crosshair">` snaps to the nearest era, and the tooltip lists every visible model's value at that era. Scoped to the dashboard root (`#dashboard-root`), never `document`-level globals beyond the IIFE (decision #24 pattern from v2, retained).

### 5.3 `dashboard_ui/static/layout.html` + `dashboard_ui/report.py`

`layout.html` holds the semantic scaffold with the spec's placeholders: `{{ KPI_CARDS }}`, `{{ METRIC_CONTROLS }}`, `{{ TIMESERIES_SVG }}`, `{{ LEADERBOARD_SVG }}`, `{{ DIVERSIFICATION_SECTION }}`, `{{ DECISION_TABLE }}`, `{{ DRAWDOWN_SVG }}`, `{{ AUDIT_ACCORDION }}`, `{{ INLINE_DATA_SCRIPT }}`.

`report.py` compiles: formats the template, inlines `style.css` + `app.js`, injects the `dashboard-data` node, and server-renders the deterministic slots — KPI cards, decision table rows (badges, `gate-fail` tinting, `FULL` chips, group headers: Champion → Promoted Full → Fleet → Benchmark), diversification badge + ensemble card, technical accordion. All existing pure helpers (`_kpi_cards`, `_table_rows`, `_row_html`, `_technical_entries`, `_diversification_stats`, `_ensemble_sharpe`) are preserved as-is. The five report sections and the technical accordion are preserved 1:1 from the current output.

**Portable asset resolution (P0):** static assets are anchored to the module location — `_STATIC_DIR = Path(__file__).resolve().parent / "static"` — so the compiler works from any CWD (this pattern already exists in `report.py`/`charts.py`; it is retained and made explicit in the new compiler).

## 6. Streamlit Rewrite (`dashboard_ui/app.py`)

Only the three render functions change; every pure shaping helper is untouched:

- `render_leaderboard` → `st.bar_chart` (horizontal) for evaluable rows + `st.dataframe` carrying Sharpe, CI bounds, CAGR, Max DD, DSR (CI rendered as columns — native charts have no error bars).
- `render_fleet` → `st.scatter_chart` (neutralization proportion vs metric).
- `render_robustness_matrix` → `st.dataframe` of the numeric matrix styled with `Styler.background_gradient(cmap="RdYlGn")` **and `.format(na_rep="—")`**, so `pl.DataFrame.to_pandas()` nulls render as `—` (never raw float `nan` text and never a formatter error).

Imports drop `plotly.express`; `streamlit` stays. `tests/test_scripts.py` adds a guard that the module source contains no `plotly` reference.

## 7. Testing Plan

### `tests/test_dashboard.py` (engine — plotly assertions removed)

Unchanged engine tests: payload extraction, gate status, capital reconcile, similarity matrix, determinism, missing-asset degradation. Deleted: every `from plotly.colors import ...` import, `build_leaderboard_bar_chart`/`build_drawdown_chart`/`build_similarity_matrix_chart` figure-object assertions, `multimetric_chart_html` block tests, `Plotly.newPlot(` counts, `plotly-engine-embed` marker, "size unbounded" assertions, `_build_html` figure-injection tests (moved/replaced below).

### `tests/test_dashboard_ui.py` (new — presentation)

1. **Geometry reference math** (`charts.data_to_svg_path` / `svg_area_path`):
   - Exact path strings for known inputs; `N=1` and `N=2` edge cases.
   - **Vertical inversion:** a larger value yields a *smaller* SVG `y` (returns rise on screen).
   - **Zero-span guard:** flat input (all-equal values) still yields a finite, well-formed path (never `NaN`/`Inf`).
   - **Area closure:** `svg_area_path` ends with baseline + `Z` (closed polygon).
   - Empty input → `""`.
   - `cumulative_series` parity with `nmr.dashboard._cumulative_from_standard`; `drawdown_series` peak-trough correctness; determinism across calls.
   - Global-y-range helper: min/max computed across all series of a metric.
2. **Payload contract:** `build_dashboard_payload` emits the exact schema; **metric-first parity** — payload `metrics` keys equal the engine's `extract_multimetric_timeseries` metric keys with per-model entries intact; leaderboard/similarity/hurdle/ensemble fields present; sorted-key JSON round-trips through the data node.
3. **HTML compiler:** `_build_html`/`generate_dashboard()` output contains the five section headings, `id="dashboard-data"`, `class="badge"` pills, `gate-fail` tinting, group headers; escapes hostile strings (`<script>`, `"><img onerror>`); byte-deterministic across two calls; compiles when invoked from a different CWD (asset anchoring).
4. **Artifact contract (hard gate):** `generate_dashboard()` writes the file; `size < 100 KB`; zero occurrences of `plotly` (case-insensitive); no `<script src=`; `id="dashboard-data"` present; report still compiles with an empty registry and with missing v5.3 assets.

### `tests/test_scripts.py`

Streamlit pure-helper tests unchanged; add the no-plotly-import guard for `dashboard_ui/app.py`.

## 8. Docs & Hygiene (same-commit)

- `requirements.txt`: remove `plotly==6.6.0`.
- **Plotly removal audit (P1):** after the rewrite, `grep -ri "plotly"` across the tree — the string may appear only in historical `docs/superpowers/` specs; **zero** imports or references in `dashboard_ui/`, `tests/`, `configs/`, and root scripts. (Pre-rewrite audit: references exist only in `tests/test_dashboard.py`, `dashboard_ui/{charts,app,report}.py`, and `requirements.txt` — all rewritten by this work; nothing in `tests/test_parity.py`, `tests/test_scripts.py`, or config files.)
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

1. `dashboard_ui/charts.py` — delete Plotly builders; add `data_to_svg_path`, `svg_area_path`, `cumulative_series`, `drawdown_series`, global-y-range helper, `build_dashboard_payload` (metric-first); re-export in `dashboard_ui/__init__.py`.
2. `dashboard_ui/static/style.css` — full design system.
3. `dashboard_ui/static/app.js` — controller + `dataToSvgPath` + area closure + grid lines + crosshair/tooltip.
4. `dashboard_ui/static/layout.html` + `dashboard_ui/report.py` — compiler rewrite; `dashboard-data` node; drop `plotly.io`/`get_plotlyjs`.
5. `dashboard_ui/app.py` — native Streamlit rewrite; drop `plotly.express`.
6. `tests/test_dashboard_ui.py` (new) + prune `tests/test_dashboard.py` + `tests/test_scripts.py` guard.
7. `requirements.txt`, `AGENTS.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md` — same-commit sync + plotly audit; regenerate `artifacts/dashboard.html`.
