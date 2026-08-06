# Design: Docs System 2.0 — Automated Drift Guard, Agent Onboarding, Trim

- **Date:** 2026-08-06
- **Status:** Approved by user (via brainstorming skill)
- **Scope:** Documentation-hygiene automation + targeted doc edits. Adds one test module; no code changes to `nmr/`, no CI changes.

## 1. Context & Goal

The docs audit (2026-08-06) rated the documentation system 80/100. The content and structure are strong; the gap is the **machinery that keeps docs true**. Known drift already occurred: the hardcoded test count in AGENTS/README/CONTRIBUTING moved 203 → 239 → 301 with manual updates; `docs/README.md` was renamed and its references had to be chased by hand; deleted/merged research files required manual ripple sweeps.

**Goal (the remaining 20 points):** make doc hygiene a CI-enforced gate rather than a discipline. Deliverables:

1. `tests/test_docs_hygiene.py` — six automated checks (links, anchors, test-count consistency, AGENTS byte budget, docs coverage, stale references).
2. AGENTS.md §6 first-session orientation block + "tests are the executable spec" pointer.
3. DOCS_README §7 trim (explicit subset-of-§1 framing).

## 2. Principles (non-negotiable)

- **Pytest is the sole gate.** The hygiene checks are pytest tests in `tests/`; they ride the existing `.github/workflows/ci.yml` (`python -m pytest -q`) with zero CI changes.
- **Stdlib only.** `pathlib`, `re`, `subprocess`. No new dependencies (AGENTS.md absolute prohibition).
- **Tested boundary intact.** The new module is a test file, not business logic; it does not import `nmr` business modules except `REPO_ROOT` (same pattern as `tests/conftest.py`).
- **Determinism.** Pure file reads + one `--collect-only` subprocess; no wall-clock, no absolute-path-sensitive output.
- **TDD.** Write failing tests first (they fail against current drift — e.g., before the count fix), then fix docs, then green.
- **SSOT / no duplication.** The checks cross-reference; the doc edits (onboarding block) do not duplicate content owned by README/CONTRIBUTING/ARCHITECTURE — they point.

## 3. Component 1 — `tests/test_docs_hygiene.py`

### 3.1 Module layout

```
tests/test_docs_hygiene.py
├── NAV_DOCS = (AGENTS.md, ARCHITECTURE.md, README.md, CONTRIBUTING.md, docs/DOCS_README.md)
├── TEST_COUNT_DOCS = (AGENTS.md, README.md, CONTRIBUTING.md)      # hardcode "N tests"
├── AGENTS_BUDGET_BYTES = 32768
├── DOCS_ROOT = REPO_ROOT / "docs"
├── COVERAGE_EXEMPT_DIRS = ("superpowers", "99-archive/raw-source")
├── STALE_REFERENCES = ("docs/README.md", "docs/04-research/the-state-of-the-art.md", "docs/04-research/neural-networks.md", "docs/99-archive/general-ml-cookbook.md")
├── helpers: _markdown_targets(path) -> list[(target, line_no)]    # markdown link extraction
├── helpers: _gh_slug(heading) -> str                              # GitHub-compatible anchor slug
├── helpers: _headings(path) -> set[str]                           # slugified headings of a file
└── helpers: _coverage_evidence() -> set[Path]                     # every file reachable from the nav docs via links or backticks
```

### 3.2 The six checks

**T1 `test_golden_doc_relative_links_resolve`**
For each file in `NAV_DOCS`: extract `[text](target)`; keep only targets that are relative paths (exclude `http://`, `https://`, `mailto:`, bare `#anchor`); for each, split off any `#fragment`, resolve the path against the containing doc's directory; assert `path.exists()` (fragment checked in T2). Failures report the doc, line, and raw target.

**T2 `test_golden_doc_anchors_resolve`**
File set: `NAV_DOCS` + `docs/06-evaluation/evaluation-suite-bible.md` (it carries a 15-entry internal TOC that must not rot). For each relative target with a `#fragment` (and each bare `#fragment`): resolve the file (or the current doc for bare fragments), slugify every ATX heading (lines starting with 1–6 `#`), assert the slug is in the set. Slugger rules (must match GitHub):

```
lowercase → keep only Unicode word chars (\w: letters/digits/underscore), hyphens, and spaces
→ replace spaces with '-'  (do NOT collapse repeated hyphens)
```

Code spans in headings: backticks stripped by the regex; the inner text stays (e.g. `` `nmr/config.py` `` → `nmrconfigpy`).
Section-number headings like `## 5) Full File Map` → `5-full-file-map`.
Fragments that are intentionally `§N`-style text references are not links and are unaffected.

**T3 `test_docs_test_count_matches_suite`**
Run `subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"])`, parse the trailing `N tests collected` (or the `N passed`/`N failed` equivalents) from stderr/stdout. For each file in `TEST_COUNT_DOCS`, regex `\b(\d+)\s+tests?\b`; assert every captured number equals the collected count. (Self-consistent: adding tests forces updating the docs in the same commit.)

**T4 `test_agents_md_within_byte_budget`**
`(REPO_ROOT / "AGENTS.md").stat().st_size <= AGENTS_BUDGET_BYTES` (32 768).

**T5 `test_docs_coverage_complete`**
Walk `DOCS_ROOT` recursively for `.md`/`.ipynb` files; exclude paths under `COVERAGE_EXEMPT_DIRS` (`superpowers/` = process artifacts; `99-archive/raw-source/` = provenance, covered at directory level by DOCS_README §6). Collect **coverage evidence** from all `NAV_DOCS` — two forms, because the knowledge map and DOCS_README tier/file-map tables reference files with backticks, not markdown links:
1. relative markdown link targets (excluding `http(s)://`, `mailto:`, bare `#fragment`);
2. backticked path tokens: `docs/…` (resolve against `REPO_ROOT`) or a bare numbered topdir (`01-canon/…`, …, `99-archive/…` — resolve against `DOCS_ROOT`); tokens are backtick-delimited so paths with spaces match.
Assert every non-exempt docs file resolves into the evidence set. Failure lists the first uncovered file(s) with the instruction "add it to the map in docs/DOCS_README.md §4/§5 or the AGENTS.md knowledge map."
Known current gap this check exists to catch: `docs/05-notebooks/example-model-sunshine.ipynb` is referenced only from `02-strategy/strategy-bible.md` prose (not a nav doc) — it must be added to DOCS_README §4 Tier T1.

**T6 `test_no_stale_doc_references`**
For each file in `NAV_DOCS` + `docs/06-evaluation/evaluation-suite-bible.md`: assert none of `STALE_REFERENCES` appears as a substring (prose or link). List is a module constant with a one-line comment (renamed `docs/README.md`; files deleted in the 2026-08-06 trim). Do NOT include `neural-networks` alone — legitimate merge notes reference it; the banned entries are the exact paths.

### 3.3 Edge cases & error handling

- Non-UTF-8 or binary docs files: not in scope (all docs are UTF-8 markdown).
- Links with spaces/percent-encoding: docs use plain spaces in one filename (`State-of-the-Art Deep Learning …`); assert on the raw literal path (match the file exactly as written). If a link uses URL-encoding, T1 fails loudly — keep docs canonical (no encoding).
- `pytest --collect-only` cost ~2–5 s: acceptable for one test.
- Windows vs CI (Ubuntu): `Path` handling is OS-agnostic; subprocess uses `sys.executable`. No path separators hardcoded in regexes beyond `/` inside markdown targets (converted via `Path`).

## 4. Component 2 — Agent onboarding (AGENTS.md §6)

Add, after the paragraph "Start with the agent reading order in `docs/DOCS_README.md` §1; the 15-minute version is §2–§3.", a compact block:

```
**First-session orientation (10 minutes):**

1. `.\.venv\Scripts\python -m pytest -q` — establish the green baseline (test count is CI-enforced against this file's claims).
2. `nmr/__init__.py` — the public API surface (imports + `__all__`); nothing outside it is public.
3. `configs/first_model.yaml` — the current competitive config; `configs/example.yaml` — annotated schema.
4. `ARCHITECTURE.md` §1 (pipeline diagram) and §3 (module dependency graph) — the system map.

**The tests are the executable spec.** Before touching a metric or formula, read `tests/test_parity.py` + `tests/test_risk_parity.py`; before touching scorecards, `tests/test_scorecard.py`; before benchmark gates, `tests/test_benchmark_*.py`. The tests encode the contracts prose can only summarize.
```

Budget check: AGENTS.md is ~20.0 KB; the block adds ~600 B → ~20.6 KB ≤ 32 768 (enforced by T4).

## 5. Component 3 — DOCS_README §7 trim

Current §7 "Minimal Traversal Recipes" lists recipes A (scoring), B (data-to-submission), C (modeling intuition) that partially restate the §1 reading order. Change: add an intro line making the relationship explicit and compress duplicated phrasing:

- New intro: "Subsets of the §1 fast-start order, for focused goals:"
- Recipe A: keep the 5-file list (scoring-only) — it is genuinely a subset.
- Recipe B: keep (data/submission lifecycle; pulls in `03-reference` not in §1).
- Recipe C: keep (strategy/research path).
- Remove the duplicated sentence fragments that repeat §1 item descriptions; each recipe keeps only its file list + one-line purpose.

Delivered: the plan's exact compact block — §7 compressed from 32 to 7 lines with zero information loss.

## 6. Verification (before sign-off)

1. `pytest tests/test_docs_hygiene.py -q` — all six new tests green; T3 drives the docs count update in the same commit.
2. Full `pytest -q` — green (≈ 301 + 6 = 307 tests); update the "N tests" claims in AGENTS/README/CONTRIBUTING to the new collected count (T3 enforces).
3. `wc -c AGENTS.md` ≤ 32 768 (T4 enforces).
4. `git status --short` — only: `tests/test_docs_hygiene.py` (new), `AGENTS.md`, `README.md`, `CONTRIBUTING.md` (counts), `docs/DOCS_README.md` (trim), spec/plan under `docs/superpowers/`.
5. CI: unchanged workflow; the suite now covers docs hygiene automatically.

## 7. Risks & Notes

- **Anchor slugger fidelity:** GitHub's slug rules are the reference; the slugger is tested against the repo's actual headings (TDD), and T2 asserts against them — any mismatch surfaces immediately as a failing test to fix, not silent rot.
- **Count-check brittleness:** `--collect-only` output formats vary by pytest version. Parse defensively (regex over the last lines for `\d+ tests? collected|passed|failed|error`).
- **Future docs additions:** T5 forces new knowledge files to be mapped (AGENTS map or DOCS_README §4/§5) in the same commit — by design.
- **Process artifacts:** `docs/superpowers/` stays exempt from coverage; specs/plans are dated records, not knowledge.
