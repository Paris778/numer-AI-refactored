# Docs Hygiene Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make documentation hygiene a CI-enforced gate: a `tests/test_docs_hygiene.py` drift guard (links, anchors, test-count claims, AGENTS byte budget, docs coverage, stale references) plus AGENTS.md first-session onboarding and a DOCS_README trim.

**Architecture:** One pytest module in `tests/` (stdlib only), riding the existing `.github/workflows/ci.yml` (`python -m pytest -q`) with zero CI changes. Six focused checks over the five "nav docs" (AGENTS, ARCHITECTURE, README, CONTRIBUTING, `docs/DOCS_README.md`) plus the evaluation bible. TDD at suite level: the module is written first; it proves current drift (stale test counts, one unmapped notebook), then the docs are fixed to green.

**Tech Stack:** Python stdlib (`pathlib`, `re`, `subprocess`, `sys`, `unicodedata`), pytest. No new dependencies.

## Global Constraints

- **Pytest is the sole gate.** The hygiene checks are pytest tests; no CI workflow changes, no standalone scripts.
- **Stdlib only.** No new third-party dependencies (AGENTS.md absolute prohibition).
- **No business logic outside `nmr/`.** The new module is a test file; it imports only `REPO_ROOT` from `nmr.config` (same pattern as `tests/conftest.py`).
- **Determinism.** Pure file reads + one `pytest --collect-only` subprocess; no wall-clock.
- **Spec:** `docs/superpowers/specs/2026-08-06-docs-hygiene-suite-design.md` is the design of record; if a check in this plan differs from the spec, this plan wins for execution and the spec must be updated in the same commit.
- **Git:** no `git commit` without explicit user confirmation — ask before each commit step.

---

### Task 1: Create `tests/test_docs_hygiene.py` (all six checks + helpers)

**Files:**
- Create: `tests/test_docs_hygiene.py`

**Interfaces:**
- Consumes: `nmr.config.REPO_ROOT` (already exported; used by `tests/conftest.py`).
- Produces: module-level constants `NAV_DOCS`, `ANCHOR_DOCS`, `TEST_COUNT_DOCS`, `AGENTS_BUDGET_BYTES`, `DOCS_ROOT`, `COVERAGE_EXEMPT_DIRS`, `STALE_REFERENCES`, and helpers `_gh_slug`, `_headings`, `_markdown_targets`, `_coverage_evidence`, `_collected_test_count` — consumed by Tasks 2–3 (count fix, coverage fix) to verify green.

- [ ] **Step 1: Write the module**

```python
"""Docs hygiene suite — CI-enforced drift guard for the golden docs and knowledge base.

Spec: docs/superpowers/specs/2026-08-06-docs-hygiene-suite-design.md

Checks:
  T1  relative markdown links in the nav docs resolve to existing files
  T2  #fragment anchors resolve to headings (GitHub slug rules)
  T3  hardcoded "N tests" claims match `pytest --collect-only`
  T4  AGENTS.md stays within its 32 KiB byte budget
  T5  every knowledge file under docs/ is referenced from a nav doc
  T6  no stale references to renamed/deleted files
"""

from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from pathlib import Path

from nmr.config import REPO_ROOT

# --- constants ---------------------------------------------------------------

NAV_DOCS = (
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "ARCHITECTURE.md",
    REPO_ROOT / "README.md",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "docs" / "DOCS_README.md",
)
# The evaluation bible carries a 15-entry internal TOC that must not rot.
ANCHOR_DOCS = NAV_DOCS + (REPO_ROOT / "docs" / "06-evaluation" / "evaluation-suite-bible.md",)
TEST_COUNT_DOCS = (
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "README.md",
    REPO_ROOT / "CONTRIBUTING.md",
)
AGENTS_BUDGET_BYTES = 32768
DOCS_ROOT = REPO_ROOT / "docs"
COVERAGE_EXEMPT_DIRS = ("superpowers", "99-archive/raw-source")
# Files renamed or deleted in the 2026-08-06 docs trim. Exact paths only, so
# legitimate prose like "merged with former neural-networks.md" is untouched.
STALE_REFERENCES = (
    "docs/README.md",
    "docs/04-research/the-state-of-the-art.md",
    "docs/04-research/neural-networks.md",
    "docs/99-archive/general-ml-cookbook.md",
)
DOCS_TOPDIRS = (
    "01-canon", "02-strategy", "03-reference",
    "04-research", "05-notebooks", "06-evaluation", "99-archive",
)
_EXTERNAL = ("http://", "https://", "mailto:")
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
BACKTICK_TOKEN = re.compile(r"`([^`]*)`")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
TEST_COUNT_CLAIM = re.compile(r"\b(\d+)[-\s]tests?\b")


# --- helpers -----------------------------------------------------------------

def _gh_slug(heading: str) -> str:
    """GitHub-compatible anchor slug: NFKD, lowercase, strip non-word chars, spaces -> '-'."""
    text = unicodedata.normalize("NFKD", heading).lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return text.replace(" ", "-")


def _headings(path: Path) -> set[str]:
    return {_gh_slug(m.group(1)) for m in HEADING.finditer(path.read_text(encoding="utf-8"))}


def _markdown_targets(path: Path) -> list[tuple[str, int]]:
    """(target, line_no) for every markdown link target in path."""
    hits: list[tuple[str, int]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for m in MARKDOWN_LINK.finditer(line):
            hits.append((m.group(1), lineno))
    return hits


# --- T1: relative links resolve ----------------------------------------------

def _check_links() -> list[str]:
    problems: list[str] = []
    for doc in NAV_DOCS:
        for target, lineno in _markdown_targets(doc):
            if target.startswith(_EXTERNAL) or target.startswith("#"):
                continue
            rel = target.split("#", 1)[0]
            if not (doc.parent / rel).exists():
                problems.append(f"{doc.name}:{lineno}: link target does not exist: {target!r}")
    return problems


def test_golden_doc_relative_links_resolve():
    problems = _check_links()
    assert not problems, "Broken relative links:\n" + "\n".join(problems)


# --- T2: anchors resolve ------------------------------------------------------

def _check_anchors() -> list[str]:
    problems: list[str] = []
    for doc in ANCHOR_DOCS:
        doc_headings: set[str] | None = None
        for target, lineno in _markdown_targets(doc):
            if "#" not in target:
                continue
            file_part, frag = target.split("#", 1)
            if not file_part:  # bare fragment -> same file
                if doc_headings is None:
                    doc_headings = _headings(doc)
                if frag not in doc_headings:
                    problems.append(f"{doc.name}:{lineno}: anchor not found: #{frag}")
                continue
            if file_part.startswith(_EXTERNAL):
                continue
            target_path = doc.parent / file_part
            if not target_path.exists():
                problems.append(f"{doc.name}:{lineno}: anchor target file missing: {file_part!r}")
                continue
            if frag not in _headings(target_path):
                problems.append(f"{doc.name}:{lineno}: anchor not found in {file_part}: #{frag}")
    return problems


def test_golden_doc_anchors_resolve():
    problems = _check_anchors()
    assert not problems, "Broken anchors:\n" + "\n".join(problems)


# --- T3: test-count claims match the collected suite --------------------------

def _collected_test_count() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True, text=True, check=False,
    )
    output = proc.stdout + proc.stderr
    counts = [int(m) for m in re.findall(r"(\d+) tests? (?:collected|passed|failed|error)", output)]
    if not counts:
        raise AssertionError(f"could not parse pytest --collect-only output: {output[-500:]!r}")
    return counts[-1]


def test_docs_test_count_matches_suite():
    collected = _collected_test_count()
    problems: list[str] = []
    for doc in TEST_COUNT_DOCS:
        text = doc.read_text(encoding="utf-8")
        for m in TEST_COUNT_CLAIM.finditer(text):
            if int(m.group(1)) != collected:
                problems.append(f"{doc.name}: claim {m.group(0)!r} != collected {collected}")
    assert not problems, "Stale test-count claims (update the docs):\n" + "\n".join(problems)


# --- T4: AGENTS.md byte budget -----------------------------------------------

def test_agents_md_within_byte_budget():
    size = (REPO_ROOT / "AGENTS.md").stat().st_size
    assert size <= AGENTS_BUDGET_BYTES, (
        f"AGENTS.md is {size} B; budget is {AGENTS_BUDGET_BYTES} B"
    )


# --- T5: docs coverage -------------------------------------------------------

def _coverage_evidence() -> set[Path]:
    """Every file reachable from the nav docs via markdown links or backticked paths."""
    evidence: set[Path] = set()
    for doc in NAV_DOCS:
        for target, _ in _markdown_targets(doc):
            if target.startswith(_EXTERNAL) or target.startswith("#"):
                continue
            rel = target.split("#", 1)[0]
            if rel:
                evidence.add((doc.parent / rel).resolve())
        for token in BACKTICK_TOKEN.findall(doc.read_text(encoding="utf-8")):
            token = token.strip()
            if token.startswith("docs/"):
                evidence.add((REPO_ROOT / token).resolve())
            elif any(token.startswith(d + "/") for d in DOCS_TOPDIRS):
                evidence.add((DOCS_ROOT / token).resolve())
    return evidence


def test_docs_coverage_complete():
    evidence = _coverage_evidence()
    uncovered: list[Path] = []
    for path in sorted(DOCS_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in (".md", ".ipynb"):
            continue
        if any(part in COVERAGE_EXEMPT_DIRS for part in path.relative_to(DOCS_ROOT).parts):
            continue
        if path.resolve() not in evidence:
            uncovered.append(path)
    assert not uncovered, (
        "Docs files not reachable from the nav docs (add to docs/DOCS_README.md §4/§5 "
        "or the AGENTS.md knowledge map):\n" + "\n".join(str(p) for p in uncovered[:10])
    )


# --- T6: stale references ----------------------------------------------------

def test_no_stale_doc_references():
    files = NAV_DOCS + (REPO_ROOT / "docs" / "06-evaluation" / "evaluation-suite-bible.md",)
    problems: list[str] = []
    for doc in files:
        text = doc.read_text(encoding="utf-8")
        for stale in STALE_REFERENCES:
            if stale in text:
                problems.append(f"{doc.name}: stale reference to {stale!r}")
    assert not problems, "Stale doc references:\n" + "\n".join(problems)
```

- [ ] **Step 2: Run the module and observe expected failures**

Run: `.\.venv\Scripts\python -m pytest tests/test_docs_hygiene.py -q`
Expected: **T3 FAILS** ("301 != 307") and **T5 FAILS** (lists `docs/05-notebooks/example-model-sunshine.ipynb` as uncovered). T1, T2, T4, T6 PASS. If any of T1/T2/T4/T6 fails, fix the underlying doc problem (broken link/anchor, budget, stale string) per the failure message before continuing — do NOT weaken the test.

- [ ] **Step 3: Commit (requires user confirmation)**

```bash
git add tests/test_docs_hygiene.py
git commit -m "test: add docs hygiene drift guard (links, anchors, counts, budget, coverage, stale refs)"
```
Ask the user for explicit confirmation before running these commands.

---

### Task 2: Fix the stale test-count claims (T3)

**Files:**
- Modify: `AGENTS.md` (§1 tech stack, §7 verification gates)
- Modify: `README.md` (stack line, project-tree comment)
- Modify: `CONTRIBUTING.md` (full-suite line)

**Interfaces:**
- Consumes: `_collected_test_count()` semantics from Task 1 — the collected count is now **307** (301 + 6 new tests).
- Produces: T3 green; the three docs' counts now self-consistent.

- [ ] **Step 1: Read and update the five count locations**

Run `Read AGENTS.md`, `Read README.md`, `Read CONTRIBUTING.md` to locate the exact strings, then apply `Edit` per file (one `Edit` per file per message):

1. `AGENTS.md`: `Test: pytest (301 tests).` → `Test: pytest (307 tests).`
2. `AGENTS.md`: `# full 301-test suite` → `# full 307-test suite`
3. `README.md`: `pytest (301 tests)` → `pytest (307 tests)`
4. `README.md`: `301 unit / parity / determinism / real-data tests` → `307 unit / parity / determinism / real-data tests`
5. `CONTRIBUTING.md`: `(301 tests; from the repo root)` → `(307 tests; from the repo root)`

- [ ] **Step 2: Verify T3 passes**

Run: `.\.venv\Scripts\python -m pytest tests/test_docs_hygiene.py::test_docs_test_count_matches_suite -q`
Expected: PASS.

- [ ] **Step 3: Commit (requires user confirmation)**

```bash
git add AGENTS.md README.md CONTRIBUTING.md
git commit -m "docs: sync hardcoded test counts to 307 (docs hygiene suite)"
```
Ask the user for explicit confirmation before running these commands.

---

### Task 3: Fix the coverage gap + add agent onboarding (T5 + AGENTS §6)

**Files:**
- Modify: `docs/DOCS_README.md` (§4 Tier T1 — add the sunshine notebook)
- Modify: `AGENTS.md` (§6 knowledge map — first-session orientation + executable-spec pointer)

**Interfaces:**
- Consumes: T5's evidence rule (backtick tokens count as coverage) from Task 1.
- Produces: T5 green; AGENTS.md gains the onboarding block (budget still enforced by T4).

- [ ] **Step 1: Map the sunshine notebook in DOCS_README §4**

`Read docs/DOCS_README.md` to confirm the Tier T1 list, then `Edit`:

old:
```
- `05-notebooks/1_hello_numerai.ipynb`
- `05-notebooks/2_feature_neutralization.ipynb`
- `05-notebooks/3_target_ensemble.ipynb`
```
new:
```
- `05-notebooks/1_hello_numerai.ipynb`
- `05-notebooks/2_feature_neutralization.ipynb`
- `05-notebooks/3_target_ensemble.ipynb`
- `05-notebooks/example-model-sunshine.ipynb` (community example: multi-target + 25% neutralization + model upload)
```

- [ ] **Step 2: Add the first-session orientation block to AGENTS.md §6**

`Read AGENTS.md` to locate the paragraph `Never invent a \`numerai_tools\` / \`numerapi\` signature — open the installed source: ...` (end of the knowledge-map section), then `Edit` — append after it:

```
**First-session orientation (10 minutes):**

1. `.\.venv\Scripts\python -m pytest -q` — establish the green baseline (the test count is CI-enforced against this file's claims).
2. `nmr/__init__.py` — the public API surface (imports + `__all__`); nothing outside it is public.
3. `configs/first_model.yaml` — the current competitive config; `configs/example.yaml` — annotated schema.
4. `ARCHITECTURE.md` §1 (pipeline diagram) and §3 (module dependency graph) — the system map.

**The tests are the executable spec.** Before touching a metric or formula, read `tests/test_parity.py` + `tests/test_risk_parity.py`; before touching scorecards, `tests/test_scorecard.py`; before benchmark gates, `tests/test_benchmark_*.py`. The tests encode the contracts prose can only summarize.
```

- [ ] **Step 3: Verify T5 passes and AGENTS stays in budget**

Run: `.\.venv\Scripts\python -m pytest tests/test_docs_hygiene.py::test_docs_coverage_complete tests/test_docs_hygiene.py::test_agents_md_within_byte_budget -q`
Expected: both PASS (AGENTS should be ~20.6 KB ≤ 32 768 B).

- [ ] **Step 4: Commit (requires user confirmation)**

```bash
git add docs/DOCS_README.md AGENTS.md
git commit -m "docs: map sunshine notebook; add first-session orientation and executable-spec pointers"
```
Ask the user for explicit confirmation before running these commands.

---

### Task 4: DOCS_README §7 trim

**Files:**
- Modify: `docs/DOCS_README.md` (§7 Minimal Traversal Recipes)

**Interfaces:**
- Consumes: nothing (independent edit).
- Produces: ~30-line §7 compressed to ~7 lines with zero information loss; T5 still green (backtick tokens retained).

- [ ] **Step 1: Read and replace §7**

`Read docs/DOCS_README.md` (the `## 7) Minimal Traversal Recipes` section through the end of recipe C), then `Edit`:

old:
```
## 7) Minimal Traversal Recipes

### A) Scoring comprehension only

Read:

1. `01-canon/scoring/00-definitions.md`
2. `01-canon/scoring/01-correlation.md`
3. `01-canon/scoring/02-mmc-bmc.md`
4. `01-canon/scoring/03-fnc.md`
5. `01-canon/staking.md`

### B) Data-to-submission full lifecycle

Read:

1. `01-canon/data.md`
2. `01-canon/models.md`
3. `01-canon/submissions.md`
4. `01-canon/staking.md`
5. `03-reference/numerapi.md`

### C) Robust modeling intuition

Read:

1. `02-strategy/strategy-bible.md`
2. `02-strategy/community-wisdom.md`
3. `02-strategy/why-it-works.md`
4. `04-research/research-program.md`
```
new:
```
## 7) Minimal Traversal Recipes

Subsets of the §1 fast-start order, for focused goals:

- **Scoring comprehension:** `01-canon/scoring/00-definitions.md` → `01-canon/scoring/01-correlation.md` → `01-canon/scoring/02-mmc-bmc.md` → `01-canon/scoring/03-fnc.md` → `01-canon/staking.md`
- **Data-to-submission lifecycle:** `01-canon/data.md` → `01-canon/models.md` → `01-canon/submissions.md` → `01-canon/staking.md` → `03-reference/numerapi.md`
- **Robust modeling intuition:** `02-strategy/strategy-bible.md` → `02-strategy/community-wisdom.md` → `02-strategy/why-it-works.md` → `04-research/research-program.md`
```

- [ ] **Step 2: Verify the suite still passes**

Run: `.\.venv\Scripts\python -m pytest tests/test_docs_hygiene.py -q`
Expected: all 6 PASS.

- [ ] **Step 3: Commit (requires user confirmation)**

```bash
git add docs/DOCS_README.md
git commit -m "docs: compress DOCS_README traversal recipes into §1 subsets"
```
Ask the user for explicit confirmation before running these commands.

---

### Task 5: Full-suite verification and final review

**Files:**
- None (verification only).

**Interfaces:**
- Consumes: Tasks 1–4 outputs.

- [ ] **Step 1: Full suite**

Run: `.\.venv\Scripts\python -m pytest -q`
Expected: **307 passed**, 0 failures. (T3 inside the suite verifies the docs counts against `--collect-only` — including its own subprocess.)

- [ ] **Step 2: Byte budget + scope**

Run:
```bash
wc -c AGENTS.md
git status --short
```
Expected: AGENTS.md ≤ 32768; status shows only: `tests/test_docs_hygiene.py` (new), `AGENTS.md`, `README.md`, `CONTRIBUTING.md`, `docs/DOCS_README.md`, plus the spec/plan under `docs/superpowers/`.

- [ ] **Step 3: Docs-consistency spot check (spec vs plan)**

Confirm `docs/superpowers/specs/2026-08-06-docs-hygiene-suite-design.md` reflects the final behavior (T2 file set incl. eval bible; T5 evidence = links ∪ backticks; sunshine map entry). If any wording drifted during execution, update the spec in the same commit.

- [ ] **Step 4: Commit (requires user confirmation)**

```bash
git add tests/test_docs_hygiene.py AGENTS.md README.md CONTRIBUTING.md docs/DOCS_README.md docs/superpowers/specs/2026-08-06-docs-hygiene-suite-design.md docs/superpowers/plans/2026-08-06-docs-hygiene-suite.md
git commit -m "docs: enforce doc hygiene via CI test suite; add agent onboarding; trim DOCS_README"
```
Ask the user for explicit confirmation before running these commands.
