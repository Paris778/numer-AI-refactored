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
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$", re.MULTILINE)
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
        rel = path.relative_to(DOCS_ROOT)
        # Exempt "paths under" each COVERAGE_EXEMPT_DIRS entry (entries may span
        # multiple path components, e.g. "99-archive/raw-source").
        if any(
            tuple(exempt.split("/")) == rel.parts[: len(exempt.split("/"))]
            for exempt in COVERAGE_EXEMPT_DIRS
        ):
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
