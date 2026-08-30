"""Docs hygiene suite — CI-enforced drift guard for the golden docs and knowledge base.

Checks:
  T1  relative markdown links in the nav docs resolve to existing files
  T2  #fragment anchors resolve to headings (GitHub slug rules)
    T3  core docs contain no volatile hardcoded test-count claims
  T4  AGENTS.md stays within its 32 KiB byte budget
  T5  every knowledge file under docs/ is referenced from a nav doc
  T6  no stale references to renamed/deleted files
    T7  repository-root Markdown is explicitly allowlisted
    T8  active design records are indexed and completed plans stay removed
    T9  ARCHITECTURE.md covers every module and control-plane script

Checks scan whole files including fenced code blocks — extra phantom anchors
can only make T2 more lenient, never false-fail; this is intentional.
"""

from __future__ import annotations

import re
from pathlib import Path

from nmr.config import REPO_ROOT

# --- constants ---------------------------------------------------------------

NAV_DOCS = (
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "ARCHITECTURE.md",
    REPO_ROOT / "CODEBASE.md",
    REPO_ROOT / "README.md",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "docs" / "DOCS_README.md",
    REPO_ROOT / "docs" / "01-canon" / "NUMERAI-CANON-DOCS-README.md",
)
# The evaluation bible carries a 15-entry internal TOC that must not rot.
ANCHOR_DOCS = NAV_DOCS + (
    REPO_ROOT / "docs" / "06-evaluation" / "evaluation-suite-bible.md",
)
TEST_COUNT_DOCS = (
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "README.md",
    REPO_ROOT / "CONTRIBUTING.md",
)
OPERATIONAL_OWNER_DOCS = (
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "ARCHITECTURE.md",
    REPO_ROOT / "CONTRIBUTING.md",
)
AGENTS_BUDGET_BYTES = 32768
DOCS_ROOT = REPO_ROOT / "docs"
COVERAGE_EXEMPT_DIRS = ("99-archive/raw-source",)
ACTIVE_DESIGN_ROOT = DOCS_ROOT / "superpowers"
ACTIVE_DESIGN_INDEX = ACTIVE_DESIGN_ROOT / "README.md"
ACTIVE_DESIGN_SPECS = ACTIVE_DESIGN_ROOT / "specs"
COMPLETED_PLAN_ROOT = ACTIVE_DESIGN_ROOT / "plans"
# Files renamed or deleted in the 2026-08-06 docs trim. Exact paths only, so
# legitimate prose like "merged with former neural-networks.md" is untouched.
STALE_REFERENCES = (
    "docs/README.md",
    "docs/01-canon/overview.md",
    "docs/01-canon/data.md",
    "docs/01-canon/models.md",
    "docs/01-canon/submissions.md",
    "docs/01-canon/staking.md",
    "docs/01-canon/scoring/00-definitions.md",
    "docs/01-canon/scoring/01-correlation.md",
    "docs/01-canon/scoring/02-mmc-bmc.md",
    "docs/01-canon/scoring/03-fnc.md",
    "01-canon/staking.md",
    "01-canon/scoring/00-definitions.md",
    "01-canon/scoring/01-correlation.md",
    "01-canon/scoring/02-mmc-bmc.md",
    "01-canon/scoring/03-fnc.md",
    "../01-canon/staking.md",
    "../01-canon/scoring/00-definitions.md",
    "../01-canon/scoring/01-correlation.md",
    "../01-canon/scoring/02-mmc-bmc.md",
    "../01-canon/scoring/03-fnc.md",
    "docs/04-research/the-state-of-the-art.md",
    "docs/04-research/neural-networks.md",
    "docs/99-archive/general-ml-cookbook.md",
)
DOCS_TOPDIRS = (
    "01-canon",
    "02-strategy",
    "03-reference",
    "04-research",
    "05-notebooks",
    "06-evaluation",
    "99-archive",
    "superpowers",
)
ROOT_MARKDOWN_ALLOWLIST = {
    "AGENTS.md",
    "ARCHITECTURE.md",
    "CODEBASE.md",
    "CONTRIBUTING.md",
    "README.md",
    "TODO-NOTES.md",
}
_EXTERNAL = ("http://", "https://", "mailto:")
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
BACKTICK_TOKEN = re.compile(r"`([^`]*)`")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$", re.MULTILINE)
TEST_COUNT_CLAIM = re.compile(r"(?<![\d.])(\d+)[ \t-]tests?\b")
OBSOLETE_COUNT_POLICY = re.compile(
    r"test-count claims?|count must match|net count change synced",
    re.IGNORECASE,
)


# --- helpers -----------------------------------------------------------------


# GitHub keeps Unicode letters (do not NFKD); subscript digits like `₅` (category No) are not matched by Python `\w` and stay approximate — acceptable, nothing links to them today.
def _gh_slug(heading: str) -> str:
    """GitHub-compatible anchor slug: lowercase, strip non-word chars, spaces -> '-'."""
    text = heading.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return text.replace(" ", "-")


def _headings(path: Path) -> set[str]:
    return {
        _gh_slug(m.group(1)) for m in HEADING.finditer(path.read_text(encoding="utf-8"))
    }


def _markdown_targets(path: Path) -> list[tuple[str, int]]:
    """(target, line_no) for every markdown link target in path."""
    hits: list[tuple[str, int]] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        for m in MARKDOWN_LINK.finditer(line):
            hits.append((m.group(1), lineno))
    return hits


# --- T1: relative links resolve ----------------------------------------------


def _maintained_markdown() -> list[Path]:
    return [
        path
        for path in sorted(DOCS_ROOT.rglob("*.md"))
        if "raw-source" not in path.parts
    ]


def _check_links(files: tuple[Path, ...] | list[Path] = NAV_DOCS) -> list[str]:
    problems: list[str] = []
    for doc in files:
        for target, lineno in _markdown_targets(doc):
            if target.startswith(_EXTERNAL) or target.startswith("#"):
                continue
            rel = target.split("#", 1)[0]
            if not (doc.parent / rel).exists():
                problems.append(
                    f"{doc.name}:{lineno}: link target does not exist: {target!r}"
                )
    return problems


def test_golden_doc_relative_links_resolve():
    problems = _check_links()
    assert not problems, "Broken relative links:\n" + "\n".join(problems)


def test_all_maintained_markdown_links_resolve():
    problems = _check_links(_maintained_markdown())
    assert not problems, "Broken maintained-doc links:\n" + "\n".join(problems)


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
                problems.append(
                    f"{doc.name}:{lineno}: anchor target file missing: {file_part!r}"
                )
                continue
            if frag not in _headings(target_path):
                problems.append(
                    f"{doc.name}:{lineno}: anchor not found in {file_part}: #{frag}"
                )
    return problems


def test_golden_doc_anchors_resolve():
    problems = _check_anchors()
    assert not problems, "Broken anchors:\n" + "\n".join(problems)


def test_all_maintained_markdown_anchors_resolve():
    problems: list[str] = []
    for doc in _maintained_markdown():
        doc_headings: set[str] | None = None
        for target, lineno in _markdown_targets(doc):
            if "#" not in target:
                continue
            file_part, frag = target.split("#", 1)
            if file_part.startswith(_EXTERNAL):
                continue
            if not file_part:
                if doc_headings is None:
                    doc_headings = _headings(doc)
                headings = doc_headings
            else:
                target_path = doc.parent / file_part
                if not target_path.exists() or not target_path.is_file():
                    continue
                headings = _headings(target_path)
            if frag not in headings:
                problems.append(f"{doc.relative_to(REPO_ROOT)}:{lineno}: #{frag}")
    assert not problems, "Broken maintained-doc anchors:\n" + "\n".join(problems)


# --- T3: core docs contain no volatile test counts ---------------------------


def test_core_docs_do_not_hardcode_test_counts():
    problems: list[str] = []
    for doc in TEST_COUNT_DOCS:
        text = doc.read_text(encoding="utf-8")
        for m in TEST_COUNT_CLAIM.finditer(text):
            problems.append(f"{doc.name}: volatile claim {m.group(0)!r}")
    assert not problems, (
        "Hardcoded test counts drift whenever the suite changes; report executed "
        "counts in review output instead:\n" + "\n".join(problems)
    )


def test_active_designs_do_not_require_test_count_sync():
    problems: list[str] = []
    for spec in sorted(ACTIVE_DESIGN_SPECS.glob("*.md")):
        for lineno, line in enumerate(
            spec.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if OBSOLETE_COUNT_POLICY.search(line):
                problems.append(f"{spec.name}:{lineno}: {line.strip()}")
    assert not problems, (
        "Active designs must report executed suite counts in review output, not "
        "require synchronized count claims:\n" + "\n".join(problems)
    )


# --- T4: AGENTS.md byte budget -----------------------------------------------


def test_agents_md_within_byte_budget():
    size = (REPO_ROOT / "AGENTS.md").stat().st_size
    assert (
        size <= AGENTS_BUDGET_BYTES
    ), f"AGENTS.md is {size} B; budget is {AGENTS_BUDGET_BYTES} B"


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
    files = NAV_DOCS + (
        REPO_ROOT / "docs" / "06-evaluation" / "evaluation-suite-bible.md",
    )
    problems: list[str] = []
    for doc in files:
        text = doc.read_text(encoding="utf-8")
        for stale in STALE_REFERENCES:
            if stale in text:
                problems.append(f"{doc.name}: stale reference to {stale!r}")
    assert not problems, "Stale doc references:\n" + "\n".join(problems)


# --- T7: root documentation stays intentional -------------------------------


def test_root_markdown_is_allowlisted():
    unexpected = sorted(
        path.name
        for path in REPO_ROOT.glob("*.md")
        if path.name not in ROOT_MARKDOWN_ALLOWLIST
    )
    assert not unexpected, (
        "Unexpected root Markdown (move maintained knowledge into docs/ or remove "
        "superseded reports):\n" + "\n".join(unexpected)
    )


# --- T8: active design records stay intentional -----------------------------


def test_active_design_records_are_indexed_and_declare_status():
    index_text = ACTIVE_DESIGN_INDEX.read_text(encoding="utf-8")
    problems: list[str] = []
    for spec in sorted(ACTIVE_DESIGN_SPECS.glob("*.md")):
        if spec.name not in index_text:
            problems.append(f"not indexed: {spec.relative_to(REPO_ROOT)}")
        header = "\n".join(spec.read_text(encoding="utf-8").splitlines()[:10])
        if "Status:" not in header:
            problems.append(
                f"status missing from header: {spec.relative_to(REPO_ROOT)}"
            )
    assert not problems, "Active design record problems:\n" + "\n".join(problems)


def test_completed_implementation_plans_are_not_active_docs():
    plans = sorted(COMPLETED_PLAN_ROOT.glob("*.md"))
    assert not plans, (
        "Completed implementation plans belong in review history, not the active "
        "knowledge base:\n" + "\n".join(str(path) for path in plans)
    )


def test_operational_owner_docs_do_not_depend_on_archive():
    problems: list[str] = []
    for doc in OPERATIONAL_OWNER_DOCS:
        for target, lineno in _markdown_targets(doc):
            if "99-archive" in target:
                problems.append(f"{doc.name}:{lineno}: archive dependency {target!r}")
    assert not problems, (
        "Operational owners cannot use archived provenance as current authority:\n"
        + "\n".join(problems)
    )


# --- T9: architecture SSOT covers every module and control-plane script ------


def _architecture_text() -> str:
    return (REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")


def test_architecture_documents_every_module():
    text = _architecture_text()
    missing = sorted(
        p.name
        for p in (REPO_ROOT / "nmr").glob("*.py")
        if p.stem != "__init__" and p.name not in text
    )
    assert not missing, (
        "nmr modules absent from ARCHITECTURE.md (add to the module dependency "
        "graph in section 3):\n" + "\n".join(missing)
    )


def test_architecture_documents_every_control_plane_script():
    text = _architecture_text()
    missing = sorted(p.name for p in REPO_ROOT.glob("*.py") if p.name not in text)
    assert not missing, (
        "Root control-plane scripts absent from ARCHITECTURE.md (add to the "
        "section O scripts table):\n" + "\n".join(missing)
    )
