# CI Coverage Floor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the measured ~90% coverage a *defended* number: CI fails when coverage regresses, using the same package-level coverage command that works locally — plus document the coverage-invocation hazard (dotted submodule `--cov` specs crash on py3.12) so no agent or CI job reintroduces it.

**Architecture:** Three small changes. (1) Pin the two missing dev tools (`pytest-cov`, `coverage`) in `requirements-dev.txt` at the exact versions installed in the venv. (2) Extend the existing CI test step (`.github/workflows/ci.yml`) with package-level coverage reporting, measure the CI numbers, then add a `scripts/coverage_gate.py` step that fails under per-module minimums for `nmr/promote.py`/`nmr/models.py` and a global floor bound to the CI-measured total (ratchet: floors only move up). (3) Document the working form and the dotted-spec root cause in the two docs that own test commands (`CONTRIBUTING.md`) and operational hazards (`AGENTS.md` §8), keeping the AGENTS.md size budget.

**Tech Stack:** GitHub Actions (ubuntu-latest, Python 3.12), pytest + pytest-cov 7.1.0 + coverage 7.13.5, ruff.

## Global Constraints

- **Exact-pinned dev deps** (repo convention): `pytest-cov==7.1.0`, `coverage==7.13.5` — the versions already installed in `.venv`. Do not upgrade anything else; pinning a new version is a deliberate act (AGENTS.md).
- **Package-level `--cov` specs only.** `--cov=nmr --cov=dashboard_ui`. **Never** `--cov=nmr.promote` or any dotted submodule: coverage 7.13.5 resolves dotted sources lazily via `find_spec('nmr.promote')` from inside a trace callback; that imports the parent package `nmr`, re-entering numpy's in-flight extension init → `ImportError: cannot load module more than once per process` (root-caused 2026-08-19: reproduced with `python -c "import coverage; c=coverage.Coverage(source=['nmr.promote']); c.start(); import nmr.config"`; the spy run showed `find_spec('nmr.promote')` firing mid-import; `source=['nmr']` is immune because resolving a top-level package never executes its `__init__`; `COVERAGE_CORE=pytrace` does NOT help).
- **CI stays the fast gate.** The coverage floor guards statement coverage of the suite CI can run; real-data verification remains the local receipt gate (`scripts/real_data_gate.py`) — do not change that division.
- **Floor calibration is honest.** Local full-suite coverage measured 90% *with* `data/v5.3/` present; CI skips the 12 data-gated tests and runs on Linux (slightly different ctypes branches). Set the initial floor at 88 and add a self-correcting step: after the first CI run, set floor = CI-measured total − 1 (floor bounded [85, 90]).
- **AGENTS.md size budget ≤ 32 KB.** It is 32,394 bytes today. After editing, `wc -c AGENTS.md` must stay ≤ 32768; if over, trim dated hazard prose (candidates listed in Task 4) without losing facts.
- **Docs SSOT:** the full hazard explanation lives in `CONTRIBUTING.md` (owns exact test commands); `AGENTS.md` §8 gets one cross-referencing line (hazards live there). No duplication of the explanation.
- **Verify per task:** `ruff check .` + the targeted command; final task runs the exact CI pytest command locally.

---

### Task 1: Pin the missing dev dependencies

**Files:**
- Modify: `requirements-dev.txt`

**Interfaces:**
- Consumes: the installed versions (verify with `pip show pytest-cov coverage`).
- Produces: CI installs `pytest-cov`/`coverage`; the local and CI coverage commands are identical.

- [ ] **Step 1: Confirm the installed versions**

Run: `./.venv/Scripts/python -m pip show pytest-cov coverage | grep -E "^(Name|Version):"`
Expected: `pytest-cov 7.1.0`, `coverage 7.13.5`.

- [ ] **Step 2: Add the pins**

Edit `requirements-dev.txt` — append after the ruff line:

```
pytest-cov==7.1.0
coverage==7.13.5
```

Keep the existing header comment; optionally extend it: these are dev tooling, not runtime deps, same as ruff.

- [ ] **Step 3: Verify a clean install resolves them**

Run: `./.venv/Scripts/python -m pip install -r requirements-dev.txt --dry-run 2>&1 | grep -E "pytest-cov|coverage|ruff" || echo "dry-run unsupported; pins verified by pip show"`
Expected: the pins resolve (or the grep reports the installed versions match). On this box the packages are already installed, so the dry-run may be a no-op — the pin correctness is the deliverable, verified by matching `pip show` output.

- [ ] **Step 4: Lint (file is not Python; check trailing whitespace only) + commit**

```bash
git diff --check
git add requirements-dev.txt
git commit -m "build: pin pytest-cov and coverage for the CI coverage gate"
```

---

### Task 2: Add coverage to the CI test step

**Files:**
- Modify: `.github/workflows/ci.yml` (the `Run test suite` step, currently `run: python -m pytest -q -rs`)

- [ ] **Step 1: Update the step**

Edit `.github/workflows/ci.yml` — replace the test step's `run:` line and extend the comment:

```yaml
      - name: Run test suite
        # -rs surfaces every skip reason so a green build does not read as
        # "real-data verified": the v5.3-gated tests (oracle parity, real
        # determinism, dashboard real-registry checks) skip in CI by design.
        # The real-data gate is the local pre-sign-off receipt
        # (scripts/real_data_gate.py, CONTRIBUTING.md) — CI green is the fast
        # gate only.
        #
        # Coverage: package-level specs ONLY. Never --cov=nmr.<submodule>:
        # coverage resolves dotted sources via find_spec inside a trace
        # callback, re-entering numpy's in-flight extension init on py3.12
        # ("cannot load module more than once per process"). See
        # CONTRIBUTING.md (coverage footgun).
        run: python -m pytest -q -rs --cov=nmr --cov=dashboard_ui --cov-report=term-missing
```

- [ ] **Step 2: Run the exact command locally (without the floor yet)**

Run: `./.venv/Scripts/python -m pytest -q -rs --cov=nmr --cov=dashboard_ui --cov-report=term-missing -p no:cacheprovider 2>&1 | tail -5`
Expected: full suite passes; a `TOTAL` coverage row prints (~90% locally). This proves the command form CI will run.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: measure coverage in the test step (package-level specs)"
```

---

### Task 3: First CI measurement — read the real CI numbers

**Files:** none (a push + log read)

**Why:** a floor set below the current value is a subsidy, not a gate. CI coverage differs from local (12 data-gated tests skip; Linux ctypes branches); the only honest floor is the CI-measured total. Local total is ~90% — the CI number is what the floor will bind to.

- [ ] **Step 1: Push Task 2's state (coverage, no floor) and wait for CI**

`git push` the Task 2 commit (or open the PR), then open the CI run log for the `Run test suite` step.

- [ ] **Step 2: Record the measured numbers**

From the CI log's `term-missing` report, record into the commit message of Task 4:
- `TOTAL` statement percentage (the global floor basis)
- the `nmr\promote.py` and `nmr\models.py` row percentages (the per-module floor basis)

If CI is unavailable right now (no push permitted in this session), fall back to the local full-suite numbers and mark the floor **provisional** — the first CI run then confirms or adjusts it by the same arithmetic.

---

### Task 4: The coverage gate — per-module minimums + a ratcheting global floor

**Files:**
- Create: `scripts/coverage_gate.py`
- Modify: `.github/workflows/ci.yml` (add the gate step after the test step)

**Interfaces:**
- Consumes: `coverage.json` produced by `pytest --cov=nmr --cov=dashboard_ui --cov-report=json` (pytest-cov writes it alongside `term-missing`).
- Produces: `scripts/coverage_gate.py --global-min G --module-min "nmr/promote.py:90" --module-min "nmr/models.py:88" [coverage.json]` — exits 0 when every floor is met, 1 otherwise, printing per-module rows and branch percentages. Precedent: `scripts/real_data_gate.py` (thin gate script, stdlib only).

- [ ] **Step 1: Write the gate script**

```python
"""Coverage gate: fail the build when coverage regresses below its floors.

Reads the JSON report produced by pytest-cov (`--cov-report=json`) and checks:
  - a global statement-coverage floor (--global-min),
  - per-module statement floors (--module-min path:pct, repeatable),
and prints branch coverage per gated module (reported, not gated in v1).

Ratchet rule (enforced by review, stated in ci.yml): floors only ever move UP.
The numbers below the current measurement are a subsidy, not a gate — a PR
that lowers a floor must carry its own written justification.

Usage (CI):
  python scripts/coverage_gate.py --global-min <T-0.5> \
      --module-min "nmr/promote.py:<P-1>" --module-min "nmr/models.py:<M-1>"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _file_summary(path: str, files: dict) -> tuple[float, float]:
    for key, summary in files.items():
        if key.replace("\\", "/").endswith(path.replace("\\", "/")):
            return (
                100.0 * summary["summary"]["covered_lines"]
                / max(1, summary["summary"]["num_statements"]),
                100.0 * summary["summary"]["covered_branches"]
                / max(1, summary["summary"]["num_branches"]),
            )
    raise SystemExit(f"coverage.json has no entry for {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--global-min", type=float, required=True)
    parser.add_argument("--module-min", action="append", default=[], metavar="PATH:PCT")
    parser.add_argument("report", nargs="?", default="coverage.json")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    totals = payload["totals"]
    global_pct = 100.0 * totals["covered_lines"] / max(1, totals["num_statements"])
    failures: list[str] = []
    print(f"coverage gate: global {global_pct:.1f}% (floor {args.global_min}%)")
    if global_pct < args.global_min:
        failures.append(f"global {global_pct:.1f}% < floor {args.global_min}%")

    for spec in args.module_min:
        path, _, pct_s = spec.partition(":")
        pct = float(pct_s)
        stmt, branch = _file_summary(path, payload["files"])
        print(f"  {path}: {stmt:.1f}% statements / {branch:.1f}% branches "
              f"(floor {pct}%)")
        if stmt < pct:
            failures.append(f"{path} {stmt:.1f}% < floor {pct}%")

    if failures:
        print("coverage gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("coverage gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify the script against the local JSON report**

Run: `./.venv/Scripts/python -m pytest -q -rs --cov=nmr --cov=dashboard_ui --cov-report=json --cov-report=term-missing -p no:cacheprovider 2>&1 | tail -2 && ./.venv/Scripts/python scripts/coverage_gate.py --global-min 0 --module-min "nmr/promote.py:0" --module-min "nmr/models.py:0"`
Expected: the gate prints the global and both module rows; exit 0. Record the printed percentages.

- [ ] **Step 3: Wire the gate into CI with the measured floors**

Edit `.github/workflows/ci.yml` — change the test step's report flags to also emit JSON, and add a gate step after it:

```yaml
      - name: Run test suite
        # ... (existing comment) ...
        run: python -m pytest -q -rs --cov=nmr --cov=dashboard_ui --cov-report=term-missing --cov-report=json
      - name: Coverage gate
        # Floors are the CI-MEASURED numbers (see git history for the
        # measurement commit), minus a 0.5pt calibration margin. RATCHET:
        # floors only move up — lowering one requires written justification.
        run: python scripts/coverage_gate.py --global-min <T-0.5> --module-min "nmr/promote.py:<P-1>" --module-min "nmr/models.py:<M-1>"
```

Replace `<T-0.5>`, `<P-1>`, `<M-1>` with the numbers recorded in Task 3 Step 2 (CI-measured; provisional from the local run if CI was unavailable), formatted to one decimal place.

- [ ] **Step 4: Prove the gate fails when it should**

Run: `./.venv/Scripts/python scripts/coverage_gate.py --global-min 99.9 --module-min "nmr/promote.py:99.9"`
Expected: exit 1 with the failure list printed.

- [ ] **Step 5: Lint + commit**

```bash
./.venv/Scripts/python -m ruff check scripts/coverage_gate.py
git add scripts/coverage_gate.py .github/workflows/ci.yml
git commit -m "ci: coverage gate with per-module minimums (measured floors: global T, promote P, models M)"
```

---

### Task 5: Document the working command and the dotted-spec hazard

**Files:**
- Modify: `CONTRIBUTING.md` (testing section + critical footguns)
- Modify: `AGENTS.md` (§8, one hazard line — mind the 32 KB budget)

- [ ] **Step 1: Read the two doc regions**

Read `CONTRIBUTING.md` lines 20-30 (critical footguns area), 53-90 (test commands + CI paragraph), and `AGENTS.md` §8 around the "Ruff lint gate" hazard. Locate the exact insertion points before editing.

- [ ] **Step 2: Add the footgun + working command to CONTRIBUTING.md**

In the critical-footguns list, add:

```markdown
- **Never pass dotted submodule `--cov` specs.** `--cov=nmr.promote` (or `nmr.models`, any `nmr.<module>`) crashes at conftest import with `ImportError: cannot load module more than once per process` on Python 3.12 + numpy 2.x + coverage 7.13.5. Root cause (measured 2026-08-19): coverage resolves dotted sources lazily via `find_spec('nmr.promote')` from inside a trace callback; resolving a submodule imports its parent package `nmr`, re-entering numpy's extension-module initialization that is still in flight. Top-level packages are immune (`find_spec('nmr')` never executes `__init__`), and `COVERAGE_CORE=pytrace` does not avoid it. For per-module numbers run `pytest ... --cov=nmr --cov=dashboard_ui --cov-report=term-missing` and read the per-file rows.
```

In the testing commands area (near line 65), add after the functional-gate line:

```markdown
# coverage (package-level specs only — see the dotted-spec footgun above)
.\.venv\Scripts\python -m pytest -q --cov=nmr --cov=dashboard_ui --cov-report=term-missing
```

- [ ] **Step 3: Add the AGENTS.md hazard line**

In `AGENTS.md` §8, extend the "Ruff lint gate" hazard entry with one sentence:

```markdown
Coverage commands must use package-level `--cov` specs only (`--cov=nmr --cov=dashboard_ui`) — dotted submodule specs (`--cov=nmr.promote`) crash at conftest import on py3.12 + coverage 7.x (root cause + working form: `CONTRIBUTING.md` coverage footgun).
```

- [ ] **Step 4: Enforce the AGENTS.md size budget**

Run: `wc -c AGENTS.md`
Expected: ≤ 32768. If over: trim dated prose first (candidates, in order of preference, without deleting facts: (a) the retired clause "The earlier conflicting full-universe figures (40-45 vs ~71 GiB) are retired" in the RAM hazard — compress to "the measured curve supersedes earlier figures"; (b) compress the `embargo_eras` hazard's second sentence to its mechanism only). Re-check `wc -c` after trimming.

- [ ] **Step 5: Verify no doc contradiction + commit**

```bash
grep -rn "cov-fail-under\|pytest-cov" README.md ARCHITECTURE.md CONTRIBUTING.md AGENTS.md | head
git add CONTRIBUTING.md AGENTS.md
git commit -m "docs: document the dotted-spec coverage footgun and the CI floor"
```

Expected: the grep shows the new references only in CONTRIBUTING.md/AGENTS.md (README and ARCHITECTURE must not duplicate the hazard — cross-reference instead).

---

### Task 6: Final verification — the exact CI commands

**Files:** none (verification only)

- [ ] **Step 1: Lint gate (as CI runs it)**

Run: `./.venv/Scripts/python -m ruff check .`
Expected: clean.

- [ ] **Step 2: The exact CI test + gate commands**

Run:

```bash
./.venv/Scripts/python -m pytest -q -rs --cov=nmr --cov=dashboard_ui --cov-report=term-missing --cov-report=json -p no:cacheprovider 2>&1 | tail -3
./.venv/Scripts/python scripts/coverage_gate.py --global-min <T-0.5> --module-min "nmr/promote.py:<P-1>" --module-min "nmr/models.py:<M-1>"
```

Expected: all tests pass; the gate prints global + both module rows and exits 0.

- [ ] **Step 3: Report the measured numbers**

State in the commit/PR description: global total and the two module percentages printed by Step 2, the floors set in `ci.yml`, and — if they came from the provisional local measurement — that the first CI run confirms them by the same arithmetic (floors only move up). Do not mark the task done without the actual printed numbers.

---

## Self-Review Notes

- **Spec coverage:** the remediation report's gap 3 ("no coverage floor in CI") is fully addressed: dependency pins (T1), measurement (T2), CI-measured numbers (T3), the gate itself with per-module minimums + a ratcheting global floor (T4), and the hazard documentation (T5). The floor binds to CI-measured values — never below what we stand on — and the per-module minimums make the money path impossible to regress while the global average stays green.
- **Placeholder scan:** no TBDs; all commands literal; the floor arithmetic (`T - 0.5`, `P - 1`, `M - 1`) is defined at the point the measurements are recorded, and the gate's failure path is proven in Task 4 Step 4.
- **Type consistency:** dev-pin versions (7.1.0 / 7.13.5) match the installed environment; the CI `run:` strings are verified verbatim in Task 2 Step 2 and Task 6 Step 2; the gate script reads pytest-cov's `coverage.json` schema (`totals.num_statements`, `totals.covered_lines`, `files[path].summary.*`); the AGENTS.md budget check has a concrete fallback (two named compression candidates).
- **Reviewer notes (carried forward, 2026-08-19):** (1) the ratchet is enforced by review, not code — a true ratchet needs stored state; accepted for v1, logged as a known soft spot. (2) Branch coverage is reported, not gated, in v1 — with a stated path to gating; do not let v1 become permanent. When branch numbers stabilize across the first CI runs, add a branch floor to `scripts/coverage_gate.py` the same way the statement floors were calibrated.
