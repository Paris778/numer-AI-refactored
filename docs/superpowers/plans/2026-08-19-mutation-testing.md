# Mutation Testing Implementation Plan (pinned tool)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce mutation testing with the mature tool instead of a bespoke harness: pin `mutmut` in `requirements-dev.txt` (dev-tool precedent: `ruff`, `pytest-cov`, `coverage` are already pinned there — the §3 dependency prohibition governs *runtime* `requirements.txt`), measure a baseline survivor count on the four correctness-core modules, and gate on it — a mutation gate that **can fail a build**, on a schedule, not an opt-in ornament.

**Architecture:** `mutmut` does the mutation; a thin `scripts/mutation_gate.py` (precedent: `scripts/real_data_gate.py`) orchestrates per-module runs with per-module test subsets (so each mutant runs a bounded subset, not the full 865-test suite), aggregates `mutmut results` into `configs/mutation_receipt.json`, and enforces the survivor floor. CI: a weekly cron + `workflow_dispatch` job runs the gate. The floor ratchets DOWN only: after the baseline is measured, any change that increases survivors fails. No logic lives in `nmr/` — mutation testing is infrastructure, not part of the tested product boundary (corrected from the earlier bespoke-harness draft).

**Tech Stack:** Python 3.12, `mutmut` (exact-pinned in `requirements-dev.txt`), pytest as mutmut's test runner, GitHub Actions.

## Global Constraints

- **Pinned dev tool, not a bespoke harness.** Install mutmut via `./.venv/Scripts/python -m pip install mutmut` (never the `Scripts/pip` shim), record the installed version with `pip show mutmut`, pin it in `requirements-dev.txt`. `requirements.txt` (runtime) is untouched.
- **Verify mutmut's CLI against the installed version before trusting it.** Task 1 Step 2 runs `mutmut --help` and `mutmut run --help`; every flag used later must appear there. If a flag differs, adapt the command and note the change in the commit message — never guess CLI shapes.
- **Bounded subsets.** Each mutant runs its module's test files only (`tests-dir` per module), not the full suite. Mutants in `nmr/evaluation.py` run `tests/test_evaluation.py tests/test_parity.py`; `nmr/risk.py` → `tests/test_risk.py tests/test_risk_parity.py`; `nmr/_transforms.py` → `tests/test_transforms.py tests/test_parity.py`; `nmr/splitter.py` → `tests/test_splitter.py`. If mutmut's installed version accepts only one tests-dir, run one module × one test file per invocation and aggregate (8 runs).
- **The gate must be able to fail.** Floor = baseline survivor count per module; any increase fails the job. The floor ratchets down only. A PR lowering the floor needs written justification, like the coverage floors.
- **Determinism of the receipt.** The receipt records per-module killed/survived/timeout counts only — no wall-clock timestamps in anything compared across runs (timestamps are allowed as display fields, clearly marked).
- **Speed budget.** Per-mutant timeout 15 s (`--test-timeout`), workers 1 (deterministic, avoids Windows spawn contention). Full four-module run ≈ 1-2 h — weekly cron + manual dispatch only, never in the push path.
- **Abort criterion (reviewer note, 2026-08-19).** If mutmut proves unusable on this box (Windows spawn issues, 3.x CLI churn, unresolvable timeouts), **report and stop** — do NOT silently revert to a bespoke harness. The fallback decision is made by the human partner, not improvised mid-plan.
- **Docs follow code.** `CONTRIBUTING.md` (receipt command), `AGENTS.md` (§6 toolkit row + §8 hazard, budget ≤ 32768 bytes), `ARCHITECTURE.md` (one paragraph) in the same commit as the tooling.
- **Verify per task:** `ruff check .` + the targeted command; final task runs the full fast gate.

---

### Task 1: Install and pin mutmut

**Files:**
- Modify: `requirements-dev.txt`

- [ ] **Step 1: Install and record the version**

```bash
./.venv/Scripts/python -m pip install mutmut
./.venv/Scripts/python -m pip show mutmut | grep -E "^(Name|Version):"
```

Expected: a recent release (as of 2026-08) — record the exact version.

- [ ] **Step 2: Verify the CLI surface**

Run: `./.venv/Scripts/python -m mutmut --help && ./.venv/Scripts/python -m mutmut run --help 2>&1 | head -40`
Expected: `run` supports `--paths-to-mutate`, `--tests-dir`, `--test-timeout`, `--simple-output`; `results` and `junitxml` subcommands exist. Record any deviations — later tasks use exactly what this step shows.

- [ ] **Step 3: Pin it**

Append to `requirements-dev.txt`: `mutmut==<recorded version>` (exact, matching the header comment's convention).

- [ ] **Step 4: Commit**

```bash
git add requirements-dev.txt
git commit -m "build: pin mutmut for the mutation gate"
```

---

### Task 2: Baseline run and the gate script

**Files:**
- Create: `scripts/mutation_gate.py`
- Run-only: `configs/mutation_receipt.json` (machine-generated)

**Interfaces:**
- Consumes: `mutmut run` + `mutmut results` (via subprocess), the module/test-file map from Global Constraints.
- Produces: `scripts/mutation_gate.py --baseline` (measure, write receipt, always exit 0) and `scripts/mutation_gate.py` (gate mode: fail when any module's survivor count exceeds its receipt floor).

- [ ] **Step 1: Write the gate script**

```python
"""Mutation gate: run mutmut over the correctness core and enforce the floor.

Thin control plane (precedent: scripts/real_data_gate.py). Each module is
mutated against its own bounded test subset; `mutmut results` provides the
per-module killed/survived/timeout counts, aggregated into
configs/mutation_receipt.json.

Modes:
  --baseline   measure and write the receipt (exit 0 regardless of survivors)
  (default)    fail when any module's survivors exceed its receipt floor

The floor ratchets DOWN only: a PR that raises a floor needs written
justification, exactly like the coverage floors (ci.yml).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MODULE_TESTS = {
    "nmr/evaluation.py": "tests/test_evaluation.py tests/test_parity.py",
    "nmr/risk.py": "tests/test_risk.py tests/test_risk_parity.py",
    "nmr/_transforms.py": "tests/test_transforms.py tests/test_parity.py",
    "nmr/splitter.py": "tests/test_splitter.py",
}
RECEIPT = Path("configs/mutation_receipt.json")
TIMEOUT_SECONDS = 15


def _run(module_path: str, tests: str) -> dict[str, int]:
    """One mutmut run: mutate one module, test against its subset, parse results."""
    subprocess.run(
        [
            sys.executable, "-m", "mutmut", "run",
            "--paths-to-mutate", module_path,
            "--tests-dir", tests,
            "--test-timeout", str(TIMEOUT_SECONDS),
            "--simple-output",
        ],
        check=False,
        cwd=".",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "mutmut", "results"],
        check=False, capture_output=True, text=True,
    )
    killed = survived = timed_out = 0
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("Killed mutants"):
            killed = int(line.split()[-1].replace("(", ""))
        elif line.startswith("Survived mutants"):
            survived = int(line.split()[-1].replace("(", ""))
        elif line.startswith("Timeout mutants"):
            timed_out = int(line.split()[-1].replace("(", ""))
    return {"killed": killed, "survived": survived, "timeout": timed_out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args(argv)

    receipt: dict[str, dict] = {}
    for module_path, tests in MODULE_TESTS.items():
        print(f"[mutation] {module_path}")
        counts = _run(module_path, tests)
        receipt[module_path] = counts
        print(f"  killed={counts['killed']} survived={counts['survived']} "
              f"timeout={counts['timeout']}")

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps({"modules": receipt, "mode": "baseline" if args.baseline else "gate"},
                   sort_keys=True, indent=2),
        encoding="utf-8",
    )

    if args.baseline:
        print(f"[mutation] baseline written to {RECEIPT}")
        return 0

    previous = json.loads(Path(RECEIPT).read_text(encoding="utf-8"))["modules"] if False else None
    # Gate mode reads the committed receipt (the baseline) as the floor; the
    # comparison runs after the receipt above is restored from git, so load it
    # BEFORE overwriting: caller passes --baseline on the first run only.
    print("gate mode: compare against the committed receipt floors (ratchet down only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note for the implementer: the gate-mode comparison in the draft above is deliberately skeletal — finalize it as: load the committed receipt (from git `HEAD`) BEFORE overwriting `RECEIPT`, then `return 1` for any module whose fresh `survived` exceeds the committed floor, printing the failing rows. Keep the logic in the script (gate infrastructure, not product logic).

- [ ] **Step 2: Run the baseline**

Run: `./.venv/Scripts/python scripts/mutation_gate.py --baseline`
Expected: 4 module runs (each a few minutes at this scale), then a receipt with per-module killed/survived/timeout counts. Record the totals in the commit message.

- [ ] **Step 3: Commit**

```bash
./.venv/Scripts/python -m ruff check scripts/mutation_gate.py
git add scripts/mutation_gate.py configs/mutation_receipt.json
git commit -m "feat: mutation gate baseline (killed X, survived Y across 4 modules)"
```

---

### Task 3: The CI mutation job — weekly + manual, able to fail

**Files:**
- Create: `.github/workflows/mutation.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: mutation-gate

on:
  workflow_dispatch:
  schedule:
    # Weekly, off the :00 mark (anti-herd): catches test-suite regressions that
    # weaken mutant detection without slowing the push path.
    - cron: "17 3 * * 1"

jobs:
  mutation:
    runs-on: ubuntu-latest
    timeout-minutes: 180
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Mutation gate (fail on increased survivors)
        # Synthetic test subsets only — no v5.3 data needed; real-data tests
        # inside the subsets skip as in the main CI.
        run: python scripts/mutation_gate.py
```

Note: the gate script's final gate-mode logic (Task 2 Step 1) reads the committed receipt as the floor. On CI, `artifacts/` may be git-ignored — verify the receipt is committed (it is, per Task 2 Step 3) and that the checkout restores it.

- [ ] **Step 2: Verify the workflow syntax**

Run: `git diff --check && ./.venv/Scripts/python -c "import yaml" 2>/dev/null && ./.venv/Scripts/python -c "import yaml; yaml.safe_load(open('.github/workflows/mutation.yml', encoding='utf-8')); print('yaml ok')"`
Expected: clean diff, valid YAML. (If PyYAML is not installed in the venv, validate by eye — the file is 40 lines.)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/mutation.yml
git commit -m "ci: weekly mutation gate on the correctness core"
```

---

### Task 4: Documentation

**Files:**
- Modify: `CONTRIBUTING.md` (testing section: the receipt + gate commands)
- Modify: `AGENTS.md` (§6 toolkit row + §8 hazard line; budget check)
- Modify: `ARCHITECTURE.md` (one paragraph in the tooling/gates area)

- [ ] **Step 1: CONTRIBUTING.md**

After the real-data gate paragraph in the testing section:

```markdown
**Mutation gate** (do the tests catch bugs, not just visit lines): `.\.venv\Scripts\python scripts\mutation_gate.py --baseline` measures, the plain run gates. mutmut mutates `nmr/evaluation.py`, `nmr/risk.py`, `nmr/_transforms.py`, `nmr/splitter.py` against their bounded test subsets; `configs/mutation_receipt.json` holds the floors. The floor ratchets down only — a PR that raises a survivor floor needs written justification. CI runs the gate weekly + on manual dispatch (`mutation.yml`).
```

- [ ] **Step 2: AGENTS.md**

Toolkit row: `| Run the mutation gate / audit survivors | `scripts/mutation_gate.py` + pinned `mutmut` (receipt: `configs/mutation_receipt.json`) |`

§8 hazard line:

```markdown
### Mutation gate (mutmut, pinned dev tool)
Mutants run only against per-module test subsets (15 s timeout each); the full four-module gate is ~1-2 h — weekly CI + manual dispatch only, never the push path. Survivor floors ratchet down only. mutmut is dev tooling (`requirements-dev.txt`), not a runtime dependency.
```

Run `wc -c AGENTS.md` — ≤ 32768 bytes; if over, apply the compression candidates named in the CI-coverage-floor plan (Task 5 Step 4) first.

- [ ] **Step 3: ARCHITECTURE.md**

One paragraph in the tooling/gates section:

```markdown
### Mutation gate (dev tooling)

`scripts/mutation_gate.py` runs pinned `mutmut` over the correctness core
(`evaluation.py`, `risk.py`, `_transforms.py`, `splitter.py`) with per-module
bounded test subsets; survivor counts are floor-ed by
`configs/mutation_receipt.json` (ratchet down only). Weekly +
manual-dispatch CI (`mutation.yml`). Dev dependency only — nothing in
`requirements.txt` or the deploy closure.
```

- [ ] **Step 4: Commit**

```bash
git add CONTRIBUTING.md AGENTS.md ARCHITECTURE.md
git commit -m "docs: mutation gate usage, hazard, and architecture note"
```

---

### Task 5: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Smoke the gate on one module**

Run: `./.venv/Scripts/python -m mutmut run --paths-to-mutate nmr/splitter.py --tests-dir tests/test_splitter.py --test-timeout 15 --simple-output && ./.venv/Scripts/python -m mutmut results 2>&1 | tail -5`
Expected: a killed/survived/timeout summary for splitter mutants (single-digit minutes).

- [ ] **Step 2: Full fast gate**

```bash
./.venv/Scripts/python -m ruff check .
./.venv/Scripts/python -m pytest -q -p no:cacheprovider
```

Expected: ruff clean; the full suite green (no test changes in this plan — the suite is the same 865+ tests as after the previous plans).

---

## Self-Review Notes

- **Spec coverage:** the remediation gap 1 (tests may visit code without catching bugs) is addressed with a tool that is fit for purpose (mutmut handles mutant equivalence, timeouts, and result aggregation — none of which the bespoke draft would have done well), a measured baseline (T2), a floor that can actually fail a build on a schedule (T3), and documentation (T4). The bespoke `nmr/mutation.py` from the earlier draft is dropped entirely: mutation tooling is infrastructure, not product logic, and writing tests to test the test-tool was scope with no payoff.
- **Placeholder scan:** the only deliberately skeletal block (gate-mode floor comparison in Task 2 Step 1) carries an explicit finalization note; every mutmut flag is verified against the installed CLI in Task 1 Step 2 before use.
- **Type consistency:** the receipt schema (`modules[path] = {killed, survived, timeout}`) is written by Task 2 and read by Task 3's CI run; `MODULE_TESTS` keys match the four `--paths-to-mutate` values verbatim.
