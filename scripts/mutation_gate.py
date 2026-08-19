"""Mutation gate (CI-only): run mutmut over the correctness core, enforce the floor.

mutmut's runner is fork-based and refuses native Windows ("use WSL", mutmut
issue #397) — this gate therefore runs ONLY on Linux CI (mutation.yml,
weekly + manual dispatch; never on the push path). The receipt is durable:
written to ``artifacts/reports/mutation_receipt.json``, committed back to the
``ci/mutation-receipt`` branch by the CI job (a regression shows up as a diff),
uploaded as a workflow artifact, and summarized in the job summary.

SCOPE (read this before comparing numbers): CI skips the 12 data-gated tests,
so a mutant that only the real-data parity tests would kill registers as a
SURVIVOR here. The floors mean "survivors under the CI-runnable suite" and are
NOT comparable to any local measurement. This sentence is embedded in the
receipt itself.

Modes:
  --mode measure   first run: write the receipt, always exit 0 (floors from
                   measurement, never from guesses — report numbers first)
  --mode gate      compare fresh survivors against the committed receipt and
                   fail when any module's survivor count increases (ratchet
                   down only; raising a floor needs written justification)
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
RECEIPT = Path("artifacts/reports/mutation_receipt.json")
TIMEOUT_SECONDS = 15
SCOPE_NOTE = (
    "floors mean 'survivors under the CI-runnable suite' (12 data-gated tests "
    "skip in CI); NOT comparable to any local measurement"
)


def _run_module(module_path: str, tests: str) -> dict[str, int]:
    subprocess.run(
        [
            sys.executable, "-m", "mutmut", "run",
            "--paths-to-mutate", module_path,
            "--tests-dir", tests,
            "--test-timeout", str(TIMEOUT_SECONDS),
            "--simple-output",
        ],
        check=False,
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


def _compare(previous: dict[str, dict], fresh: dict[str, dict]) -> list[str]:
    failures: list[str] = []
    for module_path, counts in fresh.items():
        floor = previous.get(module_path, {}).get("survived")
        if floor is None:
            failures.append(f"{module_path}: no committed floor (measure first)")
        elif counts["survived"] > floor:
            failures.append(
                f"{module_path}: survivors {counts['survived']} > floor {floor} "
                "(ratchet down only — raising needs written justification)"
            )
    return failures


def _job_summary(lines: list[str]) -> None:
    import os

    env_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not env_path:
        return
    Path(env_path).write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["measure", "gate"], required=True)
    args = parser.parse_args(argv)

    previous: dict[str, dict] = {}
    if args.mode == "gate" and RECEIPT.is_file():
        previous = json.loads(RECEIPT.read_text(encoding="utf-8")).get("modules", {})

    fresh: dict[str, dict] = {}
    for module_path, tests in MODULE_TESTS.items():
        print(f"[mutation] {module_path}")
        counts = _run_module(module_path, tests)
        fresh[module_path] = counts
        print(f"  killed={counts['killed']} survived={counts['survived']} "
              f"timeout={counts['timeout']}")

    payload = {
        "scope_note": SCOPE_NOTE,
        "mode": args.mode,
        "modules": fresh,
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    print(f"[mutation] receipt written to {RECEIPT}")

    summary_lines = [
        f"## mutation gate ({args.mode})",
        f"- scope: {SCOPE_NOTE}",
    ]
    for module_path, counts in fresh.items():
        summary_lines.append(
            f"- {module_path}: killed={counts['killed']} "
            f"survived={counts['survived']} timeout={counts['timeout']}"
        )
    _job_summary(summary_lines)

    if args.mode == "measure":
        print("[mutation] measure mode: exit 0; report the numbers before setting floors")
        return 0

    failures = _compare(previous, fresh)
    if failures:
        print("mutation gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("mutation gate passed (no survivor increase over the committed floor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
