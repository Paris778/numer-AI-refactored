"""Mutation gate (CI-only): run mutmut over the correctness core, enforce the floor.

mutmut's runner is fork-based and refuses native Windows ("use WSL", mutmut
issue #397) — this gate therefore runs ONLY on Linux CI (mutation.yml, weekly
+ manual dispatch; never on the push path).

mutmut 3.x is CONFIG-DRIVEN: `mutmut run` reads a ``[tool.mutmut]`` section
from ``pyproject.toml`` in the CWD (there is no repo pyproject.toml, so the
gate writes a scratch one per run with RELATIVE ``source_paths`` and RELATIVE
test-selection args — absolute paths make mutmut derive the wrong import key
and abort). Counts come from the machine-readable
``mutmut export-cicd-stats`` JSON — never from scraping progress output.

Failure discipline (SEV-1 lesson, 2026-08-20): a mutmut invocation that dies,
a stats file that is missing/malformed, or a zero-mutant run all RAISE. A
measurement that cannot be parsed must never be silently minted into a floor.

The receipt is SOURCE-OF-TRUTH evidence, not a machine artifact: it lives at
``configs/mutation_receipt.json`` (committed; NOT under gitignored
``artifacts/``). CI uploads it as an artifact and summarises it in the job
summary; a HUMAN commits it to main via a normal PR — merging is the act of
setting the floors (GITHUB_TOKEN cannot push workflow files, so there is no
bot commit-back).

The receipt records ALL NINE categories mutmut 3.7.0 serializes in
``export-cicd-stats`` (killed, survived, timeout, total, no_tests, skipped,
suspicious, check_was_interrupted_by_user, segfault) — never a silent
four-key subset. ``caught_by_type_check`` and ``not_checked`` exist in
mutmut's internal model but are NOT written to the JSON, so the gate cannot
record them.

Timeout semantics (proven 2026-08-21): a wedged child parks in ``pipe_write``
with zero CPU, so its own RLIMIT_CPU never fires and only the parent's
wall-clock SIGXCPU reaps it — timeouts are KILLS misclassified by a harness
defect (fork-after-polars-engine-init), not test-quality signals. Therefore:
  - the timeout ratio is computed against ADJUDICATED mutants only
    (killed + survived + timeout); no_tests/skipped/suspicious never produced
    a verdict and must not dilute it;
  - the refusal threshold is a HARNESS ALARM, not a quality gate;
  - the floor ratchets on SURVIVORS ONLY — counting timeouts in the floor
    yields an unfailable gate (evaluation 6+490=496).

SCOPE (read this before comparing numbers): CI skips the data-gated tests, so
a mutant that only the real-data parity tests would kill registers as a
SURVIVOR here. The floors mean "survivors under the CI-runnable suite" and are
NOT comparable to any local measurement. This sentence is embedded in the
receipt itself.

Modes:
  --mode measure   write the receipt, always exit 0 (floors from measurement)
  --mode gate      compare fresh survivors against the committed receipt and
                   fail when any module's survivor count increases (ratchet
                   down only; raising a floor needs written justification)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODULE_TESTS: dict[str, list[str]] = {
    "nmr/evaluation.py": ["tests/test_evaluation.py", "tests/test_parity.py"],
    "nmr/risk.py": ["tests/test_risk.py", "tests/test_risk_parity.py"],
    "nmr/_transforms.py": ["tests/test_transforms.py", "tests/test_parity.py"],
    "nmr/splitter.py": ["tests/test_splitter.py"],
}
RECEIPT = Path("configs/mutation_receipt.json")
REPO_ROOT = Path(__file__).resolve().parent.parent
STATS_KEYS = (
    "killed",
    "survived",
    "timeout",
    "total",
    "no_tests",
    "skipped",
    "suspicious",
    "check_was_interrupted_by_user",
    "segfault",
)
# mutmut 3.7.0 serializes exactly these NINE keys in save_cicd_stats. The
# internal Stat model also carries not_checked and caught_by_type_check, but
# they are NOT written to the JSON — so the gate cannot see them and must not
# claim to record them (a documented boundary, not a silent drop).
# A receipt that cannot see a category can mint a wrong floor, so every
# serialized key is validated as present (SEV-1 lesson applied to the table).
MAX_TIMEOUT_RATIO = 0.10
# Per-module timeout constants, from MEASURED subset durations (dataless
# Linux container, 2026-08-20): evaluation 5.1s, risk 2.7s, transforms 2.9s,
# splitter 2.2s. Constant = 10x subset duration + margin — the 15s guess
# produced 74-92% timeouts on the slow subsets and a vacuous risk.py
# survived=0. If the timeout-ratio refusal fires, re-measure on the CI runner
# and adjust; never raise the ratio threshold to make a bad run pass.
MODULE_TIMEOUTS: dict[str, float] = {
    "nmr/evaluation.py": 60.0,
    "nmr/risk.py": 40.0,
    "nmr/_transforms.py": 40.0,
    "nmr/splitter.py": 40.0,
}
SCOPE_NOTE = (
    "floors mean 'survivors under the CI-runnable suite' (data-gated tests "
    "skip in CI); NOT comparable to any local measurement"
)


def _run_module(module_path: str, test_paths: list[str]) -> dict[str, int]:
    """One mutmut run from the repo root; raises on ANY unparseable outcome.

    mutmut 3.x is config-driven and derives each mutant's import key from the
    RELATIVE source path — absolute paths make it guess wrong and abort with a
    recorded-vs-expected key mismatch. The scratch ``[tool.mutmut]`` config is
    therefore written to the repo root (a throwaway file in CI, removed in
    ``finally``), with relative ``source_paths`` and test-selection args.
    """
    config_path = REPO_ROOT / "pyproject.toml"
    if config_path.exists():
        raise RuntimeError(
            f"{config_path} exists — refusing to overwrite it. Merge the "
            "[tool.mutmut] section into the real file and adjust this script."
        )
    config_lines = [
        "[tool.mutmut]",
        f"source_paths = [{json.dumps(module_path)}]",
        # mutmut removes the checkout root from sys.path so the mutated file
        # shadows the original — but the mutants tree then lacks the REST of
        # the package and `import nmr.config` (tests/conftest.py) dies.
        # Copy the full package in; the mutated module overwrites its twin.
        "also_copy = ['nmr']",
        "pytest_add_cli_args_test_selection = ["
        + ", ".join(json.dumps(t) for t in test_paths)
        + "]",
        "timeout_multiplier = 1.0",
        f"timeout_constant = {MODULE_TIMEOUTS[module_path]}",
        # The repo's pytest.ini `pythonpath = .` re-inserts the checkout root
        # after mutmut removes it, so the stats phase imports the ORIGINAL
        # module and attributes zero tests ("Stopping early"). Override it to
        # the mutants tree (mutmut's own insertion) instead of the root; an
        # EMPTY value is a pytest usage error (exit 4).
        'pytest_add_cli_args = ["--override-ini", "pythonpath=mutants"]',
        # Run EVERY mutant against the subset: mutmut's covered-lines
        # pre-check depends on its own coverage mapping, which finds nothing
        # under this repo's pytest.ini pythonpath setup and aborts early.
        "mutate_only_covered_lines = false",
        "",
    ]
    config_path.write_text("\n".join(config_lines), encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "mutmut", "run"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"mutmut run failed for {module_path} (exit {proc.returncode}): "
                f"{proc.stderr[-600:] or proc.stdout[-600:]}"
            )
        stats_proc = subprocess.run(
            [sys.executable, "-m", "mutmut", "export-cicd-stats"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if stats_proc.returncode != 0:
            raise RuntimeError(
                f"mutmut export-cicd-stats failed for {module_path} "
                f"(exit {stats_proc.returncode})"
            )
        stats_path = REPO_ROOT / "mutants" / "mutmut-cicd-stats.json"
        if not stats_path.is_file():
            raise RuntimeError(
                f"mutmut produced no stats file for {module_path} — a run that "
                "cannot be measured must not mint floors"
            )
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        missing = [k for k in STATS_KEYS if k not in stats]
        if missing:
            raise RuntimeError(
                f"mutmut stats for {module_path} lack keys {missing}: {stats}"
            )
        if stats["total"] == 0:
            raise RuntimeError(
                f"mutmut found ZERO mutants in {module_path} — vacuous "
                "measurement; refusing to record it"
            )
        # Ratio against ADJUDICATED mutants only (killed + survived + timeout).
        # no_tests/skipped/suspicious/segfault/interrupted never produced a
        # verdict — including them dilutes the ratio and lets a wedged harness
        # hide behind a large unkillable population (the 38-vs-696 gap).
        adjudicated = stats["killed"] + stats["survived"] + stats["timeout"]
        timeout_ratio = stats["timeout"] / adjudicated if adjudicated else 0.0
        if timeout_ratio > MAX_TIMEOUT_RATIO:
            raise RuntimeError(
                f"mutmut timed out {stats['timeout']}/{adjudicated} adjudicated "
                f"mutants ({timeout_ratio:.0%}) in {module_path} — above the "
                f"{MAX_TIMEOUT_RATIO:.0%} refusal threshold; the numbers would "
                "measure the clock, not the tests. Re-measure the subset "
                "duration and raise MODULE_TIMEOUTS (never the ratio)."
            )
        return {key: int(stats[key]) for key in STATS_KEYS}
    finally:
        config_path.unlink(missing_ok=True)
        shutil.rmtree(REPO_ROOT / "mutants", ignore_errors=True)


def _compare(previous: dict[str, dict], fresh: dict[str, dict]) -> list[str]:
    """Ratchet on SURVIVORS only. Timeouts are kills (harness wedge: a wedged
    child is classified timeout at any budget), so counting them in the floor
    yields an unfailable gate (6+490=496) — the defect this session proved.
    The timeout-ratio refusal, not the ratchet, is the harness alarm."""
    failures: list[str] = []
    for module_path, counts in fresh.items():
        prev_counts = previous.get(module_path)
        if prev_counts is None:
            failures.append(f"{module_path}: no committed floor (measure first)")
            continue
        floor = prev_counts.get("survived", 0)
        fresh_survived = counts.get("survived", 0)
        if fresh_survived > floor:
            failures.append(
                f"{module_path}: survived {fresh_survived} > floor {floor} "
                "(ratchet down only — raising needs written justification)"
            )
    return failures


def _job_summary(lines: list[str]) -> None:
    env_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not env_path:
        return
    Path(env_path).write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["measure", "gate"], required=True)
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="module paths to run (default: measure=all, gate=floored modules)",
    )
    args = parser.parse_args(argv)

    previous: dict[str, dict] = {}
    if args.mode == "gate" and RECEIPT.is_file():
        previous = json.loads(RECEIPT.read_text(encoding="utf-8")).get("modules", {})

    if args.targets is not None:
        targets = list(args.targets)
    elif args.mode == "gate":
        # Scope the scheduled gate run to modules that have a committed floor.
        # Running every module here fails on the unfloored ones ("no committed
        # floor") and on evaluation.py's wedged timeouts every single week —
        # a permanently red gate trains people to ignore it. Gate exactly what
        # is measured; new floors join automatically when their receipt lands.
        targets = sorted(previous)
    else:
        targets = list(MODULE_TESTS)
    for target in targets:
        if target not in MODULE_TESTS:
            raise ValueError(f"unknown target {target!r}; known: {sorted(MODULE_TESTS)}")

    if args.mode == "gate" and not targets:
        print("mutation gate FAILED: no committed floors to gate (measure first)")
        return 1

    fresh: dict[str, dict] = {}
    for module_path in targets:
        print(f"[mutation] {module_path}")
        counts = _run_module(module_path, MODULE_TESTS[module_path])
        fresh[module_path] = counts
        print(f"  killed={counts['killed']} survived={counts['survived']} "
              f"timeout={counts['timeout']}")

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
        # Measure mode alone writes the receipt. Gate mode must NEVER rewrite
        # it: a failing local gate run that overwrites the committed file would
        # ratchet the floor up silently, manufacturing the exact false-alarm
        # situation the ratchet rule exists to prevent.
        payload = {
            "scope_note": SCOPE_NOTE,
            "mode": args.mode,
            "modules": fresh,
        }
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        print(f"[mutation] receipt written to {RECEIPT}")
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
