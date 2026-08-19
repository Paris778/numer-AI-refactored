"""Local pre-sign-off real-data gate: run the v5.3-gated suites and write a
machine-checkable receipt.

CI cannot run these suites (no v5.3 parquet on ubuntu-latest; every
real-data/parity test skips there by design), so CI green is the FAST gate
only. This script is the authoritative real-data verification
(CONTRIBUTING.md pre-sign-off gate): oracle parity, real-data determinism,
and the benchmark fast-mode smoke, with a receipt JSON recording commands,
exit codes, and per-suite pass/fail.

Usage:
    python scripts/real_data_gate.py [--report-dir artifacts/reports]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPORT_FILENAME = "real_data_gate_receipt.json"

_STEPS = [
    (
        "oracle_parity",
        [
            sys.executable, "-m", "pytest", "-q",
            "tests/test_parity.py", "tests/test_risk_parity.py",
        ],
    ),
    (
        "real_determinism",
        [sys.executable, "-m", "pytest", "-q", "tests/test_benchmark_hierarchy.py"],
    ),
    (
        "benchmark_fast_mode",
        [sys.executable, "benchmark_runner.py", "--fast-mode"],
    ),
]


def _run(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "passed": proc.returncode == 0,
        "tail": (proc.stdout + proc.stderr)[-2000:],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir", type=Path, default=Path("artifacts") / "reports"
    )
    args = parser.parse_args(argv)

    results = {name: _run(cmd) for name, cmd in _STEPS}
    receipt = {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "steps": results,
        "all_passed": all(r["passed"] for r in results.values()),
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    out = args.report_dir / REPORT_FILENAME
    out.write_text(json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print(f"\nreceipt written to {out}")
    return 0 if receipt["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
