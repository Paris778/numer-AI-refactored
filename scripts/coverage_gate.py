"""Coverage gate: fail the build when coverage regresses below its floors.

Reads the JSON report produced by pytest-cov (`--cov-report=json`) and checks:
  - a global statement-coverage floor (--global-min),
  - per-module statement floors (--module-min path:pct, repeatable),
and prints branch coverage per gated module (reported, not gated in v1).

Ratchet rule (enforced by review, stated in ci.yml): floors only ever move UP.
Numbers below the current measurement are a subsidy, not a gate — a PR that
lowers a floor must carry its own written justification.

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
            data = summary["summary"]
            branches = data.get("covered_branches", 0)
            return (
                100.0 * data["covered_lines"]
                / max(1, data["num_statements"]),
                100.0 * branches / max(1, data.get("num_branches", 0)),
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
