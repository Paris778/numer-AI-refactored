"""Hardware specs and live resource status — thin control plane.

Usage:
    python hardware_status.py                # print specs + live status
    python hardware_status.py --record       # also write artifacts/reports/hardware_specs.json
    python hardware_status.py --record PATH  # write to PATH

All discovery/status logic lives in ``nmr.hardware`` (stdlib only: nvidia-smi
CLI + ctypes//proc). Specs are machine-constant and safe to record; the live
status snapshot must never enter canonical hashes or run_id payloads.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from nmr._atomicio import atomic_write_text
from nmr.hardware import discover_hardware, hardware_status

_SPECS_DEFAULT = Path("artifacts") / "reports" / "hardware_specs.json"


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print hardware specs and live status.")
    parser.add_argument(
        "--record", nargs="?", const=_SPECS_DEFAULT, type=Path, default=None,
        help="write machine specs to a JSON artifact (default: artifacts/reports/hardware_specs.json)",
    )
    args = parser.parse_args(argv)

    spec = discover_hardware()
    status = hardware_status()

    print("== Hardware spec (machine-constant) ==")
    print(f"OS: {spec.os_name}")
    print(f"Python: {spec.python_version}")
    print(f"CPU logical cores: {spec.cpu_logical_cores}")
    print(f"RAM total: {spec.ram_total_gib:.1f} GiB")
    if spec.gpus:
        for gpu in spec.gpus:
            print(
                f"GPU {gpu.index}: {gpu.name} - {gpu.memory_total_mib} MiB VRAM, "
                f"CUDA {gpu.compute_capability}, driver {gpu.driver_version}"
            )
    else:
        print("GPU: none detected (no nvidia-smi / no NVIDIA device)")
    print()
    print("== Live status (instantaneous, never hashed) ==")
    print(
        f"RAM used: {status.ram_used_gib:.1f} GiB / {status.ram_total_gib:.1f} GiB "
        f"(free {status.ram_free_gib:.1f} GiB)"
    )
    print(f"CPU usage: {round(status.cpu_usage_pct, 1) if status.cpu_usage_pct is not None else 'n/a'}%")
    if status.gpu_status:
        for row in status.gpu_status:
            print(
                f"GPU {row['index']}: util {row['utilization_pct']}%, "
                f"VRAM used {row['memory_used_mib']} MiB / "
                f"used+free {row['memory_used_mib'] + row['memory_free_mib']} MiB"
            )
    else:
        print("GPU status: n/a")

    if args.record is not None:
        # Record only the machine-constant spec — the live status snapshot is
        # instantaneous and must never be treated as reproducible.
        payload = {"spec": asdict(spec)}
        args.record.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.record, json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nwrote {args.record}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
