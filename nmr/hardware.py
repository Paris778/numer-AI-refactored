"""Hardware discovery and live resource status — stdlib only, no new deps.

Static discovery (``discover_hardware``) is machine-constant and safe to
record in report manifests; live status (``hardware_status``) is
instantaneous usage and must never enter canonical hashes or run_id payloads.
GPU data comes from the ``nvidia-smi`` CLI (present with NVIDIA drivers); RAM
and CPU usage come from ctypes (Windows) or ``/proc`` (Linux). Pure parsing
helpers are exposed for tests; the ``_run_cli`` seam is the only subprocess
boundary.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass

__all__ = [
    "GpuDevice",
    "HardwareSpec",
    "HardwareStatus",
    "discover_hardware",
    "gpu_devices",
    "gpu_status",
    "hardware_status",
    "parse_cpu_times",
    "parse_gpu_devices",
    "parse_gpu_status",
    "parse_meminfo",
]

_MIB = 1024**2
_GIB = 1024**3


@dataclass(frozen=True)
class GpuDevice:
    """Static per-GPU facts from ``nvidia-smi --query-gpu``."""

    index: int
    name: str
    memory_total_mib: int
    driver_version: str
    compute_capability: str


@dataclass(frozen=True)
class HardwareSpec:
    """Machine-constant hardware facts (safe to record in manifests)."""

    os_name: str
    python_version: str
    cpu_logical_cores: int
    ram_total_gib: float
    gpus: tuple[GpuDevice, ...]


@dataclass(frozen=True)
class HardwareStatus:
    """Instantaneous resource usage (never hashed)."""

    ram_total_gib: float
    ram_used_gib: float
    ram_free_gib: float
    cpu_usage_pct: float | None
    gpu_status: tuple[dict[str, int], ...]


def _run_cli(args: list[str]) -> str | None:
    """Run a CLI and return stdout, or None when unavailable/failing."""
    try:
        completed = subprocess.run(
            args, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def parse_gpu_devices(text: str) -> tuple[GpuDevice, ...]:
    """Parse ``nvidia-smi --query-gpu=index,name,memory.total,driver_version,
    compute_cap --format=csv,noheader,nounits`` output. Malformed lines are
    skipped (a partial listing is a real signal, not a crash)."""
    devices: list[GpuDevice] = []
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 5:
            continue
        try:
            devices.append(
                GpuDevice(
                    index=int(fields[0]),
                    name=fields[1],
                    memory_total_mib=int(fields[2]),
                    driver_version=fields[3],
                    compute_capability=fields[4],
                )
            )
        except ValueError:
            continue
    return tuple(devices)


def parse_gpu_status(text: str) -> tuple[dict[str, int], ...]:
    """Parse ``nvidia-smi --query-gpu=index,utilization.gpu,memory.used,
    memory.free --format=csv,noheader,nounits`` output."""
    rows: list[dict[str, int]] = []
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 4:
            continue
        try:
            rows.append(
                {
                    "index": int(fields[0]),
                    "utilization_pct": int(fields[1]),
                    "memory_used_mib": int(fields[2]),
                    "memory_free_mib": int(fields[3]),
                }
            )
        except ValueError:
            continue
    return tuple(rows)


def parse_meminfo(text: str) -> tuple[float, float]:
    """Parse Linux ``/proc/meminfo`` -> (total_gib, available_gib)."""
    total = available = 0.0
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            total = float(line.split()[1]) * 1024 / _GIB
        elif line.startswith("MemAvailable:"):
            available = float(line.split()[1]) * 1024 / _GIB
    return total, available


def parse_cpu_times(
    sample_a: tuple[int, int, int],
    sample_b: tuple[int, int, int],
    idle_index: int,
) -> float:
    """Busy CPU fraction between two counter samples.

    Counters are ``(busy_kernel, busy_user, idle)``-shaped tuples; the idle
    counter sits at ``idle_index``. Returns percent busy in [0, 100].
    """
    deltas = [b - a for a, b in zip(sample_a, sample_b)]
    total = float(sum(deltas))
    if total <= 0.0:
        return 0.0
    idle = float(deltas[idle_index])
    return 100.0 * max(0.0, total - idle) / total


def gpu_devices() -> tuple[GpuDevice, ...]:
    """Discover CUDA devices via nvidia-smi; () when no NVIDIA tooling."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return ()
    out = _run_cli(
        [
            exe,
            "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_gpu_devices(out) if out is not None else ()


def gpu_status() -> tuple[dict[str, int], ...]:
    """Live per-GPU utilization and VRAM, or () when unavailable."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return ()
    out = _run_cli(
        [
            exe,
            "--query-gpu=index,utilization.gpu,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_gpu_status(out) if out is not None else ()


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _FILETIME(ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", ctypes.c_ulong),
        ("dwHighDateTime", ctypes.c_ulong),
    ]


def _ram_stats_windows() -> tuple[float, float, float] | None:
    try:
        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
    except (OSError, AttributeError):
        return None
    total = status.ullTotalPhys / _GIB
    free = status.ullAvailPhys / _GIB
    return total, total - free, free


def _ram_stats_linux() -> tuple[float, float, float] | None:
    try:
        text = open("/proc/meminfo", encoding="utf-8").read()
    except OSError:
        return None
    total, available = parse_meminfo(text)
    if total <= 0.0:
        return None
    return total, total - available, available


def _ram_stats() -> tuple[float, float, float] | None:
    if os.name == "nt":
        return _ram_stats_windows()
    return _ram_stats_linux()


def _cpu_usage_pct_windows(sample_seconds: float = 0.2) -> float | None:
    try:
        kernel32 = ctypes.windll.kernel32

        def _sample() -> tuple[int, int, int]:
            idle, kernel, user = _FILETIME(), _FILETIME(), _FILETIME()
            if not kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            ):
                raise OSError("GetSystemTimes failed")

            def to_int(ft) -> int:
                return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

            # kernel includes idle; report (kernel - idle, user, idle) so the
            # shared parse_cpu_times contract counts each tick once.
            return to_int(kernel) - to_int(idle), to_int(user), to_int(idle)

        first = _sample()
        time.sleep(sample_seconds)
        second = _sample()
    except (OSError, AttributeError):
        return None
    return parse_cpu_times(first, second, idle_index=2)


def _cpu_usage_pct_linux(sample_seconds: float = 0.2) -> float | None:
    try:
        first = _proc_stat_cpu()
        time.sleep(sample_seconds)
        second = _proc_stat_cpu()
    except (OSError, ValueError):
        return None
    if first is None or second is None:
        return None
    return parse_cpu_times(first, second, idle_index=2)


def _proc_stat_cpu() -> tuple[int, int, int] | None:
    with open("/proc/stat", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("cpu "):
                continue
            parts = line.split()[1:]  # user nice system idle iowait irq softirq steal
            if len(parts) < 4:
                return None
            values = [int(p) for p in parts]
            busy = values[0] + values[1] + values[2] + sum(values[4:])
            idle = values[3]
            return busy, 0, idle
    return None


def _cpu_usage_pct() -> float | None:
    if os.name == "nt":
        return _cpu_usage_pct_windows()
    return _cpu_usage_pct_linux()


def discover_hardware() -> HardwareSpec:
    """Machine-constant hardware facts. Never raises; missing data degrades."""
    ram = _ram_stats()
    return HardwareSpec(
        os_name=platform.system(),
        python_version=platform.python_version(),
        cpu_logical_cores=os.cpu_count() or 0,
        ram_total_gib=ram[0] if ram is not None else 0.0,
        gpus=gpu_devices(),
    )


def hardware_status() -> HardwareStatus:
    """Instantaneous resource usage. Never raises; missing data degrades."""
    ram = _ram_stats()
    return HardwareStatus(
        ram_total_gib=ram[0] if ram is not None else 0.0,
        ram_used_gib=ram[1] if ram is not None else 0.0,
        ram_free_gib=ram[2] if ram is not None else 0.0,
        cpu_usage_pct=_cpu_usage_pct(),
        gpu_status=gpu_status(),
    )
