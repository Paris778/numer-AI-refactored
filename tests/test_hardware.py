"""Unit tests for nmr.hardware — parsing and discovery, platform-neutral."""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import pytest

from nmr.hardware import (
    GpuDevice,
    HardwareSpec,
    apply_thread_limits,
    discover_hardware,
    gpu_devices,
    hardware_status,
    parse_cpu_times,
    parse_gpu_devices,
    parse_gpu_status,
    parse_meminfo,
)

_GPU_QUERY = """0, NVIDIA RTX A1000 Laptop GPU, 4096, 580.97, 8.6
1, NVIDIA RTX 4090, 24564, 580.97, 8.9
"""

_GPU_STATUS = """0, 12, 1024, 3072
1, 98, 22000, 2564
"""

_MEMINFO = """MemTotal:       33554432 kB
MemFree:         8388608 kB
MemAvailable:   12582912 kB
"""

_POOL_VARS = (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def test_parse_gpu_devices() -> None:
    devs = parse_gpu_devices(_GPU_QUERY)
    assert len(devs) == 2
    assert devs[0] == GpuDevice(
        index=0,
        name="NVIDIA RTX A1000 Laptop GPU",
        memory_total_mib=4096,
        driver_version="580.97",
        compute_capability="8.6",
    )
    assert devs[1].index == 1
    assert devs[1].memory_total_mib == 24564


def test_parse_gpu_devices_empty() -> None:
    assert parse_gpu_devices("") == ()
    assert parse_gpu_devices("   \n") == ()


def test_parse_gpu_devices_malformed_line_skipped() -> None:
    devs = parse_gpu_devices("0, GPU with missing fields\n1, Ok GPU, 8192, 1.0, 7.5\n")
    assert len(devs) == 1
    assert devs[0].name == "Ok GPU"


def test_parse_gpu_status() -> None:
    status = parse_gpu_status(_GPU_STATUS)
    assert status == (
        {"index": 0, "utilization_pct": 12, "memory_used_mib": 1024, "memory_free_mib": 3072},
        {"index": 1, "utilization_pct": 98, "memory_used_mib": 22000, "memory_free_mib": 2564},
    )


def test_parse_meminfo() -> None:
    total, avail = parse_meminfo(_MEMINFO)
    assert total == pytest.approx(32.0, abs=0.01)  # GiB
    assert avail == pytest.approx(12.0, abs=0.01)


def test_parse_cpu_times() -> None:
    # 100 units elapsed, 80 idle -> 20% busy
    sample_a = (0, 0, 0)  # (busy_kernel, busy_user, idle)
    sample_b = (10, 10, 80)
    pct = parse_cpu_times(sample_a, sample_b, idle_index=2)
    assert pct == pytest.approx(20.0, abs=0.5)


def test_gpu_devices_without_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr("nmr.hardware._run_cli", lambda args: None)
    assert gpu_devices() == ()


def test_gpu_devices_live(monkeypatch) -> None:
    # Gate on which("nvidia-smi") too — the CLI patch alone only works on
    # machines where the binary exists; CI containers have no NVIDIA tooling.
    monkeypatch.setattr(
        "shutil.which",
        lambda name: "nvidia-smi" if name == "nvidia-smi" else None,
    )
    monkeypatch.setattr("nmr.hardware._run_cli", lambda args: _GPU_QUERY)
    devs = gpu_devices()
    assert len(devs) == 2
    assert devs[0].name.startswith("NVIDIA")


def test_discover_hardware_smoke() -> None:
    spec = discover_hardware()
    assert isinstance(spec, HardwareSpec)
    assert spec.cpu_logical_cores >= 1
    assert spec.ram_total_gib > 0.0
    assert isinstance(spec.gpus, tuple)
    assert spec.python_version  # non-empty
    assert spec.os_name  # non-empty


def test_hardware_status_smoke() -> None:
    status = hardware_status()
    assert status.ram_total_gib > 0.0
    assert status.ram_used_gib >= 0.0
    assert status.ram_free_gib >= 0.0
    assert status.cpu_usage_pct is None or 0.0 <= status.cpu_usage_pct <= 100.0
    assert isinstance(status.gpu_status, tuple)


def test_hardware_spec_serializable() -> None:
    spec = discover_hardware()
    payload = {
        "os": spec.os_name,
        "python": spec.python_version,
        "cpu_logical_cores": spec.cpu_logical_cores,
        "ram_total_gib": spec.ram_total_gib,
        "gpus": [asdict(g) for g in spec.gpus],
    }
    json.dumps(payload)  # must not raise


def test_hardware_symbols_exported() -> None:
    import nmr

    for name in [
        "GpuDevice",
        "HardwareSpec",
        "HardwareStatus",
        "apply_thread_limits",
        "discover_hardware",
        "gpu_devices",
        "gpu_status",
        "hardware_status",
    ]:
        assert name in nmr.__all__, name
        assert hasattr(nmr, name), name


def test_apply_thread_limits_default_caps_at_eight(monkeypatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 20)
    monkeypatch.delenv("NMR_MAX_THREADS", raising=False)
    for name in _POOL_VARS:
        monkeypatch.delenv(name, raising=False)
    assert apply_thread_limits() == 8
    for name in _POOL_VARS:
        assert os.environ[name] == "8"


def test_apply_thread_limits_env_override(monkeypatch) -> None:
    monkeypatch.setenv("NMR_MAX_THREADS", "3")
    for name in _POOL_VARS:
        monkeypatch.delenv(name, raising=False)
    assert apply_thread_limits() == 3
    for name in _POOL_VARS:
        assert os.environ[name] == "3"


def test_apply_thread_limits_explicit_wins_over_env(monkeypatch) -> None:
    monkeypatch.setenv("NMR_MAX_THREADS", "3")
    for name in _POOL_VARS:
        monkeypatch.delenv(name, raising=False)
    assert apply_thread_limits(5) == 5
    for name in _POOL_VARS:
        assert os.environ[name] == "5"


def test_apply_thread_limits_preserves_user_set_vars(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.delenv("NMR_MAX_THREADS", raising=False)
    for name in ("POLARS_MAX_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    limit = apply_thread_limits()
    assert os.environ["OMP_NUM_THREADS"] == "2"
    for name in ("POLARS_MAX_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ[name] == str(limit)


def test_apply_thread_limits_invalid_env_raises(monkeypatch) -> None:
    monkeypatch.setenv("NMR_MAX_THREADS", "banana")
    with pytest.raises(ValueError, match="NMR_MAX_THREADS"):
        apply_thread_limits()
    monkeypatch.setenv("NMR_MAX_THREADS", "0")
    with pytest.raises(ValueError):
        apply_thread_limits()
