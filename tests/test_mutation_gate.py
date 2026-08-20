"""Unit tests for the mutation gate's failure discipline (SEV-1 lesson).

The gate must NEVER mint floors from a failed or vacuous measurement: these
tests pin every raise path in ``_run_module`` with a stubbed ``subprocess.run``
and the repo-root constant pointed at ``tmp_path`` (no real mutmut, no repo
pollution — mutmut itself only runs on Linux CI, per its Windows refusal).
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

import scripts.mutation_gate as gate


def _stub_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    run_rc: int = 0,
    stats_rc: int = 0,
    stats_payload: dict | None = None,
    stats_exists: bool = True,
) -> None:
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)

    def fake_run(cmd, **kwargs):
        if "export-cicd-stats" in cmd:
            return types.SimpleNamespace(returncode=stats_rc, stdout="", stderr="stats boom")
        return types.SimpleNamespace(returncode=run_rc, stdout="", stderr="run boom")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    if stats_payload is not None:
        stats_dir = tmp_path / "mutants"
        stats_dir.mkdir(parents=True, exist_ok=True)
        (stats_dir / "mutmut-cicd-stats.json").write_text(
            json.dumps(stats_payload), encoding="utf-8"
        )


_GOOD_STATS = {"killed": 115, "survived": 31, "total": 146, "timeout": 0}


def test_run_module_nonzero_exit_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_subprocess(monkeypatch, tmp_path, run_rc=1, stats_payload=_GOOD_STATS)
    with pytest.raises(RuntimeError, match="mutmut run failed"):
        gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])


def test_run_module_stats_export_nonzero_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_subprocess(monkeypatch, tmp_path, stats_rc=1, stats_payload=_GOOD_STATS)
    with pytest.raises(RuntimeError, match="export-cicd-stats failed"):
        gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])


def test_run_module_missing_stats_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_subprocess(monkeypatch, tmp_path)  # no stats file written
    with pytest.raises(RuntimeError, match="no stats file"):
        gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])


def test_run_module_missing_keys_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_subprocess(
        monkeypatch,
        tmp_path,
        stats_payload={"killed": 1, "survived": 0, "total": 1},  # no "timeout"
    )
    with pytest.raises(RuntimeError, match="lack keys"):
        gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])


def test_run_module_zero_mutants_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_subprocess(
        monkeypatch,
        tmp_path,
        stats_payload={"killed": 0, "survived": 0, "total": 0, "timeout": 0},
    )
    with pytest.raises(RuntimeError, match="ZERO mutants"):
        gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])


def test_run_module_success_returns_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_subprocess(monkeypatch, tmp_path, stats_payload=_GOOD_STATS)
    counts = gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])
    assert counts == {"killed": 115, "survived": 31, "timeout": 0}
    # The scratch config is cleaned up afterwards.
    assert not (tmp_path / "pyproject.toml").exists()
    assert not (tmp_path / "mutants").exists()


def test_run_module_refuses_to_overwrite_existing_pyproject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])


def test_compare_no_committed_floor_fails() -> None:
    fresh = {"nmr/splitter.py": {"killed": 10, "survived": 2, "timeout": 0}}
    failures = gate._compare({}, fresh)
    assert len(failures) == 1
    assert "no committed floor" in failures[0]


def test_compare_survivor_increase_fails_and_equality_passes() -> None:
    previous = {"nmr/splitter.py": {"killed": 10, "survived": 3, "timeout": 0}}
    fresh_ok = {"nmr/splitter.py": {"killed": 10, "survived": 3, "timeout": 0}}
    fresh_bad = {"nmr/splitter.py": {"killed": 10, "survived": 4, "timeout": 0}}
    assert gate._compare(previous, fresh_ok) == []
    failures = gate._compare(previous, fresh_bad)
    assert len(failures) == 1
    assert "survivors 4 > floor 3" in failures[0]
