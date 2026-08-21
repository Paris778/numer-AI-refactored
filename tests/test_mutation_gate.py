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


# The stats JSON schema has NINE categories (mutmut 3.7.0 save_cicd_stats);
# the gate must see and record all of them, never a silent four-key subset.
_FULL_STATS = {
    "killed": 115,
    "survived": 31,
    "timeout": 0,
    "total": 146,
    "no_tests": 0,
    "skipped": 0,
    "suspicious": 0,
    "check_was_interrupted_by_user": 0,
    "segfault": 0,
}


def test_run_module_nonzero_exit_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_subprocess(monkeypatch, tmp_path, run_rc=1, stats_payload=_FULL_STATS)
    with pytest.raises(RuntimeError, match="mutmut run failed"):
        gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])


def test_run_module_stats_export_nonzero_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_subprocess(monkeypatch, tmp_path, stats_rc=1, stats_payload=_FULL_STATS)
    with pytest.raises(RuntimeError, match="export-cicd-stats failed"):
        gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])


def test_run_module_missing_stats_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_subprocess(monkeypatch, tmp_path)  # no stats file written
    with pytest.raises(RuntimeError, match="no stats file"):
        gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])


def test_run_module_missing_any_of_nine_keys_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every one of the NINE schema keys is load-bearing; a missing one must
    # refuse to mint a floor (a receipt that cannot see a category can mint a
    # wrong floor — the SEV-1 lesson applied to the category table).
    for missing_key in _FULL_STATS:
        payload = dict(_FULL_STATS)
        payload.pop(missing_key)
        _stub_subprocess(monkeypatch, tmp_path, stats_payload=payload)
        with pytest.raises(RuntimeError, match="lack keys"):
            gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])


def test_run_module_zero_mutants_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = dict(_FULL_STATS, killed=0, survived=0, total=0, timeout=0)
    _stub_subprocess(monkeypatch, tmp_path, stats_payload=payload)
    with pytest.raises(RuntimeError, match="ZERO mutants"):
        gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])


def test_run_module_success_returns_all_nine_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_subprocess(monkeypatch, tmp_path, stats_payload=_FULL_STATS)
    counts = gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])
    assert counts == _FULL_STATS
    # The scratch config is cleaned up afterwards.
    assert not (tmp_path / "pyproject.toml").exists()
    assert not (tmp_path / "mutants").exists()


def test_run_module_timeout_ratio_refuses_to_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt full of timeouts measures the clock, not the tests."""
    payload = dict(_FULL_STATS, killed=1, survived=0, total=10, timeout=9)
    _stub_subprocess(monkeypatch, tmp_path, stats_payload=payload)
    with pytest.raises(RuntimeError, match="refusal threshold"):
        gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])


def test_run_module_timeout_ratio_counts_only_adjudicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`no_tests`/`skipped` mutants never ran — they must not dilute the
    timeout ratio denominator, or a module with many unkillable mutants hides
    a genuinely wedged harness (the 38-vs-658/696 accounting gap)."""
    payload = dict(
        _FULL_STATS,
        killed=0,
        survived=0,
        timeout=5,
        total=100,  # 95 in no_tests/skipped/suspicious/... never adjudicated
        no_tests=95,
    )
    _stub_subprocess(monkeypatch, tmp_path, stats_payload=payload)
    with pytest.raises(RuntimeError, match="refusal threshold"):
        gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])


def test_run_module_timeout_constant_from_measured_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every target module has a measured timeout constant, never a guess."""
    for module_path in gate.MODULE_TESTS:
        assert module_path in gate.MODULE_TIMEOUTS
        assert gate.MODULE_TIMEOUTS[module_path] > 15.0


def test_run_module_refuses_to_overwrite_existing_pyproject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        gate._run_module("nmr/splitter.py", ["tests/test_splitter.py"])


def test_compare_no_committed_floor_fails() -> None:
    fresh = {"nmr/splitter.py": {"killed": 10, "survived": 2, "timeout": 0, "total": 12}}
    failures = gate._compare({}, fresh)
    assert len(failures) == 1
    assert "no committed floor" in failures[0]


def test_compare_ratchets_on_survived_only() -> None:
    """Timeouts are kills (harness wedge), so they must NOT inflate the floor.
    A floor that includes timeouts can never fail (6+490=496) — the exact
    failure mode this session identified. Ratchet on survivors only."""
    previous = {"nmr/splitter.py": {"killed": 10, "survived": 3, "timeout": 1, "total": 14}}
    # More timeouts but same survivors: still at floor.
    fresh_same_survivors = {"nmr/splitter.py": {"killed": 9, "survived": 3, "timeout": 2, "total": 14}}
    assert gate._compare(previous, fresh_same_survivors) == []
    # More survivors: fails.
    fresh_more_survivors = {"nmr/splitter.py": {"killed": 10, "survived": 4, "timeout": 1, "total": 14}}
    failures = gate._compare(previous, fresh_more_survivors)
    assert len(failures) == 1
    assert "survived 4 > floor 3" in failures[0]


def _receipt(tmp_path: Path, modules: dict[str, dict]) -> Path:
    receipt = tmp_path / "configs" / "mutation_receipt.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"modules": modules}), encoding="utf-8")
    return receipt


def test_main_gate_scopes_to_floored_modules_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scheduled gate run must gate exactly what has a committed floor.
    With only splitter floored, the weekly job must NOT run evaluation.py
    (70% timeouts would refuse) nor fail on the other unfloored modules —
    a permanently red gate is worse than no gate."""
    receipt = _receipt(
        tmp_path, {"nmr/splitter.py": {"killed": 115, "survived": 31, "total": 146, "timeout": 0}}
    )
    monkeypatch.setattr(gate, "RECEIPT", receipt)
    ran: list[str] = []

    def fake_run_module(module_path, test_paths):
        ran.append(module_path)
        return {
            "killed": 115, "survived": 31, "timeout": 0, "total": 146,
            "no_tests": 0, "skipped": 0, "suspicious": 0,
            "check_was_interrupted_by_user": 0, "segfault": 0,
        }

    monkeypatch.setattr(gate, "_run_module", fake_run_module)
    rc = gate.main(["--mode", "gate"])
    assert rc == 0
    assert ran == ["nmr/splitter.py"]


def test_main_gate_empty_floors_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate mode with NO committed floor must fail, not vacuously pass —
    zero modules gated is the same vacuous-success defect as zero mutants."""
    receipt = _receipt(tmp_path, {})
    monkeypatch.setattr(gate, "RECEIPT", receipt)
    ran: list[str] = []

    def fake_run_module(module_path, test_paths):
        ran.append(module_path)
        return _FULL_STATS

    monkeypatch.setattr(gate, "_run_module", fake_run_module)
    rc = gate.main(["--mode", "gate"])
    assert rc == 1
    assert ran == []


def test_main_gate_does_not_overwrite_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate mode VERIFIES the committed floor; it must never rewrite it.
    A failing local gate run that overwrites the receipt would ratchet the
    floor up silently — measure mode alone writes the receipt."""
    floored = {"nmr/splitter.py": {"killed": 115, "survived": 31, "total": 146, "timeout": 0}}
    receipt = _receipt(tmp_path, floored)
    monkeypatch.setattr(gate, "RECEIPT", receipt)
    original = receipt.read_text(encoding="utf-8")

    def fake_run_module(module_path, test_paths):
        return {
            "killed": 115, "survived": 31, "timeout": 0, "total": 146,
            "no_tests": 0, "skipped": 0, "suspicious": 0,
            "check_was_interrupted_by_user": 0, "segfault": 0,
        }

    monkeypatch.setattr(gate, "_run_module", fake_run_module)
    rc = gate.main(["--mode", "gate"])
    assert rc == 0
    assert receipt.read_text(encoding="utf-8") == original
