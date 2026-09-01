"""Tests for the cross-family run registry and champion pointer.

Runs live under ``experiments/<slug>/runs/<run_id>/run.json``; the registry
iterates families for comparison and owns the atomic ``champion.json``
pointer. Tests monkeypatch ``paths.EXPERIMENTS_ROOT`` to ``tmp_path`` so every
write stays inside the fixture.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import polars as pl
import pytest

from nmr import experiment_store, paths
from nmr.evaluation import MetricSummary
from nmr.registry import RunRegistry
from nmr.runner import RunResult
from nmr.scorecard import MetricCell, MetricScorecard


@pytest.fixture
def registry(tmp_path, monkeypatch) -> RunRegistry:
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    return RunRegistry(tmp_path / "experiments")


def _result(run_id: str, sharpe: float, name: str = "fam-a") -> RunResult:
    return RunResult(
        run_id=run_id,
        oof=pl.DataFrame(
            {
                "id": ["a", "b"],
                "era": ["1", "1"],
                "prediction": [0.1, 0.9],
            }
        ),
        metrics=MetricSummary(mean=0.1, std=0.2, sharpe=sharpe, max_drawdown=0.05),
        artifact=None,
        manifest={"config": {"run": {"name": name}}},
    )


def _scorecard(sharpe_ac: float, *, max_drawdown: float = 0.1) -> MetricScorecard:
    def cell(v: float) -> MetricCell:
        return MetricCell(value=v, ci_low=None, ci_high=None, n_eras=10)

    return MetricScorecard(
        model_id="m",
        payout_policy_id="classic_legacy_075_225_clip005_v1",
        scoring_target="target",
        scoring_horizon="20D",
        n_eras=10,
        rank_scalar=0.0,
        deflated_sharpe=0.0,
        mean_payout=cell(0.0),
        corr=cell(0.0),
        mmc=cell(0.0),
        fnc=0.0,
        corr_sharpe_ac=cell(sharpe_ac),
        cvar5=0.0,
        max_drawdown=max_drawdown,
        burn_rate=0.0,
        mmc_sharpe_ac=0.0,
        sortino=0.0,
        calmar=0.0,
        std_corr=0.1,
        max_burn_streak=0,
        time_to_recovery=0,
        horizon_stability=None,
        horizon_reason=None,
        regime_corr=None,
        regime_reason=None,
        perturbation=None,
        max_feature_exposure=0.0,
        bmc=None,
        bmc_reason=None,
        cwmm=None,
        cwmm_reason=None,
        book_correlation=None,
        cagr_1y=0.0,
        gain_to_pain_ratio=0.0,
        kelly_fraction=0.0,
        mmc_down=None,
        mmc_down_n_eras=0,
        mmc_down_reason=None,
        turnover_mean=None,
        turnover_std=None,
        turnover_reason=None,
        sim_portfolio_cagr=0.0,
        sim_portfolio_mdd=0.0,
        sim_capital_utilization=0.0,
        capital_metrics_reason=None,
        metric_timing_seconds=None,
        eval_total_seconds=0.0,
    )


def _result_with_scorecard(
    run_id: str, sharpe_ac: float, *, name: str = "fam-a", max_drawdown: float = 0.1
) -> RunResult:
    result = _result(run_id, sharpe=0.5, name=name)
    return RunResult(
        run_id=result.run_id,
        oof=result.oof,
        metrics=result.metrics,
        artifact=result.artifact,
        manifest=result.manifest,
        scorecard=_scorecard(sharpe_ac, max_drawdown=max_drawdown),
    )


def _record(result: RunResult) -> Path:
    """Record a stub RunResult through the production path (experiment layout).

    The slug comes from the stub manifest's ``config.run.name`` — the family
    slug convention used by the scripts (``experiment_store.record_run_result``
    takes it explicitly).
    """
    name = ((result.manifest.get("config") or {}).get("run") or {}).get("name")
    assert isinstance(name, str) and name
    return experiment_store.record_run_result(name, result)


def test_cross_family_list_and_best(registry) -> None:
    experiment_store.record_run(
        "fam-a",
        "a" * 64,
        {
            "run_id": "a" * 64,
            "manifest": {"config": {"run": {"name": "fam-a"}}},
            "scorecard": {"corr_sharpe_ac": 0.5},
        },
    )
    experiment_store.record_run(
        "fam-b",
        "b" * 64,
        {
            "run_id": "b" * 64,
            "manifest": {"config": {"run": {"name": "fam-b"}}},
            "scorecard": {"corr_sharpe_ac": 0.9},
        },
    )
    assert set(registry.list()) == {"a" * 64, "b" * 64}
    assert registry.best() == ("b" * 64, "fam-b")


def test_best_rejects_mixed_policy_population_without_cohort(registry) -> None:
    first = _result_with_scorecard("a" * 64, sharpe_ac=0.8, name="fam-a")
    second = _result_with_scorecard("b" * 64, sharpe_ac=0.9, name="fam-b")
    second = dataclasses.replace(
        second,
        scorecard=dataclasses.replace(
            second.scorecard,
            payout_policy_id="classic_atomic_ender60_r1343_v1",
            scoring_target="target_ender_60",
            scoring_horizon="60D",
        ),
    )
    _record(first)
    _record(second)

    with pytest.raises(ValueError, match="requires policy_identity"):
        registry.best()
    assert registry.best(
        policy_identity=(
            "classic_atomic_ender60_r1343_v1",
            "target_ender_60",
            "60D",
        )
    ) == ("b" * 64, "fam-b")


def test_best_rejects_non_finite_metric(registry) -> None:
    result = _result_with_scorecard("a" * 64, sharpe_ac=float("inf"))
    _record(result)
    with pytest.raises(ValueError, match="non-finite"):
        registry.best()


def test_champion_pointer_has_slug(registry) -> None:
    experiment_store.record_run(
        "fam-a", "a" * 64, {"run_id": "a" * 64, "scorecard": {}}
    )
    path = registry.promote("a" * 64, "fam-a")
    payload = json.loads(path.read_text())
    assert payload == {
        "run_id": "a" * 64,
        "experiment_slug": "fam-a",
        "promoted_at": payload["promoted_at"],
    }


def test_list_returns_all_run_ids_sorted(registry) -> None:
    experiment_store.record_run(
        "fam-b", "b" * 64, {"run_id": "b" * 64, "scorecard": {}}
    )
    experiment_store.record_run(
        "fam-a", "a" * 64, {"run_id": "a" * 64, "scorecard": {}}
    )
    assert registry.list() == ["a" * 64, "b" * 64]


def test_best_skips_runs_without_metric_and_empty_registry(registry) -> None:
    assert registry.best() is None
    experiment_store.record_run(
        "fam-a", "a" * 64, {"run_id": "a" * 64, "scorecard": {"corr_sharpe_ac": 0.3}}
    )
    experiment_store.record_run(
        "fam-b", "b" * 64, {"run_id": "b" * 64, "scorecard": {}}
    )  # no metric
    assert registry.best() == ("a" * 64, "fam-a")
    assert registry.best("mmc") is None  # no run carries mmc


def test_promote_resolves_slug_by_scanning_when_omitted(registry) -> None:
    _record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    path = registry.promote("a" * 64)  # slug=None: scan the experiments layout
    assert json.loads(path.read_text(encoding="utf-8"))["experiment_slug"] == "fam-a"
    assert registry.promote("a" * 64) == path  # idempotent


def test_promote_unknown_run_fails_loud(registry) -> None:
    with pytest.raises(FileNotFoundError, match="no run record"):
        registry.promote("a" * 64, "fam-a")
    with pytest.raises(ValueError, match="not found"):
        registry.promote("a" * 64)


def test_promote_resolution_ambiguous_raises(registry) -> None:
    experiment_store.record_run("fam-a", "a" * 64, {"scorecard": {}})
    experiment_store.record_run("fam-b", "a" * 64, {"scorecard": {}})
    with pytest.raises(ValueError, match="ambiguous"):
        registry.promote("a" * 64)


def test_promote_rejects_non_hex_run_id(registry) -> None:
    with pytest.raises(ValueError, match="run_id"):
        registry.promote("../../etc/passwd", "fam-a")


def test_promote_if_better_promotes_only_strictly_better(registry) -> None:
    _record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    path, promoted = registry.promote_if_better("a" * 64)
    assert promoted is True
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "a" * 64

    _record(_result_with_scorecard("b" * 64, sharpe_ac=0.9))
    path, promoted = registry.promote_if_better("b" * 64)
    assert promoted is True
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "b" * 64

    _record(_result_with_scorecard("c" * 64, sharpe_ac=0.85))
    _, promoted = registry.promote_if_better("c" * 64)
    assert promoted is False
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "b" * 64


def test_promote_if_better_explicit_slug_cross_family(registry) -> None:
    _record(_result_with_scorecard("a" * 64, sharpe_ac=0.8, name="fam-a"))
    _record(_result_with_scorecard("b" * 64, sharpe_ac=0.9, name="fam-b"))
    registry.promote("a" * 64, "fam-a")
    path, promoted = registry.promote_if_better("b" * 64, "fam-b")
    assert promoted is True
    assert json.loads(path.read_text(encoding="utf-8"))["experiment_slug"] == "fam-b"


def test_promote_if_better_rejects_cross_policy_comparison(registry) -> None:
    _record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    registry.promote("a" * 64)
    result = _result_with_scorecard("b" * 64, sharpe_ac=0.9)
    atomic_scorecard = dataclasses.replace(
        result.scorecard,
        payout_policy_id="classic_atomic_ender60_r1343_v1",
        scoring_target="target_ender_60",
        scoring_horizon="60D",
    )
    _record(dataclasses.replace(result, scorecard=atomic_scorecard))

    with pytest.raises(ValueError, match="payout policy identities"):
        registry.promote_if_better("b" * 64)


def test_promote_if_better_direction_lower_is_better_for_drawdown(registry) -> None:
    _record(_result_with_scorecard("a" * 64, sharpe_ac=0.5, max_drawdown=0.2))
    registry.promote("a" * 64)

    # Higher drawdown is WORSE on max_drawdown -> must not promote.
    _record(_result_with_scorecard("b" * 64, sharpe_ac=0.5, max_drawdown=0.4))
    _, promoted = registry.promote_if_better("b" * 64, metric="max_drawdown")
    assert promoted is False

    _record(_result_with_scorecard("c" * 64, sharpe_ac=0.5, max_drawdown=0.1))
    _, promoted = registry.promote_if_better("c" * 64, metric="max_drawdown")
    assert promoted is True


def test_promote_if_better_champion_without_metric_is_displaced(registry) -> None:
    _record(_result("a" * 64, sharpe=0.9))  # champion without scorecard
    registry.promote("a" * 64)

    _record(_result_with_scorecard("b" * 64, sharpe_ac=0.4))
    _, promoted = registry.promote_if_better("b" * 64)
    assert promoted is True


def test_promote_if_better_refuses_candidate_without_metric(registry) -> None:
    _record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    registry.promote("a" * 64)
    _record(_result("b" * 64, sharpe=9.9))  # candidate without scorecard
    with pytest.raises(ValueError, match="scorecard"):
        registry.promote_if_better("b" * 64)


def test_promote_if_better_unknown_metric_raises(registry) -> None:
    _record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    with pytest.raises(ValueError, match="metric"):
        registry.promote_if_better("a" * 64, metric="nope")


def test_resolve_champion(registry) -> None:
    assert registry.resolve_champion() is None
    _record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    registry.promote("a" * 64, "fam-a")
    assert registry.resolve_champion() == ("a" * 64, "fam-a")


def test_resolve_champion_dangling_pointer_fails_loud(registry) -> None:
    champion = paths.champion_path()
    champion.parent.mkdir(parents=True, exist_ok=True)
    champion.write_text(
        json.dumps({"run_id": "b" * 64, "experiment_slug": "fam-a"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dangles"):
        registry.resolve_champion()


def test_resolve_champion_corrupt_pointer_fails_loud(registry) -> None:
    champion = paths.champion_path()
    champion.parent.mkdir(parents=True, exist_ok=True)
    champion.write_text(json.dumps({"run_id": "b" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        registry.resolve_champion()


def test_registry_rejects_misidentified_run_records(registry) -> None:
    """BLOCKING 1 (2026-08-29 re-review): a record stored at
    ``actual-family/<rid>/run.json`` whose embedded identity disagrees with its
    path — payload ``run_id`` != path run_id, or ``run.name`` != family slug —
    is refused by every registry read (list/best/promote/promote_if_better/
    resolve_champion). A misidentified record must never reach the champion
    pointer: no champion is written, and a pointer aimed at it fails loud."""
    run_id = "a" * 64
    run_dir = paths.experiment_dir("actual-family") / "runs" / run_id
    run_dir.mkdir(parents=True)

    # Embedded run_id disagrees with the path run_id.
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "b" * 64,
                "manifest": {"config": {"run": {"name": "actual-family"}}},
                "scorecard": {"corr_sharpe_ac": 0.9},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run_id"):
        registry.list()
    with pytest.raises(ValueError, match="run_id"):
        registry.best()
    with pytest.raises(ValueError, match="run_id"):
        registry.promote(run_id, "actual-family")
    with pytest.raises(ValueError, match="run_id"):
        registry.promote_if_better(run_id, "actual-family")
    assert not paths.champion_path().exists()  # promote never wrote a champion

    # A champion pointer already aimed at the misidentified record fails loud
    # on resolve — never returns the wrong identity.
    paths.champion_path().write_text(
        json.dumps({"run_id": run_id, "experiment_slug": "actual-family"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run_id"):
        registry.resolve_champion()

    # The same refusal when run_id agrees but run.name disagrees with the slug.
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "manifest": {"config": {"run": {"name": "other-family"}}},
                "scorecard": {"corr_sharpe_ac": 0.9},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run.name"):
        registry.list()
    with pytest.raises(ValueError, match="run.name"):
        registry.promote(run_id, "actual-family")

    # A non-object payload (valid JSON, not a dict) has no verifiable identity —
    # refused identically (ValueError, never AttributeError), never promoted.
    (run_dir / "run.json").write_text(json.dumps([run_id]), encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        registry.list()
    with pytest.raises(ValueError, match="not a JSON object"):
        registry.promote(run_id, "actual-family")

    # A matching-identity record passes every path.
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "manifest": {"config": {"run": {"name": "actual-family"}}},
                "scorecard": {"corr_sharpe_ac": 0.9},
            }
        ),
        encoding="utf-8",
    )
    assert registry.list() == [run_id]
    assert registry.best() == (run_id, "actual-family")
    champion = registry.promote(run_id, "actual-family")
    assert json.loads(champion.read_text(encoding="utf-8"))["run_id"] == run_id
    assert registry.resolve_champion() == (run_id, "actual-family")


def test_promote_if_better_dangling_champion_pointer_fails_loud(registry) -> None:
    _record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    paths.champion_path().write_text(
        json.dumps({"run_id": "b" * 64, "experiment_slug": "fam-a"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dangles"):
        registry.promote_if_better("a" * 64)


def test_champion_pointer_write_is_atomic(
    registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record(_result_with_scorecard("a" * 64, sharpe_ac=0.8))
    registry.promote("a" * 64, "fam-a")
    stable = paths.champion_path().read_text(encoding="utf-8")

    import nmr._atomicio as atomicio_module

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(atomicio_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        registry.promote("a" * 64, "fam-a")

    assert paths.champion_path().read_text(encoding="utf-8") == stable


def test_registry_isolated_root_stays_within_root(tmp_path, monkeypatch) -> None:
    """BLOCKING 4: a RunRegistry over a custom root lists/promotes/resolves
    ENTIRELY within that root — no file appears under the global
    EXPERIMENTS_ROOT (the registry must not delegate to paths.* globals)."""
    global_root = tmp_path / "global_experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", global_root)
    root = tmp_path / "iso_experiments"
    reg = RunRegistry(root)

    run_id = "a" * 64
    run_dir = root / "fam-iso" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "manifest": {"config": {"run": {"name": "fam-iso"}}},
                "scorecard": {"corr_sharpe_ac": 0.9},
            }
        ),
        encoding="utf-8",
    )

    assert reg.list() == [run_id]
    assert reg.best() == (run_id, "fam-iso")
    path = reg.promote(run_id, "fam-iso")
    assert path == root / "champion.json"
    assert reg.resolve_champion() == (run_id, "fam-iso")

    # Nothing leaked into the global experiments root (not even the lock file).
    assert not global_root.exists()
    # The lock file lives beside the pointer, inside the isolated root.
    assert (root / "champion.json.lock").is_file()


def _race_promote_if_better(root: str, run_id: str, slug: str) -> None:
    """Multiprocessing target: one champion compare-and-promote attempt."""
    from nmr.registry import RunRegistry

    RunRegistry(root).promote_if_better(run_id, slug)


def test_promote_if_better_concurrent_processes_serialize(tmp_path) -> None:
    """BLOCKING 3: N processes calling promote_if_better concurrently with
    different scorecard values must serialize on the champion lock — the final
    champion is the BEST value (without the lock, a read-compare-write race
    lets the last writer win, which may be any value)."""
    import multiprocessing as mp

    root = tmp_path / "experiments"
    values = {
        "a" * 64: 0.3,
        "b" * 64: 0.7,
        "c" * 64: 0.5,
        "d" * 64: 0.95,
        "e" * 64: 0.1,
    }
    for run_id, value in values.items():
        run_dir = root / "fam" / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps({"run_id": run_id, "scorecard": {"corr_sharpe_ac": value}}),
            encoding="utf-8",
        )

    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(target=_race_promote_if_better, args=(str(root), rid, "fam"))
        for rid in values
    ]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()
    assert all(proc.exitcode == 0 for proc in processes)

    champion = json.loads((root / "champion.json").read_text(encoding="utf-8"))
    best_id = max(values, key=values.get)
    assert champion["run_id"] == best_id
    assert champion["experiment_slug"] == "fam"
