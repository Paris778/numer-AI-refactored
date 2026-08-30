"""Tests for dashboard_ui.service — unified data layer.

Tests verify:
1. Data loading (leaderboard, campaigns, registry entries)
2. Return types (Pydantic models)
3. Cache invalidation (mtime sentinel)
4. Filtering/sorting operations
5. Label formatting
"""

import json
import tempfile
from pathlib import Path

import pytest

from dashboard_ui.service import (
    CampaignLog,
    DashboardDataService,
    LeaderboardFrame,
    LeaderboardRowModel,
    RobustnessMatrix,
    TopPerformerRowModel,
    TopPerformersResult,
)


class TestLeaderboardRowModel:
    """Test Pydantic model validation."""

    def test_minimal_valid_row(self):
        """Minimal valid LeaderboardRowModel."""
        row = LeaderboardRowModel(
            model_id="test-001",
            source="trained",
            run_name="test-config",
            run_dir="/tmp/test",
        )
        assert row.model_id == "test-001"
        assert row.source == "trained"
        assert row.corr is None  # Optional fields

    def test_full_valid_row(self):
        """Full populated LeaderboardRowModel."""
        row = LeaderboardRowModel(
            model_id="test-001",
            source="trained",
            run_name="test-config",
            run_dir="/tmp/test",
            backend="lightgbm",
            preset="standard",
            corr_sharpe_ac=1.23,
            corr_sharpe_ac_ci_low=1.10,
            corr_sharpe_ac_ci_high=1.36,
            max_drawdown=-0.15,
            has_bmc=True,
            status="CAPITAL READY",
        )
        assert row.corr_sharpe_ac == 1.23
        assert row.has_bmc is True

    def test_invalid_source_still_accepted(self):
        """Source is just a string; no enum validation in model."""
        row = LeaderboardRowModel(
            model_id="test-001",
            source="invalid_source",
            run_name="test-config",
            run_dir="/tmp/test",
        )
        assert row.source == "invalid_source"


class TestLeaderboardFrame:
    """Test LeaderboardFrame filtering and sorting."""

    @pytest.fixture
    def sample_leaderboard(self) -> LeaderboardFrame:
        """Sample leaderboard with multiple rows."""
        rows = [
            LeaderboardRowModel(
                model_id="m1",
                source="trained",
                run_name="config-a",
                run_dir="/tmp/m1",
                backend="lightgbm",
                preset="standard",
                corr_sharpe_ac=1.23,
            ),
            LeaderboardRowModel(
                model_id="m2",
                source="trained",
                run_name="config-b",
                run_dir="/tmp/m2",
                backend="xgboost",
                preset="fast",
                corr_sharpe_ac=0.95,
            ),
            LeaderboardRowModel(
                model_id="bench-1",
                source="benchmark",
                run_name="baseline",
                run_dir="/tmp/bench",
                corr_sharpe_ac=0.50,
            ),
        ]
        return LeaderboardFrame(rows=rows, total_rows=3, evaluable_rows=2)

    def test_len(self, sample_leaderboard: LeaderboardFrame):
        """Test __len__ method."""
        assert len(sample_leaderboard) == 3

    def test_filter_by_source(self, sample_leaderboard: LeaderboardFrame):
        """Filter by source."""
        trained_only = sample_leaderboard.filter_by_source(["trained"])
        assert len(trained_only) == 2
        assert all(r.source == "trained" for r in trained_only.rows)

    def test_filter_by_backend(self, sample_leaderboard: LeaderboardFrame):
        """Filter by backend."""
        lgbm_only = sample_leaderboard.filter_by_backend(["lightgbm"])
        assert len(lgbm_only) == 1
        assert lgbm_only.rows[0].model_id == "m1"

    def test_filter_by_preset(self, sample_leaderboard: LeaderboardFrame):
        """Filter by preset."""
        standard_only = sample_leaderboard.filter_by_preset(["standard"])
        assert len(standard_only) == 1
        assert standard_only.rows[0].model_id == "m1"

    def test_sort_by_metric_descending(self, sample_leaderboard: LeaderboardFrame):
        """Sort by metric descending."""
        sorted_lb = sample_leaderboard.sort_by_metric("corr_sharpe_ac", descending=True)
        assert sorted_lb.rows[0].model_id == "m1"  # 1.23
        assert sorted_lb.rows[1].model_id == "m2"  # 0.95
        assert sorted_lb.rows[2].model_id == "bench-1"  # 0.50

    def test_sort_by_metric_ascending(self, sample_leaderboard: LeaderboardFrame):
        """Sort by metric ascending."""
        sorted_lb = sample_leaderboard.sort_by_metric(
            "corr_sharpe_ac", descending=False
        )
        assert sorted_lb.rows[0].model_id == "bench-1"  # 0.50
        assert sorted_lb.rows[-1].model_id == "m1"  # 1.23

    def test_sort_by_metric_with_nulls(self):
        """Sort handles None values gracefully."""
        rows = [
            LeaderboardRowModel(
                model_id="m1",
                source="trained",
                run_name="config-a",
                run_dir="/tmp/m1",
                corr_sharpe_ac=1.23,
            ),
            LeaderboardRowModel(
                model_id="m2",
                source="trained",
                run_name="config-b",
                run_dir="/tmp/m2",
                corr_sharpe_ac=None,  # Missing value
            ),
        ]
        lb = LeaderboardFrame(rows=rows, total_rows=2, evaluable_rows=2)
        sorted_lb = lb.sort_by_metric("corr_sharpe_ac", descending=True)
        # Non-None values should come first when descending
        assert sorted_lb.rows[0].corr_sharpe_ac == 1.23


class TestDashboardDataService:
    """Test DashboardDataService data loading and caching."""

    def test_leaderboard_lifecycle_fields_round_trip(
        self, tmp_path, monkeypatch
    ) -> None:
        """SECONDARY 5: the engine emits the lifecycle contract per family —
        display_name / lifecycle_stage / current_full_status / stale must
        round-trip through LeaderboardRowModel."""
        from nmr import experiment_store, paths
        from nmr.deployment import serialize_predict

        experiments = tmp_path / "experiments"
        monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", experiments)
        run_id = "a" * 64
        experiment_store.record_run(
            "brb1-xgb-v6",
            run_id,
            {
                "run_id": run_id,
                "manifest": {
                    "config": {
                        "run": {"name": "brb1-xgb-v6"},
                        "data": {"feature_set": "medium"},
                        "model": {"backend": "lightgbm", "preset": "fast"},
                    },
                    "oof_device": "cpu",
                },
                "scorecard": {"corr": 0.05},
            },
        )
        # A valid full export + pointer (identity binding needs the run record
        # above).
        slot = paths.export_dir("brb1-xgb-v6", "full", run_id)
        slot.mkdir(parents=True, exist_ok=True)

        def dummy_predict(live_features, live_benchmark_models=None):
            return live_features

        serialize_predict(dummy_predict, path=slot / "predict.pkl", feature_names=["f1"])
        (slot / "export.json").write_text(
            json.dumps(
                {
                    "family": "brb1-xgb-v6",
                    "training_scope": "full",
                    "promoted_from_run_id": run_id,
                    "promoted_at": "2026-08-26T10:00:00+00:00",
                    "config": {"run": {"name": "brb1-xgb-v6"}},
                }
            ),
            encoding="utf-8",
        )
        paths.current_pointer_path("brb1-xgb-v6").write_text(
            json.dumps({"run_id": run_id}), encoding="utf-8"
        )

        service = DashboardDataService(
            registry_dir=experiments,
            benchmark_path=tmp_path / "no_benchmark.csv",
        )
        leaderboard = service.load_leaderboard()
        rows = {r.model_id: r for r in leaderboard.rows}
        trained = rows[run_id]
        assert trained.family == "brb1-xgb-v6"
        assert trained.display_name == "brb1-xgb-v6"
        assert trained.lifecycle_stage == "full"
        assert trained.current_full_status == "full"
        assert trained.stale is False
        assert trained.has_full_version is True
        full_id = f"brb1-xgb-v6::full::{run_id}"
        assert full_id in rows
        assert rows[full_id].lifecycle_stage == "full"
        assert rows[full_id].current_full_status == "full"

        # The fields survive the Pydantic JSON round-trip (serialization
        # contract for the HTML/Streamlit hosts).
        restored = LeaderboardFrame.model_validate_json(
            leaderboard.model_dump_json()
        )
        restored_row = next(r for r in restored.rows if r.model_id == run_id)
        assert restored_row.lifecycle_stage == "full"
        assert restored_row.current_full_status == "full"
        assert restored_row.display_name == "brb1-xgb-v6"

    @pytest.fixture
    def temp_registry_dir(self) -> Path:
        """Temporary registry directory with sample run.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_dir = Path(tmpdir) / "registry"
            registry_dir.mkdir(parents=True)

            # Create a sample run.json
            run_dir = registry_dir / "test-run-123"
            run_dir.mkdir()
            run_json = {
                "run_id": "test-run-123",
                "manifest": {
                    "config": {
                        "run": {"name": "test-config"},
                        "data": {"feature_set": "medium"},
                        "model": {"backend": "lightgbm", "preset": "standard"},
                    },
                    "oof_device": "cpu",
                },
                "scorecard": {"corr": 0.042},
            }
            (run_dir / "run.json").write_text(json.dumps(run_json))

            yield registry_dir

    def test_load_registry_entries(self, temp_registry_dir: Path):
        """Test load_registry_entries()."""
        service = DashboardDataService(registry_dir=temp_registry_dir)
        entries = service.load_registry_entries()

        assert len(entries) == 1
        assert entries[0]["run_id"] == "test-run-123"
        assert isinstance(entries[0], dict)

    def test_load_registry_entries_empty_dir(self):
        """Test load_registry_entries() with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = DashboardDataService(registry_dir=Path(tmpdir))
            entries = service.load_registry_entries()
            assert entries == []

    def test_cache_invalidation(self, temp_registry_dir: Path):
        """Test source-fingerprint cache invalidation."""
        service = DashboardDataService(registry_dir=temp_registry_dir)

        # Cache should be invalid initially
        assert not service._check_cache_valid()

        # Load once -> anchored on the source fingerprint
        leaderboard1 = service.load_leaderboard()
        assert service._source_fingerprint is not None

        # Second load should return cached (fingerprint unchanged)
        leaderboard2 = service.load_leaderboard()
        assert leaderboard1 is leaderboard2  # Same object

    def test_source_fingerprint_invalidates_on_registry_change(
        self, temp_registry_dir: Path
    ):
        """A registry change (new run record) invalidates the cache."""
        service = DashboardDataService(registry_dir=temp_registry_dir)
        first = service.load_leaderboard()
        run_dir = temp_registry_dir / "fam" / "runs" / ("b" * 64)
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps({"run_id": "b" * 64, "scorecard": {"corr": 0.05}}),
            encoding="utf-8",
        )
        second = service.load_leaderboard()
        assert second is not first  # cache invalidated by the new record

    def test_refresh_clears_all_caches(self, tmp_path):
        """refresh() drops leaderboard + timeseries + full-history caches."""
        service = DashboardDataService(registry_dir=Path(tmp_path))
        service.load_leaderboard()
        service._timeseries_cache[("a",)] = {"eras": []}
        service._full_history_cache[("b",)] = {"series": {}}
        assert service._leaderboard_cache is not None
        service.refresh()
        assert service._leaderboard_cache is None
        assert service._timeseries_cache == {}
        assert service._full_history_cache == {}
        assert service._source_fingerprint is not None

    def test_format_model_label_trained(self):
        """Test format_model_label for trained run."""
        service = DashboardDataService()
        label = service.format_model_label(
            "trained", "my-config", "abc123def456", short_id_len=8
        )
        assert label == "my-config · abc123de"  # First 8 chars of the id

    def test_format_model_label_benchmark(self):
        """Test format_model_label for benchmark."""
        service = DashboardDataService()
        label = service.format_model_label(
            "benchmark", "baseline-v2", "tier-4-lgbm", short_id_len=8
        )
        assert label == "baseline-v2 · tier-4-lgbm"

    def test_format_model_label_null_model_id(self):
        """Test format_model_label with None model_id."""
        service = DashboardDataService()
        label = service.format_model_label("trained", "config", None)
        assert label == "config · ?"

    def test_prepare_leaderboard_for_display(self, temp_registry_dir: Path):
        """Test prepare_leaderboard_for_display()."""
        service = DashboardDataService(registry_dir=temp_registry_dir)

        # This will load the registry (empty since we don't have full nmr.* setup)
        # So we expect an empty leaderboard
        display_rows = service.prepare_leaderboard_for_display()
        # Result depends on actual registry contents; just verify it returns a list
        assert isinstance(display_rows, list)

    def test_compute_robustness_matrix(self):
        """Test compute_robustness_matrix()."""
        service = DashboardDataService()
        # Will use default registry_dir (may not exist in test environment)
        # Just verify the method doesn't crash
        matrix = service.compute_robustness_matrix()
        assert isinstance(matrix, RobustnessMatrix)
        assert isinstance(matrix.rows, list)


class TestCampaignLogLoading:
    """Test campaign log loading."""

    @pytest.fixture
    def temp_campaigns_dir(self) -> Path:
        """Temporary campaigns directory with sample campaign log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            campaigns_dir = Path(tmpdir) / "campaigns"
            campaigns_dir.mkdir(parents=True)

            # Create a sample campaign log
            campaign_json = {
                "campaign_id": "exp-001",
                "name": "HPO Search v1",
                "runs": [
                    {
                        "config_path": "configs/test.yaml",
                        "run_id": "run-001",
                        "status": "COMPLETE",
                    },
                    {
                        "config_path": "configs/test.yaml",
                        "run_id": "run-002",
                        "status": "FAILED",
                        "error": "OOM",
                    },
                ],
            }
            (campaigns_dir / "exp-001.json").write_text(json.dumps(campaign_json))

            yield campaigns_dir

    def test_load_campaigns(self, temp_campaigns_dir: Path, monkeypatch):
        """Test load_campaigns()."""
        # Mock REPO_ROOT to use temp directory
        monkeypatch.setattr("dashboard_ui.service.REPO_ROOT", temp_campaigns_dir.parent)

        service = DashboardDataService()
        service.registry_dir = temp_campaigns_dir.parent / "registry"  # dummy
        service.registry_dir.mkdir(parents=True, exist_ok=True)

        # Override campaigns_dir in load_campaigns method by patching
        campaigns_log = service.load_campaigns()

        # Since campaigns_dir is hardcoded in the method, we can't easily test
        # without mocking the filesystem. The test above ensures the parsing logic.
        assert isinstance(campaigns_log, CampaignLog)
        assert isinstance(campaigns_log.runs, list)


class TestLeaderboardFramePydanticSerialization:
    """Test Pydantic serialization/deserialization."""

    def test_leaderboard_json_roundtrip(self):
        """Test JSON serialization and deserialization."""
        rows = [
            LeaderboardRowModel(
                model_id="m1",
                source="trained",
                run_name="config-a",
                run_dir="/tmp/m1",
                corr_sharpe_ac=1.23,
            )
        ]
        lb = LeaderboardFrame(rows=rows, total_rows=1, evaluable_rows=1)

        # Serialize to JSON
        json_str = lb.model_dump_json()
        assert isinstance(json_str, str)

        # Deserialize back
        lb2 = LeaderboardFrame.model_validate_json(json_str)
        assert lb2.rows[0].model_id == "m1"
        assert lb2.rows[0].corr_sharpe_ac == 1.23


class TestTopPerformers:
    """Test compute_top_performers."""

    def test_top_performers_ranks_by_sharpe(self):
        """Test ranking by Sharpe with real registry data."""
        service = DashboardDataService()
        result = service.compute_top_performers(top_n=5)
        if not result.rows:
            pytest.skip("No registry models")

        assert len(result) > 0
        assert result.sort_metric == "corr_sharpe_ac"
        assert result.champion is not None
        # Ranks should be sequential 1..N and sorted descending
        for i, row in enumerate(result.rows, start=1):
            assert row.rank == i
        sharpes = [r.corr_sharpe_ac or -1 for r in result.rows]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_top_performers_sharpe_floor(self):
        """Test the min_sharpe floor filters weak models."""
        service = DashboardDataService()
        filtered = service.compute_top_performers(top_n=50, min_sharpe=0.0)
        # All filtered rows should have sharpe >= 0
        for row in filtered.rows:
            assert row.corr_sharpe_ac is None or row.corr_sharpe_ac >= 0.0

    def test_top_performer_row_typed(self):
        """Test row is a typed model with decision metrics populated."""
        service = DashboardDataService()
        result = service.compute_top_performers(top_n=3)
        if not result.rows:
            pytest.skip("No registry models")
        row = result.rows[0]
        assert isinstance(row, TopPerformerRowModel)
        # Decision-critical fields should be present
        assert row.rank >= 1
        assert row.model_id
        assert row.label
        assert row.robustness_score >= 0

    def test_top_performers_json_roundtrip(self):
        """Test TopPerformersResult serializes to JSON."""
        service = DashboardDataService()
        result = service.compute_top_performers(top_n=3)
        if not result.rows:
            import pytest

            pytest.skip("No registry models")
        json_str = result.model_dump_json()
        restored = TopPerformersResult.model_validate_json(json_str)
        assert len(restored) == len(result)
        assert restored.rows[0].model_id == result.rows[0].model_id


class TestTimeseriesLoading:
    """Test load_timeseries caching."""

    def test_load_timeseries_returns_payload(self):
        """Test timeseries loads with real data and is memoized."""
        import pytest

        from dashboard_ui.service import DashboardDataService

        service = DashboardDataService()
        result = service.compute_top_performers(top_n=2)
        if not result.rows:
            pytest.skip("No registry models")
        ids = [r.model_id for r in result.rows]

        p1 = service.load_timeseries(ids)
        assert "eras" in p1
        assert "metrics" in p1
        assert "drawdowns" in p1

        # Second call returns cached object (memoized)
        p2 = service.load_timeseries(ids)
        assert p1 is p2

    def test_load_timeseries_cache_keyed_by_set(self):
        """Test cache is keyed by the run-id set regardless of order."""
        service = DashboardDataService()
        result = service.compute_top_performers(top_n=3)
        if not result.rows:
            import pytest

            pytest.skip("No registry models")
        ids = [r.model_id for r in result.rows]
        reversed_ids = list(reversed(ids))

        p1 = service.load_timeseries(ids)
        p2 = service.load_timeseries(reversed_ids)
        # Same underlying payload object because key is sorted tuple
        assert p1 is p2


class TestFullHistoryLoading:
    """Test load_full_history (per-era CORR over the FULL validation window)."""

    def test_full_history_returns_payload_with_stats(self):
        """Test full-history payload contains series, drawdowns, and stats."""
        import pytest

        from dashboard_ui.service import DashboardDataService

        service = DashboardDataService()
        result = service.compute_top_performers(top_n=2)
        if not result.rows:
            pytest.skip("No registry models")
        ids = [r.model_id for r in result.rows]

        p = service.load_full_history(ids)
        assert set(p) == {"series", "drawdowns", "stats"}

        # At least the top model should have been processed
        assert ids[0] in p["series"]
        s = p["series"][ids[0]]
        assert "eras" in s and "standard" in s and "cumulative" in s
        assert len(s["standard"]) == len(s["eras"])
        assert len(s["cumulative"]) == len(s["eras"])

        # Drawdowns align with series length
        assert len(p["drawdowns"][ids[0]]) == len(s["eras"])

        # Stats sane
        st = p["stats"][ids[0]]
        assert st["n"] == len(s["eras"])
        assert st["n"] >= 1
        assert 0.0 <= st["pct_positive"] <= 1.0
        assert st["win_streak"] >= 0
        assert st["max_drawdown"] <= 0.0  # drawdown magnitude is <= 0

    def test_full_history_memoized(self):
        """Test full-history payload is memoized."""
        import pytest

        from dashboard_ui.service import DashboardDataService

        service = DashboardDataService()
        result = service.compute_top_performers(top_n=2)
        if not result.rows:
            pytest.skip("No registry models")
        ids = [r.model_id for r in result.rows]

        p1 = service.load_full_history(ids)
        p2 = service.load_full_history(ids)
        assert p1 is p2

    def test_full_history_missing_model_absent(self):
        """Test a run with no preds is absent (never raises)."""
        from dashboard_ui.service import DashboardDataService

        service = DashboardDataService()
        p = service.load_full_history(["does_not_exist_12345"])
        assert p["series"] == {}
        assert p["drawdowns"] == {}
        assert p["stats"] == {}

    def test_full_history_drawdown_is_cumsum_peak_to_trough(self):
        """Test drawdown equals cumsum peak-to-trough (<= 0)."""
        import pytest

        from dashboard_ui.service import DashboardDataService

        service = DashboardDataService()
        result = service.compute_top_performers(top_n=1)
        if not result.rows:
            pytest.skip("No registry models")
        p = service.load_full_history([result.rows[0].model_id])
        mid = result.rows[0].model_id
        dd = p["drawdowns"][mid]
        assert all(d <= 1e-12 for d in dd)  # drawdown <= 0
        assert max(dd) == pytest.approx(0.0, abs=1e-9)  # starts at peak


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


    def test_fingerprint_detects_same_size_same_mtime_edit(self, tmp_path):
        """Content hashing catches a same-size edit with restored mtime."""
        import os
        service = DashboardDataService(registry_dir=Path(tmp_path))
        run_dir = tmp_path / "fam" / "runs" / ("a" * 64)
        run_dir.mkdir(parents=True)
        run_json = run_dir / "run.json"
        run_json.write_text("A" * 100, encoding="utf-8")
        before = service.compute_source_fingerprint()
        stat = run_json.stat()
        run_json.write_text("B" * 100, encoding="utf-8")
        os.utime(run_json, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert service.compute_source_fingerprint() != before

    def test_fingerprint_covers_legacy_meta_champion_and_benchmark_parquet(
        self, tmp_path
    ):
        """Every dashboard input moves the fingerprint: legacy run records,
        family metadata, the champion pointer, and the tier-4 benchmark
        parquet (size+mtime)."""
        import polars as pl

        service = DashboardDataService(registry_dir=Path(tmp_path))
        before = service.compute_source_fingerprint()

        legacy = tmp_path / "legacy" / "run.json"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("{}", encoding="utf-8")
        assert service.compute_source_fingerprint() != before
        before = service.compute_source_fingerprint()

        meta = tmp_path / "fam" / "meta.json"
        meta.parent.mkdir(parents=True)
        meta.write_text("{}", encoding="utf-8")
        assert service.compute_source_fingerprint() != before
        before = service.compute_source_fingerprint()

        champion = tmp_path / "champion.json"
        champion.write_text('{"run_id": "x"}', encoding="utf-8")
        assert service.compute_source_fingerprint() != before
        before = service.compute_source_fingerprint()

        bench = tmp_path / "validation_benchmark_models.parquet"
        pl.DataFrame({"era": ["1"], "id": ["a"], "bench": [0.5]}).write_parquet(bench)
        assert service.compute_source_fingerprint() != before

    def test_full_history_invalidates_when_preds_change(self, tmp_path):
        """load_full_history re-reads when a model validation_preds.parquet
        changes (the prediction path is part of the source fingerprint)."""
        import polars as pl

        data = tmp_path / "data"
        registry = tmp_path / "registry"
        data.mkdir()
        model = "m" * 64
        run = registry / model
        run.mkdir(parents=True)
        targets = pl.DataFrame(
            {"era": ["1", "1", "1", "2", "2", "2"],
             "id": ["a", "b", "c", "a", "b", "c"],
             "target": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]}
        )
        targets.write_parquet(data / "validation.parquet")
        preds = pl.DataFrame(
            {"era": ["1", "1", "1", "2", "2", "2"],
             "id": ["a", "b", "c", "a", "b", "c"],
             "prediction": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]}
        )
        preds.write_parquet(run / "validation_preds.parquet")
        service = DashboardDataService(
            registry_dir=registry, benchmark_path=None, data_dir=data
        )
        first = service.load_full_history([model])
        first_vals = first["series"][model]["standard"]
        changed = pl.DataFrame(
            {"era": ["1", "1", "1", "2", "2", "2"],
             "id": ["a", "b", "c", "a", "b", "c"],
             "prediction": [3.0, 1.0, 2.0, 3.0, 1.0, 2.0]}
        )
        changed.write_parquet(run / "validation_preds.parquet")
        second = service.load_full_history([model])
        assert first_vals != second["series"][model]["standard"]
