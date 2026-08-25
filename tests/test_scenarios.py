"""Tests for nmr.scenarios module."""

import pytest

from nmr.scenarios import (
    AllocationConstraints,
    AllocationScenario,
    ModelAllocation,
    ScenarioEngine,
)


class TestAllocationConstraints:
    """Test AllocationConstraints."""

    def test_default_constraints(self):
        """Test default constraint values."""
        c = AllocationConstraints()
        assert c.min_sharpe_threshold == 0.5
        assert c.max_single_weight == 0.25
        assert c.min_models == 3
        assert c.max_models == 10


class TestModelAllocation:
    """Test ModelAllocation."""

    def test_allocation_creation(self):
        """Test ModelAllocation creation."""
        a = ModelAllocation(
            model_id="m1",
            weight=0.25,
            corr_sharpe_ac=1.2,
            conviction="high",
            reason="Top Sharpe",
        )
        assert a.model_id == "m1"
        assert a.weight == 0.25


class TestAllocationScenario:
    """Test AllocationScenario."""

    def test_empty_scenario(self):
        """Test empty scenario."""
        s = AllocationScenario(scenario_id="empty", name="Empty")
        assert s.num_models == 0
        assert s.total_weight == 0.0
        assert not s.is_valid

    def test_valid_scenario(self):
        """Test valid scenario."""
        s = AllocationScenario(
            scenario_id="test",
            name="Test",
            allocations=[
                ModelAllocation("m1", 0.20, 1.0, "high", "reason"),
                ModelAllocation("m2", 0.20, 0.9, "high", "reason"),
                ModelAllocation("m3", 0.20, 0.8, "medium", "reason"),
            ],
        )
        assert s.num_models == 3
        assert s.is_valid


class TestScenarioEngine:
    """Test ScenarioEngine."""

    @pytest.fixture
    def base_models(self):
        """Sample base models."""
        return [
            {"model_id": "m1", "corr_sharpe_ac": 1.2, "max_drawdown": -0.05},
            {"model_id": "m2", "corr_sharpe_ac": 1.0, "max_drawdown": -0.08},
            {"model_id": "m3", "corr_sharpe_ac": 0.8, "max_drawdown": -0.12},
            {"model_id": "m4", "corr_sharpe_ac": 0.6, "max_drawdown": -0.15},
        ]

    @pytest.fixture
    def engine(self, base_models):
        """ScenarioEngine fixture."""
        return ScenarioEngine(base_models)

    def test_apply_constraints(self, engine, base_models):
        """Test constraint-based scenario."""
        constraints = AllocationConstraints(min_sharpe_threshold=0.7)
        scenario = engine.scenario_apply_constraints(base_models, constraints)

        assert len(scenario.allocations) == 3
        assert all(a.corr_sharpe_ac >= 0.7 for a in scenario.allocations)

    def test_remove_model(self, engine, base_models):
        """Test remove model scenario."""
        baseline = engine.scenario_apply_constraints(
            base_models, AllocationConstraints()
        )
        scenario = engine.scenario_remove_model(baseline, "m1", rebalance=True)

        assert "m1" not in [a.model_id for a in scenario.allocations]
        assert scenario.sharpe_vs_baseline is not None

    def test_reweight_model(self, engine, base_models):
        """Test reweight model scenario."""
        baseline = engine.scenario_apply_constraints(
            base_models, AllocationConstraints()
        )
        scenario = engine.scenario_reweight_model(baseline, "m1", 0.35)

        m1 = next(a for a in scenario.allocations if a.model_id == "m1")
        assert m1.weight == 0.35

    def test_regime_robust(self, engine, base_models):
        """Test regime-robust scenario."""
        scenario = engine.scenario_regime_robust(base_models)

        assert len(scenario.allocations) > 0
        assert scenario.name == "Regime-Robust Portfolio"

    def test_portfolio_metrics_computed(self, engine, base_models):
        """Test portfolio metrics computation."""
        scenario = engine.scenario_apply_constraints(
            base_models, AllocationConstraints()
        )

        assert scenario.portfolio_sharpe is not None
        assert scenario.portfolio_max_drawdown is not None
        assert scenario.portfolio_volatility is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
