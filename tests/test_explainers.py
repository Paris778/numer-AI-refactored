"""Tests for nmr.explainers module."""

import pytest

from nmr.explainers import (
    ModelProfile,
    get_all_profiles,
    get_model_profile,
    get_profiles_by_architecture,
    get_profiles_by_tag,
)


class TestModelProfile:
    """Test ModelProfile data class."""

    def test_profile_creation(self):
        """Test creating a ModelProfile."""
        profile = ModelProfile(
            model_id="test-001",
            run_name="Test Model",
            architecture="tree",
            universe="medium",
            target="20d",
            description="A test model",
            long_description="Longer description of the test model",
            tags=["stable"],
            context="Test context",
            risk_profile="moderate",
            diversification="moderate",
            research_intent="Testing",
        )
        assert profile.model_id == "test-001"
        assert profile.summary == "Test Model (TREE)"

    def test_get_model_profile_known(self):
        """Test looking up a known profile."""
        profile = get_model_profile("lgbm_standard_v1")
        assert profile is not None
        assert profile.model_id == "lgbm_standard_v1"
        assert "baseline" in profile.description.lower()

    def test_get_model_profile_unknown(self):
        """Test looking up an unknown profile."""
        profile = get_model_profile("does_not_exist")
        assert profile is None

    def test_get_all_profiles(self):
        """Test getting all profiles."""
        profiles = get_all_profiles()
        assert len(profiles) >= 6
        assert all(isinstance(p, ModelProfile) for p in profiles)

    def test_filter_by_tag(self):
        """Test filtering by tag."""
        profiles = get_profiles_by_tag("stable")
        assert len(profiles) > 0
        assert all("stable" in p.tags for p in profiles)

    def test_filter_by_architecture(self):
        """Test filtering by architecture."""
        profiles = get_profiles_by_architecture("tree")
        assert len(profiles) > 0
        assert all(p.architecture == "tree" for p in profiles)

    def test_dynamic_profile_generation(self):
        """Test that a dynamic profile can be built from registry metadata."""
        from dashboard_ui.service import DashboardDataService

        service = DashboardDataService()
        lb = service.load_leaderboard()
        if not lb.rows:
            pytest.skip("No registry models available")

        first_id = lb.rows[0].model_id
        profile = get_model_profile(first_id)
        assert profile is not None
        assert profile.model_id == first_id
        assert profile.summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
