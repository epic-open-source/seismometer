from pathlib import Path

import pytest
from conftest import res  # noqa:  flake cant detect fixture usage

import seismometer.configuration.config as undertest

TEST_CONFIG = Path("config") / "config.yml"


# Could share temp directory across all tests
@pytest.mark.usefixtures("tmp_as_current")
class TestConfigProvider:
    @pytest.mark.parametrize(
        "property, value",
        [
            ("info_dir", Path("outputs")),
            ("prediction_path", Path("/path/to/data/files/predictions.parquet")),
            ("event_path", Path("/path/to/data/files/events.parquet")),
            ("metadata_path", Path("/path/to/data/files/metadata.json")),
            ("entity_id", "id"),
            ("context_id", "encounter_id"),
            ("entity_keys", ["id", "encounter_id"]),
            ("target", "TargetLabel"),
            ("output", "Score"),
            ("output_list", ["Score"]),
            ("predict_time", "ScoringTime"),
            ("features", ["Input"]),
            ("comparison_time", "ScoringTime"),
            ("outcomes", {}),
            ("interventions", {}),
            (
                "prediction_columns",
                ["Age", "Input", "Score", "ScoringTime", "department", "encounter_id", "facility", "id"],
            ),
            ("censor_min_count", 15),
        ],
    )
    def test_testconfig_is_valid_simple_object(self, property, value, res):
        config = undertest.ConfigProvider(res / TEST_CONFIG)
        actual = getattr(config, property)

        assert actual == value

    def test_testconfig_uses_tmp(self, tmp_path, res):
        config = undertest.ConfigProvider(res / TEST_CONFIG)
        assert config.output_dir == tmp_path / "outputs"

    @pytest.mark.parametrize(
        "outputs, output_list",
        [
            (["Score2"], ["Score", "Score2"]),
            (["Score"], ["Score"]),
            (["Score2", "Score"], ["Score2", "Score"]),
            (["Score1", "Score2"], ["Score", "Score1", "Score2"]),
        ],
    )
    def test_provider_groups_primary_output_with_output_list(self, outputs, output_list, res):
        config = undertest.ConfigProvider(res / TEST_CONFIG)
        config.usage.outputs = outputs
        assert config.output_list == output_list

    def test_cohort_hierarchies_property(self, res):
        config = undertest.ConfigProvider(res / TEST_CONFIG)
        assert config.cohort_hierarchies == config.usage.cohort_hierarchies


# ============================================================================
# ADDITIONAL EDGE CASE TESTS
# ============================================================================


class TestConfigProviderInitialization:
    """Test ConfigProvider initialization with optional parameters."""

    @pytest.mark.usefixtures("tmp_as_current")
    def test_initialization_with_all_optional_parameters(self, tmp_path, res):
        """Test ConfigProvider with all optional parameters specified."""
        custom_info_dir = tmp_path / "custom_info"
        custom_data_dir = res / "data"

        config = undertest.ConfigProvider(
            res / TEST_CONFIG,
            info_dir=custom_info_dir,
            data_dir=custom_data_dir,
        )

        # Paths are resolved to absolute paths, so compare resolved versions
        assert config.config.info_dir == Path(custom_info_dir).resolve()
        assert config.config.data_dir == Path(custom_data_dir).resolve()

    @pytest.mark.usefixtures("tmp_as_current")
    def test_initialization_with_definitions_dict(self, res):
        """Test ConfigProvider with pre-loaded definitions dictionary."""
        definitions = {
            "predictions": [{"name": "custom_feature", "display_name": "Custom Feature"}],
            "events": [{"name": "custom_event", "display_name": "Custom Event"}],
        }

        config = undertest.ConfigProvider(res / TEST_CONFIG, definitions=definitions)

        assert config.prediction_defs.predictions[0].name == "custom_feature"
        assert config.event_defs.events[0].name == "custom_event"

    @pytest.mark.usefixtures("tmp_as_current")
    def test_initialization_with_partial_optional_parameters(self, tmp_path, res):
        """Test ConfigProvider with only some optional parameters."""
        custom_data_dir = res / "data"

        config = undertest.ConfigProvider(res / TEST_CONFIG, data_dir=custom_data_dir)

        assert config.config.data_dir == custom_data_dir
        # Other parameters should use defaults from config file
        assert config.entity_id == "id"


class TestConfigProviderFileNotFound:
    """Test ConfigProvider with file not found scenarios."""

    @pytest.mark.usefixtures("tmp_as_current")
    def test_missing_event_definition_file_uses_empty_list(self, tmp_path, res):
        """Test that missing event definition file results in empty events list."""
        config = undertest.ConfigProvider(res / TEST_CONFIG)

        # Config specifies event_definition but file doesn't exist
        # Should handle gracefully and return empty EventDictionary
        event_defs = config.event_defs
        assert event_defs is not None
        assert isinstance(event_defs.events, list)

    @pytest.mark.usefixtures("tmp_as_current")
    def test_missing_prediction_definition_file_uses_empty_list(self, tmp_path, res):
        """Test that missing prediction definition file results in empty predictions list."""
        config = undertest.ConfigProvider(res / TEST_CONFIG)

        # Should handle missing file gracefully
        prediction_defs = config.prediction_defs
        assert prediction_defs is not None
        assert isinstance(prediction_defs.predictions, list)


class TestUsagePropertyCaching:
    """Test usage property caching behavior."""

    @pytest.mark.usefixtures("tmp_as_current")
    def test_usage_property_is_cached(self, res):
        """Test that usage property is cached and not reloaded on each access."""
        config = undertest.ConfigProvider(res / TEST_CONFIG)

        # First access loads from file
        usage1 = config.usage

        # Second access should return cached instance
        usage2 = config.usage

        # Should be the exact same object (not just equal)
        assert usage1 is usage2

    @pytest.mark.usefixtures("tmp_as_current")
    def test_usage_property_loaded_during_init(self, res):
        """Test that usage is loaded during __init__ (not lazy loaded).

        Note: usage is accessed in _load_metrics() during __init__, so _usage
        is set during initialization, not on first property access.
        """
        config = undertest.ConfigProvider(res / TEST_CONFIG)

        # Usage is loaded during __init__ via _load_metrics()
        assert config._usage is not None


class TestLoadMetricsDeduplication:
    """Test _load_metrics() deduplication logic."""

    @pytest.mark.usefixtures("tmp_as_current")
    def test_metrics_with_duplicate_sources_are_deduplicated(self, caplog, res):
        """Test that metrics with duplicate sources trigger warning and are skipped."""
        from seismometer.configuration.model import Metric

        config = undertest.ConfigProvider(res / TEST_CONFIG)

        # Add duplicate metrics manually
        config.usage.metrics = [
            Metric(source="score1", display_name="Score 1", type="binary classification"),
            Metric(source="score1", display_name="Score 1 Duplicate", type="binary classification"),  # Duplicate
            Metric(source="score2", display_name="Score 2", type="binary classification"),
        ]

        with caplog.at_level("WARNING"):
            config._load_metrics()

        # Should log warning about duplicate
        assert "score1" in caplog.text

        # Should only have 2 metrics (first occurrence of score1, plus score2)
        assert len(config.metrics) == 2
        assert "score1" in config.metrics
        assert "score2" in config.metrics
        assert config.metrics["score1"].display_name == "Score 1"  # First one kept

    @pytest.mark.usefixtures("tmp_as_current")
    def test_metrics_grouped_by_group_keys(self, res):
        """Test that metrics are correctly grouped by group_keys."""
        from seismometer.configuration.model import Metric

        config = undertest.ConfigProvider(res / TEST_CONFIG)

        config.usage.metrics = [
            Metric(source="metric1", display_name="Metric 1", group_keys="group_a"),
            Metric(source="metric2", display_name="Metric 2", group_keys="group_a"),
            Metric(source="metric3", display_name="Metric 3", group_keys="group_b"),
        ]

        config._load_metrics()

        assert "group_a" in config.metric_groups
        assert sorted(config.metric_groups["group_a"]) == ["metric1", "metric2"]
        assert "group_b" in config.metric_groups
        assert config.metric_groups["group_b"] == ["metric3"]

    @pytest.mark.usefixtures("tmp_as_current")
    def test_metrics_with_multiple_group_keys(self, res):
        """Test that metrics with multiple group_keys appear in all groups."""
        from seismometer.configuration.model import Metric

        config = undertest.ConfigProvider(res / TEST_CONFIG)

        config.usage.metrics = [
            Metric(source="metric1", display_name="Metric 1", group_keys=["group_a", "group_b"]),
            Metric(source="metric2", display_name="Metric 2", group_keys="group_a"),
        ]

        config._load_metrics()

        assert "metric1" in config.metric_groups["group_a"]
        assert "metric1" in config.metric_groups["group_b"]
        assert "metric2" in config.metric_groups["group_a"]
        assert "metric2" not in config.metric_groups.get("group_b", [])


class TestEventTypesFallback:
    """Test event_types() fallback logic.

    NOTE: Bug found in event_types() implementation - see BUGS_FOUND.md Bug #6.
    Tests below work around the bug to test fallback logic only.
    """

    @pytest.mark.usefixtures("tmp_as_current")
    def test_event_types_returns_primary_target_when_no_events(self, res):
        """Test that event_types returns primary_target when events list is empty."""
        config = undertest.ConfigProvider(res / TEST_CONFIG)

        # Clear events to test fallback
        config.usage.events = []

        result = config.event_types()

        # Should return primary_target as fallback
        assert result == config.usage.primary_target

    @pytest.mark.usefixtures("tmp_as_current")
    def test_event_types_works_with_empty_events_dict(self, res):
        """Test event_types fallback when events dict is empty.

        Note: Testing with non-empty events skipped due to Bug #6 in event_types().
        The implementation iterates over dict keys instead of values, causing AttributeError.
        """
        config = undertest.ConfigProvider(res / TEST_CONFIG)

        # Clear events to test fallback without triggering Bug #6
        config.usage.events = []

        result = config.event_types()

        # Should return primary_target when no events
        assert result == config.usage.primary_target


class TestInvalidConfigFilePath:
    """Test invalid config file path handling."""

    @pytest.mark.usefixtures("tmp_as_current")
    def test_nonexistent_config_file_raises_error(self, tmp_path):
        """Test that nonexistent config file raises appropriate error."""
        nonexistent_path = tmp_path / "nonexistent" / "config.yml"

        with pytest.raises(FileNotFoundError):
            undertest.ConfigProvider(nonexistent_path)

    @pytest.mark.usefixtures("tmp_as_current")
    def test_invalid_config_directory_raises_error(self, tmp_path):
        """Test that invalid config directory raises error."""
        # Create an empty directory with no config.yml
        empty_dir = tmp_path / "empty_config_dir"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            undertest.ConfigProvider(empty_dir)
