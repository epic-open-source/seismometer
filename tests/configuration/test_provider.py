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
            ("prediction_columns", ["Age", "Input", "Score", "ScoringTime", "encounter_id", "id"]),
            ("censor_min_count", 15),
            ("output_notebook", "classifier_bin.ipynb"),
        ],
    )
    def test_testconfig_is_valid_simple_object(self, property, value, res):
        config = undertest.ConfigProvider(res / TEST_CONFIG)
        actual = getattr(config, property)

        assert actual == value

    def test_testconfig_uses_tmp(self, tmp_path, res):
        config = undertest.ConfigProvider(res / TEST_CONFIG)
        assert config.output_dir == tmp_path / "outputs"
