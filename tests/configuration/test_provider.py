from importlib.resources import files as _files
from pathlib import Path

import pytest

import seismometer.builder.resources
import seismometer.configuration.config as undertest

BUILDER_CONFIG = _files(seismometer.builder.resources) / "config.yml"


class TestConfigProvider:
    def test_builder_default_template_is_binary(self):
        config = undertest.ConfigProvider(BUILDER_CONFIG)
        assert config.template.name == "binary"
        assert config.template.value == BUILDER_CONFIG.parent / "classifier_bin.ipynb"

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
            ("output_dir", Path.cwd()),
            ("output_notebook", ""),
        ],
    )
    def test_build_config_is_valid_simple_object(self, property, value):
        config = undertest.ConfigProvider(BUILDER_CONFIG)
        actual = getattr(config, property)

        assert actual == value
