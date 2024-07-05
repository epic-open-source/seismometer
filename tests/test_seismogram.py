from unittest.mock import Mock, patch

import pandas as pd
import pytest

import seismometer.seismogram
from seismometer.configuration import ConfigProvider
from seismometer.seismogram import Seismogram


def fake_load_config(self, *args, definitions=None):
    mock_config = Mock(autospec=ConfigProvider)
    mock_config.output_dir.return_value
    mock_config.events = ["event1", "event2"]
    mock_config.cohorts = ["cohort1", "cohort2", "cohort3"]
    mock_config.features = ["one"]

    self.config = mock_config
    self.template = "TestTemplate"
    self.entity_keys = ["entity"]
    self.predict_time = "time"


def get_test_data():
    return pd.DataFrame(
        {
            "entity": ["A", "A", "B", "C"],
            "prediction": [1, 2, 3, 4],
            "time": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
        }
    )


@pytest.fixture
def fake_seismo(tmp_path):
    with patch.object(seismometer.seismogram, "loader_factory"), patch.object(
        Seismogram, "load_config", fake_load_config
    ):
        Seismogram(config_path=tmp_path / "config", output_path=tmp_path / "output")
    yield
    Seismogram.kill()


class Test__set_df_counts:
    @pytest.mark.parametrize(
        "attr_name,expected",
        [
            ("prediction_count", 4),
            ("entity_count", 3),
            ("start_time", "2022-01-01"),
            ("end_time", "2022-01-04"),
            ("event_types_count", 2),
            ("cohort_attribute_count", 3),
        ],
    )
    def test_set_df_count_freezes_values(self, attr_name, expected, fake_seismo, tmp_path):
        # Arrange
        sg = Seismogram()
        sg.dataframe = get_test_data()
        sg.config.features = ["one"]

        # Act
        sg._set_df_counts()

        # Assert
        assert getattr(sg, attr_name) == expected

    @pytest.mark.parametrize(
        "features, predict_cols, expected",
        [
            ([], [], "~3"),
            ([], ["entity", "time"], "~1"),
            ([], ["entity", "notacol"], "~2"),
            (["feature1", "feature2"], [], 2),
            (["feature1", "feature2"], ["entity", "time"], 2),
            (["feature1", "feature2", "feature3"], [], 3),
        ],
    )
    def test_set_df_count_handles_feature_counts(self, features, predict_cols, expected, fake_seismo, tmp_path):
        # Arrange
        sg = Seismogram()
        sg.dataframe = get_test_data()
        sg.config.features = features
        sg.config.prediction_columns = predict_cols

        # Act
        sg._set_df_counts()

        # Assert
        assert sg.feature_count == expected
