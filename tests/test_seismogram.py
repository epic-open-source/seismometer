from unittest.mock import Mock, patch

import pandas as pd
import pytest

import seismometer.seismogram
from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import Event
from seismometer.seismogram import Seismogram


def fake_load_config(self, *args, definitions=None):
    mock_config = Mock(autospec=ConfigProvider)
    mock_config.output_dir.return_value
    mock_config.events = {
        "event1": Event(source="event1", display_name="event1", window_hr=1),
        "event2": Event(source="event1", display_name="event1", window_hr=2, aggregation_method="min"),
    }
    mock_config.primary_target = "event1"
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
            "event1_Value": [0, 1, 0, -1],
            "event1_Time": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "event2_Value": [0, 1, 0, 1],
            "event2_Time": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
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
            ([], [], "~7"),
            ([], ["entity", "time"], "~5"),
            ([], ["entity", "notacol"], "~6"),
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


class TestSeismogramDataMethod:
    def test_data_with_named_target(self, fake_seismo, tmp_path):
        # Arrange
        sg = Seismogram()
        sg.dataframe = get_test_data()

        assert len(sg.data("event2")) == 4

    def test_data_filters_target_events(self, fake_seismo, tmp_path):
        # Arrange
        sg = Seismogram()
        sg.dataframe = get_test_data()

        assert len(sg.data("event1")) == 3

    def test_data_defaults_to_primary_target(self, fake_seismo, tmp_path):
        # Arrange
        sg = Seismogram()
        sg.target_event = "event1"
        sg.dataframe = get_test_data()

        assert len(sg.data()) == 3


class TestSeismogramConfigRetrievalMethods:
    @pytest.mark.parametrize(
        "method_name,method_args,expected",
        [
            ("event_aggregation_method", ["event1"], "max"),
            ("event_aggregation_method", ["event2"], "min"),
            ("event_aggregation_window_hours", ["event1"], 1),
            ("event_aggregation_window_hours", ["event2"], 2),
        ],
    )
    def test_methods_get_value_from_config(self, method_name, method_args, expected, fake_seismo, tmp_path):
        # Arrange
        sg = Seismogram()
        sg.dataframe = get_test_data()

        # Act
        sg._set_df_counts()

        # Assert
        assert getattr(sg, method_name)(*method_args) == expected

    @pytest.mark.parametrize(
        "method_name,method_args,exception_type",
        [
            ("event_aggregation_method", ["not_an_event"], ValueError),
            ("event_aggregation_window_hours", ["not_an_event"], ValueError),
        ],
    )
    def test_raises_exception_when_missing_from_config(
        self, method_name, method_args, exception_type, fake_seismo, tmp_path
    ):
        # Arrange
        sg = Seismogram()
        sg.dataframe = get_test_data()

        # Act
        sg._set_df_counts()

        # Assert
        with pytest.raises(exception_type):
            getattr(sg, method_name)(*method_args)
