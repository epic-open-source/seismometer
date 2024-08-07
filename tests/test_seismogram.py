import logging
from unittest.mock import Mock, patch

import pandas as pd
import pytest

import seismometer.seismogram  # noqa : needed for patching
from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import Cohort, Event
from seismometer.data.loader import SeismogramLoader
from seismometer.seismogram import Seismogram


def get_test_config(tmp_path):
    mock_config = Mock(autospec=ConfigProvider)
    mock_config.output_dir.return_value
    mock_config.events = {
        "event1": Event(source="event1", display_name="event1", window_hr=1),
        "event2": Event(source="event1", display_name="event1", window_hr=2, aggregation_method="min"),
    }
    mock_config.target = "event1"
    mock_config.entity_keys = ["entity"]
    mock_config.predict_time = "time"
    mock_config.cohorts = [Cohort(source=name) for name in ["cohort1", "cohort2"]]
    mock_config.features = ["one"]
    mock_config.config_dir = tmp_path / "config"
    mock_config.censor_min_count = 0

    return mock_config


def get_test_loader(config):
    mock_loader = Mock(autospec=SeismogramLoader)
    mock_loader.config = config

    return mock_loader


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
            "cohort1": [1, 0, 1, 0],
            "cohort2": [0, 1, 0, 1],
        }
    )


@pytest.fixture
def fake_seismo(tmp_path):
    config = get_test_config(tmp_path)
    loader = get_test_loader(config)
    Seismogram(config, loader)
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
            ("cohort_attribute_count", 2),
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
            ([], [], "~9"),
            ([], ["entity", "time"], "~7"),
            ([], ["entity", "notacol"], "~8"),
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


class TestSeismogramCreateCohorts:
    def test_create_cohorts_creates_str_list(self, fake_seismo, tmp_path):
        # Arrange
        sg = Seismogram()
        sg.dataframe = get_test_data()

        # Act
        sg.create_cohorts()

        # Assert
        assert sg.cohort_cols == ["cohort1", "cohort2"]

    def test_unseen_column_logs_warning_and_drops(self, fake_seismo, tmp_path, caplog):
        # Arrange
        sg = Seismogram()
        sg.dataframe = get_test_data()
        sg.config.cohorts = sg.config.cohorts + [Cohort(source="unseen")]

        # Act
        with caplog.at_level(logging.WARNING, logger="seismometer"):
            sg.create_cohorts()

        # Assert
        assert len(caplog.records) == 1
        assert "Source column unseen" in caplog.text
        assert sg.cohort_cols == ["cohort1", "cohort2"]

    @patch.object(seismometer.seismogram, "MAXIMUM_NUM_COHORTS", 2)
    def test_large_cardinality_warns_and_drops(self, fake_seismo, tmp_path, caplog):
        # Arrange
        sg = Seismogram()
        sg.dataframe = get_test_data()
        sg.dataframe["card3"] = [0, 1, 2, 2]
        sg.config.cohorts = sg.config.cohorts + [Cohort(source="card3")]

        # Act
        with caplog.at_level(logging.WARNING, logger="seismometer"):
            sg.create_cohorts()

        # Assert
        assert len(caplog.records) == 1
        assert "unique" in caplog.text
        assert "card3" in caplog.text
        assert sg.cohort_cols == ["cohort1", "cohort2"]

    def test_censor_limit_logs_warning_and_drops(self, fake_seismo, tmp_path, caplog):
        # Arrange
        sg = Seismogram()
        sg.dataframe = get_test_data()
        sg.dataframe["uniqVals"] = [0, 1, 2, 3]
        sg.config.censor_min_count = 1
        sg.config.cohorts = sg.config.cohorts + [Cohort(source="uniqVals")]

        # Act
        with caplog.at_level(logging.WARNING, logger="seismometer"):
            sg.create_cohorts()

        # Assert
        assert len(caplog.records) == 1
        assert "No cohort" in caplog.text
        assert "uniqVals" in caplog.text
        assert sg.cohort_cols == ["cohort1", "cohort2"]

    def test_censor_some_logs_warning(self, fake_seismo, tmp_path, caplog):
        # Arrange
        sg = Seismogram()
        sg.dataframe = get_test_data()
        sg.dataframe["rareVals"] = pd.Series([0, 0, 0, 1], dtype=bool)
        sg.config.censor_min_count = 1
        sg.config.cohorts = sg.config.cohorts + [Cohort(source="rareVals")]

        # Act
        with caplog.at_level(logging.DEBUG, logger="seismometer"):
            sg.create_cohorts()

        # Assert
        assert len(caplog.records) == 2
        assert "Some cohorts" in caplog.records[0].message
        assert "rareVals" in caplog.records[0].message
        assert sg.cohort_cols == ["cohort1", "cohort2", "rareVals"]
