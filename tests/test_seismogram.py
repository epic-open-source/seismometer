import logging
from unittest.mock import Mock, patch

import pandas as pd
import pytest

import seismometer.seismogram  # noqa : needed for patching
from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import Cohort, Event, Metric, MetricDetails
from seismometer.data.loader import SeismogramLoader
from seismometer.seismogram import Seismogram


def get_test_config(tmp_path):
    mock_config = Mock(autospec=ConfigProvider)
    mock_config.output_dir.return_value
    mock_config.events = {
        "event1": Event(source="event1", display_name="event1", window_hr=1),
        "event2": Event(source="event1", display_name="event1", window_hr=2, aggregation_method="min"),
    }
    mock_config.metrics = {
        "Metric1": Metric(
            source="Metric1",
            display_name="Metric1",
            type="ordinal/categorical",
            group_keys=["Group1", "Group2"],
            metric_details=MetricDetails(values=["disagree", "neutral", "agree"]),
        ),
        "Metric2": Metric(
            source="Metric2",
            display_name="Metric2",
            type="Type2",
            group_keys="Group1",
            metric_details=MetricDetails(values=["disagree", "neutral", "agree"]),
        ),
        "Metric3": Metric(
            source="Metric3",
            display_name="Metric3",
            type="ordinal/categorical",
            group_keys="Group2",
            metric_details=MetricDetails(values=["cold", "warm", "hot"]),
        ),
    }
    mock_config.metric_groups = {"Group1": ["Metric1", "Metric2"], "Group2": ["Metric1", "Metric3"]}
    mock_config.metric_types = {"ordinal/categorical": ["Metric1", "Metric3"], "Type2": ["Metric2"]}
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
            "event1_Time": ["2022-01-01", "2022-01-02", "2022-01-03", "2021-12-31"],
            "event2_Value": [0, 1, 0, 1],
            "event2_Time": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "cohort1": [1, 0, 1, 0],
            "cohort2": [0, 1, 0, 1],
            "Metric1": ["disagree", "neutral", "agree", "neutral"],
            "Metric2": ["disagree", "agree", "agree", "disagree"],
            "Metric3": ["cold", "warm", "cold", "cold"],
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
            ([], [], "~12"),
            ([], ["entity", "time"], "~10"),
            ([], ["entity", "notacol"], "~11"),
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


class TestSeismogramSingleton:
    def test_singleton_behavior(self, fake_seismo):
        sg1 = Seismogram()
        sg2 = Seismogram()
        assert sg1 is sg2

    def test_kill_resets_instance(self, fake_seismo, tmp_path):
        sg1 = Seismogram()
        Seismogram.kill()

        # Re-initialize singleton with config/loader again
        config = get_test_config(tmp_path)
        loader = get_test_loader(config)
        sg2 = Seismogram(config, loader)

        assert sg1 is not sg2


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

    def test_event_merge_strategy_returns_expected(self, fake_seismo):
        sg = Seismogram()
        sg.config.events["event1"].merge_strategy = "last"
        result = sg.event_merge_strategy("event1")
        assert result == "last"

    def test_event_merge_strategy_raises_on_unknown_event(self, fake_seismo):
        sg = Seismogram()
        with pytest.raises(ValueError):
            sg.event_merge_strategy("unknown_event")


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

    def test_cohort_with_splits_raises_indexerror_and_logs_warning(self, fake_seismo, tmp_path, caplog):
        sg = Seismogram()
        sg.dataframe = get_test_data()

        # Inject a Cohort with a bad `splits` definition that will break resolve_cohorts
        bad_splits = [{"name": "Low", "upper": 1}, {"name": "High"}]  # Missing 'lower' or invalid logic
        broken_cohort = Cohort(source="cohort1", splits=bad_splits)
        sg.config.cohorts = [broken_cohort]

        # Patch resolve_cohorts to raise IndexError
        with patch("seismometer.seismogram.resolve_cohorts", side_effect=IndexError("bad split")), caplog.at_level(
            "WARNING", logger="seismometer"
        ):
            sg.create_cohorts()

        assert "Failed to resolve cohort" in caplog.text
        assert sg.cohort_cols == []


class TestSeismogramAttrs:
    @pytest.mark.parametrize(
        "attr_name",
        [
            "start_time",
            "end_time",
            "prediction_count",
            "entity_count",
            "event_types_count",
            "cohort_attribute_count",
            "feature_count",
            "target_event",
            "dataframe",
            "metric_groups",
            "metric_types",
        ],
    )
    def test_attribute_exists(self, fake_seismo, attr_name):
        sg = Seismogram()

        # Ensure attribute is available
        assert hasattr(sg, attr_name)

    def test_score_bins_uses_thresholds(self, fake_seismo):
        sg = Seismogram()
        sg.thresholds = [0.2, 0.8]
        assert sg.score_bins() == [0.0, 0.2, 0.8, 1.0]
        path = sg.output_path
        assert path == sg.config.output_dir


class TestSeismogramMetricExtraction:
    @pytest.mark.parametrize(
        "max_cat_size, expected_metrics",
        [
            (3, ["Metric1", "Metric3"]),
            (2, ["Metric3"]),
            (1, []),
        ],
    )
    def test_get_ordinal_categorical_metrics(self, fake_seismo, max_cat_size, expected_metrics):
        sg = Seismogram()
        sg.dataframe = get_test_data()
        result = sg.get_ordinal_categorical_metrics(max_cat_size=max_cat_size)
        assert result == expected_metrics

    @pytest.mark.parametrize(
        "max_cat_size, expected_groups",
        [
            (3, ["Group1", "Group2"]),
            (2, ["Group2"]),
            (1, []),
        ],
    )
    def test_get_ordinal_categorical_groups(self, fake_seismo, max_cat_size, expected_groups):
        sg = Seismogram()
        sg.dataframe = get_test_data()
        result = sg.get_ordinal_categorical_groups(max_cat_size=max_cat_size)
        assert result == expected_groups

    @pytest.mark.parametrize(
        "metric_name, max_cat_size, expected",
        [
            ("Metric1", 3, True),
            ("Metric2", 3, False),
            ("Metric3", 3, True),
            ("Metric1", 2, False),
            ("Metric2", 2, False),
            ("Metric3", 2, True),
        ],
    )
    def test_is_ordinal_categorical_metric(self, fake_seismo, metric_name, max_cat_size, expected):
        sg = Seismogram()
        sg.dataframe = get_test_data()
        result = sg._is_ordinal_categorical_metric(metric_name, max_cat_size=max_cat_size)
        assert result == expected


class TestSeismogramEventProperties:
    def test_all_event_properties_resolve_correctly(self, fake_seismo):
        sg = Seismogram()
        sg.config.target = "event1"
        sg.target_event = "event1"
        sg.config.output_list = ["score1"]
        sg.config.targets = ["event1", "event2"]
        sg.config.comparison_time = "comparison_time_col"
        sg.config.interventions = {"int1": {}}
        sg.config.outcomes = {"out1": {}}

        assert sg.target == "event1_Value"
        assert sg.time_zero == "event1_Time"
        assert sg.target_cols == ["event1", "event2"]
        assert sg.comparison_time == "comparison_time_col"
        assert sg.intervention == "int1"
        assert sg.outcome == "out1"

    def test_missing_intervention_and_outcome_raises_index_error(self, fake_seismo):
        sg = Seismogram()
        sg.config.interventions = {}
        sg.config.outcomes = {}

        with pytest.raises(IndexError):
            _ = sg.intervention
        with pytest.raises(IndexError):
            _ = sg.outcome


class TestSeismogramCopyConfig:
    def test_copy_config_metadata_assigns_attributes(self, fake_seismo):
        sg = Seismogram()
        sg.config.events = {"event1": Event(source="event1", window_hr=1)}
        sg.config.metrics = {"MetricA": Metric(source="MetricA", display_name="A")}
        sg.config.metric_groups = {"Group1": ["MetricA"]}
        sg.config.metric_types = {"some_type": ["MetricA"]}
        sg.config.target = "event1"
        sg.config.features = ["f1"]
        sg.config.entity_keys = ["entity"]
        sg.config.predict_time = "time"
        sg.config.output_list = ["score"]
        sg.config.config_dir = sg.config.config_dir

        # Act
        sg.copy_config_metadata()

        # Assert
        assert sg.predict_time == "time"
        assert sg.output_list == ["score"]
        assert sg.entity_keys == ["entity"]
        assert sg.metrics == sg.config.metrics
        assert sg.metric_groups == sg.config.metric_groups
        assert sg.metric_types == sg.config.metric_types
        assert sg.target_event == "event1"


class TestSeismogramGetBinaryTargets:
    def test_get_binary_targets_filters_properly(self, fake_seismo):
        sg = Seismogram()
        df = pd.DataFrame(
            {
                "event1_Value": [0, 1, 1, 0],
                "event2_Value": [0, 1, 2, 1],  # not binary
            }
        )
        sg.config.targets = ["event1", "event2"]
        sg.dataframe = df

        binary_targets = sg.get_binary_targets()
        assert binary_targets == ["event1_Value"]


class TestSeismogramLoadData:
    def test_load_data_sets_dataframe_and_cohort_groups(self, tmp_path):
        config = get_test_config(tmp_path)
        loader = get_test_loader(config)

        # write a real metadata.json file
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text('{"thresholds": [0.2, 0.8], "modelname": "TestModel"}')
        config.metadata_path = metadata_path

        sg = Seismogram(config, loader)

        df = pd.DataFrame(
            {
                "entity": [1, 2],
                "time": pd.to_datetime(["2022-01-01", "2022-01-02"]),
                "cohort1": pd.Series([0, 1], dtype="category"),
                "event1_Value": [0, 1],
                "event1_Time": pd.to_datetime(["2022-01-01", "2022-01-02"]),
            }
        )

        loader.load_data.return_value = df
        config.cohorts = [Cohort(source="cohort1")]
        config.features = []
        config.events = {"event1": Mock(window_hr=1)}
        config.prediction_columns = ["entity", "time"]

        sg.load_data()

        assert "cohort1" in sg.available_cohort_groups
        assert isinstance(sg.dataframe, pd.DataFrame)

    def test_load_data_does_not_reload_if_data_present(self, fake_seismo):
        sg = Seismogram()
        sg.dataframe = pd.DataFrame({"entity": [1]})

        with patch.object(sg, "_load_metadata") as mock_meta, patch.object(sg.dataloader, "load_data") as mock_loader:
            sg.load_data()
            mock_meta.assert_not_called()
            mock_loader.assert_not_called()

    def test_load_data_forces_reload_if_reset_true(self, tmp_path):
        config = get_test_config(tmp_path)
        loader = get_test_loader(config)

        # Create a real metadata file
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text('{"thresholds": [0.2], "modelname": "MyModel"}')
        config.metadata_path = metadata_path

        sg = Seismogram(config, loader)

        sg.dataframe = pd.DataFrame({"entity": [1]})
        df = pd.DataFrame(
            {"entity": [2], "time": pd.to_datetime(["2022-01-01"]), "cohort1": pd.Series([1], dtype="category")}
        )

        loader.load_data.return_value = df
        config.cohorts = [Cohort(source="cohort1")]
        config.features = []
        config.events = {"event1": Mock(window_hr=1)}
        config.prediction_columns = ["entity", "time"]

        sg.load_data(reset=True)

        assert sg.dataframe["entity"].iloc[0] == 2

    def test_warns_and_defaults_on_missing_thresholds(self, tmp_path, fake_seismo, caplog):
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text('{"modelname": "TestModel"}')

        sg = Seismogram()
        sg.config.metadata_path = metadata_path

        with caplog.at_level("WARNING", logger="seismometer"):
            sg._load_metadata()

        assert sg.thresholds == [0.8, 0.5]
        assert "No thresholds set in metadata.json" in caplog.text
