import logging
from unittest.mock import Mock, patch

import pandas as pd
import pytest

import seismometer.seismogram  # noqa : needed for patching
from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import (
    Cohort,
    CohortHierarchy,
    Event,
    FilterConfig,
    FilterRange,
    Metric,
    MetricDetails,
)
from seismometer.data.filter import FilterRule
from seismometer.data.loader import SeismogramLoader
from seismometer.seismogram import MAXIMUM_NUM_COHORTS, Seismogram


@pytest.fixture(autouse=True, scope="class")
def disable_min_rows_for_filterrule():
    original = FilterRule.MIN_ROWS
    FilterRule.MIN_ROWS = 0
    yield
    FilterRule.MIN_ROWS = original


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
    mock_config.cohort_hierarchies = None

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
    def test_methods_get_value_from_filter_config(self, method_name, method_args, expected, fake_seismo, tmp_path):
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
    def test_raises_exception_when_missing_from_filter_config(
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
        config.usage.load_time_filters = None
        config.cohort_hierarchies = []

        sg.load_data()

        assert "cohort1" in sg.available_cohort_groups
        assert isinstance(sg.dataframe, pd.DataFrame)
        Seismogram.kill()

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
        config.usage.load_time_filters = None
        config.cohort_hierarchies = []

        sg.load_data(reset=True)

        assert sg.dataframe["entity"].iloc[0] == 2
        Seismogram.kill()

    def test_warns_and_defaults_on_missing_thresholds(self, tmp_path, fake_seismo, caplog):
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text('{"modelname": "TestModel"}')

        sg = Seismogram()
        sg.config.metadata_path = metadata_path

        with caplog.at_level("WARNING", logger="seismometer"):
            sg._load_metadata()

        assert sg.thresholds == [0.8, 0.5]
        assert "No thresholds set in metadata.json" in caplog.text


@pytest.mark.usefixtures("disable_min_rows_for_filterrule")
class TestSeismogramFilterConfigs:
    def test_keep_top_filters_excessive_categories(self, fake_seismo):
        sg = Seismogram()
        # Create MAX + 1 categories with descending frequencies
        values = []
        for i in range(MAXIMUM_NUM_COHORTS + 1):
            values += [f"cat_{i}"] * (MAXIMUM_NUM_COHORTS + 1 - i)
        df = pd.DataFrame({"col": values})

        sg.config.usage.load_time_filters = [FilterConfig(source="col", action="keep_top")]
        result = sg._apply_load_time_filters(df)

        top_values = df["col"].value_counts().nlargest(MAXIMUM_NUM_COHORTS).index
        assert set(result["col"].unique()) == set(top_values)

    def test_keep_top_with_few_categories_returns_all(self, fake_seismo):
        sg = Seismogram()
        values = ["A", "B", "C", "A", "B", "C", "C"]
        assert len(set(values)) < MAXIMUM_NUM_COHORTS
        df = pd.DataFrame({"col": values})

        sg.config.usage.load_time_filters = [FilterConfig(source="col", action="keep_top")]
        result = sg._apply_load_time_filters(df)

        # Expect all original rows to remain
        assert result.equals(df)

    @pytest.mark.parametrize(
        "action, values, range_, expected",
        [
            ("include", ["A", "C"], None, ["A", "C"]),
            ("include", None, FilterRange(min=5, max=15), [5, 10]),  # 5 included, 15 excluded
            ("include", None, FilterRange(min=5, max=10), [5]),  # 10 excluded
            ("include", None, FilterRange(min=10, max=20), [10]),  # only 10 is in range
            ("include", None, FilterRange(min=5, max=None), [5, 10, 20]),  # unbounded max
            ("include", None, FilterRange(min=None, max=10), [1, 5]),  # 10 excluded
            ("exclude", ["B", "D"], None, ["A", "C"]),
            ("exclude", None, FilterRange(min=5, max=15), [1, 20]),  # exclude 5, 10
            ("exclude", None, FilterRange(min=5, max=10), [1, 10, 20]),  # only 5 excluded
            ("exclude", None, FilterRange(min=10, max=20), [1, 5, 20]),  # exclude 10
            ("exclude", None, FilterRange(min=5, max=None), [1]),  # exclude 5, 10, 20
            ("exclude", None, FilterRange(min=None, max=10), [10, 20]),  # exclude 1, 5
        ],
        ids=[
            "include-values",
            "include-range-5-15",
            "include-range-5-10",
            "include-range-10-20",
            "include-min-only",
            "include-max-only",
            "exclude-values",
            "exclude-range-5-15",
            "exclude-range-5-10",
            "exclude-range-10-20",
            "exclude-min-only",
            "exclude-max-only",
        ],
    )
    def test_include_exclude_with_values_or_range(self, action, values, range_, expected, fake_seismo):
        sg = Seismogram()
        col_data = ["A", "B", "C", "D"] if values else [1, 5, 10, 20]
        df = pd.DataFrame({"col": col_data})
        sg.config.usage.load_time_filters = [FilterConfig(source="col", action=action, values=values, range=range_)]

        result = sg._apply_load_time_filters(df)
        assert sorted(result["col"].tolist()) == sorted(expected)

    @pytest.mark.parametrize(
        "action, values, range_, expected_warning, expected_result",
        [
            ("include", [1], FilterRange(min=0), "both 'values' and 'range'", [1]),
            ("exclude", [1], FilterRange(min=0), "both 'values' and 'range'", [2, 3]),
        ],
        ids=["include-both", "exclude-both"],
    )
    def test_include_exclude_with_both(
        self, action, values, range_, expected_warning, expected_result, fake_seismo, caplog
    ):
        sg = Seismogram()
        df = pd.DataFrame({"col": [1, 2, 3]})
        sg.config.usage.load_time_filters = [FilterConfig(source="col", action=action, values=values, range=range_)]

        with caplog.at_level("WARNING", logger="seismometer"):
            result = sg._apply_load_time_filters(df)

        assert expected_warning in caplog.text
        assert sorted(result["col"].tolist()) == sorted(expected_result)

    def test_missing_column_raises_error(self, fake_seismo):
        sg = Seismogram()
        df = pd.DataFrame({"col": [1, 2, 3]})

        sg.config.usage.load_time_filters = [FilterConfig(source="missing_col", action="include", values=[1])]

        with pytest.raises(ValueError, match="missing_col"):
            sg._apply_load_time_filters(df)

    def test_range_filter_raises_if_column_values_not_comparable(self, fake_seismo):
        sg = Seismogram()
        df = pd.DataFrame({"col": ["a", "b", "c"]})  # strings

        sg.config.usage.load_time_filters = [
            FilterConfig(source="col", action="include", range=FilterRange(min=1, max=10))
        ]

        with pytest.raises(ValueError, match="Values in 'col' must be comparable to '1'."):
            sg._apply_load_time_filters(df)


class TestSeismogramResolveCohortHierarchies:
    def test_valid_cohort_hierarchies_are_resolved(self, fake_seismo):
        sg = Seismogram()
        sg.dataframe = pd.DataFrame({"cohort1": [0], "cohort2": [1]})
        sg._cohorts = [
            Cohort(source="cohort1", display_name="Cohort 1"),
            Cohort(source="cohort2", display_name="Cohort 2"),
        ]
        sg.config.cohort_hierarchies = [CohortHierarchy(name="Test Hierarchy", column_order=["cohort1", "cohort2"])]

        sg._validate_and_resolve_cohort_hierarchies()

        assert len(sg.cohort_hierarchies) == 1
        resolved = sg.cohort_hierarchies[0]
        assert resolved.name == "Test Hierarchy"
        assert resolved.column_order == ["Cohort 1", "Cohort 2"]

    def test_raises_error_on_unknown_cohort_source(self, fake_seismo):
        sg = Seismogram()
        sg.dataframe = pd.DataFrame({"cohort1": [0]})
        sg._cohorts = [Cohort(source="cohort1", display_name="Cohort 1")]

        sg.config.cohort_hierarchies = [CohortHierarchy(name="Bad", column_order=["cohort1", "unknown"])]

        with pytest.raises(ValueError, match="references undefined cohort source: 'unknown'"):
            sg._validate_and_resolve_cohort_hierarchies()

    def test_skips_when_no_hierarchies_configured(self, fake_seismo):
        sg = Seismogram()
        sg.config.cohort_hierarchies = []

        # Should not raise or modify anything
        sg._validate_and_resolve_cohort_hierarchies()

        assert sg.cohort_hierarchies == []


class TestSeismogramBuildCohortHierarchyCombinations:
    @pytest.mark.parametrize(
        "cohorts, hierarchies, df_data, expected_results",
        [
            # Valid single hierarchy
            (
                [
                    Cohort(source="cohort1", display_name="Cohort 1"),
                    Cohort(source="cohort2", display_name="Cohort 2"),
                ],
                [CohortHierarchy(name="Demo", column_order=["cohort1", "cohort2"])],
                pd.DataFrame(
                    {
                        "cohort1": ["A", "A", "B", None],
                        "cohort2": [1, 1, 2, 2],
                    }
                ).assign(
                    **{
                        "Cohort 1": lambda df: df["cohort1"],
                        "Cohort 2": lambda df: df["cohort2"],
                    }
                ),
                {
                    ("Cohort 1", "Cohort 2"): pd.DataFrame(
                        {
                            "Cohort 1": ["A", "B"],
                            "Cohort 2": [1, 2],
                        }
                    )
                },
            ),
            # Invalid hierarchy (missing column)
            (
                [
                    Cohort(source="c1", display_name="C1"),
                    Cohort(source="c2", display_name="C2"),
                    Cohort(source="missing", display_name="MISSING"),
                ],
                [CohortHierarchy(name="Invalid", column_order=["c1", "missing"])],
                pd.DataFrame(
                    {
                        "c1": ["a", "b"],
                        "c2": ["x", "y"],
                        "C1": ["a", "b"],
                        "C2": ["x", "y"],
                        # No "MISSING" column → should skip
                    }
                ),
                {},
            ),
            # Multiple valid hierarchies
            (
                [
                    Cohort(source="a", display_name="A"),
                    Cohort(source="b", display_name="B"),
                    Cohort(source="c", display_name="C"),
                ],
                [
                    CohortHierarchy(name="H1", column_order=["a", "b"]),
                    CohortHierarchy(name="H2", column_order=["b", "c"]),
                ],
                pd.DataFrame(
                    {
                        "a": ["x", "x", "u"],
                        "b": ["y", "y", "v"],
                        "c": ["z", "w", "w"],
                    }
                ).assign(
                    **{
                        "A": lambda df: df["a"],
                        "B": lambda df: df["b"],
                        "C": lambda df: df["c"],
                    }
                ),
                {
                    ("A", "B"): pd.DataFrame({"A": ["u", "x"], "B": ["v", "y"]}),
                    ("B", "C"): pd.DataFrame({"B": ["v", "y", "y"], "C": ["w", "w", "z"]}),
                },
            ),
        ],
        ids=["valid-hierarchy", "missing-column", "multiple-hierarchies"],
    )
    def test_build_cohort_hierarchy_combinations(self, cohorts, hierarchies, df_data, expected_results, fake_seismo):
        sg = Seismogram()
        sg._cohorts = cohorts
        sg.config.cohort_hierarchies = hierarchies
        sg.dataframe = df_data

        sg._build_cohort_hierarchy_combinations()

        actual_keys = set(sg.cohort_hierarchy_combinations.keys())
        expected_keys = set(expected_results.keys())
        assert actual_keys == expected_keys

        for key in expected_keys:
            actual_df = sg.cohort_hierarchy_combinations[key].reset_index(drop=True)
            expected_df = expected_results[key].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual_df, expected_df)


class TestSeismogram:
    def test_value_counts_match_after_cohort_filtering(self, tmp_path):
        # --- Setup config and loader manually (mimics fake_seismo) ---
        config = get_test_config(tmp_path)
        config.cohorts = [Cohort(source=name, display_name=f"Display_{name}") for name in ["cohort1", "cohort2"]]
        config.usage.load_time_filters = [FilterConfig(source="cohort1", action="include", values=[1])]
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text('{"thresholds": [0.2, 0.8], "modelname": "TestModel"}')
        config.metadata_path = metadata_path
        config.cohort_hierarchies = []

        # Use filtered data
        df = get_test_data()

        loader = get_test_loader(config)
        loader.load_data.return_value = df

        # --- Act: Create Seismogram instance manually ---
        Seismogram.kill()
        sg = Seismogram(config=config, dataloader=loader)
        sg.load_data()

        # --- Assert: source vs. display column value_counts match ---
        for cohort in config.cohorts:
            source_col = cohort.source
            display_col = cohort.display_name or source_col

            if display_col not in sg.dataframe.columns:
                continue  # skipped due to censoring or config

            source_counts = sg.dataframe[source_col].value_counts(sort=False).sort_index()
            display_counts = sg.dataframe[display_col].value_counts(sort=False).sort_index()

            mismatch_msg = f"Counts for {display_col} do not match {source_col} after filtering."
            assert dict(display_counts) == dict(source_counts), mismatch_msg
        # Cleanup
        Seismogram.kill()
