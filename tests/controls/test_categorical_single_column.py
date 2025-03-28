from unittest.mock import Mock

import pandas as pd
import pytest
from IPython.display import HTML, SVG

from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import Cohort, Event, Metric, MetricDetails
from seismometer.controls.categorical_single_column import (
    OrdinalCategoricalSinglePlot,
    ordinal_categorical_single_col_plot,
)
from seismometer.data.loader import SeismogramLoader
from seismometer.seismogram import Seismogram


def get_test_config(tmp_path):
    mock_config = Mock(autospec=ConfigProvider)
    mock_config.output_dir.return_value
    mock_config.events = {
        "event1": Event(source="event1", display_name="event1", window_hr=1),
        "event2": Event(source="event2", display_name="event2", window_hr=2, aggregation_method="min"),
        "event3": Event(source="event3", display_name="event3", window_hr=1),
    }
    mock_config.metrics = {
        "Metric1": Metric(
            source="Metric1",
            display_name="Metric1",
            type="categorical_feedback",
            group_keys=["Group1", "Group2"],
            metric_details=MetricDetails(values=["disagree", "neutral", "agree"]),
        ),
        "Metric2": Metric(
            source="Metric2",
            display_name="Metric2",
            type="categorical_feedback",
            group_keys="Group1",
            metric_details=MetricDetails(values=["disagree", "neutral", "agree"]),
        ),
    }
    mock_config.metric_groups = {"Group1": ["Metric1", "Metric2"], "Group2": ["Metric1"]}
    mock_config.metric_types = {"categorical_feedback": ["Metric1", "Metric2"]}
    mock_config.target = "event1"
    mock_config.entity_keys = ["entity"]
    mock_config.predict_time = "time"
    mock_config.cohorts = [Cohort(source=name) for name in ["Cohort"]]
    mock_config.features = ["one"]
    mock_config.config_dir = tmp_path / "config"
    mock_config.censor_min_count = 0
    mock_config.targets = ["event1", "event2", "event3"]
    mock_config.output_list = ["prediction", "score1", "score2"]

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
            "event1_Value": [0, 1, 0, 1],
            "event1_Time": ["2022-01-01", "2022-01-02", "2022-01-03", "2021-12-31"],
            "event2_Value": [0, 1, 0, 1],
            "event2_Time": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "event3_Value": [0, 2, 5, 1],
            "event3_Time": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "Cohort": ["C1", "C2", "C1", "C3"],
            "Metric1": ["disagree", "neutral", "agree", "disagree"],
            "Metric2": ["agree", "neutral", "disagree", "agree"],
            "score1": [0.1, 0.4, 0.35, 0.8],
            "score2": [0.2, 0.5, 0.3, 0.7],
            "target1": [0, 1, 0, 1],
            "target2": [1, 0, 1, 0],
            "target3": [1, 1, 1, 0],
        }
    )


@pytest.fixture
def fake_seismo(tmp_path):
    config = get_test_config(tmp_path)
    loader = get_test_loader(config)
    sg = Seismogram(config, loader)
    sg.dataframe = get_test_data()
    sg.available_cohort_groups = {"Cohort": ["C1", "C2", "C3"]}
    sg.metric_groups = {"Group1": "Metric1"}
    yield sg

    Seismogram.kill()


class TestOrdinalCategoricalSinglePlot:
    def test_initialize_plot_functions(self, fake_seismo):
        plot = OrdinalCategoricalSinglePlot(metric_col="Metric1")
        assert "Likert Plot" in plot.plot_functions

    def test_extract_metric_values(self, fake_seismo):
        plot = OrdinalCategoricalSinglePlot(metric_col="Metric1")
        plot._extract_metric_values()
        assert plot.values == ["disagree", "neutral", "agree"]

    def test_count_cohort_group_values(self, fake_seismo):
        plot = OrdinalCategoricalSinglePlot(metric_col="Metric1", cohort_dict={"Cohort": ("C1", "C2")})
        plot.values = ["disagree", "neutral", "agree"]
        counts_df = plot._count_cohort_group_values()

        # Expected DataFrame
        expected_df = pd.DataFrame({"disagree": [1, 0], "neutral": [0, 1], "agree": [1, 0]}, index=["C1", "C2"])

        # Set the index name to match counts_df
        expected_df.columns.name = "Metric1"
        expected_df.index.name = "Cohort"

        # Check equality of the two DataFrames
        pd.testing.assert_frame_equal(counts_df, expected_df)

    def test_plot_likert(self, fake_seismo):
        plot = OrdinalCategoricalSinglePlot(metric_col="Metric1", cohort_dict={"Cohort": ("C1", "C2")})
        plot.values = ["disagree", "neutral", "agree"]
        svg = plot.plot_likert()
        assert isinstance(svg, SVG)

    def test_generate_plot(self, fake_seismo):
        plot = OrdinalCategoricalSinglePlot(metric_col="Metric1", cohort_dict={"Cohort": ("C1", "C2")})
        plot.values = ["disagree", "neutral", "agree"]
        html = plot.generate_plot()
        assert isinstance(html, HTML)

    def test_initialization_with_title(self, fake_seismo):
        plot = OrdinalCategoricalSinglePlot(
            metric_col="Metric1", title="Test Title", cohort_dict={"Cohort": ("C1", "C2")}
        )
        assert plot.title == "Test Title"

    def test_initialization_with_cohort_dict(self, fake_seismo):
        cohort_dict = {"Cohort": ("C", "D")}
        plot = OrdinalCategoricalSinglePlot(metric_col="Metric1", cohort_dict=cohort_dict)
        assert plot.dataframe is not None  # Assuming the dataframe is filtered correctly

    def test_extract_metric_values_no_values(self, fake_seismo):
        plot = OrdinalCategoricalSinglePlot(metric_col="Metric1", cohort_dict={"Cohort": ("C1", "C2")})
        plot.dataframe = pd.DataFrame({"Metric1": ["disagree", "neutral", "agree"], "Cohort": ["A", "B", "A"]})
        plot._extract_metric_values()
        assert plot.values == ["disagree", "neutral", "agree"]

    def test_generate_plot_with_different_plot_type(self, fake_seismo):
        class CustomPlot(OrdinalCategoricalSinglePlot):
            @staticmethod
            def custom_plot(self):
                return SVG()

        plot = CustomPlot(metric_col="Metric1", plot_type="Custom Plot", cohort_dict={"Cohort": ("C1", "C2")})
        plot.plot_functions["Custom Plot"] = CustomPlot.custom_plot
        html = plot.generate_plot()
        assert isinstance(html, HTML)

    def test_extract_metric_values_discard_none(self, fake_seismo):
        plot = OrdinalCategoricalSinglePlot(metric_col="Metric1", cohort_dict={"Cohort": ("C1", "C2")})
        plot.dataframe = pd.DataFrame(
            {"Metric1": ["disagree", None, "neutral", "agree", None], "Cohort": ["C1", "C1", "C2", "C2", "C1"]}
        )
        plot._extract_metric_values()
        assert plot.values == ["disagree", "neutral", "agree"]


class TestOrdinalCategoricalSinglePlotFunction:
    def test_ordinal_categorical_single_plot(self, fake_seismo):
        metric_col = "Metric1"
        cohort_dict = {"Cohort": ["C1"]}
        html = ordinal_categorical_single_col_plot(metric_col, cohort_dict, title="Test Plot")
        assert isinstance(html, HTML)
