from unittest.mock import Mock

import pandas as pd
import pytest
from IPython.display import HTML, SVG
from ipywidgets import HTML as WidgetHTML
from ipywidgets import VBox

from seismometer import Seismogram
from seismometer.api.explore import (
    ExplorationWidget,
    ExploreBinaryModelMetrics,
    ExploreCohortEvaluation,
    ExploreCohortHistograms,
    ExploreCohortLeadTime,
    ExploreCohortOutcomeInterventionTimes,
    ExploreModelEvaluation,
    ExploreModelScoreComparison,
    ExploreModelTargetComparison,
    ExploreSubgroups,
    cohort_list_details,
)
from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import Cohort, Event, Metric, MetricDetails
from seismometer.data.filter import FilterRule
from seismometer.data.loader import SeismogramLoader
from seismometer.seismogram import Seismogram


@pytest.fixture(autouse=True)
def set_min_rows_zero(monkeypatch):
    monkeypatch.setattr(FilterRule, "MIN_ROWS", 0)


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
    mock_config.entity_keys = ["entity", "context_id"]
    mock_config.predict_time = "time"
    mock_config.cohorts = [Cohort(source=name) for name in ["Cohort"]]
    mock_config.features = ["one"]
    mock_config.config_dir = tmp_path / "config"
    mock_config.censor_min_count = 0
    mock_config.targets = ["event1", "event2", "event3"]
    mock_config.output_list = ["prediction", "score1", "score2"]
    mock_config.interventions = {"intervention1": {}, "intervention2": {}}
    mock_config.outcomes = {"outcome1": {}, "outcome2": {}}
    mock_config.entity_id = "entity"
    mock_config.context_id = "context_id"

    return mock_config


def get_test_loader(config):
    mock_loader = Mock(autospec=SeismogramLoader)
    mock_loader.config = config

    return mock_loader


def get_test_data():
    return pd.DataFrame(
        {
            "entity": ["A", "A", "B", "C"],
            "context_id": ["ctx1", "ctx1", "ctx1", "ctx2"],
            "prediction": [1, 2, 3, 4],
            "time": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "event1_Value": [0, 1, 0, 1],
            "event1_Time": ["2022-01-01", "2022-01-02", "2022-01-03", "2021-12-31"],
            "event2_Value": [0, 1, 0, 1],
            "event2_Time": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "event3_Value": [0, 2, 5, 1],
            "event3_Time": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "intervention1_Value": [0, 1, 1, 0],
            "intervention2_Value": [1, 0, 0, 1],
            "outcome1_Value": [1, 1, 0, 0],
            "outcome2_Value": [0, 0, 1, 1],
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
    df = get_test_data()
    # Convert time columns to datetime
    df["time"] = pd.to_datetime(df["time"])
    df["event1_Time"] = pd.to_datetime(df["event1_Time"])
    sg.dataframe = df
    sg.create_cohorts()
    sg.thresholds = [0.2]
    sg.available_cohort_groups = {"Cohort": ["C1", "C2"]}
    yield sg

    Seismogram.kill()


class TestExploreSubgroups:
    def test_generate_plot(self, fake_seismo):
        self.widget = ExploreSubgroups()
        result = self.widget._try_generate_plot()
        assert isinstance(result, HTML)
        assert "Summary" in result.data
        assert "cohort_list_details" in self.widget.current_plot_code

    def test_multiple_targets_and_interventions(self, fake_seismo):
        result = cohort_list_details(fake_seismo.available_cohort_groups)
        assert "event1" in result.data
        assert "event2" in result.data

    def test_cohort_list_censored_output(self, fake_seismo, monkeypatch):
        fake_seismo.config.censor_min_count = 10
        result = cohort_list_details(fake_seismo.available_cohort_groups)
        assert "censored" in result.data.lower()

    def test_cohort_list_widget_rendering(self, fake_seismo):
        # This test checks the structure of the widget output
        from seismometer.api.explore import cohort_list

        result = cohort_list()
        assert isinstance(result, VBox)
        assert len(result.children) == 2
        assert "Cohort" in result.children[0].title  # MultiSelectionListWidget title

    def test_cohort_list_details_single_target_index_rename(self, fake_seismo):
        # Setup Seismogram with a single target
        fake_seismo.config.targets = ["event1"]  # Only one target
        result = cohort_list_details(fake_seismo.available_cohort_groups)
        assert isinstance(result, HTML)
        assert "Summary" in result.data


class TestExploreModelEvaluation:
    def test_generate_plot(self, fake_seismo):
        self.widget = ExploreModelEvaluation()
        result = self.widget._try_generate_plot()
        assert isinstance(result, HTML)
        assert "Overall Performance for event1 (Per Observation)" in result.data and "Sensitivity" in result.data
        assert "plot_model_evaluation" in self.widget.current_plot_code

    def test_generate_plot_code_with_args_and_kwargs(self):
        class DummyWidget(ExplorationWidget):
            def __init__(self):
                def dummy_plot(*args, **kwargs):
                    pass

                dummy_plot.__name__ = "plot_model_evaluation"
                dummy_plot.__module__ = "seismometer.api"
                super().__init__("Dummy", WidgetHTML(""), dummy_plot)

        widget = DummyWidget()
        code = widget.generate_plot_code(("arg1",), {"threshold": 0.2})
        assert "plot_model_evaluation('arg1', threshold=0.2)" in code


class TestExploreModelScoreComparison:
    def test_generate_plot(self, fake_seismo):
        self.widget = ExploreModelScoreComparison()
        result = self.widget._try_generate_plot()
        assert isinstance(result, HTML)
        assert "Model Metrics: prediction" in result.data
        assert "plot_model_score_comparison" in self.widget.current_plot_code


class TestExploreModelTargetComparison:
    def test_generate_plot(self, fake_seismo):
        self.widget = ExploreModelTargetComparison()
        result = self.widget._try_generate_plot()
        assert isinstance(result, HTML)
        assert "Model Metrics: event1" in result.data
        assert "plot_model_target_comparison" in self.widget.current_plot_code


class TestExploreCohortEvaluation:
    def test_generate_plot(self, fake_seismo):
        self.widget = ExploreCohortEvaluation()
        result = self.widget._try_generate_plot()
        assert isinstance(result, HTML)
        assert "Model Performance Metrics on Cohort across Thresholds" in result.data
        assert "plot_cohort_evaluation" in self.widget.current_plot_code


class TestExploreCohortHistograms:
    def test_generate_plot(self, fake_seismo):
        self.widget = ExploreCohortHistograms()
        result = self.widget._try_generate_plot()
        assert isinstance(result, HTML)
        assert "Predicted Probabilities by Cohort" in result.data
        assert "plot_cohort_group_histograms" in self.widget.current_plot_code


class TestExploreCohortLeadTime:
    def test_generate_plot(self, fake_seismo):
        self.widget = ExploreCohortLeadTime()
        result = self.widget._try_generate_plot()
        assert isinstance(result, HTML)
        assert "Lead Time" in result.data
        assert "plot_cohort_lead_time" in self.widget.current_plot_code


class TestExploreBinaryModelMetrics:
    def test_generate_plot(self, fake_seismo):
        self.widget = ExploreBinaryModelMetrics()
        result = self.widget._try_generate_plot()
        assert isinstance(result, SVG)
        assert result.data.startswith("<svg")
        assert "plot_binary_classifier_metrics" in self.widget.current_plot_code


class TestExploreCohortOutcomeInterventionTimes:
    def test_generate_plot(self, fake_seismo):
        self.widget = ExploreCohortOutcomeInterventionTimes()
        result = self.widget._try_generate_plot()
        assert isinstance(result, HTML)
        assert "Outcome" in result.data and "Intervention" in result.data
        assert "plot_intervention_outcome_timeseries" in self.widget.current_plot_code


class TestExplorationWidgetErrorHandling:
    def test_try_generate_plot_with_exception(self):
        class FailingWidget(ExplorationWidget):
            def __init__(self):
                super().__init__("Failing", WidgetHTML("Placeholder"), lambda *_: 1 / 0)

            def generate_plot_args(self):
                return (), {}

        widget = FailingWidget()
        result = widget._try_generate_plot()
        assert "Traceback" in result.value
