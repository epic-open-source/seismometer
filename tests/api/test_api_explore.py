from unittest.mock import Mock, patch

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
from seismometer.api.plots import (
    _model_evaluation,
    _plot_cohort_evaluation,
    _plot_cohort_hist,
    _plot_leadtime_enc,
    plot_binary_classifier_metrics,
    plot_cohort_group_histograms,
    plot_cohort_hist,
    plot_cohort_lead_time,
    plot_intervention_outcome_timeseries,
    plot_leadtime_enc,
    plot_model_score_comparison,
    plot_model_target_comparison,
    plot_trend_intervention_outcome,
)
from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import Cohort, Event, Metric, MetricDetails
from seismometer.data.filter import FilterRule
from seismometer.data.loader import SeismogramLoader
from seismometer.data.performance import BinaryClassifierMetricGenerator
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
    mock_config.comparison_time = ""
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
    @patch("seismometer.api.explore.cohort_list_details")
    def test_widget_calls_plot_function(self, mock_plot_func, fake_seismo):
        mock_plot_func.return_value = HTML("Mock Summary")

        widget = ExploreSubgroups()
        widget.option_widget.value = {"Cohort": ["C1", "C2"]}
        widget.plot_function(widget.option_widget.value)

        mock_plot_func.assert_called_once_with({"Cohort": ["C1", "C2"]})

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


class TestExploreBinaryModelMetrics:
    def test_widget_calls_plot_function(self, fake_seismo):
        # Define a spy-style dummy function to replace the real plot function
        def dummy_plot_func(
            metric_generator, metrics, cohort_dict, target, score_column, *, per_context=False, table_only=False
        ):
            dummy_plot_func.called = True
            dummy_plot_func.call_args = (metric_generator, metrics, cohort_dict, target, score_column)
            dummy_plot_func.call_kwargs = dict(per_context=per_context, table_only=table_only)
            return HTML("Mocked plot")

        dummy_plot_func.__name__ = "plot_binary_classifier_metrics"
        dummy_plot_func.__module__ = "seismometer.api"

        # Set up widget with dummy function
        widget = ExploreBinaryModelMetrics(rho=0.5)
        widget.plot_function = dummy_plot_func

        # Simulate user selections
        widget.option_widget.metric_list.value = ["Accuracy"]
        widget.option_widget.model_options.target_list.value = "event1"
        widget.option_widget.model_options.score_list.value = "prediction"
        widget.option_widget.model_options.per_context_checkbox.value = False
        widget.option_widget.cohort_list.value = {"Cohort": ["C1"]}

        widget.update_plot()

        # Validate that dummy_plot_func was called correctly
        assert getattr(dummy_plot_func, "called", False) is True
        assert dummy_plot_func.call_args[1] == ("Accuracy",)
        assert dummy_plot_func.call_args[2] == {"Cohort": ["C1"]}
        assert dummy_plot_func.call_args[3] == "event1"
        assert dummy_plot_func.call_args[4] == "prediction"

    def test_plot_binary_classifier_metrics_basic(self, fake_seismo):
        metric_gen = BinaryClassifierMetricGenerator(rho=0.5)
        html = plot_binary_classifier_metrics(
            metric_generator=metric_gen,
            metrics=["Accuracy", "PPV"],
            cohort_dict={},
            target="event1",
            score_column="score1",
        )
        assert isinstance(html, SVG)
        assert "Accuracy" in html.data and "PPV" in html.data

    def test_plot_binary_classifier_metrics_table_only(self, fake_seismo):
        metric_gen = BinaryClassifierMetricGenerator(rho=0.5)
        html = plot_binary_classifier_metrics(
            metric_generator=metric_gen,
            metrics="Sensitivity",
            cohort_dict={},
            target="event1",
            score_column="score1",
            table_only=True,
        )
        assert isinstance(html, HTML)
        assert "Sensitivity" in html.data

    def test_plot_binary_classifier_metrics_nonbinary_target(self, fake_seismo):
        df = fake_seismo.dataframe.copy()
        df["event1_Value"] = 1  # No variation

        metric_gen = BinaryClassifierMetricGenerator(rho=0.5)
        # Temporarily override the dataframe
        fake_seismo.dataframe = df

        html = plot_binary_classifier_metrics(
            metric_generator=metric_gen,
            metrics=["Accuracy"],
            cohort_dict={"Cohort": ("C1",)},
            target="event1",
            score_column="score1",
        )
        assert isinstance(html, HTML)
        assert "requires exactly two classes" in html.data


class TestExploreModelEvaluation:
    def test_widget_calls_plot_function(self, fake_seismo):
        widget = ExploreModelEvaluation()

        # Simulate user selections
        widget.option_widget.cohort_list.value = {"Cohort": ["C1", "C2"]}
        widget.option_widget.model_options.target_list.value = "event1"
        widget.option_widget.model_options.score_list.value = "score1"
        slider_key = next(iter(widget.option_widget.model_options.threshold_list.sliders))
        widget.option_widget.model_options.threshold_list.value = {slider_key: 0.2}
        widget.option_widget.model_options.per_context_checkbox.value = False

        result = widget.plot_function(
            cohort_dict={"Cohort": ["C1", "C2"]},
            target_column="event1",
            score_column="score1",
            thresholds=[0.2],
            per_context=False,
        )

        assert isinstance(result, HTML)
        assert "Overall Performance for" in result.data

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

    def test_model_evaluation_not_binary_target(self, fake_seismo):
        df = fake_seismo.dataframe
        df["event1_Value"] = 1  # no variation
        html = _model_evaluation(df, ["entity"], "event1", "event1_Value", "score1", [0.2], 0)
        assert "requires exactly two classes" in html.data


class TestExploreModelScoreComparison:
    def test_widget_calls_plot_function(self, fake_seismo):
        widget = ExploreModelScoreComparison()

        widget.option_widget.cohort_list.value = {"Cohort": ["C1", "C2"]}
        widget.option_widget.model_options.target_list.value = "event1"
        widget.option_widget.model_options.score_list.value = ("prediction", "score1")
        widget.option_widget.model_options.per_context_checkbox.value = False

        result = widget.plot_function(
            cohort_dict={"Cohort": ["C1", "C2"]},
            target="event1",
            scores=("prediction", "score1"),
            per_context=False,
        )

        assert isinstance(result, HTML)
        assert "Model Metrics" in result.data

    def test_plot_model_score_comparison(self, fake_seismo):
        html = plot_model_score_comparison(
            cohort_dict={"Cohort": ("C1", "C2")},
            target="event1",
            scores=("prediction", "score1"),
            per_context=False,
        )

        assert isinstance(html, HTML)
        assert "Model Metrics" in html.data


class TestExploreModelTargetComparison:
    def test_widget_calls_plot_function(self, fake_seismo):
        widget = ExploreModelTargetComparison()

        # Simulate user input
        widget.option_widget.cohort_list.value = {"Cohort": ["C1", "C2"]}
        widget.option_widget.model_options.target_list.value = ("event1", "event2")
        widget.option_widget.model_options.score_list.value = "score1"
        widget.option_widget.model_options.per_context_checkbox.value = True

        result = widget.plot_function(
            cohort_dict={"Cohort": ["C1", "C2"]},
            targets=("event1", "event2"),
            score="score1",
            per_context=True,
        )

        assert isinstance(result, HTML)
        assert "Model Metrics" in result.data

    def test_plot_model_target_comparison(self, fake_seismo):
        html = plot_model_target_comparison(
            cohort_dict={"Cohort": ("C1", "C2")},
            targets=("event1", "event2"),
            score="score1",
            per_context=False,
        )

        assert isinstance(html, HTML)
        assert "Model Metrics" in html.data


class TestExploreCohortEvaluation:
    def test_widget_calls_plot_function(self, fake_seismo):
        widget = ExploreCohortEvaluation()

        widget.option_widget.cohort_list.value = ("Cohort", ("C1", "C2"))
        widget.option_widget.model_options.target_list.value = "event1"
        widget.option_widget.model_options.score_list.value = "score1"

        slider_key = next(iter(widget.option_widget.model_options.threshold_list.sliders))
        widget.option_widget.model_options.threshold_list.value = {slider_key: 0.2}

        widget.option_widget.model_options.per_context_checkbox.value = False

        result = widget.plot_function(
            cohort_col="Cohort",
            subgroups=["C1", "C2"],
            target_column="event1",
            score_column="score1",
            thresholds=[0.2],
            per_context=False,
        )

        assert isinstance(result, HTML)
        assert "Model Performance Metrics on Cohort across Thresholds" in result.data

    def test_plot_cohort_evaluation_invalid_data(self, fake_seismo):
        df = fake_seismo.dataframe
        df["event1_Value"] = 0  # only negatives
        html = _plot_cohort_evaluation(
            df, ["entity"], "event1_Value", "score1", [0.2], "Cohort", ["C1"], censor_threshold=10
        )
        assert "censored" in html.data.lower()


class TestExploreCohortHistograms:
    def test_widget_calls_plot_function(self, fake_seismo):
        widget = ExploreCohortHistograms()

        widget.option_widget.cohort_list.value = ("Cohort", ("C1", "C2"))
        widget.option_widget.model_options.target_list.value = "event1"
        widget.option_widget.model_options.score_list.value = "score1"

        result = widget.plot_function(
            cohort_col="Cohort",
            subgroups=["C1", "C2"],
            target_column="event1",
            score_column="score1",
        )

        assert isinstance(result, HTML)
        assert "Predicted Probabilities by Cohort" in result.data

    def test_plot_cohort_group_histograms(self, fake_seismo):
        html = plot_cohort_group_histograms(
            cohort_col="Cohort",
            subgroups=["C1", "C2"],
            target_column="event1",
            score_column="score1",
        )

        assert isinstance(html, HTML)
        assert "Predicted Probabilities by Cohort" in html.data

    def test_plot_cohort_hist_empty_after_filter(self, fake_seismo):
        df = fake_seismo.dataframe
        df["Cohort"] = pd.Categorical(["Z", "Z", "Z", "Z"])  # mark as categorical
        html = _plot_cohort_hist(df, "event1_Value", "score1", "Cohort", ["C1", "C2"])
        assert "censored" in html.data.lower()

    def test_plot_cohort_hist_plot_fails(self, fake_seismo, monkeypatch):
        from seismometer.api import plots

        monkeypatch.setattr(
            plots.plot, "cohorts_vertical", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("fail"))
        )
        html = plots._plot_cohort_hist(fake_seismo.dataframe, "event1_Value", "score1", "Cohort", ["C1", "C2"], 0)
        assert "Error" in html.data

    def test_plot_cohort_hist(self, fake_seismo):
        fake_seismo.selected_cohort = ("Cohort", ["C1", "C2"])
        result = plot_cohort_hist()
        assert isinstance(result, HTML)
        assert "Predicted Probabilities by" in result.data


class TestExploreCohortLeadTime:
    def test_widget_calls_plot_function(self, fake_seismo):
        widget = ExploreCohortLeadTime()

        widget.option_widget.cohort_list.value = ("Cohort", ("C1", "C2"))
        widget.option_widget.model_options.target_list.value = "event1"
        widget.option_widget.model_options.score_list.value = "score1"

        slider_key = next(iter(widget.option_widget.model_options.threshold_list.sliders))
        widget.option_widget.model_options.threshold_list.value = {slider_key: 0.2}

        result = widget.plot_function(
            cohort_col="Cohort",
            subgroups=["C1", "C2"],
            event_column="event1",
            score_column="score1",
            threshold=0.2,
        )

        assert isinstance(result, HTML)
        assert "Lead Time" in result.data

    def test_plot_cohort_lead_time(self, fake_seismo):
        html = plot_cohort_lead_time(
            cohort_col="Cohort",
            subgroups=["C1", "C2"],
            event_column="event1",
            score_column="score1",
            threshold=0.2,
        )

        assert isinstance(html, HTML)
        assert "Lead Time" in html.data

    def test_leadtime_enc_no_positives(self, fake_seismo):
        df = fake_seismo.dataframe
        df["event1_Value"] = 0  # all negatives
        result = _plot_leadtime_enc(
            df,
            ["entity"],
            "event1_Value",
            "event1_Time",
            "score1",
            0.2,
            "time",
            "Cohort",
            ["C1"],
            48,
            "Lead Time (hours)",
            0,
        )
        assert result is None

    def test_plot_leadtime_enc(self, fake_seismo):
        fake_seismo.selected_cohort = ("Cohort", ["C1", "C2"])
        result = plot_leadtime_enc()
        assert "Lead Time" in result.data

    def test_leadtime_enc_missing_target_column(self, fake_seismo, caplog):
        df = fake_seismo.dataframe.drop(columns=["event1_Value"])
        with caplog.at_level("ERROR"):
            result = _plot_leadtime_enc(
                df,
                ["entity"],
                "event1_Value",
                "event1_Time",
                "score1",
                0.2,
                "time",
                "Cohort",
                ["C1"],
                48,
                "Lead Time (hours)",
                0,
            )
        assert result is None
        assert "Target event (event1_Value) not found" in caplog.text

    def test_leadtime_enc_missing_target_zero_column(self, fake_seismo, caplog):
        df = fake_seismo.dataframe.drop(columns=["event1_Time"])
        with caplog.at_level("ERROR"):
            result = _plot_leadtime_enc(
                df,
                ["entity"],
                "event1_Value",
                "event1_Time",
                "score1",
                0.2,
                "time",
                "Cohort",
                ["C1"],
                48,
                "Lead Time (hours)",
                0,
            )
        assert result is None
        assert "Target event time-zero (event1_Time) not found" in caplog.text

    def test_leadtime_enc_no_positive_events(self, fake_seismo, caplog):
        df = fake_seismo.dataframe
        df["event1_Value"] = 0  # force all negative
        with caplog.at_level("ERROR"):
            result = _plot_leadtime_enc(
                df,
                ["entity"],
                "event1_Value",
                "event1_Time",
                "score1",
                0.2,
                "time",
                "Cohort",
                ["C1"],
                48,
                "Lead Time (hours)",
                0,
            )
        assert result is None
        assert "No positive events (event1_Value=1) were found" in caplog.text

    def test_leadtime_enc_below_censor_threshold(self, fake_seismo):
        df = fake_seismo.dataframe
        df = df[df["event1_Value"] == 1].iloc[:1]  # one row with positive event
        result = _plot_leadtime_enc(
            df,
            ["entity"],
            "event1_Value",
            "event1_Time",
            "score1",
            0.2,
            "time",
            "Cohort",
            ["C1"],
            48,
            "Lead Time (hours)",
            censor_threshold=5,
        )
        assert "censored" in result.data.lower()

    def test_leadtime_enc_subgroup_filter_excludes_all(self, fake_seismo):
        df = fake_seismo.dataframe
        df["Cohort"] = "Z"  # not in subgroups
        result = _plot_leadtime_enc(
            df,
            ["entity"],
            "event1_Value",
            "event1_Time",
            "score1",
            0.2,
            "time",
            "Cohort",
            ["C1", "C2"],
            48,
            "Lead Time (hours)",
            0,
        )
        assert "censored" in result.data.lower()


class TestExploreCohortOutcomeInterventionTimes:
    def test_widget_calls_plot_function(self, fake_seismo):
        widget = ExploreCohortOutcomeInterventionTimes()

        widget.option_widget.cohort_list.value = ("Cohort", ("C1", "C2"))
        widget.option_widget.model_options.outcome_list.value = "outcome1"
        widget.option_widget.model_options.intervention_list.value = "intervention1"
        widget.option_widget.model_options.reference_time_list.value = "time"

        result = widget.plot_function(
            cohort_col="Cohort",
            subgroups=["C1", "C2"],
            outcome="outcome1",
            intervention="intervention1",
            reference_time_col="time",
        )

        assert isinstance(result, HTML)
        assert "Outcome" in result.data
        assert "Intervention" in result.data

    def test_plot_trend_intervention_outcome(self, fake_seismo):
        fake_seismo.selected_cohort = ("Cohort", ["C1", "C2"])
        result = plot_trend_intervention_outcome()
        assert isinstance(result, HTML)
        assert "Outcome" in result.data and "Intervention" in result.data

    def test_plot_intervention_outcome_timeseries(self, fake_seismo):
        result = plot_intervention_outcome_timeseries(
            cohort_col="Cohort",
            subgroups=["C1", "C2"],
            outcome="outcome1",
            intervention="intervention1",
            reference_time_col="time",
            censor_threshold=0,
        )
        assert isinstance(result, HTML)
        assert "Outcome" in result.data and "Intervention" in result.data


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
