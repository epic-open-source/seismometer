from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from IPython.display import HTML, SVG
from ipywidgets import HTML as WidgetHTML

from seismometer import Seismogram
from seismometer.api.explore import ExplorationWidget, ExploreBinaryModelMetrics, cohort_list_details
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
    def test_cohort_list_details_summary_generated(self, fake_seismo):
        with patch("seismometer.seismogram.Seismogram", return_value=fake_seismo), patch(
            "seismometer.data.filter.filter_rule_from_cohort_dictionary", return_value=MagicMock()
        ) as mock_filter, patch(
            "seismometer.html.template.render_title_message", return_value=HTML("mock summary")
        ) as mock_render:
            rule = MagicMock()
            rule.filter.return_value = fake_seismo.dataframe
            mock_filter.return_value = rule

            result = cohort_list_details({"Cohort": ["C1", "C2"]})

        mock_filter.assert_called_once()
        mock_render.assert_called_once()
        assert "mock summary" in result.data

    def test_cohort_list_details_censored_output(self, fake_seismo):
        fake_seismo.config.censor_min_count = 10
        with patch("seismometer.seismogram.Seismogram", return_value=fake_seismo), patch(
            "seismometer.data.filter.filter_rule_from_cohort_dictionary", return_value=MagicMock()
        ) as mock_filter, patch(
            "seismometer.html.template.render_censored_plot_message", return_value=HTML("censored")
        ) as mock_render:
            rule = MagicMock()
            rule.filter.return_value = fake_seismo.dataframe.iloc[:0]  # no rows
            mock_filter.return_value = rule

            result = cohort_list_details({"Cohort": ["C1", "C2"]})

        assert "censored" in result.data.lower()
        mock_render.assert_called_once()

    def test_cohort_list_details_single_target_index_rename(self, fake_seismo):
        fake_seismo.config.targets = ["event1"]  # reduce to 1 target
        with patch("seismometer.seismogram.Seismogram", return_value=fake_seismo), patch(
            "seismometer.data.filter.filter_rule_from_cohort_dictionary", return_value=MagicMock()
        ) as mock_filter, patch("seismometer.html.template.render_title_message", return_value=HTML("summary")):
            rule = MagicMock()
            rule.filter.return_value = fake_seismo.dataframe
            mock_filter.return_value = rule

            result = cohort_list_details({"Cohort": ["C1", "C2"]})

        assert "summary" in result.data

    def test_cohort_list_widget_rendering(self, fake_seismo):
        with patch("seismometer.seismogram.Seismogram", return_value=fake_seismo), patch(
            "seismometer.controls.selection.MultiSelectionListWidget", return_value=WidgetHTML("MockWidget")
        ) as mock_widget, patch("seismometer.html.template.render_title_message", return_value=HTML("initial")), patch(
            "seismometer.data.filter.filter_rule_from_cohort_dictionary"
        ) as mock_filter:
            # Set up real filtering logic
            rule = Mock()
            rule.filter.return_value = fake_seismo.dataframe  # real df with proper column types
            mock_filter.return_value = rule
            from seismometer.api.explore import cohort_list

            output = cohort_list()

        mock_widget.assert_called_once()
        assert hasattr(output, "children") and len(output.children) == 2


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

    @patch(
        "seismometer.plot.mpl.binary_classifier.plot_metric_list",
        return_value=SVG('<svg xmlns="http://www.w3.org/2000/svg"></svg>'),
    )
    @patch("seismometer.data.performance.BinaryClassifierMetricGenerator.calculate_binary_stats")
    def test_plot_binary_classifier_metrics_basic(self, mock_calc, mock_plot, fake_seismo):
        mock_stats = pd.DataFrame({"Accuracy": [0.9], "PPV": [0.8]}, index=["value"])
        mock_calc.return_value = (mock_stats, None)

        result = plot_binary_classifier_metrics(
            metric_generator=BinaryClassifierMetricGenerator(rho=0.5),
            metrics=["Accuracy", "PPV"],
            cohort_dict={},
            target="event1",
            score_column="score1",
        )

        mock_calc.assert_called_once()
        mock_plot.assert_called_once()
        assert isinstance(result, SVG)
        assert "http://www.w3.org/2000/svg" in result.data

    @patch("seismometer.data.performance.BinaryClassifierMetricGenerator.calculate_binary_stats")
    def test_plot_binary_classifier_metrics_table_only(self, mock_calc, fake_seismo):
        mock_stats = pd.DataFrame({"Sensitivity": [0.88]}, index=["value"])
        mock_calc.return_value = (mock_stats, None)

        result = plot_binary_classifier_metrics(
            metric_generator=BinaryClassifierMetricGenerator(rho=0.5),
            metrics="Sensitivity",
            cohort_dict={},
            target="event1",
            score_column="score1",
            table_only=True,
        )

        assert isinstance(result, HTML)
        assert "Sensitivity" in result.data

    @patch("seismometer.html.template.render_title_message", return_value=HTML("requires exactly two classes"))
    @patch("seismometer.data.filter.FilterRule.filter")
    @patch("seismometer.api.plots.pdh.event_value", return_value="event1_Value")
    def test_plot_binary_classifier_metrics_nonbinary_target(
        self, mock_event_value, mock_filter, mock_render, fake_seismo
    ):
        df = fake_seismo.dataframe.copy()
        df["event1_Value"] = 1  # only one class present

        mock_filter.return_value = df

        with patch("seismometer.seismogram.Seismogram", return_value=fake_seismo):
            fake_seismo.dataframe = df

            result = plot_binary_classifier_metrics(
                metric_generator=BinaryClassifierMetricGenerator(rho=0.5),
                metrics=["Accuracy"],
                cohort_dict={"Cohort": ("C1",)},
                target="event1",
                score_column="score1",
            )

        assert isinstance(result, HTML)
        assert "requires exactly two classes" in result.data


class TestExploreModelEvaluation:
    @patch("seismometer.html.template.render_title_message", return_value=HTML("requires exactly two classes"))
    @patch("seismometer.data.filter.FilterRule.filter")
    def test_model_evaluation_not_binary_target(self, mock_filter, mock_render, fake_seismo):
        df = fake_seismo.dataframe.copy()
        df["event1_Value"] = 1  # single class
        mock_filter.return_value = df

        result = _model_evaluation(
            df,
            entity_keys=["entity"],
            target_event="event1",
            target="event1_Value",
            score_col="score1",
            thresholds=[0.2],
            censor_threshold=0,
        )

        assert isinstance(result, HTML)
        assert "requires exactly two classes" in result.data

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

    @patch("seismometer.api.plots.pdh.get_model_scores")
    @patch("seismometer.api.plots.plot.evaluation", return_value=SVG('<svg xmlns="http://www.w3.org/2000/svg"></svg>'))
    @patch("seismometer.html.template.render_title_with_image", return_value=HTML("rendered"))
    @patch("seismometer.data.filter.FilterRule.filter")
    def test_model_evaluation_valid_path(self, mock_filter, mock_render, mock_plot, mock_scores, fake_seismo):
        df = fake_seismo.dataframe.copy()
        df["event1_Value"] = [0, 1, 1, 0]  # binary target
        mock_filter.return_value = df
        mock_scores.return_value = df

        result = _model_evaluation(
            df,
            entity_keys=["entity"],
            target_event="event1",
            target="event1_Value",
            score_col="score1",
            thresholds=[0.2],
            censor_threshold=0,
        )

        mock_plot.assert_called_once()
        mock_render.assert_called_once()
        assert isinstance(result, HTML)
        assert "rendered" in result.data


class TestExploreModelScoreComparison:
    @patch("seismometer.html.template.render_title_with_image", return_value=HTML("Mock Rendered"))
    @patch("seismometer.api.plots.plot.cohort_evaluation_vs_threshold", return_value="svg-obj")
    @patch("seismometer.api.plots.assert_valid_performance_metrics_df")
    @patch("seismometer.api.plots.get_cohort_performance_data")
    @patch("seismometer.api.plots.pdh.event_score")
    @patch("seismometer.data.filter.FilterRule.filter")
    @patch("seismometer.data.filter.FilterRule.from_cohort_dictionary")
    def test_plot_model_score_comparison(
        self,
        mock_from_dict,
        mock_filter,
        mock_event_score,
        mock_perf_data,
        mock_assert_valid,
        mock_plot,
        mock_render,
        fake_seismo,
    ):
        # Base input DataFrame with expected columns
        df = fake_seismo.dataframe[["score1", "event1_Value"]].copy()

        # Mocks return real data for processing
        mock_from_dict.return_value = MagicMock()
        mock_filter.return_value = df  # initial filter from cohort
        mock_event_score.return_value = df  # per_context=False path
        mock_perf_data.return_value = pd.DataFrame({"metric": [1]})

        result = plot_model_score_comparison(
            cohort_dict={"Cohort": ("C1", "C2")},
            target="event1",
            scores=("score1",),
            per_context=False,
        )

        assert isinstance(result, HTML)
        assert "Mock Rendered" in result.data
        mock_render.assert_called_once()
        mock_plot.assert_called_once()


class TestExploreModelTargetComparison:
    @patch("seismometer.html.template.render_title_with_image", return_value=HTML("Mock Rendered"))
    @patch("seismometer.api.plots.plot.cohort_evaluation_vs_threshold", return_value="svg-obj")
    @patch("seismometer.api.plots.assert_valid_performance_metrics_df")
    @patch("seismometer.api.plots.get_cohort_performance_data")
    @patch("seismometer.api.plots.pdh.event_score")
    @patch("seismometer.api.plots.pdh.event_value", side_effect=lambda x: f"{x}_Value")
    @patch("seismometer.data.filter.FilterRule.filter")
    @patch("seismometer.data.filter.FilterRule.from_cohort_dictionary")
    def test_plot_model_target_comparison(
        self,
        mock_from_dict,
        mock_filter,
        mock_event_value,
        mock_event_score,
        mock_perf_data,
        mock_assert_valid,
        mock_plot,
        mock_render,
        fake_seismo,
    ):
        # Input DataFrame
        df = fake_seismo.dataframe[["score1", "event1_Value"]].copy()

        # Mocked behaviors
        mock_from_dict.return_value = MagicMock()
        mock_filter.return_value = df  # .filter(dataframe) returns df
        mock_event_score.return_value = df
        mock_perf_data.return_value = pd.DataFrame({"metric": [1]})

        result = plot_model_target_comparison(
            cohort_dict={"Cohort": ("C1", "C2")},
            targets=("event1",),
            score="score1",
            per_context=False,
        )

        assert isinstance(result, HTML)
        assert "Mock Rendered" in result.data
        mock_render.assert_called_once()
        mock_plot.assert_called_once()


class TestExploreCohortEvaluation:
    @patch("seismometer.html.template.render_censored_plot_message", return_value=HTML("censored"))
    @patch("seismometer.api.plots.assert_valid_performance_metrics_df", side_effect=ValueError("invalid"))
    @patch("seismometer.api.plots.get_cohort_performance_data")
    @patch("seismometer.api.plots.pdh.get_model_scores")
    def test_plot_cohort_evaluation_invalid_data(
        self,
        mock_get_scores,
        mock_get_perf,
        mock_assert_valid,
        mock_render,
        fake_seismo,
    ):
        df = fake_seismo.dataframe.copy()
        mock_get_scores.return_value = df
        mock_get_perf.return_value = pd.DataFrame()

        result = _plot_cohort_evaluation(
            dataframe=df,
            entity_keys=["entity"],
            target="event1_Value",
            output="score1",
            thresholds=[0.2],
            cohort_col="Cohort",
            subgroups=["C1"],
            censor_threshold=10,
        )

        assert isinstance(result, HTML)
        assert "censored" in result.data.lower()
        mock_render.assert_called_once()

    @patch("seismometer.html.template.render_title_with_image", return_value=HTML("rendered"))
    @patch("seismometer.api.plots.plot.cohort_evaluation_vs_threshold", return_value="svg-object")
    @patch("seismometer.api.plots.assert_valid_performance_metrics_df")
    @patch("seismometer.api.plots.get_cohort_performance_data")
    @patch("seismometer.api.plots.pdh.get_model_scores")
    def test_plot_cohort_evaluation_success(
        self,
        mock_get_scores,
        mock_get_perf,
        mock_assert_valid,
        mock_plot,
        mock_render,
        fake_seismo,
    ):
        df = fake_seismo.dataframe.copy()
        mock_get_scores.return_value = df
        mock_get_perf.return_value = pd.DataFrame({"metric": [1]})

        result = _plot_cohort_evaluation(
            dataframe=df,
            entity_keys=["entity"],
            target="event1_Value",
            output="score1",
            thresholds=[0.2],
            cohort_col="Cohort",
            subgroups=["C1"],
            censor_threshold=0,
        )

        assert isinstance(result, HTML)
        assert "rendered" in result.data
        mock_render.assert_called_once()
        mock_plot.assert_called_once()


class TestExploreCohortHistograms:
    @patch("seismometer.seismogram.Seismogram")
    @patch("seismometer.api.plots.pdh.event_value", return_value="event1_Value")
    @patch("seismometer.data.filter.FilterRule.filter")
    @patch("seismometer.api.plots.plot.cohorts_vertical", return_value="svg-mock")
    @patch("seismometer.html.template.render_title_with_image", return_value=HTML("histogram"))
    def test_plot_cohort_group_histograms(
        self, mock_render, mock_plot, mock_filter, mock_event_value, mock_seismo, fake_seismo
    ):
        mock_seismo.return_value = fake_seismo
        mock_filter.return_value = fake_seismo.dataframe

        result = plot_cohort_group_histograms(
            cohort_col="Cohort",
            subgroups=["C1", "C2"],
            target_column="event1",
            score_column="score1",
        )

        assert isinstance(result, HTML)
        assert "histogram" in result.data
        mock_render.assert_called_once()

    @patch("seismometer.html.template.render_censored_plot_message", return_value=HTML("censored"))
    def test_plot_cohort_hist_empty_after_filter(self, mock_render, fake_seismo):
        # Simulate filtered-out result
        empty_df = fake_seismo.dataframe.iloc[0:0].copy()

        result = _plot_cohort_hist(
            dataframe=empty_df,
            target="event1_Value",
            output="score1",
            cohort_col="Cohort",
            subgroups=["C1", "C2"],
        )

        assert "censored" in result.data.lower()
        mock_render.assert_called_once()

    @patch("seismometer.html.template.render_title_message", return_value=HTML("error"))
    @patch("seismometer.api.plots.plot.cohorts_vertical", side_effect=ValueError("fail"))
    def test_plot_cohort_hist_plot_fails(self, mock_plot, mock_render, fake_seismo):
        df = fake_seismo.dataframe.copy()
        result = _plot_cohort_hist(df, "event1_Value", "score1", "Cohort", ["C1", "C2"])
        assert "error" in result.data.lower()

    @patch("seismometer.seismogram.Seismogram")
    @patch("seismometer.api.plots._plot_cohort_hist", return_value=HTML("cohort plot"))
    def test_plot_cohort_hist(self, mock_plot_fn, mock_seismo, fake_seismo):
        fake_seismo.selected_cohort = ("Cohort", ["C1", "C2"])
        mock_seismo.return_value = fake_seismo

        result = plot_cohort_hist()
        assert "cohort plot" in result.data
        mock_plot_fn.assert_called_once()


class TestExploreCohortLeadTime:
    @patch("seismometer.seismogram.Seismogram")
    @patch("seismometer.api.plots._plot_leadtime_enc", return_value=HTML("wrapped"))
    def test_plot_leadtime_enc(self, mock_plot, mock_seismo, fake_seismo):
        fake_seismo.selected_cohort = ("Cohort", ["C1", "C2"])
        mock_seismo.return_value = fake_seismo
        result = plot_leadtime_enc()
        assert "wrapped" in result.data
        mock_plot.assert_called_once()

    @patch("seismometer.seismogram.Seismogram")
    @patch("seismometer.api.plots._plot_leadtime_enc", return_value=HTML("cohort lead time"))
    def test_plot_cohort_lead_time(self, mock_plot, mock_seismo, fake_seismo):
        mock_seismo.return_value = fake_seismo
        result = plot_cohort_lead_time(
            cohort_col="Cohort",
            subgroups=["C1", "C2"],
            event_column="event1",
            score_column="score1",
            threshold=0.2,
        )
        assert "cohort lead time" in result.data
        mock_plot.assert_called_once()

    def test_leadtime_enc_missing_target_column(self, fake_seismo, caplog):
        df = fake_seismo.dataframe.drop(columns=["event1_Value"])
        with caplog.at_level("ERROR"):
            result = _plot_leadtime_enc(
                df,
                entity_keys=["entity"],
                target_event="event1_Value",
                target_zero="event1_Time",
                score="score1",
                threshold=0.2,
                ref_time="time",
                cohort_col="Cohort",
                subgroups=["C1"],
                max_hours=48,
                x_label="Lead Time (hours)",
            )
        assert result is None
        assert "Target event (event1_Value) not found" in caplog.text

    def test_leadtime_enc_missing_target_zero_column(self, fake_seismo, caplog):
        df = fake_seismo.dataframe.drop(columns=["event1_Time"])
        with caplog.at_level("ERROR"):
            result = _plot_leadtime_enc(
                df,
                entity_keys=["entity"],
                target_event="event1_Value",
                target_zero="event1_Time",
                score="score1",
                threshold=0.2,
                ref_time="time",
                cohort_col="Cohort",
                subgroups=["C1"],
                max_hours=48,
                x_label="Lead Time (hours)",
            )
        assert result is None
        assert "Target event time-zero (event1_Time) not found" in caplog.text

    def test_leadtime_enc_no_positive_events(self, fake_seismo, caplog):
        df = fake_seismo.dataframe.copy()
        df["event1_Value"] = 0  # force all negative
        with caplog.at_level("ERROR"):
            result = _plot_leadtime_enc(
                df,
                entity_keys=["entity"],
                target_event="event1_Value",
                target_zero="event1_Time",
                score="score1",
                threshold=0.2,
                ref_time="time",
                cohort_col="Cohort",
                subgroups=["C1"],
                max_hours=48,
                x_label="Lead Time (hours)",
            )
        assert result is None
        assert "No positive events (event1_Value=1) were found" in caplog.text

    @patch("seismometer.api.plots.pdh.event_score", return_value=None)
    def test_leadtime_enc_below_censor_threshold(self, mock_score, fake_seismo):
        df = fake_seismo.dataframe.copy()
        df = df[df["event1_Value"] == 1].iloc[:1]
        result = _plot_leadtime_enc(
            df,
            entity_keys=["entity"],
            target_event="event1_Value",
            target_zero="event1_Time",
            score="score1",
            threshold=0.2,
            ref_time="time",
            cohort_col="Cohort",
            subgroups=["C1"],
            max_hours=48,
            x_label="Lead Time (hours)",
            censor_threshold=5,
        )
        assert "censored" in result.data.lower()

    @patch("seismometer.api.plots.pdh.event_score")
    @patch("seismometer.data.filter.FilterRule.filter")
    @patch("seismometer.html.template.render_censored_plot_message", return_value=HTML("censored"))
    def test_leadtime_enc_subgroup_filter_excludes_all(self, mock_render, mock_filter, mock_score, fake_seismo):
        df = fake_seismo.dataframe
        mock_filter.return_value = df[:0]
        mock_score.return_value = df[:0]

        result = _plot_leadtime_enc(
            df,
            entity_keys=["entity"],
            target_event="event1_Value",
            target_zero="event1_Time",
            score="score1",
            threshold=0.2,
            ref_time="time",
            cohort_col="Cohort",
            subgroups=["C1", "C2"],
            max_hours=48,
            x_label="Lead Time (hours)",
            censor_threshold=0,
        )

        assert "censored" in result.data.lower()


class TestExploreCohortOutcomeInterventionTimes:
    @patch("seismometer.seismogram.Seismogram")
    @patch("seismometer.api.plots._plot_trend_intervention_outcome", return_value=HTML("wrapped outcome/intervention"))
    def test_plot_intervention_outcome_timeseries(self, mock_plot, mock_seismo, fake_seismo):
        mock_seismo.return_value = fake_seismo
        result = plot_intervention_outcome_timeseries(
            cohort_col="Cohort",
            subgroups=["C1", "C2"],
            outcome="outcome1",
            intervention="intervention1",
            reference_time_col="time",
            censor_threshold=0,
        )
        assert isinstance(result, HTML)
        assert "wrapped outcome/intervention" in result.data

    @patch("seismometer.api.plots.pdh.event_value", side_effect=lambda x: f"{x}_Value")
    @patch("seismometer.api.plots._plot_ts_cohort", return_value="svg")
    @patch("seismometer.html.template.render_title_with_image", side_effect=lambda title, svg: HTML(title))
    def test_plot_trend_intervention_outcome_combines_both(
        self, mock_render, mock_plot, mock_event_value, fake_seismo
    ):
        fake_seismo.selected_cohort = ("Cohort", ["C1", "C2"])
        result = plot_trend_intervention_outcome()
        assert isinstance(result, HTML)
        assert "Outcome" in result.data
        assert "Intervention" in result.data
        assert mock_render.call_count == 2

    @patch("seismometer.api.plots.pdh.event_value", side_effect=lambda x: f"{x}_Value")
    @patch("seismometer.api.plots._plot_ts_cohort", side_effect=[IndexError("missing"), "svg"])
    @patch("seismometer.html.template.render_title_with_image", side_effect=lambda title, svg: HTML(title))
    @patch("seismometer.html.template.render_title_message", side_effect=lambda title, msg: HTML(f"{title}: {msg}"))
    def test_plot_trend_intervention_outcome_missing_intervention(
        self, mock_msg, mock_render, mock_plot, mock_event_value, fake_seismo
    ):
        fake_seismo.selected_cohort = ("Cohort", ["C1", "C2"])
        result = plot_trend_intervention_outcome()
        assert isinstance(result, HTML)
        assert "Missing Intervention" in result.data
        assert "Outcome" in result.data

    @patch("seismometer.api.plots.pdh.event_value", side_effect=lambda x: f"{x}_Value")
    @patch("seismometer.api.plots._plot_ts_cohort", side_effect=["svg", IndexError("missing")])
    @patch("seismometer.html.template.render_title_with_image", side_effect=lambda title, svg: HTML(title))
    @patch("seismometer.html.template.render_title_message", side_effect=lambda title, msg: HTML(f"{title}: {msg}"))
    def test_plot_trend_intervention_outcome_missing_outcome(
        self, mock_msg, mock_render, mock_plot, mock_event_value, fake_seismo
    ):
        fake_seismo.selected_cohort = ("Cohort", ["C1", "C2"])
        result = plot_trend_intervention_outcome()
        assert isinstance(result, HTML)
        assert "Missing Outcome" in result.data
        assert "Intervention" in result.data


class TestExplorationWidgetErrorHandling:
    def test_try_generate_plot_with_exception(self):
        class FailingWidget(ExplorationWidget):
            def __init__(self):
                # Define a plot function that raises
                def bad_plot(*args, **kwargs):
                    raise RuntimeError("kaboom")

                bad_plot.__name__ = "plot_func"
                bad_plot.__module__ = "seismometer.api"
                super().__init__("Failing", WidgetHTML("placeholder"), bad_plot)

            def generate_plot_args(self):
                return (), {}

        widget = FailingWidget()
        result = widget._try_generate_plot()

        assert isinstance(result, WidgetHTML)
        assert "Traceback" in result.value
        assert "kaboom" in result.value
