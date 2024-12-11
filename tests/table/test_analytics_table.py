from unittest.mock import Mock

import pandas as pd
import pytest

from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import Cohort, Event
from seismometer.data.loader import SeismogramLoader
from seismometer.seismogram import Seismogram
from seismometer.table.analytics_table import AnalyticsTable
from seismometer.table.analytics_table_config import AnalyticsTableConfig


def get_test_config(tmp_path):
    mock_config = Mock(autospec=ConfigProvider)
    mock_config.output_dir.return_value
    mock_config.events = {
        "event1": Event(source="event1", display_name="event1", window_hr=1),
        "event2": Event(source="event2", display_name="event2", window_hr=2, aggregation_method="min"),
        "event3": Event(source="event3", display_name="event3", window_hr=1),
    }
    mock_config.target = "event1"
    mock_config.entity_keys = ["entity"]
    mock_config.predict_time = "time"
    mock_config.cohorts = [Cohort(source=name) for name in ["cohort1", "cohort2"]]
    mock_config.features = ["one"]
    mock_config.config_dir = tmp_path / "config"
    mock_config.censor_min_count = 0
    mock_config.targets = ["event1", "event2", "event3"]
    mock_config.output_list = ["prediction"]

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
            "cohort1": [1, 0, 1, 0],
            "cohort2": [0, 1, 0, 1],
        }
    )


@pytest.fixture
def fake_seismo(tmp_path):
    config = get_test_config(tmp_path)
    loader = get_test_loader(config)
    sg = Seismogram(config, loader)
    sg.dataframe = get_test_data()
    yield sg

    Seismogram.kill()


class TestAnalyticsTable:
    def test_initialization(self, fake_seismo):
        statistics_data = pd.DataFrame(
            {
                "Score": ["score1", "score1", "score2", "score2"],
                "Target": ["target1", "target2", "target1", "target2"],
                "AUROC": [0.75, 0.85, 0.6, 0.5],
                "PPV": [0.6, 0.7, 0.5, 0.3],
            }
        )
        scores = ["score1", "score2"]
        targets = ["target1", "target2"]
        metric_values = [0.7, 0.8]
        table_config = AnalyticsTableConfig(
            columns_show_bar={"AUROC": "lightblue", "PPV": "lightgreen"},
            columns_show_percentages=["Prevalence"],
        )
        table = AnalyticsTable(
            score_columns=scores,
            target_columns=targets,
            metric="sensitivity",
            metric_values=metric_values,
            statistics_data=statistics_data,
            table_config=table_config,
        )

        assert table.score_columns == scores
        assert table.target_columns == targets
        assert table.metric == "Sensitivity"
        assert table.metric_values == sorted(metric_values)
        assert table.title == "Model Performance Statistics"
        assert table.top_level == "Score"
        assert table.decimals == 3
        assert table.columns_show_percentages == ["Prevalence"]
        assert table.columns_show_bar == {"AUROC": "lightblue", "PPV": "lightgreen"}
        assert table.percentages_decimals == 0
        assert table.data_bar_stroke_width == 4
        assert table.statistics_data.equals(statistics_data)

    def test_invalid_metric(self, fake_seismo):
        with pytest.raises(
            ValueError,
            match="Invalid metric name: invalid_metric. The metric needs to be one of: "
            "\\['sensitivity', 'specificity', 'flagrate', 'threshold'\\]",
        ):
            table_config = AnalyticsTableConfig()
            AnalyticsTable(metric="invalid_metric", table_config=table_config)

    def test_invalid_top_level(self, fake_seismo):
        scores = ["score1", "score2"]
        targets = ["target1", "target2"]
        metric_values = [0.7, 0.8]
        with pytest.raises(
            ValueError,
            match="Invalid top_level name: invalid_top_level. "
            "The top_level needs to be one of: \\['score', 'target'\\]",
        ):
            AnalyticsTable(
                score_columns=scores,
                target_columns=targets,
                metric="sensitivity",
                metric_values=metric_values,
                top_level="invalid_top_level",
            )

    def test_empty_df_and_statistics_data(self, fake_seismo):
        sg = Seismogram()
        sg.dataframe = None
        with pytest.raises(ValueError, match="At least one of 'df' or 'statistics_data' needs to be provided."):
            AnalyticsTable(metric="sensitivity")

    def test_non_empty_df_empty_scores(self, fake_seismo):
        df = pd.DataFrame(
            {
                "score1": [0.1, 0.4, 0.35, 0.8],
                "score2": [0.2, 0.5, 0.3, 0.7],
                "target1": [0, 1, 0, 1],
                "target2": [1, 0, 1, 0],
            }
        )
        sg = Seismogram()
        sg.dataframe = df
        sg.output_list = None
        with pytest.raises(
            ValueError,
            match="When df is provided, both 'score_columns' and 'target_columns' need " "to be provided as well.",
        ):
            AnalyticsTable(target_columns=["target1", "target2"], metric="sensitivity")

    def test_non_empty_df_empty_targets(self, fake_seismo):
        sg = Seismogram()
        sg.config.targets = ["event3"]
        with pytest.raises(
            ValueError,
            match="When df is provided, both 'score_columns' and 'target_columns' " "need to be provided as well.",
        ):
            AnalyticsTable(score_columns=["score1", "score2"], metric="sensitivity")

    def test_analytics_table_with_statistics_data(self, fake_seismo):
        df = pd.DataFrame(
            {
                "score1": [0.1, 0.4, 0.35, 0.8],
                "score2": [0.2, 0.5, 0.3, 0.7],
                "target1": [0, 1, 0, 1],
                "target2": [1, 0, 1, 0],
            }
        )
        statistics_data = pd.DataFrame(
            {
                "Score": ["score1", "score2", "score1", "score2"],
                "Target": ["target1", "target2", "target2", "target1"],
                "extra-stats-1": [0.75, 0.85, 0.65, 0.95],
                "extra-stats-2": [0.6, 0.7, 0.55, 0.75],
            }
        )
        sg = Seismogram()
        sg.dataframe = df
        scores = ["score1", "score2"]
        targets = ["target1", "target2"]
        metric_values = [0.7, 0.8]
        table = AnalyticsTable(
            score_columns=scores,
            target_columns=targets,
            metric="sensitivity",
            metric_values=metric_values,
            statistics_data=statistics_data,
        )
        gt = table.analytics_table()
        assert gt is not None

    def test_analytics_table_with_statistics_data_only(self, fake_seismo):
        statistics_data = pd.DataFrame(
            {"Score": ["prediction"], "Target": ["event1_Value"], "Stat1": [0.75], "Stat2": [0.6]}
        )
        metric_values = [0.7, 0.8]

        table = AnalyticsTable(
            metric="sensitivity",
            metric_values=metric_values,
            statistics_data=statistics_data,
        )
        gt = table.analytics_table()
        assert gt is not None

    def test_generate_initial_table(self, fake_seismo):
        df = pd.DataFrame(
            {
                "score1": [0.1, 0.4, 0.35, 0.8],
                "score2": [0.2, 0.5, 0.3, 0.7],
                "target1": [0, 1, 0, 1],
                "target2": [1, 0, 1, 0],
                "target3": [1, 1, 1, 0],
            }
        )
        scores = ["score1", "score2"]
        targets = ["target1", "target2", "target3"]
        sg = Seismogram()
        sg.dataframe = df
        table = AnalyticsTable(score_columns=scores, target_columns=targets, metric="sensitivity")
        data = table._generate_table_data()
        gt = table.generate_initial_table(data)
        assert gt is not None

    def test_generate_initial_table_group_by_target(self, fake_seismo):
        df = pd.DataFrame(
            {
                "score1": [0.1, 0.4, 0.35, 0.8],
                "score2": [0.2, 0.5, 0.3, 0.7],
                "target1": [0, 1, 0, 1],
                "target2": [1, 0, 1, 0],
                "target3": [1, 1, 1, 0],
            }
        )
        scores = ["score1", "score2"]
        targets = ["target1", "target2", "target3"]
        sg = Seismogram()
        sg.dataframe = df
        table = AnalyticsTable(score_columns=scores, target_columns=targets, top_level="Target", metric="sensitivity")
        data = table._generate_table_data()
        gt = table.generate_initial_table(data)
        assert gt is not None

    def test_generate_color_bar(self, fake_seismo):
        df = pd.DataFrame(
            {
                "score1": [0.1, 0.4, 0.35, 0.8],
                "score2": [0.2, 0.5, 0.3, 0.7],
                "target1": [0, 1, 0, 1],
                "target2": [1, 0, 1, 0],
            }
        )
        scores = ["score1", "score2"]
        targets = ["target1", "target2"]
        sg = Seismogram()
        sg.dataframe = df
        table = AnalyticsTable(score_columns=scores, target_columns=targets, metric="sensitivity")
        data = table._generate_table_data()
        gt = table.generate_initial_table(data)
        gt = table.generate_color_bar(gt, data.columns)
        assert gt is not None

    def test_group_columns_by_metric_value(self, fake_seismo):
        df = pd.DataFrame(
            {
                "score1": [0.1, 0.4, 0.35, 0.8],
                "score2": [0.2, 0.5, 0.3, 0.7],
                "target1": [0, 1, 0, 1],
                "target2": [1, 0, 1, 0],
            }
        )
        scores = ["score1", "score2"]
        targets = ["target1", "target2"]
        sg = Seismogram()
        sg.dataframe = df
        table = AnalyticsTable(score_columns=scores, target_columns=targets, metric="sensitivity")
        data = table._generate_table_data()
        gt = table.generate_initial_table(data)
        for value in table.metric_values:
            columns = [col for col in data.columns if col.startswith(f"{value}_")]
            gt = table.group_columns_by_metric_value(gt, columns, value)
        assert gt is not None

    def test_analytics_table(self, fake_seismo):
        df = pd.DataFrame(
            {
                "score1": [0.1, 0.4, 0.35, 0.8],
                "score2": [0.2, 0.5, 0.3, 0.7],
                "target1": [0, 1, 0, 1],
                "target2": [1, 0, 1, 0],
            }
        )
        scores = ["score1", "score2"]
        targets = ["target1", "target2"]
        sg = Seismogram()
        sg.dataframe = df
        table = AnalyticsTable(score_columns=scores, target_columns=targets, metric="sensitivity")
        gt = table.analytics_table()
        assert gt is not None
