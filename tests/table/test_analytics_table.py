from unittest.mock import Mock

import pandas as pd
import pytest

from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import Cohort, Event
from seismometer.data.loader import SeismogramLoader
from seismometer.seismogram import Seismogram
from seismometer.table.analytics_table import PerformanceMetrics
from seismometer.table.analytics_table_config import AnalyticsTableConfig


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
    mock_config.targets = ["event1"]
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
    yield

    Seismogram.kill()


class TestPerformanceMetrics:
    def test_initialization(self, fake_seismo):
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
        pm = PerformanceMetrics(
            df=df,
            score_columns=scores,
            target_columns=targets,
            metric="sensitivity",
            metric_values=metric_values,
            statistics_data=statistics_data,
            table_config=table_config,
        )

        assert pm.df.equals(df)
        assert pm.score_columns == scores
        assert pm.target_columns == targets
        assert pm.metric == "Sensitivity"
        assert pm.metric_values == sorted(metric_values)
        assert pm.title == "Model Performance Statistics"
        assert pm.top_level == "Score"
        assert pm.decimals == 3
        assert pm.columns_show_percentages == ["Prevalence"]
        assert pm.columns_show_bar == {"AUROC": "lightblue", "PPV": "lightgreen"}
        assert pm.percentages_decimals == 0
        assert pm.data_bar_stroke_width == 4
        assert pm.rows_group_length == len(targets)
        assert pm.num_of_rows == len(scores) * len(targets)
        assert pm.statistics_data.equals(statistics_data)

    def test_invalid_metric(self, fake_seismo):
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
        with pytest.raises(
            ValueError,
            match="Invalid metric name: invalid_metric. The metric needs to be one of: "
            "\\['sensitivity', 'specificity', 'flagged', 'threshold'\\]",
        ):
            table_config = AnalyticsTableConfig()
            PerformanceMetrics(
                metric="invalid_metric", df=df, score_columns=scores, target_columns=targets, table_config=table_config
            )

    def test_invalid_top_level(self, fake_seismo):
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
        metric_values = [0.7, 0.8]
        with pytest.raises(
            ValueError,
            match="Invalid top_level name: invalid_top_level. "
            "The top_level needs to be one of: \\['score', 'target'\\]",
        ):
            PerformanceMetrics(
                df=df,
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
            PerformanceMetrics(metric="sensitivity")

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
        sg.dataframe = None
        sg.output_list = None
        with pytest.raises(
            ValueError,
            match="When df is provided, both 'score_columns' and 'target_columns' need " "to be provided as well.",
        ):
            PerformanceMetrics(df=df, target_columns=["target1", "target2"], metric="sensitivity")

    def test_non_empty_df_empty_targets(self, fake_seismo):
        df = pd.DataFrame(
            {
                "score1": [0.1, 0.4, 0.35, 0.8],
                "score2": [0.2, 0.5, 0.3, 0.7],
                "target1": [0, 1, 0, 1],
                "target2": [1, 0, 1, 0],
            }
        )
        sg = Seismogram()
        sg.dataframe = None
        with pytest.raises(
            ValueError,
            match="When df is provided, both 'score_columns' and 'target_columns' " "need to be provided as well.",
        ):
            PerformanceMetrics(df=df, score_columns=["score1", "score2"], metric="sensitivity")

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
        scores = ["score1", "score2"]
        targets = ["target1", "target2"]
        metric_values = [0.7, 0.8]
        pm = PerformanceMetrics(
            df=df,
            score_columns=scores,
            target_columns=targets,
            metric="sensitivity",
            metric_values=metric_values,
            statistics_data=statistics_data,
        )
        gt = pm.analytics_table()
        assert gt is not None

    def test_analytics_table_with_statistics_data_only(self, fake_seismo):
        statistics_data = pd.DataFrame(
            {"Score": ["prediction"], "Target": ["event1_Value"], "Stat1": [0.75], "Stat2": [0.6]}
        )
        metric_values = [0.7, 0.8]

        pm = PerformanceMetrics(
            metric="sensitivity",
            metric_values=metric_values,
            statistics_data=statistics_data,
        )
        gt = pm.analytics_table()
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
        pm = PerformanceMetrics(df=df, score_columns=scores, target_columns=targets, metric="sensitivity")
        data = pm._generate_table_data()
        gt = pm.generate_initial_table(data)
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
        pm = PerformanceMetrics(
            df=df, score_columns=scores, target_columns=targets, top_level="Target", metric="sensitivity"
        )
        data = pm._generate_table_data()
        gt = pm.generate_initial_table(data)
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
        pm = PerformanceMetrics(df=df, score_columns=scores, target_columns=targets, metric="sensitivity")
        data = pm._generate_table_data()
        gt = pm.generate_initial_table(data)
        gt = pm.generate_color_bar(gt, data.columns)
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
        pm = PerformanceMetrics(df=df, score_columns=scores, target_columns=targets, metric="sensitivity")
        data = pm._generate_table_data()
        gt = pm.generate_initial_table(data)
        for value in pm.metric_values:
            columns = [col for col in data.columns if col.startswith(f"{value}_")]
            gt = pm.group_columns_by_metric_value(gt, columns, value)
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
        pm = PerformanceMetrics(df=df, score_columns=scores, target_columns=targets, metric="sensitivity")
        gt = pm.analytics_table()
        assert gt is not None
