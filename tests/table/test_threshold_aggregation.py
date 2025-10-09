from unittest.mock import Mock

import pandas as pd
import pytest

from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import Cohort, Event
from seismometer.data.loader import SeismogramLoader
from seismometer.data.performance import THRESHOLD
from seismometer.seismogram import Seismogram
from seismometer.table.analytics_table import AnalyticsTable


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
    mock_config.cohorts = [Cohort(source=name) for name in ["cohort1"]]
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
            "cohort1": ["A", "A", "A", "B"],
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
    sg.available_cohort_groups = {"cohort1": ["A", "B"]}
    yield sg

    Seismogram.kill()


class TestThresholdAggregationIntegration:
    """Integration-level smoke tests for threshold-specific aggregation methods.

    These tests ensure that the new `first_above_threshold` aggregation
    mode integrates cleanly with AnalyticsTable and produces valid output.
    """

    @pytest.mark.parametrize(
        "score_cols,target_cols,thresholds,group_by",
        [
            (["score1"], ["target1"], [0.5], "Score"),
            (["score1", "score2"], ["target1", "target2"], [0.5], "Target"),
            (["score1"], ["target1"], [0.3, 0.6, 0.9], "Score"),
        ],
    )
    def test_first_above_threshold_runs(self, fake_seismo, score_cols, target_cols, thresholds, group_by):
        """Ensure first_above_threshold works for multiple score/target setups."""
        table = AnalyticsTable(
            score_columns=score_cols,
            target_columns=target_cols,
            metric=THRESHOLD,
            metric_values=thresholds,
            metrics_to_display=["AUROC", "PPV"],
            censor_threshold=1,
            cohort_dict={"cohort1": ("A", "B")},
            aggregation_method="first_above_threshold",
            top_level=group_by,
        )

        data = table._generate_table_data()
        assert isinstance(data, pd.DataFrame)
        assert "Score" in data.columns
        assert "Target" in data.columns

        html_table = table.analytics_table()
        assert html_table is not None

    @pytest.mark.parametrize("censor_threshold,expected_none", [(100, True), (1, False)])
    def test_censor_threshold_behavior(self, fake_seismo, censor_threshold, expected_none):
        """Verify censor_threshold filtering behaves consistently."""
        table = AnalyticsTable(
            score_columns=["score1"],
            target_columns=["target1"],
            metric=THRESHOLD,
            metric_values=[0.8],
            metrics_to_display=["AUROC"],
            censor_threshold=censor_threshold,
            cohort_dict={"cohort1": ("A", "B")},
            aggregation_method="first_above_threshold",
        )

        data = table._generate_table_data()
        if expected_none:
            assert data is None
        else:
            assert isinstance(data, pd.DataFrame)

    @pytest.mark.parametrize("group_by", ["Score", "Target"])
    def test_top_level_grouping_variants(self, fake_seismo, group_by):
        """Verify top_level variations work for threshold aggregation."""
        table = AnalyticsTable(
            score_columns=["score1", "score2"],
            target_columns=["target1", "target2"],
            metric=THRESHOLD,
            metric_values=[0.7],
            metrics_to_display=["AUROC", "Sensitivity"],
            censor_threshold=1,
            cohort_dict={"cohort1": ("A", "B")},
            aggregation_method="first_above_threshold",
            top_level=group_by,
        )

        data = table._generate_table_data()
        assert isinstance(data, pd.DataFrame)
        html = table.analytics_table()
        assert html is not None
