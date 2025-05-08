from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import pytest
from IPython.display import HTML

from seismometer.api import templates as undertest
from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import Cohort, Event
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
    mock_config.target = "event1"
    mock_config.entity_keys = ["entity"]
    mock_config.predict_time = "time"
    mock_config.cohorts = [Cohort(source=name) for name in ["cohort1"]]
    mock_config.features = ["one"]
    mock_config.config_dir = tmp_path / "config"
    mock_config.censor_min_count = 0
    mock_config.targets = ["event1", "event2", "event3"]
    mock_config.output_list = ["prediction", "score1", "score2"]
    mock_config.entity_id = "entity"

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
            "time": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"]),
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
    sg._start_time = datetime(2024, 1, 1, 1, 1, 1)
    sg._end_time = datetime(2025, 1, 1, 1, 1, 1)
    sg.thresholds = [0.2, 0.5, 0.9]
    yield sg

    Seismogram.kill()


class TestTemplates:
    def test_show_info_returns_html(self, fake_seismo, monkeypatch):
        # Covers: _get_info_dict and show_info
        monkeypatch.setattr(undertest, "Seismogram", lambda: fake_seismo)
        result = undertest.show_info(plot_help=True)
        assert isinstance(result, HTML)
        assert "Scores" in result.data or "predictions" in result.data

    def test_show_cohort_summaries_default(self, fake_seismo, monkeypatch):
        # Covers: _get_cohort_summary_dataframes and _style_cohort_summaries
        monkeypatch.setattr(undertest, "Seismogram", lambda: fake_seismo)
        result = undertest.show_cohort_summaries()
        assert isinstance(result, HTML)
        for cohort in fake_seismo.available_cohort_groups:
            assert f"Counts by {cohort}" in result.data

    def test_show_cohort_summaries_with_target_and_score(self, fake_seismo, monkeypatch):
        # Covers: _score_target_levels_and_index and _style_score_target_cohort_summaries
        monkeypatch.setattr(undertest, "Seismogram", lambda: fake_seismo)
        result = undertest.show_cohort_summaries(by_target=True, by_score=True)
        assert isinstance(result, HTML)
        assert "Counts by cohort1" in result.data
        assert "0" in result.data or "1" in result.data  # categories like target or score bins

    def test_show_cohort_summaries_with_empty_df(self, fake_seismo, monkeypatch):
        # Covers: filtering behavior when df is empty
        fake_seismo.dataframe = pd.DataFrame(columns=fake_seismo.dataframe.columns)
        monkeypatch.setattr(undertest, "Seismogram", lambda: fake_seismo)
        result = undertest.show_cohort_summaries(by_target=True, by_score=True)
        assert isinstance(result, HTML)
        assert "Counts by cohort1" in result.data
