from datetime import datetime
from unittest.mock import Mock, patch

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
            "prediction": [0.1, 0.4, 0.35, 0.8],
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
    def test_show_info_high_level(self, fake_seismo):
        with patch.object(undertest.template, "render_info_template", return_value=HTML("info_html")) as mock_render:
            _ = undertest.show_info(plot_help=True)

        mock_render.assert_called_once()
        data = mock_render.call_args[0][0]
        assert data["num_predictions"] == fake_seismo.prediction_count
        assert data["num_entities"] == fake_seismo.entity_count
        assert data["start_date"] == fake_seismo.start_time.strftime("%Y-%m-%d")
        assert data["end_date"] == fake_seismo.end_time.strftime("%Y-%m-%d")
        assert data["plot_help"] is True

    def test_show_cohort_summaries_with_target_and_score(self, fake_seismo):
        multi_index = pd.MultiIndex.from_tuples(
            [
                ("A", pd.Interval(0.0, 0.5), 0),
                ("A", pd.Interval(0.5, 1.0), 1),
                ("B", pd.Interval(0.5, 1.0), 0),
            ],
            names=["cohort1", "score1", "event1"],
        )
        mock_summary = pd.DataFrame({"Predictions": [10, 20, 5], "Entities": [8, 19, 4]}, index=multi_index)

        with (
            patch.object(undertest, "default_cohort_summaries", return_value=fake_seismo.dataframe),
            patch.object(undertest, "score_target_cohort_summaries", return_value=mock_summary),
            patch.object(
                undertest.template, "render_cohort_summary_template", return_value=HTML("summary_html")
            ) as mock_render,
            patch("pandas.io.formats.style.Styler.to_html", return_value="<table>Counts by cohort1</table>"),
        ):
            undertest.show_cohort_summaries(by_target=True, by_score=True)

        mock_render.assert_called_once()

    def test_show_cohort_summaries_with_empty_df(self, fake_seismo):
        # simulate empty input
        fake_seismo.dataframe = pd.DataFrame(columns=fake_seismo.dataframe.columns)

        # empty MultiIndex that matches expected structure
        empty_index = pd.MultiIndex.from_tuples([], names=["cohort1", "score1", "event1"])
        mock_empty_summary = pd.DataFrame(columns=["Predictions", "Entities"], index=empty_index)

        with (
            patch.object(undertest, "default_cohort_summaries", return_value=fake_seismo.dataframe),
            patch.object(undertest, "score_target_cohort_summaries", return_value=mock_empty_summary),
            patch.object(
                undertest.template, "render_cohort_summary_template", return_value=HTML("empty_html")
            ) as mock_render,
            patch("pandas.io.formats.style.Styler.to_html", return_value="<table>Counts by cohort1</table>"),
        ):
            undertest.show_cohort_summaries(by_target=True, by_score=True)

        mock_render.assert_called_once()

    def test_score_target_levels_and_index_variants(self, fake_seismo):
        thresholds = fake_seismo.thresholds
        prediction_col = "prediction"
        expected_cut = pd.cut(fake_seismo.dataframe[prediction_col], [0] + thresholds + [1])

        # by_score=False, by_target=False
        g1, gg1, idx1 = undertest._score_target_levels_and_index("cohort1", False, False)
        assert g1 == ["cohort1"]
        assert gg1 == ["cohort1"]
        assert idx1 == ["Cohort"]

        # by_score=True
        g2, gg2, idx2 = undertest._score_target_levels_and_index("cohort1", False, True)
        assert g2[0] == "cohort1"
        assert isinstance(g2[1], pd.Series)
        assert g2[1].equals(expected_cut)
        assert gg2 == ["cohort1", prediction_col]
        assert idx2 == ["Cohort", prediction_col]

        # by_target=True
        expected_target = fake_seismo.target
        g3, gg3, idx3 = undertest._score_target_levels_and_index("cohort1", True, False)
        assert g3 == ["cohort1", expected_target]
        assert gg3 == ["cohort1", expected_target]
        assert idx3 == ["Cohort", expected_target[:-6]]

        # by_target=True, by_score=True
        expected_target = fake_seismo.target  # event1_Value
        g4, gg4, idx4 = undertest._score_target_levels_and_index("cohort1", True, True)
        expected_cut2 = pd.cut(fake_seismo.dataframe[prediction_col], [0] + thresholds + [1])

        assert g4[0] == "cohort1"
        assert isinstance(g4[1], pd.Series)
        assert g4[1].equals(expected_cut2)
        assert g4[2] == expected_target  # event1_Value
        assert gg4 == ["cohort1", prediction_col, expected_target]
        assert idx4 == ["Cohort", prediction_col, expected_target[:-6]]  # user-facing label
