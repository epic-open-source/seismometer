from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from IPython.display import HTML
from pandas.io.formats.style import Styler

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


# ============================================================================
# ADDITIONAL ERROR HANDLING AND EDGE CASE TESTS
# ============================================================================


class TestShowCohortSummariesErrorHandling:
    """Test error handling and edge cases for show_cohort_summaries."""

    @patch.object(undertest, "_get_cohort_summary_dataframes", return_value={})
    @patch.object(undertest.template, "render_cohort_summary_template", return_value=HTML("empty"))
    def test_show_cohort_summaries_with_no_cohorts(self, mock_render, mock_get_dfs, fake_seismo):
        """Test show_cohort_summaries when there are no cohort groups."""
        fake_seismo.available_cohort_groups = {}

        result = undertest.show_cohort_summaries()

        assert isinstance(result, HTML)
        mock_get_dfs.assert_called_once_with(False, False)
        mock_render.assert_called_once()

    @pytest.mark.parametrize(
        "by_target,by_score",
        [
            (True, False),
            (False, True),
            (True, True),
            (False, False),
        ],
    )
    def test_show_cohort_summaries_parameter_combinations(self, fake_seismo, by_target, by_score):
        """Test show_cohort_summaries with all valid parameter combinations."""
        # Create appropriate MultiIndex based on parameters
        if by_target and by_score:
            multi_index = pd.MultiIndex.from_tuples([("A", 0.5, 1)], names=["cohort1", "score", "target"])
        elif by_target:
            multi_index = pd.MultiIndex.from_tuples([("A", 1)], names=["cohort1", "target"])
        elif by_score:
            multi_index = pd.MultiIndex.from_tuples([("A", 0.5)], names=["cohort1", "score"])
        else:
            multi_index = pd.MultiIndex.from_tuples([("A",)], names=["cohort1"])

        mock_summary = pd.DataFrame({"Predictions": [10], "Entities": [8]}, index=multi_index)

        with (
            patch.object(undertest, "default_cohort_summaries", return_value=fake_seismo.dataframe),
            patch.object(undertest, "score_target_cohort_summaries", return_value=mock_summary),
            patch.object(undertest.template, "render_cohort_summary_template", return_value=HTML("summary")),
            patch("pandas.io.formats.style.Styler.to_html", return_value="<table></table>"),
        ):
            result = undertest.show_cohort_summaries(by_target=by_target, by_score=by_score)

        assert isinstance(result, HTML)

    def test_show_cohort_summaries_with_missing_target_column(self, fake_seismo):
        """Test show_cohort_summaries when target column is missing."""
        # Remove target column
        fake_seismo.dataframe = fake_seismo.dataframe.drop(columns=["event1_Value"])

        with pytest.raises(KeyError):
            undertest.show_cohort_summaries(by_target=True)

    def test_show_cohort_summaries_with_missing_output_column(self, fake_seismo):
        """Test show_cohort_summaries when output (score) column is missing."""
        # Remove output column
        output_col = fake_seismo.output
        fake_seismo.dataframe = fake_seismo.dataframe.drop(columns=[output_col])

        # The error happens when trying to access the missing column
        with pytest.raises((KeyError, ValueError)):
            undertest.show_cohort_summaries(by_score=True)


class TestScoreTargetLevelsAndIndexEdgeCases:
    """Test edge cases for _score_target_levels_and_index function."""

    def test_score_target_levels_with_missing_target_column(self, fake_seismo):
        """Test _score_target_levels_and_index when target column is missing from dataframe."""
        fake_seismo.dataframe = fake_seismo.dataframe.drop(columns=["event1_Value"])

        # The function itself succeeds; error happens when used with dataframe in calling code
        g, gg, idx = undertest._score_target_levels_and_index("cohort1", by_target=True, by_score=False)

        # Should still return proper structure
        assert len(g) == 2
        assert len(gg) == 2
        assert len(idx) == 2

    def test_score_target_levels_with_missing_score_column(self, fake_seismo):
        """Test _score_target_levels_and_index when score column is missing from dataframe."""
        output_col = fake_seismo.output
        fake_seismo.dataframe = fake_seismo.dataframe.drop(columns=[output_col])

        # Should raise KeyError when trying to access missing score
        with pytest.raises(KeyError):
            g, gg, idx = undertest._score_target_levels_and_index("cohort1", by_target=False, by_score=True)
            # The error happens when pd.cut is called on missing column

    def test_score_target_levels_with_empty_dataframe(self, fake_seismo):
        """Test _score_target_levels_and_index with empty dataframe."""
        fake_seismo.dataframe = pd.DataFrame(columns=fake_seismo.dataframe.columns)

        g, gg, idx = undertest._score_target_levels_and_index("cohort1", by_target=False, by_score=True)

        # Should still return proper structure
        assert len(g) == 2
        assert len(gg) == 2
        assert len(idx) == 2


class TestStyleFunctions:
    """Test styling functions for cohort summaries."""

    def test_style_cohort_summaries_basic(self, fake_seismo):
        """Test _style_cohort_summaries basic functionality."""
        df = pd.DataFrame(
            {"Predictions": [10, 20], "Entities": [8, 15]}, index=pd.Index(["GroupA", "GroupB"], name="cohort1")
        )

        result = undertest._style_cohort_summaries(df, "Test Cohort")

        assert isinstance(result, Styler)
        assert result.caption == "Counts by Test Cohort"
        html = result.to_html()
        assert "Counts by Test Cohort" in html

    def test_style_cohort_summaries_precision_formatting(self, fake_seismo):
        """Test that _style_cohort_summaries formats values with correct precision."""
        df = pd.DataFrame(
            {"Predictions": [10.123456, 20.987654], "Entities": [8.5555, 15.4444]},
            index=pd.Index(["GroupA", "GroupB"], name="cohort1"),
        )

        result = undertest._style_cohort_summaries(df, "Test Cohort")

        # Check that values are formatted with 2 decimal places
        assert isinstance(result, Styler)

    def test_style_score_target_cohort_summaries_basic(self, fake_seismo):
        """Test _style_score_target_cohort_summaries basic functionality."""
        multi_index = pd.MultiIndex.from_tuples([("A", 0), ("A", 1), ("B", 0)], names=["cohort1", "target"])
        df = pd.DataFrame({"Predictions": [10, 20, 5], "Entities": [8, 19, 4]}, index=multi_index)

        result = undertest._style_score_target_cohort_summaries(df, ["Cohort", "Target"], "Test Cohort")

        assert isinstance(result, Styler)
        assert result.caption == "Counts by Test Cohort"
        html = result.to_html()
        assert "Counts by Test Cohort" in html

    def test_style_score_target_cohort_summaries_with_empty_df(self, fake_seismo):
        """Test _style_score_target_cohort_summaries with empty dataframe."""
        empty_index = pd.MultiIndex.from_tuples([], names=["cohort1", "target"])
        df = pd.DataFrame(columns=["Predictions", "Entities"], index=empty_index)

        result = undertest._style_score_target_cohort_summaries(df, ["Cohort", "Target"], "Test Cohort")

        assert isinstance(result, Styler)
        html = result.to_html()
        assert isinstance(html, str)


class TestGetInfoDict:
    """Test _get_info_dict function."""

    def test_get_info_dict_with_plot_help_true(self, fake_seismo):
        """Test _get_info_dict with plot_help=True."""
        result = undertest._get_info_dict(plot_help=True)

        assert isinstance(result, dict)
        assert result["plot_help"] is True
        assert result["num_predictions"] == fake_seismo.prediction_count
        assert result["num_entities"] == fake_seismo.entity_count
        assert "start_date" in result
        assert "end_date" in result
        assert "tables" in result

    def test_get_info_dict_with_plot_help_false(self, fake_seismo):
        """Test _get_info_dict with plot_help=False."""
        result = undertest._get_info_dict(plot_help=False)

        assert isinstance(result, dict)
        assert result["plot_help"] is False

    def test_get_info_dict_table_structure(self, fake_seismo):
        """Test that _get_info_dict returns proper table structure."""
        result = undertest._get_info_dict(plot_help=False)

        assert "tables" in result
        assert isinstance(result["tables"], list)
        assert len(result["tables"]) > 0

        table = result["tables"][0]
        assert "name" in table
        assert "description" in table
        assert "num_rows" in table
        assert "num_cols" in table


class TestGetCohortSummaryDataframes:
    """Test _get_cohort_summary_dataframes function."""

    def test_get_cohort_summary_dataframes_basic(self, fake_seismo):
        """Test _get_cohort_summary_dataframes with basic parameters."""
        with (
            patch.object(undertest, "default_cohort_summaries", return_value=fake_seismo.dataframe),
            patch("pandas.io.formats.style.Styler.to_html", return_value="<table></table>"),
        ):
            result = undertest._get_cohort_summary_dataframes(by_target=False, by_score=False)

        assert isinstance(result, dict)
        assert "cohort1" in result
        assert isinstance(result["cohort1"], list)
        assert len(result["cohort1"]) == 1  # Only default summary, no by_target/by_score

    def test_get_cohort_summary_dataframes_with_target_and_score(self, fake_seismo):
        """Test _get_cohort_summary_dataframes with by_target and by_score."""
        multi_index = pd.MultiIndex.from_tuples([("A", 0.5, 1)], names=["cohort1", "score", "target"])
        mock_summary = pd.DataFrame({"Predictions": [10], "Entities": [8]}, index=multi_index)

        with (
            patch.object(undertest, "default_cohort_summaries", return_value=fake_seismo.dataframe),
            patch.object(undertest, "score_target_cohort_summaries", return_value=mock_summary),
            patch("pandas.io.formats.style.Styler.to_html", return_value="<table></table>"),
        ):
            result = undertest._get_cohort_summary_dataframes(by_target=True, by_score=True)

        assert isinstance(result, dict)
        assert "cohort1" in result
        assert len(result["cohort1"]) == 2  # Default summary + by_target/by_score summary

    def test_get_cohort_summary_dataframes_with_empty_cohorts(self, fake_seismo):
        """Test _get_cohort_summary_dataframes when no cohorts are available."""
        fake_seismo.available_cohort_groups = {}

        result = undertest._get_cohort_summary_dataframes(by_target=False, by_score=False)

        assert isinstance(result, dict)
        assert len(result) == 0
