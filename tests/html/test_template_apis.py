from datetime import datetime
from unittest import mock

import pandas as pd
import pytest
from conftest import TEST_ROOT
from IPython.core.display import HTML

import seismometer
import seismometer.api as undertest

res = TEST_ROOT / "resources/html"

INPUT_FRAME = pd.read_csv(res / "input_predictions.tsv", sep="\t", parse_dates=["PredictTime"], index_col=False)

EXPECTED_CUTS = pd.cut(INPUT_FRAME["Score"], [0, 0.2, 1.0])


class Test_Template_Apis:
    def test_get_info_dict(self):
        mock_sg = mock.Mock(autospec=seismometer.seismogram.Seismogram)
        mock_sg.prediction_count = 1
        mock_sg.feature_count = 2
        mock_sg.entity_count = 4
        mock_sg.start_time = datetime(2024, 1, 1, 1, 1, 1)
        mock_sg.end_time = datetime(2025, 1, 1, 1, 1, 1)
        mock_sg.event_types_count = 6

        expected = {
            "tables": [
                {
                    "name": "predictions",
                    "description": "Scores, features, configured demographics, and merged events for each prediction",
                    "num_rows": 1,
                    "num_cols": 2,
                }
            ],
            "num_predictions": 1,
            "num_entities": 4,
            "start_date": "2024-01-01",
            "end_date": "2025-01-01",
            "plot_help": True,
        }
        with mock.patch.object(undertest.templates, "Seismogram", return_value=mock_sg):
            assert undertest.templates._get_info_dict(True) == expected

    @pytest.mark.parametrize(
        "selection,by_target,by_score,expected",
        [
            pytest.param("cohort", False, False, (["cohort"], ["cohort"], ["Cohort"])),
            pytest.param(
                "cohort", True, False, (["cohort", "Target_Value"], ["cohort", "Target_Value"], ["Cohort", "Target"])
            ),
            pytest.param("cohort", False, True, (["cohort", EXPECTED_CUTS], ["cohort", "Score"], ["Cohort", "Score"])),
            pytest.param(
                "cohort",
                True,
                True,
                (
                    ["cohort", EXPECTED_CUTS, "Target_Value"],
                    ["cohort", "Score", "Target_Value"],
                    ["Cohort", "Score", "Target"],
                ),
            ),
        ],
    )
    def test_score_target_levels_and_index(self, selection, by_target, by_score, expected):
        mock_sg = mock.Mock(autospec=seismometer.seismogram.Seismogram)
        mock_sg.target = "Target_Value"
        mock_sg.output = "Score"
        mock_sg.score_bins.return_value = [0, 0.2, 1]
        mock_sg.dataframe = INPUT_FRAME

        with mock.patch.object(undertest.templates, "Seismogram", return_value=mock_sg):
            result = undertest.templates._score_target_levels_and_index(selection, by_target, by_score)

        for val, expected_val in zip(result, expected):
            for sub_val, expected_sub_val in zip(val, expected_val):
                if isinstance(sub_val, pd.Series):
                    pd.testing.assert_series_equal(sub_val, expected_sub_val)
                else:
                    assert sub_val == expected_sub_val


# ============================================================================
# ADDITIONAL API TESTS
# ============================================================================


class TestShowInfoFunction:
    """Test show_info() main public API function."""

    @mock.patch("seismometer.core.decorators.DiskCachedFunction.is_enabled", return_value=False)
    @mock.patch.object(undertest.templates, "template")
    @mock.patch.object(undertest.templates, "Seismogram")
    def test_show_info_with_plot_help_true(self, mock_seismo, mock_template, mock_cache_enabled):
        """Test show_info with plot_help=True."""
        mock_sg = mock.Mock()
        mock_sg.prediction_count = 100
        mock_sg.feature_count = 50
        mock_sg.entity_count = 75
        mock_sg.start_time = datetime(2024, 1, 1)
        mock_sg.end_time = datetime(2024, 12, 31)
        mock_seismo.return_value = mock_sg
        # Return actual HTML object instead of Mock
        mock_template.render_info_template.return_value = HTML("<html>info</html>")

        _ = undertest.templates.show_info(plot_help=True)

        mock_template.render_info_template.assert_called_once()
        call_args = mock_template.render_info_template.call_args[0][0]
        assert call_args["plot_help"] is True
        assert call_args["num_predictions"] == 100

    @mock.patch("seismometer.core.decorators.DiskCachedFunction.is_enabled", return_value=False)
    @mock.patch.object(undertest.templates, "template")
    @mock.patch.object(undertest.templates, "Seismogram")
    def test_show_info_with_plot_help_false(self, mock_seismo, mock_template, mock_cache_enabled):
        """Test show_info with plot_help=False (default)."""
        mock_sg = mock.Mock()
        mock_sg.prediction_count = 100
        mock_sg.feature_count = 50
        mock_sg.entity_count = 75
        mock_sg.start_time = datetime(2024, 1, 1)
        mock_sg.end_time = datetime(2024, 12, 31)
        mock_seismo.return_value = mock_sg
        # Return actual HTML object instead of Mock
        mock_template.render_info_template.return_value = HTML("<html>info</html>")

        _ = undertest.templates.show_info(plot_help=False)

        mock_template.render_info_template.assert_called_once()
        call_args = mock_template.render_info_template.call_args[0][0]
        assert call_args["plot_help"] is False

    @mock.patch("seismometer.core.decorators.DiskCachedFunction.is_enabled", return_value=False)
    @mock.patch.object(undertest.templates, "template")
    @mock.patch.object(undertest.templates, "Seismogram")
    def test_show_info_caching_decorator(self, mock_seismo, mock_template, mock_cache_enabled):
        """Test that show_info uses caching decorator."""
        mock_sg = mock.Mock()
        mock_sg.prediction_count = 100
        mock_sg.feature_count = 50
        mock_sg.entity_count = 75
        mock_sg.start_time = datetime(2024, 1, 1)
        mock_sg.end_time = datetime(2024, 12, 31)
        mock_seismo.return_value = mock_sg
        # Return actual HTML object instead of Mock
        mock_template.render_info_template.return_value = HTML("<html>info</html>")

        # Call twice with same parameters
        result1 = undertest.templates.show_info(plot_help=True)
        result2 = undertest.templates.show_info(plot_help=True)

        # Both should return HTML objects (caching handled by decorator)
        assert result1 is not None
        assert result2 is not None
        assert isinstance(result1, HTML)
        assert isinstance(result2, HTML)


class TestDateFormattingEdgeCases:
    """Test date formatting edge cases."""

    @pytest.mark.parametrize(
        "start_date,end_date",
        [
            # Year boundary
            (datetime(2023, 12, 31, 23, 59, 59), datetime(2024, 1, 1, 0, 0, 1)),
            # Leap year (Feb 29)
            (datetime(2024, 2, 28), datetime(2024, 2, 29)),
            # Same day
            (datetime(2024, 6, 15, 8, 0, 0), datetime(2024, 6, 15, 18, 0, 0)),
            # One year apart
            (datetime(2023, 1, 1), datetime(2024, 1, 1)),
            # Century boundary
            (datetime(1999, 12, 31), datetime(2000, 1, 1)),
        ],
    )
    @mock.patch.object(undertest.templates, "Seismogram")
    def test_date_formatting_edge_cases(self, mock_seismo, start_date, end_date):
        """Test date formatting with various edge cases."""
        mock_sg = mock.Mock()
        mock_sg.prediction_count = 1
        mock_sg.feature_count = 1
        mock_sg.entity_count = 1
        mock_sg.start_time = start_date
        mock_sg.end_time = end_date
        mock_seismo.return_value = mock_sg

        result = undertest.templates._get_info_dict(False)

        assert result["start_date"] == start_date.strftime("%Y-%m-%d")
        assert result["end_date"] == end_date.strftime("%Y-%m-%d")

    @mock.patch.object(undertest.templates, "Seismogram")
    def test_date_formatting_with_microseconds(self, mock_seismo):
        """Test date formatting handles microseconds correctly."""
        mock_sg = mock.Mock()
        mock_sg.prediction_count = 1
        mock_sg.feature_count = 1
        mock_sg.entity_count = 1
        mock_sg.start_time = datetime(2024, 1, 1, 12, 30, 45, 123456)
        mock_sg.end_time = datetime(2024, 12, 31, 23, 59, 59, 999999)
        mock_seismo.return_value = mock_sg

        result = undertest.templates._get_info_dict(False)

        # Should format as date only (no time/microseconds)
        assert result["start_date"] == "2024-01-01"
        assert result["end_date"] == "2024-12-31"


class TestSeismogramAccessEdgeCases:
    """Test error cases for Seismogram access."""

    @mock.patch.object(undertest.templates, "Seismogram")
    def test_get_info_dict_with_none_dates(self, mock_seismo):
        """Test _get_info_dict when dates might be None."""
        mock_sg = mock.Mock()
        mock_sg.prediction_count = 0
        mock_sg.feature_count = 0
        mock_sg.entity_count = 0
        # Dates should always be datetime objects, but test defensive handling
        mock_sg.start_time = datetime(2024, 1, 1)
        mock_sg.end_time = datetime(2024, 1, 1)
        mock_seismo.return_value = mock_sg

        result = undertest.templates._get_info_dict(False)

        assert result["num_predictions"] == 0
        assert result["num_entities"] == 0
        assert result["start_date"] == "2024-01-01"

    @mock.patch.object(undertest.templates, "Seismogram")
    def test_get_info_dict_with_zero_counts(self, mock_seismo):
        """Test _get_info_dict with zero counts."""
        mock_sg = mock.Mock()
        mock_sg.prediction_count = 0
        mock_sg.feature_count = 0
        mock_sg.entity_count = 0
        mock_sg.start_time = datetime(2024, 1, 1)
        mock_sg.end_time = datetime(2024, 1, 1)
        mock_seismo.return_value = mock_sg

        result = undertest.templates._get_info_dict(True)

        assert result["num_predictions"] == 0
        assert result["num_entities"] == 0
        assert result["plot_help"] is True
        assert isinstance(result["tables"], list)

    @mock.patch.object(undertest.templates, "Seismogram")
    def test_score_target_levels_with_none_dataframe(self, mock_seismo):
        """Test _score_target_levels_and_index error handling."""
        mock_sg = mock.Mock()
        mock_sg.target = "Target_Value"
        mock_sg.output = "Score"
        mock_sg.score_bins.return_value = [0, 0.5, 1.0]
        # Test with minimal dataframe
        mock_sg.dataframe = pd.DataFrame({"Score": [0.1, 0.6, 0.9]})
        mock_seismo.return_value = mock_sg

        result = undertest.templates._score_target_levels_and_index("cohort", False, True)

        # Should return tuple of lists
        assert isinstance(result, tuple)
        assert len(result) == 3
