from datetime import datetime
from unittest import mock

import pandas as pd
import pytest
from conftest import TEST_ROOT

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
