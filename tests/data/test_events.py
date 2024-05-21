from pathlib import Path

import pandas as pd
import pytest

import seismometer.data.pandas_helpers as undertest

RES_SUB_DIR = Path("data") / "score_selection"


@pytest.fixture
def event_data(res):
    res = res / RES_SUB_DIR

    input_frame = pd.read_csv(
        res / "input_event_scores.tsv",
        sep="\t",
        parse_dates=["PredictTime", "Target_Time", "Reference_5_15_Time"],
        index_col=False,
    )
    expected_frame = pd.read_csv(res / "expected_event_scores.tsv", sep="\t", index_col=False)

    return input_frame, expected_frame


class Test_Event_Score:
    @pytest.mark.parametrize("ref_event", ["Target", "PredictTime", "Reference_5_15_Time"])
    @pytest.mark.parametrize("summary_method", ["min", "max", "first", "last"])
    @pytest.mark.parametrize("id_, csn", [pytest.param(1, 0, id="monotonic-increasing")])
    def test_bad_event(self, id_, csn, summary_method, ref_event, event_data):
        input_frame, expected_frame = event_data
        expected_score = expected_frame.loc[
            (expected_frame["Id"] == id_)
            & (expected_frame["CSN"] == csn)
            & (expected_frame["ref_event"] == ref_event),
            summary_method,
        ]

        actual = undertest.event_score(input_frame, ["Id", "CSN"], "ModelScore", ref_event, summary_method)
        assert actual["ModelScore"].tolist() == expected_score.tolist()
