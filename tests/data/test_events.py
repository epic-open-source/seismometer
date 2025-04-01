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
    @pytest.mark.parametrize("aggregation_method", ["min", "max", "first", "last"])
    @pytest.mark.parametrize("id_, ctx", [pytest.param(1, 0, id="monotonic-increasing")])
    def test_bad_event(self, id_, ctx, aggregation_method, ref_event, event_data):
        input_frame, expected_frame = event_data
        expected_score = expected_frame.loc[
            (expected_frame["Id"] == id_)
            & (expected_frame["CtxId"] == ctx)
            & (expected_frame["ref_event"] == ref_event),
            aggregation_method,
        ]
        if ref_event in ["PredictTime", "Reference_5_15_Time"] and aggregation_method in ["min", "max"]:
            actual = undertest.event_score(input_frame, ["Id", "CtxId"], "ModelScore", "Target", aggregation_method)
            assert actual["ModelScore"].tolist() == expected_score.tolist()
        else:
            actual = undertest.event_score(input_frame, ["Id", "CtxId"], "ModelScore", ref_event, aggregation_method)
            assert actual["ModelScore"].tolist() == expected_score.tolist()

    def test_max_aggregation(self):
        input_frame = pd.DataFrame(
            {"Id": [1, 1, 1, 2], "CtxId": [10, 10, 10, 20], "ModelScore": [1, 2, 3, 1], "Target_Value": [1, 1, 0, 0]}
        )
        actual = undertest.max_aggregation(input_frame, ["Id", "CtxId"], "ModelScore", "Target")
        expected = pd.DataFrame(
            {"Id": [1, 2], "CtxId": [10, 20], "ModelScore": [2, 1], "Target_Value": [1, 0]}, index=[1, 3]
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_min_aggregation(self):
        input_frame = pd.DataFrame(
            {"Id": [1, 1, 1, 2], "CtxId": [10, 10, 10, 20], "ModelScore": [1, 2, 3, 1], "Target_Value": [0, 1, 0, 1]}
        )
        actual = undertest.min_aggregation(input_frame, ["Id", "CtxId"], "ModelScore", "Target")
        expected = pd.DataFrame(
            {"Id": [1, 2], "CtxId": [10, 20], "ModelScore": [1, 1], "Target_Value": [0, 1]}, index=[0, 3]
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_first_aggregation(self):
        input_frame = pd.DataFrame(
            {
                "Id": [1, 1, 1, 2],
                "CtxId": [10, 10, 10, 20],
                "ModelScore": [1, 2, 3, 1],
                "EventTime": pd.to_datetime([None, "2024-02-01 08:00", "2024-02-01 10:00", None]),
            }
        )
        actual = undertest.first_aggregation(input_frame, ["Id", "CtxId"], "ModelScore", "EventTime")
        expected = pd.DataFrame(
            {"Id": [1], "CtxId": [10], "ModelScore": [2], "EventTime": pd.to_datetime(["2024-02-01 08:00"])}, index=[1]
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_last_aggregation(self):
        input_frame = pd.DataFrame(
            {
                "Id": [1, 1, 1, 2],
                "CtxId": [10, 10, 10, 20],
                "ModelScore": [1, 2, 3, 1],
                "EventTime": pd.to_datetime(["2024-02-01 08:00", None, "2024-02-01 10:00", "2024-02-01 20:00"]),
            }
        )
        actual = undertest.last_aggregation(input_frame, ["Id", "CtxId"], "ModelScore", "EventTime")
        expected = pd.DataFrame(
            {
                "Id": [1, 2],
                "CtxId": [10, 20],
                "ModelScore": [3, 1],
                "EventTime": pd.to_datetime(["2024-02-01 10:00", "2024-02-01 20:00"]),
            },
            index=[2, 3],
        ).sort_values(by="EventTime", ascending=False)
        pd.testing.assert_frame_equal(actual, expected)

    def test_invalid_aggregation_method(self):
        input_frame = pd.DataFrame(
            {
                "Id": [1, 1, 1, 2],
                "CtxId": [10, 10, 10, 20],
                "ModelScore": [1, 2, 3, 1],
                "PredictTime": pd.to_datetime(
                    ["2024-02-01 08:00", "2024-02-01 09:00", "2024-02-01 10:00", "2024-02-01 13:00"]
                ),
            }
        )
        with pytest.raises(ValueError, match="Unknown aggregation method: invalid"):
            _ = undertest.event_score(input_frame, ["Id", "CtxId"], "ModelScore", "PredictTime", "invalid")
