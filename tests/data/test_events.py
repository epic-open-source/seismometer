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


# fmt: off

class Test_Event_Score:
    @pytest.mark.parametrize(
            "ref_time,ref_event", [("Target", "Target"), ("PredictTime", "Target"), ("Reference_5_15_Time", "Target")]
            )
    @pytest.mark.parametrize("aggregation_method", ["min", "max", "first", "last"])
    @pytest.mark.parametrize("id_, ctx", [pytest.param(1, 0, id="monotonic-increasing")])
    def test_bad_event(self, id_, ctx, aggregation_method, ref_time, ref_event, event_data):
        input_frame, expected_frame = event_data

        ref_col = ref_event if aggregation_method in ["min", "max"] else ref_time
        expected_score = expected_frame.loc[
            (expected_frame["Id"] == id_)
            & (expected_frame["CtxId"] == ctx)
            & (expected_frame["ref_event"] == ref_col),
            aggregation_method,
        ]

        actual = undertest.event_score(
            input_frame, ["Id", "CtxId"], "ModelScore", ref_time, ref_event, aggregation_method
            )

        assert actual["ModelScore"].tolist() == expected_score.tolist()

    @pytest.mark.parametrize(
        "aggregation_method,entities,scores,targets,event_times,selected_rows",
        [
            # Select max with target_value=1
            (
                "max", [1, 1, 2, 2], [1, 2, 3, 1], [1, 1, 0, 1],
                ["2024-02-01 08:00", "2024-02-01 10:00", None, "2024-02-01 14:00"],
                [1, 3],
            ),
            (
                "max", [1, 1, 2, 2], [1, 2, 3, 1], [0, 1, 0, 1],
                [None, "2024-02-01 10:00", None, "2024-02-01 14:00"],
                [1, 3],
            ),
            (
                "max", [1, 1, 2, 2], [1, 2, 3, 1], [0, 0, 1, 0],
                [None, None, "2024-02-01 12:00", None],
                [2, 1],
            ),
            # Select min with target_value=1
            (
                "min", [1, 1, 2, 2], [1, 2, 3, 1], [1, 1, 0, 1],
                ["2024-02-01 08:00", "2024-02-01 10:00", None, "2024-02-01 14:00"],
                [0, 3],
            ),
            (
                "first", [1, 1, 2, 2], [1, 2, 3, 1], [1, 1, 0, 1],
                ["2024-02-01 10:00", "2024-02-01 08:00", None, "2024-02-01 14:00"],
                [1, 3],
            ),
            (
                "last", [1, 1, 2, 2], [1, 2, 3, 1], [1, 1, 1, 0],
                ["2024-02-01 08:00", "2024-02-01 10:00", "2024-02-01 12:00", None],
                [2, 1],
            ),
        ],
    )
    def test_event_score(self, aggregation_method, entities, scores, targets, event_times, selected_rows):
        input_frame = pd.DataFrame(
            {
                "Id": entities,
                "CtxId": [entity * 10 for entity in entities],
                "ModelScore": scores,
                "Target_Value": targets,
                "EventTime": pd.to_datetime(event_times),
            }
        )
        expected = input_frame.loc[pd.Index(selected_rows)]
        # ref_event = "Target_Value" if aggregation_method in ["max", "min"] else "EventTime"
        actual = undertest.event_score(
            input_frame, ["Id", "CtxId"], "ModelScore", "EventTime", "Target", aggregation_method=aggregation_method
        )
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "entities,scores,targets,selected_rows",
        [
            # For entity 1, we take max score over scores associated with target = 1.
            ([1, 1, 1, 2], [1, 2, 3, 1], [1, 1, 0, 0], [1, 3]),
            # For entity 1, there are no scores associated with target = 1, we just take the max score.
            ([1, 1, 1, 2], [1, 2, 3, 1], [0, 0, 0, 1], [3, 2]),
            # Since we order by target (descending) then score (descending), the order of selected rows is [3, 1].
            ([1, 1, 1, 2], [1, 2, 3, 4], [0, 1, 0, 1], [3, 1]),
        ],
    )
    def test_max_aggregation(self, entities, scores, targets, selected_rows):
        input_frame = pd.DataFrame(
            {
                "Id": entities,
                "CtxId": [entity * 10 for entity in entities],
                "ModelScore": scores,
                "Target_Value": targets,
            }
        )
        expected = input_frame.loc[pd.Index(selected_rows)]
        actual = undertest.max_aggregation(input_frame, ["Id", "CtxId"], "ModelScore", None, "Target")
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "entities,scores,targets,selected_rows",
        [
            # For entity 1, we take min score over scores associated with target = 1.
            ([1, 1, 1, 2], [1, 2, 3, 1], [0, 1, 0, 1], [3, 1]),
            # For entity 1, there are no scores associated with target = 1, we just take the min score.
            ([1, 1, 1, 2], [1, 2, 3, 1], [0, 0, 0, 1], [3, 0]),
            # Since we order by target (descending) then score (ascending), the order of selected rows is [3,2].
            ([1, 1, 1, 2], [1, 2, 3, 1], [0, 0, 1, 1], [3, 2]),
        ],
    )
    def test_min_aggregation(self, entities, scores, targets, selected_rows):
        input_frame = pd.DataFrame(
            {
                "Id": entities,
                "CtxId": [entity * 10 for entity in entities],
                "ModelScore": scores,
                "Target_Value": targets,
            }
        )
        expected = input_frame.loc[pd.Index(selected_rows)]
        actual = undertest.min_aggregation(input_frame, ["Id", "CtxId"], "ModelScore", None, "Target")
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "entities,scores,ref_times,selected_rows",
        [
            # For entity 1, we take the row associated with the first non-na time in ref_times.
            ([1, 1, 1, 2], [1, 2, 3, 1], [None, "2024-02-01 08:00", "2024-02-01 10:00", None], [1]),
            # Since we order by event time (ascending), the order of selected rows is [2, 0].
            ([1, 1, 2, 2], [5, 6, 3, 4], ["2024-02-01 01:00", "2024-02-01 08:00", "2024-01-31 10:00", None], [2, 0]),
            ([1, 1, 2, 2], [1, 2, 3, 1], [None, None, None, None], []),
        ],
    )
    def test_first_aggregation(self, entities, scores, ref_times, selected_rows):
        input_frame = pd.DataFrame(
            {
                "Id": entities,
                "CtxId": [entity * 10 for entity in entities],
                "ModelScore": scores,
                "EventTime": pd.to_datetime(ref_times),
            }
        )
        expected = input_frame.loc[pd.Index(selected_rows)]
        actual = undertest.first_aggregation(input_frame, ["Id", "CtxId"], "ModelScore", "EventTime", None)
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "entities,scores,ref_times,selected_rows",
        [
            # For entity 1, we take the row associated with the last non-na time in ref_times.
            ([1, 1, 1, 2], [1, 2, 3, 1], ["2024-02-01 08:00", None, "2024-02-02 10:00", None], [2],),
            # Since we order by event time (descending), the order of selected rows is [2, 1].
            ([1, 1, 2, 2], [5, 6, 3, 4], ["2024-02-01 01:00", "2024-02-01 08:00", "2024-02-21 10:00", None], [2, 1],),
            ([1, 1, 2, 2], [1, 2, 3, 1], [None, None, None, None], [],),
        ],
    )
    def test_last_aggregation(self, entities, scores, ref_times, selected_rows):
        input_frame = pd.DataFrame(
            {
                "Id": entities,
                "CtxId": [entity * 10 for entity in entities],
                "ModelScore": scores,
                "EventTime": pd.to_datetime(ref_times),
            }
        )
        expected = input_frame.loc[pd.Index(selected_rows)]
        actual = undertest.last_aggregation(input_frame, ["Id", "CtxId"], "ModelScore", "EventTime", None)
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "aggregation_method, expected_rows",
        [
            ("max", [1, 3]),
            ("min", [0, 3]),
            ("first", [0, 3]),
            ("last", [3, 1]),
        ],
    )
    def test_aggregation_multiple_context_per_id(self, aggregation_method, expected_rows):
        input_frame = pd.DataFrame(
            {
                "Id": [1, 1, 1, 1],  # only one ID
                "CtxId": [10, 10, 20, 20],
                "ModelScore": [1, 2, 3, 1],
                "Target_Value": [1, 1, 0, 1],
                "EventTime": pd.to_datetime(["2024-02-01 08:00", "2024-02-01 10:00", None, "2024-02-01 12:00"]),
            }
        )
        expected = input_frame.loc[pd.Index(expected_rows)]

        if aggregation_method == "max":
            actual = undertest.max_aggregation(
                input_frame, ["Id", "CtxId"], "ModelScore", "EventTime", "Target_Value"
                )
        elif aggregation_method == "min":
            actual = undertest.min_aggregation(
                input_frame, ["Id", "CtxId"], "ModelScore", "EventTime", "Target_Value"
                )
        elif aggregation_method == "first":
            actual = undertest.first_aggregation(
                input_frame, ["Id", "CtxId"], "ModelScore", "EventTime", "Target_Value"
                )
        elif aggregation_method == "last":
            actual = undertest.last_aggregation(
                input_frame, ["Id", "CtxId"], "ModelScore", "EventTime", "Target_Value"
                )

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
            _ = undertest.event_score(input_frame, ["Id", "CtxId"], "ModelScore", "PredictTime", None, "invalid")

    @pytest.mark.parametrize(
            "agg_method, ref_col", [
                ("min", "ref_event"), ("max", "ref_event"), ("first", "ref_time"), ("last", "ref_time")
                ]
    )
    def test_aggregation_missing_ref_col(self, agg_method, ref_col):
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
        with pytest.raises(ValueError, match=f"With aggregation_method '{agg_method}', {ref_col} is required."):
            _ = undertest.event_score(input_frame, ["Id", "CtxId"], "ModelScore", None, None, agg_method)


class TestEventScoreErrorHandling:
    """Test error handling and edge cases for event_score and aggregation functions"""

    def test_missing_entity_keys_column(self):
        """Test event_score() with missing entity_keys column"""
        input_frame = pd.DataFrame(
            {
                "Id": [1, 1, 2, 2],
                "ModelScore": [1, 2, 3, 4],
                "Target_Value": [0, 1, 0, 1],
            }
        )
        # Request a column that doesn't exist - should silently filter it out
        # (line 627: pks = [c for c in pks if c in merged_frame.columns])
        result = undertest.event_score(
            input_frame, ["Id", "NonExistentColumn"], "ModelScore", None, "Target", "max"
        )
        # Should still work, just using the columns that do exist
        assert result is not None
        assert len(result) == 2  # One row per Id

    def test_all_entity_keys_missing(self):
        """Test event_score() when all entity_keys columns are missing"""
        input_frame = pd.DataFrame(
            {
                "Id": [1, 1, 2, 2],
                "ModelScore": [1, 2, 3, 4],
                "Target_Value": [0, 1, 0, 1],
            }
        )
        # EDGE CASE: When all requested columns don't exist, pks becomes empty list
        # This causes drop_duplicates(subset=[]) to fail with confusing error
        # Better error message would be helpful here
        with pytest.raises(ValueError, match="not enough values to unpack"):
            _ = undertest.event_score(
                input_frame, ["NonExistent1", "NonExistent2"], "ModelScore", None, "Target", "max"
            )

    def test_case_sensitivity_aggregation_method(self):
        """Test event_score() case sensitivity for aggregation_method"""
        input_frame = pd.DataFrame(
            {
                "Id": [1, 1, 2, 2],
                "ModelScore": [1, 2, 3, 4],
                "Target_Value": [0, 1, 0, 1],
            }
        )
        # "Max" (capitalized) should not match "max"
        with pytest.raises(ValueError, match="Unknown aggregation method: Max"):
            _ = undertest.event_score(input_frame, ["Id"], "ModelScore", None, "Target", "Max")

    def test_event_score_both_ref_none_with_max(self):
        """Test event_score() with both ref_time and ref_event = None using max aggregation"""
        input_frame = pd.DataFrame(
            {
                "Id": [1, 1, 2, 2],
                "ModelScore": [1, 2, 3, 4],
            }
        )
        # max_aggregation requires ref_event
        with pytest.raises(ValueError, match="With aggregation_method 'max', ref_event is required."):
            _ = undertest.event_score(input_frame, ["Id"], "ModelScore", None, None, "max")

    def test_event_score_both_ref_none_with_min(self):
        """Test event_score() with both ref_time and ref_event = None using min aggregation"""
        input_frame = pd.DataFrame(
            {
                "Id": [1, 1, 2, 2],
                "ModelScore": [1, 2, 3, 4],
            }
        )
        # min_aggregation requires ref_event
        with pytest.raises(ValueError, match="With aggregation_method 'min', ref_event is required."):
            _ = undertest.event_score(input_frame, ["Id"], "ModelScore", None, None, "min")

    def test_max_aggregation_with_nan_in_target(self):
        """Test max_aggregation() with NaN values in Target column"""
        import numpy as np

        input_frame = pd.DataFrame(
            {
                "Id": [1, 1, 1, 2, 2],
                "ModelScore": [1, 2, 3, 4, 5],
                "Target_Value": [np.nan, 1, 0, np.nan, np.nan],
            }
        )
        # NaN values in target should be handled gracefully (sorted to end by pandas)
        result = undertest.max_aggregation(input_frame, ["Id"], "ModelScore", None, "Target")

        # Should return one row per Id
        assert len(result) == 2
        # For Id=1, should select row with Target=1 (highest target, then highest score)
        assert result[result["Id"] == 1]["ModelScore"].iloc[0] == 2
        # For Id=2, all targets are NaN, should select highest score
        assert result[result["Id"] == 2]["ModelScore"].iloc[0] == 5

    def test_min_aggregation_with_nan_in_target(self):
        """Test min_aggregation() with NaN values in Target column"""
        import numpy as np

        input_frame = pd.DataFrame(
            {
                "Id": [1, 1, 1, 2, 2],
                "ModelScore": [1, 2, 3, 4, 5],
                "Target_Value": [np.nan, 0, 1, np.nan, np.nan],
            }
        )
        # NaN values in target should be handled gracefully
        result = undertest.min_aggregation(input_frame, ["Id"], "ModelScore", None, "Target")

        # Should return one row per Id
        assert len(result) == 2
        # For Id=1, should select row with Target=1 (highest target, then lowest score among Target=1)
        assert result[result["Id"] == 1]["ModelScore"].iloc[0] == 3
        # For Id=2, all targets are NaN, should select lowest score
        assert result[result["Id"] == 2]["ModelScore"].iloc[0] == 4

    def test_all_nan_targets_max_aggregation(self):
        """Test max_aggregation() when all Target values are NaN"""
        import numpy as np

        input_frame = pd.DataFrame(
            {
                "Id": [1, 1, 2, 2],
                "ModelScore": [1, 2, 3, 4],
                "Target_Value": [np.nan, np.nan, np.nan, np.nan],
            }
        )
        # Should still work, selecting max score when all targets are NaN
        result = undertest.max_aggregation(input_frame, ["Id"], "ModelScore", None, "Target")

        assert len(result) == 2
        # Should select highest scores (2 for Id=1, 4 for Id=2)
        assert result[result["Id"] == 1]["ModelScore"].iloc[0] == 2
        assert result[result["Id"] == 2]["ModelScore"].iloc[0] == 4

# fmt: on
