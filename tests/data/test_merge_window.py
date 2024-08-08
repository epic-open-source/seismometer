import logging

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import seismometer.data.pandas_helpers as undertest

# region test helpers
DEFAULT_PRED_TIME_STR = "2024-02-01 01:00:00"
DEFAULT_TIME = pd.Timestamp("2024-02-01 12:00:00")
DAY = pd.Timedelta(days=1)


def create_event_table(ids, ctxs, event_labels, event_offsets=None, event_values=None, event_times=None):
    count = len(event_times)
    return pd.DataFrame(
        {
            "Id": ids * count if len(ids) == 1 else ids,
            "CtxId": ctxs * count if len(ctxs) == 1 else ctxs,
            "Type": event_labels,
            "Time": event_times,
            "Value": event_values,
        }
    )


def create_prediction_table(ids, ctxs, predtimes):
    count = len(predtimes)
    return pd.DataFrame(
        {
            "Id": ids * count if len(ids) == 1 else ids,
            "CtxId": ctxs * count if len(ctxs) == 1 else ctxs,
            "PredictTime": pd.to_datetime(predtimes),
        }
    )


def create_pred_event_frame(event, labels, times):
    return pd.DataFrame({undertest.event_value(event): labels, undertest.event_time(event): pd.to_datetime(times)})


# endregion


class TestMergeWindowedEvent:
    @pytest.mark.parametrize(
        "id_, ctx, event_val, event_time, window, window_offset, out_time, out_val",
        [
            #
            # No event impute to negative
            pytest.param(0, 0, 1, None, 10, 0, None, 1, id="pos val no event: no change"),
            pytest.param(0, 1, 0, None, 10, 0, None, 0, id="neg val no event: no change"),
            pytest.param(0, 2, None, None, 10, 0, None, 0, id="null val no event: impute negative"),
            #
            # Predictions far away from event (early-predicition) have associated event cleared out
            pytest.param(
                1, 0, 1, "2024-02-01 12:00:00", 10, 0, None, 0, id="pos val before_window: clear time+impute neg"
            ),
            pytest.param(1, 1, 0, "2024-02-01 12:00:00", 10, 0, None, 0, id="neg val before_window: clear time"),
            pytest.param(
                1,
                2,
                None,
                "2024-02-01 12:00:00",
                10,
                0,
                None,
                0,
                id="null val before_window: no time or event",
            ),
            #
            # Predictions near event (in-window predicition) stay the same
            pytest.param(
                2, 0, 1, "2024-02-01 12:00:00", 12, 0, "2024-02-01 12:00:00", 1, id="pos val in_window: no change"
            ),
            pytest.param(
                2, 1, 0, "2024-02-01 12:00:00", 12, 0, "2024-02-01 12:00:00", 0, id="neg val in_window: no change"
            ),
            pytest.param(
                2, 2, None, "2024-02-01 12:00:00", 12, 0, "2024-02-01 12:00:00", 1, id="null in_window: impute+"
            ),
            #
            # Predictions for positive labels after last event(late prediction) are no time and imputed to negative,
            # but keep time
            pytest.param(
                3,
                0,
                1,
                "2024-01-31 12:00:00",
                12,
                0,
                None,
                0,
                id="pos val late time: NaT+0",
            ),
            pytest.param(
                3,
                1,
                0,
                "2024-01-31 12:00:00",
                12,
                0,
                None,
                0,
                id="neg val late time:  NaT+0",
            ),
            pytest.param(
                3,
                2,
                None,
                "2024-01-31 12:00:00",
                12,
                0,
                None,
                0,
                id="null val late time:  NaT+0",
            ),
            #
            # predictions at edge of window count
            pytest.param(
                4, 0, 1, "2024-02-01 12:00:00", 11, 0, "2024-02-01 12:00:00", 1, id="pos val window start: no change"
            ),
            pytest.param(
                4, 1, 0, "2024-02-01 12:00:00", 11, 0, "2024-02-01 12:00:00", 0, id="neg val window start: no change"
            ),
            pytest.param(
                4,
                2,
                None,
                "2024-02-01 12:00:00",
                11,
                0,
                "2024-02-01 12:00:00",
                1,
                id="null val window start: impute positive",
            ),
            pytest.param(
                4,
                3,
                1,
                "2024-02-01 01:00:00",
                10,
                0,
                "2024-02-01 01:00:00",
                1,
                id="pos val pred=event times: no change",
            ),
            pytest.param(
                4,
                4,
                0,
                "2024-02-01 01:00:00",
                10,
                0,
                "2024-02-01 01:00:00",
                0,
                id="neg val pred=event times: no change",
            ),
            pytest.param(
                4,
                5,
                None,
                "2024-02-01 01:00:00",
                10,
                0,
                "2024-02-01 01:00:00",
                1,
                id="null val pred=event times: impute positve",
            ),
            #
            # offsets
            pytest.param(
                5,
                0,
                1,
                "2024-02-01 12:00:00",
                10,
                2,
                "2024-02-01 12:00:00",
                1,
                id="offset catches earlier case; like 1.0",
            ),
            pytest.param(
                5,
                0,
                1,
                "2024-02-01 01:00:00",
                10,
                2,
                None,
                0,
                id="offset misses near event; like 4.3",
            ),
            pytest.param(
                5, 0, 1, "2024-02-11 12:00:00", None, 0, "2024-02-11 12:00:00", 1, id="no window keep longago event"
            ),
        ],
    )
    def test_merge_with_single_prediction(
        self, id_, ctx, event_val, event_time, window, window_offset, out_time, out_val
    ):
        event_name = "TestEvent"

        events = create_event_table(
            [id_], [ctx], event_name, event_times=[pd.Timestamp(event_time)], event_values=[event_val]
        )
        predictions = create_prediction_table([id_], [ctx], [DEFAULT_PRED_TIME_STR])

        expected = create_pred_event_frame(event_name, [out_val], [out_time])
        actual = undertest.merge_windowed_event(
            predictions,
            "PredictTime",
            events,
            event_name,
            ["Id", "CtxId"],
            min_leadtime_hrs=window_offset,
            window_hrs=window,
        )

        # Assert the output contains the expected columns and values
        pdt.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)

    @pytest.mark.parametrize(
        "id_, ctx, event_vals, event_times, window, window_offset, out_time, out_val",
        [
            # grouped events (both late/early/missing)
            #                  event_vals    event_times window
            pytest.param(
                0,
                0,
                [1, 0],
                ["2024-02-01 12:00:00", "2024-02-01 13:00:00"],
                24,
                0,
                "2024-02-01 12:00:00",
                1,
                id="two later events: closest value",
            ),
            pytest.param(
                0,
                0,
                [0, 1],
                ["2024-02-01 12:00:00", "2024-02-01 13:00:00"],
                24,
                0,
                "2024-02-01 12:00:00",
                0,
                id="two later events: closest",
            ),
            pytest.param(
                0,
                0,
                [1, 1],
                ["2024-02-01 20:00:00", "2024-02-01 16:00:00"],
                24,
                0,
                "2024-02-01 16:00:00",
                1,
                id="two later out of order events: closest by time",
            ),
            pytest.param(
                0,
                0,
                [None, None],
                ["2024-02-01 12:00:00", "2024-02-01 13:00:00"],
                24,
                0,
                "2024-02-01 12:00:00",
                1,
                id="two later events: impute positive",
            ),
            pytest.param(0, 0, [None, None], [None, None], 24, 0, None, 0, id="two without times: impute negative"),
            pytest.param(
                0,
                0,
                [0, 1],
                ["2024-01-31 12:00:00", "2024-01-31 11:00:00"],
                1,
                0,
                None,
                0,
                id="two events before prediction: NaT+0",
            ),
            pytest.param(
                0,
                0,
                [0, 1],
                ["2024-01-31 11:00:00", "2024-01-31 12:00:00"],
                1,
                0,
                None,
                0,
                id="two events before prediction: Nat+0",
            ),
            pytest.param(
                0,
                0,
                [1, 0],
                ["2024-02-01 20:00:00", "2024-02-01 16:00:00"],
                1,
                0,
                None,
                0,
                id="two early out of order events: keep neither",
            ),
            pytest.param(
                0,
                0,
                [1, 0],
                ["2024-02-01 16:00:00", "2024-02-01 20:00:00"],
                1,
                0,
                None,
                0,
                id="two early events: keep neither",
            ),
            # split events
            # -24hr event is before prediction (late score)
            pytest.param(
                0,
                0,
                [1, 0],
                ["2024-02-01 12:00:00", "2024-01-31 12:00:00"],
                12,
                0,
                "2024-02-01 12:00:00",
                1,
                id="late and inwindow score: in window",
            ),
            pytest.param(
                0,
                0,
                [1, 0],
                ["2024-01-31 12:00:00", "2024-02-01 12:00:00"],
                12,
                0,
                "2024-02-01 12:00:00",
                0,
                id="late and inwindow ooo: in window",
            ),
            pytest.param(
                0,
                0,
                [1, 0],
                ["2024-01-31 12:00:00", "2024-02-01 20:00:00"],
                12,
                0,
                None,
                0,
                id="late and early score: NaT+0",
            ),
            pytest.param(
                0,
                0,
                [1, 0],
                ["2024-02-01 20:00:00", "2024-01-31 12:00:00"],
                12,
                0,
                None,
                0,
                id="late and early score ooo: NaT+0",
            ),
            pytest.param(
                0,
                0,
                [1, 0],
                ["2024-01-31 12:00:00", None],
                12,
                0,
                None,
                0,
                id="late and no time: prioritize known time ('late score')",
            ),
            pytest.param(
                0,
                0,
                [1, 0],
                [None, "2024-01-31 12:00:00"],
                12,
                0,
                None,
                0,
                id="late and no time ooo: ",
            ),
            #
            # +8hr event is 19hr after prediction (early score)
            pytest.param(
                0,
                0,
                [1, 0],
                ["2024-02-01 12:00:00", "2024-02-01 20:00:00"],
                12,
                0,
                "2024-02-01 12:00:00",
                1,
                id="inwindow and early score: keeps inwindow",
            ),
            pytest.param(
                0,
                0,
                [1, 0],
                ["2024-02-01 20:00:00", "2024-02-01 12:00:00"],
                12,
                0,
                "2024-02-01 12:00:00",
                0,
                id="inwindow and early score ooo: keeps inwindow",
            ),
            pytest.param(
                0,
                0,
                [1, 0],
                ["2024-02-01 20:00:00", None],
                12,
                0,
                None,
                0,
                id="early score and no time: no time or event ",
            ),
            pytest.param(
                0,
                0,
                [1, 0],
                [None, "2024-02-01 20:00:00"],
                12,
                0,
                None,
                0,
                id="early score and no time ooo: no time or event",
            ),
            #
            # inwindow
            pytest.param(
                0,
                0,
                [1, 0],
                [None, "2024-02-01 12:00:00"],
                12,
                0,
                "2024-02-01 12:00:00",
                0,
                id="inwindow and no time: keep in window",
            ),
            pytest.param(
                0,
                0,
                [1, 0],
                ["2024-02-01 12:00:00", None],
                12,
                0,
                "2024-02-01 12:00:00",
                1,
                id="inwindow and no time ooo: keep in window",
            ),
        ],
    )
    def test_merge_two_events(self, id_, ctx, event_vals, event_times, window, window_offset, out_time, out_val):
        event_name = "TestEvent"
        events = create_event_table(
            [id_], [ctx], event_name, event_times=[pd.Timestamp(time) for time in event_times], event_values=event_vals
        )
        predictions = create_prediction_table([id_], [ctx], [DEFAULT_PRED_TIME_STR])

        expected = create_pred_event_frame(event_name, [out_val], [out_time])
        actual = undertest.merge_windowed_event(
            predictions,
            "PredictTime",
            events,
            event_name,
            ["Id", "CtxId"],
            min_leadtime_hrs=window_offset,
            window_hrs=window,
        )

        # Assert the output contains the expected columns and values
        pdt.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)

    @pytest.mark.parametrize(
        "window, window_offset, expected_dates, expected_vals",
        [
            pytest.param(
                12,
                0,
                [None, "2024-02-02 12:00:00", "2024-02-02 12:00:00", None, None],
                [0.0, 2, 2, 0, 0],
                id="base lookback",
            ),
            pytest.param(
                12,
                2,
                [None, "2024-02-02 12:00:00", None, None, None],
                [0.0, 2, 0, 0, 0],
                id="offset out of middle pred window",
            ),
            pytest.param(
                12,
                -2,
                [None, "2024-02-02 12:00:00", "2024-02-02 12:00:00", "2024-02-02 12:00:00", None],
                [0.0, 2, 2, 2, 0],
                id="offset allow future for third event",
            ),
            pytest.param(
                1.5,
                0,
                [None, None, "2024-02-02 12:00:00", None, None],
                [0.0, 0, 2, 0, 0],
                id="reduce window out of second event",
            ),
        ],
    )
    def test_merge_multi_predictions(self, window, window_offset, expected_dates, expected_vals):
        id_ = 1
        ctx = 1
        event_name = "TestEvent"
        predtimes = [
            "2020-01-01 01:00:00",
            "2024-02-02 10:00:00",
            "2024-02-02 12:00:00",
            "2024-02-02 14:00:00",
            "2030-12-31 23:59:59",
        ]
        event_times = pd.to_datetime(
            [
                "2024-01-31 12:00:00",
                "2024-02-01 12:00:00",
                "2024-02-02 12:00:00",
                "2024-02-03 12:00:00",
                "2024-02-04 12:00:00",
            ]
        )
        expected_times = [pd.Timestamp(d) for d in expected_dates]

        predictions = create_prediction_table([id_], [ctx], predtimes)
        events = create_event_table(
            [id_], [ctx], event_name, event_times=event_times, event_values=range(len(event_times))
        )

        expected = create_pred_event_frame(event_name, expected_vals, expected_times)
        actual = undertest.merge_windowed_event(
            predictions,
            "PredictTime",
            events,
            event_name,
            ["Id", "CtxId"],
            min_leadtime_hrs=window_offset,
            window_hrs=window,
        )

        pdt.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)

    def test_multiple_ids(self):
        # First two multi predictions cases, adjusted to have same offset
        ids = np.repeat([1, 2], 5)
        ctxs = np.repeat([1], 10)
        event_name = "TestEvent"
        predtimes = [
            "2020-01-01 01:00:00",
            "2024-02-02 10:00:00",
            "2024-02-02 12:00:00",
            "2024-02-02 14:00:00",
            "2030-12-31 23:59:59",
            "2020-01-01 01:00:00",
            "2024-02-02 12:00:00",
            "2024-02-02 14:00:00",
            "2024-02-02 16:00:00",
            "2030-12-31 23:59:59",
        ]
        event_times = pd.to_datetime(
            [
                "2024-01-31 12:00:00",
                "2024-02-01 12:00:00",
                "2024-02-02 12:00:00",
                "2024-02-03 12:00:00",
                "2024-02-04 12:00:00",
                "2024-01-31 12:00:00",
                "2024-02-01 12:00:00",
                "2024-02-02 12:00:00",
                "2024-02-03 12:00:00",
                "2024-02-04 12:00:00",
            ]
        )
        expected_times = pd.to_datetime(
            [
                None,
                "2024-02-02 12:00:00",
                "2024-02-02 12:00:00",
                None,
                None,
                None,
                "2024-02-02 12:00:00",
                None,
                None,
                None,
            ]
        )
        expected_vals = np.hstack(([0.0, 2, 2, 0, 0], np.add([-5.0, 2, -5, -5, -5], 5)))  # -5+5=0
        predictions = create_prediction_table(ids, ctxs, predtimes)
        events = create_event_table(
            ids, ctxs, event_name, event_times=event_times, event_values=range(len(event_times))
        )

        expected = create_pred_event_frame(event_name, expected_vals, expected_times)
        actual = undertest.merge_windowed_event(
            predictions, "PredictTime", events, event_name, ["Id", "CtxId"], min_leadtime_hrs=0, window_hrs=12
        )
        actual = actual.sort_values(by=["Id", "CtxId", "PredictTime"]).reset_index(
            drop=True
        )  # sort is for human comparison

        pdt.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)

    def test_multiple_ctxs(self):
        # First two multi predictions cases, adjusted to have same offset
        ids = np.repeat([1], 10)
        ctxs = np.repeat([1, 2], 5)
        event_name = "TestEvent"
        predtimes = [
            "2020-01-01 01:00:00",
            "2024-02-02 10:00:00",
            "2024-02-02 12:00:00",
            "2024-02-02 14:00:00",
            "2030-12-31 23:59:59",
            "2020-01-01 01:00:00",
            "2024-02-02 12:00:00",
            "2024-02-02 14:00:00",
            "2024-02-02 16:00:00",
            "2030-12-31 23:59:59",
        ]
        event_times = pd.to_datetime(
            [
                "2024-01-31 12:00:00",
                "2024-02-01 12:00:00",
                "2024-02-02 12:00:00",
                "2024-02-03 12:00:00",
                "2024-02-04 12:00:00",
                "2024-01-31 12:00:00",
                "2024-02-01 12:00:00",
                "2024-02-02 12:00:00",
                "2024-02-03 12:00:00",
                "2024-02-04 12:00:00",
            ]
        )
        expected_times = pd.to_datetime(
            [
                None,
                "2024-02-02 12:00:00",
                "2024-02-02 12:00:00",
                None,
                None,
                None,
                "2024-02-02 12:00:00",
                None,
                None,
                None,
            ]
        )
        expected_vals = np.hstack(([0.0, 2, 2, 0, 0], np.add([-5.0, 2, -5, -5, -5], 5)))  # -5+5=0
        predictions = create_prediction_table(ids, ctxs, predtimes)
        events = create_event_table(
            ids, ctxs, event_name, event_times=event_times, event_values=range(len(event_times))
        )

        expected = create_pred_event_frame(event_name, expected_vals, expected_times)
        actual = undertest.merge_windowed_event(
            predictions, "PredictTime", events, event_name, ["Id", "CtxId"], min_leadtime_hrs=0, window_hrs=12
        )
        actual = actual.sort_values(by=["Id", "CtxId", "PredictTime"]).reset_index(
            drop=True
        )  # sort is for human comparison

        pdt.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)

    def test_no_shared_keys_errors(self):
        # First two multi predictions cases, adjusted to have same offset
        ids = np.repeat([1], 10)
        ctxs = np.repeat([1, 2], 5)
        event_name = "TestEvent"
        predtimes = [
            "2020-01-01 01:00:00",
            "2024-02-02 10:00:00",
            "2024-02-02 12:00:00",
            "2024-02-02 14:00:00",
            "2030-12-31 23:59:59",
            "2020-01-01 01:00:00",
            "2024-02-02 12:00:00",
            "2024-02-02 14:00:00",
            "2024-02-02 16:00:00",
            "2030-12-31 23:59:59",
        ]
        event_times = pd.to_datetime(
            [
                "2024-01-31 12:00:00",
                "2024-02-01 12:00:00",
                "2024-02-02 12:00:00",
                "2024-02-03 12:00:00",
                "2024-02-04 12:00:00",
                "2024-01-31 12:00:00",
                "2024-02-01 12:00:00",
                "2024-02-02 12:00:00",
                "2024-02-03 12:00:00",
                "2024-02-04 12:00:00",
            ]
        )
        predictions = create_prediction_table(ids, ctxs, predtimes)
        events = create_event_table(
            ids, ctxs, event_name, event_times=event_times, event_values=range(len(event_times))
        )
        events.rename(columns={"Id": "Id_", "CtxId": "CtxId_"}, inplace=True)  # break key match

        with pytest.raises(ValueError, match="No common keys"):
            _ = undertest.merge_windowed_event(
                predictions, "PredictTime", events, event_name, ["Id", "CtxId"], min_leadtime_hrs=0, window_hrs=12
            )

    def test_no_events_adds_nothing(self):
        # First two multi predictions cases, adjusted to have same offset
        ids = np.repeat([1], 10)
        ctxs = np.repeat([1, 2], 5)
        event_name = "TestEvent"
        predtimes = [
            "2020-01-01 01:00:00",
            "2024-02-02 10:00:00",
            "2024-02-02 12:00:00",
            "2024-02-02 14:00:00",
            "2030-12-31 23:59:59",
            "2020-01-01 01:00:00",
            "2024-02-02 12:00:00",
            "2024-02-02 14:00:00",
            "2024-02-02 16:00:00",
            "2030-12-31 23:59:59",
        ]
        predictions = create_prediction_table(ids, ctxs, predtimes)
        events = pd.DataFrame(columns=["Id", "CtxId", "Type", "Time", "Value"])

        actual = undertest.merge_windowed_event(
            predictions, "PredictTime", events, event_name, ["Id", "CtxId"], min_leadtime_hrs=0, window_hrs=12
        )

        pdt.assert_frame_equal(actual, predictions, check_dtype=False)

    def test_ctx_not_key_not_added(self):
        # Use two ctxs
        ids = np.repeat([1], 10)
        ctxs = np.repeat([1, 2], 5)
        event_name = "TestEvent"
        predtimes = [
            "2024-01-01 01:00:00",
            "2024-02-02 10:00:00",
            "2024-02-02 12:00:00",
            "2024-02-02 14:00:00",
            "2024-12-31 23:59:59",
            "2024-01-01 01:00:00",
            "2024-02-02 12:00:00",
            "2024-02-02 14:00:00",
            "2024-02-02 16:00:00",
            "2024-12-31 23:59:59",
        ]
        event_times = pd.to_datetime(
            [
                "2024-01-01 12:00:00",
                pd.NA,
            ]
        )
        predictions = create_prediction_table(ids, ctxs, predtimes)
        events = create_event_table([1, 1], [1, 2], event_name, event_times=event_times, event_values=[10, 20])

        expected_vals = [10, 0, 0, 0, 0, 10, 0, 0, 0, 0]
        expected_times = ["2024-01-01 12:00:00", None, None, None, None, "2024-01-01 12:00:00", None, None, None, None]
        expected = create_pred_event_frame(event_name, expected_vals, expected_times)

        actual = undertest.merge_windowed_event(
            predictions, "PredictTime", events, event_name, ["Id"], min_leadtime_hrs=0, window_hrs=12
        )

        # No CtxId_x
        actual = actual.sort_values(by=["Id", "CtxId", "PredictTime"]).reset_index(drop=True)
        pdt.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)

    def test_some_nat_warn_and_dropped(self, caplog):
        # Use two ctxs
        ids = np.repeat([1], 10)
        ctxs = np.repeat([1, 2], 5)
        event_name = "TestEvent"
        predtimes = [
            "2024-01-01 01:00:00",
            "2024-02-02 10:00:00",
            "2024-02-02 12:00:00",
            "2024-02-02 14:00:00",
            "2024-12-31 23:59:59",
            "2024-01-01 01:00:00",
            "2024-02-02 12:00:00",
            "2024-02-02 14:00:00",
            "2024-02-02 16:00:00",
            "2024-12-31 23:59:59",
        ]
        event_times = pd.to_datetime(
            [
                "2024-01-01 12:00:00",
                pd.NA,
            ]
        )
        predictions = create_prediction_table(ids, ctxs, predtimes)
        events = create_event_table([1, 1], [1, 2], event_name, event_times=event_times, event_values=[10, 20])

        expected_vals = [10, 0, 0, 0, 0, 10, 0, 0, 0, 0]
        expected_times = ["2024-01-01 12:00:00", None, None, None, None, "2024-01-01 12:00:00", None, None, None, None]
        expected = create_pred_event_frame(event_name, expected_vals, expected_times)

        with caplog.at_level(logging.WARNING, logger="seismometer"):
            actual = undertest.merge_windowed_event(
                predictions, "PredictTime", events, event_name, ["Id"], min_leadtime_hrs=0, window_hrs=12
            )

        actual = actual.sort_values(by=["Id", "CtxId", "PredictTime"]).reset_index(drop=True)
        pdt.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)

        assert len(caplog.records) == 1
        assert "Inconsistent" in caplog.text

    def test_no_times_warn_and_merges(self, caplog):
        # Use two ctxs
        ids = np.repeat([1], 10)
        ctxs = np.repeat([1, 2], 5)
        event_name = "TestEvent"
        predtimes = [
            "2024-01-01 01:00:00",
            "2024-02-02 10:00:00",
            "2024-02-02 12:00:00",
            "2024-02-02 14:00:00",
            "2024-12-31 23:59:59",
            "2024-01-01 01:00:00",
            "2024-02-02 12:00:00",
            "2024-02-02 14:00:00",
            "2024-02-02 16:00:00",
            "2024-12-31 23:59:59",
        ]
        event_times = pd.to_datetime(
            [
                pd.NA,
                pd.NA,
            ]
        )
        predictions = create_prediction_table(ids, ctxs, predtimes)
        events = create_event_table([1, 1], [1, 2], event_name, event_times=event_times, event_values=[10, 20])

        expected_vals = [10.0, 10, 10, 10, 10, 20, 20, 20, 20, 20]
        expected_times = [None] * 10
        expected = create_pred_event_frame(event_name, expected_vals, expected_times)

        with caplog.at_level(logging.WARNING, logger="seismometer"):
            actual = undertest.merge_windowed_event(
                predictions, "PredictTime", events, event_name, ["Id", "CtxId"], min_leadtime_hrs=0, window_hrs=12
            )

        actual = actual.sort_values(by=["Id", "CtxId", "PredictTime"]).reset_index(drop=True)
        pdt.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)

        assert len(caplog.records) == 1
        assert "No times" in caplog.text
