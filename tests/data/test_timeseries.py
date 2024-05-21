import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import seismometer.data.timeseries as undertest

ONEDAY = pd.to_timedelta("1D")


def one_entity(id, group, days_offset=0):
    N = 4
    # Generate the dates then add random offset
    dates = pd.date_range(start="2024-01-01", periods=N, freq="D")
    offset = pd.to_timedelta(np.random.randint(-6, 7, size=N), unit="h")
    dates = dates + pd.to_timedelta("12:00:00") + offset + (days_offset * ONEDAY)

    # Create the dataframe
    df = pd.DataFrame(
        {"EntityId": [id] * N, "Reference": dates, "Group": [group] * N, "Value": [i + id * 10 for i in [-1, 0, 1, 2]]}
    )

    return df


def create_frame(ids, groups, days_offset=0):
    return pd.concat([one_entity(ids[ix], groups[ix], days_offset * ix) for ix in range(len(ids))], ignore_index=True)


class TestCreateMetricTimeseries:
    def test_filter_small_group(self):
        # Group C is dropped, others keep *first* value per entity
        input_frame = create_frame(range(1, 7), ["Group A", "Group B", "Group C", "Group B", "Group A", "Group A"])
        expected = pd.DataFrame(
            {
                "Reference": [pd.Timestamp("2024-01-01")] * 5,
                "Group": ["Group A", "Group A", "Group A", "Group B", "Group B"],
                "Value": [9, 49, 59, 19, 39],
            }
        )

        actual = undertest.create_metric_timeseries(
            input_frame, "Reference", "Value", ["EntityId"], "Group", censor_threshold=1
        )

        pdt.assert_frame_equal(actual, expected)

    def test_multi_key_filters_small_group(self):
        # Group C is dropped, others keep *first* value per entity
        key1 = [1, 2, 3, 1, 2, 3]
        key2 = [1, 1, 1, 2, 2, 2]
        input_frame = create_frame(key1, ["Group A", "Group B", "Group C", "Group B", "Group A", "Group A"])
        input_frame["ContextId"] = np.repeat(key2, 4)  # create frame has N=4
        expected = pd.DataFrame(
            {
                "Reference": [pd.Timestamp("2024-01-01")] * 5,
                "Group": ["Group A", "Group A", "Group A", "Group B", "Group B"],
                "Value": [9, 19, 29, 9, 19],
            }
        )

        actual = undertest.create_metric_timeseries(
            input_frame, "Reference", "Value", ["EntityId", "ContextId"], "Group", censor_threshold=1
        )
        compare = actual.sort_values(by=["Group", "Value"]).reset_index(drop=True)

        pdt.assert_frame_equal(compare, expected)

    def test_all_small_returns_empty(self):
        input_frame = create_frame(range(1, 7), ["Group A", "Group B", "Group C", "Group B", "Group A", "Group A"])

        # default censoring at 10
        actual = undertest.create_metric_timeseries(input_frame, "Reference", "Value", ["EntityId"], "Group")
        assert actual.empty

    @pytest.mark.parametrize(
        "start_date, expected_date",
        [
            # Jan1 is a Monday
            ("2024-01-01", "2024-01-01"),
            ("2024-01-02", "2024-01-01"),
            ("2024-01-03", "2024-01-01"),
            ("2024-01-04", "2024-01-01"),
            ("2024-01-05", "2024-01-01"),
            ("2024-01-06", "2024-01-01"),
            ("2024-01-07", "2024-01-01"),
            # Jan8 is the next Monday
            ("2024-01-08", "2024-01-08"),
            ("2024-01-09", "2024-01-08"),
            ("2024-01-10", "2024-01-08"),
        ],
    )
    def test_frequency_aligned_to_week(self, start_date, expected_date):
        # filter to first occurence
        N = 10
        V = 5
        offset = pd.to_timedelta(np.random.randint(0, 1439), unit="min")
        dates = pd.date_range(start=start_date, periods=N, freq="D") + offset
        input_frame = pd.DataFrame(
            {"Reference": dates, "Group": ["A"] * N, "EntityId": [1] * N, "Value": range(V, V + N)}
        )
        expected = pd.DataFrame({"Reference": pd.to_datetime(expected_date), "Group": ["A"], "Value": V})

        actual = undertest.create_metric_timeseries(
            input_frame, "Reference", "Value", ["EntityId"], "Group", censor_threshold=0
        )
        pdt.assert_frame_equal(actual, expected, obj=f"Frame(start={start_date}, offset={offset})")

    def test_data_limited_to_time_bound(self):
        # Last group is excluded by late
        input_frame = create_frame(
            range(1, 7), ["Group A", "Group B", "Group C", "Group A", "Group B", "Group C"], days_offset=1
        )
        bounds = pd.to_datetime(["2024-1-4", "2024-1-6"])
        expected = pd.DataFrame(
            {
                "Reference": [pd.Timestamp("2024-01-01")] * 5,
                "Group": ["Group A", "Group A", "Group B", "Group B", "Group C"],
                "Value": [12, 39, 21, 49, 30],
            }
        )

        actual = undertest.create_metric_timeseries(
            input_frame, "Reference", "Value", ["EntityId"], "Group", censor_threshold=0, time_bounds=bounds
        )

        # For true ignore-order need to sort and then reset index
        compare = actual.sort_values(by=["Reference", "Group", "Value"]).reset_index(drop=True)
        pdt.assert_frame_equal(compare, expected)

    def test_bools_filter_out_invalid(self):
        input_frame = create_frame([0, 1], ["A", "B"])
        input_frame["Value"] = [-1, -2, 1, -1, -1, -2, -1, 0]
        expected = pd.DataFrame({"Reference": [pd.Timestamp("2024-01-01")] * 2, "Group": ["A", "B"], "Value": [1, 0]})

        actual = undertest.create_metric_timeseries(
            input_frame, "Reference", "Value", ["EntityId"], "Group", censor_threshold=0, boolean_event=True
        )

        pdt.assert_frame_equal(actual, expected)

    def test_missing_does_not_count_toward_threshold(self):
        input_frame = create_frame([0, 1], ["A", "A"])
        input_frame["Value"] = [-1, -2, -1, -1, -1, -2, -1, 0]

        actual = undertest.create_metric_timeseries(
            input_frame, "Reference", "Value", ["EntityId"], "Group", censor_threshold=1, boolean_event=True
        )

        assert actual.empty
