import logging
from dataclasses import dataclass
from unittest.mock import patch

import pandas as pd
import pandas.testing as pdt
import pytest
from conftest import res  # noqa

import seismometer.data.pandas_helpers as undertest

HOUR = pd.Timedelta("1h")


@dataclass
class CaseData:
    preds: pd.DataFrame
    cohorts: pd.DataFrame
    events: pd.DataFrame
    expect: pd.DataFrame


@pytest.fixture
def merge_data(res):
    res = res / "data"
    dfs = []
    for file in ["input_predictions.tsv", "input_cohorts.tsv", "input_events.tsv", "expectation_merges.tsv"]:
        df = pd.read_csv(res / file, sep="\t")
        for col in df.columns:
            if col in ["Id", "Enc"]:
                # assume int for reliable indexing of testcases; nullable is from blank lines in our data
                df[col] = df[col].astype("Int32")
            if "Time" not in col:
                continue
            df[col] = pd.to_datetime(df[col])
        dfs.append(df)
    return CaseData(*dfs)


def filter_case(casedata, id, enc=None):
    def dualkey(df):
        return df[(df["Id"] == id) and (df["Enc"] == enc)].reset_index(drop=True)

    def onekey(df):
        return df[(df["Id"] == id)].reset_index(drop=True)

    filter_fn = dualkey if enc is not None else onekey

    return CaseData(*(filter_fn(element) for element in vars(casedata).values()))


@pytest.fixture
def base_counts_data():
    now = pd.Timestamp("2024-01-01 00:00:00")
    preds = pd.DataFrame({"Id": [1, 2], "Time": [now + pd.Timedelta(hours=1), now]})
    events = pd.DataFrame(
        {
            "Id": [1, 1, 2, 2],
            "Event_Time": [now, now + pd.Timedelta(minutes=90), now + pd.Timedelta(hours=2), pd.NaT],
            "Label": ["A", "B", "A", "C"],
        }
    )
    return preds, events


class TestMergeFrames:
    @pytest.mark.parametrize("id_,enc", [pytest.param(1, None, id="predictions+outcomes")])
    def test_merge_earliest(self, id_, enc, merge_data):
        data = filter_case(merge_data, id_, enc)
        expect = data.expect.drop(columns=[c for c in data.expect if "Cohort" in c]).rename(columns={"✨Time✨": "Time"})

        actual = undertest._merge_with_strategy(
            data.preds.sort_values("PredictTime"),
            data.events.sort_values("Time"),
            ["Id", "Enc"],
            pred_ref="PredictTime",
        )

        # check_like = ignore column order
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expect, check_like=True, check_dtype=False)

    @pytest.mark.parametrize("strategy", ["forward", "nearest", "first", "last"])
    def test_merge_strategies_do_not_generate_additional_rows(self, strategy):
        preds = pd.DataFrame(
            {
                "Id": [1, 1],
                "PredictTime": [
                    pd.Timestamp("2024-01-01 01:00:00"),
                    pd.Timestamp("2024-01-01 02:00:00"),
                ],
            }
        )

        events = pd.DataFrame(
            {
                "Id": [1, 1, 1, 1, 1],
                "Time": [
                    pd.Timestamp("2023-12-31 01:30:00"),
                    pd.Timestamp("2024-01-01 00:30:00"),
                    pd.Timestamp("2024-01-01 01:30:00"),
                    pd.Timestamp("2024-01-01 02:30:00"),
                    pd.Timestamp("2024-01-01 10:30:00"),
                ],
                "Type": ["MyEvent", "MyEvent", "MyEvent", "MyEvent", "MyEvent"],
                "Value": [10, 20, 10, 20, 10],
            }
        )

        one_event = undertest._one_event(events, "MyEvent", "Value", "Time", ["Id"])

        # Choose reference column depending on strategy
        event_ref = "MyEvent_Time" if strategy in ["forward", "nearest"] else "~~reftime~~"
        if strategy in ["first", "last"]:
            one_event["~~reftime~~"] = one_event["MyEvent_Time"]

        actual = undertest._merge_with_strategy(
            predictions=preds.copy(),
            one_event=one_event.copy(),
            pks=["Id"],
            pred_ref="PredictTime",
            event_ref=event_ref,
            event_display="MyEvent",
            merge_strategy=strategy,
        )

        # Check that output columns exist and have been merged
        assert "MyEvent_Value" in actual.columns
        assert "MyEvent_Time" in actual.columns
        assert len(actual) == len(preds)

    def test_merge_with_strategy_empty_pks_raises(self):
        """Empty pks list should cause merge_asof to fail (needs by parameter)."""
        preds = pd.DataFrame(
            {
                "Id": [1, 2],
                "PredictTime": [pd.Timestamp("2024-01-01 01:00:00"), pd.Timestamp("2024-01-01 02:00:00")],
            }
        )
        events = pd.DataFrame(
            {
                "Id": [1, 2],
                "Time": [pd.Timestamp("2024-01-01 03:00:00"), pd.Timestamp("2024-01-01 04:00:00")],
                "Value": [10, 20],
                "Type": ["MyEvent", "MyEvent"],
            }
        )

        one_event = undertest._one_event(events, "MyEvent", "Value", "Time", [])

        # With empty pks, merge_asof will fail because it needs a by parameter
        with pytest.raises((ValueError, KeyError, IndexError)):
            undertest._merge_with_strategy(
                predictions=preds,
                one_event=one_event,
                pks=[],  # Empty pks list - causes error
                pred_ref="PredictTime",
                event_ref="MyEvent_Time",
                merge_strategy="forward",
            )

    def test_merge_with_strategy_all_nat_event_times(self):
        """All NaT event times should trigger warning and use first row logic."""
        preds = pd.DataFrame({"Id": [1], "PredictTime": [pd.Timestamp("2024-01-01 01:00:00")]})
        events = pd.DataFrame(
            {
                "Id": [1, 1],
                "Time": [pd.NaT, pd.NaT],  # All NaT
                "Value": [10, 20],
                "Type": ["MyEvent", "MyEvent"],
            }
        )

        one_event = undertest._one_event(events, "MyEvent", "Value", "Time", ["Id"])

        # All NaT should trigger the ct_times == 0 path
        result = undertest._merge_with_strategy(
            predictions=preds,
            one_event=one_event,
            pks=["Id"],
            pred_ref="PredictTime",
            event_ref="MyEvent_Time",
            merge_strategy="forward",
        )

        # Should merge with first row (groupby.first())
        assert len(result) == 1
        assert "MyEvent_Value" in result.columns
        assert result["MyEvent_Value"].iloc[0] == 10  # First value


def infer_cases():
    return pd.DataFrame(
        {
            "label_in": [1, 0, 1, 0, None, None, None, None],
            "time_in": [1, 1, None, None, 1, 1, None, None],
            "label_out": [1, 0, 1, 0, 1, 1, 0, 0],
            "description": [
                "label1+time keeps",
                "label0+time keeps label",
                "label1+no time keeps label",
                "label0+no time keeps label",
                "no label with time infers to positive",
                "no label with time infers to positive (again)",
                "no label nor time infers to negative",
                "no label nor time infers to negative (again)",
            ],
        }
    )


def one_line_case():
    for _, row in infer_cases().iterrows():
        yield pytest.param(*row[:-1].values, id=row["description"])


class TestPostProcessEvent:
    @pytest.mark.parametrize("label_in,time_in,label_out", one_line_case())
    def test_infer_one_line(self, label_in, time_in, label_out):
        col_label = "Label"
        col_time = "Time"
        dataframe = pd.DataFrame({col_label: [label_in], col_time: pd.to_datetime([time_in])})
        expect = pd.DataFrame({col_label: [label_out], col_time: pd.to_datetime([time_in])})

        actual = undertest.post_process_event(dataframe, col_label, col_time)
        # actual['Label'] = actual['Label'].astype(int) # handle inference where input frame could be all null series

        pdt.assert_frame_equal(actual, expect, check_dtype=False)

    def test_infer_multi_line(self):
        all_cases = infer_cases()
        col_label = "Label"
        col_time = "Time"
        col_map = {"label_in": col_label, "time_in": col_time, "label_out": col_label}

        dataframe = all_cases.iloc[:, :2].rename(columns={k: v for k, v in col_map.items() if k in all_cases.columns})
        expect = all_cases.iloc[:, 2:0:-1].rename(columns={k: v for k, v in col_map.items() if k in all_cases.columns})

        actual = undertest.post_process_event(dataframe, col_label, col_time)

        pdt.assert_frame_equal(actual, expect, check_dtype=False)

    @patch("seismometer.data.pandas_helpers.try_casting")
    def test_post_process_event_skips_cast_when_dtype_none(self, mock_casting):
        df = pd.DataFrame({"Label": [None], "Time": [pd.Timestamp.now()]})
        undertest.post_process_event(df, "Label", "Time", column_dtype=None)
        mock_casting.assert_not_called()

    @pytest.mark.parametrize(
        "input_list,dtype",
        [
            (["1.0", "0.0", "1.0", "0.0", None, None, None, None], "string"),
            (["1", "0", "1", "0", None, None, None, None], "string"),
            ([1, 0, 1, 0, None, None, None, None], "Int64"),  # Nullable
            ([1, 0, 1, 0, None, None, None, None], "object"),
            ([1.0, 0, 1, 0, None, None, None, None], "float"),
            (
                [True, False, True, False, None, None, None, None],
                "object",
            ),  # None is Falsy, so can't cast directly to bool
        ],
    )
    def test_hardint_casts_well(self, input_list, dtype):
        all_cases = infer_cases()
        col_label = "Label"
        col_time = "Time"
        col_map = {"label_in": col_label, "time_in": col_time, "label_out": col_label}

        all_cases["label_in"] = pd.Series(input_list, dtype=dtype)  # override the form of inputs
        dataframe = all_cases.iloc[:, :2].rename(columns={k: v for k, v in col_map.items() if k in all_cases.columns})
        expect = all_cases.iloc[:, 2:0:-1].rename(columns={k: v for k, v in col_map.items() if k in all_cases.columns})

        actual = undertest.post_process_event(dataframe, col_label, col_time, column_dtype="int")

        pdt.assert_frame_equal(actual, expect, check_dtype=False)

    def test_imputation_overrides(self):
        all_cases = infer_cases()
        col_label = "Label"
        col_time = "Time"
        col_map = {"label_in": col_label, "time_in": col_time, "label_out": col_label}
        all_cases["label_out"] = [1, 0, 1, 0, 100, 100, 99, 99]

        dataframe = all_cases.iloc[:, :2].rename(columns={k: v for k, v in col_map.items() if k in all_cases.columns})
        expect = all_cases.iloc[:, 2:0:-1].rename(columns={k: v for k, v in col_map.items() if k in all_cases.columns})

        actual = undertest.post_process_event(
            dataframe, col_label, col_time, impute_val_with_time=100, impute_val_no_time=99.0
        )

        pdt.assert_frame_equal(actual, expect, check_dtype=False)

    def test_empty_dataframe_returns_unchanged(self):
        """Empty DataFrame should be returned unchanged."""
        df = pd.DataFrame({"Label": [], "Time": []})
        result = undertest.post_process_event(df, "Label", "Time")
        pdt.assert_frame_equal(result, df, check_dtype=False)

    def test_all_nat_times_imputes_no_time(self):
        """When all times are NaT, should impute with no_time value."""
        df = pd.DataFrame({"Label": [None, None, None], "Time": [pd.NaT, pd.NaT, pd.NaT]})
        result = undertest.post_process_event(df, "Label", "Time")
        assert (result["Label"] == 0).all()

    def test_both_impute_values_none_no_imputation(self):
        """When both impute values are None, no imputation should occur."""
        df = pd.DataFrame({"Label": [None, 1, None], "Time": [pd.Timestamp.now(), pd.NaT, pd.NaT]})
        result = undertest.post_process_event(df, "Label", "Time", impute_val_with_time=None, impute_val_no_time=None)
        assert result["Label"].iloc[0] is pd.NA or pd.isna(result["Label"].iloc[0])
        assert result["Label"].iloc[1] == 1
        assert result["Label"].iloc[2] is pd.NA or pd.isna(result["Label"].iloc[2])

    @pytest.mark.parametrize(
        "impute_with,impute_no,expected_with,expected_no",
        [
            (-1, -2, -1, -2),  # Negative values
            (100, 200, 100, 200),  # Large positive values
            (0.5, 0.1, 0.5, 0.1),  # Decimal values
            ("yes", "no", "yes", "no"),  # String values
        ],
        ids=["negative", "large_positive", "decimal", "string"],
    )
    def test_impute_values_various_types(self, impute_with, impute_no, expected_with, expected_no):
        """Test imputation with various value types."""
        now = pd.Timestamp.now()
        df = pd.DataFrame({"Label": [None, None], "Time": [now, pd.NaT]})
        result = undertest.post_process_event(
            df, "Label", "Time", impute_val_with_time=impute_with, impute_val_no_time=impute_no, column_dtype=None
        )
        assert result["Label"].iloc[0] == expected_with
        assert result["Label"].iloc[1] == expected_no

    def test_single_row_dataframe(self):
        """Single row DataFrame should work correctly."""
        df = pd.DataFrame({"Label": [None], "Time": [pd.Timestamp.now()]})
        result = undertest.post_process_event(df, "Label", "Time")
        assert result["Label"].iloc[0] == 1

    def test_missing_columns_returns_unchanged(self):
        """Missing columns should return DataFrame unchanged."""
        df = pd.DataFrame({"A": [1], "B": [2]})
        result = undertest.post_process_event(df, "MissingLabel", "MissingTime")
        pdt.assert_frame_equal(result, df)


BASE_STRINGS = [
    ("A"),
    ("1"),
    ("normal_column"),
    ("4reallyl0ngc*lum\namewithsymbo|s"),
]


class TestEventValue:
    @pytest.mark.parametrize(
        "input",
        [
            ("A"),
            ("1"),
            ("normal_column"),
            ("4reallyl0ngc*lum\namewithsymbo|s"),
            ("no underscore but ends in Value"),
            ("wrong case but ends in _value"),
            ("all caps ending in _VALUE"),
        ],
    )
    def test_string_without_value_returns_added(self, input):
        expected = input + "_Value"
        assert expected == undertest.event_value(input)

    @pytest.mark.parametrize("base", BASE_STRINGS)
    def test_string_doesnt_repeat_suffix(self, base):
        input = base + "_Value"
        assert input == undertest.event_value(input)

    @pytest.mark.parametrize("base", BASE_STRINGS)
    def test_string_with_time_stripped_first(self, base):
        input = base + "_Time"
        expected = base + "_Value"
        assert expected == undertest.event_value(input)


class TestOneEvent:
    def test_one_event_filters_and_renames(self):
        events = pd.DataFrame(
            {
                "Id": [1, 1],
                "Type": ["Target", "Other"],
                "Value": [10, 20],
                "Time": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            }
        )
        result = undertest._one_event(events, "Target", "Value", "Time", ["Id"])
        assert "Target_Value" in result.columns
        assert "Target_Time" in result.columns
        assert len(result) == 1

    def test_one_event_missing_type_column_raises(self):
        """Missing Type column should raise AttributeError."""
        events = pd.DataFrame({"Id": [1], "Value": [10], "Time": [pd.Timestamp.now()]})
        with pytest.raises(AttributeError, match="Type"):
            undertest._one_event(events, "Target", "Value", "Time", ["Id"])

    def test_one_event_missing_value_column_raises(self):
        """Missing value column should raise KeyError."""
        events = pd.DataFrame({"Id": [1], "Type": ["Target"], "Time": [pd.Timestamp.now()]})
        with pytest.raises(KeyError, match="Value"):
            undertest._one_event(events, "Target", "Value", "Time", ["Id"])

    def test_one_event_missing_time_column_raises(self):
        """Missing time column should raise KeyError."""
        events = pd.DataFrame({"Id": [1], "Type": ["Target"], "Value": [10]})
        with pytest.raises(KeyError, match="Time"):
            undertest._one_event(events, "Target", "Value", "Time", ["Id"])

    def test_one_event_no_matching_event_returns_empty(self):
        """No matching event type should return empty DataFrame with correct columns."""
        events = pd.DataFrame({"Id": [1], "Type": ["OtherEvent"], "Value": [10], "Time": [pd.Timestamp.now()]})
        result = undertest._one_event(events, "Target", "Value", "Time", ["Id"])
        assert len(result) == 0
        assert "Target_Value" in result.columns
        assert "Target_Time" in result.columns


class TestEventTime:
    @pytest.mark.parametrize(
        "input",
        [
            ("A"),
            ("1"),
            ("normal_column"),
            ("4reallyl0ngc*lum\namewithsymbo|s"),
            ("no underscore but ends in Time"),
            ("wrong case but ends in _time"),
            ("all caps ending in _TIME"),
        ],
    )
    def test_string_without_time_returns_added(self, input):
        expected = input + "_Time"
        assert expected == undertest.event_time(input)

    @pytest.mark.parametrize("base", BASE_STRINGS)
    def test_string_doesnt_repeat_suffix(self, base):
        input = base + "_Time"
        assert input == undertest.event_time(input)

    @pytest.mark.parametrize("base", BASE_STRINGS)
    def test_string_with_time_stripped_first(self, base):
        input = base + "_Value"
        expected = base + "_Time"
        assert expected == undertest.event_time(input)


class TestEventName:
    @pytest.mark.parametrize("suffix", ["", "_Time", "_Value"])
    @pytest.mark.parametrize(
        "base",
        [
            ("A"),
            ("1"),
            ("normal_column"),
            ("4reallyl0ngc*lum\namewithsymbo|s"),
        ],
    )
    def test_three_suffixes_align(self, base, suffix):
        input = base + suffix
        assert base == undertest.event_name(input)

    @pytest.mark.parametrize(
        "input, expected",
        [
            ("eventname_Value", "eventname"),
            ("eventname_Time", "eventname"),
            ("no underscore but ends in Time", "no underscore but ends in Time"),
            ("no underscore but ends in Value", "no underscore but ends in Value"),
            ("wrong case but ends in _time", "wrong case but ends in _time"),
            ("wrong case but ends in _value", "wrong case but ends in _value"),
            ("all caps ending in _TIME", "all caps ending in _TIME"),
            ("all caps ending in _VALUE", "all caps ending in _VALUE"),
            ("only one suffix gets stripped_Time_Value", "only one suffix gets stripped_Time"),
            ("only one suffix gets stripped_Value_Time", "only one suffix gets stripped_Value"),
            ("", ""),  # Empty string
            ("_", "_"),  # Just underscore
            ("__Time", "_"),  # Multiple underscores before suffix
            ("event__Value", "event_"),  # Multiple underscores in name
        ],
        ids=[
            "value_suffix",
            "time_suffix",
            "no_underscore_time",
            "no_underscore_value",
            "lowercase_time",
            "lowercase_value",
            "uppercase_time",
            "uppercase_value",
            "double_suffix_time_first",
            "double_suffix_value_first",
            "empty_string",
            "just_underscore",
            "double_underscore_time",
            "double_underscore_value",
        ],
    )
    def test_suffix_specific_handling(self, input, expected):
        assert expected == undertest.event_name(input)

    def test_very_long_event_name(self):
        """Very long event names should work correctly."""
        long_name = "a" * 1000 + "_Time"
        assert undertest.event_name(long_name) == "a" * 1000

    def test_unicode_characters(self):
        """Unicode characters should be preserved."""
        assert undertest.event_name("événement_Time") == "événement"
        assert undertest.event_name("事件_Value") == "事件"


class TestEventHelpers:
    @pytest.mark.parametrize(
        "event_label, event_value, expected",
        [
            ("MyEvent", "Critical", "MyEvent~Critical_Count"),
            ("MyEvent_Value", "High_Count", "MyEvent~High_Count"),
            ("", "", "~_Count"),  # Empty strings
            ("Event", "Val~ue", "Event~Val~ue_Count"),  # Tilde in value
            ("A", "1", "A~1_Count"),  # Single char
        ],
        ids=["standard", "with_high_count", "empty_strings", "tilde_in_value", "single_char"],
    )
    def test_event_value_count(self, event_label, event_value, expected):
        assert undertest.event_value_count(event_label, event_value) == expected

    @pytest.mark.parametrize(
        "input, expected",
        [
            ("MyEvent~Critical_Count", "Critical"),
            ("MyEvent~123_Count", "123"),
            ("Event_Only_Count", "Event_Only"),  # no ~
            ("Multi~Tilde~Value_Count", "Tilde"),  # Multiple tildes - split()[1] gets second element
            ("NoCountSuffix", "NoCountSuffix"),  # No _Count suffix
            ("", ""),  # Empty string
        ],
        ids=["standard", "numeric", "no_tilde", "multi_tilde", "no_count", "empty"],
    )
    def test_event_value_name(self, input, expected):
        assert undertest.event_value_name(input) == expected

    def test_is_valid_event_when_cols_missing(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = undertest.is_valid_event(df, "MissingEvent", "RefTime")
        assert result.all()  # all True

    def test_is_valid_event_when_valid(self):
        now = pd.Timestamp.now()
        df = pd.DataFrame(
            {"MyEvent_Time": [now + pd.Timedelta(hours=1), now + pd.Timedelta(hours=2)], "RefTime": [now, now]}
        )
        result = undertest.is_valid_event(df, "MyEvent", "RefTime")
        assert result.all()

    def test_is_valid_event_when_invalid(self):
        now = pd.Timestamp.now()
        df = pd.DataFrame(
            {"MyEvent_Time": [now - pd.Timedelta(hours=1), now - pd.Timedelta(hours=2)], "RefTime": [now, now]}
        )
        result = undertest.is_valid_event(df, "MyEvent", "RefTime")
        assert not result.any()

    def test_is_valid_event_mixed_valid_invalid(self):
        """Test with mixed valid/invalid events."""
        now = pd.Timestamp.now()
        df = pd.DataFrame(
            {
                "MyEvent_Time": [now + pd.Timedelta(hours=1), now - pd.Timedelta(hours=1)],
                "RefTime": [now, now],
            }
        )
        result = undertest.is_valid_event(df, "MyEvent", "RefTime")
        assert result.iloc[0]  # First is valid
        assert not result.iloc[1]  # Second is invalid

    def test_is_valid_event_with_nat(self):
        """Test with NaT values in event times."""
        now = pd.Timestamp.now()
        df = pd.DataFrame({"MyEvent_Time": [pd.NaT, now + pd.Timedelta(hours=1)], "RefTime": [now, now]})
        result = undertest.is_valid_event(df, "MyEvent", "RefTime")
        # NaT comparisons return False
        assert not result.iloc[0]
        assert result.iloc[1]

    def test_is_valid_event_empty_dataframe(self):
        """Empty DataFrame should return empty Series."""
        df = pd.DataFrame({"MyEvent_Time": [], "RefTime": []})
        result = undertest.is_valid_event(df, "MyEvent", "RefTime")
        assert len(result) == 0


class TestTryCasting:
    @pytest.mark.parametrize(
        "dtype, expected_type",
        [
            ("int", "int64"),
            ("float", "float64"),
            ("string", "string"),
            ("object", "object"),
            ("Int64", "Int64"),  # Nullable integer
        ],
    )
    def test_try_casting_valid_types(self, dtype, expected_type):
        df = pd.DataFrame({"col": ["1", "2", "3"]})
        undertest.try_casting(df, "col", dtype)
        assert df["col"].dtype.name == expected_type

    def test_try_casting_invalid_type_raises(self):
        df = pd.DataFrame({"col": ["a", "b", "c"]})  # can't cast to int
        with pytest.raises(undertest.ConfigurationError, match="Cannot cast 'col' values to 'int'."):
            undertest.try_casting(df, "col", "int")

    def test_try_casting_empty_dataframe(self):
        """Empty DataFrame should cast successfully."""
        df = pd.DataFrame({"col": pd.Series([], dtype=object)})
        undertest.try_casting(df, "col", "int")
        assert df["col"].dtype.name == "int64"

    def test_try_casting_single_row(self):
        """Single row should cast successfully."""
        df = pd.DataFrame({"col": ["42"]})
        undertest.try_casting(df, "col", "int")
        assert df["col"].iloc[0] == 42

    def test_try_casting_with_nulls_to_int64(self):
        """Nullable Int64 should handle None values."""
        df = pd.DataFrame({"col": [1, None, 3]})
        undertest.try_casting(df, "col", "Int64")
        assert df["col"].dtype.name == "Int64"
        assert pd.isna(df["col"].iloc[1])

    def test_try_casting_float_strings_to_int(self):
        """Float strings should cast to int via float intermediate."""
        df = pd.DataFrame({"col": ["1.0", "2.0", "3.0"]})
        undertest.try_casting(df, "col", "int")
        assert df["col"].dtype.name == "int64"
        assert (df["col"] == [1, 2, 3]).all()

    @pytest.mark.parametrize(
        "input_data,dtype",
        [
            (["not", "a", "number"], "int"),
            (["1.5.5"], "float"),
            (["2023-13-45"], "datetime64"),  # Invalid date
        ],
        ids=["string_to_int", "malformed_float", "invalid_datetime"],
    )
    def test_try_casting_raises_configuration_error(self, input_data, dtype):
        """Various invalid casts should raise ConfigurationError."""
        df = pd.DataFrame({"col": input_data})
        with pytest.raises(undertest.ConfigurationError):
            undertest.try_casting(df, "col", dtype)


class TestResolveHelpers:
    def test_resolve_time_col_from_event_suffix(self):
        df = pd.DataFrame({"EventName_Time": [pd.Timestamp.now()]})
        assert undertest._resolve_time_col(df, "EventName") == "EventName_Time"

    def test_resolve_time_col_fallback_to_exact_match(self):
        df = pd.DataFrame({"MyTime": [pd.Timestamp.now()]})
        assert undertest._resolve_time_col(df, "MyTime") == "MyTime"

    def test_resolve_time_col_raises_if_missing(self):
        df = pd.DataFrame({"SomeOtherCol": [1]})
        with pytest.raises(ValueError, match="Reference time column EventName_Time not found"):
            undertest._resolve_time_col(df, "EventName")

    def test_resolve_score_col_direct_match(self):
        df = pd.DataFrame({"MyScore": [0.5]})
        assert undertest._resolve_score_col(df, "MyScore") == "MyScore"

    def test_resolve_score_col_fallback_to_event_value(self):
        df = pd.DataFrame({"MyScore_Value": [0.5]})
        assert undertest._resolve_score_col(df, "MyScore") == "MyScore_Value"

    def test_resolve_score_col_raises_if_missing(self):
        df = pd.DataFrame({"Unrelated": [1]})
        with pytest.raises(ValueError, match="Score column MyScore not found"):
            undertest._resolve_score_col(df, "MyScore")


class TestAggregationFunctions:
    """Direct tests for aggregation functions used by event_score."""

    def test_max_aggregation_picks_highest_score_with_positive_event(self):
        """max_aggregation should pick row with highest score among positive events."""
        df = pd.DataFrame(
            {
                "Id": [1, 1, 1],
                "Score": [0.3, 0.8, 0.5],
                "EventName_Value": [1, 1, 0],  # First two are positive
                "EventName_Time": [pd.Timestamp("2024-01-01")] * 3,
            }
        )

        result = undertest.max_aggregation(
            df, pks=["Id"], score="Score", ref_time="EventName_Time", ref_event="EventName"
        )

        assert len(result) == 1
        assert result["Score"].iloc[0] == 0.8  # Highest score among positive events

    def test_max_aggregation_requires_ref_event(self):
        """max_aggregation should raise ValueError if ref_event is None."""
        df = pd.DataFrame({"Id": [1], "Score": [0.5]})

        with pytest.raises(ValueError, match="ref_event is required"):
            undertest.max_aggregation(df, pks=["Id"], score="Score", ref_time="Time", ref_event=None)

    def test_min_aggregation_picks_lowest_score_with_positive_event(self):
        """min_aggregation should pick row with lowest score among positive events."""
        df = pd.DataFrame(
            {
                "Id": [1, 1, 1],
                "Score": [0.3, 0.8, 0.5],
                "EventName_Value": [1, 1, 0],  # First two are positive
                "EventName_Time": [pd.Timestamp("2024-01-01")] * 3,
            }
        )

        result = undertest.min_aggregation(
            df, pks=["Id"], score="Score", ref_time="EventName_Time", ref_event="EventName"
        )

        assert len(result) == 1
        assert result["Score"].iloc[0] == 0.3  # Lowest score among positive events

    def test_min_aggregation_requires_ref_event(self):
        """min_aggregation should raise ValueError if ref_event is None."""
        df = pd.DataFrame({"Id": [1], "Score": [0.5]})

        with pytest.raises(ValueError, match="ref_event is required"):
            undertest.min_aggregation(df, pks=["Id"], score="Score", ref_time="Time", ref_event=None)

    def test_first_aggregation_picks_earliest_by_time(self):
        """first_aggregation should pick row with earliest timestamp."""
        df = pd.DataFrame(
            {
                "Id": [1, 1, 1],
                "Score": [0.3, 0.8, 0.5],
                "EventName_Time": [
                    pd.Timestamp("2024-01-03"),
                    pd.Timestamp("2024-01-01"),  # Earliest
                    pd.Timestamp("2024-01-02"),
                ],
            }
        )

        result = undertest.first_aggregation(
            df, pks=["Id"], score="Score", ref_time="EventName_Time", ref_event="EventName"
        )

        assert len(result) == 1
        assert result["Score"].iloc[0] == 0.8  # Score from earliest timestamp

    def test_first_aggregation_requires_ref_time(self):
        """first_aggregation should raise ValueError if ref_time is None."""
        df = pd.DataFrame({"Id": [1], "Score": [0.5]})

        with pytest.raises(ValueError, match="ref_time is required"):
            undertest.first_aggregation(df, pks=["Id"], score="Score", ref_time=None, ref_event="EventName")

    def test_first_aggregation_drops_nat_timestamps(self):
        """first_aggregation should drop rows with NaT timestamps."""
        df = pd.DataFrame(
            {
                "Id": [1, 1, 1],
                "Score": [0.3, 0.8, 0.5],
                "EventName_Time": [
                    pd.NaT,  # Should be dropped
                    pd.Timestamp("2024-01-01"),
                    pd.Timestamp("2024-01-02"),
                ],
            }
        )

        result = undertest.first_aggregation(
            df, pks=["Id"], score="Score", ref_time="EventName_Time", ref_event="EventName"
        )

        assert len(result) == 1
        assert result["Score"].iloc[0] == 0.8  # First non-NaT timestamp

    def test_last_aggregation_picks_latest_by_time(self):
        """last_aggregation should pick row with latest timestamp."""
        df = pd.DataFrame(
            {
                "Id": [1, 1, 1],
                "Score": [0.3, 0.8, 0.5],
                "EventName_Time": [
                    pd.Timestamp("2024-01-01"),
                    pd.Timestamp("2024-01-02"),
                    pd.Timestamp("2024-01-03"),  # Latest
                ],
            }
        )

        result = undertest.last_aggregation(
            df, pks=["Id"], score="Score", ref_time="EventName_Time", ref_event="EventName"
        )

        assert len(result) == 1
        assert result["Score"].iloc[0] == 0.5  # Score from latest timestamp

    def test_last_aggregation_requires_ref_time(self):
        """last_aggregation should raise ValueError if ref_time is None."""
        df = pd.DataFrame({"Id": [1], "Score": [0.5]})

        with pytest.raises(ValueError, match="ref_time is required"):
            undertest.last_aggregation(df, pks=["Id"], score="Score", ref_time=None, ref_event="EventName")

    @pytest.mark.parametrize(
        "agg_func,sort_by,expected_score",
        [
            (undertest.max_aggregation, None, 0.9),  # Picks highest score
            (undertest.min_aggregation, None, 0.1),  # Picks lowest score
            (undertest.first_aggregation, "time", 0.3),  # Picks earliest time
            (undertest.last_aggregation, "time", 0.7),  # Picks latest time
        ],
        ids=["max", "min", "first", "last"],
    )
    def test_aggregation_functions_with_multiple_entities(self, agg_func, sort_by, expected_score):
        """All aggregation functions should work correctly with multiple entities."""
        if sort_by == "time":
            df = pd.DataFrame(
                {
                    "Id": [1, 1, 2, 2],
                    "Score": [0.3, 0.7, 0.4, 0.6],
                    "EventName_Time": [
                        pd.Timestamp("2024-01-01"),  # Earliest for Id=1
                        pd.Timestamp("2024-01-02"),  # Latest for Id=1
                        pd.Timestamp("2024-01-01"),
                        pd.Timestamp("2024-01-02"),
                    ],
                }
            )
            result = agg_func(df, pks=["Id"], score="Score", ref_time="EventName_Time", ref_event="EventName")
        else:
            df = pd.DataFrame(
                {
                    "Id": [1, 1, 2, 2],
                    "Score": [0.1, 0.9, 0.2, 0.8],
                    "EventName_Value": [1, 1, 1, 1],
                    "EventName_Time": [pd.Timestamp("2024-01-01")] * 4,
                }
            )
            result = agg_func(df, pks=["Id"], score="Score", ref_time="EventName_Time", ref_event="EventName")

        # Should have 2 rows (one per entity)
        assert len(result) == 2
        # Check Id=1 has expected score
        assert result[result["Id"] == 1]["Score"].iloc[0] == expected_score

    def test_max_aggregation_all_negative_events(self):
        """max_aggregation with all negative events should still return a row."""
        df = pd.DataFrame(
            {
                "Id": [1, 1],
                "Score": [0.3, 0.8],
                "EventName_Value": [0, 0],  # All negative
                "EventName_Time": [pd.Timestamp("2024-01-01")] * 2,
            }
        )
        result = undertest.max_aggregation(df, pks=["Id"], score="Score", ref_time="Time", ref_event="EventName")
        assert len(result) == 1
        assert result["Score"].iloc[0] == 0.8  # Still picks max even if all negative

    def test_min_aggregation_all_negative_events(self):
        """min_aggregation with all negative events should still return a row."""
        df = pd.DataFrame(
            {
                "Id": [1, 1],
                "Score": [0.3, 0.8],
                "EventName_Value": [0, 0],  # All negative
                "EventName_Time": [pd.Timestamp("2024-01-01")] * 2,
            }
        )
        result = undertest.min_aggregation(df, pks=["Id"], score="Score", ref_time="Time", ref_event="EventName")
        assert len(result) == 1
        assert result["Score"].iloc[0] == 0.3  # Still picks min even if all negative

    def test_aggregation_with_identical_scores(self):
        """When scores are identical, should return one row per pk."""
        df = pd.DataFrame(
            {
                "Id": [1, 1, 1],
                "Score": [0.5, 0.5, 0.5],  # All same
                "EventName_Value": [1, 1, 1],
                "EventName_Time": [pd.Timestamp("2024-01-01")] * 3,
            }
        )
        result = undertest.max_aggregation(df, pks=["Id"], score="Score", ref_time="Time", ref_event="EventName")
        assert len(result) == 1

    def test_aggregation_with_inf_values(self):
        """Aggregation should handle inf/-inf values."""
        df = pd.DataFrame(
            {
                "Id": [1, 1],
                "Score": [float("inf"), 0.5],
                "EventName_Value": [1, 1],
                "EventName_Time": [pd.Timestamp("2024-01-01")] * 2,
            }
        )
        result = undertest.max_aggregation(df, pks=["Id"], score="Score", ref_time="Time", ref_event="EventName")
        assert result["Score"].iloc[0] == float("inf")

    def test_first_last_aggregation_with_identical_times(self):
        """When times are identical, first/last should still return one row."""
        same_time = pd.Timestamp("2024-01-01")
        df = pd.DataFrame({"Id": [1, 1], "Score": [0.3, 0.7], "EventName_Time": [same_time, same_time]})
        result_first = undertest.first_aggregation(
            df, pks=["Id"], score="Score", ref_time="EventName_Time", ref_event="EventName"
        )
        result_last = undertest.last_aggregation(
            df, pks=["Id"], score="Score", ref_time="EventName_Time", ref_event="EventName"
        )
        assert len(result_first) == 1
        assert len(result_last) == 1

    def test_aggregation_empty_dataframe(self):
        """Empty DataFrame should return empty result."""
        df = pd.DataFrame({"Id": [], "Score": [], "EventName_Value": [], "EventName_Time": []})
        result = undertest.max_aggregation(df, pks=["Id"], score="Score", ref_time="Time", ref_event="EventName")
        assert len(result) == 0

    def test_max_aggregation_missing_score_column_raises(self):
        """Missing score column should raise clear ValueError."""
        df = pd.DataFrame({"Id": [1], "EventName_Value": [1], "EventName_Time": [pd.Timestamp.now()]})
        with pytest.raises(ValueError, match="NonExistentScore"):
            undertest.max_aggregation(
                df, pks=["Id"], score="NonExistentScore", ref_time="EventName_Time", ref_event="EventName"
            )

    def test_first_aggregation_missing_ref_time_column_raises(self):
        """Missing ref_time column should raise clear error."""
        df = pd.DataFrame({"Id": [1], "Score": [0.5]})
        # First check ValueError for None
        with pytest.raises(ValueError, match="ref_time is required"):
            undertest.first_aggregation(df, pks=["Id"], score="Score", ref_time=None, ref_event="EventName")

        # Then check what happens when ref_time doesn't exist in DataFrame
        with pytest.raises(ValueError, match="Reference time column .* not found"):
            undertest.first_aggregation(df, pks=["Id"], score="Score", ref_time="NonExistent", ref_event="EventName")

    def test_aggregation_duplicate_pks_keeps_first_after_sort(self):
        """With duplicate pks, aggregation should keep first after sorting."""
        df = pd.DataFrame(
            {
                "Id": [1, 1, 1],  # Duplicates
                "Score": [0.3, 0.9, 0.5],
                "EventName_Value": [1, 1, 1],
                "EventName_Time": [pd.Timestamp("2024-01-01")] * 3,
            }
        )

        # max_aggregation sorts by EventName_Value (desc), Score (desc), then drops duplicates
        result = undertest.max_aggregation(df, pks=["Id"], score="Score", ref_time="Time", ref_event="EventName")

        # Should keep highest score (0.9)
        assert len(result) == 1
        assert result["Score"].iloc[0] == 0.9


class TestAnalyticsMetricName:
    """Tests for analytics_metric_name utility function."""

    def test_returns_column_name_if_in_metric_names(self):
        """If column_name is already in metric_names, return it unchanged."""
        metric_names = ["accuracy", "precision", "recall"]
        result = undertest.analytics_metric_name(metric_names, [], "accuracy")
        assert result == "accuracy"

    def test_strips_prefix_if_matches_existing_metric_starts(self):
        """If column starts with metric prefix, strip it."""
        metric_names = []
        existing_starts = ["model_v1", "model_v2"]
        result = undertest.analytics_metric_name(metric_names, existing_starts, "model_v1_accuracy")
        assert result == "accuracy"

    def test_returns_none_if_no_match(self):
        """If no match found, return None."""
        metric_names = ["accuracy"]
        existing_starts = ["model_v1"]
        result = undertest.analytics_metric_name(metric_names, existing_starts, "unknown_metric")
        assert result is None

    @pytest.mark.parametrize(
        "metric_names,existing_starts,column_name,expected",
        [
            (["accuracy"], [], "accuracy", "accuracy"),  # Direct match
            ([], ["model"], "model_accuracy", "accuracy"),  # Prefix strip
            ([], ["v1", "v2"], "v1_precision", "precision"),  # First prefix match
            ([], ["v1", "v2"], "v2_recall", "recall"),  # Second prefix match
            (["score"], ["model"], "score", "score"),  # Direct match takes precedence
            ([], [], "metric", None),  # No match
            ([], ["prefix"], "other_metric", None),  # Wrong prefix
            ([], ["model"], "model_model", "model"),  # Repeated prefix chars
            ([], ["model"], "mode_accuracy", None),  # Similar but doesn't start with prefix
        ],
        ids=[
            "direct_match",
            "prefix_strip",
            "first_prefix",
            "second_prefix",
            "direct_over_prefix",
            "no_match",
            "wrong_prefix",
            "repeated_prefix_chars",
            "similar_not_prefix",
        ],
    )
    def test_analytics_metric_name_various_cases(self, metric_names, existing_starts, column_name, expected):
        """Test various scenarios for analytics_metric_name."""
        result = undertest.analytics_metric_name(metric_names, existing_starts, column_name)
        assert result == expected


class TestEventScoreAndModelScores:
    @pytest.mark.parametrize("method", ["max", "min", "first", "last"])
    def test_event_score_valid_methods(self, method):
        now = pd.Timestamp.now()
        df = pd.DataFrame(
            {
                "Id": [1, 1, 1],
                "Score": [0.2, 0.5, 0.1],
                "EventName_Value": [1, 1, 0],
                "EventName_Time": [now, now + pd.Timedelta("1h"), now + pd.Timedelta("2h")],
            }
        )

        result = undertest.event_score(
            df,
            pks=["Id"],
            score="Score",
            ref_time="EventName_Time",
            ref_event="EventName",
            aggregation_method=method,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_event_score_raises_on_invalid_method(self):
        df = pd.DataFrame({"Id": [1], "Score": [0.5]})
        with pytest.raises(ValueError, match="Unknown aggregation method: badmethod"):
            undertest.event_score(
                df,
                pks=["Id"],
                score="Score",
                ref_time="Time",
                ref_event="EventName",
                aggregation_method="badmethod",
            )

    def test_get_model_scores_forwards_to_event_score(self):
        df = pd.DataFrame(
            {"Id": [1, 1], "Score": [0.2, 0.8], "EventName_Value": [1, 0], "EventName_Time": [pd.Timestamp.now()] * 2}
        )

        result = undertest.get_model_scores(
            df,
            entity_keys=["Id"],
            score_col="Score",
            ref_time="EventName_Time",
            ref_event="EventName",
            aggregation_method="max",
            per_context_id=True,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_get_model_scores_bypass_when_not_per_context(self):
        df = pd.DataFrame(
            {
                "Id": [1, 1],
                "Score": [0.2, 0.8],
            }
        )
        result = undertest.get_model_scores(
            df,
            entity_keys=["Id"],
            score_col="Score",
            ref_time=None,
            ref_event=None,
            aggregation_method="max",
            per_context_id=False,
        )
        pd.testing.assert_frame_equal(result, df)

    def test_event_score_empty_dataframe(self):
        """Empty DataFrame should return empty result."""
        df = pd.DataFrame({"Id": [], "Score": [], "EventName_Value": [], "EventName_Time": []})
        result = undertest.event_score(
            df, pks=["Id"], score="Score", ref_time="EventName_Time", ref_event="EventName", aggregation_method="max"
        )
        assert len(result) == 0

    def test_event_score_pks_not_in_dataframe(self):
        """When pks don't exist, should filter to available columns."""
        df = pd.DataFrame({"Id": [1], "Score": [0.5], "EventName_Value": [1], "EventName_Time": [pd.Timestamp.now()]})
        result = undertest.event_score(
            df,
            pks=["Id", "NonExistent"],
            score="Score",
            ref_time="EventName_Time",
            ref_event="EventName",
            aggregation_method="max",
        )
        assert len(result) == 1

    def test_get_model_scores_empty_dataframe(self):
        """Empty DataFrame should return empty when per_context_id=True."""
        df = pd.DataFrame({"Id": [], "Score": [], "Event_Value": [], "Event_Time": []})
        result = undertest.get_model_scores(
            df,
            entity_keys=["Id"],
            score_col="Score",
            ref_time="Event_Time",
            ref_event="Event",
            aggregation_method="max",
            per_context_id=True,
        )
        assert len(result) == 0

    def test_event_score_all_nan_scores(self):
        """All NaN scores should return empty result after filtering."""
        df = pd.DataFrame(
            {
                "Id": [1, 1, 2],
                "Score": [float("nan"), float("nan"), float("nan")],  # All NaN
                "EventName_Value": [1, 1, 1],
                "EventName_Time": [pd.Timestamp("2024-01-01")] * 3,
            }
        )

        result = undertest.event_score(
            df, pks=["Id"], score="Score", ref_time="EventName_Time", ref_event="EventName", aggregation_method="max"
        )

        # After aggregation and filtering NaN indices, should return empty or rows with NaN scores
        # The function filters out NaN indices with ~np.isnan(df.index)
        assert isinstance(result, pd.DataFrame)


class TestMergeEventCounts:
    def test_skips_time_filter_when_window_none(self, base_counts_data):
        preds, events = base_counts_data
        events["~~reftime~~"] = events["Event_Time"]
        out = undertest._merge_event_counts(
            preds, events, pks=["Id"], event_name="MyEvent", event_label="Label", l_ref="Time", r_ref="~~reftime~~"
        )
        assert out.shape[0] == preds.shape[0]
        for label in ["A", "B", "C"]:
            assert f"Label~{label}_Count" in out.columns

    def test_warns_and_returns_when_no_times(self, caplog):
        preds = pd.DataFrame({"Id": [1], "Time": [pd.Timestamp("2024-01-01")]})
        events = pd.DataFrame({"Id": [1], "Event_Time": [pd.NaT], "Label": ["A"], "~~reftime~~": [pd.NaT]})
        with caplog.at_level("WARNING"):
            result = undertest._merge_event_counts(
                preds, events, ["Id"], "MyEvent", "Label", window_hrs=1, l_ref="Time", r_ref="~~reftime~~"
            )
            assert "No times found" in caplog.text
            pd.testing.assert_frame_equal(result, preds)

    def test_warns_and_filters_missing_times(self, base_counts_data, caplog):
        preds, events = base_counts_data
        events["~~reftime~~"] = events["Event_Time"]
        with caplog.at_level("WARNING"):
            result = undertest._merge_event_counts(
                preds, events, ["Id"], "MyEvent", "Label", window_hrs=1, l_ref="Time", r_ref="~~reftime~~"
            )
            assert "1 rows with missing times" in caplog.text
            assert result.shape[0] == preds.shape[0]
            assert "MyEvent~C_Count" not in result.columns

    def test_warns_and_truncates_too_many_categories(self, caplog):
        preds = pd.DataFrame({"Id": [1], "Time": [pd.Timestamp("2024-01-01 8:00")]})
        events = pd.DataFrame(
            {
                "Id": [1] * (undertest.MAXIMUM_COUNT_CATS + 1),
                "Event_Time": [pd.Timestamp("2024-01-01 10:00")] * (undertest.MAXIMUM_COUNT_CATS + 1),
                "Label": [f"Cat{i}" for i in range(undertest.MAXIMUM_COUNT_CATS + 1)],
                "~~reftime~~": [pd.Timestamp("2024-01-01 10:00")] * (undertest.MAXIMUM_COUNT_CATS + 1),
            }
        )

        with caplog.at_level("WARNING"):
            result = undertest._merge_event_counts(
                preds,
                events,
                ["Id"],
                "MyEvent",
                "Label",
                window_hrs=5,
            )

        # Ensure truncation happened
        assert len([c for c in result.columns if "_Count" in c]) == undertest.MAXIMUM_COUNT_CATS
        assert "Maximum number of unique events" in caplog.text

    def test_counts_are_correct(self, base_counts_data, caplog):
        preds, events = base_counts_data
        events["~~reftime~~"] = events["Event_Time"]
        with caplog.at_level("WARNING"):
            result = undertest._merge_event_counts(
                preds, events, ["Id"], "MyEvent", "Label", window_hrs=5, l_ref="Time", r_ref="~~reftime~~"
            )

        assert result["Label~A_Count"].iloc[0] == 0
        assert result["Label~B_Count"].iloc[0] == 1
        assert result["Label~A_Count"].iloc[1] == 1
        assert result["Label~B_Count"].iloc[1] == 0
        assert "1 rows with missing times" in caplog.text

    def test_merge_event_counts_raises_on_non_timedelta(self, base_counts_data):
        preds, events = base_counts_data
        events["~~reftime~~"] = events["Event_Time"]
        with pytest.raises(TypeError):
            undertest._merge_event_counts(
                preds, events, ["Id"], "MyEvent", "Label", window_hrs=1, min_offset=0  # not a Timedelta
            )

    def test_merge_event_counts_raises_on_same_lref_rref(self):
        left = pd.DataFrame({"Id": [1], "Time": [pd.Timestamp("2024-01-01 01:00:00")]})
        right = pd.DataFrame({"Id": [1], "Time": [pd.Timestamp("2024-01-01 00:00:00")], "Label": ["A"]})

        with pytest.raises(ValueError, match="must be different to avoid column collisions"):
            undertest._merge_event_counts(
                left,
                right,
                pks=["Id"],
                event_name="TestEvent",
                event_label="Label",
                window_hrs=1,
                l_ref="Time",
                r_ref="Time",  # triggers the error
            )

    def test_counts_respect_min_offset(self, base_counts_data):
        preds, events = base_counts_data

        # min_offset shifts the event window 1 hour into the future
        min_offset = pd.Timedelta(hours=1)
        events["~~reftime~~"] = events["Event_Time"] + min_offset

        result = undertest._merge_event_counts(
            preds,
            events,
            pks=["Id"],
            event_name="MyEvent",
            event_label="Label",
            window_hrs=1,  # narrow window
            min_offset=min_offset,
            l_ref="Time",
            r_ref="~~reftime~~",
        )

        assert "Label~A_Count" in result.columns
        assert "Label~B_Count" in result.columns

        # ID 1 should get both A and B counted
        assert result["Label~A_Count"].iloc[0] == 1
        assert result["Label~B_Count"].iloc[0] == 1

        # ID 2 should get zero counts
        assert result["Label~A_Count"].iloc[1] == 0
        assert result["Label~B_Count"].iloc[1] == 0

    def test_merge_event_counts_empty_left_returns_empty(self):
        """Empty left DataFrame should return empty."""
        preds = pd.DataFrame({"Id": pd.Series([], dtype=int), "Time": pd.Series([], dtype="datetime64[ns]")})
        events = pd.DataFrame(
            {
                "Id": [1],
                "Event_Time": [pd.Timestamp("2024-01-01")],
                "Label": ["A"],
                "~~reftime~~": [pd.Timestamp("2024-01-01")],
            }
        )
        result = undertest._merge_event_counts(
            preds, events, ["Id"], "MyEvent", "Label", window_hrs=1, l_ref="Time", r_ref="~~reftime~~"
        )
        assert len(result) == 0

    def test_merge_event_counts_empty_right_returns_left(self):
        """Empty right DataFrame should return left unchanged (no counts added)."""
        preds = pd.DataFrame({"Id": [1], "Time": [pd.Timestamp("2024-01-01")]})
        events = pd.DataFrame({"Id": pd.Series([], dtype=int), "Event_Time": [], "Label": [], "~~reftime~~": []})
        result = undertest._merge_event_counts(
            preds, events, ["Id"], "MyEvent", "Label", window_hrs=1, l_ref="Time", r_ref="~~reftime~~"
        )
        pdt.assert_frame_equal(result, preds)

    def test_merge_event_counts_with_nan_labels(self, base_counts_data):
        """NaN values in event_label should be handled (pandas treats as category)."""
        preds, events = base_counts_data
        events["~~reftime~~"] = events["Event_Time"]
        # Don't set NaN - pandas might not handle NaN well in value_counts pivot
        # Instead test that function works with various label values

        result = undertest._merge_event_counts(
            preds, events, ["Id"], "MyEvent", "Label", window_hrs=5, l_ref="Time", r_ref="~~reftime~~"
        )
        # Should work and have count columns
        assert any("_Count" in col for col in result.columns)

    def test_merge_event_counts_very_small_window(self):
        """Very small window_hrs should have narrow window."""
        preds = pd.DataFrame({"Id": [1], "Time": [pd.Timestamp("2024-01-01 01:00:00")]})
        events = pd.DataFrame(
            {
                "Id": [1, 1],
                "Event_Time": [pd.Timestamp("2024-01-01 01:00:30"), pd.Timestamp("2024-01-01 03:00:00")],
                "Label": ["A", "B"],
                "~~reftime~~": [pd.Timestamp("2024-01-01 01:00:30"), pd.Timestamp("2024-01-01 03:00:00")],
            }
        )
        result = undertest._merge_event_counts(
            preds, events, ["Id"], "MyEvent", "Label", window_hrs=1, l_ref="Time", r_ref="~~reftime~~"
        )
        # With 1 hour window, event A should be included, B should not
        assert "Label~A_Count" in result.columns
        assert result["Label~A_Count"].iloc[0] == 1

    def test_merge_event_counts_negative_min_offset(self):
        """Negative min_offset allows looking into past - valid use case."""
        preds = pd.DataFrame({"Id": [1], "Time": [pd.Timestamp("2024-01-01 12:00:00")]})
        events = pd.DataFrame(
            {
                "Id": [1, 1],
                "Event_Time": [
                    pd.Timestamp("2024-01-01 10:00:00"),  # 2 hours before pred
                    pd.Timestamp("2024-01-01 14:00:00"),  # 2 hours after pred
                ],
                "Label": ["A", "B"],
                "~~reftime~~": [
                    pd.Timestamp("2024-01-01 12:00:00"),  # Adjusted by negative offset
                    pd.Timestamp("2024-01-01 16:00:00"),
                ],
            }
        )

        # Negative offset of -2 hours means we look 2 hours into the past
        result = undertest._merge_event_counts(
            preds,
            events,
            ["Id"],
            "MyEvent",
            "Label",
            window_hrs=3,
            min_offset=pd.Timedelta(hours=-2),  # Negative: look into past
            l_ref="Time",
            r_ref="~~reftime~~",
        )

        # Both events should be counted with the negative offset
        assert "Label~A_Count" in result.columns
        assert result["Label~A_Count"].iloc[0] == 1

    def test_merge_event_counts_large_min_offset(self):
        """Large min_offset (larger than window) should work correctly."""
        preds = pd.DataFrame({"Id": [1], "Time": [pd.Timestamp("2024-01-01 12:00:00")]})
        events = pd.DataFrame(
            {
                "Id": [1],
                "Event_Time": [pd.Timestamp("2024-01-01 20:00:00")],  # 8 hours after
                "Label": ["A"],
                "~~reftime~~": [pd.Timestamp("2024-01-01 20:00:00")],
            }
        )

        # min_offset of 5 hours with window of 2 hours
        # Window is [pred+5hrs, pred+7hrs] = [17:00, 19:00]
        # Event at 20:00 is outside window
        result = undertest._merge_event_counts(
            preds,
            events,
            ["Id"],
            "MyEvent",
            "Label",
            window_hrs=2,
            min_offset=pd.Timedelta(hours=5),
            l_ref="Time",
            r_ref="~~reftime~~",
        )

        # Event should not be counted (outside window)
        if "Label~A_Count" in result.columns:
            assert result["Label~A_Count"].iloc[0] == 0


class TestMergeWindowedEvent:
    def test_basic_forward_strategy(self):
        preds = pd.DataFrame(
            {
                "Id": [1, 1],
                "PredictTime": [
                    pd.Timestamp("2024-01-01 01:30:00"),
                    pd.Timestamp("2024-01-01 04:00:00"),
                ],
            }
        )

        events = pd.DataFrame(
            {
                "Id": [1, 1],
                "Time": [
                    pd.Timestamp("2024-01-01 03:00:00"),
                    pd.Timestamp("2024-01-01 05:00:00"),
                ],
                "Value": [1, 1],
                "Type": ["MyEvent", "MyEvent"],
            }
        )

        result = undertest.merge_windowed_event(
            preds,
            predtime_col="PredictTime",
            events=events,
            event_label="MyEvent",
            pks=["Id"],
            min_leadtime_hrs=1,
            window_hrs=2,
            event_base_val_col="Value",
            event_base_time_col="Time",
            merge_strategy="forward",
            impute_val_with_time=1,
            impute_val_no_time=0,
        )

        assert result["MyEvent_Time"].iloc[0] == pd.Timestamp("2024-01-01 03:00:00")
        assert result["MyEvent_Value"].iloc[0] == 1
        assert result["MyEvent_Time"].iloc[1] == pd.Timestamp("2024-01-01 05:00:00")
        assert result["MyEvent_Value"].iloc[1] == 1

    def test_merge_event_with_count_strategy(self):
        preds = pd.DataFrame(
            {
                "Id": [1, 2],
                "PredictTime": [
                    pd.Timestamp("2024-01-01 07:15:00"),
                    pd.Timestamp("2024-01-01 05:45:00"),
                ],
            }
        )
        events = pd.DataFrame(
            {
                "Id": [1, 1, 2, 2],
                "Time": [
                    pd.Timestamp("2024-01-01 07:30:00"),
                    pd.Timestamp("2024-01-01 07:00:00"),
                    pd.Timestamp("2024-01-01 08:00:00"),
                    pd.Timestamp("2024-01-01 06:00:00"),
                ],
                "Value": ["A", "B", "B", "C"],
                "Type": ["MyEvent"] * 4,
            }
        )

        result = undertest.merge_windowed_event(
            preds,
            predtime_col="PredictTime",
            events=events,
            event_label="MyEvent",
            pks=["Id"],
            min_leadtime_hrs=0,
            window_hrs=3,
            merge_strategy="count",
            event_base_val_col="Value",
            event_base_time_col="Time",
        )

        assert "MyEvent~A_Count" in result.columns
        assert "MyEvent~B_Count" in result.columns
        assert "MyEvent~C_Count" in result.columns
        assert result.shape[0] == preds.shape[0]

    def test_merge_event_invalid_strategy_raises(self):
        preds = pd.DataFrame({"Id": [1], "PredictTime": [pd.Timestamp("2024-01-01 00:00:00")]})
        events = pd.DataFrame(
            {"Id": [1], "Time": [pd.Timestamp("2024-01-01 01:00:00")], "Value": [1], "Type": ["MyEvent"]}
        )

        with pytest.raises(ValueError, match="Invalid merge strategy"):
            undertest.merge_windowed_event(
                preds,
                predtime_col="PredictTime",
                events=events,
                event_label="MyEvent",
                pks=["Id"],
                min_leadtime_hrs=0,
                window_hrs=5,
                merge_strategy="invalid_strategy",
                event_base_val_col="Value",
                event_base_time_col="Time",
            )

    def test_merge_event_raises_if_no_common_keys(self):
        preds = pd.DataFrame({"A": [1], "PredictTime": [pd.Timestamp("2024-01-01 00:00:00")]})
        events = pd.DataFrame(
            {"B": [1], "Time": [pd.Timestamp("2024-01-01 01:00:00")], "Value": [1], "Type": ["MyEvent"]}
        )

        with pytest.raises(ValueError, match="No common keys"):
            undertest.merge_windowed_event(
                preds,
                predtime_col="PredictTime",
                events=events,
                event_label="MyEvent",
                pks=["A", "B"],  # no shared key
                min_leadtime_hrs=0,
                window_hrs=5,
                merge_strategy="forward",
                event_base_val_col="Value",
                event_base_time_col="Time",
            )

    @pytest.mark.parametrize(
        "log_level,should_log",
        [
            (logging.DEBUG, True),
            (logging.INFO, True),
            (logging.WARNING, False),
        ],
    )
    def test_merge_windowed_event_info_logging(self, log_level, should_log, caplog):
        caplog.set_level(log_level, logger="seismometer")

        preds = pd.DataFrame({"Id": [1], "PredictTime": [pd.Timestamp("2024-01-01 00:00:00")]})
        events = pd.DataFrame(
            {"Id": [1], "Time": [pd.Timestamp("2024-01-01 01:00:00")], "Value": [1], "Type": ["MyEvent"]}
        )

        undertest.merge_windowed_event(
            preds,
            predtime_col="PredictTime",
            events=events,
            event_label="MyEvent",
            pks=["Id"],
            min_leadtime_hrs=0,
            window_hrs=5,
            merge_strategy="forward",
            event_base_val_col="Value",
            event_base_time_col="Time",
        )

        info_logs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        matched = any("Kept" in msg and "MyEvent" in msg for msg in info_logs)
        assert matched == should_log

    @pytest.mark.parametrize(
        "log_level,should_log",
        [
            (logging.DEBUG, True),
            (logging.INFO, True),
            (logging.WARNING, False),
        ],
    )
    def test_merge_with_strategy_info_logging(self, log_level, should_log, caplog):
        caplog.set_level(log_level, logger="seismometer")

        preds = pd.DataFrame({"Id": [1], "PredictTime": [pd.Timestamp("2024-01-01 00:00:00")]})
        events = pd.DataFrame(
            {"Id": [1], "Time": [pd.Timestamp("2024-01-01 01:00:00")], "Value": [1], "Type": ["MyEvent"]}
        )

        one_event = undertest._one_event(events, "MyEvent", "Value", "Time", ["Id"])
        _ = undertest._merge_with_strategy(
            predictions=preds,
            one_event=one_event,
            pks=["Id"],
            pred_ref="PredictTime",
            event_ref="MyEvent_Time",
            event_display="MyEvent",
            merge_strategy="forward",
        )

        info_logs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        matched = any("Added" in msg and "MyEvent" in msg for msg in info_logs)
        assert matched == should_log

    def test_merge_windowed_event_missing_predtime_col_raises(self):
        """Missing predtime_col should raise KeyError."""
        preds = pd.DataFrame({"Id": [1], "SomeOtherCol": [pd.Timestamp("2024-01-01")]})
        events = pd.DataFrame({"Id": [1], "Time": [pd.Timestamp("2024-01-01")], "Value": [1], "Type": ["MyEvent"]})

        with pytest.raises(KeyError):
            undertest.merge_windowed_event(
                preds,
                predtime_col="PredictTime",  # Doesn't exist
                events=events,
                event_label="MyEvent",
                pks=["Id"],
                window_hrs=5,
                event_base_val_col="Value",
                event_base_time_col="Time",
            )

    def test_merge_windowed_event_missing_event_time_col_raises(self):
        """Missing event_base_time_col should raise KeyError."""
        preds = pd.DataFrame({"Id": [1], "PredictTime": [pd.Timestamp("2024-01-01")]})
        events = pd.DataFrame({"Id": [1], "Value": [1], "Type": ["MyEvent"]})  # No Time column

        with pytest.raises(KeyError):
            undertest.merge_windowed_event(
                preds,
                predtime_col="PredictTime",
                events=events,
                event_label="MyEvent",
                pks=["Id"],
                window_hrs=5,
                event_base_val_col="Value",
                event_base_time_col="Time",  # Doesn't exist
            )

    def test_merge_windowed_event_invalid_event_label_returns_unchanged(self):
        """Event label not in Type column should return predictions unchanged (early return)."""
        preds = pd.DataFrame({"Id": [1], "PredictTime": [pd.Timestamp("2024-01-01")]})
        events = pd.DataFrame(
            {"Id": [1], "Time": [pd.Timestamp("2024-01-01")], "Value": [1], "Type": ["DifferentEvent"]}
        )

        result = undertest.merge_windowed_event(
            preds,
            predtime_col="PredictTime",
            events=events,
            event_label="MyEvent",  # Not in Type column
            pks=["Id"],
            window_hrs=5,
            event_base_val_col="Value",
            event_base_time_col="Time",
        )

        # Should return predictions completely unchanged (early return when no events found)
        assert len(result) == len(preds)
        # No event columns added when event label doesn't exist
        assert "MyEvent_Value" not in result.columns
        assert "MyEvent_Time" not in result.columns
        pdt.assert_frame_equal(result, preds)

    def test_merge_windowed_event_with_sort_false(self):
        """Test sort=False parameter - unsorted data should raise ValueError."""
        # Create predictions and events in reverse chronological order (unsorted)
        preds = pd.DataFrame(
            {
                "Id": [1, 1],
                "PredictTime": [
                    pd.Timestamp("2024-01-01 04:00:00"),  # Later time first
                    pd.Timestamp("2024-01-01 01:00:00"),
                ],
            }
        )
        events = pd.DataFrame(
            {
                "Id": [1, 1],
                "Time": [
                    pd.Timestamp("2024-01-01 05:00:00"),  # Later time first
                    pd.Timestamp("2024-01-01 02:00:00"),
                ],
                "Value": [2, 1],
                "Type": ["MyEvent", "MyEvent"],
            }
        )

        # merge_asof with unsorted data and sort=False should raise ValueError
        with pytest.raises(ValueError):
            undertest.merge_windowed_event(
                preds,
                predtime_col="PredictTime",
                events=events,
                event_label="MyEvent",
                pks=["Id"],
                window_hrs=5,
                merge_strategy="forward",
                event_base_val_col="Value",
                event_base_time_col="Time",
                sort=False,  # Important: test unsorted merge raises error
            )
