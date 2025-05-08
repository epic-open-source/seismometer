from dataclasses import dataclass

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

    def test_post_process_event_skips_cast_when_dtype_none(self):
        df = pd.DataFrame({"Label": [None], "Time": [pd.Timestamp("2024-01-01")]})
        result = undertest.post_process_event(df, "Label", "Time", column_dtype=None)
        assert result["Label"].iloc[0] == 1

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
        ],
    )
    def test_suffix_specific_handling(self, input, expected):
        assert expected == undertest.event_name(input)


class TestEventHelpers:
    @pytest.mark.parametrize(
        "event_label, event_value, expected",
        [
            ("MyEvent", "Critical", "MyEvent~Critical_Count"),
            ("MyEvent_Value", "High_Count", "MyEvent~High_Count"),
        ],
    )
    def test_event_value_count(self, event_label, event_value, expected):
        assert undertest.event_value_count(event_label, event_value) == expected

    @pytest.mark.parametrize(
        "input, expected",
        [
            ("MyEvent~Critical_Count", "Critical"),
            ("MyEvent~123_Count", "123"),
            ("Event_Only_Count", "Event_Only"),  # no ~
        ],
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


class TestTryCasting:
    @pytest.mark.parametrize(
        "dtype, expected_type",
        [
            ("int", "int64"),
            ("float", "float64"),
            ("string", "string"),
            ("object", "object"),
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
            assert "rows with missing times" in caplog.text
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

    def test_counts_are_correct(self, base_counts_data):
        preds, events = base_counts_data
        events["~~reftime~~"] = events["Event_Time"]
        result = undertest._merge_event_counts(
            preds, events, ["Id"], "MyEvent", "Label", window_hrs=5, l_ref="Time", r_ref="~~reftime~~"
        )

        assert result["Label~A_Count"].iloc[0] == 0
        assert result["Label~B_Count"].iloc[0] == 1
        assert result["Label~A_Count"].iloc[1] == 1
        assert result["Label~B_Count"].iloc[1] == 0

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

        @pytest.mark.parametrize("strategy", ["forward", "nearest", "first", "last"])
        def test_merge_event_with_various_strategies(self, strategy):
            preds = pd.DataFrame(
                {
                    "Id": [1, 1],
                    "PredictTime": [
                        pd.Timestamp("2024-01-01 00:00:00"),
                        pd.Timestamp("2024-01-01 01:00:00"),
                    ],
                }
            )
            events = pd.DataFrame(
                {
                    "Id": [1, 1],
                    "Time": [
                        pd.Timestamp("2024-01-01 01:30:00"),
                        pd.Timestamp("2024-01-01 02:00:00"),
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
                merge_strategy=strategy,
                impute_val_with_time=1,
                impute_val_no_time=0,
            )

            # Result should include the matched event in _Value/_Time columns
            assert "MyEvent_Value" in result.columns
            assert "MyEvent_Time" in result.columns
            assert result["MyEvent_Value"].notna().all()

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


def test_post_process_event_skips_cast_when_dtype_none():
    df = pd.DataFrame({"Label": [None], "Time": [pd.Timestamp.now()]})
    result = undertest.post_process_event(df, "Label", "Time", column_dtype=None)
    assert "Label" in result.columns


def test_one_event_filters_and_renames():
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
