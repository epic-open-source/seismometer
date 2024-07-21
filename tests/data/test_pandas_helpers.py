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


class TestMergeFrames:
    @pytest.mark.parametrize("id_,enc", [pytest.param(1, None, id="predictions+outcomes")])
    def test_merge_earliest(self, id_, enc, merge_data):
        data = filter_case(merge_data, id_, enc)
        # TODO: rename second time
        expect = data.expect.drop(columns=[c for c in data.expect if "Cohort" in c]).rename(columns={"✨Time✨": "Time"})

        actual = undertest._merge_next(data.preds, data.events, ["Id", "Enc"], l_ref="PredictTime")

        # check_like = ignore column order
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expect, check_like=True, check_dtype=False)


def infer_cases():
    return pd.DataFrame(
        {
            "label_in": [1, 0, 1, 0, None, None],
            "time_in": [1, 1, None, None, 1, None],
            "label_out": [1, 0, 1, 0, 1, 0],
            "description": [
                "label1+time keeps",
                "label0+time keeps label",
                "label1+no time keeps label",
                "label0+no time keeps label",
                "no label with time infers to positive",
                "no label nor time infers to negative",
            ],
        }
    )


def one_line_case():
    for _, row in infer_cases().iterrows():
        yield pytest.param(*row[:-1].values, id=row["description"])


class TestInferLabel:
    @pytest.mark.parametrize("label_in,time_in,label_out", one_line_case())
    def test_infer_one_line(self, label_in, time_in, label_out):
        col_label = "Label"
        col_time = "Time"
        dataframe = pd.DataFrame({col_label: [label_in], col_time: pd.to_datetime([time_in])})
        expect = pd.DataFrame({col_label: [label_out], col_time: pd.to_datetime([time_in])})

        actual = undertest.infer_label(dataframe, col_label, col_time)
        # actual['Label'] = actual['Label'].astype(int) # handle inference where input frame could be all null series

        pdt.assert_frame_equal(actual, expect, check_dtype=False)

    def test_infer_multi_line(self):
        all_cases = infer_cases()
        col_label = "Label"
        col_time = "Time"
        col_map = {"label_in": col_label, "time_in": col_time, "label_out": col_label}

        dataframe = all_cases.iloc[:, :2].rename(columns={k: v for k, v in col_map.items() if k in all_cases.columns})
        expect = all_cases.iloc[:, 2:0:-1].rename(columns={k: v for k, v in col_map.items() if k in all_cases.columns})

        actual = undertest.infer_label(dataframe, col_label, col_time)

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
