import json
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from profile_cases import make_profile_frame

import seismometer.data.profile as undertest
from seismometer.data.profile import AlertType

GOLDEN_PATH = "profile/golden.json"
# Memory depends on the pandas version and string dtype, not on the profile logic
NOT_COMPARED = {"memory_size", "record_size"}


@pytest.fixture(scope="module")
def golden():
    from conftest import TEST_ROOT

    return json.loads((TEST_ROOT / "resources" / GOLDEN_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def profile():
    return undertest.profile_dataframe(make_profile_frame(), title="Profile")


def assert_close(actual, expected, where):
    if isinstance(expected, str) and expected in ("nan", "inf", "-inf"):
        assert str(float(actual)) == expected, where
    elif isinstance(expected, bool) or expected is None:
        assert actual == expected, where
    elif isinstance(expected, (int, float)):
        assert math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12), (where, actual, expected)
    else:
        assert str(actual) == expected, (where, actual, expected)


def all_counts(series):
    assert series.is_monotonic_decreasing, "counts not sorted by frequency"
    return {str(label): int(count) for label, count in series.items()}


class TestMatchesYdataProfiling:
    def test_table_stats(self, profile, golden):
        for key, expected in golden["table"].items():
            if key in NOT_COMPARED:
                continue
            if key == "types":
                assert profile.table["types"] == expected
            else:
                assert_close(profile.table[key], expected, key)

    def test_column_types(self, profile, golden):
        assert {col: d["type"] for col, d in profile.variables.items()} == {
            col: d["type"] for col, d in golden["variables"].items()
        }

    @pytest.mark.parametrize("column", list(make_profile_frame().columns))
    def test_column_statistics(self, profile, golden, column):
        actual, expected = profile.variables[column], golden["variables"][column]
        for key, value in expected.items():
            if key in NOT_COMPARED:
                continue
            where = f"{column}.{key}"
            if key in ("value_counts_without_nan", "character_counts", "word_counts"):
                assert all_counts(actual[key]) == value, where
            elif key == "first_rows":
                assert [str(v) for v in actual[key]] == value, where
            elif key in ("histogram", "histogram_length"):
                np.testing.assert_allclose(actual[key][0], value[0], err_msg=where)
                np.testing.assert_allclose(actual[key][1], value[1], err_msg=where)
            else:
                assert_close(actual[key], value, where)

    def test_alerts(self, profile, golden):
        assert [(a.alert_type.name, a.column_name) for a in profile.alerts] == [
            (a["type"], a["column"]) for a in golden["alerts"]
        ]


class TestProfile:
    def test_does_not_modify_input(self):
        df = make_profile_frame().rename(columns={"text": "index"})
        before = df.copy()
        profile = undertest.profile_dataframe(df)
        pd.testing.assert_frame_equal(df, before)
        assert "df_index" in profile.variables

    def test_empty_frame_alert(self):
        profile = undertest.profile_dataframe(pd.DataFrame({"a": pd.Series([], dtype=float)}))
        # an empty column has no type, so it is also unsupported
        assert [a.alert_type for a in profile.alerts] == [AlertType.EMPTY, AlertType.REJECTED, AlertType.UNSUPPORTED]
        assert profile.table["n"] == 0

    def test_alert_values_are_the_column_description(self, profile):
        alert = next(a for a in profile.alerts if a.alert_type == AlertType.ZEROS)
        assert alert.values is profile.variables[alert.column_name]
        assert alert.alert_type_name == "Zeros"

    def test_nullable_floats_with_infinite_values_match_floats(self):
        values = [1.0, 2.0, 3.0, np.inf, np.inf]
        nullable = undertest.describe_series(pd.Series(values, dtype="Float64"))
        plain = undertest.describe_series(pd.Series(values))
        keys = ["5%", "25%", "50%", "75%", "95%", "iqr"]
        assert [nullable[k] for k in keys] == pytest.approx([plain[k] for k in keys], nan_ok=True)

    @pytest.mark.parametrize("value,dtype", [(7, "int64"), (7, "Int64"), (0.1, "float64"), (0.1, "Float64")])
    def test_constant_columns_have_zero_kurtosis_and_skewness(self, value, dtype):
        description = undertest.describe_series(pd.Series([value] * 1000, dtype=dtype))
        assert (description["kurtosis"], description["skewness"]) == (0.0, 0.0)

    def test_alert_type_name(self):
        assert undertest.Alert(AlertType.HIGH_CARDINALITY).alert_type_name == "High cardinality"

    @pytest.mark.parametrize(
        "series,expected",
        [
            (pd.Series([1.0, None]), "Numeric"),
            (pd.Series([True, False]), "Boolean"),
            (pd.Series([True, None], dtype=object), "Boolean"),
            (pd.Series(["a", "b"], dtype="category"), "Categorical"),
            (pd.Series(["a", None], dtype=object), "Text"),
            (pd.Series(["a", None], dtype="string"), "Text"),
            (pd.Series([date(2024, 1, 1)], dtype=object), "DateTime"),
            (pd.Series(["a", 1], dtype=object), "Unsupported"),
            (pd.Series([None, None], dtype=float), "Unsupported"),
            (pd.Series([], dtype=float), "Unsupported"),
            (pd.Series([[1], [2]], dtype=object), "Unsupported"),
        ],
    )
    def test_detect_type(self, series, expected):
        assert undertest.detect_type(series) == expected

    def test_unhashable_values(self):
        description = undertest.describe_series(pd.Series([[1], [2], None]))
        assert description["type"] == "Unsupported"
        assert not description["hashable"]
        assert description["n_missing"] == 1

    @pytest.mark.parametrize(
        "counts,expected",
        [([10], 0), ([5, 5], 0.0), ([99, 1], 1 - 0.08079313589591118), ([1, 1, 1, 1], 0.0)],
    )
    def test_imbalance_score(self, counts, expected):
        assert undertest.imbalance_score(pd.Series(counts)) == pytest.approx(expected)

    def test_arrow_strings_match_object_strings(self):
        values = ["x", "y y", None, "x", "z!"]
        as_object = undertest.describe_series(pd.Series(values, dtype=object))
        as_arrow = undertest.describe_series(pd.Series(values, dtype="string[pyarrow]"))
        for key in ["type", "n_distinct", "n_missing", "max_length", "mean_length", "n_characters"]:
            assert as_object[key] == as_arrow[key], key
        assert all_counts(as_object["word_counts"]) == all_counts(as_arrow["word_counts"])
