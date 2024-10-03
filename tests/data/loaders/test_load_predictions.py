import logging
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from conftest import tmp_as_current  # noqa

import seismometer.data.loader.prediction as undertest
from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import DictionaryItem


# region Fakes and Data Prep
def fake_config(prediction_file):
    # Create a fake configuration object
    class FakeConfigProvider:
        def __init__(self):
            self.entity_keys = ["id"]
            self.predict_time = "Time"
            self.prediction_path = prediction_file
            self.target = "target"  # unseen in base data
            self.output_list = ["score1", "score2"]

            # Intentionally empty, but used in logic for "all feattures"
            self.features = []  # This should be overwritten in tests
            self.cohorts = []

        @property
        def prediction_columns(self):
            # Benefit of focusing the Fake outweighs cost of duplicating this property logic
            col_set = set(
                self.entity_keys
                + [self.predict_time]
                + self.features
                + self.output_list
                + [c.source for c in self.cohorts]
            )
            return sorted(col_set)

    return FakeConfigProvider()


def pred_frame():
    # Create a mock predictions dataframe
    return pd.DataFrame(
        {
            "id": ["0", "1", "2"],
            "column1": [1, 2, 3],
            "column2": [4, 5, 6],
            "column3": [7, 8, 9],
            "maybe_target": [10, 20, 30],
            "score1": [0.5, 0.6, 0.7],
            "score2": [0.8, 0.9, 1.0],
            "Time": ["2022-01-01", "2022-01-02", "2022-01-03"],
        }
    ).sort_index(axis=1)


# endregion
# region File-type setup functions
def parquet_setup():
    file = Path("predictions.parquet")

    data = pred_frame()
    data.to_parquet(file)

    return fake_config(file)


# endregion


# region Tests
@pytest.mark.parametrize("setup_fn,load_fn", [[parquet_setup, undertest.parquet_loader]])
@pytest.mark.usefixtures("tmp_as_current")
class TestPredictionLoad:
    def test_load_all_columns(self, setup_fn, load_fn):
        config = setup_fn()
        expected = pred_frame()

        actual = load_fn(config)

        pdt.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "extra_columns",
        [
            pytest.param([], id="no_extra_columns"),
            pytest.param(["not_a_column"], id="one extra column"),
            pytest.param(["not_a_column", "another_extra"], id="multiple extra columns"),
        ],
    )
    def test_load_selected_columns(self, extra_columns, setup_fn, load_fn):
        column_subset = ["column1", "column2"]
        non_feature_columns = ["id", "score1", "score2", "Time"]

        config = setup_fn()
        config.features = column_subset if not extra_columns else column_subset + extra_columns
        expected = pred_frame()[sorted(column_subset + non_feature_columns)]

        actual = load_fn(config)

        pdt.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "extra_columns",
        [pytest.param([], id="all features"), pytest.param(["column1", "column2", "column3"], id="full features")],
    )
    def test_target_inclusion_is_renamed(self, extra_columns, setup_fn, load_fn):
        config = setup_fn()
        config.target = "maybe_target"
        config.features = extra_columns
        expected = pred_frame().rename(columns={"maybe_target": "maybe_target_Value"})

        actual = load_fn(config)

        pdt.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "desired_columns",
        [
            pytest.param(["column1", "column2", "not_in_file"], id="one unseen columns"),
            pytest.param(["column1", "column2", "not_in_file", "another_unseen"], id="multiple unseen columns"),
            pytest.param(["not_in_file"], id="only missing feature"),
            pytest.param(["column1", "not_in_file"], id="one present, one missing feature"),
        ],
    )
    @pytest.mark.parametrize(
        "log_level,debug_present,warning_present",
        [
            pytest.param(logging.WARNING, False, True, id="warning only"),
            pytest.param(logging.DEBUG, True, True, id="debug has both"),
        ],
    )
    def test_column_mismatch_logs_warning(
        self, log_level, debug_present, warning_present, desired_columns, setup_fn, load_fn, caplog
    ):
        config = setup_fn()
        config.features = desired_columns

        with caplog.at_level(log_level, logger="seismometer"):
            _ = load_fn(config)

        assert ("Not all requested columns are present" in caplog.text) == warning_present
        assert ("Requested columns are" in caplog.text) == debug_present
        assert ("Columns present are" in caplog.text) == debug_present


class TestAssumedTypes:
    @pytest.mark.parametrize(
        "time_col",
        [
            pytest.param("Time", id="exact"),
            pytest.param("~~Time~~", id="match middle"),
        ],
    )
    def test_assumed_types_convert_times(self, time_col):
        config = Mock(spec=ConfigProvider)
        config.output_list = []

        dataframe = pd.DataFrame({time_col: np.datetime64("2022-01-01 13:27:56") + (np.arange(5) * 100_000)})

        expected = dataframe.copy()
        expected[time_col] = pd.to_datetime(expected[time_col]).astype("<M8[ns]")

        actual = undertest.assumed_types(config, dataframe)
        pdt.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "target_values,expected_values",
        [
            pytest.param([1, 50, 99], [0.01, 0.50, 0.99], id="reduces from percentage"),
            pytest.param([0.02, 0.5, 0.79], [0.02, 0.5, 0.79], id="no change from probability"),
        ],
    )
    def test_assumed_types_convert_scores(self, target_values, expected_values):
        config = Mock(spec=ConfigProvider)
        config.output_list = ["target1"]

        percentage_like = [1, 50, 99]
        proba_like = [0.01, 0.50, 0.99]
        dataframe = pd.DataFrame(
            {"target1": target_values, "nottarget_big": percentage_like, "nottarget_small": proba_like}
        )

        expected = dataframe.copy()
        expected["target1"] = expected_values

        actual = undertest.assumed_types(config, dataframe)
        pdt.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "input_series",
        [
            pytest.param(pd.Series([1.0, 2.0, 3.0], dtype="Float64"), id="float64"),
            pytest.param(pd.Series([1, np.nan, 2]), id="nullable inferred"),
            pytest.param(pd.Series([1.0, 2.0, 3.0], dtype=np.float32), id="numpy"),
            pytest.param(pd.Series([1.0, 2.0, 3.0], dtype="float"), id="generic"),
        ],
    )
    def test_assumed_types_avoids_pandasFloat(self, input_series):
        config = Mock(spec=ConfigProvider)
        config.output_list = []

        dataframe = pd.DataFrame({"number": input_series})

        expected = dataframe.copy()
        expected["number"] = expected["number"].astype(np.float64)

        actual = undertest.assumed_types(config, dataframe)
        pdt.assert_frame_equal(actual, expected)


class TestDictionaryTypes:
    @pytest.mark.parametrize(
        "defined_types,expected_dtypes",
        [
            pytest.param(
                [{"name": "downcast", "dtype": "float16"}, {"name": "hardint", "dtype": "int"}],
                {"downcast": "float16", "assumedTime": "datetime64[ns]", "keep": "int64", "hardint": "int"},
                id="basic types",
            )
        ],
    )
    def test_defined_type_conversions_succeed(self, defined_types, expected_dtypes):
        config = Mock(spec=ConfigProvider)
        config.output_list = []

        config.prediction_defs = Mock()
        config.prediction_defs.predictions = [DictionaryItem(**di) for di in defined_types]

        dataframe = pd.DataFrame(
            {
                "downcast": [1.0, 2.0, 3.0],
                "assumedTime": ["2022-01-01", "2022-01-02", "2022-01-03"],
                "keep": [4, 5, 6],
            }
        )
        # avoid pandas inferring
        dataframe["hardint"] = pd.Series(["1.0", "0.0", "3.0"], dtype="string")

        actual_frame = undertest.dictionary_types(config, dataframe)

        for k, v in expected_dtypes.items():
            assert actual_frame[k].dtype == v

    def test_value_error_raises_configuration_error(
        self,
    ):
        config = Mock(spec=ConfigProvider)
        config.output_list = []
        config.prediction_defs = Mock()
        config.prediction_defs.predictions = [
            DictionaryItem(name="bad_col1", dtype="int64"),
            DictionaryItem(name="bad_col2", dtype="datetime64[ns]"),
        ]

        dataframe = pd.DataFrame({"bad_col1": ["a", "b", "c"], "bad_col2": ["a", "b", "c"], "keep_col": [4, 5, 6]})

        with pytest.raises(undertest.ConfigurationError, match="Could not convert") as cerr:
            undertest.dictionary_types(config, dataframe)

        # Check all bad columns are mentioned in the error message
        assert "bad_col1" in str(cerr.value)
        assert "bad_col2" in str(cerr.value)
        assert "keep_col" not in str(cerr.value)


# endregion
