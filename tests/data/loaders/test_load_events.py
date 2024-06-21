import logging
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import seismometer.data.loader.event as undertest
from seismometer.configuration import ConfigProvider


# region Fakes and Data Prep
def fake_config(event_file):
    # Create a fake configuration object
    class FakeConfigProvider:
        def __init__(self):
            self.entity_keys = ["id"]
            self.predict_time = "Time"
            self.event_path = event_file
            self.target = "target"

            self.ev_type = "origType"
            self.ev_time = "origTime"
            self.ev_value = "origValue"

    return FakeConfigProvider()


def event_frame(rename=False):
    # Create a mock predictions dataframe
    df = pd.DataFrame(
        {
            "id": ["0", "1", "2"],
            "origType": [1, 2, 3],
            "origTime": [4, 5, 6],
            "origValue": [7, 8, 9],
        }
    )
    if rename:
        df.columns = ["id", "Type", "Time", "Value"]
    return df.sort_index(axis=1)


# endregion
# region File-type setup functions
def parquet_setup():
    file = Path("events.parquet")

    data = event_frame()
    data.to_parquet(file)

    return fake_config(file)


# region Tests
@pytest.mark.parametrize("setup_fn,load_fn", [[parquet_setup, undertest.parquet_loader]])
@pytest.mark.usefixtures("tmp_as_current")
class TestPredictionLoad:
    def test_nofile_warns_and_returns_empty(self, setup_fn, load_fn, caplog):
        config = setup_fn()
        config.event_path = "not_a_file.parquet"

        with caplog.at_level(logging.DEBUG):
            actual = load_fn(config)
        print(caplog.text)
        assert "No events found" in caplog.text
        assert actual.empty
        assert set(actual.columns) == set(["id", "Type", "Time", "Value"])

    def test_load_remaps_columns(self, setup_fn, load_fn):
        config = setup_fn()
        expected = event_frame(1)

        actual = load_fn(config).sort_index(axis=1)

        pdt.assert_frame_equal(actual, expected)


class TestPostTransformFn:
    def test_assumed_types_convert_times(self):
        time_col = "Time"
        config = Mock(spec=ConfigProvider)

        dataframe = pd.DataFrame({time_col: np.datetime64("2022-01-01 13:27:56") + (np.arange(5) * 100_000)})

        expected = dataframe.copy()
        expected[time_col] = pd.to_datetime(expected[time_col]).astype("<M8[ns]")

        actual = undertest.post_transform_fn(config, dataframe)
        pdt.assert_frame_equal(actual, expected)
