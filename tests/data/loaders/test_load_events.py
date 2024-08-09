import logging
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import seismometer.data.loader.event as undertest
from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import Event


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

        # Aid setting a list of events (like that from config) but getting a dict like ConfigProvider returns
        @property
        def events(self):
            return self._events

        @events.setter
        def events(self, event_list: list[Event]):
            self._events = {event.display_name: event for event in event_list}

    return FakeConfigProvider()


EVENT_DATES = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]


def event_frame(rename=False):
    # Create a mock predictions dataframe
    df = pd.DataFrame(
        {
            "id": ["0", "0", "1", "1", "2", "2"],
            "origType": ["a", "b", "a", "a", "b", "a"],
            "origTime": EVENT_DATES * 3,
            "origValue": [7, 8, 9, 10, 11, 12.0],
        }
    )
    if rename:
        df.columns = ["id", "Type", "Time", "Value"]
    return df.sort_index(axis=1)


# endregion
# region File-type setup functions
def parquet_setup():
    # Expects current directory to have been modified by the testcase,
    # such as using tmp_as_current
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

        with caplog.at_level(logging.DEBUG, logger="seismometer"):
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


class TestMergeOntoPredictions:
    def test_merge_event(self):
        config = fake_config("unused_file")
        config.events = [Event(source="a")]

        event_df = event_frame(rename=True)
        predictions = pd.DataFrame(
            {
                "id": ["0", "1", "2"],
                # push backwards so all events are in the future
                "Time": [EVENT_DATES[0] - pd.to_timedelta(10, unit="D")] * 3,
                "Prediction": [4, 5, 6],
            }
        )

        expected = predictions.copy()
        expected[["a_Value", "a_Time"]] = event_df[["Value", "Time"]].values[[0, 2, 5]]
        expected["a_Value"] = expected["a_Value"].astype(float)

        actual = undertest.merge_onto_predictions(config, event_df, predictions)

        pdt.assert_frame_equal(actual, expected)

    def test_merge_event_two_sources(self):
        config = fake_config("unused_file")
        config.events = [Event(source=["a", "c"], display_name="ac")]

        event_df = event_frame(rename=True)
        # Types : ["a", "b", "a", "a", "b", "a"]
        event_df["Type"] = ["a", "b", "c", "c", "b", "c"]

        predictions = pd.DataFrame(
            {
                "id": ["0", "1", "2"],
                # push backwards so all events are in the future
                "Time": [EVENT_DATES[0] - pd.to_timedelta(10, unit="D")] * 3,
                "Prediction": [4, 5, 6],
            }
        )

        expected = predictions.copy()
        expected[["ac_Value", "ac_Time"]] = event_df[["Value", "Time"]].values[[0, 2, 5]]
        expected["ac_Value"] = expected["ac_Value"].astype(float)

        actual = undertest.merge_onto_predictions(config, event_df, predictions)

        pdt.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "event,str_inclusions",
        [
            (Event(source="a"), ["Merging event a"]),
            (Event(source="a", window_hr=1), ["Windowing event a", "lookback 1", "offset by 0"]),
            pytest.param(Event(source="a", offset_hr=6), ["Merging event a"], id="only offset, doesnt window"),
            (Event(source="a", window_hr=2, offset_hr=12), ["Windowing event a", "lookback 2", "offset by 12"]),
        ],
    )
    def test_merge_info_logged(self, event, str_inclusions, caplog):
        config = fake_config("unused_file")
        config.events = [event]

        event_df = event_frame(rename=True)
        predictions = pd.DataFrame(
            {
                "id": ["0", "1", "2"],
                # push inbetween two events
                "Time": [EVENT_DATES[0] + pd.to_timedelta(12, unit="h")] * 3,
                "Prediction": [4, 5, 6],
            }
        )

        with caplog.at_level(logging.DEBUG, logger="seismometer"):
            _ = undertest.merge_onto_predictions(config, event_df, predictions)

        for pattern in str_inclusions:
            assert pattern in caplog.text

    def test_imputation_on_target(self, caplog):
        config = fake_config("unused_file")
        config.events = [Event(source="a", impute_val=10)]

        event_df = event_frame(rename=True)
        event_df = event_df[event_df["id"] == "0"]

        predictions = pd.DataFrame(
            {
                "id": ["0", "1", "2"],
                # push backwards so all events are in the future
                "Time": [EVENT_DATES[0] - pd.to_timedelta(10, unit="D")] * 3,
                "Prediction": [4, 5, 6],
            }
        )

        with caplog.at_level(logging.WARNING, logger="seismometer"):
            _ = undertest.merge_onto_predictions(config, event_df, predictions)

        assert "Event a specified impute" in caplog.text
