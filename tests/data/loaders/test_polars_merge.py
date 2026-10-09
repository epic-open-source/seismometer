import logging
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import seismometer.data.loader.event as pandas_event
from seismometer.configuration.model import Event

pytest.importorskip("polars")
from seismometer.data.loader import polars_merge as undertest  # noqa: E402

BASE = pd.Timestamp("2024-01-01")


def fake_config(events, event_dtypes=None, keys=("Id", "Ctx")):
    return SimpleNamespace(
        entity_keys=list(keys),
        predict_time="PredictTime",
        events={event.display_name: event for event in events},
        event_defs={name: SimpleNamespace(dtype=dtype) for name, dtype in (event_dtypes or {}).items()},
    )


def frames(seed=0, n_ids=40, n_preds=400, n_events=600):
    rng = np.random.default_rng(seed)
    ids = np.array([f"P{i:03d}" for i in range(n_ids)], dtype=object)
    predictions = pd.DataFrame(
        {
            "Id": ids[rng.integers(0, n_ids, n_preds)],
            "Ctx": rng.integers(0, 3, n_preds),
            # hourly times so duplicate key + time pairs and ties with events occur
            "PredictTime": BASE + pd.to_timedelta(rng.integers(0, 24 * 10, n_preds), unit="h"),
            "Score": rng.random(n_preds),
        }
    )
    predictions.loc[rng.integers(0, n_preds, 5), "PredictTime"] = pd.NaT
    events = pd.DataFrame(
        {
            "Id": ids[rng.integers(0, n_ids, n_events)],
            "Ctx": rng.integers(0, 3, n_events),
            "Type": rng.choice(["A", "B", "C", "AB"], n_events),
            "Time": BASE + pd.to_timedelta(rng.integers(0, 24 * 12, n_events), unit="h"),
            "Value": rng.integers(0, 3, n_events).astype(float),
        }
    )
    events.loc[rng.integers(0, n_events, 10), "Time"] = pd.NaT
    events.loc[rng.integers(0, n_events, 10), "Value"] = np.nan
    return predictions, events


EVENTS = [
    Event(source="A", display_name="Windowed", window_hr=12, offset_hr=2),
    Event(source=["B", "C"], display_name="Two sources", window_hr=48),
    Event(source="A", display_name="Offset without window", offset_hr=5),
    Event(source="B", display_name="Zero window", window_hr=0),
    Event(source="C", display_name="AB", window_hr=24, impute_val=2),  # display name is also an event type
    Event(source="Missing", display_name="No rows", window_hr=24),
]


def assert_same_merge(config, predictions, events):
    expected = pandas_event.merge_onto_predictions(config, events.copy(), predictions.copy())
    with patch.object(undertest.pandas_event, "merge_onto_predictions", side_effect=AssertionError("fell back")):
        actual = undertest.merge_onto_predictions(config, events.copy(), predictions.copy())
    pdt.assert_frame_equal(actual, expected)


class TestPolarsMerge:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_pandas_merge(self, seed):
        predictions, events = frames(seed)
        assert_same_merge(fake_config(EVENTS), predictions, events)

    def test_matches_with_event_dtypes(self):
        predictions, events = frames()
        events["Value"] = events["Value"].fillna(1)
        config = fake_config(EVENTS[:2], event_dtypes={"A": "int", "B": "float"})
        assert_same_merge(config, predictions, events)

    @pytest.mark.parametrize("values", [[1, 0], [True, False]], ids=["int", "bool"])
    def test_matches_with_unmatched_rows_upcast(self, values):
        predictions, events = frames()
        events["Value"] = np.resize(values, len(events))
        assert_same_merge(fake_config(EVENTS[:2]), predictions, events)

    def test_matches_with_categorical_and_string_keys(self):
        predictions, events = frames()
        predictions["Id"] = predictions["Id"].astype("category")
        events["Id"] = events["Id"].astype("string")
        assert_same_merge(fake_config(EVENTS), predictions, events)

    def test_matches_with_single_key(self):
        predictions, events = frames()
        assert_same_merge(fake_config(EVENTS, keys=["Id"]), predictions, events)

    def test_does_not_modify_inputs(self):
        predictions, events = frames()
        before_predictions, before_events = predictions.copy(), events.copy()
        _ = undertest.merge_onto_predictions(fake_config(EVENTS), events, predictions)
        pdt.assert_frame_equal(predictions, before_predictions)
        pdt.assert_frame_equal(events, before_events)


def use_nearest(predictions, events, config):
    config.events["Windowed"].merge_strategy = "nearest"


def null_key(predictions, events, config):
    predictions.loc[0, "Id"] = None


def mismatched_key_kinds(predictions, events, config):
    events["Ctx"] = events["Ctx"].astype(str)


def no_event_times(predictions, events, config):
    events.loc[events["Type"] == "A", "Time"] = pd.NaT


def existing_event_column(predictions, events, config):
    predictions["Windowed_Value"] = 0.0


class TestFallsBackToPandas:
    @pytest.mark.parametrize(
        "change,reason",
        [
            (use_nearest, "nearest"),
            (null_key, "missing"),
            (mismatched_key_kinds, "differ"),
            (no_event_times, "no event times"),
            (existing_event_column, "already have"),
        ],
    )
    def test_falls_back(self, change, reason, caplog):
        predictions, events = frames()
        config = fake_config([event.model_copy() for event in EVENTS])
        change(predictions, events, config)

        with caplog.at_level(logging.INFO, logger="seismometer"):
            try:
                expected = pandas_event.merge_onto_predictions(config, events.copy(), predictions.copy())
            except Exception as exc:  # the pandas merge rejects some of these; the fallback must do the same
                with pytest.raises(type(exc)):
                    undertest.merge_onto_predictions(config, events.copy(), predictions.copy())
            else:
                actual = undertest.merge_onto_predictions(config, events.copy(), predictions.copy())
                pdt.assert_frame_equal(actual, expected)

        assert "merging events with pandas" in caplog.text
        assert reason in caplog.text
