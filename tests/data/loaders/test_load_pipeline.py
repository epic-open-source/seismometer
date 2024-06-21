from unittest.mock import Mock

import pandas as pd
import pytest

import seismometer.data.loader.pipeline as undertest


def test_passthru_framehook():
    pass_var = "dataframe"
    assert pass_var == undertest._passthru_framehook("config", pass_var)


def test_passthru_mergehook():
    pass_var = "dataframe"
    assert pass_var == undertest._passthru_mergehook("config", "other", pass_var)


class TestSeismogramLoader:
    @pytest.mark.parametrize(
        "attr_name,expected",
        [
            # Constructor exposed
            ("config", "CONFIGPROVIDER"),
            ("prediction_fn", None),
            ("event_fn", None),
            ("post_predict_fn", undertest._passthru_framehook),
            ("post_event_fn", undertest._passthru_framehook),
            ("merge_fn", undertest._passthru_mergehook),
            # Internal passthru_frame
            ("prediction_from_memory", undertest._passthru_framehook),
            ("event_from_memory", undertest._passthru_framehook),
        ],
    )
    def test_loader_defaults(self, attr_name, expected):
        config = "CONFIGPROVIDER"
        actualLoader = undertest.SeismogramLoader(config)

        assert getattr(actualLoader, attr_name) == expected

    @pytest.mark.parametrize(
        "input_kwargs",
        [
            {"prediction_fn": "PREDICTION_FN"},
            {"event_fn": "EVENT_FN"},
            {"post_predict_fn": "POST_PREDICT_FN"},
            {"post_event_fn": "POST_EVENT_FN"},
            {"merge_fn": "MERGE_FN"},
        ],
    )
    def test_loader_kwarg_assignment(self, input_kwargs):
        config = "CONFIGPROVIDER"
        attr_name = list(input_kwargs)[0]
        expected = input_kwargs[attr_name]

        actualLoader = undertest.SeismogramLoader(config, **input_kwargs)

        assert getattr(actualLoader, attr_name) == expected

    def test_default_behavior_can_execute(self, fake_config):
        # Default allows single frame passthough
        input = "INPUT"
        loader = undertest.SeismogramLoader(fake_config)

        output = loader.load_data(input)

        assert input == output

    def test_loading_can_log_location(self, fake_config, caplog):
        # Default allows single frame passthough
        input = "INPUT"

        loader = undertest.SeismogramLoader(fake_config)
        with caplog.at_level("INFO"):
            _ = loader.load_data(input)

        assert "Importing" in caplog.text

    @pytest.mark.parametrize(
        "input,expected",
        [
            pytest.param(None, "PREDICTION", id="prediction function is called"),
            pytest.param("INPUT", "INPUT", id="inmemory is prioritized"),
        ],
    )
    def test_load_predictions_use_expected_fn(self, input, expected, fake_config):
        def prediction_fn(config):
            return "PREDICTION"

        loader = undertest.SeismogramLoader(fake_config, prediction_fn)

        actual = loader.load_data(input)

        assert actual == expected

    @pytest.mark.parametrize(
        "input,expected",
        [
            pytest.param(None, "EVENT", id="prediction function is called"),
            pytest.param(pd.DataFrame(data={"name": ["INPUT"]}), "INPUT", id="inmemory is prioritized"),
        ],
    )
    def test_load_events_use_expected_fn(self, input, expected, fake_config):
        def event_fn(config):
            return pd.DataFrame({"name": ["EVENT"]})

        def merge_fn(config, event_frame, dataframe):
            return event_frame

        loader = undertest.SeismogramLoader(fake_config, event_fn=event_fn, merge_fn=merge_fn)

        # Transform dataframe to a comparable string
        actual = loader.load_data(None, input).iloc[0, 0]

        assert actual == expected


@pytest.fixture
def fake_config():
    fake_config = Mock(spec=undertest.ConfigProvider)
    fake_config.config_dir = "TestDir"

    return fake_config
