from pathlib import Path
from typing import Optional
from unittest.mock import Mock

import pytest

import seismometer.data.loader as undertest
from seismometer.data.loader.pipeline import _passthru_framehook


def get_fake_config(prediction_path: Optional[str] = "predict.parquet", event_path: Optional[str] = "event.parquet"):
    fake_config = Mock(spec=undertest.ConfigProvider)
    fake_config.prediction_path = Path(prediction_path)
    fake_config.event_path = Path(event_path)

    return fake_config


class TestLoaderFactory:
    def test_loader_factory_sets_config(self):
        fake_config = "CONFIGPROVIDER"
        actualLoader = undertest.loader_factory(fake_config)

        assert actualLoader.config == fake_config

    @pytest.mark.parametrize(
        "attr_name,expected",
        [
            # Constructor exposed
            ("prediction_fn", undertest.prediction.parquet_loader),
            ("event_fn", undertest.event.parquet_loader),
            ("post_predict_fn", undertest.prediction.assumed_types),
            ("post_event_fn", undertest.event.post_transform_fn),
            ("merge_fn", undertest.event.merge_onto_predictions),
            # Internal passthru_frame
            ("prediction_from_memory", _passthru_framehook),
            ("event_from_memory", _passthru_framehook),
        ],
    )
    def test_parquet_loader_functions(self, attr_name, expected):
        actualLoader = undertest.loader_factory(get_fake_config())

        assert getattr(actualLoader, attr_name) == expected
