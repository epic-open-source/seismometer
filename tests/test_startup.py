import logging
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from seismometer import run_startup
from seismometer.configuration import ConfigProvider
from seismometer.seismogram import Seismogram


def fake_load_config(self, *args):
    mock_config = Mock(autospec=ConfigProvider)
    mock_config.output_dir.return_value
    self.config = mock_config

    self.template = "TestTemplate"


# TODO: update this to create testing Loader and have factory return it
def fake_load_data(self, *args):
    self.dataframe = pd.DataFrame()


@pytest.fixture
def fake_seismo(tmp_path):
    with patch.object(Seismogram, "load_data", fake_load_data):
        with patch.object(Seismogram, "load_config", fake_load_config):
            Seismogram(config_path=tmp_path / "config", output_path=tmp_path / "output")
        yield
    Seismogram.kill()


class TestStartup:
    def test_debug_logs_with_formatter(self, fake_seismo, tmp_path, capsys):
        expected_date_str = "[" + datetime.now().strftime("%Y-%m-%d")

        run_startup(config_path=tmp_path / "new_config", log_level=logging.DEBUG)

        sterr = capsys.readouterr().err
        assert sterr.startswith(expected_date_str)

    def test_seismo_killed_between_tests(self, fake_seismo, tmp_path):
        sg = Seismogram()
        assert sg.config_path == (tmp_path / "config")

    @pytest.mark.parametrize("log_level", [30, 40, 50])
    def test_logger_initialized(self, log_level, fake_seismo, tmp_path):
        run_startup(config_path=tmp_path / "new_config", log_level=log_level)

        logger = logging.getLogger("seismometer")
        assert logger.getEffectiveLevel() == log_level
