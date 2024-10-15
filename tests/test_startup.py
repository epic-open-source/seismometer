import logging
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

import seismometer
import seismometer.configuration
from seismometer import run_startup
from seismometer.configuration import ConfigProvider
from seismometer.seismogram import Seismogram


@pytest.fixture
def mock_config(tmp_path):
    mock = Mock(autospec=ConfigProvider)
    mock.config_path = tmp_path / "config"

    with patch.object(seismometer, "ConfigProvider", new=mock):
        yield mock


def fake_data_loader(*args):
    return "LOADER"


@pytest.fixture
def fake_seismo(tmp_path):
    old_level = logging.getLogger("seismometer").getEffectiveLevel()

    with patch.object(Seismogram, "copy_config_metadata"), patch.object(Seismogram, "load_data"):
        yield
    Seismogram.kill()

    # DO NOT delete logger: it unsynchs the module objects with test setup (and wouldn't happen during a real session)
    logging.getLogger("seismometer").setLevel(old_level)


@pytest.mark.usefixtures("fake_seismo")
@pytest.mark.usefixtures("mock_config")
@patch.object(seismometer.data.loader, "loader_factory", new=fake_data_loader)
class TestStartup:
    def test_debug_logs_with_formatter(self, capsys):
        expected_date_str = "[" + datetime.now().strftime("%Y-%m-%d")

        run_startup(log_level=logging.DEBUG)

        sterr = capsys.readouterr().err
        assert sterr.startswith(expected_date_str)

    def test_seismo_killed_between_tests(self):
        sg = Seismogram("bad_config", "bad_loader")
        assert sg.config == "bad_config"

    @pytest.mark.parametrize("log_level", [30, 40, 50])
    def test_logger_initialized(self, log_level):
        run_startup(config_path="new_config", log_level=log_level)

        logger = logging.getLogger("seismometer")
        assert logger.getEffectiveLevel() == log_level
