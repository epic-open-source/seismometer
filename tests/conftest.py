import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

from pytest import fixture

from seismometer.data import metric_apis, otel

TEST_ROOT = Path(__file__).parent


@fixture
def res():
    return TEST_ROOT / "resources"


@fixture
def tmp_as_current(tmp_path):
    with working_dir_as(tmp_path):
        yield tmp_path


@contextmanager
def working_dir_as(path: Path) -> Generator:
    """
    Temporarily changes the current working directory
    Useful for testing when the model root is assumed

    Parameters
    ----------
    path : Path
        Directory to treat as working directory
    """
    oldpath = Path().absolute()

    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(oldpath)


@fixture(scope="module", autouse=True)
def sg_decorator_mock():
    with patch("seismometer.core.autometrics._store_call_parameters"):
        yield


@fixture(scope="module", autouse=True)
def export_manager_mock():
    mock_export_manager = Mock()
    mock_export_manager.active = True
    otel.export_manager = mock_export_manager
    yield


@fixture(scope="module", autouse=True)
def set_datapoint_mock():
    with patch.object(metric_apis.RealOpenTelemetryRecorder, "_set_one_datapoint"):
        yield
