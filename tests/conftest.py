import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, Mock, patch

from pytest import fixture

from seismometer.data import metric_apis

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
    class ExportManagerMock:
        def __init__(self, *args, **kwargs):
            self.active = True
            self.meter_provider = Mock()

    with patch("seismometer.data.otel.RealExportManager", ExportManagerMock):
        yield


@fixture(scope="module", autouse=True)
def set_datapoint_mock():
    with patch.object(metric_apis.RealOpenTelemetryRecorder, "_set_one_datapoint"):
        yield


@fixture(scope="session", autouse=True)
def patch_opentelemetry_modules():
    with patch("opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter", new=MagicMock()), patch(
        "opentelemetry.metrics.Meter", new=MagicMock()
    ), patch("opentelemetry.metrics.set_meter_provider", new=MagicMock()), patch(
        "opentelemetry.sdk.metrics.MeterProvider", new=MagicMock()
    ), patch(
        "opentelemetry.sdk.metrics.export.ConsoleMetricExporter", new=MagicMock()
    ), patch(
        "opentelemetry.sdk.metrics.export.PeriodicExportingMetricReader", new=MagicMock()
    ), patch(
        "opentelemetry.sdk.resources.SERVICE_NAME", new="mocked_service"
    ), patch(
        "opentelemetry.sdk.resources.Resource", new=MagicMock()
    ):
        yield
