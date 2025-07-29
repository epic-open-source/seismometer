from unittest.mock import MagicMock, patch

from seismometer.data.otel import RealExportManager


@patch("seismometer.data.otel.Resource", new=MagicMock())
@patch("seismometer.data.otel.SERVICE_NAME", new="mocked_service")
@patch("seismometer.data.otel.PeriodicExportingMetricReader", new=MagicMock())
@patch("seismometer.data.otel.ConsoleMetricExporter", new=MagicMock())
@patch("seismometer.data.otel.MeterProvider", new=MagicMock())
@patch("seismometer.data.otel.set_meter_provider", new=MagicMock())
@patch("seismometer.data.otel.Meter", new=MagicMock())
@patch("seismometer.data.otel.OTLPMetricExporter", new=MagicMock())
class TestExportManager:
    def test_class_activate(self):
        r = RealExportManager(hostname="", file_output_paths=[], export_ports=[], dump_to_stdout=True)
        r.activate_exports()
        assert r.active

    def test_class_deactivate(self):
        r = RealExportManager(hostname="", file_output_paths=[], export_ports=[], dump_to_stdout=True)
        r.deactivate_exports()
        assert not r.active
