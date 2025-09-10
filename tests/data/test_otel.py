from seismometer.data.otel import RealExportManager


class TestExportManager:
    def test_class_activate(self):
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter  # noqa: F401

            r = RealExportManager(hostname="", file_output_paths=[], export_ports=[], dump_to_stdout=True)
            r.activate_exports()
            assert r.active
        except ModuleNotFoundError:
            assert True  # No point in testing the export manager if we can't export

    def test_class_deactivate(self):
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter  # noqa: F401

            r = RealExportManager(hostname="", file_output_paths=[], export_ports=[], dump_to_stdout=True)
            r.deactivate_exports()
            assert not r.active
        except ModuleNotFoundError:
            assert True
