from seismometer.data.otel import RealMetricTelemetryManager, TelemetryConfig


class TestMetricTelemetryManager:
    def test_class_activate(self):
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter  # noqa: F401

            config = TelemetryConfig({"otel_export": {"stdout": True}})
            r = RealMetricTelemetryManager(config)
            r.activate_exports()
            assert r.active
        except ModuleNotFoundError:
            assert True  # No point in testing the export manager if we can't export

    def test_class_deactivate(self):
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter  # noqa: F401

            config = TelemetryConfig({"otel_export": {"stdout": True}})
            r = RealMetricTelemetryManager(config)
            r.deactivate_exports()
            assert not r.active
        except ModuleNotFoundError:
            assert True
