from collections import defaultdict
from unittest.mock import patch

import pytest

try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.sdk.metrics import MeterProvider
except ImportError:
    # No OTel!
    pytest.skip("No OpenTelemetry, nothing to test here", allow_module_level=True)

from seismometer.configuration.metrics import MetricConfig, SingleMetricConfig
from seismometer.core.autometrics import AutomationManager
from seismometer.data import telemetry
from seismometer.data.otel import MetricTelemetryManager, TelemetryConfig

# We will mock it so that all metrics that are "logged"
# actually just end up going here.
RECEIVED_METRICS = defaultdict(list)


def mock_set_one_datapoint(_, attributes, instrument, data):
    RECEIVED_METRICS[instrument].append({"attributes": attributes, "value": data})


@pytest.fixture
def recorder(tmp_path):
    RECEIVED_METRICS.clear()
    MetricTelemetryManager.kill()
    MetricTelemetryManager(
        TelemetryConfig({"otel_export": {"files": [tmp_path / "test_metrics.txt"], "stdout": True}})
    )
    AutomationManager.kill()
    AutomationManager(fake_config(tmp_path))
    r = telemetry.OpenTelemetryRecorder(metric_names=[], name="Test")
    r.instruments = {"A": "A", "B": "B", "C": "C", "D": "D"}
    r.metric_names = ["A", "B", "C", "D"]
    return r


def fake_config(tmp_path):
    # Create a fake configuration object
    class FakeConfigProvider:
        def __init__(self):
            self.automation_config = {}
            self.automation_config_path = tmp_path / "automation_config.yml"
            self.metric_config = MetricConfig(
                metric_configs={
                    "A": SingleMetricConfig(log_all=True),
                    "B": SingleMetricConfig(log_all=False),
                }
            )

    return FakeConfigProvider()


# We want recorder and exporter init methods to do nothing, so we don't actually have to
# open any files or make anything.
@patch.object(telemetry.OpenTelemetryRecorder, "__init__", new=lambda self: None)
@patch.object(telemetry.RealOpenTelemetryRecorder, "_set_one_datapoint", new=mock_set_one_datapoint)
class TestMetricLogging:
    def test_log_to_instrument(self, recorder):
        """We're just testing some basic logging outside of the context of any widget."""
        recorder._log_to_instrument(attributes={"test?": True}, instrument="Test Instrument", data=4354.6444)
        assert {"attributes": {"test?": True}, "value": 4354.6444} in RECEIVED_METRICS["Test Instrument"]

    def test_populate_metrics(self, recorder):
        """Now, try the populate_metrics method to make sure it works as expected."""
        recorder.populate_metrics(attributes={"test?": True}, metrics={"A": 1})
        recorder.populate_metrics(attributes={"bar?": "foo"}, metrics={"B": 3, "C": 4})
        assert {"attributes": {"test?": True}, "value": 1} in RECEIVED_METRICS["A"]
        assert {"attributes": {"bar?": "foo"}, "value": 3} in RECEIVED_METRICS["B"]
        assert {"attributes": {"bar?": "foo"}, "value": 4} in RECEIVED_METRICS["C"]


@pytest.fixture(scope="session", autouse=True)
def otel_metrics_cleanup():
    yield
    # Cleanup ... close the meter provider so that we don't erroneously try to log late
    provider = otel_metrics.get_meter_provider()
    if isinstance(provider, MeterProvider):
        provider.shutdown()
