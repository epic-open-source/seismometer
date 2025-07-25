from collections import defaultdict
from unittest.mock import patch

import pandas as pd
import pytest

try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.sdk.metrics import MeterProvider
except ImportError:
    # No OTel!
    pytest.skip("No OpenTelemetry, nothing to test here", allow_module_level=True)

from seismometer.core import autometrics
from seismometer.data import metric_apis, otel

# We will mock it so that all metrics that are "logged"
# actually just end up going here.
RECEIVED_METRICS = defaultdict(list)


def mock_set_one_datapoint(_, attributes, instrument, data):
    RECEIVED_METRICS[instrument].append({"attributes": attributes, "value": data})


@pytest.fixture
def recorder():
    r = metric_apis.OpenTelemetryRecorder(metric_names=[], name="Test")
    r.instruments = {"A": "A", "B": "B", "C": "C", "D": "D"}
    r.metric_names = ["A", "B", "C", "D"]
    return r


def mock_get_metric_config(_, metric_name):
    return {
        "log_all": metric_name == "A",
        "output_metrics": True,
    }  # We are logging all for specifically the metric "A"


# We want recorder and exporter init methods to do nothing, so we don't actually have to
# open any files or make anything.
@patch.object(metric_apis.OpenTelemetryRecorder, "__init__", new=lambda self: None)
@patch.object(otel.ExportManager, "__init__", new=lambda self: None)
@patch.object(metric_apis.RealOpenTelemetryRecorder, "_set_one_datapoint", new=mock_set_one_datapoint)
@patch.object(autometrics.AutomationManager, "get_metric_config", new=mock_get_metric_config)
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

    def test_log_by_cohort(self, recorder):
        """Finally, we test the most complicated method in both fashions: intersecting and not."""
        dataframe = pd.DataFrame(
            {
                "Name": ["U", "V", "W", "X", "Y", "Z"],
                "Age": [10, 10, 20, 20, 30, 30],
                "Birth Season": ["Spring", "Summer", "Spring", "Summer", "Fall", "Winter"],
                "A": [1, 2, 3, 4, 5, 6],  # What we ostensibly want to log.
            }
        )
        cohorts = {"Age": [10, 20], "Birth Season": ["Spring", "Summer"]}
        recorder.log_by_cohort(
            dataframe=dataframe, base_attributes={"foo": "bar"}, cohorts=cohorts, intersecting=False
        )
        assert {"attributes": {"foo": "bar", "Age": 10, "key": 1}, "value": 2} in RECEIVED_METRICS["A"]
        recorder.log_by_cohort(dataframe=dataframe, base_attributes={"foo": "bar"}, cohorts=cohorts, intersecting=True)
        assert {
            "attributes": {"foo": "bar", "Age": 10, "Birth Season": "Spring", "key": 0},
            "value": 1,
        } in RECEIVED_METRICS["A"]

    def test_log_by_column(self, recorder):
        dataframe = pd.DataFrame(
            {
                "Threshold": [1, 2, 3, 4, 5, 6],
                "Cohort": ["Cohort A", "Cohort B"] * 3,
                "A": [10, 20, 30, 40, 50, 60],
                "B": [100, 200, 300, 400, 500, 600],
            }
        )
        cohorts = {"Cohort": ["Cohort A"]}
        recorder.log_by_column(
            dataframe, col_name="Threshold", cohorts=cohorts, base_attributes={"foo": "bar"}, col_values=[1, 4]
        )
        assert {10, 30, 50} == (
            set([datapoint["value"] for datapoint in RECEIVED_METRICS["A"]])
        )  # We expect all logging here ...
        assert {100} == set([datapoint["value"] for datapoint in RECEIVED_METRICS["B"]])  # ... but not here.


@pytest.fixture(autouse=True)
def clear_otel_metrics():
    RECEIVED_METRICS.clear()


@pytest.fixture(scope="session", autouse=True)
def otel_metrics_cleanup():
    yield
    # Cleanup ... close the meter provider so that we don't erroneously try to log late
    provider = otel_metrics.get_meter_provider()
    if isinstance(provider, MeterProvider):
        provider.shutdown()
