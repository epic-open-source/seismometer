from collections import defaultdict
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
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


class TestRecordDataframeMatrix:
    """Tests for record_dataframe_matrix function."""

    @patch("seismometer.data.telemetry.OpenTelemetryRecorder")
    def test_basic_single_index(self, mock_recorder_class):
        """Test recording from a DataFrame with single index."""
        # Arrange
        mock_recorder = Mock()
        mock_recorder_class.return_value = mock_recorder

        df = pd.DataFrame({"Low": [10, 20], "Medium": [15, 25], "High": [5, 10]}, index=["GroupA", "GroupB"])
        df.index.name = "cohort"

        # Act
        telemetry.record_dataframe_matrix(df, "response_count")

        # Assert
        mock_recorder_class.assert_called_once_with(metric_names=["response_count"], name="DataFrameCounts")

        # Verify all 6 recordings (2 rows × 3 columns)
        assert mock_recorder.populate_metrics.call_count == 6

        # Check specific calls
        expected_calls = [
            call(attributes={"cohort": "GroupA", "score": "Low"}, metrics={"response_count": 10}),
            call(attributes={"cohort": "GroupA", "score": "Medium"}, metrics={"response_count": 15}),
            call(attributes={"cohort": "GroupA", "score": "High"}, metrics={"response_count": 5}),
            call(attributes={"cohort": "GroupB", "score": "Low"}, metrics={"response_count": 20}),
            call(attributes={"cohort": "GroupB", "score": "Medium"}, metrics={"response_count": 25}),
            call(attributes={"cohort": "GroupB", "score": "High"}, metrics={"response_count": 10}),
        ]
        mock_recorder.populate_metrics.assert_has_calls(expected_calls, any_order=True)

    @patch("seismometer.data.telemetry.OpenTelemetryRecorder")
    def test_multi_index(self, mock_recorder_class):
        """Test recording from a DataFrame with multi-level index."""
        # Arrange
        mock_recorder = Mock()
        mock_recorder_class.return_value = mock_recorder

        df = pd.DataFrame(
            {"Disagree": [10, 5], "Neutral": [15, 10], "Agree": [35, 45]},
            index=pd.MultiIndex.from_tuples([("A", "C"), ("B", "D")], names=["Group1", "Group2"]),
        )

        # Act
        telemetry.record_dataframe_matrix(df, "likes_cats")

        # Assert
        assert mock_recorder.populate_metrics.call_count == 6

        # Verify multi-index attributes
        expected_calls = [
            call(attributes={"Group1": "A", "Group2": "C", "score": "Disagree"}, metrics={"likes_cats": 10}),
            call(attributes={"Group1": "A", "Group2": "C", "score": "Neutral"}, metrics={"likes_cats": 15}),
            call(attributes={"Group1": "A", "Group2": "C", "score": "Agree"}, metrics={"likes_cats": 35}),
            call(attributes={"Group1": "B", "Group2": "D", "score": "Disagree"}, metrics={"likes_cats": 5}),
            call(attributes={"Group1": "B", "Group2": "D", "score": "Neutral"}, metrics={"likes_cats": 10}),
            call(attributes={"Group1": "B", "Group2": "D", "score": "Agree"}, metrics={"likes_cats": 45}),
        ]
        mock_recorder.populate_metrics.assert_has_calls(expected_calls, any_order=True)

    @patch("seismometer.data.telemetry.OpenTelemetryRecorder")
    def test_empty_dataframe(self, mock_recorder_class):
        """Test that empty DataFrame is handled gracefully without recording."""
        # Arrange
        mock_recorder = Mock()
        mock_recorder_class.return_value = mock_recorder

        df = pd.DataFrame()

        # Act
        telemetry.record_dataframe_matrix(df, "test_metric")

        # Assert
        mock_recorder_class.assert_not_called()
        mock_recorder.populate_metrics.assert_not_called()

    @patch("seismometer.data.telemetry.OpenTelemetryRecorder")
    def test_nan_values_skipped(self, mock_recorder_class):
        """Test that NaN values are skipped and not recorded."""
        # Arrange
        mock_recorder = Mock()
        mock_recorder_class.return_value = mock_recorder

        df = pd.DataFrame({"Low": [10, np.nan], "Medium": [np.nan, 25], "High": [5, 10]}, index=["GroupA", "GroupB"])
        df.index.name = "cohort"

        # Act
        telemetry.record_dataframe_matrix(df, "response_count")

        # Assert
        # Should only record 4 values (skipping 2 NaNs)
        assert mock_recorder.populate_metrics.call_count == 4

        # Verify NaN values were skipped
        recorded_calls = mock_recorder.populate_metrics.call_args_list
        for call_args in recorded_calls:
            value = call_args[1]["metrics"]["response_count"]
            assert not pd.isna(value)

    @patch("seismometer.data.telemetry.OpenTelemetryRecorder")
    def test_custom_attributes(self, mock_recorder_class):
        """Test that custom attributes are merged with DataFrame-derived attributes."""
        # Arrange
        mock_recorder = Mock()
        mock_recorder_class.return_value = mock_recorder

        df = pd.DataFrame({"Low": [10], "High": [5]}, index=["GroupA"])
        df.index.name = "cohort"

        custom_attrs = {"experiment": "test_1", "version": 2}

        # Act
        telemetry.record_dataframe_matrix(df, "response_count", attributes=custom_attrs)

        # Assert
        expected_calls = [
            call(
                attributes={"cohort": "GroupA", "experiment": "test_1", "version": 2, "score": "Low"},
                metrics={"response_count": 10},
            ),
            call(
                attributes={"cohort": "GroupA", "experiment": "test_1", "version": 2, "score": "High"},
                metrics={"response_count": 5},
            ),
        ]
        mock_recorder.populate_metrics.assert_has_calls(expected_calls, any_order=True)

    @patch("seismometer.data.telemetry.OpenTelemetryRecorder")
    def test_custom_source(self, mock_recorder_class):
        """Test that custom source name is passed to recorder."""
        # Arrange
        mock_recorder = Mock()
        mock_recorder_class.return_value = mock_recorder

        df = pd.DataFrame({"Low": [10]}, index=["GroupA"])

        # Act
        telemetry.record_dataframe_matrix(df, "test_metric", source="CustomSource")

        # Assert
        mock_recorder_class.assert_called_once_with(metric_names=["test_metric"], name="CustomSource")

    @patch("seismometer.data.telemetry.OpenTelemetryRecorder")
    def test_unnamed_index(self, mock_recorder_class):
        """Test that unnamed indices get default names."""
        # Arrange
        mock_recorder = Mock()
        mock_recorder_class.return_value = mock_recorder

        df = pd.DataFrame({"Low": [10], "High": [5]}, index=["GroupA"])
        # Don't set index.name - leave it as None

        # Act
        telemetry.record_dataframe_matrix(df, "response_count")

        # Assert
        expected_calls = [
            call(attributes={"index_0": "GroupA", "score": "Low"}, metrics={"response_count": 10}),
            call(attributes={"index_0": "GroupA", "score": "High"}, metrics={"response_count": 5}),
        ]
        mock_recorder.populate_metrics.assert_has_calls(expected_calls, any_order=True)

    @patch("seismometer.data.telemetry.OpenTelemetryRecorder")
    def test_numeric_index_values(self, mock_recorder_class):
        """Test that numeric index values are handled correctly."""
        # Arrange
        mock_recorder = Mock()
        mock_recorder_class.return_value = mock_recorder

        df = pd.DataFrame({"Low": [10, 20], "High": [5, 10]}, index=[0, 1])
        df.index.name = "id"

        # Act
        telemetry.record_dataframe_matrix(df, "response_count")

        # Assert
        assert mock_recorder.populate_metrics.call_count == 4

        # Verify numeric indices are preserved
        expected_calls = [
            call(attributes={"id": 0, "score": "Low"}, metrics={"response_count": 10}),
            call(attributes={"id": 0, "score": "High"}, metrics={"response_count": 5}),
            call(attributes={"id": 1, "score": "Low"}, metrics={"response_count": 20}),
            call(attributes={"id": 1, "score": "High"}, metrics={"response_count": 10}),
        ]
        mock_recorder.populate_metrics.assert_has_calls(expected_calls, any_order=True)

    @patch("seismometer.data.telemetry.OpenTelemetryRecorder")
    def test_float_values(self, mock_recorder_class):
        """Test that float metric values are recorded correctly."""
        # Arrange
        mock_recorder = Mock()
        mock_recorder_class.return_value = mock_recorder

        df = pd.DataFrame({"Low": [10.5, 20.7], "High": [5.3, 10.1]}, index=["GroupA", "GroupB"])
        df.index.name = "cohort"

        # Act
        telemetry.record_dataframe_matrix(df, "response_rate")

        # Assert
        assert mock_recorder.populate_metrics.call_count == 4

        # Verify float values
        for call_args in mock_recorder.populate_metrics.call_args_list:
            value = call_args[1]["metrics"]["response_rate"]
            assert isinstance(value, float)


class TestRecordSingleMetric:
    """Tests for record_single_metric function."""

    @patch("seismometer.data.telemetry.OpenTelemetryRecorder")
    def test_basic_recording(self, mock_recorder_class):
        """Test basic single metric recording."""
        # Arrange
        mock_recorder = Mock()
        mock_recorder_class.return_value = mock_recorder

        # Act
        telemetry.record_single_metric("accuracy", 0.92)

        # Assert
        mock_recorder_class.assert_called_once_with(metric_names=["accuracy"], name="SingleMetric")
        mock_recorder.populate_metrics.assert_called_once_with(attributes={}, metrics={"accuracy": 0.92})

    @patch("seismometer.data.telemetry.OpenTelemetryRecorder")
    def test_with_attributes(self, mock_recorder_class):
        """Test recording with custom attributes."""
        # Arrange
        mock_recorder = Mock()
        mock_recorder_class.return_value = mock_recorder

        attrs = {"model": "A", "experiment": "test_1"}

        # Act
        telemetry.record_single_metric("accuracy", 0.92, attributes=attrs)

        # Assert
        mock_recorder.populate_metrics.assert_called_once_with(
            attributes={"model": "A", "experiment": "test_1"}, metrics={"accuracy": 0.92}
        )

    @patch("seismometer.data.telemetry.OpenTelemetryRecorder")
    def test_custom_source(self, mock_recorder_class):
        """Test recording with custom source name."""
        # Arrange
        mock_recorder = Mock()
        mock_recorder_class.return_value = mock_recorder

        # Act
        telemetry.record_single_metric("latency", 100, source="PerformanceMonitor")

        # Assert
        mock_recorder_class.assert_called_once_with(metric_names=["latency"], name="PerformanceMonitor")
