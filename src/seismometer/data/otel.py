import logging
import socket
import sys
import time
from pathlib import Path
from typing import Optional

from seismometer.configuration.telemetry_config import TelemetryConfig
from seismometer.core.decorators import export
from seismometer.core.patterns import Singleton

logger = logging.getLogger("seismometer.telemetry")

try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.metrics import set_meter_provider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource

    OTEL_AVAILABLE = True
except ImportError:
    # No OTel.
    OTEL_AVAILABLE = False


@export
class MetricTelemetryManager(object, metaclass=Singleton):
    def __new__(cls, telemetry_config: Optional[TelemetryConfig] = None):
        if not OTEL_AVAILABLE:
            logger.info("Telementry not installed, export manager disabled.")
            return NoOpMetricTelemetryManager()
        elif telemetry_config is None:
            logger.info("No export configured, export manager disabled.")
            return NoOpMetricTelemetryManager()
        elif not telemetry_config.is_exporting_possible():
            logger.info("No export locations specified, export manager disabled.")
            return NoOpMetricTelemetryManager()
        return RealMetricTelemetryManager(telemetry_config)


class NoOpMetricTelemetryManager:
    def __init__(self):
        self.active = False

    def deactivate_exports(self):
        pass

    def activate_exports(self):
        logger.warning("Telemetry disabled! Exports cannot be activated.")


# Class which stores info about exporting metrics.
class RealMetricTelemetryManager:
    def __init__(self, telemetry_config: TelemetryConfig):
        """Create a place to export files.

        Parameters
        ----------
        telemetry_config: TelemetryConfig
            Configuration for exports: files, hostnames/ports, or logging to stdout
        """
        if not telemetry_config.is_exporting_possible():
            self.deactivate_exports()
            return  # We're done already!
        self.readers = []
        self.otlp_exhausts = []
        for file_output_path in telemetry_config.otel_files:
            file_path = Path(file_output_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_exhaust = open(file_path, "a")  # Does not overwrite existing data
            self.readers.append(
                PeriodicExportingMetricReader(ConsoleMetricExporter(out=file_exhaust), export_interval_millis=5000)
            )
            self.otlp_exhausts.append(file_exhaust)
        for export_port in telemetry_config.otel_ports:
            if self._can_connect_to_socket(host=telemetry_config.hostname, port=export_port):
                otlp_exporter = OTLPMetricExporter(
                    endpoint=f"{telemetry_config.hostname}:{export_port}", insecure=True
                )
                otel_collector_reader = PeriodicExportingMetricReader(otlp_exporter, export_interval_millis=5000)
                self.readers.append(otel_collector_reader)
        if telemetry_config.otel_stdout:
            self.readers.append(
                PeriodicExportingMetricReader(ConsoleMetricExporter(out=sys.stdout), export_interval_millis=5000)
            )
        self.resource = Resource.create(attributes={SERVICE_NAME: "Seismometer"})
        self.meter_provider = MeterProvider(resource=self.resource, metric_readers=self.readers)
        set_meter_provider(self.meter_provider)
        self.active = True  # by default!

    def _can_connect_to_socket(self, host: str, port: int, timeout: float = 1.0, retries: int = 10) -> bool:
        """See if we can connect to host:port.

        Parameters
        ----------
        host : str
            The host name.
        port : int
            Which port number we are connecting to.
        timeout : float, optional
            How long of a timeout to set when connecting, by default 1.0
        retries : int, optional
            How many tries we can do at most, by default 10

        Returns
        -------
        bool
            Whether connection succeeded or failed.

        """
        for i in range(retries):
            try:
                with socket.create_connection((host, port), timeout=timeout):
                    logger.info(f"Connection to {host}:{port} succeeded.")
                    return True
            except (socket.timeout, ConnectionRefusedError, OSError) as e:
                logger.debug(f"Attempt {i+1} out of {retries} attempts to {host}:{port} failed: {e}")
                time.sleep(1)
            logger.warning(f"Max retries ({retries}) passed, could not connect to {host}:{port}.")
            return False

    def deactivate_exports(self):
        """Make it so that all metric emission calls are no-ops."""
        self.active = False

    def activate_exports(self):
        """Revert to the default state of allowing metric emission."""
        # added a hasattr check to see if the meter_provider was not set
        if not hasattr(self, "meter_provider"):
            logger.warning("Telemetry configuration not loaded! Cannot activate exports.")
            return
        self.active = True

    def __del__(self):
        for file_exhaust in self.otlp_exhausts:
            file_exhaust.close()


@export
def deactivate_exports():
    """Make it so that all metric emission calls are no-ops."""
    MetricTelemetryManager().deactivate_exports()


@export
def activate_exports():
    """Revert to the default state of allowing metric emission."""
    MetricTelemetryManager().activate_exports()
