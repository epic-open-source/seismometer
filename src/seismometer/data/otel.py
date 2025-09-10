import logging
import socket
import sys
import time
from pathlib import Path
from typing import Optional

from seismometer.configuration.export_config import ExportConfig
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
class ExportManager(object, metaclass=Singleton):
    def __new__(cls, export_config: Optional[ExportConfig] = None):
        if not OTEL_AVAILABLE or export_config is None or not export_config.is_exporting_possible():
            return NoOpExportManager()
        return RealExportManager(export_config)


class NoOpExportManager:
    def __init__(self):
        self.active = False

    def deactivate_exports(self):
        logger.warning("Telemetry packages have not been installed! Exports are already deactivated.")

    def activate_exports(self):
        logger.warning("Telemetry packages have not been installed! Exports cannot be activated.")


# Class which stores info about exporting metrics.
class RealExportManager:
    def __init__(self, export_config: ExportConfig):
        """Create a place to export files.

        Parameters
        ----------
        export_config: ExportConfig
            Configuration for exports: files, hostnames/ports, or logging to stdout
        """
        if not export_config.is_exporting_possible():
            self.deactivate_exports()
            return  # We're done already!
        self.readers = []
        self.otlp_exhausts = []
        for file_output_path in export_config.otel_files:
            file_path = Path(file_output_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_exhaust = open(file_path, "a")  # Does not overwrite existing data
            self.readers.append(
                PeriodicExportingMetricReader(ConsoleMetricExporter(out=file_exhaust), export_interval_millis=5000)
            )
            self.otlp_exhausts.append(file_exhaust)
        for export_port in export_config.otel_ports:
            if self._can_connect_to_socket(host=export_config.hostname, port=export_port):
                otlp_exporter = OTLPMetricExporter(endpoint=f"{export_config.hostname}:{export_port}", insecure=True)
                otel_collector_reader = PeriodicExportingMetricReader(otlp_exporter, export_interval_millis=5000)
                self.readers.append(otel_collector_reader)
        if export_config.otel_stdout:
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
    ExportManager().deactivate_exports()


@export
def activate_exports():
    """Revert to the default state of allowing metric emission."""
    ExportManager().activate_exports()
