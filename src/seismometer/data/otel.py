import logging
import socket
import sys
import time
from typing import Callable

from seismometer.core.autometrics import AutomationManager
from seismometer.core.decorators import export
from seismometer.core.patterns import Singleton

logger = logging.getLogger("seismometer.telemetry")

try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.metrics import Meter, set_meter_provider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource

    TELEMETRY_POSSIBLE = True
except ImportError:
    # No OTel.
    TELEMETRY_POSSIBLE = False


def get_metric_creator(metric_name: str, meter) -> Callable:
    """Takes in the name of a metric and determines the OTel function which creates
    the corresponding instrument.

    Parameters
    ----------
    metric_name : str
        Which metric, like "Accuracy".
    meter: Meter
        Which meter is providing these instruments.

    Returns
    -------
    Callable
        The function creating the right metric instrument.

    """
    if not TELEMETRY_POSSIBLE:
        # This should not happen, but just in case ...
        logger.warning("Tried to get a metric creator, but metrics packages are not installed!")
        return None
    TYPES = {"Gauge": meter.create_gauge, "Counter": meter.create_up_down_counter, "Histogram": meter.create_histogram}
    typestring = AutomationManager().get_metric_config(metric_name)["measurement_type"]
    return TYPES[typestring] if typestring in TYPES else Meter.create_gauge


@export
class ExportManager(object, metaclass=Singleton):
    def __new__(cls, *args, **kwargs):
        if TELEMETRY_POSSIBLE:
            return RealExportManager(*args, **kwargs)
        else:
            return NoOpExportManager(*args, **kwargs)


class NoOpExportManager:
    def __init__(self, *args, **kwargs):
        pass

    def deactivate_exports(self, *args, **kwargs):
        logger.warning("Telemetry packages have not been installed! Exports are already deactivated.")

    def activate_exports(self, *args, **kwargs):
        logger.warning("Telemetry packages have not been installed! Exports cannot be activated.")


# Class which stores info about exporting metrics.
class RealExportManager:
    def __init__(self, hostname, file_output_paths, export_ports, dump_to_stdout):
        """Create a place to export files.

        Parameters
        ----------
        hostname: str
            The host name to use for the ports.
        file_output_path : list[str], optional
            Where metrics are to be dumped to for debugging purposes.
        prom_port : int, optional
            What ports (local HTTP server for instance) metrics are to be dumped to,
            for Prometheus or OTel collector purposes.
        dump_to_stdout: bool, optional
            Whether to dump the metrics to stdout.
        """

        if file_output_paths == export_ports == [] and not dump_to_stdout:
            self.deactivate_exports()
            return  # We're done already!
        self.readers = []
        self.otlp_exhausts = []
        for file_output_path in file_output_paths:
            file_exhaust = open(file_output_path, "a")  # Does not overwrite existing data
            self.readers.append(
                PeriodicExportingMetricReader(ConsoleMetricExporter(out=file_exhaust), export_interval_millis=5000)
            )
            self.otlp_exhausts.append(file_exhaust)
        for export_port in export_ports:
            if self._can_connect_to_socket(host=hostname, port=export_port):
                otlp_exporter = OTLPMetricExporter(endpoint=f"{hostname}:{export_port}", insecure=True)
                otel_collector_reader = PeriodicExportingMetricReader(otlp_exporter, export_interval_millis=5000)
                self.readers.append(otel_collector_reader)
        if dump_to_stdout:
            self.readers.append(
                PeriodicExportingMetricReader(ConsoleMetricExporter(out=sys.stdout), export_interval_millis=5000)
            )
        self.resource = Resource.create(attributes={SERVICE_NAME: "Seismometer"})
        self.meter_provider = MeterProvider(resource=self.resource, metric_readers=self.readers)
        set_meter_provider(self.meter_provider)
        self.active = False  # by default!

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
