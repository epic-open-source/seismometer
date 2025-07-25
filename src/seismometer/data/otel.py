import logging
import socket
import sys
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
        logger.warning("Tried to get a metric creator, but no metrics are active!")
        return None
    TYPES = {"Gauge": meter.create_gauge, "Counter": meter.create_up_down_counter, "Histogram": meter.create_histogram}
    typestring = AutomationManager().get_metric_config(metric_name)["measurement_type"]
    return TYPES[typestring] if typestring in TYPES else Meter.create_gauge


@export
class ExportManager:
    def __new__(cls, *args, **kwargs):
        if TELEMETRY_POSSIBLE:
            return RealExportManager(*args, **kwargs)
        else:
            return NoOpExportManager(*args, **kwargs)


class NoOpExportManager(object, metaclass=Singleton):
    def __init__(self, *args, **kwargs):
        pass

    def deactivate_exports(self, *args, **kwargs):
        logger.warning("Telemetry packages have not been installed! Exports are already deactivated.")

    def activate_exports(self, *args, **kwargs):
        logger.warning("Telemetry packages have not been installed! Exports cannot be activated.")


# Class which stores info about exporting metrics.
class RealExportManager(object, metaclass=Singleton):
    def __init__(self, file_output_paths=[], export_ports=[], dump_to_stdout=False):
        """Create a place to export files.

        Parameters
        ----------
        file_output_path : list[str], optional
            Where metrics are to be dumped to for debugging purposes.
        prom_port : int, optional
            What ports (local HTTP server for instance) metrics are to be dumped to,
            for Prometheus or OTel collector purposes.
        dump_to_stdout: bool, optional
            Whether to dump the metrics to stdout.
        """

        if file_output_paths == export_ports == [] and not dump_to_stdout:
            raise Exception("Metrics must go somewhere!")
        self.readers = []
        self.otlp_exhausts = []
        for file_output_path in file_output_paths:
            file_exhaust = open(file_output_path, "w")
            self.readers.append(
                PeriodicExportingMetricReader(ConsoleMetricExporter(out=file_exhaust), export_interval_millis=5000)
            )
            self.otlp_exhausts.append(file_exhaust)
        for export_port in export_ports:
            otel_collector_reader = None
            try:
                otlp_exporter = OTLPMetricExporter(endpoint=f"otel-collector:{export_port}", insecure=True)
                otel_collector_reader = PeriodicExportingMetricReader(otlp_exporter, export_interval_millis=5000)
                with socket.create_connection(("otel-collector", export_port), timeout=1):
                    pass
            except OSError:
                logger.warning("Connecting to port failed. Ignoring ...")
                if otel_collector_reader is not None:
                    otel_collector_reader.shutdown()
            else:
                self.readers.append(otel_collector_reader)
        if dump_to_stdout:
            self.readers.append(
                PeriodicExportingMetricReader(ConsoleMetricExporter(out=sys.stdout), export_interval_millis=5000)
            )
        self.resource = Resource.create(attributes={SERVICE_NAME: "Seismometer"})
        self.meter_provider = MeterProvider(resource=self.resource, metric_readers=self.readers)
        set_meter_provider(self.meter_provider)
        self.active = False  # by default!

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
    ExportManager().deactivate_exports()


@export
def activate_exports():
    ExportManager().activate_exports()
