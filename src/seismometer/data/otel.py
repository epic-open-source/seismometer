import logging
import socket
import sys
from typing import Callable

import yaml

from seismometer.core.decorators import export

logger = logging.getLogger("Seismometer OpenTelemetry")

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


def read_otel_info(file_path: str) -> dict:
    """Reads in the OTel information (which metrics to load, etc.) from a file.

    Parameters
    ----------
    file_path : str
        Where to read from.

    Returns
    -------
    dict
        The YAML object.

    Raises
    ------
    Exception
        If there is no file at the specified location.
    """
    try:
        file = open(file_path, "r")
        return yaml.safe_load(file)["otel_info"]
    except FileNotFoundError:
        raise Exception("Could not find usage config file for metric setup!")
    except (KeyError, TypeError):
        logger.warning(f"No OTel config found in {file_path}. Will be defaulting ...")
        return {}


# This will be set once usage_config.yml is done downloading, in run_startup.
OTEL_INFO = {}


def get_metric_config(metric_name: str) -> dict:
    """_summary_

    Parameters
    ----------
    metric_name : str
        The metric.

    Returns
    -------
    dict
        The configuration, as described in RFC #4 as a dictionary.
        E.g. {"output_metrics": True}, etc.
    """

    METRIC_DEFAULTS = {"output_metrics": True, "log_all": False, "granularity": 4, "measurement_type": "Gauge"}

    if metric_name in OTEL_INFO:
        ret = OTEL_INFO[metric_name]
    else:
        ret = {}
    # Overwrite defaults with whatever is in the dictionary.
    return METRIC_DEFAULTS | ret


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
    typestring = get_metric_config(metric_name)["measurement_type"]
    return TYPES[typestring] if typestring in TYPES else Meter.create_gauge


class ExportManager:
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
    def __init__(self, file_output_path=None, export_port=None, dump_to_stdout=False):
        """Create a place to export files.

        Parameters
        ----------
        file_output_path : str, optional
            Where metrics are to be dumped to for debugging purposes, if needed.
            Set this to the object sys.stdout (NOT the string) in order to just
            log metrics to the console.
        prom_port : int, optional
            What port (local HTTP server for instance) metrics are to be dumped,
            for Prometheus exporting purposes.
        dump_to_stdout: bool, optional
            Whether to dump the metrics to stdout.
        """

        if file_output_path is None and export_port is None:
            raise Exception("Metrics must go somewhere!")
        self.readers = []
        self.otlp_exhaust = None
        if file_output_path is not None:
            if file_output_path == sys.stdout:
                self.otlp_exhaust = sys.stdout
            else:
                self.otlp_exhaust = open(
                    file_output_path, "w"
                )  # Save this for closing files down at the end of execution
            self.readers.append(
                PeriodicExportingMetricReader(
                    ConsoleMetricExporter(out=self.otlp_exhaust), export_interval_millis=5000
                )
            )
        if export_port is not None:
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
        if self.otlp_exhaust is not None and (sys is None or self.otlp_exhaust != sys.stdout):
            self.otlp_exhaust.close()


# For debug purposes: dump to stdout, and also to exporter path
export_manager = ExportManager(file_output_path=sys.stdout, export_port=4317)


@export
def deactivate_exports():
    export_manager.deactivate_exports()


@export
def activate_exports():
    export_manager.activate_exports()
