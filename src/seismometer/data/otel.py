import logging
import os
from typing import Any, Callable, List

import numpy as np
import pandas as pd
import yaml
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import Histogram, Meter, UpDownCounter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from prometheus_client import start_http_server

from seismometer.core.io import slugify

logger = logging.getLogger("Seismometer OpenTelemetry")


def config_otel_stoppage() -> bool:
    """Get from the environment whether we should export metrics at all, or not.

    Returns
    -------
    bool
        Whether or not all OTel outputs will be disabled.
    """
    raw_stop = os.getenv("SEISMO_NO_OTEL", "FALSE")
    if raw_stop not in ["TRUE", "FALSE"]:
        logger.warn("Unrecognized value for SISMO_NO_OTEL. Defaulting to false (metrics will be output ...)")
        raw_stop = "FALSE"
    return raw_stop == "TRUE"


STOP_ALL_OTEL = config_otel_stoppage()


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
    except KeyError:
        logger.warning("No OTel config found. Will be defaulting ...")
        return {}


# This will be set once usage_config.yml is done downloading, in run_startup.
OTEL_INFO = None


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


def get_metric_creator(metric_name: str, meter: Meter) -> Callable:
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
    TYPES = {"Gauge": meter.create_gauge, "Counter": meter.create_up_down_counter, "Histogram": meter.create_histogram}
    typestring = get_metric_config(metric_name)["measurement_type"]
    if "Time Lead" in metric_name.lower():
        return TYPES["Histogram"]
    return TYPES[typestring] if typestring in TYPES else Meter.create_gauge


# Class which stores info about exporting metrics.
class ExportManager:
    def __init__(self, file_output_path=None, prom_port=None):
        """Create a place to export files.

        Parameters
        ----------
        file_output_path : str, optional
            Where metrics are to be dumped to for debugging purposes, if needed.
        prom_port : int, optional
            What port (local HTTP server for instance) metrics are to be dumped,
            for Prometheus exporting purposes.
        """

        if STOP_ALL_OTEL:
            self.otlp_exhaust = None
            return

        if file_output_path is None and prom_port is None:
            raise Exception("Metrics must go somewhere!")
        self.readers = []
        self.otlp_exhaust = None
        if file_output_path is not None:
            self.otlp_exhaust = open(file_output_path, "w")
            self.readers.append(
                PeriodicExportingMetricReader(
                    ConsoleMetricExporter(out=self.otlp_exhaust), export_interval_millis=5000
                )
            )
        else:
            self.otlp_exhaust = None  # Save this for closing files down at the end of execution
        if prom_port is not None:
            try:
                start_http_server(port=prom_port, addr="0.0.0.0")
                self.readers.append(PrometheusMetricReader())
            except OSError:
                logger.warning("Port is already in use. Ignoring ...")
        self.resource = Resource.create(attributes={SERVICE_NAME: "Seismometer"})
        self.meter_provider = MeterProvider(resource=self.resource, metric_readers=self.readers)

    def __del__(self):
        if self.otlp_exhaust is not None:
            self.otlp_exhaust.close()


# For debug purposes: dump to stdout, and also to exporter path
export_manager = ExportManager(file_output_path="/dev/stdout", prom_port=9464)


class OpenTelemetryRecorder:
    def __init__(self, metric_names: List[str], name: str = "Seismo-meter"):
        """_summary_

        Parameters
        ----------
        metric_names : List[str]
            These are the kinds of metrics we want to be collecting. E.g. fairness, accuracy.
        output : str, optional
            This is the file path where outputted metrics should be dumped. For now, we are just dealing
            with file path, but further support for OTel exporters will be added in the future.
            Leave this blank for output to be stdout -- e.g. dumped to the console.S
        """

        # If we are not recording metrics, don't bother.
        if STOP_ALL_OTEL:
            self.metric_names = []
            return

        meter_provider = export_manager.meter_provider
        # OpenTelemetry: use this new object to spawn new "Instruments" (measuring devices)
        self.meter = meter_provider.get_meter(name)
        # Keep it like this for now: just make an instrument for each metric we are measuring
        # TODO: get better descriptions for each metric besides just the name
        # This is a map from metric name to corresponding instrument
        self.instruments: dict[str, Any] = {}
        self.metric_names = metric_names
        for mn in metric_names:
            creator_fn = get_metric_creator(mn, self.meter)
            self.instruments[mn] = creator_fn(slugify(mn), description=mn)

    def populate_metrics(self, attributes, metrics):
        """Populate the OpenTelemetry instruments with data from
        the model.

        Parameters
        ----------
        attributes: dict[str, Union[str, int]]
            All information associated with this metric. For instance,
                - what cohort is this a part of?
                - what metric is this actually?
                - what are the score and fairness thresholds?
                - etc.

        metrics : dict[str, float].
            The actual data we are populating.
        """

        if STOP_ALL_OTEL:
            return

        if metrics is None:
            # metrics = self()
            raise Exception()
        for name in self.instruments.keys():
            # I think metrics.keys() is a subset of self.instruments.keys()
            # but I am not 100% on it. So this stays for now.
            if name in metrics and get_metric_config(name)["output_metrics"]:
                self._log_to_instrument(attributes, self.instruments[name], metrics[name])
        if not any([name in metrics for name in self.instruments.keys()]):
            logger.warning("No metrics populated with this call!")

    def _log_to_instrument(self, attributes, instrument: Any, data):
        """Write information to a single instrument. We need this
        wrapper function because the data we are logging may be a
        type such as a series, in which case we need to log each
        entry separately or at least do some extra preprocessing.

        Parameters
        ----------
        attributes : dict[str, tuple[Any]]
            Which parameters go into this measurement.
        instrument : Any
            The OpenTelemetry instrument that we are recording a
            measurement to. Should be one of self.instruments.
        data
            The data we are recording. Could conceivably be either
            a numeric value or a series.
        """

        def set_one_datapoint(value):
            # The SDK doesn't expose Gauge as a type, so we need to get creative here.
            if type(instrument).__name__ == "_Gauge":
                instrument.set(value, attributes=attributes)
            elif isinstance(instrument, UpDownCounter):
                instrument.add(value, attributes=attributes)
            elif isinstance(instrument, Histogram):
                instrument.record(value, attributes=attributes)
            else:
                raise Exception(
                    f"Internal error: one of the instruments is not a recognized type: {str(type(instrument))}"
                )

        # Some code seems to be logging numpy int64s so here we are.
        if isinstance(data, (int, float)):
            set_one_datapoint(data)
        elif isinstance(data, (np.int64, np.float64)):
            set_one_datapoint(data.item())
        elif isinstance(data, (list, pd.Series)):
            for datapoint in data:
                set_one_datapoint(datapoint)
        elif isinstance(data, dict):
            for k, v in data.items():
                # Augment attributes with keys of dictionary
                self._log_to_instrument(attributes | {"key": k}, instrument, v)
        elif isinstance(data, str):
            raise Exception(f"Cannot log strings as metrics in OTel. Tried to log {data} as a metric.")
        else:
            raise Exception(f"Unrecognized data format for OTel logging: {str(type(data))}")
