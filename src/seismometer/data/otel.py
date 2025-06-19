import functools
import itertools
import logging
import operator
import os
import sys
from typing import Any, Callable, Dict, List

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
    except (KeyError, TypeError):
        logger.warning("No OTel config found. Will be defaulting ...")
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
    return TYPES[typestring] if typestring in TYPES else Meter.create_gauge


# Class which stores info about exporting metrics.
class ExportManager:
    def __init__(self, file_output_path=None, prom_port=None, dump_to_stdout=False):
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

        if STOP_ALL_OTEL:
            self.otlp_exhaust = None
            return

        if file_output_path is None and prom_port is None:
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
        if prom_port is not None:
            try:
                start_http_server(port=prom_port, addr="0.0.0.0")
                self.readers.append(PrometheusMetricReader())
            except OSError:
                logger.warning("Port is already in use. Ignoring ...")
        self.resource = Resource.create(attributes={SERVICE_NAME: "Seismometer"})
        self.meter_provider = MeterProvider(resource=self.resource, metric_readers=self.readers)

    def __del__(self):
        if self.otlp_exhaust is not None and self.otlp_exhaust != sys.stdout:
            self.otlp_exhaust.close()


# For debug purposes: dump to stdout, and also to exporter path
export_manager = ExportManager(file_output_path=sys.stdout, prom_port=9464)


class OpenTelemetryRecorder:
    def __init__(self, metric_names: List[str], name: str = "Seismo-meter"):
        """_summary_

        Parameters
        ----------
        metric_names : List[str]
            These are the kinds of metrics we want to be collecting. E.g. fairness, accuracy.
        """

        # If we are not recording metrics, don't bother.
        if STOP_ALL_OTEL:
            self.metric_names = []
            return

        meter_provider = export_manager.meter_provider
        # OpenTelemetry: use this new object to spawn new "Instruments" (measuring devices)
        self.meter = meter_provider.get_meter(name)
        # Keep it like this for now: just make an instrument for each metric we are measuring
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

        metrics : dict[str, Any].
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
            logger.warning(f"Instruments available: {self.metric_names}")
            logger.warning(f"Metrics provided for population: {metrics.keys()}")

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
        elif isinstance(data, list):
            for datapoint in data:
                set_one_datapoint(datapoint)
        elif isinstance(data, (dict, pd.Series)):
            # Same logic for both dictionary and (labeled) series,
            # but we would need to get the dictionary representation of the series first.
            for k, v in dict(data).items():
                # Augment attributes with keys of dictionary
                self._log_to_instrument(attributes | {"key": k}, instrument, v)
        elif isinstance(data, str):
            raise Exception(f"Cannot log strings as metrics in OTel. Tried to log {data} as a metric.")
        else:
            raise Exception(f"Unrecognized data format for OTel logging: {str(type(data))}")

    def log_by_cohort(
        self,
        base_attributes: Dict[str, Any],
        dataframe: pd.DataFrame,
        cohorts: Dict[str, List[str]],
        intersecting: bool = False,
        metric_maker: Callable = None,
    ):
        """Take data from a dataframe and log it, selecting by all cohorts provided.

        Parameters
        ----------
        base_attributes : Dict[str, Any]
            The information we want to store with every individual metric log. This might be, for example, every
            parameter used to generate this data.
        dataframe : pd.DataFrame
            The actual dataframe containing the data we want to log.
        cohorts : Dict[str, List[str]]
            Which cohorts we want to select on. For instance:
            {"Age": ["[10-20)", "70+"], "Race": ["AfricanAmerican", "Caucasian"]}
        intersecting: bool
            Whether we are logging each combination of separate cohorts or not.
            Given the example cohorts above:
                - intersecting=False would log data for Age=[10-20), Age=70+, Race=AfricanAmerican, and Race=Caucasian.
                - intersecting=True: Age=[10,20) and Race=AfricanAmerican, Age=[20, 50) and Race=Caucasian, etc.
        metric_maker: Callable
            Produce a metric to log from the Series we will create.
            For example, in plot_cohort_hist, what we want is the length of each dataframe.
            If not, we will log each row separately.
        """
        if not intersecting:
            # Simpler case: just go through each label provided.
            for cohort_category in cohorts.keys():
                for cohort_value in cohorts[cohort_category]:
                    # We want to log all of the attributes passed in, but also what cohorts we are selecting on.
                    attributes = base_attributes | {cohort_category: cohort_value}
                    metrics = dataframe[dataframe[cohort_category] == cohort_value]
                    if metric_maker is not None:
                        self.populate_metrics(attributes=attributes, metrics=metric_maker(metrics))
                    else:
                        self.populate_metrics(attributes=attributes, metrics=metrics)
            # More complex: go through each combination of attributes.
        else:
            if cohorts:
                keys = cohorts.keys()
                # So if column "A" has attributes A_false and A_true, and B has 1, 2, 3, then
                # we will exhaust through all combinations of these attributes.
                selections = list(itertools.product(*[cohorts[key].unique() for key in keys]))
                # Now we put them into a dictionary for ease of processing
                selections = [dict(zip(keys, s)) for s in selections]
            else:
                # If there are in fact no other attributes, get a list so we don't just skip logging entirely
                selections = [{}]
            if len(selections) > 100:
                logger.warning("More than 100 cohort groups were provided. This might take a while.")
            for selection in selections:
                attributes = base_attributes | selection
                selection_condition = (
                    functools.reduce(operator.and_, (dataframe[k] == v for k, v in selection.items()))
                    if selection
                    else pd.Series([True] * len(dataframe), index=dataframe.index)  # Condition which always succeeds
                )
                metrics = dataframe[selection_condition]
                if metric_maker is not None:
                    self.populate_metrics(attributes=attributes, metrics=metric_maker(metrics))
                else:
                    for row in metrics:
                        self.populate_metrics(attributes=attributes, metrics=row)
