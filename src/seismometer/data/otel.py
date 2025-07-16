import functools
import itertools
import logging
import operator
import os
import socket
import sys
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import yaml

from seismometer.configuration.model import OtherInfo
from seismometer.core.decorators import export
from seismometer.core.io import slugify
from seismometer.data.performance import BinaryClassifierMetricGenerator
from seismometer.seismogram import Seismogram

logger = logging.getLogger("Seismometer OpenTelemetry")

try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.metrics import Histogram, Meter, UpDownCounter, set_meter_provider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource

    TELEMETRY = True
except ImportError as e:
    logger.warning(f"Was not able to use metric exporting (message {str(e)}). Not using telemetry ...")
    TELEMETRY = False


def config_otel_stoppage() -> bool:
    """Get from the environment whether we should export metrics at all, or not.

    Returns
    -------
    bool
        Whether or not all OTel outputs will be disabled.
    """
    raw_stop = os.getenv("SEISMO_NO_OTEL", "FALSE")
    if raw_stop not in ["TRUE", "FALSE"]:
        logger.warn("Unrecognized value for SEISMO_NO_OTEL. Defaulting to false (metrics will be output ...)")
        raw_stop = "FALSE"
    return raw_stop == "TRUE"


STOP_ALL_OTEL = False


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
    TYPES = {"Gauge": meter.create_gauge, "Counter": meter.create_up_down_counter, "Histogram": meter.create_histogram}
    typestring = get_metric_config(metric_name)["measurement_type"]
    return TYPES[typestring] if typestring in TYPES else Meter.create_gauge


# Class which stores info about exporting metrics.
class ExportManager:
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

        if STOP_ALL_OTEL:
            self.otlp_exhaust = None
            return

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
        if export_port is not None and not STOP_ALL_OTEL:
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

    def __del__(self):
        if self.otlp_exhaust is not None and (sys is None or self.otlp_exhaust != sys.stdout):
            self.otlp_exhaust.close()


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
        # OTel doesn't like fancy Unicode characters.
        metrics = {metric_name.replace("\xa0", " "): metrics[metric_name] for metric_name in metrics}
        for name in self.instruments.keys():
            # I think metrics.keys() is a subset of self.instruments.keys()
            # but I am not 100% on it. So this stays for now.
            if name in metrics and get_metric_config(name)["output_metrics"]:
                self._log_to_instrument(attributes, self.instruments[name], metrics[name])
        if not any([name in metrics for name in self.instruments.keys()]):
            logger.warning("No metrics populated with this call!")
            logger.warning(f"Instruments available: {self.metric_names}")
            logger.warning(f"Metrics provided for population: {metrics.keys()}")

    def _set_one_datapoint(self, attributes, instrument, value):
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
        if data is None:
            logger.warning("otel: Tried to log 'None'.")
            return
        if isinstance(data, (int, float)):
            self._set_one_datapoint(attributes, instrument, data)
        # Some code seems to be logging numpy int64s so here we are.
        elif isinstance(data, (np.int64, np.float64)):
            self._set_one_datapoint(attributes, instrument, data.item())
        elif isinstance(data, list):
            for datapoint in data:
                self._set_one_datapoint(attributes, instrument, datapoint)
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
                for cohort_value in set(cohorts[cohort_category]):
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
                selections = list(itertools.product(*[cohorts[key] for key in keys]))
                # Now we put them into a dictionary for ease of processing
                selections = [dict(zip(keys, s)) for s in selections]
            else:
                # If there are in fact no other attributes, get a list so we don't just skip logging entirely
                selections = [{}]
            if len(selections) > 100:
                logger.warning(
                    f"More than 100 cohort groups ({len(selections)}) were provided. This might take a while."
                )
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
                    self.populate_metrics(attributes=attributes, metrics=metrics)

    def log_by_column(self, df: pd.DataFrame, col_name: str, cohorts: dict, base_attributes: dict):
        """Log with a particular column as the index, only if each metric is set to log_all.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe the data resides in.
        col_name : str
            Which column we want the index to be.
        cohorts : dict
            Which cohorts we want to extract data from when it's time.
        base_attributes : dict
            The attributes we want to associate to all of the logs from this.
        """
        for metric in df.columns:

            def maker(frame):
                return frame.set_index(col_name)[[metric]].to_dict()

            log_all = get_metric_config(metric)["log_all"]
            if log_all:
                self.log_by_cohort(base_attributes=base_attributes, dataframe=df, cohorts=cohorts, metric_maker=maker)


# If we don't have telemetry, we make everything into a no-op.
if not TELEMETRY:

    def noop(*args, **kwargs):
        pass

    class ExportManager:  # noqa: F811
        def __init__(self, *args, **kwargs):
            pass

    class OpenTelemetryRecorder:  # noqa: F811
        def __init__(self, *args, **kwargs):
            self.instruments = {}
            self.metric_names = []

        def populate_metrics(self, *args, **kwargs):  # noqa: F811
            pass

        def log_by_cohort(self, *args, **kwargs):  # noqa: F811
            pass

        def log_by_column(self, *args, **kwargs):  # noqa: F811
            pass

    get_metric_creator = noop  # noqa: F811


# For debug purposes: dump to stdout, and also to exporter path
export_manager = ExportManager(file_output_path=sys.stdout, export_port=4317)


def ready_for_serialization(obj):
    """
    Recursively convert:
      - Python objects (with __dict__) to dicts,
      - tuples to lists,
      - all contents deeply.

    This helps us because some of the plot functions take internal seismometer
    objects as arguments, and what we really care about are the attributes
    within said objects.
    """
    if isinstance(obj, (str, int, float, type(None), bool)):
        return obj
    elif isinstance(obj, dict):
        return {k: ready_for_serialization(v) for k, v in obj.items()}
    # Also turn tuples into lists because YAML doesn't love the latter.
    elif isinstance(obj, (list, tuple)):
        return [ready_for_serialization(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        return ready_for_serialization(vars(obj))
    else:
        return str(obj)


@export
def initialize_otel_config(config: OtherInfo):
    """Read all metric exporting and automation info.

    Parameters
    ----------
    config : OtherInfo
        The configuration object handed in during Seismogram initialization.
    """
    global OTEL_INFO, STOP_ALL_OTEL
    OTEL_INFO = read_otel_info(config.usage_config)
    Seismogram().load_automation_config(config.automation_config)
    STOP_ALL_OTEL = config_otel_stoppage()


@export
def export_config():
    """Produce a configuration file specifying which metrics to export,
    based on which functions have been run in the notebook.

    To note: this only counts the most recent run of each function,
    because this is what we might expect output to look like for a
    given run (each type of cell is only run once, and we don't want to
    store the old runs that have been overwritten as users figure out which
    plots and metrics they want to see). It also does not accommodate
    cells being deleted, because this would require some more in-depth
    access to the Jupyter frontend.
    """
    with open("metric-automation.yml", "w") as automation_file:
        sg = Seismogram()
        call_history = ready_for_serialization(sg._call_history)
        yaml.dump(call_history, automation_file)


# Writing a full list here for security reasons.
allowed_export_names = {
    "feature_alerts",
    "feature_summary",
    "plot_model_evaluation",
    "plot_cohort_evaluation",
    "plot_cohort_hist",
    "plot_leadtime_enc",
    "plot_binary_classifier_metrics",
    "plot_trend_intervention_outcome",
    "show_cohort_summaries",
    "target_feature_summary",
}


def do_auto_export(function_name: str, fn_settings: dict):
    """Run a (metric-generating) function with
    predetermined settings. To be used when reading in
    an auto-generated config file, as opposed to a
    manually-written one which takes a bit more
    preprocessing.

    Parameters
    ----------
    function_name : str
        The name of the function to export.
    fn_settings : dict
        What settings (see output config) to apply:
        args, kwargs: function parameters
        extra_params: the current settings of Seismogram,
        saved at the time of export.
    """
    from seismometer.api.plots import (
        _plot_cohort_hist,
        _plot_leadtime_enc,
        plot_binary_classifier_metrics,
        plot_cohort_evaluation,
        plot_model_evaluation,
        plot_model_score_comparison,
        plot_trend_intervention_outcome,
    )
    from seismometer.api.reports import feature_alerts, feature_summary, target_feature_summary
    from seismometer.api.templates import show_cohort_summaries

    args = fn_settings["args"]
    kwargs = fn_settings["kwargs"]
    # We need to have these here for circular import reasons.
    match function_name:
        case "feature_alerts":
            fn = feature_alerts
        case "feature_summary":
            fn = feature_summary
        case "plot_model_evaluation":
            fn = plot_model_evaluation
        case "plot_cohort_evaluation":
            fn = plot_cohort_evaluation
        case "plot_cohort_hist":
            fn = _plot_cohort_hist
        case "plot_leadtime_enc":
            fn = _plot_leadtime_enc
        case "plot_binary_classifier_metrics":
            fn = plot_binary_classifier_metrics
        case "plot_trend_intervention_outcome":
            fn = plot_trend_intervention_outcome
        case "show_cohort_summaries":
            fn = show_cohort_summaries
        case "target_feature_summary":
            fn = target_feature_summary
        case "plot_model_score_comparison":
            fn = plot_model_score_comparison
        case _:
            raise ValueError(f"Unknown function name: {function_name}")
    fn = allowed_export_names[function_name]
    fn(*args, **kwargs)


def extract_arguments(argument_names: list[str], run_settings: dict) -> dict:
    """The YAML of a handwritten config file looks like this:
    function_name:
        cohorts: # ...
        options:
            # here's where we store the "extra information" per function call
    So we get, out of options, the call parameters.

    Parameters
    ----------
    argument_names : list[str]
        Which arguments / other info (like Seismogram settings) we want to find.
    run_settings : dict
        The segment of the YAML we want to read.

    Returns
    -------
    dict
        The options and values we found in options.
    """
    if "options" in run_settings:
        return {arg: run_settings["options"][arg] for arg in argument_names if arg in run_settings["options"]}
    else:
        return {}


def do_one_manual_export(function_name: str, run_settings):
    """Perform an export from handwritten config.

    The process is roughly:
    - extract function call parameters
    - extract Seismogram info (which is set prior to function call)
    - extra cohort info (for looping purposes)

    Parameters
    ----------
    function_name : str
        The name of the plot function we will be calling.
    run_settings : dict
        The appropriate section of YAML.
    """

    from seismometer.api.plots import (  # plot_trend_intervention_outcome,
        _plot_cohort_hist,
        _plot_leadtime_enc,
        plot_binary_classifier_metrics,
        plot_cohort_evaluation,
        plot_model_evaluation,
        plot_model_score_comparison,
    )
    from seismometer.api.reports import feature_alerts, feature_summary  # target_feature_summary

    # from seismometer.api.templates import show_cohort_summaries

    match function_name:
        # These first three are super repetitive, fix them
        case "feature_alerts":
            # The only possibility here is the exclude_cols option so let's look for that.
            kwargs = extract_arguments(["exclude_cols"], run_settings)
            feature_alerts(**kwargs)
        case "feature_summary":
            kwargs = extract_arguments(["exclude_cols", "inline"], run_settings)
            feature_summary(**kwargs)
        case "plot_model_evaluation":
            kwargs = extract_arguments(["target_column", "score_column", "thresholds", "per_context"], run_settings)
            kwargs["cohort_dict"] = run_settings["cohort"]
            plot_model_evaluation(**kwargs)
        case "plot_cohort_evaluation":
            kwargs = extract_arguments(["target_column", "score_column", "thresholds", "per_context"], run_settings)
            # Now we loop over cohort columns and subgroups specified.
            for cohort in run_settings["cohorts"]:
                subgroups = run_settings["cohorts"][cohort]
                plot_cohort_evaluation(cohort_col=cohort, subgroups=subgroups, **kwargs)
        case "plot_cohort_hist":
            sg = Seismogram()
            kwargs = extract_arguments(["target", "output", "censor_threshold", "filter_zero_one"], run_settings)
            for cohort_col in run_settings["cohorts"]:
                subgroups = run_settings["cohorts"][cohort_col]
                _plot_cohort_hist(dataframe=sg.dataframe, cohort_col=cohort_col, subgroups=subgroups, **kwargs)
        case "plot_leadtime_enc":
            sg = Seismogram()
            kwargs = extract_arguments(
                [
                    "entity_keys",
                    "target_event",
                    "target_zero",
                    "score",
                    "threshold",
                    "ref_time",
                    "max_hours",
                    "x_label",
                    "censor_threshold",
                ],
                run_settings,
            )
            for cohort_col in run_settings["cohorts"]:
                subgroups = run_settings["cohorts"][cohort_col]
                _plot_leadtime_enc(dataframe=sg.dataframe, cohort_col=cohort_col, subgroups=subgroups, **kwargs)
        case "plot_binary_classifier_metrics":
            kwargs = extract_arguments(
                ["metrics", "target", "score_column", "per_context", "table_only", run_settings]
            )
            # We treat cohorts differently in automation, so we'll have to build it up specially here.
            kwargs["cohort_dict"] = run_settings["cohorts"]
            # This also takes a binary classifier metric generator as input, so we'll need to create one too.
            try:
                rho = run_settings["options"]["rho"]
            except KeyError:
                rho = None
            metric_generator = BinaryClassifierMetricGenerator(rho)
            plot_binary_classifier_metrics(metric_generator=metric_generator, **kwargs)
        case "plot_model_score_comparison":
            kwargs = extract_arguments(["target", "scores", "per_context"], run_settings)
            kwargs["cohort_dict"] = run_settings["cohorts"]
            plot_model_score_comparison(**kwargs)
        case "plot_trend_intervention_outcome":
            pass  # Possibly add metric logging for this in the first place
        case "show_cohort_summaries":
            pass


def do_manual_export(function_name: str, fn_settings: list | dict):
    """Because a handwritten config can have multiple
    sets of parameters for one function, here we differentiate
    them and provide a uniform interface.

    Parameters
    ----------
    function_name : str
        The name of the function
    fn_settings : list | dict
        Either the set of parameters, or a list of such sets.
    """
    if isinstance(fn_settings, dict):
        do_one_manual_export(function_name, fn_settings)
    elif isinstance(fn_settings, list):
        for setting in fn_settings:
            do_one_manual_export(function_name, setting)


@export
def do_metric_exports() -> None:
    """This function does automated metric exporting for
    everything specified in Seismogram.
    """
    sg = Seismogram()
    for function_name in sg._automation_info.keys():
        fn_settings = sg._automation_info[function_name]
        if function_name not in allowed_export_names:
            logger.warning(f"Unrecognized auto-export function name {function_name}. Continuing ...")
            continue
        # See if this is auto-generated or if it was hand-written.
        # Different processing will be needed in each case.
        if fn_settings is not None and "args" in fn_settings:
            do_auto_export(function_name, fn_settings)
        else:
            do_manual_export(function_name, fn_settings)
