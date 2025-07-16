import logging
import os
import socket
import sys
from typing import Callable

import yaml

from seismometer.configuration.model import OtherInfo
from seismometer.core.decorators import export
from seismometer.data.performance import BinaryClassifierMetricGenerator
from seismometer.seismogram import Seismogram

logger = logging.getLogger("Seismometer OpenTelemetry")

try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.metrics import Meter, set_meter_provider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource

    TELEMETRY = True
except ImportError:
    # No OTel.
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


# If we don't have telemetry, we make everything into a no-op.
if not TELEMETRY:

    def noop(*args, **kwargs):
        pass

    class ExportManager:  # noqa: F811
        def __init__(self, *args, **kwargs):
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

    args = fn_settings["args"]
    kwargs = fn_settings["kwargs"]
    # We need to have these here for circular import reasons.
    fn = Seismogram().get_function_from_export_name(function_name)
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

    sg = Seismogram()
    fn = sg.get_function_from_export_name(function_name)

    match function_name:
        # These first three are super repetitive, fix them
        case "feature_alerts":
            # The only possibility here is the exclude_cols option so let's look for that.
            kwargs = extract_arguments(["exclude_cols"], run_settings)
            fn(**kwargs)
        case "feature_summary":
            kwargs = extract_arguments(["exclude_cols", "inline"], run_settings)
            fn(**kwargs)
        case "plot_model_evaluation":
            kwargs = extract_arguments(["target_column", "score_column", "thresholds", "per_context"], run_settings)
            kwargs["cohort_dict"] = run_settings["cohort"]
            fn(**kwargs)
        case "plot_cohort_evaluation":
            kwargs = extract_arguments(["target_column", "score_column", "thresholds", "per_context"], run_settings)
            # Now we loop over cohort columns and subgroups specified.
            for cohort in run_settings["cohorts"]:
                subgroups = run_settings["cohorts"][cohort]
                fn(cohort_col=cohort, subgroups=subgroups, **kwargs)
        case "plot_cohort_lead_time":
            kwargs = extract_arguments(["event_column", "score_column", "threshold"], run_settings)
            for cohort_col in run_settings["cohorts"]:
                subgroups = run_settings["cohorts"][cohort_col]
                fn(cohort_col=cohort_col, subgroups=subgroups, **kwargs)
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
            fn(metric_generator=metric_generator, **kwargs)
        case "plot_model_score_comparison":
            kwargs = extract_arguments(["target", "scores", "per_context"], run_settings)
            kwargs["cohort_dict"] = run_settings["cohorts"]
            fn(**kwargs)
        case "plot_trend_intervention_outcome":
            pass  # Possibly add metric logging for this in the first place


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
        if not sg.is_allowed_export_function(function_name):
            logger.warning(f"Unrecognized auto-export function name {function_name}. Continuing ...")
            continue
        # See if this is auto-generated or if it was hand-written.
        # Different processing will be needed in each case.
        if fn_settings is not None and "args" in fn_settings:
            do_auto_export(function_name, fn_settings)
        else:
            do_manual_export(function_name, fn_settings)
