import functools
import logging
import os
from collections import defaultdict
from inspect import signature
from pathlib import Path
from typing import Any, Callable

import yaml

from seismometer.configuration.config import ConfigProvider
from seismometer.core.decorators import export
from seismometer.core.patterns import Singleton
from seismometer.data.performance import BinaryClassifierMetricGenerator

logger = logging.getLogger("seismometer.telemetry")


automation_function_map: dict[str, dict] = {}
""" Maps the name of a function to the actual function to automate metric exporting from,
as well as information about its cohorts."""


def get_function_args(function_name: str) -> list[str]:
    """Get a list of arguments from a function.

    def foo(x: int, y: int)

    Parameters
    ----------
    function_name : str
        "foo"

    Returns
    -------
    list[str]
        ["x", "y"]
    """
    name = AutomationManager().get_function_from_export_name(function_name)
    return list(signature(name).parameters.keys())


def _transform_item(item: Any) -> Any:
    if isinstance(item, tuple):
        return list(item)
    if isinstance(item, BinaryClassifierMetricGenerator):
        return item.rho
    return item


def _call_transform(call: dict) -> dict:
    return {k: _transform_item(v) for k, v in call.items()}


@export
class AutomationManager(object, metaclass=Singleton):
    _call_history: dict[str, dict]
    """ plot function name -> {"args": args, "kwargs": kwargs } """
    _automation_info: dict[str, dict]
    """ Mapping function names to the corresponding automation settings. """
    automation_file_path: Path
    """ Where we are reading or dumping automation data. """

    def __init__(self, config_provider: ConfigProvider = None):
        """
        Parameters
        ----------
        config_provider : ConfigProvider
            Tells us where the automation file lives (for instance, metric-automation.yml)
        """
        self._call_history = defaultdict(list)
        # If no setup is provided, set up effective no-ops for everything.
        if config_provider is None:
            self._automation_info = {}
            self._metric_info = {}
            return
        self.automation_file_path = config_provider.automation_config_path
        self.load_automation_config(config_provider)
        self.load_metric_config(config_provider)

    def load_automation_config(self, config_provider: ConfigProvider) -> None:
        """Copy in the metric automation config."""
        self._automation_info = config_provider.automation_config

    def load_metric_config(self, config_provider: ConfigProvider) -> None:
        """Copy in the metric config (how much granularity, etc.)"""
        self._metric_info = config_provider.metric_config

    def store_call_params(self, fn_name, fn, args, kwargs, extra_info):
        """_summary_

        Parameters
        ----------
        fn_name : str
            The name of the function that has been called. By default it is
            the actual name of the function in the code, but if needed it can be
            set to a more readable or expected name (e.g. _plot_cohort_hist is set
            to plot_cohort_hist without an underscore).
        fn: Callable
            The actual fuinction itself.
        args : list
            The arguments the function was called with.
        kwargs : dict
            The keyword arguments the function was called with.
        """

        # Get kwargs from args, because a list of unlabeled arguments would be extremely confusing.
        sig = signature(fn)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        argument_set = dict(bound.arguments)
        self._call_history[fn_name].append(
            {"options": _call_transform(argument_set), "extra_info": extra_info(args, kwargs)}
        )

    def is_allowed_export_function(self, fn_name: str) -> bool:
        """Whether or not a function is an allowed export.

        Parameters
        ----------
        fn_name : str
            The name of the function.

        """
        return fn_name in automation_function_map

    def get_function_from_export_name(self, fn_name: str) -> Callable:
        """Get the actual function to export metrics with, from its name.
        This is not necessarily the function name itself: ex. we may use
        plot_xyz instead of _plot_xyz.

        Parameters
        ----------
        fn_name : str
            The name of the function itself.

        Returns
        -------
        Callable
            Which function we should call when we see this in automation.
        """

        # Special case: plot_binary_classifier_metrics (see autometrics.py)
        if fn_name == "plot_binary_classifier_metrics":
            from seismometer.api import plots

            return plots._autometric_plot_binary_classifier_metrics
        return automation_function_map[fn_name]["function"]

    def export_config(self, overwrite_existing=False):
        """Produce a configuration file specifying which metrics to export,
        based on which functions have been run in the notebook.

        This counts all runs of each function, but does not accommodate
        cells being deleted, because this would require some more in-depth
        access to the Jupyter frontend.

        Parameters
        ----------
        overwrite_existing : bool
            Whether to overwrite an existing automation file at the same path.
        """
        if self.automation_file_path is None:
            logger.warning("Cannot export config without a file set to export to!")
            return
        if not overwrite_existing and os.path.exists(self.automation_file_path):
            return
        with open(self.automation_file_path, "w") as automation_file:
            call_history = dict(self._call_history)
            yaml.dump(call_history, automation_file)

    def get_metric_config(self, metric_name: str) -> dict:
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

        if metric_name in self._metric_info:
            ret = self._metric_info[metric_name]
        else:
            ret = {}
        # Overwrite defaults with whatever is in the dictionary.
        return METRIC_DEFAULTS | ret


# Internal implementation -- stored separately here for mocking purposes.
def _store_call_parameters(name: str, fn: Callable, args: list, kwargs: dict, extra_info: dict) -> None:
    AutomationManager().store_call_params(name, fn, args, kwargs, extra_info)


def store_call_parameters(
    func: Callable[..., Any] = None,
    name: str = None,
    extra_params: Callable[[tuple, dict], dict] = lambda x, y: {},
    cohort_col: str = None,
    subgroups: str = None,
    cohort_dict: str = None,
) -> Callable[..., Any]:
    """_summary_

    Parameters
    ----------
    func : Callable[..., Any], optional
        The function we are wrapping.
    name : str, optional
        What name to store in the call.
        (Maybe we want to represent _internal_generate_widget_actually as just generate_widget, for example).
    extra_params : Callable[tuple, dict, dict], optional
        Extra arguments we need to reconstruct the call.
    cohort_col: str, optional
        Which function parameter represents a cohort (like Age).
    subgroups: str, optional
        Which function parameter represents a list of subgroups (like 70+, 10-20).
    cohort_dict: dict, optional
        Which function parameter represents a dict like {"Age": ["70+", "[10-20)"]}

    Returns
    -------
    Callable[..., Any]
        _description_
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        call_name = name if name is not None else fn.__name__

        @functools.wraps(fn)
        def new_fn(*args, **kwargs):
            _store_call_parameters(call_name, fn, list(args), kwargs, extra_params)
            return fn(*args, **kwargs)

        automation_function_map[call_name] = {
            k: v
            for k, v in {
                "function": fn,
                "cohort_col": cohort_col,
                "subgroups": subgroups,
                "cohort_dict": cohort_dict,
            }.items()
            if v is not None
        }
        return new_fn

    if func is not None and callable(func):
        return decorator(func)
    else:
        return decorator


@export
def initialize_otel_config(config: ConfigProvider):
    """Read all metric exporting and automation info.

    Parameters
    ----------
    config : OtherInfo
        The configuration object handed in during Seismogram initialization.
    """
    am = AutomationManager(config_provider=config)
    am.load_automation_config(config)
    am.load_metric_config(config)


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


"""
Because automation has it built in so that we can loop over cohorts, the structure
of plotting function automation depends heavily on how each function uses cohorts.

There are three main patterns in terms of plotting functions:
- (1) The plot function takes no cohorts as arguments. We can then just pass all arguments
directly to the function.
- (2) The plot function takes a cohort dictionary as an argument. We then pass the cohort dict
from run settings directly in.
- (3) The plot function takes a cohort column and a list of subgroups as arguments. We then
iterate over the cohort dict, selecting key (cohort) and value (subgroup) pairs one at a time,
calling the function with each.
"""


def _call_cohortless_function(function: Callable, arg_names: list[str], run_settings: dict):
    """Call a function in the style of (1) above.

    Parameters
    ----------
    function : Callable
        Which function we are trying to call.
    arg_names : list[str]
        The list of parameters this function takes, which we then read values for from the YAML.
    run_settings : dict
        The relevant section of YAML we are reading.
    """
    kwargs = extract_arguments(arg_names, run_settings)
    function(**kwargs)


def _call_cohort_dict_function(function: Callable, arg_names: list[str], run_settings: dict, cohort_dict: str):
    """Call a function in the style of (2) above.

    Parameters
    ----------
    function : Callable
        Which function we are trying to call.
    arg_names : list[str]
        The list of parameters this function takes, which we then read values for from the YAML.
    run_settings : dict
        The relevant section of YAML we are reading.
    cohort_dict : str
        What argument in the function is the cohort dictionary. E.g.
        def my_plot_function(cohort_dict: ...) would give us "cohort_dict".
    """
    kwargs = extract_arguments(arg_names, run_settings)
    kwargs[cohort_dict] = run_settings["cohorts"]
    function(**kwargs)


def _call_single_cohort_function(
    function: Callable, arg_names: list[str], run_settings: dict, cohort_col: str, subgroups: str
):
    """Call a function the style of (3) above.

    Parameters
    ----------
    function : Callable
        Which function we are trying to call.
    arg_names : list[str]
        The list of parameters this function takes, which we then read values for from the YAML.
    run_settings : dict
        The relevant section of YAML we are reading.
    cohort_col : str
        What argument in the function is the cohort column name.
    subgroups : str
        What argument to the function is the list of subgroups for that cohort column.
    """
    kwargs = extract_arguments(arg_names, run_settings)
    for cohort in run_settings["cohorts"]:
        subgroups_list = run_settings["cohorts"][cohort]
        kwargs_copy = kwargs.copy()
        kwargs_copy[cohort_col] = cohort
        kwargs_copy[subgroups] = subgroups_list
        function(**kwargs_copy)


def _dispatch_appropriate_call(fn: Callable, arg_info: dict, run_settings: dict, argument_names: list[str]):
    """Based on the set of parameters provided for an automated metric call, use the appropriate
    function from the three above.

    Parameters
    ----------
    fn : Callable
        The function itself we want to call.
    arg_info: dict
        The stored information about special cohort-related arguments.
    run_settings: dict
        Which settings we have loaded in from YAML for this one specific run.
    argument_names: list[str]
        What arguments are being passed to the function.
    """

    kwargs = {"function": fn, "arg_names": argument_names, "run_settings": run_settings}
    kwargs |= {k: v for k, v in arg_info.items() if k != "function"}

    if "cohort_col" in arg_info:
        _call_single_cohort_function(**kwargs)
    elif "cohort_dict" in arg_info:
        _call_cohort_dict_function(**kwargs)
    else:
        _call_cohortless_function(**kwargs)


def do_one_export(function_name: str, run_settings):
    """Perform an export from one particular config.

    The process is roughly:
    - extract function call parameters
    - extract Seismogram info (which is set prior to function call)
    - extract cohort info (for looping purposes)

    Parameters
    ----------
    function_name : str
        The name of the plot function we will be calling.
    run_settings : dict
        The appropriate section of YAML.
    """

    am = AutomationManager()
    arg_info = automation_function_map[function_name]
    argument_names = get_function_args(function_name)
    fn = am.get_function_from_export_name(function_name)
    _dispatch_appropriate_call(fn, arg_info, run_settings, argument_names)


def do_export(function_name: str, fn_settings: list | dict):
    """Because a config can have multiple
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
        do_one_export(function_name, fn_settings)
    elif isinstance(fn_settings, list):
        for setting in fn_settings:
            do_one_export(function_name, setting)


@export
def do_metric_exports() -> None:
    """This function does automated metric exporting for
    everything specified in Seismogram.
    """
    am = AutomationManager()
    for function_name in am._automation_info.keys():
        if not am.is_allowed_export_function(function_name):
            logger.warning(f"Unrecognized auto-export function name {function_name}. Continuing ...")
            continue
        fn_settings = am._automation_info[function_name]
        do_export(function_name, fn_settings)
