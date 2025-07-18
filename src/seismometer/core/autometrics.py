import functools
import logging
from collections import defaultdict
from typing import Any, Callable

import yaml

from seismometer.configuration.model import OtherInfo
from seismometer.core.decorators import export
from seismometer.core.patterns import Singleton
from seismometer.data.otel import read_otel_info
from seismometer.seismogram import Seismogram

logger = logging.getLogger("Seismometer Metric Automation")


automation_function_map: dict[str, Callable] = {}
""" Maps the name of a function to the actual function to automate metric exporting from. """


class AutomationManager(object, metaclass=Singleton):
    _call_history: dict[str, dict]
    """ plot function name -> {"args": args, "kwargs": kwargs } """
    _automation_info: dict[str, Callable]
    """ Mapping function names to the corresponding callable. """

    def __init__(self):
        self._call_history = defaultdict(list)
        self.automation_function_map = {}

    def load_automation_config(self, automation_file_path: str) -> None:
        """Load in a config from YAML.

        Parameters
        ----------
        automation_file_path : str
            Where the automation file lives (metric-automation.yml)
        """
        try:
            with open(automation_file_path, "r") as automation_file:
                self._automation_info = yaml.safe_load(automation_file)
        except (FileNotFoundError, TypeError):  # TypeError is for when the path is None
            self._automation_info = {}

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

        self._call_history[fn_name].append({"args": args, "kwargs": kwargs, "extra_info": extra_info(args, kwargs)})
        automation_function_map[fn_name] = fn

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

            return plots._plot_binary_classifier_metrics
        return automation_function_map[fn_name]


# Internal implementation -- stored separately here for mocking purposes.
def _store_call_parameters(name: str, fn: Callable, args: list, kwargs: dict, extra_info: dict) -> None:
    AutomationManager().store_call_params(name, fn, args, kwargs, extra_info)


def store_call_parameters(
    func: Callable[..., Any] = None, name: str = None, extra_params: Callable[[tuple, dict], dict] = lambda x, y: {}
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

        automation_function_map[call_name] = fn
        return new_fn

    if func is not None and callable(func):
        return decorator(func)
    else:
        return decorator


@export
def initialize_otel_config(config: OtherInfo):
    """Read all metric exporting and automation info.

    Parameters
    ----------
    config : OtherInfo
        The configuration object handed in during Seismogram initialization.
    """
    global OTEL_INFO
    OTEL_INFO = read_otel_info(config.usage_config)
    AutomationManager().load_automation_config(config.automation_config)


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
        call_history = sg._call_history
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


def _call_cohort_dict_function(function: Callable, arg_names: list[str], run_settings: dict, cohort_arg_name: str):
    """Call a function in the style of (2) above.

    Parameters
    ----------
    function : Callable
        Which function we are trying to call.
    arg_names : list[str]
        The list of parameters this function takes, which we then read values for from the YAML.
    run_settings : dict
        The relevant section of YAML we are reading.
    cohort_arg_name : str
        What argument in the function is the cohort dictionary. E.g.
        def my_plot_function(cohort_dict: ...) would give us "cohort_dict".
    """
    kwargs = extract_arguments(arg_names, run_settings)
    kwargs[cohort_arg_name] = run_settings["cohorts"]
    function(**kwargs)


def _call_single_cohort_function(
    function: Callable, arg_names: list[str], run_settings: dict, cohort_arg_name: str, subgroup_arg_name: str
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
    cohort_arg_name : str
        What argument in the function is the cohort column name.
    subgroup_arg_name : str
        What argument to the function is the list of subgroups for that cohort column.
    """
    kwargs = extract_arguments(arg_names, run_settings)
    for cohort in run_settings["cohorts"]:
        subgroups = run_settings["cohorts"][cohort]
        kwargs_copy = kwargs.copy()
        kwargs_copy[cohort_arg_name] = cohort
        kwargs_copy[subgroup_arg_name] = subgroups
        function(**kwargs_copy)


def _dispatch_appropriate_call(kwargs: dict):
    """Based on the set of parameters provided for an automated metric call, use the appropriate
    function from the three above.

    Parameters
    ----------
    kwargs : dict
        All function arguments
    """
    if "subgroup_arg_name" in kwargs:
        _call_single_cohort_function(**kwargs)
    elif "cohort_arg_name" in kwargs:
        _call_cohort_dict_function(**kwargs)
    else:
        _call_cohortless_function(**kwargs)


_call_information = {
    "feature_alerts": {"arg_names": ["exclude_cols"]},
    "feature_summary": {"arg_names": ["exclude_cols", "inline"]},
    "plot_model_evaluation": {
        "arg_names": ["target_column", "score_column", "thresholds", "per_context"],
        "cohort_arg_name": "cohort_dict",
    },
    "plot_cohort_evaluation": {
        "arg_names": ["target_column", "score_column", "thresholds", "per_context"],
        "cohort_arg_name": "cohort_col",
        "subgroup_arg_name": "subgroups",
    },
    "plot_cohort_lead_time": {
        "arg_names": ["event_column", "score_column", "threshold"],
        "cohort_arg_name": "cohort_col",
        "subgroup_arg_name": "subgroups",
    },
    "plot_binary_classifier_metrics": {
        "arg_names": ["metrics", "target", "score_column", "per_context", "table_only", "rho"],
        "cohort_arg_name": "cohort_dict",
    },
    "plot_model_score_comparison": {
        "arg_names": ["target", "scores", "per_context"],
        "cohort_arg_name": "cohort_dict",
    },
}


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

    arg_info = _call_information[function_name]
    arg_info["function"] = fn
    arg_info["run_settings"] = run_settings
    _dispatch_appropriate_call(arg_info)


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
    am = AutomationManager()
    for function_name in am._automation_info.keys():
        if not am.is_allowed_export_function(function_name):
            logger.warning(f"Unrecognized auto-export function name {function_name}. Continuing ...")
            continue
        fn_settings = am._automation_info[function_name]
        # See if this is auto-generated or if it was hand-written.
        # Different processing will be needed in each case.
        if fn_settings is not None and "args" in fn_settings:
            do_auto_export(function_name, fn_settings)
        else:
            do_manual_export(function_name, fn_settings)
