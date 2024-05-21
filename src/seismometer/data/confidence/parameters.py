import abc
from copy import deepcopy
from typing import Callable

from . import calculations as calc

_SUPPORTED_PLOTS = ["roc", "pr", "metric"]


class ConfidenceParam(abc.ABC):
    """
    An object that holds parameters for confidence interval and confidence region calculations.
    Inherit from this class to define defaults for a particular plot or calculation.

    Parameters
    ----------
    conf : None | float | dict
        None means no confidence values will be plotted. A float will use default methods at the given confidence
        level.
        Entries of 'level' in a dict will be for the confidence level.
        Entries of 'region' and 'interval' are functions whose signatures vary based on child class usage. See the
        defaults for examples.

        Values for the level must be bounded by 0 and 1 (exclusive), where 0.95 represents a 95% confidence level.
    """

    # Method signatures for these depend on the nature of the calculation and may vary with implementation
    region: Callable
    interval: Callable
    level: float

    _default_region: Callable = None
    _default_interval: Callable = None
    _default_level: float = 0.95

    _region_allowed: bool = True
    _interval_allowed: bool = True

    @abc.abstractmethod
    def __init__(self, conf: None | float | dict) -> None:
        self.region = self._default_region
        self.interval = self._default_interval
        self.level = self._default_level
        self.apply_conf(_get_specific_conf_dict(conf))
        super().__init__()

    def apply_conf(self, conf: dict) -> None:
        """
        Takes a confidence dictionary and applies the relevant attributes.
        """
        if "level" in conf and _validate_level(conf["level"]):
            self.level = conf["level"]
        # Built this way for ease in future enhancements
        for func in ("region", "interval"):
            if func not in conf:
                continue
            if not getattr(self, f"_{func}_allowed"):
                raise ValueError(f"{type(self)} does not support the parameter '{func}'")
            setattr(self, func, conf[func])


class ROCConfidenceParam(ConfidenceParam):
    """
    Specifies the CI and CR calculation for ROC curves.
    """

    def __init__(self, conf: None | float | dict) -> None:
        # Defaults are assigned this way due to semantics of python classes and callable member fields
        # This allows for consistency between passed functions vs default ones
        self._default_region = calc.simultaneous_joint_confidence
        self._default_interval = calc.hanley_mcneill_confidence
        super().__init__(conf)


class PRConfidenceParam(ConfidenceParam):
    """
    Specifies the CI calculation for Precision-recall curves.
    """

    def __init__(self, conf: None | float | dict) -> None:
        # Future work may enable regions for this plot
        self._region_allowed: bool = False
        self._default_interval = calc.logit_interval
        super().__init__(conf)


class PerformanceMetricConfidenceParam(ConfidenceParam):
    """
    Specifies the CR calculation for the performance metric curves (e.g. PPV vs threshold).
    """

    def __init__(self, conf: None | float | dict) -> None:
        self._interval_allowed: bool = False
        self._default_region = calc.agresti_coull_interval
        super().__init__(conf)


def confidence_dict(conf: None | float | dict) -> dict:
    """
    Converts a flexible confidence instruction for a function like seismometer.plot.mpl.evaluation() into a dict of
      confidence parameters that can each be used to create a ConfidenceParam instance.

    Parameters
    -----------
    conf: None | float | dict
        If None, returns a dictionary of None entries for the delegate plotting functions.
        If a float, returns a dictionary of 'level' entries for the plotting functions.
        Entries of 'level' in a dict will be for the confidence level.
        Entries of 'region' and 'interval' are functions that must match the method signature of the default function
        they are replacing. See `calculations.simultaneous_joint_confidence` and `calculations.Logit` for examples.
    """
    if isinstance(conf, float):
        return _confidence_dict_from_float(conf)
    if conf is None:
        return _confidence_dict_from_none()

    apply_to_all = {} if "all" not in conf else _get_specific_conf_dict(conf["all"])
    conf = _mergecopy(_confidence_dict_from_none(), conf)
    a = {key: _mergecopy(apply_to_all, _get_specific_conf_dict(conf[key])) for key in _SUPPORTED_PLOTS}
    return a


def _get_specific_conf_dict(conf: None | float | dict) -> dict:
    """Given the flexible conf argument, parse it according to its actual type"""
    if conf is None:
        return {}
    if isinstance(conf, (float, int)):
        return {"level": conf}
    return conf


def _confidence_dict_from_float(conf: float) -> dict:
    """Create a dict of confidence parameters for use in a function like seismometer.plot.mpl.evaluation()"""
    return {key: {"level": conf} for key in _SUPPORTED_PLOTS}


def _confidence_dict_from_none() -> dict:
    """Create a dict of confidence parameters for use in a function like seismometer.plot.mpl.evaluation()"""
    return {key: None for key in _SUPPORTED_PLOTS}


def _validate_level(level: float) -> bool:
    """Validate a float 'level' parameter"""
    if level <= 0 or level >= 1:
        raise ValueError("Specified confidence level must be between 0 and 1 (exclusive).")
    return True


def _mergecopy(a: dict, b: dict) -> dict:
    """Create a merge dictionary starting with a copy of dictionary 'a', and merging in dictionary 'b'"""
    # Python passes dicts by reference and we don't want to overwrite anything
    a = deepcopy(a)
    for key in b.keys():
        a[key] = b[key]
    return a
