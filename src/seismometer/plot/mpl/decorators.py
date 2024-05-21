# -*- coding: utf-8 -*-
"""
This module adds the plot decorator to wrap axis and saving for different plots.

Examples:
Given a plot function which takes an axis and filename::

    @model_plot
    new_model_plot(some, variables, for, plot, axis=None, filename=None, **kwargs)

The wrapped new_model_plot will handle axis instantiation and figure saving in a
uniform way.
"""

from functools import wraps
from inspect import Parameter as param
from inspect import signature
from typing import Any, Callable

import matplotlib as matplotlib
import matplotlib.pyplot as plt

from seismometer.core._decorators import disk_cached_function, export

from ._util import save_figure


def is_disp_axis(ax: plt.Axes) -> bool:
    """Infer if the axis is used for display."""
    return bool(ax.get_xlabel() or ax.get_ylabel())


def can_draw_empty_perf(plot_fn: Callable) -> Any:
    """Decorator to handle empty performance data."""

    @wraps(plot_fn)
    def plot_wrapper(*args, **kwargs):
        # If an axis is being passed, increment the line count
        ax = kwargs.get("axis", None)
        if ax is not None:
            data = args[0]
            if data is None or data.empty:
                ax.plot([], [])
                return None
        return plot_fn(*args, **kwargs)

    return plot_wrapper


def model_plot(plot_fn: Callable) -> Any:
    """
    A model_plot is any function that should include axis and filename parameters.
    If an axis is given, we will use it, or if not we will create one based on plt.subplots.

    If we are given a filename, and we have created our own axis for plotting, we will also save
    the image as expected.

    Parameters
    ----------
    axis
        Optional matplotlib axis for the current plot; will create a new figure if none is passed in.
    filename
        Optional file name to save output; if no filename is passed in nothing will be saved.
        Assumes that the path exists.
    <arg_name>
        Any additional parameters needed by your plot must be named, kwargs will not be sent to wrapped function.
    kwargs
        Optional arguments passed directly to matplotlib subplots if no axis is specified.

    Notes
    -----
    Parameters are the expected function parameters.
    All named variables will be passed to wrapped function, and all remaining parameters will be passed on to
    matplotlib.pyplot.subplots. Important: If subplots does not accept your kwargs, it will crash!
    """

    @wraps(plot_fn)
    def plot_wrapper(*args, **kwargs):
        """
        This doc string will be overridden by the source function.
        """
        named_vars = signature(plot_fn).parameters
        filename = kwargs.pop("filename", None)

        plotargs = {k: v for k, v in kwargs.items() if k in named_vars and k not in ("filename", "axis")}

        if "axis" in named_vars:
            axis = kwargs.pop("axis", named_vars["axis"].default)
            if axis is None or axis == param.empty:
                figsize = kwargs.pop("figsize", (5, 5))
                subplotsargs = {k: v for k, v in kwargs.items() if k not in named_vars}
                fig, ax = plt.subplots(figsize=figsize, **subplotsargs)
            else:
                ax = axis
            source_value = plot_fn(*args, axis=ax, filename=filename, **plotargs)
        else:
            axis = None
            source_value = plot_fn(*args, filename=filename, **plotargs)

        if axis is None:
            if filename is not None:
                save_figure(filename)
            plt.show()

        return source_value

    return plot_wrapper
