# -*- coding: utf-8 -*-
"""
This module adds the plot decorator to wrap axis and saving for different plots.

Examples:
Given a plot function which takes an axis:

    @model_plot
    new_model_plot(some, variables, for, plot, axis=None, **kwargs)

The wrapped new_model_plot will handle axis instantiation and figure creation in a uniform way.
Results will be ether an svg string or a figure (if used by another plot)
"""

from functools import wraps
from inspect import Parameter as param
from inspect import signature
from typing import Any, Callable

import matplotlib as matplotlib
import matplotlib.pyplot as plt
from IPython.display import SVG

from seismometer.core.decorators import export

from ._util import to_svg


def render_as_svg(plot_fn: Callable[..., plt.Figure]) -> Callable[..., SVG]:
    """
    Given a plot function that retuns a figure, render to SVG and close the Figure

    Parameters
    ----------
    plot_fn : Callable[..., plt.Figure]
        function that returns a matplotlib figure

    Returns
    -------
    Callable[..., SVG]
        function that returns and SVG object
    """

    @wraps(plot_fn)
    def plot_wrapper(*args, **kwargs):
        plot_fn(*args, **kwargs)
        svg = to_svg()
        plt.close()
        return svg

    return plot_wrapper


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


def model_plot(plot_fn: Callable) -> Callable[..., plt.Figure | str]:
    """
    A model_plot is any function that potentially needs to create its own axis.
    If an axis is given, we will use it, or if not we will create one based on plt.subplots.

    Parameters
    ----------
    axis
        Optional matplotlib axis for the current plot; will create a new figure if none is passed in.
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

    sig = signature(plot_fn)
    if sig.return_annotation != plt.Figure:
        raise TypeError("The source function must return a matplotlib figure object.")
    named_vars = sig.parameters

    @wraps(plot_fn)
    def plot_wrapper(*args, **kwargs):
        """
        This doc string will be overridden by the source function.
        """
        is_primary_plot = True

        plotargs = {k: v for k, v in kwargs.items() if k in named_vars and k not in ("", "axis")}

        if "axis" in named_vars:
            axis = kwargs.pop("axis", named_vars["axis"].default)
            if axis is None or axis == param.empty:
                # create the default axis
                figsize = kwargs.pop("figsize", (4, 4))
                subplotsargs = {k: v for k, v in kwargs.items() if k not in named_vars}
                fig, ax = plt.subplots(figsize=figsize, **subplotsargs)
            else:
                is_primary_plot = False
                ax = axis
            source_value = plot_fn(*args, axis=ax, **plotargs)
        else:
            source_value = plot_fn(*args, **plotargs)

        if not isinstance(source_value, plt.Figure):
            raise TypeError("The function must return a matplotlib figure object.")

        if is_primary_plot:
            svg_format = to_svg()
            plt.close()
            return svg_format
        return source_value

    return plot_wrapper
