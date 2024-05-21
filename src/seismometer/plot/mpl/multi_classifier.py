# This module primarily supports multiple cohorts of binary classification models
from numbers import Number
from typing import Callable, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import seismometer.plot.mpl._lines as lines

from . import binary_classifier as mpPlots
from ._util import add_unseen, axis_clear, cohort_legend
from .decorators import export, model_plot

Axes = plt.Axes | plt.Subplot
SeriesOrArray = pd.Series | np.ndarray
CENSOR_THRESHOLD = 10


@export
@model_plot
def cohort_evaluation_vs_threshold(
    stats: pd.DataFrame,
    cohort_feature: str,
    *,
    splits: Optional[list] = None,
    labels: Optional[list[str]] = None,
    highlight: Optional[list[float]] = None,
    filename: Optional[str] = None,
) -> None:
    """
    Creates a 2x3 grid of individual performance metrics across cohorts.

    Plots include Sensitivity, Flagged, PPV, Specificity, NPV vs Thresholds.
    Includes a legend with cohort size.

    Parameters
    ----------
    stats : pd.DataFrame
        Table of performance metrics, of the form given by calculate_bin_stats.
    cohort_feature : str
        Display string for the cohort groupings.
    splits : Optional[list], optional
        Primarily for non-categorical cohort info, a subset of cohort values to plot, by default None; plots all.
    labels : Optional[list[str]], optional
        Optional list of display labels for cohorts, by default None; uses the cohort category value.
    highlight : Optional[list[float]], optional
        An optional list of thresholds to highlight on the plots, by default None.
    filename : Optional[str], optional
        Filename to save the plot, by default None.

    """
    cohort_col = "cohort"

    # validate labels against cat values
    if labels is None:
        labels = stats[cohort_col].cat.categories

    if splits is not None:  # Unneeded for categoricals
        stats = add_unseen(stats.loc[stats[cohort_col].isin(splits)], col=cohort_col)

    # Plot
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, fig)

    metrics = ["Sensitivity", "Flagged", "PPV", "Specificity", "NPV"]  # determines ordering of plots
    func_kws = {"highlight": highlight}
    for i, metric in enumerate(metrics):
        func_kws["metric"] = metric

        if i == 4:  # skip subplot in gridspec 4 to allow legend to go there
            i = 5

        cohorts_overlay(stats.copy(), mpPlots.metric_vs_threshold, axis=fig.add_subplot(gs[i]), func_kws=func_kws)

    legend_axes = fig.add_subplot(gs[4])
    if highlight:
        lines.vertical_threshold_lines(
            legend_axes, highlight, color_alerts=True, legend_position="upper center", plot=False, show_marker=False
        )

    cohort_legend(stats, legend_axes, cohort_feature)

    fig.suptitle(f"Model Performance Metrics on {cohort_feature} across Thresholds")
    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])


@export
@model_plot
def leadtime_whiskers(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    xmax: Optional[Number] = None,
    axis: Optional[plt.Axes] = None,
    filename: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Box and whisker plot of leadtime across cohorts.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the leadtime and cohort information.
    x_col : str
        The leadtimes to be plotted along the x-axis.
    y_col : str
        The cohort values to split data.
    xmax : Optional[Number], optional
        An optional maximum leadtime to display, by default None.
    axis : Optional[plt.Axes], optional
        The matplotlib axis to draw, by default None; creates a new figure.
    filename : Optional[str], optional
        Filename to save the plot, by default None.
    title : Optional[str], optional
        An override title for the plot, by default None; uses the y_col display name to derive the title.
    """
    M = data[y_col].nunique()
    if axis is None:
        fig, axis = plt.subplots(figsize=(20, min(10, 2 * M)))
    title = title or f"Leadtime across {y_col.replace('_', ' ')}"

    sns.boxplot(
        data=data, x=x_col, y=data[y_col].cat.remove_unused_categories(), hue=data[y_col], ax=axis, saturation=1
    )

    if xmax is not None:
        axis.set_xlim(-abs(xmax) - 0.01, 0)
    axis.set_title(title)
    axis.set_ylabel(y_col.replace("_", " "))
    axis.set_xlabel(x_col.replace("_", " "))


##########################################################################
#  Multi-plot wrappers for performance plots
##########################################################################


@export
@model_plot
def cohorts_overlay(
    data: pd.DataFrame,
    plot_func: Callable,
    axis: Optional[Axes] = None,
    labels: Optional[list[str]] = None,
    func_kws: Optional[dict] = None,
    censor_threshold: int = None,
    filename=None,
) -> None:
    """
    Uses a passed plotting function to plot a line per given split.

    Parameters
    ----------
    data : pd.DataFrame
        Data in format of either get_cohort_data[0] OR get_cohort_performance_data.
    plot_func : Callable (@model_plot decorator compatible)
        Should accept data in the first parameter and all other parameters as keyword arguments.
        Axis must be passed in by keyword axis, and has special handling.
    axis : Optional[Axes], default=None
        Matplotlib axis on which to plot, creates a new figure if None.
    labels : Optional[list[str]], default=None
        List of labels to optionally pass to the plot_func callable, function must be able to handle
        a kwarg of 'label'.
    func_kws : Optional[dict], default=None
        A dictionary to pass to callable. Function must be able to handle all keywords.
    censor_threshold : int, default=None
        Minimum number of samples to plot a line, otherwise it will be censored.
    filename : Optional[str], optional
        Filename to save the plot, by default None.

    """
    if censor_threshold is None:
        censor_threshold = CENSOR_THRESHOLD
    if axis is None:
        axis = plt.subplots(figsize=(5, 5))[1]

    func_kws = func_kws or {}

    grouped_data = data.groupby("cohort", observed=False)

    num_labels = sum([0 if len(group_stats.index) < censor_threshold else 1 for _, group_stats in grouped_data])

    if num_labels > 3:
        func_kws["conf"] = None

    # Loop over cohorts
    for label, group_stats in grouped_data:
        if labels is not None and label not in labels:
            # May just be unselected, this forces it into censored
            group_stats = group_stats.head(0)

        if len(group_stats.index) < censor_threshold:
            # Increment cyclers with empty line
            plot_func(None, None, axis=axis, label=label)
            continue

        plot_func(group_stats, axis=axis, label=label, **func_kws)

    if axis is None:
        plt.show()


@export
@model_plot
def cohorts_vertical(
    df: pd.DataFrame,
    plot_func: Callable,
    gs: Optional[gridspec.GridSpec] = None,
    labels: Optional[list[str]] = None,
    func_kws: Optional[dict] = None,
    filename=None,
) -> None:
    """
    Uses a passed plotting function to plot a line per given split.

    Parameters
    ----------
    df : pd.DataFrame
        Data in format of get_cohort_data[0]
        Currently expects a pandas Dataframe that has three columns: split, true, prob.
    plot_func : Callable (@model_plot decorator compatible)
        Plotting function that takes y_true and y_proba as first two inputs, and allows
        axis to be passed in by keyword axis.
    gs : Optional[Axes], default=None
        Specific gridsearch subplot spec on which to plot, creates a new figure if None.
    labels : Optional[list[str]], default=None
        List of labels to optionally pass to the plot_func callable, function must be able to handle
        a kwarg of 'label'.
    func_kws : Optional[dict], default=None
        A dictionary to pass to callable. Function must be able to handle all keywords.
    filename : Optional[str], optional
        filename to save the plot, by default None.

    """
    cohort_count = (df["cohort"].value_counts() > 0).sum()
    if cohort_count == 0:
        raise ValueError("No cohorts had data to plot")

    if gs is None:
        fig = plt.figure(figsize=(5, 5))
        gs1 = gridspec.GridSpec(cohort_count, 1, fig)
    else:
        fig = plt.gcf()
        gs1 = gridspec.GridSpecFromSubplotSpec(cohort_count, 1, subplot_spec=gs)

    # TODO: differentiate override labels vs subset of labels

    active_ix = 0
    for label, group_data in df.groupby("cohort", observed=True):
        # drop missing labels
        group_data = group_data.loc[group_data.iloc[:, 0].notna()]
        if len(group_data.index) == 0:
            continue
        if labels is not None and active_ix < len(labels):
            label = labels[active_ix]
        axis = fig.add_subplot(gs1[active_ix])
        _plot_one_vertical(group_data, plot_func, axis, label, func_kws)

        # Hardcoded behavior around our legend position
        if active_ix < cohort_count - 1:
            axis_clear(axis)

        active_ix += 1


def _plot_one_vertical(
    data: pd.DataFrame,
    plot_func: Callable,
    axis: Axes,
    label: Optional[str] = None,
    func_kws: Optional[dict] = None,
) -> None:
    """
    Plots a single subpanel of a vertical subplot (called by plot_cohorts_vertical)

    Parameters
    ----------
    data : np.ndarray
        2d Array of data - true and prediction pairs.
    plot_func : Callable (@model_plot decorator compatible)
        Plotting function that takes two series, and allows
        axis to be passed in by keyword axis.
    axis : Axes
        Matplotlib axis on which to plot.
    label : Optional[str], default=None
        Name of subplot, will print this to the right of the figure if specified.
    func_kws : Optional[dict], default=None
        A dictionary to pass to callable. Function must be able to handle all keywords.
    """
    func_kws = func_kws or {}

    plot_func(data.iloc[:, 0].astype(int), data.iloc[:, 1], axis=axis, **func_kws)
    axis.set_xlim(0, 1)
    if label:
        axis.text(1.01, 0.5, s=label, transform=axis.transAxes)
