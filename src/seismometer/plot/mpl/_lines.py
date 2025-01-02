# -*- coding: utf-8 -*-
import logging
from numbers import Number
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seismometer.data.confidence import PerformanceMetricConfidenceParam
from seismometer.data.performance import as_percentages, as_probabilities

from ._util import plot_diagonal, plot_horizontal, plot_polygon
from ._ux import alert_colors, area_colors, text_colors

SeriesOrArray = pd.Series | np.ndarray
DataFrameOrArray = pd.DataFrame | np.ndarray
Axes = plt.Axes | plt.Subplot


def vertical_threshold_lines(axis, highlight, color_alerts=True, legend_position=None, plot=True, show_marker=True):
    colors = [None]
    highlight = sorted(highlight, reverse=True)
    if color_alerts:
        colors = [alert_colors[i % len(alert_colors)] for i in range(len(highlight))]
    handles = []
    for i, x in enumerate(highlight):
        c = colors[i % len(colors)]
        if plot:
            axis.axvline(x=x, color=c, linestyle="--")
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=c,
                linestyle="--",
                marker="o" if show_marker else None,
                label=f"Threshold of {highlight[i]*100:.0f}",
            )
        )
    if legend_position:
        axis.add_artist(axis.legend(handles=handles, loc=legend_position))


# region Axis plotting


def roc_plot(axis, fpr, tpr, label=None) -> None:
    """Actual plot call and axis definition, specifies 0-1.01 square."""
    plot_diagonal(axis)
    axis.plot(fpr, tpr, label=label)
    axis.legend(loc="lower right")
    axis.set_xlim(0, 1.01)
    axis.set_xlabel("1 - Specificity")

    axis.set_ylim(0, 1.01)
    axis.set_ylabel("Sensitivity")


def recall_condition_plot(axis, ppcr, recall, prevalence, show_reference=False) -> None:
    """
    Plots line for recall condition on a 0-1.01 square.
    Has optional flag for reference shading.
    """
    if show_reference:
        plot_polygon(axis, [0, prevalence, 1], [0, 1, 1])
    axis.plot(ppcr, recall)
    axis.set_xlim(0, 1.01)
    axis.set_xlabel("Flag Rate")

    axis.set_ylim(0, 1.01)
    axis.set_ylabel("Sensitivity")


def reliability_plot(axis, mean_predicted, fraction_positive, label=None) -> None:
    """Actual plot call and axis definition. Specifies 0-1.01 square."""
    plot_diagonal(axis)
    axis.plot(mean_predicted, fraction_positive, "x-", label=label)
    axis.set_xlim(0, 1.01)
    axis.set_xlabel("Predicted Probability")

    axis.set_ylim(0, 1.01)
    axis.set_ylabel("Observed Rate")


def hist_stacked(axis, probabilities, labels, show_legend=True, bins=20) -> None:
    """Actual plot call and axis definition. Specifies 0-1.01 xlim. Has optional flag for legend."""
    axis.hist(probabilities, bins=bins, label=labels, stacked=True)
    if show_legend:
        axis.legend(loc="lower right")

    axis.set_xlim([0, 1.01])
    axis.set_xlabel("Predicted Probability")

    # do not call set_ylim - count can be arbitrarily large
    axis.set_ylabel("Count")


def hist_single(
    axis: Axes,
    data_series: SeriesOrArray,
    label: str,
    bins: list[float] | int = 20,
    scale: int = 1,
) -> None:
    """Actual plot call and axis definition of a single histogram line using step and fill_between."""
    if scale <= 0:
        scale = 1

    hist, x_data = np.histogram(data_series.dropna(), bins=bins)
    y_data = np.insert(hist, 0, hist[0]) / scale  # Plotting needs a duplicate of first value

    p = axis.step(x_data, y_data, where="pre", label=label, zorder=100)
    c = p[-1].get_color()
    axis.fill_between(x_data, y_data, step="pre", alpha=0.05, color=c, zorder=1)
    axis.set_xlabel("Predicted Probability")
    if scale == 1:
        axis.set_ylabel("Count")
    else:
        axis.set_ylabel("Proportion")

    # For now return y_data for legend positioning
    return y_data


def single_ppv(axis, thresholds, precision, precision_threshold) -> None:
    """Actual plot call and axis definition. Specifies 0-1.01 xlim."""
    thresholds = as_probabilities(thresholds)
    axis.plot(thresholds, precision)

    if precision_threshold is not None:
        plot_horizontal(axis, precision_threshold)
    axis.set_xlim([0, 1.01])
    axis.set_xlabel("Threshold")

    axis.set_ylim([0, 1.01])
    axis.set_ylabel("PPV")


def ppv_sensitivity_curve(axis, recall, precision, label=None):
    """Actual plot call and axis definition. Specifies 0-1.01 xlim. Allows custom label."""
    axis.step(recall, precision, where="post", label=label)
    axis.legend(loc="upper left")

    axis.set_xlim([0, 1.01])
    axis.set_xlabel("Sensitivity")

    axis.set_ylim([0, 1.01])
    axis.set_ylabel("PPV")


def metric_vs_threshold_curve(axis, metric, thresholds, label="Metric") -> None:
    """Actual plot call and axis definition. Specifies 0-1.01 xlim."""
    thresholds = as_probabilities(thresholds)
    axis.plot(thresholds, metric)

    axis.set_xlim([0, 1.01])
    axis.set_xlabel("Threshold")

    axis.set_ylim([0, 1.01])
    axis.set_ylabel(label)


def performance_metrics_plot(axis, sensitivity, specificity, ppv, thresholds) -> None:
    """Actual plot call and axis definition. Specifies 0-1.01 xlim."""
    thresholds = as_probabilities(thresholds)

    axis.plot(thresholds, sensitivity, label="Sensitivity")
    axis.plot(thresholds, specificity, label="Specificity")
    axis.plot(thresholds, ppv, label="PPV")
    axis.legend(loc="lower right")

    axis.set_xlim([0, 1.01])
    axis.set_xlabel("Threshold")
    axis.set_ylim([0, 1.01])
    axis.set_ylabel("Metric")


def roc_region_plot(axis, lower_x, lower_y, upper_x, upper_y) -> None:
    """Plots an ROC confidence region bounded between the curves specified by (x1,y1) and (x2,y2)."""

    region_xs = np.concatenate([lower_x, [1], upper_x[::-1], [0]])
    region_ys = np.concatenate([lower_y, [1], upper_y[::-1], [0]])
    axis.fill(region_xs, region_ys, alpha=0.25)


def performance_confidence(axis, perf_stats, conf, metric, color=None) -> None:
    """Private helper function for plotting performance metric confidence regions."""
    conf = PerformanceMetricConfidenceParam(conf)

    # Each metric has a different definition of "N" for the sake of calculating confidence
    if metric == "Sensitivity":
        metric_N = perf_stats["TP"] + perf_stats["FN"]
    elif metric == "Specificity":
        metric_N = perf_stats["TN"] + perf_stats["FP"]
    elif metric == "PPV":
        metric_N = perf_stats["TP"] + perf_stats["FP"]
    elif metric == "NPV":
        metric_N = perf_stats["TN"] + perf_stats["FN"]
    elif metric == "Flag Rate":
        return
    else:
        raise ValueError(f"Tried to plot a confidence region on an unsupported metric '{metric}'.")

    perf_stats.loc[:, metric], lower, upper = conf.region(conf, perf_stats[metric], metric_N)
    performance_region_plot(axis, lower, upper, perf_stats["Threshold"], color=color)


def performance_region_plot(axis, lower, upper, thresholds, color=None) -> None:
    """Plots a confidence region around the specified performance statistic."""

    axis.fill_between(thresholds, lower, upper, alpha=0.25, color=color)


# endregion


# region Plotting Extras


def get_last_line_color(axis):
    lines = axis.get_lines()
    if len(lines) == 0:
        return None
    return lines[-1].get_color()


# endregion
# region Private Helpers
def _add_radial_score_thresholds(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    thresholds: list[Number],
    Q: int = 1,
    colorIx: int = 0,
) -> None:
    """
    Add threshold annotations on points as if it were a quarter-circle.

    Expected use is for ROC curves with either scores or n_scores roughly spaced along the curve.

    Parameters
    ----------
    axis : plt.Axes
        Axis to modify.
    x : np.ndarray
        Array of x-values.
    y : np.ndarray
        Array of y-values.
    labels : list[str]
        Array of labels to display.
    thresholds : list[Number]
        List of threshold points to label.
    Q : int, optional
        The quadrant 1-4 to assume for offsets, by default 1.
    colorIx : int, optional
        Index of line being annotated in order to match color, by default 0.
    """
    if labels is None:
        return

    # For safety convert to handle indexing lists
    x = np.array(x)
    y = np.array(y)
    labels = np.array(labels)

    if x.max() > 1 or x.min() < 0:
        logging.warning(
            "Adding labels when some x-values are outside range of [0-1] " "may produce unexpected results."
        )
    if y.max() > 1 or y.min() < 0:
        logging.warning(
            "Adding labels when some y-values are outside range of [0-1] " "may produce unexpected results."
        )

    # searchsorted assumes value is above the minimum in list
    thresholds = sorted(thresholds, reverse=True)
    val_ix = _find_thresholds(labels, thresholds)

    colors = []
    if len(thresholds) == 2:
        colors = alert_colors[:2]
        legend_str = None
    else:
        colors = [area_colors[colorIx]] * len(thresholds)
        legend_str = "Threshold"
    for i, ix in enumerate(val_ix):
        axis.plot(x[ix], y[ix], "o", color=colors[i], label=legend_str)

    thresholds = as_percentages(np.array(thresholds))
    for i, ix in enumerate(val_ix):
        axis.annotate(
            f"{thresholds[i]:.0f}",
            _radial_annotations(x[ix], y[ix], Q=Q),
            color=colors[i],
        )


def _find_thresholds(labels: list[float], thresholds: list[float]) -> list[int]:
    """Finds the threshold values in the labels list."""
    labels = as_probabilities(np.array(labels))
    if labels[0] > labels[-1]:
        label_compare = np.insert(labels[::-1], 0, -np.inf)
        val_ix = [max(x, 0) for x in len(label_compare) - np.searchsorted(label_compare, thresholds, side="right") - 1]
    else:
        label_compare = np.insert(labels, 0, -np.inf)
        val_ix = [max(x, 0) for x in np.searchsorted(label_compare, thresholds, side="right") - 2]

    return val_ix


def _add_radial_score_labels(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    colorIx: int = 0,
    n_scores: int = 10,
    Q: int = 1,
    highlight: Optional[Number] = None,
) -> None:
    """
    Add numeric annotations on points as if it were a quareter-circle.

    Expected use is for ROC curves with either scores or n_scores roughly spaced along the curve.

    If the three arrays (x, y, and labels) have more values then n_scores, labels will only be added
    for a reduced list of len(n_scores) == 10 by default.

    Parameters
    ----------
    axis : plt.Axes
        Axis to modify.
    x : np.ndarray
        Array of x-values.
    y : np.ndarray
        Array of y-values.
    labels : list[str]
        Array of labels corresponding to the x, y value order.
    colorIx : int, optional
        Index of line being annotated in order to match color, by default 0.
    n_scores : int, optional
        Integer number of scores to display, if length of arrays is greater, will automatically
        select 10 values to shows based on an assumed range of labels being 0-1 (default: 10).
    Q : int, optional
        The quadrant 1-4 to assume for offsets, by default 1.
    highlight : Optional[Number], optional
        List of thresholds to plot instead of labels, by default None.

    """
    if highlight is not None:
        return _add_radial_score_thresholds(axis, x, y, labels, thresholds=highlight, Q=Q)

    # For safety convert to handle indexing lists
    x = np.array(x)
    y = np.array(y)
    labels = np.array(labels)

    if x.max() > 1 or x.min() < 0:
        logging.warning("Adding labels when some x-values are outside range of [0-1] may produce unexpected results.")
    if y.max() > 1 or y.min() < 0:
        logging.warning("Adding labels when some y-values are outside range of [0-1] may produce unexpected results.")

    if len(labels) > n_scores:
        thresholds = np.linspace(0, 2, n_scores + 1)[1:]
        labelIx = np.searchsorted(x + y, thresholds)

        labels = as_percentages(labels)
    else:
        labelIx = range(len(labels))

    axis.plot(x[labelIx], y[labelIx], "o", color=area_colors[colorIx], label="Threshold")
    axis.legend(loc="lower right")

    for ix in labelIx:
        axis.annotate(
            f"{labels[ix]:.1f}",
            _radial_annotations(x[ix], y[ix], Q),
            color=text_colors[colorIx],
        )


def _radial_annotations(x, y, Q=1) -> tuple[float, float]:
    """
    Takes a single point and offsets it, returning the modified values.

    Used for summary statistic plots that are approximated to a quarter-circle.
    Without any offset, annotate would print directly on top of the line.

    Parameters
    ----------
    x : float
        The initial x-coordinate.
    y : float
        The initial y-coordinate.
    Q : int, optional
        The quadrant of the circle to offset the annotation (default: 1).

    Returns
    -------
    Tuple[float, float]
        Modified x, y values.
    """
    sgn = [1, 1, -1, -1, 1, 1]
    return (x + (sgn[Q] * 0.05 * np.cos(1.57 * x)), y + (sgn[Q + 1] * 0.05 * np.sin(1.57 * x)))


# endregion
