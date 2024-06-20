# -*- coding: utf-8 -*-
"""
This module supports plots in other seismometer.plot.mpl modules. It supports saving figures,
drawing plot backgrounds (diagonals, polygons), and simple accented curves.

Examples::
    utils.save_figure("my_figure.png")
    utils.plot_curve(axis, line, accent_dict={"accent1": points, "accent2": points})

"""
import logging
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import SVG
from ipywidgets import Checkbox
from matplotlib.patches import Rectangle


def to_svg() -> SVG:
    """matplot lib, write current plot to svg string"""
    buffer = StringIO()
    plt.savefig(buffer, format="svg")
    return SVG(buffer.getvalue())


def save_figure(filename, **kwargs):
    """Wrapper for saving matplotlib images."""
    bbox_inches = kwargs.pop("bbox_inches", "tight")
    plt.savefig(filename, bbox_inches=bbox_inches, **kwargs)


def create_checkboxes(value_list: list) -> list[Checkbox]:
    """Creates a list of checkbox widgets for a list of values."""
    checkboxes = []
    # Create a checkbox for each value in the list
    for value in value_list:
        checkbox = Checkbox(description=str(value), value=True)
        checkboxes.append(checkbox)

    return checkboxes


def add_unseen(df: pd.DataFrame, col="cohort") -> pd.DataFrame:
    """Updates a dataframe so that all category values have at least one row."""
    keys = df[col].cat.categories
    obs = df[col].unique()
    unseen = [k for k in keys if k not in obs]

    rv = pd.concat([df, pd.DataFrame({col: unseen})], ignore_index=True)
    rv[col] = rv[col].astype(pd.CategoricalDtype(df[col].cat.categories))
    return rv


def cohort_legend(
    data: pd.DataFrame,
    ax: plt.Axes,
    feature: str,
    labellist: list[str] = None,
    ref_axis: int = -2,
    censor_threshold: int = 10,
) -> None:
    """
    Function to create a table like legend for a cohort performance plot.
    Uses the lines from a neighboring axis on the same figure as reference, which MUST be plotted before calling
    this function.

    Table includes the cohort's label, and calculates the prevalence and cohort size based on the data passed in.

    Parameters
    ----------
    data : pd.DataFrame
        Data in format of get_cohort_data[0] OR get_cohort_performance_data.
    ax : plt.Axes
        Matplotlib axis on which to draw.
    feature : str
        Name of feature that is being split; used for display only.
    labels : list[str]
        List of descriptions for each of the cohorts being displayed. Expects a label for each line plotted.
    ref_axis : int,
        The index of the matplotlib axis to use as a reference for legend lines, by default -2.
        By default looks to the previous axis, with the assumption that the legend is being drawn on a NEW blank axis.
    """
    # skip dashed lines - e.g., highlighted thresholds
    linelist = [line for line in plt.gcf().axes[ref_axis].get_lines() if not line.is_dashed()]
    ax.axis("off")

    if labellist is None:
        labellist = data["cohort"].cat.categories
    if len(labellist) < len(linelist):  # More cohorts than plotted lines
        raise IndexError("More lines than cohorts. Possibly a bad reference axis")
    if len(labellist) > len(linelist):  # More cohorts than plotted lines
        logging.warning("More cohorts than lines, using first labels found")

    blank = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)

    legend_labels = [[f"{feature} Cohorts", "Count", "Prevalence"]]
    legend_handle = [[blank] * 3]
    for line, label in zip(linelist, labellist):
        cohort_df = data[data["cohort"] == label]

        if len(cohort_df.index) <= censor_threshold:
            count_str = ""
            prevalence_str = ""
        else:
            count = 0
            TP = 0

            if "true" in cohort_df.columns:
                count = len(cohort_df.index)
                TP = len(cohort_df[cohort_df["true"] == 1].index)
            elif all([item in cohort_df.columns for item in ["cohort-count", "cohort-targetcount"]]):
                count = cohort_df.loc[cohort_df.index[0], "cohort-count"]
                TP = cohort_df.loc[cohort_df.index[0], "cohort-targetcount"]

            prevalence = TP / count if count > 0 else 0

            count_str = f"{count:>}"
            prevalence_str = f"{prevalence:.2f}"

        row_labels = [label, count_str, prevalence_str]
        row_handles = [line, blank, blank]

        legend_labels.append(row_labels)
        legend_handle.append(row_handles)

    ax.legend(np.ravel(legend_handle, "F"), np.ravel(legend_labels, "F"), loc="center", ncol=3)


# region Reference Lines
REFERENCE_GREY = "#e6e7e8"


def plot_polygon(axis, x, y):
    axis.fill(x, y, c="C0", alpha=0.10)


def plot_diagonal(axis):
    axis.plot([0, 1], [0, 1], "--", c=REFERENCE_GREY)


def plot_horizontal(axis, y):
    axis.plot([0, 1], [y, y], "r--", c=REFERENCE_GREY)


def plot_vertical(axis, x):
    axis.plot([x, x], [0, 1], "r--", c=REFERENCE_GREY)


def needed_colors(series: pd.Series, color_list: list[str]) -> list[str]:
    """Identifies which colors are needed to plot while maintaining colors to category order."""
    try:  # Categorical
        observed_codes = sorted(series.cat.codes.unique())
    except BaseException:
        observed_codes = range(series.nunique())
    return [color_list[i] for i in observed_codes]


# endregion

# region Curve with accent marks


def plot_curve(axis, line, accent_dict=None, title=None, curve_name=None, y_label=None) -> None:
    """
    Actual matplotlib plot creation for transform data.

    Parameters
    ----------
    axis
        Current axis from matplotlib.
    line
        [xList, yList] - ordered list of x and y's plotted as connected lines.
    accent_dict
        {'AccentName':[xList, yList],...}. Labeled sets of x's and y's such as a spline's knots; order between x-y
        pairs does not matter as these are plotted with 'x' symbols.
        If not none, the plot will show a legend containing the accent names.
    title
        Title of the graph.
    label
        Label for the primary curve.

    Returns
    -------
        None - will add to the passed in graphing axis.
    """
    axis.plot(line[0], line[1], label=curve_name)

    if accent_dict is not None:
        for key in accent_dict.keys():
            axis.plot(accent_dict[key][0], accent_dict[key][1], "x", label=key)
        axis.legend()
    if title is not None:
        axis.set_xlabel(title)
    if y_label is not None:
        axis.set_ylabel(y_label)


# endregion

# region Axis cleaning


def axis_clear(axis: plt.Axes, clear_x: bool = True, clear_y: bool = False) -> None:
    """
    Utility to clear out the tick and labels for an axis.

    Parameters
    ----------
    axis : plt.Axes
        A matplotlib.pyplot Axis to modify.
    clear_x : bool, optional
        Boolean flag on whether to alter the x-axis, by default True.
    clear_y : bool, optional
        Boolean flag on whether to alter the x-axis, by default False.
    """
    if clear_x:
        axis.set_xticklabels([])
        axis.set_xlabel(None)
    if clear_y:
        axis.set_yticklabels([])
        axis.set_ylabel(None)


# endregion
