import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .decorators import model_plot

logger = logging.getLogger("seismometer")


@model_plot
def compare_series(
    plotdata: pd.DataFrame,
    cohort_col: str,
    ref_str: str,
    event_col: str,
    ylabel: Optional[str] = None,
    counts: Optional[pd.DataFrame] = None,
    show_legend: bool = True,
) -> plt.Figure:
    """
    Creates a line plot of the data using cohorts as hue.

    Parameters
    ----------
    plotdata : pd.DataFrame
        The input data.
    cohort_col : str
        The column name of the cohort, used as hue.
    ref_str : str
        The column name of the reference time, used as the x-axis.
    event_col : Optional[str], optional
        The column name of the value, used as y-axis, by default None.
    ylabel : Optional[str], optional
        The label for the y-axis, by default None; uses the event_col.
    counts : Optional[pd.DataFrame], optional
        Optional data to plot in a second axis, by default None.
    show_legend : bool, optional
        A flag when set will show the legend on the plot, by default True.
    """
    n_vert = 1 if counts is None else 2
    fig, axes = plt.subplots(n_vert, 1, figsize=(9, 2 * n_vert))

    disp_event = ylabel or event_col
    plotdata = plotdata.rename(columns={event_col: disp_event})

    try:
        ax0 = axes[0]
    except TypeError:
        ax0 = axes

    with np.errstate(divide="ignore"):
        sns.lineplot(x=ref_str, y=disp_event, hue=cohort_col, data=plotdata, ax=ax0, legend=show_legend)
        if show_legend:
            ax0.legend(loc="upper right", title=cohort_col)
        if n_vert == 2:
            axes[0].set_xlabel("")
            counts["log(count)"] = np.log10(counts[event_col])
            sns.lineplot(x=ref_str, y="log(count)", hue=cohort_col, data=counts.reset_index(), ax=axes[1])

    return fig
