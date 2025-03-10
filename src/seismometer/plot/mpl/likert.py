from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, MaxNLocator

from ._ux import area_colors


def likert_plot(
    df: pd.DataFrame,
    colors: Iterable[str] = area_colors,
    border: int = 5,
):
    """
    Creates a Likert plot (horizontal stacked bar) from the given DataFrame.

    Expects dataframe to be of the form:
        1. column names are categorical values (e.g., Disagree, Neutral, Agree),
        2. row indexes are corresponding metric columns (e.g., feedback questions 1,2,3),
        3. values in each row are counts of each categorical value in the row.

    Example
    -------
    >>> import pandas as pd
    >>> data = {
    >>>     "Disagree": [10, 5],
    >>>     "Neutral": [15, 10],
    >>>     "Agree": [35, 45]
    >>> }
    >>> df = pd.DataFrame(data, index=["Likes Cat", "Likes Dog"])
    >>> print(df)
               Disagree  Neutral  Agree
    Likes Cat        10       15     35
    Likes Dog         5       10     45

    Parameters
    ----------
    df : pd.DataFrame
       DataFrame containing the counts of each category.
    colors : Iterable[str], optional
        Iterable of colors for the bars, by default area_colors.
    border : int, optional
        Border space around the plot, by default 5.

    Returns
    -------
    matplotlib.figure.Figure
        The generated Likert plot figure.
    """
    row_sums = df.sum(axis=1)
    matplotlib.use("Agg")
    if (row_sums != row_sums.iloc[0]).any():
        fig, (ax, ax_count) = plt.subplots(ncols=2, figsize=(15, 6), gridspec_kw={"width_ratios": [2, 1]})
        _plot_counts(df, ax_count)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
    plt.close(fig)

    df_percentages = df.div(row_sums, axis=0) * 100

    cumulative_data = df_percentages.cumsum(axis=1)
    first_half_columns = df.columns[: (len(df.columns) + 1) // 2]
    second_half_columns = df.columns[len(df.columns) // 2 :]  # noqa: E203
    middle_adjustment = df_percentages[df.columns[len(df.columns) // 2]] / 2 if len(df.columns) % 2 == 1 else 0
    for i, col in enumerate(df.columns):
        bars = ax.barh(
            df.index,
            df_percentages[col],
            left=cumulative_data[col]
            - df_percentages[col]
            - cumulative_data[df.columns[(len(df.columns) - 1) // 2]]
            + middle_adjustment,
            color=colors[i],
            label=col,
        )
        for index, bar in enumerate(bars):
            width = bar.get_width()
            count = df[col].iloc[index]
            percentage = df_percentages[col].iloc[index]
            if width >= 10:  # Adjust this threshold as needed
                ax.text(
                    bar.get_x() + width / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{percentage:.0f}%\n({count})",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
    max_left = abs(df_percentages[first_half_columns].sum(axis=1) - middle_adjustment).max()
    max_right = abs(df_percentages[second_half_columns].sum(axis=1) - middle_adjustment).max()
    ax.set_xlim(-max_left - border, max_right + border)
    # Add labels and title
    ax.set_xlabel("Percentages of Responses")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
    ax.set_xticklabels([f"{int(x)}%" for x in ax.get_xticks()])
    # ax.set_ylabel('Questions')
    ax.set_title("Likert Plot")
    return fig


def _plot_counts(df: pd.DataFrame, ax_count: matplotlib.axes.Axes, color: str = area_colors[0], border: int = 5):
    """
    Plots the counts of each row in the dataframe as horizontal bars.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the counts of each category.
    ax_count : matplotlib.axes.Axes
        The axes on which to plot the counts.
    color : str, optional
        Color for the bars, by default area_colors[0].
    border : int, optional
        Border space around the plot, by default 5.
    """
    total_counts = df.sum(axis=1)
    bars = ax_count.barh(df.index, total_counts, color=color)
    max_count = max(total_counts)
    for bar in bars:
        width = bar.get_width()
        if width >= 0.1 * max_count:
            ax_count.text(
                bar.get_x() + width / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{width}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
        else:
            ax_count.text(
                bar.get_x() + width + 0.01 * max_count,
                bar.get_y() + bar.get_height() / 2,
                f"{width}",
                ha="left",
                va="center",
                fontsize=8,
                color="black",
            )

    ax_count.set_xlabel("Counts")
    ax_count.set_title("Counts of Each Row")
    ax_count.set_xlim(0, total_counts.max() + border)
    ax_count.xaxis.set_major_locator(MaxNLocator(nbins=5))
