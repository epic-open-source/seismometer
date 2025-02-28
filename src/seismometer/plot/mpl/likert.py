import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, MaxNLocator

PLOT_COLORS = ["red", "orange", "yellow", "lightgreen", "green"]


def likert_plot(df: pd.DataFrame, colors: list[str] = PLOT_COLORS, border: int = 5, include_counts_plot: bool = False):
    """
    Expects dataframe to be of the form:
        1. columns are categorical values (e.g., Disagree, Neutral, Agree),
        2. rows are corresponding columns (e.g., feedback questions 1,2,3),
        3. values are counts of each categorical value in the corresponding column.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    colors : list[str]
        _description_
    border : int, optional
        _description_, by default 5
    """
    if include_counts_plot:
        fig, (ax, ax_count) = plt.subplots(ncols=2, figsize=(15, 6), gridspec_kw={"width_ratios": [2, 1]})
        _plot_counts(df, ax_count)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
    plt.close(fig)

    df_percentages = df.div(df.sum(axis=1), axis=0) * 100

    cumulative_data = df_percentages.cumsum(axis=1)
    first_half_columns = df.columns[: (len(df.columns) + 1) // 2]
    second_half_columns = df.columns[len(df.columns) // 2 :]  # noqa: E203
    middle_adjustment = df[df.columns[len(df.columns) // 2]] / 2 if len(df.columns) % 2 == 1 else 0
    for i, col in enumerate(df.columns):
        bars = ax.barh(
            df.index,
            df_percentages[col],
            left=cumulative_data[col]
            - df_percentages[col]
            - cumulative_data[(len(df.columns) - 1) // 2]
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
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
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


def _plot_counts(df: pd.DataFrame, ax: matplotlib.axes.Axes, color: str = "lightblue", border: int = 5):
    """
    Plots the counts of each row in the dataframe as horizontal bars.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the counts of each category.
    ax : matplotlib.axes.Axes
        The axes on which to plot the counts.
    color : str
        Color for the bars.
    border : int, optional
        Border space around the plot, by default 5
    """
    total_counts = df.sum(axis=1)
    bars = ax.barh(df.index, total_counts, color=color)

    for bar in bars:
        width = bar.get_width()
        width = bar.get_width()
        if width >= 10:
            ax.text(
                bar.get_x() + width / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{width}",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )

    ax.set_xlabel("Counts")
    ax.set_title("Counts of Each Row")
    ax.set_xlim(0, total_counts.max() + border)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
