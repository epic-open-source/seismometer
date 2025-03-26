import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from seismometer.plot.mpl.decorators import render_as_svg

from ._ux import area_colors


@render_as_svg
def likert_plot(
    df: pd.DataFrame,
    border: int = 5,
    title: str = "Likert Plot",
) -> plt.Figure:
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
    border : int, optional
        Border space around the plot, by default 5.
    title : str, optional
        The title of the plot, by default "Likert Plot".

    Returns
    -------
    matplotlib.figure.Figure
        The generated Likert plot figure.
    """
    row_sums = df.sum(axis=1)
    matplotlib.use("Agg")
    if len(df) == 0:
        raise ValueError("df is empty.")
    fig_height = len(df) * 1.5 + 1
    if (row_sums != row_sums.iloc[0]).any():
        fig, (ax, ax_count) = plt.subplots(
            ncols=2, figsize=(21, fig_height), gridspec_kw={"width_ratios": [2, 1], "wspace": 0.5}
        )
        _plot_counts(df, ax_count)
    else:
        fig, ax = plt.subplots(figsize=(14, fig_height))

    df_percentages = df.div(row_sums, axis=0) * 100
    df_percentages.fillna(0, inplace=True)

    cumulative_data = df_percentages.cumsum(axis=1)
    first_half_columns = df.columns[: (len(df.columns) + 1) // 2]
    second_half_columns = df.columns[len(df.columns) // 2 :]  # noqa: E203
    middle_adjustment = df_percentages[df.columns[len(df.columns) // 2]] / 2 if len(df.columns) % 2 == 1 else 0
    for i, col in enumerate(df.columns):
        bars = ax.barh(
            df.index,
            df_percentages[col],
            height=0.7,
            left=cumulative_data[col]
            - df_percentages[col]
            - cumulative_data[df.columns[(len(df.columns) - 1) // 2]]
            + middle_adjustment,
            color=area_colors[i % len(area_colors)],
            label=col,
        )
        for index, bar in enumerate(bars):
            width = bar.get_width()
            count = df[col].iloc[index]
            percentage = df_percentages[col].iloc[index]
            if width >= 9.5:  # Adjust this threshold as needed
                ax.text(
                    bar.get_x() + width / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{percentage:.0f}%\n({_format_count(count)})",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
            elif width >= 5:
                ax.text(
                    bar.get_x() + width / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{percentage:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                )

    ax.set_yticklabels(_wrap_labels(df.index), fontsize=12)
    ax.set_ylim(-0.5, len(df.index) - 0.5)
    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
    max_left = abs(df_percentages[first_half_columns].sum(axis=1) - middle_adjustment).max()
    max_right = abs(df_percentages[second_half_columns].sum(axis=1) - middle_adjustment).max()
    ax.set_xlim(-max_left - border, max_right + border)
    # Add labels and title
    ax.set_xlabel("Percentages of Responses", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.set_xticklabels([f"{int(x)}%" for x in ax.get_xticks()], fontsize=12)
    ax.set_title(title)
    return fig


def _plot_counts(df: pd.DataFrame, ax_count: matplotlib.axes.Axes, border: int = 5):
    """
    Plots the counts of each row in the dataframe as horizontal bars.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the counts of each category.
    ax_count : matplotlib.axes.Axes
        The axes on which to plot the counts.
    border : int, optional
        Border space around the plot, by default 5.
    """
    total_counts = df.sum(axis=1)
    bars = ax_count.barh(df.index, total_counts, height=0.7, color=area_colors[0])
    max_count = max(total_counts)
    for bar in bars:
        width = bar.get_width()
        if width >= 0.1 * max_count:
            ax_count.text(
                bar.get_x() + width / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{_format_count(width)}",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )
        else:
            ax_count.text(
                bar.get_x() + width + 0.01 * max_count,
                bar.get_y() + bar.get_height() / 2,
                f"{_format_count(width)}",
                ha="left",
                va="center",
                fontsize=10,
                color="black",
            )

    ax_count.set_yticklabels(_wrap_labels(df.index), fontsize=12)
    ax_count.set_xlabel("Counts", fontsize=12)
    ax_count.set_title("Counts of Each Row")
    ax_count.set_xlim(0, total_counts.max() + border)
    ax_count.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_count.set_xticklabels([int(x) for x in ax_count.get_xticks()], fontsize=12)


def _wrap_labels(labels, width=12):
    wrapped_labels = []
    for label in labels:
        words = label.split()
        new_label = ""
        line = ""
        for word in words:
            if len(line) + len(word) + 1 <= width:
                line += word + " "
            else:
                new_label += line.strip() + "\n"
                line = word + " "
        new_label += line.strip()
        wrapped_labels.append(new_label)
    return wrapped_labels


def _format_count(value):
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3g}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    elif value >= 1_000:
        return f"{value / 1_000:.3g}K"
    else:
        return f"{value:.0f}"
