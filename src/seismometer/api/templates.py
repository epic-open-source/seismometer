import logging

import pandas as pd
from IPython.display import HTML
from pandas.io.formats.style import Styler

from seismometer.controls.decorators import disk_cached_html_segment
from seismometer.core.decorators import export
from seismometer.data import default_cohort_summaries, score_target_cohort_summaries
from seismometer.html import template
from seismometer.seismogram import Seismogram

logger = logging.getLogger("seismometer")


def _get_info_dict(plot_help: bool) -> dict[str, str | list[str]]:
    """
    Gets the required data dictionary for the info template.

    Parameters
    ----------
    plot_help : bool
        If True, displays additional information about available plotting utilities, by default False.

    Returns
    -------
    dict[str, str | list[str]]
        The data dictionary.
    """
    sg = Seismogram()

    info_vals = {
        "tables": [
            {
                "name": "predictions",
                "description": "Scores, features, configured demographics, and merged events for each prediction",
                "num_rows": sg.prediction_count,
                "num_cols": sg.feature_count,
            }
        ],
        "num_predictions": sg.prediction_count,
        "num_entities": sg.entity_count,
        "start_date": sg.start_time.strftime("%Y-%m-%d"),
        "end_date": sg.end_time.strftime("%Y-%m-%d"),
        "plot_help": plot_help,
    }

    return info_vals


@disk_cached_html_segment
@export
def show_info(plot_help: bool = False) -> HTML:
    """
    Displays information about the dataset

    Parameters
    ----------
    plot_help : bool, optional
        If True, displays additional information about available plotting utilities, by default False.
    """

    info_vals = _get_info_dict(plot_help)

    return template.render_info_template(info_vals)


def _style_cohort_summaries(df: pd.DataFrame, attribute: str) -> Styler:
    """
    Adds required styling to a cohort dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The output of default_cohort_summaries().
    attribute : str
        The display name of the cohort.

    Returns
    -------
    Styler
        Styled dataframe.
    """
    df.index = df.index.rename("Cohort")
    style = df.style.format(precision=2)
    style = style.format_index(precision=2)
    return style.set_caption(f"Counts by {attribute}")


def _score_target_levels_and_index(
    selected_attribute: str, by_target: bool, by_score: bool
) -> tuple[list[str], list[str], list[str]]:
    """
    Gets the summary levels for the cohort summary tables.

    Parameters
    ----------
    selected_attribute : str
        The name of the current attribute to generate summary levels for.
    by_target : bool
        If True, adds an additional summary table to break down the population by target prevalence.
    by_score : bool
        If True, adds an additional summary table to break down the population by model output.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        groupby_groups: The levels in the dataframe to groupby when summarizing
        grab_groups: The columns in the dataframe to grab to summarize
        index_rename: The display names for the indices
    """
    sg = Seismogram()

    score_bins = sg.score_bins()
    cut_bins = pd.cut(sg.dataframe[sg.output], score_bins)

    groupby_groups = [selected_attribute]
    grab_groups = [selected_attribute]
    index_rename = ["Cohort"]

    if by_score:
        groupby_groups.append(cut_bins)
        grab_groups.append(sg.output)
        index_rename.append(sg.output)

    if by_target:
        groupby_groups.append(sg.target)
        grab_groups.append(sg.target)
        index_rename.append(sg.target.removesuffix("_Value"))

    return groupby_groups, grab_groups, index_rename


def _style_score_target_cohort_summaries(df: pd.DataFrame, index_rename: list[str], cohort: str) -> Styler:
    """
    Adds required styling to a multiple summary level cohort summary dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The output of score_target_cohort_summaries.
    index_rename : list[str]
        The display names to put in the index from _score_target_levels_and_index().
    cohort : str
        The display name of the cohort.

    Returns
    -------
    Styler
        Styled dataframe.
    """
    df.index = df.index.rename(index_rename)
    style = df.style.format(precision=2)
    style = style.format_index(precision=2)
    return style.set_caption(f"Counts by {cohort}")


def _get_cohort_summary_dataframes(by_target: bool, by_score: bool) -> dict[str, list[str]]:
    """
    Gets the formatted summary cohort dataframes to display in the cohort summary template.

    Parameters
    ----------
    by_target : bool
        If True, adds an additional summary table to break down the population by target prevalence.
    by_score : bool
        If True, adds an additional summary table to break down the population by model output.

    Returns
    -------
    dict[str, list[str]]
        The dictionary, indexed by cohort attribute (e.g. Race), of summary dataframes.
    """
    sg = Seismogram()

    dfs: dict[str, list[str]] = {}

    available_cohort_groups = sg.available_cohort_groups

    for attribute, options in available_cohort_groups.items():
        df = default_cohort_summaries(sg.dataframe, attribute, options, sg.config.entity_id)
        styled = _style_cohort_summaries(df, attribute)

        dfs[attribute] = [styled.to_html()]

        if by_score or by_target:
            groupby_groups, grab_groups, index_rename = _score_target_levels_and_index(attribute, by_target, by_score)

            results = score_target_cohort_summaries(sg.dataframe, groupby_groups, grab_groups, sg.config.entity_id)
            results_styled = _style_score_target_cohort_summaries(results, index_rename, attribute)

            dfs[attribute].append(results_styled.to_html())

    return dfs


@disk_cached_html_segment
@export
def show_cohort_summaries(by_target: bool = False, by_score: bool = False) -> HTML:
    """
    Displays a table of selectable attributes and their associated counts.
    Use `by_target` and `by_score` to add additional summary levels to the tables.

    Parameters
    ----------
    by_target : bool, optional
        If True, adds an additional summary table to break down the population by target prevalence, by default False.
    by_score : bool, optional
        If True, adds an additional summary table to break down the population by model output, by default False.
    """
    dfs = _get_cohort_summary_dataframes(by_target, by_score)

    return template.render_cohort_summary_template(dfs)
