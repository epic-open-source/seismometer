import pandas as pd

from .decorators import export


@export
def default_cohort_summaries(
    dataframe: pd.DataFrame, attribute: str, options: list[str], entity_id_col: str
) -> pd.DataFrame:
    """
    Generate a dataframe of summary counts from the input dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.
    attribute : str
        The attribute to generate summary levels for.
    options : list[str]
        An ordered list of options to reindex the dataframe on.
    entity_id_col : str
        The column name for the dataframe column containing the entity identifier.

    Returns
    -------
    pd.DataFrame
        A dataframe of summary counts.
    """
    left = dataframe[attribute].value_counts().rename("Predictions")
    right = dataframe.drop_duplicates(subset=entity_id_col)[attribute].value_counts().rename("Entities")

    return pd.concat([left, right], axis=1).reindex(options)


@export
def score_target_cohort_summaries(
    dataframe: pd.DataFrame, groupby_groups: list[str], grab_groups: list[str], entity_id_col: str
) -> pd.DataFrame:
    """
    Generate a dataframe of summary counts from the input dataframe.
    Also, summarizes by additional summary levels in groupby_groups.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.
    groupby_groups : list[str]
        Selections to groupby when generating summaries (attribute, score bins, target, etc.).
    grab_groups : list[str]
        Columns to grab while summarizing.
    entity_id_col : str
        The column name for the dataframe column containing the entity identifier.

    Returns
    -------
    pd.DataFrame
        A dataframe of summary counts.
    """
    predictions = dataframe[grab_groups].groupby(groupby_groups, observed=False).size().rename("Predictions")
    entities = (
        dataframe.drop_duplicates(subset=entity_id_col)[grab_groups]
        .groupby(groupby_groups, observed=False)
        .size()
        .rename("Entities")
        .astype("Int64")
    )

    return pd.DataFrame(pd.concat([predictions, entities], axis=1)).fillna(0)
