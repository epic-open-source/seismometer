from typing import Optional

import pandas as pd

from .pandas_helpers import is_valid_event


def create_metric_timeseries(
    dataframe: pd.DataFrame,
    reftime: str,
    event_col: str,
    entity_keys: list[str],
    cohort_col: str,
    *,
    time_bounds: Optional[tuple] = None,
    boolean_event: bool = False,
    censor_threshold: int = 10,
) -> pd.DataFrame:
    """
    Summarize a dataframe into a frame for plotting a timeseries.

    Manipulates a dataframe with reference to

        - entity_keys - will reduce to the earliest value for each unique set of keys,
        - reftime - the time to use as the reference for the timeseries,
        - event_col - the value to use for the timeseries,
        - cohort_col - an additional metadata column for stratifying.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input data.
    reftime : str
        The column name of the reference time.
    event_col : str
        The columns name of the value.
    entity_keys : list[str]
        A list of column names to use for summarizing the data.
    cohort_col : str
        A column containing the cohort information.
    time_bounds : Optional[tuple], optional
        An optional tuple (min, max) of inclusive times to bound the data to, by default None.
        If present the data is reduced to times within bounds prior to summarization.
        While the selection is inclusive, midnight is used if no time is passed making a maximum
        date effectively exclusive on times.
    boolean_event : bool, optional
        If True, indicates the event is boolean so negative values are filtered out, by default False.
    censor_threshold : int, optional
        The minimum number of values for a given time that are needed to not be filtered, by default 10.

    Returns
    -------
    pd.DataFrame
        The filtered and summarized data.
    """
    reduced = _limit_data(dataframe, event_col, reftime, boolean_event=boolean_event, time_bounds=time_bounds)

    line_data = _orient_frequency_per_entity(reduced, entity_keys, reftime)

    return _censor_small_groups(
        line_data, event_col, group_columns=[reftime, cohort_col], censor_threshold=censor_threshold
    )


def _orient_frequency_per_entity(dataframe: pd.DataFrame, entity_keys: list[str], reftime: str) -> pd.DataFrame:
    """Reduces and aligns the data to the earliest instance of a given week."""
    line_data = dataframe[~dataframe[entity_keys].duplicated()].copy()
    # Note: W-SUN specifies the LAST day of the week
    line_data[reftime] = line_data[reftime].dt.to_period("W-SUN").dt.start_time
    return line_data.reset_index(drop=True)


def _limit_data(
    dataframe: pd.DataFrame,
    event_col: str,
    reftime: str,
    boolean_event: bool = False,
    time_bounds: Optional[tuple] = None,
) -> pd.DataFrame:
    """Reduces the data to only include valid data points."""
    include = dataframe[reftime].notna()

    if boolean_event:
        include = include & (is_valid_event(dataframe, event_col, reftime))
    reduced = dataframe[include]

    if time_bounds is None:
        return reduced

    return reduced[
        (reduced[reftime] >= pd.to_datetime(time_bounds[0])) & (reduced[reftime] <= pd.to_datetime(time_bounds[1]))
    ]


def _censor_small_groups(
    dataframe: pd.DataFrame, event_col: str, group_columns: list[str], censor_threshold: int
) -> pd.DataFrame:
    """Reduces the data to only the data where each group has sufficient size per timestamp."""
    counts = dataframe.groupby(group_columns, observed=True).count()
    msk_min = counts[event_col] <= censor_threshold
    invmask = counts[msk_min].reset_index()[group_columns]
    invmask["DROPPING"] = 1

    return_data = dataframe.merge(invmask, on=group_columns, how="outer")
    return_data = return_data.loc[return_data["DROPPING"] != 1]
    return return_data[group_columns + [event_col]]
