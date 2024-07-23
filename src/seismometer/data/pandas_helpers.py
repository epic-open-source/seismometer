import logging
import warnings
from numbers import Number
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("seismometer")


def merge_windowed_event(
    predictions: pd.DataFrame,
    predtime_col: str,
    events: pd.DataFrame,
    event_label: str,
    pks: list[str],
    min_leadtime_hrs: Number = 0,
    window_hrs: Optional[Number] = None,
    event_base_val_col: str = "Value",
    event_base_time_col: str = "Time",
    sort: bool = True,
) -> pd.DataFrame:
    """
    Merges a single windowed event into a predictions dataframe

    Adds two new event columns: a _Value column with the event value and a _Time column with the event time.
    Ground-truth labeling for a model is considered an event and can have a time associated with it.

    Joins on a set of keys and associates the first event occurring after the prediction time.  The following special
    cases are also applied:

    Invalidate late predictions - if a prediction occurs after all recorded events of the type, the prediction is
    considered invalid wrt to the event and the _Value is set to -1.
    Early predictions drop timing - if a prediction occurs before all recorded events of the type, the label is kept
    for analyses but the time is removed.
    Imputation of no event to negative label - if no row in the events frame is present for the prediction keys, it is
    assumed to be a Negative label (default 0) but will not have an event time.


    Parameters
    ----------
    predictions : pd.DataFrame
        The predictions or features frame where each row represents a prediction.
    predtime_col : str
        The column in the predictions frame indicating the timestamp when inference occurred.
    events : pd.DataFrame
        The narrow events dataframe
    event_label : str
        The category name of the event to merge, expected to be a value in events.Type.
    pks : list[str]
        A list of primary keys on which to perform the merge, keys are column names occurring in both predictions and
        events dataframes.
    min_leadtime_hrs : Number, optional
        The number of hour offset to be required for prediction, by default 0.
        If set to 1, a prediction made within the hour before the last associated event will be invalidated and set
        to -1 even though it occurred before the event time.
    window_hrs : Optional[Number], optional
        The number of hours the window of predictions of interest should be limited to, by default None.
        If None, then all predictions occurring before a known event will be included.
        If used with min_leadtime_hrs, the entire window is shifted maintaining its size. The maximum lookback for a
        prediction is window_hrs + min_leadtime_hrs.
    event_base_val_col : str, optional
        The name of the column in the events frame to merge as the _Value, by default 'Value'.
    event_base_time_col : str, optional
        The name of the column in the events frame to merge as the _Time, by default 'Time'.
    sort : bool
        Whether or not to sort the predictions/events dataframes, by default True.

    Returns
    -------
    pd.DataFrame
        The predictions dataframe with the new time and value columns for the event specified.

    Raises
    ------
    ValueError
        At least one column in pks must be in both the predictions and events dataframes.
    """

    # Validate and resolve
    min_offset = pd.Timedelta(min_leadtime_hrs, unit="hr")

    r_ref = "~~reftime~~"
    pks = [col for col in pks if col in events and col in predictions]  # Ensure existence in both frames
    if len(pks) == 0:
        raise ValueError("No common keys found between predictions and events.")

    # Preprocess events : reduce and rename
    one_event = _one_event(events, event_label, event_base_val_col, event_base_time_col)
    if len(one_event.index) == 0:
        return predictions
    event_time_col = event_time(event_label)
    event_val_col = event_value(event_label)

    # one_event = infer_label(one_event, event_val_col, event_time_col)
    one_event[r_ref] = one_event[event_time_col] - min_offset

    # merge next event for each prediction
    predictions = _merge_next(
        predictions, one_event, pks, l_ref=predtime_col, r_ref=r_ref, merge_cols_without_times=event_val_col, sort=sort
    )
    predictions = infer_label(predictions, event_val_col, event_time_col)

    if window_hrs is not None:
        max_lookback = pd.Timedelta(window_hrs, unit="hr") + min_offset  # keep window the specified size
        predictions.loc[predictions[predtime_col] < (predictions[r_ref] - max_lookback), event_time_col] = pd.NaT

    # refactor to generalize
    predictions.loc[predictions[predtime_col] > predictions[r_ref], event_val_col] = -1

    return predictions.drop(columns=r_ref)

    # assume one event per


def _one_event(
    events: pd.DataFrame, event_label: str, event_base_val_col: str, event_base_time_col: str
) -> pd.DataFrame:
    """Reduces the events dataframe to those rows associated with the event_label, preemptively renaming to the
    columns to what a join should use."""
    one_event = events[events.Type == event_label].drop(columns=["Type"])
    return one_event.rename(
        columns={event_base_time_col: event_time(event_label), event_base_val_col: event_value(event_label)}
    )


def infer_label(dataframe: pd.DataFrame, label_col: str, time_col: str) -> pd.DataFrame:
    """
    Infers boolean label for event columns that have no label, based on existence of time value.
    In the context of a merge_window event, a prediction-row does not have any documentation of an event and is assumed
    to have a negative (0) label.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to modify.
    label_col : str
        The column specifying the value to infer.
    time_col : time_col
        The time column associated with the value to infer.

    Returns
    -------
    pd.DataFrame
        The dataframe with potentially modified labels.
    """
    if label_col not in dataframe.columns or time_col not in dataframe.columns:
        return dataframe

    try:  # Assume numeric labels: edge case Float and Int incompatibilities
        dataframe[label_col] = dataframe[label_col].astype(float)
    except BaseException:  # Leave as nonnumeric
        pass
    dataframe.loc[dataframe[label_col].isna(), label_col] = (
        dataframe.loc[dataframe[label_col].isna(), time_col].notna().astype(int)
    )
    return dataframe


def event_score(
    merged_frame: pd.DataFrame,
    pks: list[str],
    score: str,
    ref_event: Optional[str] = None,
    aggregation_method: str = "max",
) -> pd.DataFrame:
    """
    Reduces a dataframe of all predictions to a single row of significance; such as the max or most recent value for
    an entity.
    Supports max/min for value only scores, and last/first if a reference timestamp is provided.

    Parameters
    ----------
    merged_frame : pd.DataFrame
        The dataframe with score and event data, such as those having an event added via merge_windowed_event.
    pks : list[str]
        A list of identifying keys on which to aggregate, such as Id.
    score : str
        The column name containing the score value.
    ref_event : Optional[str], optional
        The column name containing the time to consider, by default None.
    aggregation_method : str, optional
        A string describing the method to select a value, by default 'max'.

    Returns
    -------
    pd.DataFrame
        The reduced dataframe with one row per combination of pks.
    """
    logger.debug(f"Combining scores using {aggregation_method} for {score} on {ref_event}")
    # groupby.agg works on columns individually - this wants entire row where a condition is met
    # start with first/last/max/min

    ref_score = _resolve_score_col(merged_frame, score)
    if aggregation_method == "max":
        ref_col = ref_score

        def apply_fn(gf):
            return gf.idxmax()

    elif aggregation_method == "min":
        ref_col = ref_score

        def apply_fn(gf):
            return gf.idxmin()

    # merged frame has time columns only for events in appropriate time window,
    # implicitly reduces to positive label (need method to re-add negative samples)
    elif aggregation_method == "last":

        def apply_fn(gf):
            return gf.idxmax()

        ref_col = _resolve_time_col(merged_frame, ref_event)
    elif aggregation_method == "first":

        def apply_fn(gf):
            return gf.idxmin()

        ref_col = _resolve_time_col(merged_frame, ref_event)

    df = merged_frame
    if ref_event is not None:
        event_time = _resolve_time_col(merged_frame, ref_event)
        df = merged_frame[merged_frame[event_time].notna()]

    if len(df.index) == 0:
        return

    pks = [c for c in pks if c in df.columns]
    ix = df.groupby(pks)[ref_col].apply(apply_fn).values
    ix = ix[~np.isnan(ix)]  # dropna

    return merged_frame.loc[ix]


# region Core Methods
def event_value(event: str) -> str:
    """Converts an event name into the value column name."""
    if event.endswith("_Value"):
        return event

    if event.endswith("_Time"):
        event = event[:-5]
    return f"{event}_Value"


def event_time(event: str) -> str:
    """Converts an event name into the time column name."""
    if event.endswith("_Time"):
        return event

    if event.endswith("_Value"):
        event = event[:-6]
    return f"{event}_Time"


def event_name(event: str) -> str:
    """Converts an event column name into the the event name."""
    if event.endswith("_Time"):
        return event[:-5]

    if event.endswith("_Value"):
        return event[:-6]
    return event


def valid_event(dataframe: pd.DataFrame, event: str) -> pd.DataFrame:
    """Filters a dataframe to valid predictions, where the event value has not set to -1."""
    return dataframe[dataframe[event_value(event)] >= 0]


def _resolve_time_col(dataframe: pd.DataFrame, ref_event: str) -> str:
    """
    Determines the time column to use based on existence in the dataframe.
    First assumes it is an event, and checks the time column associated with that name.
    Defaults to the ref_event being the exact column.
    """
    if ref_event is None:
        raise ValueError("Reference event must be specified for last/first summarization")
    ref_time = event_time(ref_event)
    if ref_time not in dataframe.columns:
        if ref_event not in dataframe.columns:
            raise ValueError(f"Reference time column {ref_time} not found in dataframe")
        ref_time = ref_event
    return ref_time


def _resolve_score_col(dataframe: pd.DataFrame, score: str) -> str:
    """
    Determines the value column to use based on existence in the dataframe.
    First assumes the score is a column.
    Defaults to the ref_event being the exact column.
    """
    if score not in dataframe.columns:
        if event_value(score) not in dataframe.columns:
            raise ValueError(f"Score column {score} not found in dataframe.")
        score = event_value(score)
    return score


def _merge_next(
    left: pd.DataFrame,
    right: pd.DataFrame,
    pks: list[str],
    *,
    l_ref: str = "Time",
    r_ref: str = "Time",
    merge_cols_without_times: str = None,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Merges the right frame into the left based on a set of exact match primary keys, prioritizing the first row
    in the right frame occurring after the row in the left.

    Delegates initial distance logic to pandas.DataFrame.merge_asof looking forward to find the next event.
    When indicated, will perform a second left-join to coalesce data from events that were not matching.

    Parameters
    ----------
    left : pd.DataFrame
        The original dataframe.
    right : pd.DataFrame
        The dataframe to join onto left.
    pks : list[str]
        The list of columns to require exact matches during the merge.
    l_ref : str, optional
        The column in the left frame to use as a reference point in the distance match, by default 'Time'.
    r_ref : str, optional
        The column in the right frame to use reference in the distance match, by default 'Time'.
    merge_cols_without_times : Optional[str], optional
        A column name from the right frame, which when provided, will perform a second merge to attempt
        filling in data where the distance merge did not find results, by default None.
    sort : bool
        Whether or not to sort the left/right dataframes, by default True.

    Returns
    -------
    pd.DataFrame
        The merged dataframe.
    """
    if sort:
        left = left.sort_values(l_ref).drop_duplicates(subset=pks + [l_ref]).dropna(subset=[l_ref])
        right = right.sort_values(r_ref)

    rv = pd.merge_asof(left, right.dropna(subset=[r_ref]), left_on=l_ref, right_on=r_ref, by=pks, direction="forward")

    # Second pass to fill in values for no times or no right-event after the left one
    #   This isn't rare when most predictions have no event (have negative labels)..
    if merge_cols_without_times is not None:
        needs_update = rv[merge_cols_without_times].isna()
        direct_merge = pd.merge(left, right.sort_values(r_ref, ascending=False), how="left", on=pks)

        if sort:
            direct_merge = direct_merge.drop_duplicates(subset=pks + [l_ref])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            rv.iloc[needs_update] = direct_merge.sort_values(l_ref).iloc[np.where(needs_update)[0]]

    return rv


# endregion
