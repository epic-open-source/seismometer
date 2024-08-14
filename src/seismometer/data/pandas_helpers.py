import logging
from numbers import Number
from typing import Optional

from seismometer.configuration.model import MergeStrategies

import numpy as np
import pandas as pd

logger = logging.getLogger("seismometer")

MAXIMUM_NUM_COUNTS = 15


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
    merge_strategy: str = "forward",
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
    merge_strategy : str
        The method to use when merging the event data, by default 'forward'.

    Returns
    -------
    pd.DataFrame
        The predictions dataframe with the new time and value columns for the event specified.

    Raises
    ------
    ValueError
        At least one column in pks must be in both the predictions and events dataframes.
    """
    # Validate merge strategy
    if merge_strategy not in MergeStrategies.__args__:
        raise ValueError(f"Invalid merge strategy {merge_strategy} for {event_label}. Must be one of: {', '.join(MergeStrategies.__args__)}.")
    
    # Validate and resolve
    r_ref = "~~reftime~~"
    pks = [col for col in pks if col in events and col in predictions]  # Ensure existence in both frames
    if len(pks) == 0:
        raise ValueError("No common keys found between predictions and events.")

    min_offset = pd.Timedelta(min_leadtime_hrs, unit="hr")

    if sort:
        predictions.sort_values(predtime_col, inplace=True)
        events.sort_values(event_base_time_col, inplace=True)

    # Preprocess events : reduce and rename
    one_event = _one_event(events, event_label, event_base_val_col, event_base_time_col, pks)
    if len(one_event.index) == 0:
        return predictions

    event_time_col = event_time(event_label)
    event_val_col = event_value(event_label)
    one_event[r_ref] = one_event[event_time_col] - min_offset

    if merge_strategy=="count":
        return _merge_event_counts(predictions, one_event, pks, event_label, event_val_col, window_hrs=window_hrs, min_offset=min_offset, l_ref=predtime_col, r_ref=r_ref)

    # merge next event for each prediction
    predictions = _merge_with_strategy(
        predictions, one_event, pks, pred_ref=predtime_col, event_ref=r_ref, event_display=event_label, merge_strategy=merge_strategy
    )


    if window_hrs is not None:  # Clear out events outside window
        max_lookback = pd.Timedelta(window_hrs, unit="hr") + min_offset  # keep window the specified size
        filter_map = predictions[predtime_col] < (predictions[r_ref] - max_lookback)
        predictions.loc[filter_map, [event_val_col, event_time_col]] = pd.NA

    predictions = infer_label(predictions, event_val_col, event_time_col)

    # refactor to generalize
    if merge_strategy=="forward": #For forward merges, don't count events that happen before the prediction
        predictions.loc[predictions[predtime_col] > predictions[r_ref], event_val_col] = -1
    
    return predictions.drop(columns=r_ref)


def _one_event(
    events: pd.DataFrame, event_label: str, event_base_val_col: str, event_base_time_col: str, pks: list[str]
) -> pd.DataFrame:
    """Reduces the events dataframe to those rows associated with the event_label, preemptively renaming to the
    columns to what a join should use and reducing columns to pks + event value and time."""
    expected_columns = pks + [event_base_val_col, event_base_time_col]
    one_event = events.loc[events.Type == event_label, expected_columns][expected_columns]
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

def _merge_event_counts(
    left: pd.DataFrame,
    right: pd.DataFrame,
    pks: list[str],
    event_name: str,
    event_label: str,
    window_hrs: Optional[Number] = None,
    min_offset: Number = 0,
    l_ref: str = "Time",
    r_ref: str = "Time",
) -> pd.DataFrame:
    """Creates a new column for each event in the right frame's event_label column, 
    counting the number of times that event has occurred"""

    if (N := right[event_label].nunique()) > MAXIMUM_NUM_COUNTS:
        logger.warning(f"Maximum number of unique events to count is {MAXIMUM_NUM_COUNTS}, but {N} were found for {event_name}. Only the first {MAXIMUM_NUM_COUNTS} will be counted.")
        #Filter right frame to the only contain the first MAXIMUM_NUM_COUNTS events
        events_to_count = right[event_label].unique()[:MAXIMUM_NUM_COUNTS]
        right = right[right[event_label].isin(events_to_count)]

    event_name_map = {event: str(event)+"_Count" for event in right[event_label].unique()} #Create dictionary to map column names with

    if window_hrs is not None:
        max_lookback = pd.Timedelta(window_hrs, unit="hr") + min_offset  #Keep window the specified size
        right = pd.merge(right, left[pks+[l_ref]], on=pks, how="left")
        right = right[(right[r_ref] - max_lookback) < right[l_ref]] #Filter to only events that happened within the window
    
    #Create a value counts dataframe where each event is a column containing the count of that event grouped by the primary keys
    val_counts = right.groupby(pks, as_index=False)[event_label].value_counts()
    val_counts = val_counts.pivot(index=pks, columns=event_label, values='count').reset_index().fillna(0).rename(columns=event_name_map)

    left = pd.merge(left, val_counts, on=pks, how="left") #Merge counts into left frame
    
    return left


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


def _merge_with_strategy(
    predictions: pd.DataFrame,
    one_event: pd.DataFrame,
    pks: list[str],
    *,
    pred_ref: str = "Time",
    event_ref: str = "Time",
    event_display: str = "an event",
    merge_strategy: str = "forward",
) -> pd.DataFrame:
    """
    Merges the right frame into the left based on a set of exact match primary keys and merge strategy. 

    Parameters
    ----------
    predictions : pd.DataFrame
        The left frame, usually of predictions. Assumed to be sorted by time.
    one_event : pd.DataFrame
        The right frame to merge, assumed to be of events. Assumed to be sorted by time if applicable.
    pks : list[str]
        The list of columns to require exact matches during the merge.
    pred_ref : str, optional
        The column in the left (prediction) frame to use as a reference point in the distance match, by default 'Time'.
    event_ref : str, optional
        The column in the right (event) frame to use reference in the distance match, by default 'Time'.
    event_display : str, optional
        The name of the event to display in warning messages, by default "an event"
    merge_strategy : str
        The method to use when merging the event data, by default 'forward'.

    Returns
    -------
    pd.DataFrame
        The merged dataframe.
    """
    if merge_strategy=="forward" or merge_strategy=="nearest":
        ct_times = one_event[event_ref].notna().sum()
        if ct_times == 0:
            logger.warning(f"No times found for {event_display}, merging 'first'")
            return pd.merge(predictions, one_event.groupby(pks).first(), how="left", on=pks)

        if ct_times != len(one_event.index):
            logger.warning(f"Inconsistent event times for {event_display}, only considering events with times.")
            one_event = one_event.dropna(subset=[event_ref])
        return pd.merge_asof(
            predictions,
            one_event.dropna(subset=[event_ref]),
            left_on=pred_ref,
            right_on=event_ref,
            by=pks,
            direction=merge_strategy,
        )
     
    #If there's multiple events with matching event_ref vals, idxmax and idxmin will return the first row.
    #So we sort the index before grabbing the value to default to the first and last index if multiple events happen simultaneously.
    if merge_strategy=="first":
        one_event_filtered = one_event.loc[one_event.sort_index().groupby(pks)[event_ref].idxmin()]
    if merge_strategy=="last": 
        one_event_filtered = one_event.loc[one_event.sort_index().groupby(pks)[event_ref].idxmax()]
    
    return pd.merge(predictions, one_event_filtered, on=pks, how="left")


# endregion
