import logging
from numbers import Number
from typing import Optional, get_args

import numpy as np
import pandas as pd

from seismometer.configuration import ConfigurationError
from seismometer.configuration.model import MergeStrategies

logger = logging.getLogger("seismometer")

MAXIMUM_COUNT_CATS = 15


def merge_windowed_event(
    predictions: pd.DataFrame,
    predtime_col: str,
    events: pd.DataFrame,
    event_label: str,
    pks: list[str],
    *,
    min_leadtime_hrs: Number = 0,
    window_hrs: Optional[Number] = None,
    event_base_val_col: str = "Value",
    event_base_time_col: str = "Time",
    event_base_val_dtype: str = "float",
    sort: bool = True,
    merge_strategy: str = "forward",
    impute_val_with_time: Optional[Number | str] = 1,
    impute_val_no_time: Optional[Number | str] = 0,
) -> pd.DataFrame:
    """
    Merges a single windowed event into a predictions dataframe

    Adds two new event columns: a _Value column with the event value and a _Time column with the event time.
    Ground-truth labeling for a model is considered an event and can have a time associated with it.

    Joins on a set of keys and associates the first event occurring after the prediction time.  The following special
    cases are also applied:

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
    window_hrs : Optional[Number], optional
        The number of hours the window of predictions of interest should be limited to, by default None.
        If None, then all predictions occurring before a known event will be included.
        If used with min_leadtime_hrs, the entire window is shifted maintaining its size. The maximum lookback for a
        prediction is window_hrs + min_leadtime_hrs.
    event_base_val_col : str, optional
        The name of the column in the events frame to merge as the _Value, by default 'Value'.
    event_base_val_dtype : str
        The data type to cast the event value column to, by default 'float'.
    event_base_time_col : str, optional
        The name of the column in the events frame to merge as the _Time, by default 'Time'.
    sort : bool
        Whether or not to sort the predictions/events dataframes, by default True.
    merge_strategy : str
        The method to use when merging the event data, by default 'forward'.
        Options are 'forward', 'nearest', 'first', 'last', and 'count'.
        See seismometer.configuration.model for more information.
    impute_val_with_time : Optional[Number|str], optional
        The value to impute for the label if timestamp exists, defaults to 1.
    impute_val_no_time : Optional[Number|str], optional
        The value to impute for the label if no timestamp exists, defaults to 0.

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
    if merge_strategy not in get_args(MergeStrategies):
        raise ValueError(
            f"Invalid merge strategy {merge_strategy} for {event_label}."
            + f" Must be one of: {', '.join(get_args(MergeStrategies))}."
        )

    # Validate and resolve
    r_ref = "~~reftime~~"
    pks = [col for col in pks if col in events and col in predictions]  # Ensure existence in both frames
    if len(pks) == 0:
        raise ValueError("No common keys found between predictions and events.")

    min_offset = pd.Timedelta(min_leadtime_hrs, unit="hr")

    if sort:
        predictions.sort_values(predtime_col, kind="mergesort", inplace=True)
        events.sort_values(event_base_time_col, kind="mergesort", inplace=True)

    # Preprocess events : reduce and rename
    one_event = _one_event(events, event_label, event_base_val_col, event_base_time_col, pks)
    if len(one_event.index) == 0:
        return predictions

    event_time_col = event_time(event_label)
    event_val_col = event_value(event_label)
    one_event[r_ref] = one_event[event_time_col] - min_offset

    # When merging counts we want to apply the windowing BEFORE the merge.
    # So this case is handled separately due to needing some additional arguments.
    if merge_strategy == "count":
        # Immediately return the merged frame with the counts to avoid unnecessary processing.
        return _merge_event_counts(
            predictions,
            one_event,
            pks,
            event_label,
            event_val_col,
            window_hrs=window_hrs,
            min_offset=min_offset,
            l_ref=predtime_col,
            r_ref=r_ref,
        )

    # merge event specified by merge_strategy for each prediction
    event_ref = event_time_col if merge_strategy in ["forward", "nearest"] else r_ref
    predictions = _merge_with_strategy(
        predictions,
        one_event,
        pks,
        pred_ref=predtime_col,
        event_ref=event_ref,
        event_display=event_label,
        merge_strategy=merge_strategy,
    )

    # Note that filtering happens after merging.
    if window_hrs is not None:  # Clear out events outside window
        max_lookback = pd.Timedelta(window_hrs, unit="hr")  # r_ref has already been moved by min_offset.
        filter_map = (predictions[predtime_col] < (predictions[r_ref] - max_lookback)) | (
            predictions[predtime_col] > (predictions[r_ref])
        )
        predictions.loc[filter_map, [event_val_col, event_time_col]] = pd.NA

    predictions = post_process_event(
        predictions,
        event_val_col,
        event_time_col,
        column_dtype=event_base_val_dtype,
        impute_val_with_time=impute_val_with_time,
        impute_val_no_time=impute_val_no_time,
    )

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


def post_process_event(
    dataframe: pd.DataFrame,
    label_col: str,
    time_col: str,
    *,
    column_dtype: str = "float",
    impute_val_with_time: Optional[Number | str] = 1,
    impute_val_no_time: Optional[Number | str] = 0,
) -> pd.DataFrame:
    """
    Infers and casts events.

    Default assumptions are for binary classifications (cast as float to maximize compatibility with analyses).
    A row that does not have any documentation of an event defaults to a negative (0) label - impute_val_no_time.
    A row that has a timestamp but no event value defaults to a positive (1) label - impute_val_with_time.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to modify.
    label_col : str
        The column specifying the value to infer.
    time_col : time_col
        The time column associated with the value to infer.
    column_dtype : str
        The data type to cast the label column to, done after imputation; by default 'float'.
    impute_val_with_time : Optional[Number|str], optional
        The value to impute for the label if timestamp exist, defaults to 1.
    impute_val_no_time : Optional[Number|str], optional
        The value to impute for the label if no timestamp exist, defaults to 0.

    Returns
    -------
    pd.DataFrame
        The dataframe with potentially modified labels.
    """
    if label_col not in dataframe.columns or time_col not in dataframe.columns:
        return dataframe

    # use pandas for compatibility of imputations -- handle Nones
    impute_val = pd.Series(
        [impute_val_no_time or 0, impute_val_with_time or 1],
        dtype=dataframe[label_col].dtype,
        index=["no_time", "with_time"],
    )

    label_na_map = dataframe[label_col].isna()
    time_na_map = dataframe[time_col].isna()

    if impute_val_with_time is not None:
        dataframe.loc[(label_na_map & ~time_na_map), label_col] = impute_val.with_time
    if impute_val_no_time is not None:
        dataframe.loc[dataframe[label_col].isna(), label_col] = impute_val.no_time

    if column_dtype is None:
        return dataframe

    # cast after imputation - supports nonnullable types
    try_casting(dataframe, label_col, column_dtype)

    return dataframe


def try_casting(dataframe: pd.DataFrame, column: str, column_dtype: str) -> None:
    """
    Attempts to cast a column to a specified data type inplace.

    Will convert the specified column to a data type, raising a ConfigurationError if the conversion fails.
    Does multistep casts to get strings "1.0" into format int 1.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to modify.
    column : str
        The column to cast.
    column_dtype : str
        The data type to cast the column to.

    Raises
    ------
    ConfigurationError
        If the column cannot be cast to the specified data type.
    """
    try:
        if "int" in column_dtype.lower():  # "1.0" -> 1.0 then 1.0 -> 1
            dataframe[column] = dataframe[column].astype(float)
        dataframe[column] = dataframe[column].astype(column_dtype)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(
            f"Cannot cast '{event_name(column)}' values to '{column_dtype}'. "
            + "Update dictionary config or contact the model owner."
        ) from exc


def _merge_event_counts(
    left: pd.DataFrame,
    right: pd.DataFrame,
    pks: list[str],
    event_name: str,
    event_label: str,
    window_hrs: Optional[Number] = None,
    min_offset: pd.Timedelta = pd.Timedelta(0, unit="hr"),
    l_ref: str = "Time",
    r_ref: str = "~~reftime~~",
) -> pd.DataFrame:
    """Creates a new column for each event in the right frame's event_label column,
    counting the number of times that event has occurred"""

    if l_ref == r_ref:
        raise ValueError(
            f"`l_ref` and `r_ref` must be different to avoid column collisions during merge (both are '{l_ref}')."
        )

    if window_hrs is not None:
        # Filter out rows with missing times if checking window hours
        if len(right_filtered := right[right[r_ref].notna()]) == 0:
            logger.warning(f"No times found for {event_name}! Unable to merge any counts.")
            return left
        if diff := len(right) - len(right_filtered) > 0:
            logger.warning(f"Found {diff} rows with missing times for {event_name}. These rows will be ignored.")
            right = right_filtered

        max_lookback = pd.Timedelta(window_hrs, unit="hr") + min_offset  # Keep window the specified size
        right = pd.merge(right, left[pks + [l_ref]], on=pks, how="left")
        right = right[right[l_ref] <= right[r_ref]]  # Filter to only events that happened at or after the prediction
        right = right[
            right[l_ref] > (right[r_ref] - max_lookback)
        ]  # Filter to only events that happened within the window

    # Validate number of categories to create columns for
    pop_counts = right[event_label].value_counts()
    if (N := len(pop_counts)) > MAXIMUM_COUNT_CATS:
        logger.warning(
            f"Maximum number of unique events to count is {MAXIMUM_COUNT_CATS}, but {N} were found for {event_name}. "
            + f"Only the top {MAXIMUM_COUNT_CATS} by number of appearances will be included."
        )
        # Filter right frame to the only contain the top MAXIMUM_COUNT_CATS events
        events_to_count = pop_counts.iloc[:MAXIMUM_COUNT_CATS].keys()
        right = right[right[event_label].isin(events_to_count)]
    del pop_counts

    event_name_map = {
        event: event_value_count(str(event_label), str(event)) for event in right[event_label].unique()
    }  # Create dictionary to map column names with

    # Create a value counts dataframe where each event is a column containing the count of that
    # event grouped by the primary keys.
    val_counts: pd.DataFrame = right.groupby(pks, as_index=False)[event_label].value_counts()
    val_counts = (
        val_counts.pivot(index=pks, columns=event_label, values="count")
        .reset_index()
        .fillna(0)
        .rename(columns=event_name_map)
    )

    left = pd.merge(left, val_counts, on=pks, how="left")  # Merge counts into left frame
    count_cols = list(event_name_map.values())
    left[count_cols] = left[count_cols].fillna(0)  # Fill any missing counts for rows that didn't have any events

    return left


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
    try:
        ct_times = one_event[event_ref].notna().sum()

        # If there are no times in the event frame, merge the first row for each group
        if ct_times == 0:
            # Set the filtered frame to the first row for each group and throw a value error
            # which is passed before merging.
            one_event_filtered = one_event.groupby(pks).first()
            raise ValueError(f"No times found for {event_display}, merging first row for each group.")

        if ct_times != len(one_event.index):
            logger.warning(f"Inconsistent event times for {event_display}, only considering events with times.")
            one_event = one_event.dropna(subset=[event_ref])

        if merge_strategy == "forward" or merge_strategy == "nearest":
            return pd.merge_asof(
                predictions,
                one_event.dropna(subset=[event_ref]),
                left_on=pred_ref,
                right_on=event_ref,
                by=pks,
                direction=merge_strategy,
            )

        # Assume sorted on event_ref before being passed in
        if merge_strategy == "first":
            one_event_filtered = one_event.groupby(pks).first().reset_index()
        if merge_strategy == "last":
            one_event_filtered = one_event.groupby(pks).last().reset_index()

    except ValueError as e:
        logger.warning(e)
        pass

    return pd.merge(predictions, one_event_filtered, on=pks, how="left")


def max_aggregation(df: pd.DataFrame, pks: list[str], score: str, ref_time: str, ref_event: str) -> pd.DataFrame:
    """
    Aggregates the DataFrame by selecting the maximum score value.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to aggregate.
    pks : list[str]
        A list of identifying keys on which to aggregate.
    score : str
        The column name containing the score value.
    ref_time : Optional[str], optional
        The column name containing the time to consider, by default None.
    ref_event : Optional[str], optional
        The column name containing the event to consider, by default None.

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame.
    """
    if ref_event is None:
        raise ValueError("With aggregation_method 'max', ref_event is required.")

    event_val = event_value(ref_event)
    ref_score = _resolve_score_col(df, score)
    df = df.sort_values(by=[event_val, ref_score], ascending=False)
    return df.drop_duplicates(subset=pks)


def min_aggregation(df: pd.DataFrame, pks: list[str], score: str, ref_time: str, ref_event: str) -> pd.DataFrame:
    """
    Aggregates the DataFrame by selecting the minimum score value.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to aggregate.
    pks : list[str]
        A list of identifying keys on which to aggregate.
    score : str
        The column name containing the score value.
    ref_time : Optional[str], optional
        The column name containing the time to consider, by default None.
    ref_event : Optional[str], optional
        The column name containing the event to consider, by default None.

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame.
    """
    if ref_event is None:
        raise ValueError("With aggregation_method 'min', ref_event is required.")

    event_val = event_value(ref_event)
    ref_score = _resolve_score_col(df, score)
    df = df.sort_values(by=[event_val, ref_score], ascending=[False, True])
    return df.drop_duplicates(subset=pks)


def first_aggregation(df: pd.DataFrame, pks: list[str], score: str, ref_time: str, ref_event: str) -> pd.DataFrame:
    """
    Aggregates the DataFrame by selecting the first occurrence based on event time.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to aggregate.
    pks : list[str]
        A list of identifying keys on which to aggregate.
    score : str
        The column name containing the score value.
    ref_time : Optional[str], optional
        The column name containing the time to consider, by default None.
    ref_event : Optional[str], optional
        The column name containing the event to consider, by default None.

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame.
    """
    if ref_time is None:
        raise ValueError("With aggregation_method 'first', ref_time is required.")

    reference_time = _resolve_time_col(df, ref_time)
    df = df[df[reference_time].notna()]
    df = df.sort_values(by=reference_time)
    return df.drop_duplicates(subset=pks)


def last_aggregation(df: pd.DataFrame, pks: list[str], score: str, ref_time: str, ref_event: str) -> pd.DataFrame:
    """
    Aggregates the DataFrame by selecting the last occurrence based on event time.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to aggregate.
    pks : list[str]
        A list of identifying keys on which to aggregate.
    score : str
        The column name containing the score value.
    ref_time : Optional[str], optional
        The column name containing the time to consider, by default None.
    ref_event : Optional[str], optional
        The column name containing the event to consider, by default None.

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame.
    """
    if ref_time is None:
        raise ValueError("With aggregation_method 'last', ref_time is required.")

    reference_time = _resolve_time_col(df, ref_time)
    df = df[df[reference_time].notna()]
    df = df.sort_values(by=reference_time, ascending=False)
    return df.drop_duplicates(subset=pks)


def event_score(
    merged_frame: pd.DataFrame,
    pks: list[str],
    score: str,
    ref_time: Optional[str] = None,
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
    ref_time : Optional[str], optional
        The column name containing the time to consider, by default None.
        Required when aggregation_method requires a time reference (e.g., 'first', 'last').
        Note that we drop NaT rows first and consequently we pick the row satisfying the
        aggregation_method that also corresponds to a positive case for the associated event.
    ref_event : Optional[str], optional
        The column name containing the event to consider, by default None.
        Required when aggregation_method requires an event reference to prioritize positive cases (e.g., 'max', 'min')
        Note that we pick the row satisfying the aggregation_method among scores associated with a positive case of
        ref_event if there are any positive cases. In case there are no positive case, we just pick the row satisfying
        the aggregation_method.
    aggregation_method : str, optional
        A string describing the method to select a value, by default 'max'.

    Returns
    -------
    pd.DataFrame
        The reduced dataframe with one row per combination of pks.
    """
    logger.debug(
        f"Combining scores using {aggregation_method} for {score} on ref_time: {ref_time} "
        + f"and ref_event: {ref_event}"
    )
    pks = [c for c in pks if c in merged_frame.columns]

    aggregation_methods = {
        "max": max_aggregation,
        "min": min_aggregation,
        "first": first_aggregation,
        "last": last_aggregation,
    }

    if aggregation_method not in aggregation_methods:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    df = aggregation_methods[aggregation_method](merged_frame, pks, score, ref_time, ref_event)
    return df.loc[~np.isnan(df.index)]


def get_model_scores(
    dataframe: pd.DataFrame,
    entity_keys: list[str],
    score_col: str,
    ref_time: Optional[str],
    ref_event: Optional[str],
    aggregation_method: str = "max",
    per_context_id: bool = False,
) -> pd.DataFrame:
    """
    Reduces a dataframe of all predictions to a single row of significance; such as the max or most recent value for
    an entity.
    Supports max/min for value only scores, and last/first if a reference timestamp is provided.

    Parameters
    ----------
    merged_frame : pd.DataFrame
        The dataframe with score and event data, such as those having an event added via merge_windowed_event.
    entity_keys : list[str]
        A list of identifying keys on which to aggregate, such as Id.
    score_col : str
        The column name containing the score value.
    ref_time : Optional[str], optional
        The column name containing the time to consider, by default None.
    ref_event : Optional[str], optional
        The column name containing the event to consider, by default None.
    aggregation_method : str, optional
        A string describing the method to select a value, by default 'max'.
    per_context_id : bool, optional
        If True, limits data to one row per context_id, by default False.

    Returns
    -------
    pd.DataFrame
        The reduced dataframe with one row per combination of pks.
    """
    if per_context_id:
        return event_score(
            dataframe,
            entity_keys,
            score=score_col,
            ref_time=ref_time,
            ref_event=ref_event,
            aggregation_method=aggregation_method,
        )
    return dataframe


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


def event_value_count(event_label: str, event_value: str) -> str:
    """Converts a value of an event column into the count column name."""
    event_label = event_name(event_label)

    if event_value.endswith("_Count"):
        return f"{event_label}~{event_value}"

    return f"{event_label}~{event_value}_Count"


def event_name(event: str) -> str:
    """Converts an event column name into the the event name."""
    if event.endswith("_Time"):
        return event[:-5]

    if event.endswith("_Value"):
        return event[:-6]
    return event


def event_value_name(event_value: str) -> str:
    """Converts event value count column into the event value name."""
    val = event_value
    if "~" in val:
        val = val.split("~")[1]
    if val.endswith("_Count"):
        val = val[:-6]

    return val


def is_valid_event(dataframe: pd.DataFrame, event: str, ref: str) -> pd.DataFrame:
    """
    Creates a mask excluding rows (False) where the event occurs before the reference time.
    If the comparison cannot be made, all rows will be considered valid (True).
    """
    if event_time(event) not in dataframe.columns or ref not in dataframe.columns:
        return pd.Series([True] * len(dataframe), index=dataframe.index)
    return dataframe[ref] <= dataframe[event_time(event)]


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


# endregion
