import logging
from typing import Optional

import numpy as np
import pandas as pd

import seismometer.data.pandas_helpers as pdh
from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import Event

from .pipeline import ConfigOnlyHook

logger = logging.getLogger("seismometer")


def get_data_loader(config: ConfigProvider) -> ConfigOnlyHook:
    """
    Returns the proper data loader function from the event file extension.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.

    Returns
    -------
    ConfigOnlyHook
        The predictions dataframe.
    """
    loaders = {
        ".csv": csv_loader,
        ".tsv": tsv_loader,
        ".parquet": parquet_loader,
    }
    return loaders.get(config.event_path.suffix.lower(), parquet_loader)


def csv_loader(config: ConfigProvider) -> pd.DataFrame:
    """
    Loads the events frame from a csv file based on config.event_path.

    Maps the non-key columns to the standard names "Type", "Time", "Value".
    Will log debug message and return an empty dataframe if read_csv fails.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.

    Returns
    -------
    pd.DataFrame
        the events dataframe with standarized column names.
    """
    return _sv_loader(config, ",")


def tsv_loader(config: ConfigProvider) -> pd.DataFrame:
    """
    Loads the events frame from a csv file based on config.event_path.

    Maps the non-key columns to the standard names "Type", "Time", "Value".
    Will log debug message and return an empty dataframe if read_csv fails.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.

    Returns
    -------
    pd.DataFrame
        the events dataframe with standarized column names.
    """
    return _sv_loader(config, "\t")


def parquet_loader(config: ConfigProvider) -> pd.DataFrame:
    """
    Loads the events frame from a parquet file based on config.event_path.

    Maps the non-key columns to the standard names "Type", "Time", "Value".
    Will log debug message and return an empty dataframe if read_parquet fails.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.

    Returns
    -------
    pd.DataFrame
        the events dataframe with standarized column names.
    """
    logger.debug(f"Loading events from {config.event_path}.")
    try:
        events = pd.read_parquet(config.event_path)
        _rename_event_columns(config, events)
        logger.debug(f"Loaded {len(events)} events from {config.event_path}.")
    except BaseException:
        logger.warning(f"No events found at {config.event_path}")
        events = pd.DataFrame(columns=config.entity_keys + ["Type", "Time", "Value"])

    return events


def post_transform_fn(config: ConfigProvider, events: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the Time column in events to a datetime64[ns] type, to be compatible with other operations.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.
    events : pd.DataFrame
        The events dataframe.

    Returns
    -------
    pd.DataFrame
        The transformed events dataframe.
    """
    # Time column in events is known
    events["Time"] = events["Time"].astype("<M8[ns]")

    logger.debug("Transformed events dataframe with Time as datetime64[ns].")
    return events


# data merges
def merge_onto_predictions(config: ConfigProvider, event_frame: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Merges each configured event onto the predictions dataframe.

    Will create event columns (_Value + _Time) for each configured event.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.
    event_frame : pd.DataFrame
        The events dataframe.
    dataframe : pd.DataFrame
        The target predictions dataframe.

    Returns
    -------
    pd.DataFrame
        The merged dataframe of predictions plus new event columns.
    """
    logger.debug("Merging events onto predictions dataframe.")
    dataframe = dataframe.sort_values(config.predict_time, kind="mergesort")
    dataframe.index = pd.RangeIndex(len(dataframe))
    event_frame = event_frame.sort_values("Time", kind="mergesort")
    event_frame.index = pd.RangeIndex(len(event_frame))
    pks = [key for key in config.entity_keys if key in dataframe and key in event_frame]
    uses_asof = any(event.merge_strategy in ("forward", "nearest") for event in config.events.values())
    key_codes = pdh.factorize_keys(dataframe, event_frame, pks) if pks and uses_asof else None
    dataframe, key_codes = _drop_repeated_predictions(config, dataframe, event_frame, key_codes)
    logger.debug(
        f"Sorted predictions dataframe by {config.predict_time}, dropping duplicate (entity time, prediction time) "
        "and dropping predictions with Null prediction time."
    )

    for one_event in config.events.values():
        logger.debug(f"Processing event {one_event.display_name} with sources {one_event.source}")
        # Get dtype
        event_dtype = _get_source_type(config, one_event)

        # Merge
        if one_event.window_hr:
            logger.debug(
                f"Windowing event {one_event.display_name} to lookback {one_event.window_hr} "
                + f"offset by {one_event.offset_hr}"
            )
            dataframe = _merge_event(
                config,
                one_event.source,
                dataframe,
                event_frame,
                event_dtype=event_dtype,
                window_hrs=one_event.window_hr,
                offset_hrs=one_event.offset_hr,
                display=one_event.display_name,
                sort=False,
                impute_pos=one_event.impute_val,
                key_codes=key_codes,
            )
        else:  # No lookback
            logger.debug(f"Merging event {one_event.display_name}")
            dataframe = _merge_event(
                config,
                one_event.source,
                dataframe,
                event_frame,
                event_dtype=event_dtype,
                display=one_event.display_name,
                sort=False,
                impute_pos=one_event.impute_val,
                key_codes=key_codes,
            )

        # Impute no event
        if one_event.impute_val and one_event.impute_val != 0:
            event_val = pdh.event_value(one_event.display_name)
            logger.warning(
                f"Event {one_event.display_name} specified impute; "
                + "currently missing event value is being inferred based on timestamp existence."
            )
            impute = one_event.impute_val
            dataframe[event_val] = dataframe[event_val].fillna(impute)

    return dataframe


def _drop_repeated_predictions(
    config: ConfigProvider, dataframe: pd.DataFrame, event_frame: pd.DataFrame, key_codes: Optional[pdh.KeyCodes]
) -> tuple[pd.DataFrame, Optional[pdh.KeyCodes]]:
    """
    Keeps the first prediction for each entity keys + prediction time, dropping those without a prediction time.

    Compares the integer key codes rather than the key columns when the codes cover every entity key, which is much
    faster for string keys. Returns the frame with a fresh index and the codes for the kept rows.
    """
    predict_time = config.predict_time
    if (
        key_codes is not None
        and key_codes.pks == tuple(config.entity_keys)
        # the codes come from both frames' keys combined, so they only keep equality when the dtypes match in kind
        and all(pdh._key_kind(dataframe[key]) == pdh._key_kind(event_frame[key]) for key in key_codes.pks)
    ):
        keys = pd.DataFrame({"key": key_codes.predictions.to_numpy(), "time": dataframe[predict_time].array})
        repeated = keys.duplicated().to_numpy()
    else:
        repeated = dataframe.duplicated(subset=config.entity_keys + [predict_time]).to_numpy()
    rows = np.flatnonzero(~repeated & dataframe[predict_time].notna().to_numpy())

    dataframe = dataframe.take(rows)
    # The index merge_asof would return; keeps the cached key codes aligned across merges.
    dataframe.index = pd.RangeIndex(len(dataframe))
    if key_codes is not None:
        codes = pd.Series(key_codes.predictions.to_numpy()[rows], index=dataframe.index)
        key_codes = key_codes._replace(predictions=codes)
    return dataframe, key_codes


def _merge_event(
    config,
    event_cols,
    dataframe,
    event_frame,
    event_dtype=None,
    offset_hrs=0,
    window_hrs=None,
    display="",
    sort=True,
    impute_pos=1,
    impute_neg=0,
    key_codes=None,
) -> pd.DataFrame:
    """Wrapper for calling merge_windowed_event with the correct event column names."""
    disp_event = display if display else event_cols[0]
    logger.debug(f"Merging event {disp_event} with columns {event_cols} onto predictions dataframe.")
    # Copies only this event's rows instead of the whole events frame; rows already named disp_event count too
    rows = event_frame["Type"].isin(list(event_cols) + [disp_event])

    return pdh.merge_windowed_event(
        dataframe,
        config.predict_time,
        event_frame.loc[rows].assign(Type=disp_event),
        disp_event,
        config.entity_keys,
        min_leadtime_hrs=offset_hrs,
        window_hrs=window_hrs,
        event_base_time_col="Time",
        event_base_val_dtype=event_dtype,
        sort=sort,
        merge_strategy=config.events[disp_event].merge_strategy,
        impute_val_with_time=impute_pos,
        impute_val_no_time=impute_neg,
        key_codes=key_codes,
    )


def _get_source_type(config: ConfigProvider, event: Event) -> str:
    """Get the dtype of the first source of the specified event."""
    dtype = None
    multitypes = set()  # Aggregate woarning messages
    for source in event.source:
        new_type = None
        if (source_defn := config.event_defs.get(source, None)) is not None:
            new_type = source_defn.dtype

        if new_type is None:
            continue
        if dtype is None:
            dtype = new_type
            continue

        if dtype != new_type:
            multitypes.add(new_type)

    if len(multitypes) > 1:
        logger.warning(
            f"Multiple types found for event {event.display_name}. "
            + f"Will use the first source of type {dtype} and ignore others: {','.join(multitypes)}"
        )

    return dtype


# other


def _rename_event_columns(config: ConfigProvider, events: pd.DataFrame) -> None:
    """Renames the event columns in place, avoiding rename's data copy on pandas 2 and its deprecated inplace."""
    names = {config.ev_type: "Type", config.ev_time: "Time", config.ev_value: "Value"}
    events.columns = [names.get(col, col) for col in events.columns]


def _sv_loader(config: ConfigProvider, sep) -> pd.DataFrame:
    """General loader for CSV or TSV files"""
    try:
        events = pd.read_csv(config.event_path, sep=sep)
        _rename_event_columns(config, events)

        # since importing CSVs automatically cast numbers to ints, make sure the columns
        # shared with predictions become strings so we don't have a type mismatch
        defined_types = config.prediction_types
        usage = config.usage
        for col in [usage.entity_id, usage.context_id, usage.predict_time]:
            if col is not None and defined_types[col] == "object":
                events[col] = events[col].astype(str)
    except BaseException:
        logger.warning(f"No events found at {config.event_path}")
        events = pd.DataFrame(columns=config.entity_keys + ["Type", "Time", "Value"])

    return events
