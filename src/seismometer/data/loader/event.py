import logging

import pandas as pd

import seismometer.data.pandas_helpers as pdh
from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import Event

logger = logging.getLogger("seismometer")


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
    try:
        events = pd.read_parquet(config.event_path).rename(
            columns={config.ev_type: "Type", config.ev_time: "Time", config.ev_value: "Value"},
            copy=False,
        )
    except BaseException:
        logger.debug(f"No events found at {config.event_path}")
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
    dataframe = (
        dataframe.sort_values(config.predict_time, kind="mergesort")
        .drop_duplicates(subset=config.entity_keys + [config.predict_time])
        .dropna(subset=[config.predict_time])
    )
    event_frame = event_frame.sort_values("Time", kind="mergesort")

    for one_event in config.events.values():
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
) -> pd.DataFrame:
    """Wrapper for calling merge_windowed_event with the correct event column names."""
    disp_event = display if display else event_cols[0]

    return pdh.merge_windowed_event(
        dataframe,
        config.predict_time,
        event_frame.replace({"Type": event_cols}, disp_event),
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
