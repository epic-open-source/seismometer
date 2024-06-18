import logging

import pandas as pd

import seismometer.data.pandas_helpers as pdh
from seismometer.configuration import ConfigProvider

logger = logging.getLogger("seismometer")


def parquet_loader(config: ConfigProvider) -> pd.DataFrame:
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
    # Time column in events is known
    events["Time"] = events["Time"].astype("<M8[ns]")

    return events


# data merges
def merge_onto_predictions(config: ConfigProvider, event_frame: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = (
        dataframe.sort_values(config.predict_time)
        .drop_duplicates(subset=config.entity_keys + [config.predict_time])
        .dropna(subset=[config.predict_time])
    )
    event_frame = event_frame.sort_values("Time")

    for one_event in config.events:
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
                window_hrs=one_event.window_hr,
                offset_hrs=one_event.offset_hr,
                display=one_event.display_name,
                sort=False,
            )
            config.target_cols.append(one_event.display_name)
        else:  # No lookback
            logger.debug(f"Merging event {one_event.display_name}")
            dataframe = _merge_event(
                config, one_event.source, dataframe, event_frame, display=one_event.display_name, sort=False
            )

        # Impute
        if one_event.impute_val or one_event.usage == "target":
            event_val = pdh.event_value(one_event.display_name)
            impute = one_event.impute_val or 0  # Enforce binary type target
            dataframe[event_val] = dataframe[event_val].fillna(impute)

    return dataframe


def _merge_event(
    config, event_col, dataframe, event_frame, offset_hrs=0, window_hrs=None, display="", sort=True
) -> pd.DataFrame:
    disp_event = display if display else event_col
    translate_event = event_col if display else ""

    return pdh.merge_windowed_event(
        dataframe,
        config.predict_time,
        event_frame.replace({"Type": translate_event}, disp_event),
        disp_event,
        config.entity_keys,
        min_leadtime_hrs=offset_hrs,
        window_hrs=window_hrs,
        event_base_time_col="Time",
        sort=sort,
    )
