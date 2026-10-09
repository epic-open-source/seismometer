"""
Event merge accelerated with polars, used by ``memory_efficient`` sessions.

Imported only by run_startup(memory_efficient=True), so polars stays an optional dependency (``seismometer[perf]``).
Polars sorts and deduplicates the predictions and finds each prediction's next event; windowing, imputation and
casting reuse the pandas code, so the result matches event.merge_onto_predictions. Configurations it doesn't cover
fall back to that function.
"""
import inspect
import logging
from typing import Optional

import numpy as np
import pandas as pd

import seismometer.data.pandas_helpers as pdh
from seismometer.configuration import ConfigProvider

from . import event as pandas_event

logger = logging.getLogger("seismometer")

try:
    import polars as pl
except ImportError:  # optional dependency
    pl = None

_ROW = "~~row~~"
_EVENT_ROW = "~~event_row~~"
_REF_TIME = "~~reftime~~"  # the reference time column merge_windowed_event uses
# Both frames are sorted before joining, so skip polars' check (it only warns when joining by groups)
_ASOF_KWARGS = (
    {"check_sortedness": False}
    if pl is not None and "check_sortedness" in inspect.signature(pl.DataFrame.join_asof).parameters
    else {}
)


def available() -> bool:
    """Whether polars is installed."""
    return pl is not None


def merge_onto_predictions(config: ConfigProvider, event_frame: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Merges each configured event onto the predictions dataframe, matching events with polars.

    Drop-in replacement for event.merge_onto_predictions with the same result. Falls back to it for merge strategies
    other than forward, missing or mismatched keys, events without any times, and event columns that already exist.

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
    reason = _unsupported_reason(config, event_frame, dataframe)
    if reason is not None:
        logger.info(f"memory_efficient: merging events with pandas, since {reason}.")
        return pandas_event.merge_onto_predictions(config, event_frame, dataframe)

    logger.debug("Merging events onto predictions dataframe with polars.")
    keys, predict_time = config.entity_keys, config.predict_time

    # Same rows and order as the pandas merge: stable sort on prediction time, then drop duplicate keys + times
    # (keeping the first) and missing times.
    left = (
        _to_polars(dataframe, keys + [predict_time])
        .with_row_index(_ROW)
        .sort(predict_time, maintain_order=True, nulls_last=True)
        .unique(subset=keys + [predict_time], keep="first", maintain_order=True)
        .drop_nulls(predict_time)
    )
    dataframe = dataframe.take(left[_ROW].to_numpy())
    dataframe.index = pd.RangeIndex(len(dataframe))
    left = left.drop(_ROW)

    # Stable sort on time, as in the pandas merge, so ties match the same event
    events = (
        _to_polars(event_frame, keys + ["Type", "Time"])
        .with_row_index(_EVENT_ROW)
        .sort("Time", maintain_order=True, nulls_last=True)
    )

    for one_event in config.events.values():
        logger.debug(f"Processing event {one_event.display_name} with sources {one_event.source}")
        dataframe = _merge_event(config, one_event, dataframe, left, events, event_frame)

        # Impute no event
        if one_event.impute_val and one_event.impute_val != 0:
            event_val = pdh.event_value(one_event.display_name)
            logger.warning(
                f"Event {one_event.display_name} specified impute; "
                + "currently missing event value is being inferred based on timestamp existence."
            )
            dataframe[event_val] = dataframe[event_val].fillna(one_event.impute_val)

    return dataframe


def _merge_event(config, one_event, dataframe, left, events, event_frame) -> pd.DataFrame:
    """Adds one event's value and time columns, matching what merge_windowed_event does for a forward merge."""
    display = one_event.display_name
    keys, predict_time = config.entity_keys, config.predict_time
    windowed = bool(one_event.window_hr)  # merge_onto_predictions only windows (and offsets) for a nonzero window
    window_hrs = one_event.window_hr if windowed else None
    offset_hrs = one_event.offset_hr if windowed else 0

    # merge_windowed_event renames sources to the display name, so rows already named that are included too
    candidates = events.filter(pl.col("Type").is_in(list(one_event.source) + [display]))
    if candidates.height == 0:
        logger.debug(f"No events found for {display} with keys {keys}.")
        return dataframe
    right = candidates.drop_nulls("Time")
    if right.height != candidates.height:
        logger.warning(f"Inconsistent event times for {display}, only considering events with times.")

    matched = left.join_asof(
        right.select(keys + ["Time", _EVENT_ROW]),
        left_on=predict_time,
        right_on="Time",
        by=keys,
        strategy="forward",
        **_ASOF_KWARGS,
    )
    rows = matched[_EVENT_ROW].fill_null(-1).to_numpy().astype(np.intp)

    time_col = pdh.event_time(display)
    dataframe[pdh.event_value(display)] = _take(event_frame["Value"], rows)
    dataframe[time_col] = _take(event_frame["Time"], rows)
    dataframe[_REF_TIME] = dataframe[time_col] - pd.Timedelta(offset_hrs, unit="hr")

    return pdh._finish_windowed_event(
        dataframe,
        predict_time,
        display,
        _REF_TIME,
        window_hrs=window_hrs,
        min_leadtime_hrs=offset_hrs,
        event_base_val_dtype=pandas_event._get_source_type(config, one_event),
        impute_val_with_time=one_event.impute_val,
        impute_val_no_time=0,
    )


def _take(series: pd.Series, rows: np.ndarray):
    """Values at the given positions, missing where the position is -1, upcast the way merge_asof does."""
    values = series.array if isinstance(series.dtype, pd.api.extensions.ExtensionDtype) else series.to_numpy()
    return pd.api.extensions.take(values, rows, allow_fill=True)


def _to_polars(frame: pd.DataFrame, columns: list[str]) -> "pl.DataFrame":
    """Converts columns to polars, with string-like columns as polars strings so both frames join on one dtype."""
    out = pl.from_pandas(frame[columns])
    strings = [col for col in columns if pdh._key_kind(frame[col]) == "string"]
    return out.with_columns(pl.col(strings).cast(pl.String)) if strings else out


def _unsupported_reason(config: ConfigProvider, event_frame: pd.DataFrame, dataframe: pd.DataFrame) -> Optional[str]:
    """Why the polars merge can't reproduce the pandas merge for this data, or None if it can."""
    if other := sorted({ev.merge_strategy for ev in config.events.values()} - {"forward"}):
        return f"merge strategies {other} need it"
    keys = config.entity_keys
    if any(key not in dataframe or key not in event_frame for key in keys) or "Value" not in event_frame:
        return "not every key is in both predictions and events"
    if not (
        pd.api.types.is_datetime64_dtype(dataframe[config.predict_time])
        and pd.api.types.is_datetime64_dtype(event_frame["Time"])
    ):
        return "prediction or event times are not timezone-naive datetimes"
    if any(pdh._key_kind(dataframe[key]) != pdh._key_kind(event_frame[key]) for key in keys):
        return "key dtypes differ between predictions and events"
    if dataframe[keys].isna().any().any() or event_frame[keys].isna().any().any():
        return "some keys are missing"  # pandas matches missing keys to each other; polars never does

    counts = event_frame.groupby("Type", observed=True)["Time"].agg(["size", "count"])
    for ev in config.events.values():
        if pdh.event_value(ev.display_name) in dataframe or pdh.event_time(ev.display_name) in dataframe:
            return f"predictions already have columns for {ev.display_name}"
        types = counts.index.intersection(list(ev.source) + [ev.display_name])
        if counts.loc[types, "size"].sum() > 0 and counts.loc[types, "count"].sum() == 0:
            return f"{ev.display_name} has no event times"
    return None
