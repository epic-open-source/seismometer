import logging
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import seismometer.data.pandas_helpers as pdh
from seismometer.configuration import ConfigProvider, ConfigurationError

logger = logging.getLogger("seismometer")


def parquet_loader(config: ConfigProvider) -> pd.DataFrame:
    """
    Load the predictions frame from a parquet file based on config.prediction_path.

    Will restrict the loaded columns to those specified in config.features, if present.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.

    Returns
    -------
    pd.DataFrame
        The predictions dataframe.
    """
    if not config.features:  # no features ==> all features
        dataframe = pd.read_parquet(config.prediction_path)
    else:
        desired_columns = set(config.prediction_columns)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            present_columns = set(pq.ParquetDataset(config.prediction_path, use_legacy_dataset=False).schema.names)

        if config.target in present_columns:
            desired_columns.add(config.target)

        actual_columns = desired_columns & present_columns
        _log_column_mismatch(actual_columns, desired_columns, present_columns)

        dataframe = pd.read_parquet(config.prediction_path, columns=actual_columns)

    dataframe = _rename_targets(config, dataframe)

    return dataframe.sort_index(axis=1)  # parquet can shuffle column order


def _log_column_mismatch(actual_columns: list[str], desired_columns: list[str], present_columns: list[str]) -> None:
    """Logs warnings if the actual columns and desired columns are a mismatch."""
    if len(actual_columns) == len(desired_columns):
        return

    logger.warning(
        "Not all requested columns are present. " + f"Missing columns are {', '.join(desired_columns-present_columns)}"
    )
    logger.debug(f"Requested columns are {', '.join(desired_columns)}")
    logger.debug(f"Columns present are {', '.join(present_columns)}")


def _rename_targets(config: ConfigProvider, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Renames the target column if already in the dataframe, to match what a event merge would produce."""
    if config.target in dataframe:
        target_value = pdh.event_value(config.target)
        logger.debug(f"Using existing column in predictions dataframe as target: {config.target} -> {target_value}")
        dataframe = dataframe.rename({config.target: target_value}, axis=1)
    return dataframe


# post loaders
def dictionary_types(config: ConfigProvider, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the loaded predictions dataframe to the expected types.

    Prioritizes the types defined in the configuration dictionary, then falls back to assumed_types.


    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.
    dataframe : pd.DataFrame
        The loaded predictions dataframe.

    Returns
    -------
    pd.DataFrame
        The predictions dataframe with adjusted types.

    Raises
    ------
    ConfigurationError
        If any columns cannot be converted to the expected types.
    """

    defined_types = _gather_defined_types(config)
    unspecified_columns = []
    value_error_columns = []

    for col in dataframe.columns:
        if col in defined_types:
            try:
                pdh.try_casting(dataframe, col, defined_types[col])
            except ConfigurationError:
                value_error_columns.append(col)
        else:
            unspecified_columns.append(col)

    if len(value_error_columns) > 0:
        raise ConfigurationError(
            f"Could not convert columns to expected types: {', '.join(value_error_columns)}. "
            + "Update dictionary config or contact the model owner columns."
        )

    # Default to assumed types
    dataframe[unspecified_columns] = assumed_types(config, dataframe[unspecified_columns])
    return dataframe


def assumed_types(config: ConfigProvider, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the loaded predictions dataframe to the expected types.

    Scope is currently restricted to time and output columns as parquet is expected to include datatypes.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.
    dataframe : pd.DataFrame
        The loaded predictions dataframe.

    Returns
    -------
    pd.DataFrame
        The predictions dataframe with adjusted types.
    """
    dataframe = _infer_datetime(dataframe)

    # datetime precisions don't play nicely - fix to pands default
    pred_times = dataframe.select_dtypes(include="datetime").columns
    dataframe[pred_times] = dataframe[pred_times].astype({col: "<M8[ns]" for col in pred_times})

    # Expand this to robust score prep
    for score in config.output_list:
        if score not in dataframe:
            continue
        if 25 < dataframe[score].max() <= 100:  # Assume out of 100, readjust
            dataframe[score] /= 100

    # Need to remove pd.FloatXxDtype as sklearn and numpy get confused
    float_cols = dataframe.select_dtypes(include=[float]).columns
    dataframe[float_cols] = dataframe[float_cols].astype(np.float64)

    return dataframe


# other
def _gather_defined_types(config: ConfigProvider) -> dict[str, str]:
    """Gathers the defined types from the configuration dictionary."""
    return {defn.name: defn.dtype for defn in config.prediction_defs.predictions if defn.dtype is not None}


def _infer_datetime(dataframe, cols=None, override_categories=None):
    """Infers datetime columns based on column name and casts to pandas.datatime."""
    if cols is None:
        cols = dataframe.columns
    for col in cols:
        if "Time" in col:
            dataframe[col] = pd.to_datetime(dataframe[col])
            continue
    return dataframe
