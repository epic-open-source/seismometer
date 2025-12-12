import csv
import logging
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import seismometer.data.pandas_helpers as pdh
from seismometer.configuration import ConfigProvider, ConfigurationError

from .pipeline import ConfigOnlyHook

logger = logging.getLogger("seismometer")


def get_data_loader(config: ConfigProvider) -> ConfigOnlyHook:
    """
    Returns the proper data loader function from the prediction file extension.

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
    return loaders.get(config.prediction_path.suffix.lower(), parquet_loader)


def csv_loader(config: ConfigProvider) -> pd.DataFrame:
    """
    Load the predictions frame from a CSV file based on config.prediction_path.

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
    return _sv_loader(config, ",")


def tsv_loader(config: ConfigProvider) -> pd.DataFrame:
    """
    Load the predictions frame from a TSV file based on config.prediction_path.

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
    return _sv_loader(config, "\t")


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
    logger.debug(f"Loading predictions from {config.prediction_path}.")
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
        dataframe = pd.read_parquet(config.prediction_path, columns=list(actual_columns))

    dataframe = _rename_targets(config, dataframe)

    logger.debug(f"Loaded {len(dataframe)} predictions from {config.prediction_path}.")
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

    defined_types = config.prediction_types
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


def _infer_datetime(dataframe, cols=None, override_categories=None):
    """Infers datetime columns based on column name and casts to pandas.datetime."""
    dataframe = dataframe.copy()
    if cols is None:
        cols = dataframe.columns
    for col in cols:
        if "Time" in col:
            dataframe[col] = pd.to_datetime(dataframe[col])
    return dataframe


def _sv_loader(config: ConfigProvider, sep) -> pd.DataFrame:
    """General loader for CSV or TSV files"""
    if not config.features:  # no features ==> all features
        dataframe = pd.read_csv(config.prediction_path, sep=sep)
    else:
        desired_columns = set(config.prediction_columns)

        with open(config.prediction_path, "r") as f:
            dict_reader = csv.DictReader(f, delimiter=sep)
            try:
                present_columns = set(dict_reader.fieldnames)
            except UnicodeDecodeError:
                # output clean error
                raise ValueError(
                    f"Unable to parse file {config.prediction_path}. Make sure it is formatted correctly."
                ) from None

        if config.target in present_columns:
            desired_columns.add(config.target)
        actual_columns = desired_columns & present_columns
        _log_column_mismatch(actual_columns, desired_columns, present_columns)
        dataframe = pd.read_csv(config.prediction_path, sep=sep, usecols=list(actual_columns))

    dataframe = _rename_targets(config, dataframe)

    # since importing CSVs automatically cast numbers to ints, make sure the columns
    # shared with events become strings so we don't have a type mismatch
    defined_types = config.prediction_types
    usage = config.usage
    for col in [usage.entity_id, usage.context_id, usage.predict_time]:
        if col is not None and defined_types[col] == "object":
            dataframe[col] = dataframe[col].astype(str)

    return dataframe.sort_index(axis=1)
