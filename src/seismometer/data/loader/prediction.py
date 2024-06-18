import logging
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import seismometer.data.pandas_helpers as pdh
from seismometer.configuration import ConfigProvider

logger = logging.getLogger("seismometer")


def parquet_loader(config: ConfigProvider) -> pd.DataFrame:
    if config.features:  # no features == all features
        desired_columns = set(config.prediction_columns)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            present_columns = set(pq.ParquetDataset(config.prediction_path, use_legacy_dataset=False).schema.names)

        if config.target in present_columns:
            desired_columns.add(config.target)

        actual_columns = desired_columns & present_columns
        if len(desired_columns) != len(actual_columns):
            logger.warning(
                "Not all requested columns are present. "
                + f"Missing columns are {', '.join(desired_columns-present_columns)}"
            )
            logger.debug(f"Requested columns are {', '.join(desired_columns)}")
            logger.debug(f"Columns present are {', '.join(present_columns)}")
        dataframe = pd.read_parquet(config.prediction_path, columns=actual_columns)
    else:
        dataframe = pd.read_parquet(config.prediction_path)

    if config.target in dataframe:
        target_value = pdh.event_value(config.target)
        logger.debug(f"Using existing column in predictions dataframe as target: {config.target} -> {target_value}")
        dataframe = dataframe.rename({config.target: target_value}, axis=1)

    return dataframe


# post loaders
def assumed_types(config: ConfigProvider, dataframe: pd.DataFrame) -> pd.DataFrame:
    # datetime precisions don't play nicely - fix to pands default
    pred_times = dataframe.select_dtypes(include="datetime").columns
    dataframe = _infer_datetime(dataframe)
    dataframe[pred_times] = dataframe[pred_times].astype({col: "<M8[ns]" for col in pred_times})

    # Expand this to robust score prep
    for score in config.output_list:
        if score not in dataframe:
            continue
        if 50 < dataframe[score].max() <= 100:  # Assume out of 100, readjust
            dataframe[score] /= 100

    # Need to remove pd.FloatXxDtype as sklearn and numpy get confused
    float_cols = dataframe.select_dtypes(include=[float]).columns
    dataframe[float_cols] = dataframe[float_cols].astype(np.float32)

    return dataframe


# other
def _infer_datetime(dataframe, cols=None, override_categories=None):
    # override_categories - allow configured dtypes to force decision
    if cols is None:
        cols = dataframe.columns
    for col in cols:
        if "Time" in col:
            dataframe[col] = pd.to_datetime(dataframe[col])
            continue
    return dataframe
