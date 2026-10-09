# flake8: noqa: F403, F405 -- allow * from api
import importlib.metadata

# typing
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from seismometer.core.autometrics import initialize_otel_config

# isort: split
from seismometer.api import *

__version__ = importlib.metadata.version("seismometer")
logger = logging.getLogger("seismometer")


def run_startup(
    *,
    config_path: str | Path = None,
    output_path: str | Path = None,
    config_provider: Optional[ConfigProvider] = None,
    predictions_frame: Optional[pd.DataFrame] = None,
    events_frame: Optional[pd.DataFrame] = None,
    definitions: Optional[dict] = None,
    log_level: int = logging.WARN,
    reset: bool = False,
    memory_efficient: bool = False,
):
    """
    Runs the required startup for instantiating seismometer.

    Parameters
    ----------
    config_path : Optional[str | Path], optional
        The path containing the config.yml and other resources, by default None.
        Optional if configProvider is provided.
    output_path : Optional[str | Path], optional
        An output path to write data to, overwriting the default path specified by info_dir in config.yml,
        by default None.
    config_provider : Optional[ConfigProvider], optional
        An optional ConfigProvider instance to use instead of loading configuration from config_path, by default None.
    predictions_frame : Optional[pd.DataFrame], optional
        An optional DataFrame containing the fully loaded predictions data, by default None.
        By default, when not specified here, these data will be loaded based on conifguration.
    events_frame : Optional[pd.DataFrame], optional
        An optional DataFrame containing the fully loaded events data, by default None.
        By default, when not specified here, these data will be loaded based on conifguration.
    definitions : Optional[dict], optional
        A dictionary of definitions to use instead of loading those specified by configuration, by default None.
        By default, when not specified here, these data will be loaded based on conifguration.
    log_level : logging._Level, optional
        The log level to set. by default, logging.WARN.
    reset : bool, optional
        A flag when True, will reset the Seismogram instance before loading configuration and data, by default False.
    memory_efficient : bool, optional
        A flag when True, lowers the memory and time to load data, by default False. String columns stay
        arrow-backed even where the data dictionary says ``object``, and if polars is installed
        (``pip install seismometer[perf]``) events are merged with it. The loaded values are the same.
    """

    _ = init_logger()
    logger.setLevel(log_level)
    logger.info(f"seismometer version {__version__} starting")

    if reset:
        Seismogram.kill()

    config = config_provider or ConfigProvider(config_path, output_path=output_path, definitions=definitions)
    config.memory_efficient = memory_efficient
    loader = loader_factory(config)
    if memory_efficient:
        from seismometer.data.loader import polars_merge  # only import polars when asked to

        if polars_merge.available():
            loader.merge_fn = polars_merge.merge_onto_predictions
        else:
            logger.info("memory_efficient: polars is not installed; merging events with pandas.")
    sg = Seismogram(config, loader)

    sg.load_data(predictions=predictions_frame, events=events_frame)

    initialize_otel_config(config)
