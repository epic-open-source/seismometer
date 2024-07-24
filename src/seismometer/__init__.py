import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from seismometer._version import __version__
from seismometer.core.logger import add_log_formatter, set_default_logger_config


def run_startup(
    *,
    config_path: str | Path = None,
    output_path: str | Path = None,
    predictions_frame: Optional[pd.DataFrame] = None,
    events_frame: Optional[pd.DataFrame] = None,
    definitions: Optional[dict] = None,
    log_level: int = logging.WARN,
    reset: bool = False,
):
    """
    Runs the required startup for instantiating seismometer.

    Parameters
    ----------
    config_path : Optional[str | Path], optional
        The path containing the config.yml and other resources, by default None.
    output_path : Optional[str | Path], optional
        An output path to write data to, overwriting the default path specified by info_dir in config.yml,
        by default None.
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
    """
    import importlib

    from seismometer.configuration import ConfigProvider
    from seismometer.data.loader import loader_factory
    from seismometer.seismogram import Seismogram

    set_default_logger_config()

    logger = logging.getLogger("seismometer")
    add_log_formatter(logger)

    logger.setLevel(log_level)
    logger.info(f"seismometer version {__version__} starting")

    if reset:
        Seismogram.kill()

    config = ConfigProvider(config_path, output_path, definitions=definitions)
    loader = loader_factory(config)
    sg = Seismogram(config, loader)
    sg.load_data(predictions=predictions_frame, events=events_frame)

    # Surface api into namespace
    s_module = importlib.import_module("seismometer._api")
    globals().update(vars(s_module))
