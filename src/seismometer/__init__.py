import logging
from pathlib import Path

from seismometer._version import __version__
from seismometer.core.logger import add_log_formatter, set_default_logger_config


def run_startup(*, config_path: str | Path = None, output_path: str | Path = None, log_level: int = logging.WARN):
    """
    Runs the required startup for instantiating seismometer.

    Parameters
    ----------
    config_path : Optional[str | Path], optional
        The path containing the config.yml and other resources, by default None.
    output_path : Optional[str | Path], optional
        An output path to write data to, overwriting the default path specified by info_dir in config.yml,
        by default None.
    log_level : logging._Level, optional
        The log level to set. by default, logging.WARN.
    """
    import importlib

    from seismometer.seismogram import Seismogram

    set_default_logger_config()

    logger = logging.getLogger("seismometer")
    add_log_formatter(logger)

    logger.setLevel(log_level)
    logger.info(f"seismometer version {__version__} starting")

    sg = Seismogram(config_path, output_path)
    sg.load()

    # Surface api into namespace
    s_module = importlib.import_module("seismometer._api")
    globals().update(vars(s_module))
