import logging

from seismometer.core.logger import logger


def test_logger_is_the_package_logger():
    assert logger is logging.getLogger("seismometer")


def test_modules_share_the_package_logger():
    from seismometer.api import plots
    from seismometer.data import pandas_helpers

    assert plots.logger is logger
    assert pandas_helpers.logger is logger
