# This is a utility to automate the running of a seismograph
# and collect metrics from it.
#
# Usage: runmetrics path/to/seismograph

import logging
import sys

import seismometer as sm

logger = logging.Logger("Seismograph run metrics logger")


def main():
    args = sys.argv
    # We expect one argument: the path to the seismograph.
    if len(args) != 2:  # ['runmetrics', path/to/seismograph]
        logger.error("Expected one argument.")
        return
    seismo_path = args[1]
    sm.run_startup(config_path=seismo_path)
    sm.do_metric_exports()
