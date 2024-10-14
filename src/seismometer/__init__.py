# flake8: noqa: F403, F405 -- allow * from api
import importlib.metadata

# typing
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# API
from seismometer.api import *

__version__ = importlib.metadata.version("seismometer")
logger = init_logger()


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
    """
    logger.setLevel(log_level)
    logger.info(f"seismometer version {__version__} starting")

    if reset:
        Seismogram.kill()

    config = config_provider or ConfigProvider(config_path, output_path=output_path, definitions=definitions)
    loader = loader_factory(config)
    sg = Seismogram(config, loader)
    sg.load_data(predictions=predictions_frame, events=events_frame)

    # Surface api into namespace
    s_module = importlib.import_module("seismometer._api")
    globals().update(vars(s_module))


def download_example_dataset(dataset_name: str, branch_name: str = "main"):  # pragma: no cover
    """
    Downloads an example dataset from the specified branch to local data/ directory.

    Parameters
    ----------
        dataset_name : str
            The name of the dataset to download.
        branch_name : str, optional
            The branch from which to download the dataset. Defaults to "main".

    Raises
    ------
        ValueError
            If the specified dataset is not available in the example datasets.
    """

    # This function does not depend on the seismometer initialization so singleton and loggers are not available.

    import urllib.request
    from collections import namedtuple
    from pathlib import Path

    DatasetItem = namedtuple("DatasetItem", ["source", "destination"])

    datasets = {}

    datasets["diabetes"] = [
        DatasetItem("data_dictionary.yml", "data_dictionary.yml"),
        DatasetItem("predictions.parquet", "data/predictions.parquet"),
        DatasetItem("events.parquet", "data/events.parquet"),
    ]

    datasets["diabetes-v2"] = [
        DatasetItem("config.yml", "config.yml"),
        DatasetItem("usage_config.yml", "usage_config.yml"),
        DatasetItem("data_dictionary.yml", "data_dictionary.yml"),
        DatasetItem("data/predictions.parquet", "data/predictions.parquet"),
        DatasetItem("data/events.parquet", "data/events.parquet"),
        DatasetItem("data/metadata.json", "data/metadata.json"),
    ]

    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} is not available in the example datasets.")

    SOURCE_REPO = "epic-open-source/seismometer-data"
    DATASET_SOURCE = f"https://raw.githubusercontent.com/{SOURCE_REPO}/refs/heads/{branch_name}/{dataset_name}"

    Path("data").mkdir(parents=True, exist_ok=True)
    for item in datasets[dataset_name]:
        try:
            _ = urllib.request.urlretrieve(f"{DATASET_SOURCE}/{item.source}", item.destination)
        except urllib.error.ContentTooShortError:
            print(f"Failed to download {item.source} from {DATASET_SOURCE}")
