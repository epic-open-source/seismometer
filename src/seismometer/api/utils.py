import urllib.request
from collections import namedtuple
from pathlib import Path
from typing import Optional

from seismometer.configuration.config_helpers import generate_dictionary_from_parquet


def download_example_dataset(
    dataset_name: str, branch_name: str = "main", source_repo: Optional[str] = None
):  # pragma: no cover
    """
    Downloads an example dataset from the specified branch to local data/ directory.

    Parameters
    ----------
        dataset_name : str
            The name of the dataset to download.
        branch_name : str, optional
            The branch from which to download the dataset. Defaults to "main".
        source_repo: Optional[str], optional
            The GitHub repository that contains the data, by default None.

    Raises
    ------
        ValueError
            If the specified dataset is not available in the example datasets.
    """

    # This function does not depend on the seismometer initialization so singleton and loggers are not available.
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

    SOURCE_REPO = source_repo or "epic-open-source/seismometer-data"
    DATASET_SOURCE = f"https://raw.githubusercontent.com/{SOURCE_REPO}/refs/heads/{branch_name}/{dataset_name}"

    Path("data").mkdir(parents=True, exist_ok=True)
    for item in datasets[dataset_name]:
        try:
            _ = urllib.request.urlretrieve(f"{DATASET_SOURCE}/{item.source}", item.destination)
        except urllib.error.ContentTooShortError:
            print(f"Failed to download {item.source} from {DATASET_SOURCE}")


__all__ = ["download_example_dataset", "generate_dictionary_from_parquet"]
