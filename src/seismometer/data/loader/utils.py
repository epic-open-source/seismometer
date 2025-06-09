from pathlib import Path

from seismometer.configuration import ConfigProvider


def gather_prediction_types(config: ConfigProvider) -> dict[str, str]:
    """
    Gathers the defined types from the configuration dictionary.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object.

    Returns
    -------
    dict[str, str]
        the type dictionary with the structure dict[column] = type
    """
    return {defn.name: defn.dtype for defn in config.prediction_defs.predictions if defn.dtype is not None}


def get_loader_from_path(loaders: dict[str, callable], path: Path, default: callable) -> callable:
    file_extension = get_file_extension(path)
    if file_extension in loaders:
        return loaders[file_extension]
    return default


def get_file_extension(path: Path) -> str:
    """
    Gets the file extension from a path in lowercase, e.g. "predictions.parquet" => ".parquet"
    """
    return path.suffix.lower()
