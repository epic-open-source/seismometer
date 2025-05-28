import logging
import os

import seismometer.data.loader.event as event
import seismometer.data.loader.prediction as prediction
from seismometer.configuration import ConfigProvider

from .pipeline import ConfigFrameHook, ConfigOnlyHook, MergeFramesHook, SeismogramLoader


"""
Dictionary of data loaders of the form prediction_loaders[file_extension] = loader.
"""
prediction_loaders = {
    ".parquet": prediction.parquet_loader,
    ".csv": prediction.csv_loader
}

"""
Dictionary of data loaders of the form event_loaders[file_extension] = loader.
"""
event_loaders = {
    ".parquet": event.parquet_loader,
    ".csv": event.csv_loader
}

def loader_factory(config: ConfigProvider, post_load_fn: ConfigFrameHook = None) -> SeismogramLoader:
    """
    Construct a SeismogramLoader from the provided configuration.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object
    post_load_fn : ConfigFrameHook, optional
        A callable taking a ConfigProvider and the fully loaded dataframe and returning a dataframe.
        Used to allow any custom manipulations of the Seismogram dataframe after all other load steps complete.
        WARNING: This can completly overwrite/discard the daframe that was loaded.

    Returns
    -------
    SeismogramLoader
        Instance of the SeismogramLoader class, with load_data() method ready to be called
    """
    
    # Default filetype: parquet
    prediction_loader: ConfigOnlyHook = prediction.parquet_loader
    event_loader: ConfigOnlyHook = event.parquet_loader

    # config should be able to be any object without immediately giving
    # an error
    if isinstance(config, ConfigProvider):
        prediction_file_extension = _get_file_extension(config.prediction_path)
        if prediction_file_extension in prediction_loaders: prediction_loader = prediction_loaders[prediction_file_extension]

        event_file_extension = _get_file_extension(config.event_path)
        if event_file_extension in event_loaders: event_loader = event_loaders[event_file_extension]

    return SeismogramLoader(
        config,
        prediction_fn=prediction_loader,
        post_predict_fn=prediction.dictionary_types,
        event_fn=event_loader,
        post_event_fn=event.post_transform_fn,
        merge_fn=event.merge_onto_predictions,
        post_load_fn=post_load_fn,
    )

# helper functions

def _get_file_extension(path: str) -> str:
    """
    Gets the file extension from a path in lowercase, e.g. "predictions.parquet" => ".parquet"
    """
    _, extension = os.path.splitext(path)
    return extension.lower()
