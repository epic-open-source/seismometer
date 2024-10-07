import seismometer.data.loader.event as event
import seismometer.data.loader.prediction as prediction
from seismometer.configuration import ConfigProvider

from .pipeline import ConfigFrameHook, ConfigOnlyHook, MergeFramesHook, SeismogramLoader


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
    prediction_loader: ConfigOnlyHook = prediction.parquet_loader
    event_loader: ConfigOnlyHook = event.parquet_loader

    return SeismogramLoader(
        config,
        prediction_fn=prediction_loader,
        post_predict_fn=prediction.dictionary_types,
        event_fn=event_loader,
        post_event_fn=event.post_transform_fn,
        merge_fn=event.merge_onto_predictions,
        post_load_fn=post_load_fn,
    )
