import seismometer.data.loader.event as event
import seismometer.data.loader.prediction as prediction
from seismometer.configuration import ConfigProvider

from .pipeline import ConfigFrameHook, ConfigOnlyHook, MergeHook, SeismogramLoader


def loader_factory(config: ConfigProvider) -> SeismogramLoader:
    """
    Construct a SeismogramLoader from the provided configuration.

    Parameters
    ----------
    config : ConfigProvider
        The loaded configuration object

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
        post_predict_fn=prediction.assumed_types,
        event_fn=event_loader,
        post_event_fn=event.post_transform_fn,
        merge_fn=event.merge_onto_predictions,
    )
