import logging
from typing import Callable, Optional, TypeAlias

import pandas as pd

from seismometer.configuration import ConfigProvider

logger = logging.getLogger("seismometer")

ConfigOnlyHook: TypeAlias = Callable[[ConfigProvider], pd.DataFrame]
ConfigOnlyHook.__doc__ = """
TypeAlias for a callable taking a ConfigProvider, which returns a DataFrame.

An example is loading the predictions from a parquet file, where the ConfigProvider
provides the path to the file which is loaded as a dataframe.
"""

ConfigFrameHook: TypeAlias = Callable[[ConfigProvider, pd.DataFrame], pd.DataFrame]
ConfigFrameHook.__doc__ = """
TypeAlias for a callable taking a ConfigProvider and a DataFrame, which returns a DataFrame.

An example is ensuring types of a loaded predictions dataframe, where ConfigProvider
provides some metadata that can then be applied to the dataframe returing the transformed object.
"""


MergeFramesHook: TypeAlias = Callable[[ConfigProvider, pd.DataFrame, pd.DataFrame], pd.DataFrame]
MergeFramesHook.__doc__ = """
TypeAlias for a callable taking a ConfigProvider and two DataFrames, which returns a DataFrame.

An example is uses ConfigProvider to understand the relevant keys, then
merging events from the first dataframe argument onto the last argument (predictions)
returning the combined dataframe.
"""


def _passthru_framehook(config: ConfigProvider, dataframe: pd.DataFrame) -> pd.DataFrame:
    """A generic pass-through used for hooks that don't need to modify the dataframe."""
    return dataframe


def _passthru_mergehook(config: ConfigProvider, eventframe: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
    """A generic pass-through used for hooks that don't need to modify the dataframe."""
    return dataframe


class SeismogramLoader:
    """
    A data loading pipeline using three types of hooks:

    * load predictions [ConfigOnlyHook]
    * transform (type) the loaded predictions [ConfigFrameHook]
    * load events [ConfigOnlyHook]
    * transform (type) the loaded events [ConfigFrameHook]
    * merge events onto predictions [MergeFramesHook]
    * post load manipulations [ConfigFrameHook]

    Each step is expected to return a dataframe, chaining the steps to get the frame driving a loaded Seismogram.
    """

    def __init__(
        self,
        config: ConfigProvider,
        prediction_fn: Optional[ConfigOnlyHook] = None,
        event_fn: Optional[ConfigOnlyHook] = None,
        post_predict_fn: Optional[ConfigFrameHook] = None,
        post_event_fn: Optional[ConfigFrameHook] = None,
        merge_fn: Optional[MergeFramesHook] = None,
        post_load_fn: Optional[ConfigFrameHook] = None,
    ):
        """
        Initialize a data loading pipeline of functions returning a dataframe for a Seismogram session.

        Parameters
        ----------
        config : ConfigProvider
            The loaded configuration object.
        prediction_fn : ConfigOnlyHook
            A callable taking a ConfigProvider and returning a dataframe.
            Used to load a (predictions) dataframe based on configuration;
            skipped if prediction_obj is provided to load_data.
        event_fn : ConfigOnlyHook, optional
            A callable taking a ConfigProvider and returning a dataframe.
            Used to load a (events) dataframe based on configuration; skipped if event_obj is provided to load_data.
        post_predict_fn : ConfigFrameHook, optional
            A callable taking a ConfigProvider and a (predictions) dataframe and returning a dataframe.
            Used to do minor transforms of predictions such as type casting.
        post_event_fn : ConfigFrameHook, optional
            A callable taking a ConfigProvider and a (events) dataframe and returning a dataframe.
            Used to do minor transforms of events such as type casting.
        merge_fn : MergeFramesHook, optional
            A callable taking a ConfigProvider, a (events) dataframe, and a (predictions) dataframe
            and returning a dataframe.
            Used to merge events onto predictions based on configuration.
        post_load_fn : ConfigFrameHook, optional
            A callable taking a ConfigProvider and the fully loaded dataframe and returning a dataframe.
            Used to allow any custom manipulations of the Seismogram dataframe during load.
            WARNING: This can completly overwrite/discard the daframe that was loaded.
        """
        self.config = config

        self.prediction_fn = prediction_fn
        self.post_predict_fn = post_predict_fn or _passthru_framehook

        self.event_fn = event_fn
        self.post_event_fn = post_event_fn or _passthru_framehook
        self.merge_fn = merge_fn or _passthru_mergehook

        self.prediction_from_memory: ConfigFrameHook = _passthru_framehook
        self.event_from_memory: ConfigFrameHook = _passthru_framehook

        # Hooks for custom transformations
        self.post_load_fn: ConfigFrameHook = post_load_fn or _passthru_framehook

    def load_data(self, prediction_obj: pd.DataFrame = None, event_obj: pd.DataFrame = None) -> pd.DataFrame:
        """
        Entry point for loading data for a Seismogram session.

        Can optionally accept a prediction and event frame that are prioritized over loading from configuration.

        Parameters
        ----------
        prediction_obj : pd.DataFrame, optional
            an optional pre-loaded predictions dataframe, if None then will load from configuration, by default None.
        event_obj : pd.DataFrame, optional
            an optional pre-loaded events dataframe, if None then will load from configuration, by default None.

        Returns
        -------
        pd.DataFrame
            the loaded and merged dataframe for a Seismogram session.
        """

        logger.info(f"Configuration speficies path {self.config.config_dir}")

        dataframe = self._load_predictions(prediction_obj)
        dataframe = self.post_predict_fn(self.config, dataframe)
        dataframe = self._add_events(dataframe, event_obj)

        dataframe = self.post_load_fn(self.config, dataframe)
        return dataframe

    def _load_predictions(self, prediction_obj: pd.DataFrame = None):
        """Load predictions from configuration if not passed in directly."""
        if (prediction_obj is None) and (self.prediction_fn is None):
            raise RuntimeError(
                "No prediction_fn provided and no prediction_obj provided. A prediction frame must be provided."
            )
        if prediction_obj is None:
            return self.prediction_fn(self.config)
        return self.prediction_from_memory(self.config, prediction_obj)

    def _add_events(self, dataframe: pd.DataFrame, event_obj: pd.DataFrame = None):
        """
        Load and merge events onto the predictions dataframe.
        Prioritizes event_obj if provided, else loads from configuration.
        """
        event_frame = self._load_events(event_obj)
        if event_frame.empty:  # No events to add
            logger.debug("No events were loaded; nothing added to frame.")
            return dataframe

        event_frame = self.post_event_fn(self.config, event_frame)
        return self.merge_fn(self.config, event_frame, dataframe)

    def _load_events(self, event_obj: pd.DataFrame = None) -> pd.DataFrame:
        """Load events from config or memory."""
        if (event_obj is None) and (self.event_fn is None):
            return pd.DataFrame()
        if event_obj is None:
            return self.event_fn(self.config)
        return self.event_from_memory(self.config, event_obj)


__all__ = ["SeismogramLoader", "ConfigOnlyHook", "ConfigFrameHook", "MergeFramesHook"]
