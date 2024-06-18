import logging
from typing import Callable

import pandas as pd

from seismometer.configuration import ConfigProvider

logger = logging.getLogger("seismometer")

# types
ConfigOnlyHook = Callable[[ConfigProvider], pd.DataFrame]
ConfigFrameHook = Callable[[ConfigProvider, pd.DataFrame], pd.DataFrame]
MergeHook = Callable[[ConfigProvider, pd.DataFrame, pd.DataFrame], pd.DataFrame]


def passthru_frame(*args, dataframe) -> pd.DataFrame:
    return dataframe


class SeismogramLoader:
    def __init__(
        self,
        config: ConfigProvider,
        prediction_fn: ConfigOnlyHook,
        event_fn: ConfigOnlyHook = None,
        post_predict_fn: ConfigFrameHook = None,
        post_event_fn: ConfigFrameHook = None,
        merge_fn: MergeHook = None,
    ):
        self.config = config

        self.prediction_fn = prediction_fn
        self.post_predict_fn = post_predict_fn or passthru_frame

        self.event_fn = event_fn
        self.post_event_fn = post_event_fn or passthru_frame
        self.merge_fn = merge_fn or passthru_frame

        self.prediction_from_memory: ConfigFrameHook = passthru_frame
        self.event_from_memory: ConfigFrameHook = passthru_frame

    def load_data(self, prediction_obj=None, event_obj=None):
        logger.info(f"Importing files from {self.config.config_dir}")

        dataframe = self._load_predictions(prediction_obj)
        dataframe = self.post_predict_fn(self.config, dataframe)
        dataframe = self._add_events(dataframe, event_obj)

        return dataframe

    def _load_predictions(self, prediction_obj=None):
        if prediction_obj is None:
            return self.prediction_fn(self.config)
        return self.prediction_from_memory(self.config, prediction_obj)

    def _add_events(self, dataframe: pd.DataFrame, event_obj=None):
        event_frame = self._load_events(event_obj)
        if event_frame.empty:  # No events to add
            return dataframe

        event_frame = self.post_event_fn(self.config, event_frame)
        return self.merge_fn(self.config, event_frame, dataframe)

    def _load_events(self, event_obj=None) -> pd.DataFrame:
        if (event_obj is None) and (self.event_fn is None):
            return pd.DataFrame()
        if event_obj is None:
            return self.event_fn(self.config)
        return self.event_from_memory(self.config, event_obj)


__all__ = ["SeismogramLoader", "ConfigOnlyHook", "ConfigFrameHook", "MergeHook"]
