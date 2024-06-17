import logging
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from seismometer.configuration import ConfigProvider
from seismometer.data import pandas_helpers as pdh

logger = logging.getLogger("seismometer")


def seismogram_loader_factory(config: ConfigProvider):
    return SeismogramParquetLoader(config)


class SeismogramParquetLoader:
    def __init__(self, config, predictions=None, events=None):
        self.config = config
        # placeholders
        self.prediction_loader = predictions
        self.event_loader = events

        # validation
        self.config.bwd = "loaded"

    def load(self, predictions=None, events=None) -> pd.DataFrame:
        """
        Xlogger.info(f"Importing files from {self.config.config_dir}")

        Xself._load_predictions()
        self._load_events()

        self._time_to_ns()

        Xself._cohorts = self.config.cohorts #-in load_config
        ‼self.prep_data()
        """
        logger.info(f"Importing files from {self.config.config_dir}")

        dataframe = self.load_predictions()
        dataframe = self.load_events(dataframe)

        return dataframe

    def load_predictions(self) -> pd.DataFrame:
        dataframe = self._load_predictions()
        return self._prediction_post_load(dataframe)

    def load_events(self, dataframe) -> pd.DataFrame:
        events = self._load_events()
        events = self._event_post_load(events)
        return self.merge_events(dataframe, events)

    def _load_predictions(self) -> pd.DataFrame:
        """
        Loads the predictions data, restricting features based on config (if any).
        """
        if self.config.features:  # no features == all features
            desired_columns = set(self.config.prediction_columns)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                present_columns = set(
                    pq.ParquetDataset(self.config.prediction_path, use_legacy_dataset=False).schema.names
                )

            if self.config.target in present_columns:
                desired_columns.add(self.config.target)

            actual_columns = desired_columns & present_columns
            if len(desired_columns) != len(actual_columns):
                logger.warning(
                    "Not all requested columns are present. "
                    + f"Missing columns are {', '.join(desired_columns-present_columns)}"
                )
                logger.debug(f"Requested columns are {', '.join(desired_columns)}")
                logger.debug(f"Columns present are {', '.join(present_columns)}")
            dataframe = pd.read_parquet(self.config.prediction_path, columns=actual_columns)
        else:
            dataframe = pd.read_parquet(self.config.prediction_path)

        if self.config.target in dataframe:
            logger.debug(
                f"Using existing column in predictions dataframe as target: {self.config.target} -> {self.target}"
            )
            dataframe = dataframe.rename({self.config.target: self.target}, axis=1)

        return dataframe

    def _load_events(self) -> pd.DataFrame:
        """
        Loads the events data if any exists, otherwise stands up an empty df with the expected columns.
        """
        try:
            events = pd.read_parquet(self.config.event_path).rename(
                columns={self.config.ev_type: "Type", self.config.ev_time: "Time", self.config.ev_value: "Value"},
                copy=False,
            )
        except BaseException:
            logger.debug(f"No events found at {self.config.event_path}")
            events = pd.DataFrame(columns=self.config.entity_keys + ["Type", "Time", "Value"])

        return events

    def _prediction_post_load(self, dataframe) -> pd.DataFrame:
        # datetime precisions don't play nicely - fix to pands default
        pred_times = dataframe.select_dtypes(include="datetime").columns
        dataframe = self._infer_datetime(dataframe)
        dataframe[pred_times] = dataframe[pred_times].astype({col: "<M8[ns]" for col in pred_times})

        # Expand this to robust score prep
        for score in self.config.output_list:
            if score not in dataframe:
                continue
            if 50 < dataframe[score].max() <= 100:  # Assume out of 100, readjust
                dataframe[score] /= 100

        # Need to remove pd.FloatXxDtype as sklearn and numpy get confused
        float_cols = dataframe.select_dtypes(include=[float]).columns
        dataframe[float_cols] = dataframe[float_cols].astype(np.float32)

        return dataframe

    def _event_post_load(self, events) -> pd.DataFrame:
        # Time column in events is known
        events["Time"] = events["Time"].astype("<M8[ns]")

        return events

    def merge_events(self, dataframe, event_frame) -> pd.DataFrame:
        """Merges a value and time column into dataframe for each configured event."""
        # -> should specify a specific outcome
        dataframe = (
            dataframe.sort_values(self.config.predict_time)
            .drop_duplicates(subset=self.config.entity_keys + [self.config.predict_time])
            .dropna(subset=[self.config.predict_time])
        )
        event_frame = event_frame.sort_values("Time")

        for one_event in self.config.events:
            # Merge
            if one_event.window_hr:
                logger.debug(
                    f"Windowing event {one_event.display_name} to lookback {one_event.window_hr} "
                    + f"offset by {one_event.offset_hr}"
                )
                dataframe = self._merge_event(
                    one_event.source,
                    dataframe,
                    event_frame,
                    window_hrs=one_event.window_hr,
                    offset_hrs=one_event.offset_hr,
                    display=one_event.display_name,
                    sort=False,
                )
                self.config.target_cols.append(one_event.display_name)
            else:  # No lookback
                logger.debug(f"Merging event {one_event.display_name}")
                dataframe = self._merge_event(
                    one_event.source, dataframe, event_frame, display=one_event.display_name, sort=False
                )

            # Impute
            if one_event.impute_val or one_event.usage == "target":
                event_val = pdh.event_value(one_event.display_name)
                impute = one_event.impute_val or 0  # Enforce binary type target
                dataframe[event_val] = dataframe[event_val].fillna(impute)

        return dataframe

    def _merge_event(
        self, event_col, dataframe, event_frame, offset_hrs=0, window_hrs=None, display="", sort=True
    ) -> pd.DataFrame:
        disp_event = display if display else event_col
        translate_event = event_col if display else ""

        return pdh.merge_windowed_event(
            dataframe,
            self.config.predict_time,
            event_frame.replace({"Type": translate_event}, disp_event),
            disp_event,
            self.config.entity_keys,
            min_leadtime_hrs=offset_hrs,
            window_hrs=window_hrs,
            event_base_time_col="Time",
            sort=sort,
        )

    @staticmethod
    def _infer_datetime(df, cols=None, override_categories=None):
        # override_categories - allow configured dtypes to force decision
        if cols is None:
            cols = df.columns
        for col in cols:
            if "Time" in col:
                df[col] = pd.to_datetime(df[col])
                continue
        return df
