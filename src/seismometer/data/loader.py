import logging
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from seismometer.configuration import ConfigProvider
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

        # TODO: time to ns
        # TODO: prep (score handling)

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
        pred_times = self.dataframe.select_dtypes(include="datetime").columns
        self.dataframe[pred_times] = self.dataframe[pred_times].astype({col: "<M8[ns]" for col in pred_times})

        # TODO event loader
        # Time column in events is known
        self._events["Time"] = self._events["Time"].astype("<M8[ns]")

    def prep_data(self):
        """Preprocesses the prediction data for use in analysis."""
        # TODO prediction loader
        # Do not infer events
        self.dataframe = self._infer_datetime(self.dataframe)

        # Expand this to robust score prep
        for score in self.output_list:
            if score not in self.dataframe:
                continue
            if 50 < self.dataframe[score].max() <= 100:  # Assume out of 100, readjust
                self.dataframe[score] /= 100

        # Need to remove pd.FloatXxDtype as sklearn and numpy get confused
        float_cols = self.dataframe.select_dtypes(include=[float]).columns
        self.dataframe[float_cols] = self.dataframe[float_cols].astype(np.float32)

        return dataframe

    def _event_post_load(self, events) -> pd.DataFrame:
        # Time column in events is known
        events["Time"] = events["Time"].astype("<M8[ns]")

        return events

    def merge_events(self, dataframe, event_frame) -> pd.DataFrame:
        """Merges a value and time column into dataframe for each configured event."""
        # -> should specify a specific outcome
        self.dataframe = (
            self.dataframe.sort_values(self.predict_time)
            .drop_duplicates(subset=self.entity_keys + [self.predict_time])
            .dropna(subset=[self.predict_time])
        )
        self.events = self.events.sort_values("Time")

        for event in self.config.events:
            # Merge
            if event.window_hr:
                logger.debug(
                    f"Windowing event {one_event.display_name} to lookback {one_event.window_hr} "
                    + f"offset by {one_event.offset_hr}"
                )
                self.dataframe = self._merge_event(
                    event.source,
                    frozenset(self.dataframe.columns),
                    window_hrs=event.window_hr,
                    offset_hrs=event.offset_hr,
                    display=event.display_name,
                    sort=False,
                )
                self.target_cols.append(event.display_name)
            else:  # No lookback
                logger.debug(f"Merging event {event.display_name}")
                self.dataframe = self._merge_event(
                    event.source, frozenset(self.dataframe.columns), display=event.display_name, sort=False
                )

            # Impute
            if event.impute_val or event.usage == "target":
                event_val = pdh.event_value(event.display_name)
                impute = event.impute_val or 0  # Enforce binary type target
                self.dataframe[event_val] = self.dataframe[event_val].fillna(impute)

        return dataframe

    def _merge_event(
        self, event_col, dataframe, event_frame, offset_hrs=0, window_hrs=None, display="", sort=True
    ) -> pd.DataFrame:
        disp_event = display if display else event_col
        translate_event = event_col if display else ""

        return pdh.merge_windowed_event(
            self.dataframe,
            self.predict_time,
            self.events.replace({"Type": translate_event}, disp_event),
            disp_event,
            self.entity_keys,
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
