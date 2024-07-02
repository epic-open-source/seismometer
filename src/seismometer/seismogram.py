import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from seismometer.configuration import ConfigProvider
from seismometer.core.patterns import Singleton
from seismometer.data import pandas_helpers as pdh
from seismometer.data import resolve_cohorts
from seismometer.data.loader import loader_factory
from seismometer.report.alerting import AlertConfigProvider

MAXIMUM_NUM_COHORTS = 25
logger = logging.getLogger("seismometer")


class Seismogram(object, metaclass=Singleton):
    """
    Seismogram is the main orchestrator for the seismometer package.
    It loads the model data and configuration metadata. It is a singleton, so only one instance can be created.

    Seismogram is responsible for:
        1. Loading static data so that they can be treated as Immutable.
            a. Model Configuration
            b. Source Data
        2. Hold the current dynamic configuration that is shared state.
            a. Cohort Selection
        3. Providing data accces for other objects without compromising the source data.
            a. Merge Event data with Prediction data for Label generation
            b. Cohort based data selection
            c. Model configuration help texts

    As a single instance, the first time it is loaded, it will load the data from the configuration.
    In order to refresh the single instance, the kernel must be restarted, or Seismogram.kill() must be called.

    """

    entity_keys: list[str]
    """ The one or two columns used as identifiers for data. """
    predict_time: str
    """ The column name for evaluation timestamp. """
    config_path: Path
    """ The location of the main configuration file. """
    output_list: list[str]
    """ The list of columns representing model outputs."""

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
        definitions: Optional[dict] = None,
    ):
        """
        Constructor for Seismogram, which can only be instantiated once.

        Subsequent calls will get the initial instance.

        Parameters
        ----------
        config_path : str or Path, optional
            Where to find the primary configuration file, defaults to the notebook's data directory.
        output_path : str or Path, optional
            Override location to place resulting data and report files.
            Defaults to the config.yml info_dir, and then the notebook's output directory.
        definitions : dict, optional
            Additional definitions to be used instead of loading based on configuration, by default None.

        """
        if config_path is None:
            config_path = Path.cwd() / "data"
        else:
            config_path = Path(config_path)

        self.dataframe: pd.DataFrame = None
        self.cohort_cols: list[str] = []
        self.config_path = config_path

        self.load_config(config_path, definitions=definitions)

        self.config.set_output(output_path)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataloader = loader_factory(self.config)

    def load_data(
        self, *, predictions: Optional[pd.DataFrame] = None, events: Optional[pd.DataFrame] = None, reset: bool = False
    ):
        """
        Loads the seismogram data.

        Uses the passed in frames if they are specified, otherwise uses configuration to load data.
        If data is already loaded, does not change state unless reset is true.

        Parameters
        ----------
        predictions : pd.DataFrame, optional
            The fully prepared predictions dataframe, by default None.
            Uses this when specified, otherwise loads based on configuration.
        events : pd.DataFrame, optional
            The pre-loaded events dataframe, by default None.
            Uses this when specified, otherwise loads based on configuration.
        reset : bool, optional
            Flag when set to true will overwrite existing dataframe, by default False
        """
        if self.dataframe is not None and not reset:
            logger.debug("Data already loaded; pass reset=True to clear data and re-evaluate.")
            return

        self._load_metadata()

        self.dataframe = self.dataloader.load_data(predictions, events)

        self.create_cohorts()
        self._set_df_counts()

        # UI Controls
        if self.cohort_cols:
            self.available_cohort_groups = {
                cohort_attr: self.dataframe[cohort_attr].cat.categories.tolist() for cohort_attr in self.cohort_cols
            }
            self.selected_cohort = (self.cohort_cols[0], self.available_cohort_groups[self.cohort_cols[0]])

    # region data accessors

    @property
    def events(self) -> pd.DataFrame:
        """
        DataFrame of events which include the target event and other outcomes.
        """
        return self._events

    @events.setter
    def events(self, df: pd.DataFrame):
        self._events = df

    @property
    def events_columns(self) -> dict[str, str]:
        """
        Event descriptions.
        """
        return self._event_column_map

    @events_columns.setter
    def events_columns(self, mapping_dict: dict[str, str]):
        self._event_column_map = mapping_dict

    @property
    def target(self):
        """
        Name of the target.
        """
        return pdh.event_value(self.target_event)

    @property
    def time_zero(self):
        """The time associated with the primary target event."""
        return pdh.event_time(self.target_event)

    @property
    def output(self):
        """The first configured model output (score)."""
        return self.output_list[0]

    @property
    def target_event(self) -> str:
        """
        Name of the target event.
        """
        return self._target

    @target_event.setter
    def target_event(self, event: str):
        if event.endswith("_Value"):
            event = event[:-6]
        self._target = event

    @property
    def target_cols(self) -> list:
        return self.config.targets

    @property
    def intervention(self):
        """First event in configuration with usage 'intervention'."""
        try:
            return self.config.interventions[0]
        except IndexError as exc:
            raise IndexError("No interventions defined in configuration") from exc

    @property
    def outcome(self):
        """First event in configuration with usage 'outcome'."""
        try:
            return self.config.outcomes[0]
        except IndexError as exc:
            raise IndexError("No outcomes defined in configuration") from exc

    @property
    def comparison_time(self):
        """Time used for reference point to intervents and outcomes."""
        return self.config.comparison_time

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        The working dataframe core to the analyses.

        This property and the event-filtered method data(event) are the primary data
        accessors to be used by tables, reports and visualizations.

        The dataframe returned contains the merged events and all prediction and cohort information, having rows
        per model evaluation and columns from the data usage configuration.
        """
        return self._dataframe

    @dataframe.setter
    def dataframe(self, df: pd.DataFrame):
        self._dataframe = df

    @property
    def output_path(self):
        """The location to write output files."""
        return self.config.output_dir

    @property
    def censor_threshold(self) -> int:
        """Minimum number of observations for a cohort to be included."""
        return self.config.censor_min_count

    @property
    def prediction_count(self) -> int:
        """Number of predictions in the data."""
        return self._prediction_count

    @property
    def entity_count(self) -> int:
        """Number of unique prediction entities in the data."""
        return self._entity_count

    @property
    def feature_count(self) -> int | str:
        """Number of features in the data."""
        return self._feature_counts

    @property
    def start_time(self) -> pd.Timestamp:
        """Earliest prediction time."""
        return self._start_time

    @property
    def end_time(self) -> pd.Timestamp:
        """Latest prediction time."""
        return self._end_time

    @property
    def event_types_count(self) -> int:
        """Number of unique outcome KPIs in the data."""
        return self._event_types_count

    @property
    def cohort_attribute_count(self) -> int:
        """Number of unique cohort attributes by usage definition."""
        return self._cohort_attribute_count

    def _set_df_counts(self):
        self._prediction_count = len(self.dataframe)
        self._entity_count = self.dataframe[self.entity_keys[0]].nunique()

        feature_counts = len(self.config.features)
        if feature_counts > 0:
            self._feature_counts = feature_counts
        else:
            self._feature_counts = f"~{max(0, len(self.dataframe.columns) - self.config.prediction_columns)}"

        self._start_time = self.dataframe[self.predict_time].min()
        self._end_time = self.dataframe[self.predict_time].max()
        self._event_types_count = len(self.config.events)
        self._cohort_attribute_count = len(self.config.cohorts)

    # endregion
    # region data accessors
    def data(self, event: str = None) -> pd.DataFrame:
        """
        Provides data for the specified target event.

        Expects the event string, defaults to the configured primary target.
        """
        event_val = pdh.event_value(event) or self.target
        event_time = pdh.event_time(event) or self.time_zero  # Assumes binary target
        if event_time in self.dataframe:
            return self.dataframe[self._data_mask(event_val) & self._time_mask(event_time)]

        return self.dataframe[self._data_mask(event_val)]

    def score_bins(self):
        """Updates the active values for notebook-scoped selections."""
        score_bins = [0] + self.thresholds + [1.0]
        return sorted(score_bins)

    # endregion

    # region initialization and preprocessing (this region knows about config)
    def load_config(self, config_path: Path, definitions: Optional[dict] = None):
        """
        Loads the base configuration and alerting congfiguration

        Parameters
        ----------
        config_path : Path
            The location of the main configuration file.
        definitions : Optional[dict], optional
            An optional dictionary containing both events and predictions lists, by default None.
            If not passed, these will be loaded based on configuration.
        """
        self.config = ConfigProvider(config_path, definitions=definitions)
        self.alert_config = AlertConfigProvider(config_path)

        if len(self.config.cohorts) == 0:
            logger.warning("No cohort columns were configured; tool may behave unexpectedly.")

        # copy over attributes [freeze values]
        self.entity_keys = self.config.entity_keys
        self.predict_time = self.config.predict_time
        self.output_list = self.config.output_list
        self.target_event = self.config.target
        self._cohorts = self.config.cohorts

    def _load_metadata(self):
        """
        Loads metadata including model name (by default, "UNDEFINED MODEL") and set of threshold values
        (by default, [0.8, 0.5]) to display on performance plots.
        """
        with open(self.config.metadata_path, "r") as file:
            self._metadata = json.load(file)
        try:
            self.thresholds: list[float] = self._metadata["thresholds"]
        except KeyError:
            logger.warn("No thresholds set in metadata.json. Using [0.8, 0.5]")
            self.thresholds = [0.8, 0.5]

        self.modelname: str = self._metadata.get("modelname", "UNDEFINED MODEL")

    def create_cohorts(self) -> None:
        """Creates data columns for each cohort defined in configuration."""
        for cohort in self.config.cohorts:
            disp_attr = cohort.display_name
            if cohort.splits:
                try:
                    new_col = resolve_cohorts(self.dataframe[cohort.source], cohort.splits)
                except IndexError as exc:
                    logger.warn(f"Failed to resolve cohort {disp_attr}: {exc}")
                    continue
            else:
                new_col = pd.Series(pd.Categorical(self.dataframe[cohort.source]))

            # validate counts (per observation)
            if (N := len(new_col.cat.categories)) > MAXIMUM_NUM_COHORTS:  # More accurate if censor first
                logger.warning(
                    f"Too many unique values to cohort {disp_attr}. Limit to {MAXIMUM_NUM_COHORTS} found {N}."
                )
                continue
            sufficient = new_col.value_counts(sort=False) > self.censor_threshold
            if not sufficient.any():
                logger.warning(
                    f"No cohort on {disp_attr} met censor limit {self.censor_threshold}; dropping cohort option."
                )
                continue
            if not sufficient.all():
                logger.debug(
                    f"Some cohorts of {disp_attr} were below censor limit: {', '.join(sufficient[~sufficient].index)}"
                )

            self.dataframe[disp_attr] = new_col.cat.set_categories(sufficient[sufficient].index.tolist(), ordered=True)
            self.cohort_cols.append(disp_attr)
        logger.debug(f"Created cohorts: {', '.join(self.cohort_cols)}")

    @lru_cache
    def _data_mask(self, event_val):
        return self.dataframe[event_val] != -1

    @lru_cache
    def _time_mask(self, event_time, keep_zero: str = None):
        # Require events with time or negative label
        neg_mask = np.ones(keep_zero).astype(bool) if keep_zero is None else self.dataframe[keep_zero] == 0
        return self.dataframe[event_time].notna() | neg_mask

    # endregion
