import logging
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("seismometer")

FileLike = str | Path
DirLike = str | Path
AggregationStrategies = Literal["min", "max", "first", "last"]
MergeStrategies = Literal["first", "last", "nearest", "forward", "count"]


class OtherInfo(BaseModel):
    """Locations of configuration and data files."""

    template: Optional[str] = None
    """ Descriptor of the template; not used. """
    info_dir: Optional[DirLike] = None
    """ Writable directors for output; used during run. """
    event_definition: Optional[FileLike] = None
    """ The location of the event dictionary. """
    prediction_definition: Optional[FileLike] = None
    """ The location of the prediction dictionary. """
    usage_config: Optional[FileLike] = None
    """ The location of the usage configuration; used during run. """

    # The following specify where to find the data at notebook runtime
    data_dir: Optional[DirLike] = None
    """ The parent location of data files; primarily used during run. """
    prediction_path: FileLike = "scores.parquet"
    """ The location of the prediction data within data_dir."""
    event_path: FileLike = "events.parquet"
    """ The location of the event data within data_dir."""
    metadata_path: FileLike = "metadata.json"
    """ The location of the metadata file within data_dir."""


class DictionaryItem(BaseModel):
    """Defines a generic dictionary item."""

    name: str
    """ The source name of the item, should match raw data. """
    display_name: str = Field(default="", validate_default=True)
    """ The name to display for the item, defaults to name, currently ignored. """
    dtype: Optional[str] = None
    """ The pandas-compatible type of data, currently ignored. """
    definition: Optional[str] = None
    """ The definition of the item. """

    @field_validator("display_name")
    def default_display_name(cls, display_name: str, values: dict) -> str:
        """
        Ensures that the display_name is set to the name if not provided.

        Parameters
        ----------
        display_name : str
            The display name of the item.
        values : dict
            The values of the instance.

        Returns
        -------
        str
            The resolved display name.
        """
        return display_name or values.data.get("name")


class PredictionDictionary(BaseModel):
    """
    The dictionary information for prediction data.

    This is the structure of a dictionary corresponding to the prediction frame.
    Generally, the predictions frame is all the information known near model execution time such as inputs, outputs,
    and cohort attributes.

    """

    predictions: list[DictionaryItem] = []
    """ The list of all columns in the predictions data."""

    def __getitem__(self, key: str) -> Optional[DictionaryItem]:
        """
        Get the definition of an item.

        Parameters
        ----------
        key : str
            The column name.

        Returns
        -------
        Optional[str]
            The definition of the column.
        """
        return _search_dictionary(self.predictions, key)

    def get(self, key: str, default: Optional[Any] = None) -> Union[DictionaryItem, Any]:
        """
        Get the definition of an item.

        Parameters
        ----------
        key : str
            The column name.
        default : Optional[Any]
            The default value to return if the key is not found, defaults to None.

        Returns
        -------
        The DictionaryItem with name specified or the default value
        """
        try:
            return self[key]
        except KeyError:
            return default


class EventDictionary(BaseModel):
    """
    The dictionary information for events data.

    This is the structure of a dictionary file corresponding to the event frame.
    Generally, the events frame is all the information not known near model execution time - such as target,
    interventions, and outcomes.
    """

    events: list[DictionaryItem] = []

    """ The list of all columns in the events data."""

    def __getitem__(self, key: str) -> Optional[DictionaryItem]:
        """
        Get the definition of an item.

        Parameters
        ----------
        key : str
            The column name.

        Returns
        -------
        Optional[str]
            The definition of the column.
        """
        return _search_dictionary(self.events, key)

    def get(self, key: str, default: Optional[Any] = None) -> Union[DictionaryItem, Any]:
        """
        Get the definition of an item.

        Parameters
        ----------
        key : str
            The column name.
        default : Optional[Any]
            The default value to return if the key is not found, defaults to None.

        Returns
        -------
        The DictionaryItem with name specified or the default value
        """
        try:
            return self[key]
        except KeyError:
            return default


class Cohort(BaseModel):
    """
    The definition of an expected cohort attribute.

    This structure defines a cohort attribute that should be available for selection in a notebook.
    For a categorical data, the splits should all be existing values and the list limits the selections available.
    For numerical data, the splits should be the inner boundaries of bucketing; with a high and low being added
    outside theses values.
    """

    source: str
    """ The source column name for a cohort. """
    display_name: str = Field(default="", validate_default=True)
    """
    The display name for the cohort.

    If not specified, defaults to the source name.
    Display names must be unique across the dataset and are what is referenced in usage configuration.
    """
    splits: Optional[list[Any]] = []
    """ An optional list of 'inner edges' used to create a set of cohorts from a continuous attribute."""

    @field_validator("display_name")
    def default_display_name(cls, display_name: str, values: dict) -> str:
        """Ensures that display_name exists, setting it to the source name if not provided."""
        return display_name or values.data.get("source")


class Event(BaseModel):
    """
    The definition of an event.

    This structure defines an event and which predictions are relevant to it.
    If a window is specified:

    - the offset_hr defines the upper bound of the window relative to the event time,
      has default value of 0 (event time),
    - the window_hr defines the size of the window looking backwards from the offset_hr.

    If an event is present but the prediction is not in the window, the predictions are ignored for the event type.
    If multiple events are present then the closest one is used.

    The impute_val is used as the value for the event if no event is present.

    Usage is used for context when selecting events, such as analyzing performance of the model with respect to a
    target or when comparing an expected intervention to a monitored outcome.
    """

    source: list[str]
    """
    The source(s) for an event.

    Supports a single event type or a list of event types.
    If a list is provided, the display_name must be specified.
    """
    display_name: str = Field(default="", validate_default=True)
    """
    The display name for the event.

    If not specified, defaults to the source name if singular, otherwise must be specified.
    Display names must be unique across the dataset and are what is referenced in usage configuration.
    """
    window_hr: Optional[float] = None
    """ The size of the valid window in hours. """
    offset_hr: float = 0
    """ The number of hours to offset the valid window before the reference time. """
    impute_val: Optional[Any] = None
    """ The value to use if no event is present. """
    usage: Optional[str] = None
    """ The type of event being defined; can be target, intervention, or outcome. """
    aggregation_method: Optional[AggregationStrategies] = "max"
    """
    The strategy for aggregating (or selecting) scores for an event.
    Supports min, max, first, and last; defaulting to max.
    """
    merge_strategy: Optional[MergeStrategies] = "forward"
    """
    The strategy for merging events with predictions.
    Supports first, last, nearest, forward, and count; defaulting to forward.

    | - first: The first event within the window is selected. If no window, the first event is selected.
    | - last: The last event within the window is selected. If no window, the last event is selected.
    | - nearest: The event closest to the prediction time within the window is selected.
                 If no window, the nearest event is selected.
    | - forward: The first event at or after the prediction time within the window is selected.
                 If no window, the first event at or after the prediction time is selected.
    | - count: Creates a count column for each event value for specified event label.
               This column tracks the occurrences of each event value for each prediction.
               Counts respects the window.

    """

    @field_validator("source", mode="before")
    @classmethod
    def coerce_source_list(cls, v) -> list[str]:
        """Coerce single values for source into a list so all downstream processing can assume a list."""
        if isinstance(v, str):
            v = [v]
        return v

    @field_validator("display_name")
    def default_display_name(cls, display_name: str, values: dict) -> str:
        """
        Ensures that display_name exists.

        If display_name is not explicitly set, it will be set to the source name if singular valued.
        When multiple sources, raise an error if no display_name is provided.
        """
        if display_name:
            return display_name
        source_list = values.data.get("source", [])
        if len(source_list) == 1:
            return source_list[0]
        raise ValueError("A display_name must be specified for multiple source event types.")


class EventTableMap(BaseModel):
    """Override mapping of event table columns."""

    type: str = "Type"
    """ The column name of the event type. """
    time: str = "EventTime"
    """ The column name of the event time. """
    value: str = "Value"
    """ The column name of the event value. """


class DataUsage(BaseModel):
    """
    The definitions of data to use in a notebook run.

    This structure defines what data to load and how to use it.
    The entity_id and context_id are the possible keys for joining events and predictions, and are also used to
    summarize predictions to a single entity.
    Primary output and target are the score and target used in default performance analysis.

    The features and scores list, when defined, limit the loading of data from the predictions file to only those
    inputs and outputs (plus primary_score and cohort attributes).
    The events similarly limits the event types that are merged into the working dataframe and available to analyses.
    """

    entity_id: str = "Id"
    """ The identifier of the entity. """
    context_id: Optional[str] = None
    """ A secondary identifier used to group an entity_id. """
    primary_output: str = "Score"
    """ Column name of the primary output of the model. """
    primary_target: str = "Target"
    """ Display_name of the primary target event. """
    predict_time: str = "Time"
    """ Column name of the timestamp for each prediction row. """
    comparison_time: str = Field("", validate_default=True)
    """ The timestamp to use as reference for comparison."""
    event_table: EventTableMap = EventTableMap()
    """ Mapping of the non-id columns in events data. """
    outputs: list[str] = []
    """ A list of all columns to consider outputs; does not need to include primary_output. """
    cohorts: list[Cohort] = []
    """ A list of all cohort attributes to make available in selections. """
    features: list[str] = []
    """
    A list of all features to load into predictions.

    Can exclude any features that are specified elsewhere. If not specified, will load all columns from the specified
    location.
    """
    events: list[Event] = []
    """
    A list of all events to load.

    Must have at least one target event.
    """

    censor_min_count: int = Field(10, ge=10)
    """ The minimum size of a cohort to be considered displayable. """

    @field_validator("comparison_time")
    def default_comparison(cls, comparison_time: str, values: dict) -> str:
        """Return the default comparison_time."""
        return comparison_time or values.data.get("predict_time")

    @field_validator("events")
    def reduce_events_to_unique_names(cls, events, values) -> list[Cohort]:
        """
        Reduces the list of events to unique names.

        Parameters
        ----------
        events : list[Event]
            List of configured events.
        values : Any
            Values of the instance.

        Returns
        -------
        list[Cohort]
            The unique list of events.
        """
        if not events:
            return []

        return DataUsage._reduce_derived_features(events, "Events")

    @field_validator("cohorts")
    def reduce_cohorts_to_unique_names(cls, cohorts, values) -> list[Cohort]:
        """
        Reduces the list of cohorts to unique names.

        Parameters
        ----------
        cohorts : List[Cohorts]
            List of configured cohorts.
        values : Any
            Values of the instance.

        Returns
        -------
        list[Cohort]
            The unique list of cohorts.
        """
        if not cohorts:
            return []

        return DataUsage._reduce_derived_features(cohorts, "Cohorts")

    @staticmethod
    def _reduce_derived_features(derived_features: list[Event | Cohort], field_name="DataUsage") -> list[Any]:
        """
        Reduces the list of derived features to unique names.

        Parameters
        ----------
        derived_features : list[Event|Cohort]
            List of derived features.
        field_name : str, optional
            Name of the field for sourcing in logs, by default 'DataUsage'.

        Returns
        -------
        list[Any]
            The unique list of derived features.
        """
        if not derived_features:
            return []

        seen_names = set()
        good_features = []
        for feature in derived_features:
            if feature.display_name in seen_names:
                logger.warning(
                    f"Duplicate display_name '{feature.display_name}' found in {field_name}. "
                    "Keeping the first instance."
                )
                continue
            seen_names.add(feature.display_name)
            good_features.append(feature)
        return good_features


def _search_dictionary(dictionary: list[DictionaryItem], key: str) -> Optional[DictionaryItem]:
    for item in dictionary:
        if item.name == key:
            return item
    raise KeyError(f"{key} not found")
