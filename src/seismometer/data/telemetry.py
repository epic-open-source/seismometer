import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from seismometer.core.autometrics import AutomationManager
from seismometer.core.io import slugify
from seismometer.data.filter import FilterRule
from seismometer.data.otel import MetricTelemetryManager

logger = logging.getLogger("seismometer.telemetry")


class OpenTelemetryRecorder:
    def __new__(cls, metric_names: List[str], name: str):
        if MetricTelemetryManager().active:
            return RealOpenTelemetryRecorder(metric_names, name)
        return NoOpOpenTelemetryRecorder()


class NoOpOpenTelemetryRecorder:
    def __init__(self):
        self.meter = None
        self.instruments = {}
        self.metric_names = []

    def populate_metrics(self, attributes, metrics):
        pass


class RealOpenTelemetryRecorder:
    def __init__(self, metric_names: List[str], name: str = "Seismo-meter"):
        """Set up a list of instruments according to the metrics we are logging.

        Parameters
        ----------
        metric_names : List[str]
            These are the kinds of metrics we want to be collecting. E.g. fairness, accuracy.
        """

        export_manager = MetricTelemetryManager()

        # If we are not recording metrics, don't bother.
        if not export_manager.active:
            self.metric_names = []
            return

        meter_provider = export_manager.meter_provider
        # OpenTelemetry: use this new object to spawn new "Instruments" (measuring devices)
        self.meter = meter_provider.get_meter(name)
        # Keep it like this for now: just make an instrument for each metric we are measuring
        # This is a map from metric name to corresponding instrument
        self.instruments: dict[str, Any] = {}
        self.metric_names = metric_names
        for mn in metric_names:
            creator_fn = self.get_metric_creator(mn)
            self.instruments[mn] = creator_fn(slugify(mn), description=mn)

    def get_metric_creator(self, metric_name: str) -> Callable:
        """Takes in the name of a metric and determines the OTel function which creates
        the corresponding instrument.

        Parameters
        ----------
        metric_name : str
            Which metric, like "Accuracy".

        Returns
        -------
        Callable
            The function creating the right metric instrument.

        """
        TYPES = {
            "Gauge": self.meter.create_gauge,
            "Counter": self.meter.create_up_down_counter,
            "Histogram": self.meter.create_histogram,
        }
        typestring = AutomationManager().get_metric_config(metric_name).measurement_type
        return TYPES[typestring] if typestring in TYPES else self.meter.create_gauge

    def populate_metrics(self, attributes: dict[str, Union[str, int]], metrics: dict[str, Any]) -> None:
        """Populate the OpenTelemetry instruments with data from
        the model.

        Parameters
        ----------
        attributes: dict[str, Union[str, int]]
            All information associated with this metric. For instance,
                - what cohort is this a part of?
                - what metric is this actually?
                - what are the score and fairness thresholds?
                - etc.

        metrics : dict[str, Any].
            The actual data we are populating.
        """

        if not MetricTelemetryManager().active:
            return

        # OTel doesn't like fancy Unicode characters.
        metrics = {metric_name.replace("\xa0", " "): metrics[metric_name] for metric_name in metrics}
        am = AutomationManager()
        for name in self.instruments.keys():
            # I think metrics.keys() is a subset of self.instruments.keys()
            # but I am not 100% on it. So this stays for now.
            if name in metrics and am.get_metric_config(name).output_metrics:
                self._log_to_instrument(attributes, self.instruments[name], metrics[name])
        if not any([name in metrics for name in self.instruments.keys()]):
            logger.warning("No metrics populated with this call!")
            logger.warning(f"Instruments available: {self.metric_names}")
            logger.warning(f"Metrics provided for population: {metrics.keys()}")

    def _log_to_instrument(self, attributes, instrument: Any, data):
        """Write information to a single instrument. We need this
        wrapper function because the data we are logging may be a
        type such as a series, in which case we need to log each
        entry separately or at least do some extra preprocessing.

        Parameters
        ----------
        attributes : dict[str, tuple[Any]]
            Which parameters go into this measurement.
        instrument : Any
            The OpenTelemetry instrument that we are recording a
            measurement to. Should be one of self.instruments.
        data
            The data we are recording. Could conceivably be either
            a numeric value or a series.
        """
        if data is None:
            logger.warning("otel: Tried to log 'None'.")
            return
        # Attributes cannot contain lists, so we make them into tuples.
        attributes = {k: tuple(v) if isinstance(v, list) else v for k, v in attributes.items()}
        if isinstance(data, (int, float)):
            self._set_one_datapoint(attributes, instrument, data)
        # Some code seems to be logging numpy int64s so here we are.
        elif isinstance(data, (np.int64, np.float64)):
            self._set_one_datapoint(attributes, instrument, data.item())
        elif isinstance(data, list):
            for datapoint in data:
                self._set_one_datapoint(attributes, instrument, datapoint)
        elif isinstance(data, (dict, pd.Series)):
            # Same logic for both dictionary and (labeled) series,
            # but we would need to get the dictionary representation of the series first.
            for k, v in dict(data).items():
                # Augment attributes with keys of dictionary
                self._log_to_instrument(attributes | {"key": k}, instrument, v)
        elif isinstance(data, str):
            raise Exception(f"Cannot log strings as metrics in OTel. Tried to log {data} as a metric.")
        else:
            raise Exception(f"Unrecognized data format for OTel logging: {str(type(data))}")

    def _set_one_datapoint(self, attributes, instrument, value):
        from opentelemetry.metrics import Histogram, UpDownCounter

        # The SDK doesn't expose Gauge as a type, so we need to get creative here.
        if type(instrument).__name__ == "_Gauge":
            instrument.set(value, attributes=attributes)
        elif isinstance(instrument, UpDownCounter):
            instrument.add(value, attributes=attributes)
        elif isinstance(instrument, Histogram):
            instrument.record(value, attributes=attributes)
        else:
            raise Exception(
                f"Internal error: one of the instruments is not a recognized type: {str(type(instrument))}"
            )


def record_dataframe_metrics(
    df: pd.DataFrame,
    metrics: Union[str, List[str]],
    *,
    attribute_cols: Optional[List[str]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    filter_rule: Optional[FilterRule] = None,
    source: str = "DataFrameMetrics",
) -> None:
    """Record metrics from a DataFrame with OpenTelemetry.

    This is the core API for metric logging. It handles recorder setup internally
    and uses AutomationManager configuration to determine what gets logged.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metrics and attributes to record.
    metrics : Union[str, List[str]]
        Column name(s) in the DataFrame that contain metric values to record.
    attribute_cols : Optional[List[str]], optional
        Column names in the DataFrame to use as attributes (labels) for metrics.
        Column names act as labels, row values become label values for that metric recording.
    attributes : Optional[Dict[str, Any]], optional
        Base attributes to apply to all metric recordings. These are merged with
        per-row label/value pairs from attribute_cols. Keys will be additional label names, and values
        will be label values.
    filter_rule : Optional[FilterRule], optional
        Filter to apply to the DataFrame before recording metrics. If None,
        all rows are processed. If any metric has log_all=True in AutomationManager
        configuration, that metric will be recorded for all rows regardless of the filter.
        Other metrics will only be recorded for rows that pass the filter.
    source : str, optional
        Source name for the OpenTelemetry recorder, by default "DataFrameMetrics".

    Examples
    --------
    Simple case - record a single metric for all rows:
    >>> df = pd.DataFrame({'accuracy': [0.85, 0.90], 'model': ['A', 'B']})
    >>> record_dataframe_metrics(df, 'accuracy')

    With attributes from DataFrame columns:
    >>> record_dataframe_metrics(df, 'accuracy', attribute_cols=['model'])

    With base attributes and filtering:
    >>> record_dataframe_metrics(
    ...     df,
    ...     ['accuracy', 'precision'],
    ...     attribute_cols=['model'],
    ...     attributes={'experiment': 'test_1'},
    ...     filter_rule=FilterRule.gt('accuracy', 0.8)
    ... )
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    # Handle empty DataFrame gracefully - this is a common case and shouldn't be an error
    if df.empty:
        return

    # Validate metric columns exist
    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Metric columns not found in DataFrame: {missing_metrics}")

    # Validate attribute columns exist
    attribute_cols = attribute_cols or []
    missing_attrs = [col for col in attribute_cols if col not in df.columns]
    if missing_attrs:
        raise ValueError(f"Attribute columns not found in DataFrame: {missing_attrs}")

    # Split metrics into log_all vs log_some based on AutomationManager configuration
    log_all_metrics = []
    log_some_metrics = []

    am = AutomationManager()
    for metric in metrics:
        if am.get_metric_config(metric).log_all:
            log_all_metrics.append(metric)
        else:
            log_some_metrics.append(metric)

    # Create single recorder for all metrics
    recorder = OpenTelemetryRecorder(metric_names=metrics, name=source)
    base_attributes = attributes or {}

    # Helper function to record a single row using apply
    def record_row(row, metrics_to_record: set):
        # Build attributes for this row
        row_attributes = base_attributes.copy()
        for attr_col in attribute_cols:
            row_attributes[attr_col] = row[attr_col]

        # Build metrics dict for this row (only the relevant metrics)
        row_metrics = {metric: row[metric] for metric in metrics_to_record}
        recorder.populate_metrics(attributes=row_attributes, metrics=row_metrics)

    # Process log_all metrics with full dataset
    if log_all_metrics:
        df[log_all_metrics + attribute_cols].apply(lambda row: record_row(row, log_all_metrics), axis=1)

    # Process log_some metrics with filtered dataset
    if log_some_metrics:
        filtered_df = df
        if filter_rule is not None:
            filtered_df = filter_rule.filter(df, ignore_min_rows=True)

        filtered_df[log_some_metrics + attribute_cols].apply(lambda row: record_row(row, log_some_metrics), axis=1)


def record_dataframe_quantiles(
    df: pd.DataFrame,
    metrics: Union[str, List[str]],
    *,
    attribute_cols: Optional[List[str]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    filter_rule: Optional[FilterRule] = None,
    source: str = "DataFrameQuantiles",
):
    """Record quantile metrics from a DataFrame with OpenTelemetry.

    This function computes quantiles for one or more metric columns, grouped by attribute_cols,
    and records per group quantiles using the core record_dataframe_metrics API.
    Quantiles are calculated per attribute_cols group and per metric using AutomationManager
    configuration, which allows for custom quantile specifications.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to compute quantiles from.
    metrics : Union[str, List[str]]
        Column name(s) containing the metric values to compute quantiles for.
        Each metric can have its own quantile configuration from AutomationManager:
        - List: custom quantile values (e.g., [0.5, 0.9, 0.95, 0.99])
    attribute_cols : Optional[List[str]], optional
        Column names to group by before computing quantiles. Each group will
        have its own set of quantile metrics recorded.
    attributes : Optional[Dict[str, Any]], optional
        Base attributes to apply to all quantile metric recordings.
    filter_rule : Optional[FilterRule], optional
        Filter to apply to the DataFrame before computing quantiles.
    source : str, optional
        Source name for the OpenTelemetry recorder, by default "DataFrameQuantiles".

    Examples
    --------
    Simple quantiles for entire dataset:
    >>> df = pd.DataFrame({'latency': [100, 200, 150, 300, 250]})
    >>> record_dataframe_quantiles(df, 'latency')

    Multiple metrics with different quantile configurations:
    >>> df = pd.DataFrame({
    ...     'latency': [100, 200, 150, 300, 250, 180],
    ...     'throughput': [50, 45, 60, 40, 55, 58],
    ...     'model': ['A', 'A', 'B', 'B', 'A', 'B']
    ... })
    >>> record_dataframe_quantiles(
    ...     df, ['latency', 'throughput'],
    ...     attribute_cols=['model'],
    ...     attributes={'experiment': 'test_1'}
    ... )
    # With AutomationManager config:
    # latency: quantiles: 10  # -> [0.1, 0.2, ..., 0.9]
    # throughput: quantiles: [0.5, 0.9, 0.95]  # -> [0.5, 0.9, 0.95]
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    if not metrics:
        raise ValueError("At least one metric must be specified")

    # Get per-metric quantile configuration from AutomationManager
    # The AutomationManager will return default configs for any metrics not explicitly configured
    metric_quantiles = {}

    for metric in metrics:
        metric_config = AutomationManager().get_metric_config(metric)
        # quantiles is now already a List[float] thanks to pydantic validation
        metric_quantiles[metric] = metric_config.quantiles

    # Apply filter if provided
    if filter_rule is not None:
        df = filter_rule.filter(df, ignore_min_rows=True)

    # Handle empty DataFrame gracefully - this is a common case and shouldn't be an error
    if df.empty:
        return pd.DataFrame()

    # Validate metric columns exist
    missing_metrics = [col for col in metrics if col not in df.columns]
    if missing_metrics:
        raise ValueError(f"Metric columns not found in DataFrame: {missing_metrics}")

    # Validate attribute columns exist
    attribute_cols = attribute_cols or []
    missing_cols = [col for col in attribute_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Attribute columns not found in DataFrame: {missing_cols}")

    # Compute quantiles
    if attribute_cols:
        # Group by specified columns and compute quantiles for each group
        quantile_data = []

        for group_values, group_df in df.groupby(attribute_cols):
            # Handle single attribute column case
            if len(attribute_cols) == 1:
                group_values = (group_values,)

            # Create a single row with all quantiles for this group
            row_data = dict(zip(attribute_cols, group_values))

            # Compute quantiles for each metric using per-metric configuration
            for metric in metrics:
                for q in metric_quantiles[metric]:
                    quantile_value = group_df[metric].quantile(q)
                    row_data[f"{metric}_quantile_{q}"] = quantile_value

            quantile_data.append(row_data)

        if not quantile_data:
            logger.warning("No quantile data computed")
            return pd.DataFrame()

        # Create DataFrame with quantile results
        quantile_df = pd.DataFrame(quantile_data)

    else:
        # No grouping - compute quantiles for entire dataset
        row_data = {}

        # Compute quantiles for each metric using per-metric configuration
        for metric in metrics:
            for q in metric_quantiles[metric]:
                quantile_value = df[metric].quantile(q)
                row_data[f"{metric}_quantile_{q}"] = quantile_value

        if not row_data:
            logger.warning("No quantile data computed")
            return pd.DataFrame()

        # Create DataFrame with quantile results (single row)
        quantile_df = pd.DataFrame([row_data])

    # Get metric columns (all quantile columns)
    metric_columns = [col for col in quantile_df.columns if "_quantile_" in col]

    # Record using the core API
    record_dataframe_metrics(
        quantile_df, metric_columns, attribute_cols=attribute_cols, attributes=attributes, source=source
    )

    return quantile_df


def record_dataframe_matrix(
    df: pd.DataFrame,
    metric: str,
    *,
    attributes: Optional[Dict[str, Any]] = None,
    source: str = "DataFrameCounts",
) -> None:
    """Record metrics from a matrix-structured DataFrame with OpenTelemetry.

    This function is designed for DataFrames where:
    - Row indices represent one dimension (e.g., cohort groups)
    - Columns represent another dimension (e.g., score categories, response options)
    - Cell values are the metrics to record

    Each cell is recorded as a separate metric observation with attributes
    derived from both the row index and column name.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing count data where:
        - Row indices represent cohort groups/subgroups
        - Columns represent score categories or response types
        - Values are the counts for each group/category combination
    metric : str
        The name of the metric to record all counts under.
    attributes : Optional[Dict[str, Any]], optional
        Base attributes to apply to all metric recordings, by default None.
        These are merged with attributes derived from the DataFrame structure.
    source : str, optional
        Source name for the OpenTelemetry recorder, by default "DataFrameCounts".

    Examples
    --------
    Recording Likert scale responses across cohorts:

    >>> df = pd.DataFrame({
    ...     'Disagree': [10, 5],
    ...     'Neutral': [15, 10],
    ...     'Agree': [35, 45]
    ... }, index=[('A', 'C'), ('B', 'D')])
    >>> df.index.names = ['Group1', 'Group2']
    >>> record_dataframe_matrix(df, 'likes_cats')

    This records 6 values:
    - likes_cats{Group1="A", Group2="C", score="Disagree"} 10
    - likes_cats{Group1="A", Group2="C", score="Neutral"} 15
    - likes_cats{Group1="A", Group2="C", score="Agree"} 35
    - likes_cats{Group1="B", Group2="D", score="Disagree"} 5
    - likes_cats{Group1="B", Group2="D", score="Neutral"} 10
    - likes_cats{Group1="B", Group2="D", score="Agree"} 45
    """
    if df.empty:
        return

    # Create recorder for this single metric
    recorder = OpenTelemetryRecorder(metric_names=[metric], name=source)
    base_attributes = attributes or {}

    # Get index column names - these become attribute labels
    index_names = df.index.names if df.index.names != [None] else [f"index_{i}" for i in range(df.index.nlevels)]

    # Iterate through each row (index values) and column (score categories)
    for row_idx, row in df.iterrows():
        # Handle multi-index case
        if isinstance(row_idx, tuple):
            row_attributes = dict(zip(index_names, row_idx))
        else:
            row_attributes = {index_names[0]: row_idx}

        # Add base attributes
        row_attributes.update(base_attributes)

        # Record each column value with the score category as an attribute
        for col_name, count_value in row.items():
            # Skip NaN values
            if pd.isna(count_value):
                continue

            # Add the score category attribute
            final_attributes = row_attributes.copy()
            final_attributes["score"] = col_name

            # Record the count
            recorder.populate_metrics(attributes=final_attributes, metrics={metric: count_value})


def record_single_metric(
    name: str,
    value: Union[int, float],
    *,
    attributes: Optional[Dict[str, Any]] = None,
    source: str = "SingleMetric",
) -> None:
    """Record a single metric value with OpenTelemetry.

    This is a convenience function for recording a single metric value
    with optional attributes.

    Parameters
    ----------
    name : str
        The name of the metric to record.
    value : Union[int, float]
        The numeric value of the metric to record.
    attributes : Optional[Dict[str, Any]], optional
        Attributes to associate with this metric recording, by default None.
    source : str, optional
        Source name for the OpenTelemetry recorder, by default "SingleMetric".

    Examples
    --------
    Recording a single accuracy metric:

    >>> record_single_metric('accuracy', 0.92, attributes={'model': 'A'})
    """
    recorder = OpenTelemetryRecorder(metric_names=[name], name=source)
    final_attributes = attributes or {}

    recorder.populate_metrics(attributes=final_attributes, metrics={name: value})
