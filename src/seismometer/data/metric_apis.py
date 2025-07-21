import functools
import itertools
import logging
import operator
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from seismometer.core.io import slugify
from seismometer.data.otel import ExportManager, get_metric_config, get_metric_creator

logger = logging.getLogger("Seismometer OpenTelemetry")

try:
    from opentelemetry.metrics import Histogram, UpDownCounter

    TELEMETRY_POSSIBLE = True
except ImportError:
    # No OTel.
    TELEMETRY_POSSIBLE = False


class OpenTelemetryRecorder:
    def __new__(cls, *args, **kwargs):
        if TELEMETRY_POSSIBLE:
            return RealOpenTelemetryRecorder(*args, **kwargs)
        else:
            return NoOpOpenTelemetryRecorder(*args, **kwargs)


class NoOpOpenTelemetryRecorder:
    def __init__(self, *args, **kwargs):
        pass

    def populate_metrics(self, *args, **kwargs):
        pass

    def log_by_cohort(self, *args, **kwargs):
        pass

    def log_by_column(self, *args, **kwargs):
        pass


class RealOpenTelemetryRecorder:
    def __init__(self, metric_names: List[str], name: str = "Seismo-meter"):
        """_summary_

        Parameters
        ----------
        metric_names : List[str]
            These are the kinds of metrics we want to be collecting. E.g. fairness, accuracy.
        """

        export_manager = ExportManager()

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
            creator_fn = get_metric_creator(mn, self.meter)
            self.instruments[mn] = creator_fn(slugify(mn), description=mn)

    def populate_metrics(self, attributes, metrics):
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

        if not ExportManager().active:
            return

        if metrics is None:
            # metrics = self()
            raise Exception()
        # OTel doesn't like fancy Unicode characters.
        metrics = {metric_name.replace("\xa0", " "): metrics[metric_name] for metric_name in metrics}
        for name in self.instruments.keys():
            # I think metrics.keys() is a subset of self.instruments.keys()
            # but I am not 100% on it. So this stays for now.
            if name in metrics and get_metric_config(name)["output_metrics"]:
                self._log_to_instrument(attributes, self.instruments[name], metrics[name])
        if not any([name in metrics for name in self.instruments.keys()]):
            logger.warning("No metrics populated with this call!")
            logger.warning(f"Instruments available: {self.metric_names}")
            logger.warning(f"Metrics provided for population: {metrics.keys()}")

    def _set_one_datapoint(self, attributes, instrument, value):
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

    def log_by_cohort(
        self,
        dataframe: pd.DataFrame,
        base_attributes: Dict[str, Any],
        cohorts: Dict[str, List[str]],
        intersecting: bool = False,
        metric_maker: Callable = None,
    ):
        """Take data from a dataframe and log it, selecting by all cohorts provided.

        Parameters
        ----------
        base_attributes : Dict[str, Any]
            The information we want to store with every individual metric log. This might be, for example, every
            parameter used to generate this data.
        dataframe : pd.DataFrame
            The actual dataframe containing the data we want to log.
        cohorts : Dict[str, List[str]]
            Which cohorts we want to select on. For instance:
            {"Age": ["[10-20)", "70+"], "Race": ["AfricanAmerican", "Caucasian"]}
        intersecting: bool
            Whether we are logging each combination of separate cohorts or not.
            Given the example cohorts above:
                - intersecting=False would log data for Age=[10-20), Age=70+, Race=AfricanAmerican, and Race=Caucasian.
                - intersecting=True: Age=[10,20) and Race=AfricanAmerican, Age=[20, 50) and Race=Caucasian, etc.
        metric_maker: Callable
            Produce a metric to log from the Series we will create.
            For example, in plot_cohort_hist, what we want is the length of each dataframe.
            If not, we will log each row separately.
        """
        if not intersecting:
            # Simpler case: just go through each label provided.
            selections = [
                {cohort_category: cohort_value}
                for cohort_category in cohorts
                for cohort_value in cohorts[cohort_category]
            ]
        # More complex: go through each combination of attributes.
        else:
            if cohorts:
                keys = cohorts.keys()
                # So if column "A" has attributes A_false and A_true, and B has 1, 2, 3, then
                # we will exhaust through all combinations of these attributes.
                selections = list(itertools.product(*[cohorts[key] for key in keys]))
                # Now we put them into a dictionary for ease of processing
                selections = [dict(zip(keys, s)) for s in selections]
            else:
                # If there are in fact no other attributes, get a list so we don't just skip logging entirely
                selections = [{}]
            if len(selections) > 100:
                logger.warning(
                    f"More than 100 cohort groups ({len(selections)}) were provided. This might take a while."
                )
        for selection in selections:
            attributes = base_attributes | selection
            selection_condition = (
                functools.reduce(operator.and_, (dataframe[k] == v for k, v in selection.items()))
                if selection
                else pd.Series([True] * len(dataframe), index=dataframe.index)  # Condition which always succeeds
            )
            metrics = dataframe[selection_condition]
            if metric_maker is not None:
                self.populate_metrics(attributes=attributes, metrics=metric_maker(metrics))
            else:
                self.populate_metrics(attributes=attributes, metrics=metrics)

    def log_by_column(
        self, df: pd.DataFrame, col_name: str, cohorts: dict, base_attributes: dict, col_values: list = []
    ):
        """Log with a particular column as the index, only if each metric is set to log_all.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe the data resides in.
        col_name : str
            Which column we want the index to be.
        cohorts : dict
            Which cohorts we want to extract data from when it's time.
        base_attributes : dict
            The attributes we want to associate to all of the logs from this.
        col_values: list
            If log_all is not set for a particular metric, we will only log using the particular values provided. An
            example use case is a set of thresholds.
        """
        for metric in df.columns:
            if metric == col_name:
                continue

            def maker(frame):
                return frame.set_index(col_name)[[metric]].to_dict()

            log_all = get_metric_config(metric)["log_all"]
            if log_all or col_values != []:
                log_df = df if log_all else df[df[col_name].isin(col_values)]
                self.log_by_cohort(
                    dataframe=log_df, base_attributes=base_attributes, cohorts=cohorts, metric_maker=maker
                )
