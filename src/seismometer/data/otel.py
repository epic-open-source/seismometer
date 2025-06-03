import sys
from typing import List

import pandas as pd
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, Gauge, PeriodicExportingMetricReader

from seismometer.core.io import slugify


class OpenTelemetryRecorder:
    def __init__(self, metric_names: List[str], output_path: str = None):
        """_summary_

        Parameters
        ----------
        metric_names : List[str]
            These are the kinds of metrics we want to be collecting. E.g. fairness, accuracy.
        output : str, optional
            This is the file path where outputted metrics should be dumped. For now, we are just dealing
            with file path, but further support for OTel exporters will be added in the future.
            Leave this blank for output to be stdout -- e.g. dumped to the console.S
        """

        # The IO object to write OTel data into.
        self.otlp_exhaust: sys.IO
        if output_path is not None:
            self.otlp_exhaust = open(output_path, "r")
            self.output_file_path = output_path  # Store for closing later
        else:
            self.otlp_exhaust = sys.stdout
            self.output_file_path = None
        reader = PeriodicExportingMetricReader(ConsoleMetricExporter(out=self.otlp_exhaust))
        meter_provider = MeterProvider(metric_readers=[reader])
        # OpenTelemetry: use this new object to spawn new "Instruments" (measuring devices)
        self.meter = meter_provider.get_meter("Seismo-meter")
        # Keep it like this for now: just make an instrument for each metric we are measuring
        # TODO: worry about the type of each metric being measured
        # TODO: get better descriptions for each metric besides just the name
        # This is a map from metric name to corresponding instrument
        self.instruments: dict[str, Gauge] = {}
        self.metric_names = metric_names
        for mn in metric_names:
            self.instruments[mn] = self.meter.create_gauge(slugify(mn), description=mn)

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

        metrics : dict[str, float].
            The actual data we are populating.
        """
        if metrics is None:
            # metrics = self()
            raise Exception()
        for name in self.instruments.keys():
            # I think metrics.keys() is a subset of self.instruments.keys()
            # but I am not 100% on it. So this stays for now.
            if name in metrics:
                self._log_to_instrument(attributes, self.instruments[name], metrics[name])

    def _log_to_instrument(self, attributes, instrument: Gauge, data):
        """Write information to a single instrument. We need this
        wrapper function because the data we are logging may be a
        type such as a series, in which case we need to log each
        entry separately or at least do some extra preprocessing.

        Parameters
        ----------
        cohort_info : dict[str, tuple[Any]]
            Which cohort we are logging a measurement from.
        instrument : opentelemetry.sdk.metrics.Gauge
            The OpenTelemetry Gauge that we are recording a
            measurement to. Should be one of self.instruments.
        data
            The data we are recording. Could conceivably be either
            a numeric value or a series.
        """

        def set_one_datapoint(value):
            instrument.set(value, attributes=attributes)

        if isinstance(data, (int, float)):
            set_one_datapoint(data)
        elif isinstance(data, pd.Series):
            for k, v in data.items():
                set_one_datapoint(v)
        else:
            raise Exception(f"Unrecognized data format for OTel logging: {type(data)}")

    def __del__(self):
        if self.output_file_path is not None:
            self.otlp_exhaust.close()
