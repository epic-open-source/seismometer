================
Metric Exporting
================

As OpenTelemetry has capabilities for exporting metrics from notebook runs,
much information about metrics may be configured. There are two main places that
metrics are configured:

- The automation settings (likely in ``metric-automation.yml``)
- The settings for which metrics to export (likely in ``usage-config.yml``).

===============================
Individual Metric Configuration
===============================

The ``otel_metric_override:`` section in the usage config file contains information about
what metrics we want to export from our plotting functions, in the ways that it differs
from the defaults given below:

.. code-block:: yaml

    otel_metric_override:
      Accuracy:
        output_metrics: true
        log_all: false
        quantiles: 4
        measurement_type: Gauge

A sequence of entries follows, each of which specifies a type of metric to be
logged (accuracy, specificity, etc). There are four settings for each metric:

- ``output_metrics``: whether metrics are to be output from this at all.
- ``log_all``: in some displays, an entire curve is plotted while only a few
  points are singled out (by threshold, for example). This options says whether
  to log the entire curve or just the singled-out points.
- ``quantiles``: for plots which display in quantiles, specifies how many quantiles
  to output in the metrics. (For instance, quartiles would be ``quantiles: 4``.)
- ``measurement_type``: specifies what the sort of data should be logged as. Default
  is ``Gauge`` (for individual data points), but also offered are ``Counter`` (for
  cumulative data) and ``Histogram`` (for data which is meant to be processed as a histogram).

The defaults for each setting when not provided are those given in the example.

=================
Metric Automation
=================

Metric automation can also be specified. A file with a name like
``metric-automation.yml`` (path specified in ``config.yml`` under ``other_info: automation_config:``)
will contain information about a series of calls to run. Upon loading a seismograph, calling
``sm.export_automated_metrics()`` will perform an export of all specified metrics.

Such a file looks like a list of plot functions and their arguments, i.e.

.. code-block:: yaml

    plot_cohort_evaluation:
    - cohorts:
        Age:
        - 70+
        - '[0-10)'
        - '[10-20)'
        - '[20-50)'
        - '[50-70)'
    options:
        per_context: false
        score_column: Risk30DayReadmission
        target_column: Readmitted within 30 Days
        thresholds:
        - 0.2
        - 0.1
    plot_cohort_lead_time:
    - cohorts:
        Age:
        - 70+
        - '[0-10)'
        - '[10-20)'
        - '[20-50)'
        - '[50-70)'
    options:
        event_column: Readmitted within 30 Days
        score_column: Risk30DayReadmission
        threshold: 0.1

The ``options`` section specifies function arguments besides cohorts,
and the cohorts themselves are specified in the ``cohorts`` section.

This is automatically loaded on startup; to export an automation file which
will do exactly what your current run has done so far, call ``sm.telemetry_config()``.
