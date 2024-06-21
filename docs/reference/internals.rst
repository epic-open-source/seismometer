
.. _api.internals:

=========
Internals
=========

Command Line
------------

Extract
~~~~~~~
.. currentmodule:: seismometer.builder.extract
.. autosummary::
    :toctree: api/

    extract_supplement
    generate_data_dict_from_parquet

Build
~~~~~
.. currentmodule:: seismometer.builder.compile
.. autosummary::
    :toctree: api/

    compile_notebook

Other
~~~~~
.. currentmodule:: seismometer.builder
.. autosummary::
    :toctree: api/

    jupyter.contrib_cells
    jupyter.get_id
    jupyter.get_text


.. currentmodule:: seismometer.plot.mpl

Plotting
--------

Binary classification
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   calibration
   compare_series
   cohort_evaluation_vs_threshold
   cohorts_overlay
   cohorts_vertical
   evaluation
   histogram_stacked
   leadtime_whiskers
   metric_vs_threshold
   performance_metrics
   ppv_vs_sensitivity
   recall_condition
   singleROC
   

.. currentmodule:: seismometer.data

Data Manipulation
-----------------

Cohorts
~~~~~~~
.. autosummary::
   :toctree: api/

   cohorts.find_bin_edges
   cohorts.get_cohort_data
   cohorts.get_cohort_performance_data
   cohorts.has_good_binning
   cohorts.label_cohorts_categorical
   cohorts.label_cohorts_numeric
   cohorts.resolve_cohorts
   cohorts.resolve_col_data


Pandas Helpers
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   pandas_helpers.event_score
   pandas_helpers.event_time
   pandas_helpers.event_value
   pandas_helpers.infer_label
   pandas_helpers.merge_windowed_event
   pandas_helpers.valid_event

Performance
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   performance.as_percentages
   performance.as_probabilities
   performance.assert_valid_performance_metrics_df
   performance.calculate_bin_stats
   performance.calculate_eval_ci

Summaries
~~~~~~~~~
.. autosummary::
   :toctree: api/

   summaries.default_cohort_summaries
   summaries.score_target_cohort_summaries

.. currentmodule:: seismometer.core.patterns

Low-level patterns
------------------

Patterns
~~~~~~~~
.. autosummary::
   :toctree: api/

   Singleton
   DiskCachedFunction
