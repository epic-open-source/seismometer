
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
   leadtime_violin
   metric_vs_threshold
   performance_metrics
   ppv_vs_sensitivity
   recall_condition
   singleROC

Utility Functions
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   decorators.render_as_svg


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
   performance.calculate_nnt

Seismogram Loaders
~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   loader.ConfigOnlyHook
   loader.ConfigFrameHook
   loader.MergeFramesHook
   loader.SeismogramLoader
   loader.SeismogramLoader.load_data
   loader.loader_factory
   loader.event.parquet_loader
   loader.event.post_transform_fn
   loader.event.merge_onto_predictions
   loader.prediction.parquet_loader
   loader.prediction.assumed_types

Summaries
~~~~~~~~~
.. autosummary::
   :toctree: api/

   summaries.default_cohort_summaries
   summaries.score_target_cohort_summaries


Low-level patterns
------------------

Patterns
~~~~~~~~
.. currentmodule:: seismometer.core.patterns
.. autosummary::
   :toctree: api/

   Singleton

Decorators
~~~~~~~~~~
.. currentmodule:: seismometer.core.decorators
.. autosummary::
   :toctree: api/

   DiskCachedFunction
