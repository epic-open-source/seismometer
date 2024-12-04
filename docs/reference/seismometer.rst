.. currentmodule:: seismometer.api

===========
Seismometer
===========

Exploration Widgets
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   ExploreSubgroups
   ExploreModelEvaluation
   ExploreCohortEvaluation
   ExploreCohortHistograms
   ExploreCohortLeadTime
   ExploreCohortOutcomeInterventionTimes
   ExploreFairnessAudit
   ExploreModelScoreComparison
   ExploreModelTargetComparison
   ExploreBinaryModelMetrics


Public APIs
~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   cohort_comparison_report
   cohort_list
   feature_alerts
   feature_summary
   model_evaluation
   plot_cohort_evaluation
   plot_cohort_hist
   plot_leadtime_enc
   plot_binary_classifier_metrics
   plot_trend_intervention_outcome
   show_info
   show_cohort_summaries
   target_feature_summary


.. _custom-visualization-controls:

Custom Visualization Controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   explore.ExplorationWidget
   explore.ExplorationSubpopulationWidget
   explore.ExplorationModelSubgroupEvaluationWidget
   explore.ExplorationCohortSubclassEvaluationWidget
   explore.ExplorationCohortOutcomeInterventionEvaluationWidget
   explore.ExplorationScoreComparisonByCohortWidget
   explore.ExplorationTargetComparisonByCohortWidget
   explore.ExplorationMetricWidget

Notebook and Configuration Helper APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: seismometer.api

.. autosummary::
   :toctree: api/

   download_example_dataset
   generate_dictionary_from_parquet
