===========
Seismometer
===========

Exploration APIs
~~~~~~~~~~~~~~~~

.. currentmodule:: seismometer._api

.. autosummary::
   :toctree: api/

   ExploreModelEvaluation
   ExploreCohortEvaluation
   ExploreCohortHistograms
   ExploreCohortLeadTime
   ExploreCohortOutcomeInterventionTimes
   ExploreFairnessAudit

Public API
~~~~~~~~~~~

.. currentmodule:: seismometer._api

.. autosummary::
   :toctree: api/

   cohort_evaluation
   cohort_comparison_report
   cohort_list
   fairness_audit
   feature_alerts
   feature_summary
   model_evaluation
   plot_cohort_hist
   plot_leadtime_enc
   plot_trend_intervention_outcome
   show_info
   show_cohort_summaries
   target_feature_summary


.. _custom-visualization-controls:

Custom Visualization Controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: seismometer.controls

.. autosummary::
   :toctree: api/


   explore.ExplorationWidget
   explore.ExplorationModelSubgroupEvaluationWidget
   explore.ExplorationCohortSubclassEvaluationWidget
   explore.ExplorationCohortOutcomeInterventionEvaluationWidget
