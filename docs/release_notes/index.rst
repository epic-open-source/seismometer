.. _release:

Changelog
=========

This is a list of changes that have been made between releases of the ``seismometer`` library. See GitHub for full details.

Breaking changes may occur between minor versions prior to the v1 release; afterwhich API changes will be restricted to major version updates.

.. towncrier release notes start

0.4.0
-----

Features
~~~~~~~~

- Implements ``ExploreAnalyticsTable`` to compare model performance across multiple model scores/targets. (`#62 <https://github.com/epic-open-source/seismometer/issues/62>`__)
- Implements ``ExploreOrdinalMetrics`` and ``ExploreCohortOrdinalMetrics`` to analyze feedback/categorical data. (`#98 <https://github.com/epic-open-source/seismometer/issues/98>`__)
- Adds cohort selection to analytics table options. (`#131 <https://github.com/epic-open-source/seismometer/issues/131>`__)


Bugfixes
~~~~~~~~

- Initialize Seismogram values to prevent ``AttributeErrors`` like ``available_cohort_group``. (`#120 <https://github.com/epic-open-source/seismometer/issues/120>`__)
- Fix a scaling issue causing annotations to be misplaced on the ROC plot of the evaluation 2x3. (`#124 <https://github.com/epic-open-source/seismometer/issues/124>`__)
- Fix an issue with reading parquet files without ``Pandas`` metadata (e.g., those created by ``Polars``). (`#134 <https://github.com/epic-open-source/seismometer/issues/134>`__)
- Make number of positives consistent when scores are combined in ``ExploreModelEvaluation`` and ``show_cohort_summaries``. (`#138 <https://github.com/epic-open-source/seismometer/issues/138>`__)
- Improve performance of combining scores. (`#138 <https://github.com/epic-open-source/seismometer/issues/138>`__)
- Updating ``cohort_list_details`` to only add data corresponding to ``context_id`` if it is not ``None``. (`#138 <https://github.com/epic-open-source/seismometer/issues/138>`__)
- Fixed a bug in ``generate_analytics_data`` where ``per_context=True`` caused data contamination across iterations due to in-place modification of the input DataFrame. (`#147 <https://github.com/epic-open-source/seismometer/issues/147>`__)
- Limit ``pyarrow`` version to ``>14.0.0,<20.0.0`` as ``use_legacy_dataset`` keyword is removed in later versions. (`#150 <https://github.com/epic-open-source/seismometer/issues/150>`__)
- Aligned empty filter dictionary handling across ``ExploreAnalyticsTable`` and ``ExploreModelEvaluation``. (`#151 <https://github.com/epic-open-source/seismometer/issues/151>`__)
- Improve threshold precision handling and formatting in analytics tables and slider widgets for more accurate AUC computation. (`#153 <https://github.com/epic-open-source/seismometer/issues/153>`__)


Improved Documentation
~~~~~~~~~~~~~~~~~~~~~~

- Updated ``integration_guide/index.rst`` to add a table documenting examples from the Seismometer Community of open source tools that are compatible/integrated with Seismometer. (`#74 <https://github.com/epic-open-source/seismometer/issues/74>`__)


Misc
~~~~

- `#152 <https://github.com/epic-open-source/seismometer/issues/152>`__, `#155 <https://github.com/epic-open-source/seismometer/issues/155>`__


0.3.0
------

Features
~~~~~~~~

- Addresses #77 by removing Aequitas and replacing with a great_tables based fairness audit. (`#77 <https://github.com/epic-open-source/seismometer/issues/77>`__)
- Added ExploreBinaryModelMetrics to see plots of individual metrics, including number needed to treat. (`#86 <https://github.com/epic-open-source/seismometer/issues/86>`__)
- Reorganize methods making initial import and public api more standard (`#102 <https://github.com/epic-open-source/seismometer/issues/102>`__)
- Includes confusion matrix rates into Binary Fairness metrics (`#108 <https://github.com/epic-open-source/seismometer/issues/108>`__)
- Renames `Flagged` to `Flag Rate` for clarity (`#108 <https://github.com/epic-open-source/seismometer/issues/108>`__)
- Add function to load example datasets (`#113 <https://github.com/epic-open-source/seismometer/issues/113>`__)


Bugfixes
~~~~~~~~

- Remove remaining references to -1 'invalidation'; validate directly on time comparison when needed (`#100 <https://github.com/epic-open-source/seismometer/issues/100>`__)
- Fixes scaling issue for binary classfier scores that use the range 0-100 rather than 0-1. (`#101 <https://github.com/epic-open-source/seismometer/issues/101>`__)
- Fixes a few minor ux issues. (`#109 <https://github.com/epic-open-source/seismometer/issues/109>`__)

  - Disable dropdowns with only one valid option.
  - Fix the Sensitivity/Specificity/PPV plot to move the label to the lower right.
  - Fix the Legend in the new Fairness Audit table to improve readability.
  - Add right border to the count column.
- Remove NotebookHost class that was no longer in use. (`#114 <https://github.com/epic-open-source/seismometer/issues/114>`__)


0.2.2
-----

Features
~~~~~~~~

- Added additional "merge_strategy" event configuration option. (`#76 <https://github.com/epic-open-source/seismometer/issues/76>`__)
- Viable merge strategies are "first", "last", "nearest", "forward", and "count". (`#76 <https://github.com/epic-open-source/seismometer/issues/76>`__)
- Added ExploreSubgroups as a drop in replacement for sm.cohort_list (`#82 <https://github.com/epic-open-source/seismometer/issues/82>`__)
- Added ExploreModelScoreComparison to compare two scores againts a shared target (`#82 <https://github.com/epic-open-source/seismometer/issues/82>`__)
- Added ExploreModelTargetComparison to compare a single score across two targets (`#82 <https://github.com/epic-open-source/seismometer/issues/82>`__)
- Added MultiselectDropdownWidget as a new widget for selecting cohort_dicts, uses a drop down and dismissalbe tags to keep the UX neater. (`#82 <https://github.com/epic-open-source/seismometer/issues/82>`__)
- Updated handling around `primary_output` and `outputs`, so that if primary_output is in outputs, it does not get added in again during startup. (`#82 <https://github.com/epic-open-source/seismometer/issues/82>`__)
- Cast features to dtypes in the dictionary yml when specified in the dictionary configuration. (`#92 <https://github.com/epic-open-source/seismometer/issues/92>`__)
- Cast event _Values to dtypes in the dictionary yml when specified in the dictionary configuration, done after imputation. (`#92 <https://github.com/epic-open-source/seismometer/issues/92>`__)
- Add a configuration helper that can generate a dictionary file for events or predictions. (`#93 <https://github.com/epic-open-source/seismometer/issues/93>`__)


Bugfixes
~~~~~~~~

- Limits control max-widths to 1200px in most cases, allowing row wrap when needed. (`#81 <https://github.com/epic-open-source/seismometer/issues/81>`__)
- Exclude time columns from ydata-profiling. (`#88 <https://github.com/epic-open-source/seismometer/issues/88>`__)
- Temporarily remove CLI - variation in templates is not stable enough to ensure robustness. (`#90 <https://github.com/epic-open-source/seismometer/issues/90>`__)


0.2.1
-----

Features
~~~~~~~~

- Add number needed to treat (NNT) with fixed rho=1/3 and net benefit, from med_metrics package. (`#78 <https://github.com/epic-open-source/seismometer/issues/78>`__)


Bugfixes
~~~~~~~~

- Warning message on censored cohorts is coerced to string before logging. (`#65 <https://github.com/epic-open-source/seismometer/issues/65>`__)
- Log a warning if cohort source column is not found in the dataframe. (`#67 <https://github.com/epic-open-source/seismometer/issues/67>`__)
- Improved support for datasets with a large number of cohort columns. Allowing row wrapping of cohort selection. (`#70 <https://github.com/epic-open-source/seismometer/issues/70>`__)
- Improved the Fairness audit iframe support by increasing the height a bit to account for cohort error messages. (`#70 <https://github.com/epic-open-source/seismometer/issues/70>`__)
- Allow long filenames by hashing the ends, this fix allow large cohort selection lists when creating reports and fairness audits which are cached to disk. (`#70 <https://github.com/epic-open-source/seismometer/issues/70>`__)
- Handle merging without context id; assumes events dataframe columns. (`#71 <https://github.com/epic-open-source/seismometer/issues/71>`__)
- Simplify merge logic for the single 'first' strategy. This removes coalecsing logic by assuming an event type always has times (or never does). (`#72 <https://github.com/epic-open-source/seismometer/issues/72>`__)
- Hardens restriction of events (with times) to occur after prediction time + window offset, not having the unintuitive partial information for early, late, and unknown timings. (`#72 <https://github.com/epic-open-source/seismometer/issues/72>`__)


Improved Documentation
~~~~~~~~~~~~~~~~~~~~~~

- Updated integration_guide/index.rst to add a table documenting examples from the Seismometer Community of open source tools that are compatible/integrated with Seismometer. (`#74 <https://github.com/epic-open-source/seismometer/issues/74>`__)


0.2.0
-----

Features
~~~~~~~~

- Changed box and whisker plots to violin plots (`#36 <https://github.com/epic-open-source/seismometer/issues/36>`__)
- Added Exploration controls to allow setting dynamic values for thresholds, targets, outcomes, etc. (`#41 <https://github.com/epic-open-source/seismometer/issues/41>`__)
- Add an optional aggregation_method to event objects of usage_config. (`#48 <https://github.com/epic-open-source/seismometer/issues/48>`__)
- Modify the accessors of ConfigProvider.events to return a dictionaries of events instead of a list. (`#48 <https://github.com/epic-open-source/seismometer/issues/48>`__)
- Update model and cohort performance plots to respect the aggregation_method: supports max (default), min, first and last. (`#48 <https://github.com/epic-open-source/seismometer/issues/48>`__)
- Configuration now supports multiple event types as the source for a single event. Values for the source events are assumed to be compatible, like both being Boolean. (`#54 <https://github.com/epic-open-source/seismometer/issues/54>`__)
- Updates the Seismogram class constructor to take precisely a DataLoader and a ConfigProvider, improving separation between the class and configuration loading. (`#56 <https://github.com/epic-open-source/seismometer/issues/56>`__)
- Adds a post-load hook for modifying the otherwise ready dataframe.  This hook is not accessible via normal run_startup, and requires direct initialization of Seismogram. (`#56 <https://github.com/epic-open-source/seismometer/issues/56>`__)


Bugfixes
~~~~~~~~

- Fixes issue with IFrames not being displayed in vscode. (`#53 <https://github.com/epic-open-source/seismometer/issues/53>`__)
- Fixes issue with ipywidgets being hidden in sphinx docs. (`#55 <https://github.com/epic-open-source/seismometer/issues/55>`__)
- Fixes issue with iframe sources not being found in sphinx docs. (`#55 <https://github.com/epic-open-source/seismometer/issues/55>`__)
- Update "Generate Report" button width for Cohort Comparison widget. (`#58 <https://github.com/epic-open-source/seismometer/issues/58>`__)
- Update various plot titles to `<h4>` headings for consistent size/theming. (`#58 <https://github.com/epic-open-source/seismometer/issues/58>`__)
- New CensoredResultException, for when a plot cannot render due to too few results. (`#58 <https://github.com/epic-open-source/seismometer/issues/58>`__)

Improved Documentation
~~~~~~~~~~~~~~~~~~~~~~

- Added a developer guide section for custom visualizations. (`#64 <https://github.com/epic-open-source/seismometer/issues/64>`__)
- Updated docstrings for exploration widgets and improved sphinx documentation layout. (`#64 <https://github.com/epic-open-source/seismometer/issues/64>`__)

0.1.1
-----

Features
~~~~~~~~

- Added `FilterRule.all()` and `FilterRule.none()` class methods for matching all or no rows of a dataframe. (`#27 <https://github.com/epic-open-source/seismometer/issues/27>`__)
- Updated plots to use HTML and SVG over pngs from matplotlib (`#28 <https://github.com/epic-open-source/seismometer/issues/28>`__)
- Added DiskCachedFunction to allow disk caching of HTML content (`#28 <https://github.com/epic-open-source/seismometer/issues/28>`__)
- Restructures Seismogram load to allow prioritizing in memory dataframe over loading predictions and/or events. (`#20 <https://github.com/epic-open-source/seismometer/issues/20>`__)
- seismometer.run_startup() can now accept preloaded prediction and event dataframes that take precendence over loading from configuration (`#34 <https://github.com/epic-open-source/seismometer/issues/34>`__)


Bugfixes
~~~~~~~~

- Fixes the header of sm.show_info() to start the table zebra stripe on the right row. (`#24 <https://github.com/epic-open-source/seismometer/issues/24>`__)
- Updated the defaulting for `censor_threshold`` in `_plot_leadtime_enc` (`#35 <https://github.com/epic-open-source/seismometer/issues/35>`__)
- Take len of column list for count  (`#42 <https://github.com/epic-open-source/seismometer/issues/42>`__)


Improved Documentation
~~~~~~~~~~~~~~~~~~~~~~

- Added documentation to the Example Notebooks section. (`#8 <https://github.com/epic-open-source/seismometer/issues/8>`__)


v0.1.0
------

Initial release!
