.. _release:

Changelog
=========

This is a list of changes that have been made between releases of the ``seismometer`` library. See GitHub for full details.

Breaking changes may occur between minor versions prior to the v1 release; afterwhich API changes will be restricted to major version updates.

.. towncrier release notes start

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
