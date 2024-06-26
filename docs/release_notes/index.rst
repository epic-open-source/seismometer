.. _release:

Changelog
=========

This is a list of changes that have been made between releases of the ``seismometer`` library. See GitHub for full details.

Breaking changes may occur between minor versions prior to the v1 release; afterwhich API changes will be restricted to major version updates.

.. towncrier release notes start

0.1.1
-----

Features
~~~~~~~~

- Added `FilterRule.all()` and `FilterRule.none()` class methods for matching all or no rows of a dataframe. (`#27 <https://github.com/epic-open-source/seismometer/issues/27>`__)
- Updated plots to use HTML and SVG over pngs from matplotlib (`#43 <https://github.com/epic-open-source/seismometer/issues/43>`__)
- Added DiskCachedFunction to allow disk caching of HTML content (`#43 <https://github.com/epic-open-source/seismometer/issues/43>`__)


Bugfixes
~~~~~~~~

- Fixes the header of sm.show_info() to start the table zebra stripe on the right row. (`#24 <https://github.com/epic-open-source/seismometer/issues/24>`__)


Improved Documentation
~~~~~~~~~~~~~~~~~~~~~~

- Added documentation to the Example Notebooks section. (`#8 <https://github.com/epic-open-source/seismometer/issues/8>`__)


v0.1.0
------

Initial release!
