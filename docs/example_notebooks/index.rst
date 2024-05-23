.. _examples:

Example Notebooks
=================

Directory Structure
-------------------

To run the notebook, we assume the existence of the following files in the same 
directory as the notebook: configuration files and the data directory.

Configuration Files
~~~~~~~~~~~~~~~~~~~

The following configuraion files are required to be in the same directory as the notebook: 

- ``config.yml``, 
- ``usage_config.yml``,
- ``data/metadata.json``.

.. seealso::
    :ref:`config-files` for more information on configuration files.


Data
~~~~

The first code cell in the notebook downloads the required data and data dictionary from 
`seismometer-data <https://github.com/epic-open-source/seismometer-data>`_
repository and creates the following files in the notebook directory:

- ``data/predictions.parquet``,
- ``data/events.parquet``,
- ``dictionary.yml``.

To use your own data, simply remove the first code cell and place your own data 
following the schema mentioned above.

.. nbgallery::
    notebooks/binary-classifier/classifier_bin
