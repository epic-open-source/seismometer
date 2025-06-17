.. _integration_guide:

=================
Integration Guide
=================

.. toctree::
    :hidden:
    :maxdepth: 1

    prometheus

This guide contains references to systems and packages that integrate with Seismometer. If you are interested in including your content here, either open a pull request or reach out to OpenSourceContributions-Python@epic.com

Vendor Specific Resources
=========================

Your model developer or health IT vendor might provide guidance on how to extract data from your system for use in ``seismometer``. The following table provides links to vendor specific integration instructions. Note that some links might only be accessible if you are licensed with the 
associated vendor.

+----------------------------+-----------------------------------------------+
| Vendor                     | Integration instructions                      |
+============================+===============================================+
| Epic                       | `Setup and Support Guide`_                    |
+----------------------------+-----------------------------------------------+

.. _Setup and Support Guide: https://galaxy.epic.com/Redirect.aspx?DocumentID=100277113        

Seismometer Community
=====================

You might also be interested in using ``seismometer`` with other open source tools that assess complementary facets of AI systems deployed in specific settings. The following table provides a list of such 
tools and links to relevant code/documentation.

+----------------------------+-------------------------------------------------------------+-----------------------------------------------+
| Package name               | Description                                                 | Link                                          |
+============================+=============================================================+===============================================+
| APLUSML                    | A Python Library for Usefulness Simulations of ML Models    | `Example notebook integrating seismometer`_   |
+----------------------------+-------------------------------------------------------------+-----------------------------------------------+
| Prometheus                 | A place for metrics to be collected for later visualization | `Integration of seismometer with Prometheus`_ |
+----------------------------+-------------------------------------------------------------+-----------------------------------------------+

.. _Example notebook integrating seismometer: https://github.com/som-shahlab/aplusml/blob/main/tutorials/synthetic_pad_seismometer.ipynb

.. _Integration of seismometer with Prometheus: :doc:`Prometheus instructions <prometheus>`
