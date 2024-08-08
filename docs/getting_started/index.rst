.. _getting_started:

===============
Getting Started
===============

``seismometer`` allows you to evaluate AI
model performance using standardized evaluation criteria that helps you
make decisions based on your own local data. ``seismometer`` helps you
validate a model's initial performance and continue to monitor its
performance over time.

Local validation of an AI model requires cross-referencing data about
patients (such as demographics and clinical outcomes) and model
performance (including inputs and outputs).

What is in Seismometer
======================

Jupyter Notebook templates
--------------------------

Open-source Python libraries and Jupyter Notebooks allow data scientists
to quickly generate a notebook to evaluate model performance.

We expect these templates to continually evolve as new validation and
analysis techniques and approaches are created.

Data Schemas
------------

Standardized data schemas to incorporate data from single or multiple source systems.

Notebook configuration definitions
----------------------------------

A config file provides instructions for how to build the notebook using
the provided data and template. Within the file, you can control details
such as the cohorts to include and the outcomes relevant to the model.
You can also provide supplemental model documentation to give data
scientists and other report consumers working with the notebook
background on the model, definitions of terms and cohorts, and tips for
working with data in the notebook.

Install Seismometer
===================

From the Python Package Index (PyPI) install the package by running the
following at your command line:

.. code-block:: bash

   pip install seismometer

If you want to utilize fairness audit visualizations, run the following at
your command line:

.. code-block:: bash

   pip install seismometer[audit]

For additional details on installing Python packages, refer to the
`Python Packaging User
Guide <https://packaging.python.org/en/latest/tutorials/installing-packages/>`__.

Gather data
===========

Begin by gathering data on model inputs and outputs from production use
of a model on your local data. Set up a process for collecting data in
the following formats so that you can rerun the notebook for ongoing
monitoring of model performance.

Your model developer or health IT vendor might have instructions on how to gather this data. Refer to the :ref:`integration_guide` for more information.

Format
------

Save data for each of the following tables in the parquet format. This
can be done with Pandas `to_parquet
function <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_parquet.html>`__.

**Predictions**

Predictions data includes the model inputs, outputs, patient
demographics, and other relevant model features.

It is a wide table where each row represents an individual output on an
entity. Each prediction must include:

-  At least one entity identifier. You can optionally include additional
   context identifiers.

-  A primary output and timestamp for when the output was created.

You can optionally, include additional outputs and feature data as
needed.

The following is an example of a predictions table.

+------+--------+----------------+----------+---------+------------+------------+
|  ID  | Contact| Predict        | Risk     | Has     | Categorical| Categorical|
|      |   ID   | Time           | Of       | Previous| Feature1   | Feature... |
|      |        |                | Scoring  | Score   |            |            |
+======+========+================+==========+=========+============+============+
| 123  | 54321  | 1/1/2024       | 0.5      | 0       | Cat        | Cat        |
|      |        | 8:00:00        |          |         | Value1     | Value1     |
+------+--------+----------------+----------+---------+------------+------------+
| 123  | 54321  | 1/1/2024       | 0.99     | 1       | Cat        | Cat        |
|      |        | 8:15:00        |          |         | Value2     | Value2     |
+------+--------+----------------+----------+---------+------------+------------+

**Events**

Events data should be formatted in a long/narrow table design that
includes relevant interventions, ground-truth outcomes, and other
events.

Lines per entity must include:

-  At least one identifier that matches the identifier used in the
   predictions table. Include any additional context identifiers used in
   the Predictions table.

-  An event label/type column.

-  One or two additional nullable columns. The example below shows an event
   value and timestamp.

The ground truth for a model must be one of the outcome values.

The following is an example of an events table.

+------+-------------+------------------------+----------+-----------+
| ID   | ContactID   | Time                   | Type     | Value     |
+======+=============+========================+==========+===========+
| 123  | 54321       | 1/1/2024 10:36:25      | SEP-3    | true      |
+------+-------------+------------------------+----------+-----------+
| 123  | 54321       | 1/1/2024 10:45:02      | CDC      | true      |
+------+-------------+------------------------+----------+-----------+

Provide Configuration Files
===========================

The configuration file provides the instructions to the template for how
to build the Notebook with the provided data. It allows you to define
relevant cohorts including things like demographics, sensitive groups,
and other criteria for evaluating model performance. You can define
outcome and intervention events that relate to actions taken based on
model output and expected measurable results driven by those actions.

The configuration includes two core elements:

1. Data definitions to map columns in your data tables to the keys used
   in the Notebook template. This includes information on how data is
   used, including associating events to relevant predictions.

2. Supplemental documentation to give report consumers working in the
   Notebook background on the model, definitions of terms and cohorts,
   and tips for working with data in the notebook.

The model developer should provide much of the detail for the
configuration file, and you can modify it as needed to fit your
population and local workflows. Refer to the :ref:`integration_guide`
for more information.

For details on creating configuration files, refer to the User Guide.
