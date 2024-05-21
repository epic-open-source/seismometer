.. _roadmap:

=======
Roadmap
=======

Contributions
=============

As an open-source project, we welcome community contributions to ``seismometer``. 
Ultimately, we want this project to be a community-led effort to codify guidelines
for ensuring the equitable and informed use of machine learning and AI tools in the 
healthcare space. Contributions to this project can be as simple as fixing typos or 
small bugs, or more complex contributions that, with the support and scrutiny of our 
development team, guide the overall direction of the project.

.. seealso::
    :ref:`development` for our Contributor's Guide.

Use Cases
=========

Templates
---------

As of ``v0.1.0``, ``seismometer`` supports evaluating model performance using standardized evaluation 
criteria binary classifier models. We plan to add support in the near future for other types of 
machine learning models, such as multiple classifier models. Similarly, we plan to add 
support for validating generative AI models. These enhancements will include changes to
the underlying ``seismometer`` tooling, as well as adding new templates for validating
generative models. 

Workflows and Pre-Live Evaluation
---------------------------------

As of ``v0.1.0``, ``seismometer`` has limited support for evaluating model performance pre-live.
We are planning to add support for workflow simulation (e.g., estimating the number of 
alerts that would be shown to end-users for a clinical model that predicts an adverse 
event, or the amount of time saved per clinician for a generative model that drafts 
messages to patients) based on particular thresholds. We will also add tools to identify
thresholds for models based on pre-live data and operational goals. These tools are intended
to help identify when a machine learning or artificial intelligence solutions will improve
current workflows and also improve efficiency when integrating models into a workflow.

Comparing to Baselines
----------------------

We plan to add support for comparing model performance to baseline statistics (e.g., statistics 
from a model train or from model performance at a separate site). These are intended to verify 
that the model feature or target drift are not adversely affecting the model's performance after
it goes live.

Functional changes
==================

Visualizations
--------------

As ``seismometer`` grows, we will add support for new types of visualizations. Our initial focus 
is to improve visualizations for interventions and outcomes stratified by sensitive groups, but 
we plan to extend our model performance visualizations as well.

Data Layer
----------

As of ``v0.1.0``, ``seismometer`` supports reading data from `parquet` files, which contain data 
type information and performance improvements that standard CSV data does not have. We plan to add
support for more file formats (alongside metadata files that will describe the data types) as well
as support for reading data directly from a database (e.g., through an ODBC connection).

Code Structure
--------------

As we gear up for ``seismometer``'s version 1.0 release, we will be working on finalizing the internals
of the tool. Prior to the version 1.0 release, we expect there will be breaking changes to APIs, after
which the goal will be to minimize those breaking changes and only release breaking changes alongside
a major version bump.

.. seealso::
    :ref:`release` for our Release Notes and any breaking changes.