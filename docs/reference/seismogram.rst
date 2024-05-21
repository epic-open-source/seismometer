.. currentmodule:: seismometer.seismogram

==========
Seismogram
==========

The Seismogram class is the data provider for a notebook session.

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Seismogram

Properties
~~~~~~~~~~
.. autosummary::
   :toctree: api/

    Seismogram.events
    Seismogram.events_columns
    Seismogram.target_event
    Seismogram.target
    Seismogram.time_zero
    Seismogram.output
    Seismogram.dataframe
    Seismogram.output_path
    Seismogram.censor_threshold
    Seismogram.prediction_count
    Seismogram.entity_count
    Seismogram.feature_count
    Seismogram.start_time
    Seismogram.end_time
    Seismogram.event_count
    Seismogram.event_types_count
    Seismogram.cohort_attribute_count
    Seismogram.score_bins
    Seismogram.data
   
Initialization
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Seismogram.prep_data
   Seismogram.add_events
   Seismogram.create_cohorts
