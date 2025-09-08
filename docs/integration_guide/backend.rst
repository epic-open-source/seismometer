=================
Metric Collection
=================

Seismometer is already configured to export metrics to the OpenTelemetry collector, from where metrics can be exported
to backends (like Prometheus and Grafana).

In the ``config.yml`` file, the ``otel_export:`` section configures where metrics should go.

.. code-block:: yaml

  # config.yml
  otel_export:
    stdout: true
    hostname: otel-collector
    ports: port1, port2, port3
    files: output/metrics.json

There are four sets of values here: ``stdout`` determines whether the metrics should be printed out, ``files`` names the
files where JSON should be dumped (omit if no files), ``ports`` lists the ports that metrics should be exported over to an
OpenTelemetry collector (omit if no ports), and ``hostname`` specifies what hostname said ports are under.

A simple example will follow, exporting over a single port to an OpenTelemetry collector instance.

===============
Getting Started
===============


Here is an example of configuring all three components -- Collector, Prometheus, and Grafana -- for metric visualization.

This requires ``docker-compose``.

Make an empty folder with the following files:

.. code-block:: yaml

  # docker-compose.yml
  services:
    prometheus:
      image: prom/prometheus
      container_name: prometheus
      ports:
        - "9090:9090"
      volumes:
        - ./prometheus.yml:/etc/prometheus/prometheus.yml
      networks:
        - metricsnetwork
    grafana:
      image: grafana/grafana
      container_name: grafana
      ports:
        - "4000:3000"
      environment:
        - GF_SECURITY_ADMIN_PASSWORD=admin
      volumes:
        - grafana-storage:/var/lib/grafana
      networks:
        - metricsnetwork
    otel-collector:
      image: otel/opentelemetry-collector-contrib
      container_name: otel-collector
      command: ["--config=/etc/otel-collector-config.yml"]
      volumes:
        - ./otel-collector-config.yml:/etc/otel-collector-config.yml
      ports:
        - "4317:4317"
        - "4318:4318"
        - "9464:9464"
      networks:
        - metricsnetwork

  networks:
    metricsnetwork:
      external: true


  volumes:
    grafana-storage:


.. code-block:: yaml

  # prometheus.yml
  global:
    scrape_interval: 15s  # Frequency of metric scraping

  scrape_configs:
    - job_name: 'otel-collector'
      static_configs:
        - targets: ['otel-collector:9464']

.. code-block:: yaml

  # otel-collector-config.yml
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317
        http:
          endpoint: 0.0.0.0:4318

  exporters:
    prometheus:
      endpoint: "0.0.0.0:9464"

  service:
    pipelines:
      metrics:
        receivers: [otlp]
        exporters: [prometheus]

This will configure a Docker container running three services.

Before starting to run this (see below), make sure you have your own
environment configured to export metrics correctly. Using Docker, make
sure your ``docker-compose.yml`` file (like the commented-out lines in
the ``seismometer`` repository on GitHub) has the following section to
tap into the shared network which these three services interact on:

.. code-block:: yaml

  # docker-compose.yml
  services:
    my-seismometer-use-case:
      # whatever other setup you need
      # For communication with a backend
      networks:
        - metricsnetwork

  networks:
    metricsnetwork:
      external: true

Before starting either Docker container, run the command
``docker network create metricsnetwork``. This will actually make the
network for passing metric information around.

Now when in your Docker container, seismometer will output metrics to
the OpenTelemetry collector, which will send it eventually to Grafana --
as long as you have the collector and backends running in the first place.

Make sure your `config.yml`, in the `otel_export:` section, has the following:

.. code-block:: yaml

  ports:
    4317

Start the instances with ``docker-compose up -d`` and then visit
``localhost:4000`` to use Grafana. Log in with ``admin/admin``
username/password to explore metrics. For example, a dashboard with a panel
whose sole query is a metric set to just ``accuracy`` will scrape all datapoints
which quantify accuracy from the exported metrics.

=================
Metric Automation
=================

Metric exporting and collection can also be automated. Call history is saved per
notebook run, and can then be exported to disk for future runs.

To view the current call history:

.. code-block:: python

  sm.preview_automation()

To do an export of the current call settings:

.. code-block:: python
  
  sm.export_config()

This will export to the path given in ``config.yml`` under ``other_info: automation_config:``.
The parameter ``overwrite_existing`` specifies whether existing config will be overwritten if
the file is already populated.

Upon Seismogram load, any config stored in this path will be loaded. You may run an
automatic export based on the saved settings using

.. code-block:: python

  sm.export_automated_metrics()

Here is an example script to run in any seismograph, which will read the metrics in
``metric-automation.yml`` (or appropriate other path) and export them automatically:

.. code-block:: python

  import seismometer as sm
  sm.run_startup(config_path='.') # load config
  sm.export_automated_metrics()

In the future, we aim for a smoother CLI to do the same thing; for now, this short Python
script will suffice.
   