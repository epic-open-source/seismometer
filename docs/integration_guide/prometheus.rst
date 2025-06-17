==========
Prometheus
==========

Seismometer is already configured to export metrics to Prometheus, from where a visualization backend like Grafana can be used to
observe metrics at a glance. Here is an example to set up a simple Prometheus instance to connect with Seismometer, running locally
on port ``9090``:

.. code-block:: yaml

   # docker-compose.yml
   version: '3.8'

   services:
     prometheus:
       image: prom/prometheus:v2.52.0
       container_name: prometheus
       ports:
         - "9090:9090"
       volumes:
         - ./prometheus.yml:/etc/prometheus/prometheus.yml
       networks:
         - monitoring
       restart: unless-stopped

   networks:
     monitoring:
       external: true

.. code-block:: yaml

   # prometheus.yml
   global:
   scrape_interval: 15s  # Frequency of metric scraping

   scrape_configs:
      - job_name: 'seismo'
      static_configs:
      - targets: ['172.19.0.1:9464']

Start the Prometheus instance with ``docker-compose up`` and then visit
``localhost:9090``.

   