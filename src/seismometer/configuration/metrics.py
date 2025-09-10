from typing import Dict

from pydantic import BaseModel


class SingleMetricConfig(BaseModel):
    """
    Configuration for a single metric.

    Parameters
    ----------
    output_metrics : bool, optional
        Whether to output metrics for this metric. Default is True.
    log_all : bool, optional
        Whether to log all values for this metric, rather than those at specified value (usually thresholds).
        Default is False.
    quantiles : int, optional
        Number of quantiles to use for this metric. Default is 4.
    measurement_type : str, optional
        Type of measurement (e.g., 'Gauge'). Default is 'Gauge'.
    """

    output_metrics: bool = True
    log_all: bool = False
    quantiles: int = 4
    measurement_type: str = "Gauge"


class MetricConfig(BaseModel):
    """
    Container for multiple metric configurations.

    Parameters
    ----------
    metric_configs : dict[str, SingleMetricConfig]
        Dictionary mapping metric names to their configurations.
    """

    metric_configs: Dict[str, SingleMetricConfig]

    def __contains__(self, key: str) -> bool:
        """
        Check if a metric name exists in the configuration.

        Parameters
        ----------
        key : str
            The metric name to check.

        Returns
        -------
        bool
            True if the metric exists, False otherwise.
        """
        return key in self.metric_configs

    def __getitem__(self, key: str) -> SingleMetricConfig:
        """
        Get the configuration for a metric by name, or return a default config if not found.

        Parameters
        ----------
        key : str
            The metric name to retrieve.

        Returns
        -------
        SingleMetricConfig
            The configuration for the metric, or a default config.
        """
        return self.metric_configs.get(key, SingleMetricConfig())
