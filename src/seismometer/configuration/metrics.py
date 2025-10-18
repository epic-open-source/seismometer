from typing import Dict, List, Union

from pydantic import BaseModel, field_validator


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
    quantiles : Union[int, List[float]], optional
        Quantiles for this metric. If int, evenly spread quantiles (e.g., 4 -> [0.25, 0.5, 0.75]).
        If list, use the values directly. Default is 4.
    measurement_type : str, optional
        Type of measurement (e.g., 'Gauge'). Default is 'Gauge'.
    """

    output_metrics: bool = True
    log_all: bool = False
    quantiles: Union[int, List[float]] = [0.25, 0.5, 0.75]  # Default to transformed list
    measurement_type: str = "Gauge"

    @field_validator("quantiles")
    @classmethod
    def transform_quantiles(cls, v: Union[int, List[float]]) -> List[float]:
        """
        Transform quantiles configuration to a list of float values.

        If int: Generate evenly spaced quantiles (e.g., 4 -> [0.25, 0.5, 0.75])
        If list: Use values directly

        Parameters
        ----------
        v : Union[int, List[float]]
            The quantiles configuration value

        Returns
        -------
        List[float]
            List of quantile values between 0 and 1
        """
        if isinstance(v, int):
            if v <= 0:
                raise ValueError("Number of quantiles must be positive")
            # Generate evenly spaced quantiles: 1/n, 2/n, ..., (n-1)/n
            return [i / v for i in range(1, v)]
        elif isinstance(v, list):
            # Validate that all values are between 0 and 1
            for q in v:
                if not (0 < q < 1):
                    raise ValueError(f"Quantile values must be between 0 and 1, got {q}")
            return sorted(v)  # Sort to ensure consistent ordering
        else:
            raise ValueError(f"Quantiles must be int or list of float, got {type(v)}")


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
