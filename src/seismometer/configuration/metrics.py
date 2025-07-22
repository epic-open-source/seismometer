from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class SingleMetricConfig:
    """The settings for outputting of a single metric."""

    output_metrics: bool = True
    log_all: bool = False
    granularity: int = 4
    measurement_type: str = "Gauge"


class MetricConfig(BaseModel):
    """The global settings, so that we can load the
    settings for all metrics at once.
    """

    metric_configs: dict[str, SingleMetricConfig]

    def __init__(self, **kwargs):
        """Populate the metric information, by type of metric, from the provided YAML dict.

        Parameters
        ----------
        kwargs : dict
            The section of YAML with the information we want to read.
        """
        super().__init__(metric_configs={})
        for metric_name in kwargs.keys():
            self.metric_configs[metric_name] = SingleMetricConfig(**kwargs[metric_name])
