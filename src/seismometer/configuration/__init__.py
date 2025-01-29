from .config import ConfigProvider
from .model import AggregationStrategies, MergeStrategies


class ConfigurationError(Exception):
    """Raised when a configuration error is detected."""

    pass
