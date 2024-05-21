from unittest.mock import Mock

import numpy as np

from seismometer.data.confidence.parameters import (
    PerformanceMetricConfidenceParam,
    PRConfidenceParam,
    ROCConfidenceParam,
)

dummy_pred = np.arange(0, 1, 0.01)
dummy_pred = np.array([1 - dummy_pred, dummy_pred]).T
dummy_true = np.random.rand(dummy_pred.shape[0]) > 0.5


_interval_confidences = [ROCConfidenceParam, PRConfidenceParam]
_region_confidences = [ROCConfidenceParam, PerformanceMetricConfidenceParam]


class Test_Function_Config:
    """Tests if the correct functions are being called according to the configuration"""

    def test_custom_intervals(self):
        mock_method = Mock(return_value=np.random.rand(3))
        for object in _interval_confidences:
            config = {"interval": mock_method}
            conf = object(config)
            _ = conf.interval()
            mock_method.assert_called_once()
            mock_method.reset_mock()

    def test_custom_regions(self):
        mock_method = Mock(return_value=np.random.rand(3))
        for object in _region_confidences:
            config = {"region": mock_method}
            conf = object(config)
            _ = conf.region(conf, 0.5, 10)
            mock_method.assert_called_once()
            mock_method.reset_mock()
