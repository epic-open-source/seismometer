import pytest

from seismometer.data.confidence.calculations import (
    agresti_coull_interval,
    logit_interval,
    simultaneous_joint_confidence,
)
from seismometer.data.confidence.parameters import (
    _SUPPORTED_PLOTS,
    PerformanceMetricConfidenceParam,
    PRConfidenceParam,
    ROCConfidenceParam,
    confidence_dict,
)


class Test_Parameters:
    """Test the input validation, object instantiantion, and other expected behaviors"""

    @pytest.mark.parametrize(
        ("object", "input", "valid"),
        [
            [ROCConfidenceParam, {"level": 0.8}, True],
            [ROCConfidenceParam, {"level": 0}, False],
            [PRConfidenceParam, {"level": 0}, False],
            [PRConfidenceParam, {"region": simultaneous_joint_confidence}, False],
            [PRConfidenceParam, {"interval": agresti_coull_interval}, True],
            [PRConfidenceParam, {"level": 0.8, "interval": agresti_coull_interval}, True],
            [PRConfidenceParam, {"level": 1e-10, "interval": agresti_coull_interval}, True],
            [PRConfidenceParam, {"level": -1, "interval": agresti_coull_interval}, False],
            [PerformanceMetricConfidenceParam, {"interval": agresti_coull_interval}, False],
            [PerformanceMetricConfidenceParam, {"region": logit_interval}, True],
            [PerformanceMetricConfidenceParam, {"region": logit_interval, "some_random_param": 1}, True],
            [PerformanceMetricConfidenceParam, {"some_random_param": 1}, True],
            [PerformanceMetricConfidenceParam, 0.8, True],
            [PerformanceMetricConfidenceParam, 0, False],
            [PerformanceMetricConfidenceParam, -1, False],
            [PerformanceMetricConfidenceParam, 1.2, False],
            [PerformanceMetricConfidenceParam, None, True],
        ],
    )
    def test_input_validation(self, object, input, valid):
        try:
            _ = object(input)
        except ValueError:
            assert not valid
        else:
            assert valid


class Test_Confidence_Dict:
    def test_all_param_level(self):
        conf = confidence_dict({"all": 0.95})
        for plot in _SUPPORTED_PLOTS:
            assert "level" in conf[plot] and conf[plot]["level"] == 0.95

    def test_all_param_merge(self):
        conf = confidence_dict({"all": {"level": 0.95}})
        for plot in _SUPPORTED_PLOTS:
            assert "level" in conf[plot] and conf[plot]["level"] == 0.95

    def test_none(self):
        conf = confidence_dict(None)
        for plot in _SUPPORTED_PLOTS:
            assert conf[plot] is None
