import numpy as np
import pandas as pd
import pytest
from sklearn import metrics

from seismometer.data.confidence import (
    PerformanceMetricConfidenceParam,
    ROCConfidenceParam,
    agresti_coull_interval,
    hanley_mcneill_confidence,
    logit_interval,
    simultaneous_joint_confidence,
)
from seismometer.data.confidence.parameters import PerformanceMetricConfidenceParam

dummy_pred = np.arange(0, 1, 0.01)
dummy_pred = np.array([1 - dummy_pred, dummy_pred]).T
dummy_true = np.random.rand(dummy_pred.shape[0]) > 0.5


class Test_SJR:
    """Tests numeric properties of the simultaneous joint confidence interval"""

    def test_Series_input(self):
        proba = pd.Series(np.random.rand(100000))
        truth = pd.Series(flip_weighted_coin(proba))
        _, tpr, fpr, *_ = simultaneous_joint_confidence(ROCConfidenceParam(0.95), truth, proba)
        assert metrics.auc(fpr, tpr) > 0.158 and metrics.auc(fpr, tpr) < 0.173

    def test_allfalse_alltrue(self):
        proba = np.random.rand(100000)
        truth = np.zeros(proba.shape)
        _, tpr, _, region = simultaneous_joint_confidence(ROCConfidenceParam(0.95), truth, proba)
        assert pd.isna(tpr).all() and pd.isna(region.lower_tpr).all() and pd.isna(region.upper_tpr).all()
        truth = np.ones(proba.shape)
        _, _, fpr, region = simultaneous_joint_confidence(ROCConfidenceParam(0.95), truth, proba)
        assert pd.isna(fpr).all() and pd.isna(region.lower_fpr).all() and pd.isna(region.upper_fpr).all()


class Test_hanley_mcneill_confidence:
    """Tests numeric properties of the Hanley-McNeill interval"""

    @pytest.mark.parametrize("alpha", np.arange(0.05, 0.95, 0.05))
    def test_alpha(self, alpha):
        proba = np.random.rand(100000)
        truth = flip_weighted_coin(proba)
        result1 = hanley_mcneill_confidence(ROCConfidenceParam(alpha), truth, proba)
        result2 = hanley_mcneill_confidence(ROCConfidenceParam(alpha + 0.05), truth, proba)
        assert result1[0] == result2[0]
        assert result1[1] > result2[1]
        assert result1[2] < result2[2]


class Test_logit_interval:
    """Tests numeric properties of the logit interval"""

    def test_uniform_probability(self):
        assert logit_interval(PerformanceMetricConfidenceParam(0.95), 0.5, 100)[0] == 0.5

    def test_variance(self):
        ns = np.logspace(np.log(10), np.log(10000000)).astype("int64")
        print(ns)
        for i in range(len(ns) - 1):
            simple_variance_test(ns[i], ns[i + 1], logit_interval, PerformanceMetricConfidenceParam)

    def test_alpha(self):
        alphas = np.arange(0.05, 1, 0.05)
        for i in range(len(alphas) - 1):
            alpha_test(alphas[i], alphas[i + 1], logit_interval, PerformanceMetricConfidenceParam)


class Test_agresti_coull_interval:
    """Tests numeric properties of the Agresti-Coull interval"""

    def test_uniform_probability(self):
        assert agresti_coull_interval(PerformanceMetricConfidenceParam(0.95), 0.5, 100)[0] == 0.5

    def test_variance(self):
        ns = np.logspace(np.log(10), np.log(10000000)).astype("int64")
        for i in range(len(ns) - 1):
            simple_variance_test(ns[i], ns[i + 1], agresti_coull_interval, PerformanceMetricConfidenceParam)

    def test_alpha(self):
        alphas = np.arange(0.05, 1, 0.05)
        for i in range(len(alphas) - 1):
            alpha_test(alphas[i], alphas[i + 1], agresti_coull_interval, PerformanceMetricConfidenceParam)


def alpha_test(level1, level2, my_calc_func, my_conf_type):
    assert level1 < level2
    conf1 = my_conf_type(level1)
    conf2 = my_conf_type(level2)
    result1 = my_calc_func(conf1, 0.5, 1000)
    result2 = my_calc_func(conf2, 0.5, 1000)
    assert result1[0] == result2[0]
    assert result1[1] > result2[1]
    assert result1[2] < result2[2]


def simple_variance_test(n1, n2, my_calc_func, my_conf_type):
    assert n1 < n2
    conf = my_conf_type(0.95)
    result1 = my_calc_func(conf, 0.5, n1)
    result2 = my_calc_func(conf, 0.5, n2)
    assert result1[0] == result2[0]
    assert result1[1] < result2[1]
    assert result1[2] > result2[2]


def flip_weighted_coin(proba):
    flip = np.random.rand(proba.shape[0])
    return proba < flip
