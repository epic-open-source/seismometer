import numpy as np
import pandas as pd
import pytest

import seismometer.data.performance as undertest


def stats_case_base():
    y_true = np.array([1, 1, 0])
    y_prob = np.array([0.1, 0.5, 0.1])

    expected = []
    # TP, FP, TN, FN, Threshold, Accuracy, Sensitivity, Specificity, PPV, NPV, Flagged, LR+ |  Threshold
    expected.append([0, 0, 1, 2, 100, 1 / 3, 0, 1, 0, 1 / 3, 0, np.nan])  # 1
    expected.append([1, 0, 1, 1, 50, 2 / 3, 0.5, 1, 1, 0.5, 1 / 3, np.inf])  # .5
    expected.append([2, 1, 0, 0, 10, 2 / 3, 1, 0, 2 / 3, 1, 1, 1])  # .1
    expected.append([2, 1, 0, 0, 0, 2 / 3, 1, 0, 2 / 3, 1, 1, 1])  # 0

    return (y_true, y_prob, expected)


def stats_case_0():
    "include prediction of exactly 0"
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0, 0.1, 0.5, 0.1])

    expected = []
    # TP, FP, TN, FN, Threshold, Accuracy, Sensitivity, Specificity, PPV, NPV, Flagged, LR+ |  Threshold
    expected.append([0, 0, 2, 2, 100, 0.5, 0, 1, 0, 0.5, 0, np.nan])  # 1
    expected.append([1, 0, 2, 1, 50, 0.75, 0.5, 1, 1, 2 / 3, 0.25, np.inf])  # .5
    expected.append([2, 1, 1, 0, 10, 0.75, 1, 0.5, 2 / 3, 1, 0.75, 2])  # .1
    expected.append([2, 2, 0, 0, 0, 0.5, 1, 0, 0.5, 1, 1, 1])  # 0

    return y_true, y_prob, expected


def stats_case_1():
    "include prediction of exactly 1"
    y_true = np.array([1, 1, 0, 1])
    y_prob = np.array([0.1, 0.5, 0.1, 1])

    expected = []
    # TP, FP, TN, FN, Threshold, Accuracy, Sensitivity, Specificity, PPV, NPV, Flagged, LR+ |  Threshold
    expected.append([1, 0, 1, 2, 100, 0.5, 1 / 3, 1, 1, 1 / 3, 0.25, np.inf])  # 1
    expected.append([2, 0, 1, 1, 50, 0.75, 2 / 3, 1, 1, 0.5, 0.5, np.inf])  # .5
    expected.append([3, 1, 0, 0, 10, 0.75, 1, 0, 0.75, 1, 1, 1])  # .1
    expected.append([3, 1, 0, 0, 0, 0.75, 1, 0, 0.75, 1, 1, 1])  # 0

    return y_true, y_prob, expected


def stats_case_01():
    "include predictions of exactly 0 and 1"
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0, 0.1, 0.5, 0.1, 1])

    expected = []
    # TP, FP, TN, FN, Threshold, Accuracy, Sensitivity, Specificity, PPV, NPV, Flagged, LR+ |  Threshold
    expected.append([1, 0, 2, 2, 100, 0.6, 1 / 3, 1, 1, 0.5, 0.2, np.inf])  # 1
    expected.append([2, 0, 2, 1, 50, 0.8, 2 / 3, 1, 1, 2 / 3, 0.4, np.inf])  # .5
    expected.append([3, 1, 1, 0, 10, 0.8, 1, 0.5, 0.75, 1, 0.8, 2])  # .1
    expected.append([3, 2, 0, 0, 0, 0.6, 1, 0, 0.6, 1, 1, 1])  # 0

    return y_true, y_prob, expected


def stats_case_0_4():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.75, 0.5, 0.25, 0])

    expected = []
    # TP, FP, TN, FN, Threshold, Accuracy, Sensitivity, Specificity, PPV, NPV, Flagged, LR+ |  Threshold
    expected.append([0, 0, 2, 2, 100, 0.5, 0, 1, 0, 0.5, 0, np.nan])  # 1
    expected.append([1, 0, 2, 1, 75, 0.75, 0.5, 1, 1, 2 / 3, 0.25, np.inf])  # .75
    expected.append([2, 0, 2, 0, 50, 1, 1, 1, 1, 1, 0.5, np.inf])  # .5
    expected.append([2, 1, 1, 0, 25, 0.75, 1, 0.5, 2 / 3, 1, 0.75, 2])  # .25
    expected.append([2, 2, 0, 0, 0, 0.5, 1, 0, 0.5, 1, 1, 1])  # 0

    return (y_true, y_prob, expected)


@pytest.mark.parametrize(
    "y_true,y_prob,expected",
    [stats_case_base(), stats_case_0(), stats_case_1(), stats_case_01(), stats_case_0_4()],
    ids=["base", "0-pred", "1-pred", "1 and 0 preds", "0 preds"],
)
class Test_Stats:
    def test_stat_keys(self, y_true, y_prob, expected):
        expected_keys = [
            "TP",
            "FP",
            "TN",
            "FN",
            "Threshold",
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
            "Flagged",
            "LR+",
        ]
        assert undertest.STATNAMES == expected_keys

    def test_score_arr(self, y_true, y_prob, expected):
        actual = undertest.calculate_bin_stats(y_true, y_prob, not_point_thresholds=True)
        assert undertest.STATNAMES == list(actual)
        assert np.isclose(actual, expected, equal_nan=True).all()

    def test_score_arr_percentile(self, y_true, y_prob, expected):
        y_prob = y_prob * 100
        actual = undertest.calculate_bin_stats(y_true, y_prob, not_point_thresholds=True)
        assert undertest.STATNAMES == list(actual)
        assert np.isclose(actual, expected, equal_nan=True).all()

    def test_score_with_y_proba_nulls(self, y_true, y_prob, expected):
        y_prob = np.hstack((y_prob, [np.nan, np.nan]))
        y_true = np.hstack((y_true, [0, 1]))
        actual = undertest.calculate_bin_stats(y_true, y_prob, not_point_thresholds=True)
        assert undertest.STATNAMES == list(actual)
        assert np.isclose(actual, expected, equal_nan=True).all()

    def test_score_with_y_true_nulls(self, y_true, y_prob, expected):
        y_prob = np.hstack((y_prob, [0.1, 0.9]))
        y_true = np.hstack((y_true, [np.nan, np.nan]))
        actual = undertest.calculate_bin_stats(y_true, y_prob, not_point_thresholds=True)
        assert undertest.STATNAMES == list(actual)
        assert np.isclose(actual, expected, equal_nan=True).all()


def test_bin_stats_point_thresholds():
    y_true, y_prob, base_expected = stats_case_base()
    actual = undertest.calculate_bin_stats(y_true, np.array(y_prob))

    expected = []
    for dup, row in zip([50, 40, 11], base_expected):
        expected.extend([row] * dup)
    expected = pd.DataFrame(expected, columns=undertest.STATNAMES)
    expected.Threshold = np.arange(100, -1, -1)
    expected = expected.reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def sumall(a, b):
    return a.sum() + b.sum()


def sumdiff(a, b):
    return (a - b).sum()


class Test_AsProbabilities:
    def test_convert(self):
        percentages = np.random.uniform(low=0, high=100, size=100)
        expected = percentages / 100
        assert np.array_equal(undertest.as_probabilities(percentages), expected)

    def test_already_in_range(self):
        percentages = np.random.uniform(low=0, high=1, size=100)
        assert np.array_equal(undertest.as_probabilities(percentages), percentages)


class Test_AsPercentages:
    def test_convert(self):
        percentages = np.random.uniform(low=0, high=1, size=100)
        expected = percentages * 100
        assert np.array_equal(undertest.as_percentages(percentages), expected)

    def test_already_in_range(self):
        percentages = np.random.uniform(low=0, high=100, size=100)
        assert np.array_equal(undertest.as_percentages(percentages), percentages)
