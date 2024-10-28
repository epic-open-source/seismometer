import numpy as np
import pandas as pd
import pytest

import seismometer.data.performance as undertest

ALL_STATS = [undertest.THRESHOLD] + undertest.STATNAMES + ["NNT@0.333"]


def stats_case_base():
    y_true = np.array([1, 1, 0])
    y_prob = np.array([0.1, 0.5, 0.1])

    expected = []
    # Threshold, TP, FP, TN, FN, Accuracy, Sensitivity, Specificity, PPV, NPV, Flagged, LR+, NBS, NNT1/3 |  Threshold
    expected.append([100, 0, 0, 1, 2, 1 / 3, 0, 1, 0, 1 / 3, 0, np.nan, np.nan, np.inf])  # 1
    expected.append([50, 1, 0, 1, 1, 2 / 3, 0.5, 1, 1, 0.5, 1 / 3, np.inf, 1 / 3, 3])  # .5
    expected.append([10, 2, 1, 0, 0, 2 / 3, 1, 0, 2 / 3, 1, 1, 1, 17 / 27, 4.5])  # .1
    expected.append([0, 2, 1, 0, 0, 2 / 3, 1, 0, 2 / 3, 1, 1, 1, 2 / 3, 4.5])  # 0

    return (y_true, y_prob, expected)


def stats_case_0():
    "include prediction of exactly 0"
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0, 0.1, 0.5, 0.1])

    expected = []
    # Threshold, TP, FP, TN, FN, Accuracy, Sensitivity, Specificity, PPV, NPV, Flagged, LR+, NBS, NNT1/3|  Threshold
    expected.append([100, 0, 0, 2, 2, 0.5, 0, 1, 0, 0.5, 0, np.nan, np.nan, np.inf])  # 1
    expected.append([50, 1, 0, 2, 1, 0.75, 0.5, 1, 1, 2 / 3, 0.25, np.inf, 1 / 4, 3])  # .5
    expected.append([10, 2, 1, 1, 0, 0.75, 1, 0.5, 2 / 3, 1, 0.75, 2, 17 / 36, 4.5])  # .1
    expected.append([0, 2, 2, 0, 0, 0.5, 1, 0, 0.5, 1, 1, 1, 1 / 2, 6])  # 0

    return y_true, y_prob, expected


def stats_case_1():
    "include prediction of exactly 1"
    y_true = np.array([1, 1, 0, 1])
    y_prob = np.array([0.1, 0.5, 0.1, 1])

    expected = []
    # Threshold, TP, FP, TN, FN, Accuracy, Sensitivity, Specificity, PPV, NPV, Flagged, LR+, NBS, NNT1/3 |  Threshold
    expected.append([100, 1, 0, 1, 2, 0.5, 1 / 3, 1, 1, 1 / 3, 0.25, np.inf, np.nan, 3])  # 1
    expected.append([50, 2, 0, 1, 1, 0.75, 2 / 3, 1, 1, 0.5, 0.5, np.inf, 1 / 2, 3])  # .5
    expected.append([10, 3, 1, 0, 0, 0.75, 1, 0, 0.75, 1, 1, 1, 13 / 18, 4])  # .1
    expected.append([0, 3, 1, 0, 0, 0.75, 1, 0, 0.75, 1, 1, 1, 3 / 4, 4])  # 0

    return y_true, y_prob, expected


def stats_case_01():
    "include predictions of exactly 0 and 1"
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0, 0.1, 0.5, 0.1, 1])

    expected = []
    # Threshold, TP, FP, TN, FN, Accuracy, Sensitivity, Specificity, PPV, NPV, Flagged, LR+, NBS, NNT1/3 |  Threshold
    expected.append([100, 1, 0, 2, 2, 0.6, 1 / 3, 1, 1, 0.5, 0.2, np.inf, np.nan, 3])  # 1
    expected.append([50, 2, 0, 2, 1, 0.8, 2 / 3, 1, 1, 2 / 3, 0.4, np.inf, 2 / 5, 3])  # .5
    expected.append([10, 3, 1, 1, 0, 0.8, 1, 0.5, 0.75, 1, 0.8, 2, 26 / 45, 4])  # .1
    expected.append([0, 3, 2, 0, 0, 0.6, 1, 0, 0.6, 1, 1, 1, 3 / 5, 5])  # 0

    return y_true, y_prob, expected


def stats_case_0_4():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.75, 0.5, 0.25, 0])

    expected = []
    # Threshold, TP, FP, TN, FN, Accuracy, Sensitivity, Specificity, PPV, NPV, Flagged, LR+, NBS, NNT1/3 |  Threshold
    expected.append([100, 0, 0, 2, 2, 0.5, 0, 1, 0, 0.5, 0, np.nan, np.nan, np.inf])  # 1
    expected.append([75, 1, 0, 2, 1, 0.75, 0.5, 1, 1, 2 / 3, 0.25, np.inf, 1 / 4, 3])  # .75
    expected.append([50, 2, 0, 2, 0, 1, 1, 1, 1, 1, 0.5, np.inf, 1 / 2, 3])  # .5
    expected.append([25, 2, 1, 1, 0, 0.75, 1, 0.5, 2 / 3, 1, 0.75, 2, 5 / 12, 4.5])  # .25
    expected.append([0, 2, 2, 0, 0, 0.5, 1, 0, 0.5, 1, 1, 1, 1 / 2, 6])  # 0

    return (y_true, y_prob, expected)


@pytest.mark.parametrize(
    "y_true,y_prob,expected",
    [stats_case_base(), stats_case_0(), stats_case_1(), stats_case_01(), stats_case_0_4()],
    ids=["base", "0-pred", "1-pred", "1 and 0 preds", "0 preds"],
)
class Test_Stats:
    def test_stat_keys(self, y_true, y_prob, expected):
        """Ensure stat manipulations are intentional"""
        expected_keys = [
            "TP",
            "FP",
            "TN",
            "FN",
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
            "Flagged",
            "LR+",
            "NetBenefitScore",
        ]
        assert undertest.STATNAMES == expected_keys

    def test_score_arr(self, y_true, y_prob, expected):
        actual = undertest.calculate_bin_stats(y_true, y_prob, not_point_thresholds=True)
        assert ALL_STATS == list(actual)
        assert np.isclose(actual, expected, equal_nan=True).all()

    def test_score_arr_percentile(self, y_true, y_prob, expected):
        y_prob = y_prob * 100
        actual = undertest.calculate_bin_stats(y_true, y_prob, not_point_thresholds=True)
        assert ALL_STATS == list(actual)
        assert np.isclose(actual, expected, equal_nan=True).all()

    def test_score_with_y_proba_nulls(self, y_true, y_prob, expected):
        y_prob = np.hstack((y_prob, [np.nan, np.nan]))
        y_true = np.hstack((y_true, [0, 1]))
        actual = undertest.calculate_bin_stats(y_true, y_prob, not_point_thresholds=True)
        assert ALL_STATS == list(actual)
        assert np.isclose(actual, expected, equal_nan=True).all()

    def test_score_with_y_true_nulls(self, y_true, y_prob, expected):
        y_prob = np.hstack((y_prob, [0.1, 0.9]))
        y_true = np.hstack((y_true, [np.nan, np.nan]))
        actual = undertest.calculate_bin_stats(y_true, y_prob, not_point_thresholds=True)
        assert ALL_STATS == list(actual)
        assert np.isclose(actual, expected, equal_nan=True).all()


def test_bin_stats_point_thresholds():
    y_true, y_prob, base_expected = stats_case_base()
    actual = undertest.calculate_bin_stats(y_true, np.array(y_prob))

    expected = []
    for dup, row in zip([50, 40, 11], base_expected):
        expected.extend([row] * dup)
    expected = pd.DataFrame(expected, columns=ALL_STATS)
    expected.Threshold = np.arange(100, -1, -1)
    expected = expected.reset_index(drop=True)

    # Verify accuracy of threshold point-wise in above cases,
    # duplication doesn't apply
    threshold_dependent_columns = ["NetBenefitScore"]
    expected = expected.drop(columns=threshold_dependent_columns)
    actual = actual.drop(columns=threshold_dependent_columns)

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
