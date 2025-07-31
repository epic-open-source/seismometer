import numpy as np
import pandas as pd
import pytest

import seismometer.data.performance as undertest

# Test scenarios - each is a tuple of (y_true, y_prob, thresholds, description)
SCENARIOS = [
    # Base scenario
    (np.array([1, 1, 0]), np.array([0.1, 0.5, 0.1]), [100, 50, 10, 0], "base"),
    # Include prediction of exactly 0
    (np.array([0, 1, 1, 0]), np.array([0, 0.1, 0.5, 0.1]), [100, 50, 10, 0], "0-pred"),
    # Include prediction of exactly 1
    (np.array([1, 1, 0, 1]), np.array([0.1, 0.5, 0.1, 1]), [100, 50, 10, 0], "1-pred"),
    # Include predictions of exactly 0 and 1
    (np.array([0, 1, 1, 0, 1]), np.array([0, 0.1, 0.5, 0.1, 1]), [100, 50, 10, 0], "1 and 0 preds"),
    # Four value case
    (np.array([1, 1, 0, 0]), np.array([0.75, 0.5, 0.25, 0]), [100, 75, 50, 25, 0], "0 preds"),
]

# Expected values for each metric, organized as:
# (metric_name, scenario_id): [values by threshold index]
EXPECTED_VALUES = {
    # TP metric values for each scenario
    ("TP", "base"): [0, 1, 2, 2],
    ("TP", "0-pred"): [0, 1, 2, 2],
    ("TP", "1-pred"): [1, 2, 3, 3],
    ("TP", "1 and 0 preds"): [1, 2, 3, 3],
    ("TP", "0 preds"): [0, 1, 2, 2, 2],
    # FP metric values for each scenario
    ("FP", "base"): [0, 0, 1, 1],
    ("FP", "0-pred"): [0, 0, 1, 2],
    ("FP", "1-pred"): [0, 0, 1, 1],
    ("FP", "1 and 0 preds"): [0, 0, 1, 2],
    ("FP", "0 preds"): [0, 0, 0, 1, 2],
    # TN metric values for each scenario
    ("TN", "base"): [1, 1, 0, 0],
    ("TN", "0-pred"): [2, 2, 1, 0],
    ("TN", "1-pred"): [1, 1, 0, 0],
    ("TN", "1 and 0 preds"): [2, 2, 1, 0],
    ("TN", "0 preds"): [2, 2, 2, 1, 0],
    # FN metric values for each scenario
    ("FN", "base"): [2, 1, 0, 0],
    ("FN", "0-pred"): [2, 1, 0, 0],
    ("FN", "1-pred"): [2, 1, 0, 0],
    ("FN", "1 and 0 preds"): [2, 1, 0, 0],
    ("FN", "0 preds"): [2, 1, 0, 0, 0],
    # Accuracy metric values for each scenario
    ("Accuracy", "base"): [1 / 3, 2 / 3, 2 / 3, 2 / 3],
    ("Accuracy", "0-pred"): [0.5, 0.75, 0.75, 0.5],
    ("Accuracy", "1-pred"): [0.5, 0.75, 0.75, 0.75],
    ("Accuracy", "1 and 0 preds"): [0.6, 0.8, 0.8, 0.6],
    ("Accuracy", "0 preds"): [0.5, 0.75, 1, 0.75, 0.5],
    # Sensitivity metric values for each scenario
    ("Sensitivity", "base"): [0, 0.5, 1, 1],
    ("Sensitivity", "0-pred"): [0, 0.5, 1, 1],
    ("Sensitivity", "1-pred"): [1 / 3, 2 / 3, 1, 1],
    ("Sensitivity", "1 and 0 preds"): [1 / 3, 2 / 3, 1, 1],
    ("Sensitivity", "0 preds"): [0, 0.5, 1, 1, 1],
    # Specificity metric values for each scenario
    ("Specificity", "base"): [1, 1, 0, 0],
    ("Specificity", "0-pred"): [1, 1, 0.5, 0],
    ("Specificity", "1-pred"): [1, 1, 0, 0],
    ("Specificity", "1 and 0 preds"): [1, 1, 0.5, 0],
    ("Specificity", "0 preds"): [1, 1, 1, 0.5, 0],
    # PPV metric values for each scenario
    ("PPV", "base"): [1, 1, 2 / 3, 2 / 3],
    ("PPV", "0-pred"): [1, 1, 2 / 3, 0.5],
    ("PPV", "1-pred"): [1, 1, 0.75, 0.75],
    ("PPV", "1 and 0 preds"): [1, 1, 0.75, 0.6],
    ("PPV", "0 preds"): [1, 1, 1, 2 / 3, 0.5],
    # NPV metric values for each scenario
    ("NPV", "base"): [1 / 3, 0.5, 1, 1],
    ("NPV", "0-pred"): [0.5, 2 / 3, 1, 1],
    ("NPV", "1-pred"): [1 / 3, 0.5, 1, 1],
    ("NPV", "1 and 0 preds"): [0.5, 2 / 3, 1, 1],
    ("NPV", "0 preds"): [0.5, 2 / 3, 1, 1, 1],
    # Flag Rate metric values for each scenario
    ("Flag Rate", "base"): [0, 1 / 3, 1, 1],
    ("Flag Rate", "0-pred"): [0, 0.25, 0.75, 1],
    ("Flag Rate", "1-pred"): [0.25, 0.5, 1, 1],
    ("Flag Rate", "1 and 0 preds"): [0.2, 0.4, 0.8, 1],
    ("Flag Rate", "0 preds"): [0, 0.25, 0.5, 0.75, 1],
    # LR+ metric values for each scenario
    ("LR+", "base"): [np.nan, np.inf, 1, 1],
    ("LR+", "0-pred"): [np.nan, np.inf, 2, 1],
    ("LR+", "1-pred"): [np.inf, np.inf, 1, 1],
    ("LR+", "1 and 0 preds"): [np.inf, np.inf, 2, 1],
    ("LR+", "0 preds"): [np.nan, np.inf, np.inf, 2, 1],
    # NNE metric values for each scenario
    ("NNE", "base"): [1, 1, 1.5, 1.5],
    ("NNE", "0-pred"): [1, 1, 1.5, 2],
    ("NNE", "1-pred"): [1, 1, 4 / 3, 4 / 3],
    ("NNE", "1 and 0 preds"): [1, 1, 4 / 3, 5 / 3],
    ("NNE", "0 preds"): [1, 1, 1, 1.5, 2],
    # NetBenefitScore metric values for each scenario
    ("NetBenefitScore", "base"): [np.nan, 1 / 3, 17 / 27, 2 / 3],
    ("NetBenefitScore", "0-pred"): [np.nan, 1 / 4, 17 / 36, 1 / 2],
    ("NetBenefitScore", "1-pred"): [np.nan, 1 / 2, 13 / 18, 3 / 4],
    ("NetBenefitScore", "1 and 0 preds"): [np.nan, 2 / 5, 26 / 45, 3 / 5],
    ("NetBenefitScore", "0 preds"): [np.nan, 1 / 4, 1 / 2, 5 / 12, 1 / 2],
    # NNT@0.333 metric values for each scenario
    ("NNT@0.333", "base"): [3, 3, 4.5, 4.5],
    ("NNT@0.333", "0-pred"): [3, 3, 4.5, 6],
    ("NNT@0.333", "1-pred"): [3, 3, 4, 4],
    ("NNT@0.333", "1 and 0 preds"): [3, 3, 4, 5],
    ("NNT@0.333", "0 preds"): [3, 3, 3, 4.5, 6],
}


def get_scenario_params():
    """Generate parameters for pytest.mark.parametrize"""
    params = []
    ids = []

    for y_true, y_prob, thresholds, scenario_id in SCENARIOS:
        # For each metric
        for metric in undertest.STATNAMES + ["NNT@0.333"]:
            expected_values = EXPECTED_VALUES.get((metric, scenario_id))
            if expected_values is not None:  # Skip if no expected value defined
                params.append((y_true, y_prob, thresholds, metric, expected_values))
                ids.append(f"{scenario_id}-{metric}")

    return params, ids


# Generate all the parameter combinations
METRIC_PARAMS, METRIC_IDS = get_scenario_params()


@pytest.mark.parametrize("y_true,y_prob,thresholds,metric,expected_values", METRIC_PARAMS, ids=METRIC_IDS)
def test_individual_metric_values(y_true, y_prob, thresholds, metric, expected_values):
    """Test individual metric values for various scenarios and thresholds"""
    stats = undertest.calculate_bin_stats(y_true, y_prob, not_point_thresholds=True)

    # Ensure thresholds and expected_values are the same length
    assert len(thresholds) >= len(
        expected_values
    ), f"Thresholds array length ({len(thresholds)}) must be >= expected values length ({len(expected_values)})"

    # Loop through thresholds and expected values using zip
    for threshold, expected_value in zip(thresholds, expected_values):
        # Find the row with the matching threshold
        row = stats.loc[stats["Threshold"] == threshold]

        if len(row) != 1:
            pytest.fail(f"Expected exactly one row with threshold={threshold}, got {len(row)}")

        actual_value = row[metric].iloc[0]

        # For nan values, check that both are nan
        if np.isnan(expected_value):
            assert np.isnan(actual_value), f"Expected {metric} to be nan for threshold {threshold}, got {actual_value}"
        else:
            assert actual_value == pytest.approx(
                expected_value, rel=0.01
            ), f"Expected {metric}={expected_value} for threshold {threshold}, got {actual_value}"


# Get unique scenarios
SCENARIOS_PARAMS = [(y_true, y_prob, thresholds, desc) for y_true, y_prob, thresholds, desc in SCENARIOS]
SCENARIOS_IDS = [params[-1] for params in SCENARIOS]


@pytest.mark.parametrize("y_true,y_prob,thresholds,scenario_id", SCENARIOS_PARAMS, ids=SCENARIOS_IDS)
class TestScenarios:
    """Group tests by scenario to make it easier to test variations on the same data"""

    def test_stat_keys(self, y_true, y_prob, thresholds, scenario_id):
        """Ensure stat manipulations are intentional"""
        expected_keys = {
            "Flag Rate",
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
            "LR+",
            "NNE",
            "NetBenefitScore",
            "TP",
            "FP",
            "TN",
            "FN",
        }
        assert set(undertest.STATNAMES) == expected_keys

    def test_all_metrics_present(self, y_true, y_prob, thresholds, scenario_id):
        """Test that all expected metrics are present in the output"""
        stats = undertest.calculate_bin_stats(y_true, y_prob, not_point_thresholds=True)
        expected_columns = [undertest.THRESHOLD] + undertest.STATNAMES + [f"NNT@{undertest.DEFAULT_RHO:0.3n}"]
        assert all(col in stats.columns for col in expected_columns)

    def test_score_arr_percentile(self, y_true, y_prob, thresholds, scenario_id):
        """Test that percentile scores produce the same results"""
        # Original score
        original_stats = undertest.calculate_bin_stats(y_true, y_prob, not_point_thresholds=True)

        # Percentile score (x100)
        percentile_stats = undertest.calculate_bin_stats(y_true, y_prob * 100, not_point_thresholds=True)

        # Should produce the same results
        pd.testing.assert_frame_equal(
            original_stats, percentile_stats, check_column_type=False, check_like=True, check_dtype=False
        )

    def test_score_with_y_proba_nulls(self, y_true, y_prob, thresholds, scenario_id):
        """Test handling of NaN values in probability scores"""
        # Original stats
        original_stats = undertest.calculate_bin_stats(y_true, y_prob, not_point_thresholds=True)

        # Add some NaN values
        y_prob_with_nulls = np.hstack((y_prob, [np.nan, np.nan]))
        y_true_with_nulls = np.hstack((y_true, [0, 1]))

        # Stats with nulls
        nulls_stats = undertest.calculate_bin_stats(y_true_with_nulls, y_prob_with_nulls, not_point_thresholds=True)

        # Should produce the same results
        pd.testing.assert_frame_equal(
            original_stats, nulls_stats, check_column_type=False, check_like=True, check_dtype=False
        )

    def test_score_with_y_true_nulls(self, y_true, y_prob, thresholds, scenario_id):
        """Test handling of NaN values in true labels"""
        # Original stats
        original_stats = undertest.calculate_bin_stats(y_true, y_prob, not_point_thresholds=True)

        # Add some NaN values
        y_prob_with_nulls = np.hstack((y_prob, [0.1, 0.9]))
        y_true_with_nulls = np.hstack((y_true, [np.nan, np.nan]))

        # Stats with nulls
        nulls_stats = undertest.calculate_bin_stats(y_true_with_nulls, y_prob_with_nulls, not_point_thresholds=True)

        # Should produce the same results
        pd.testing.assert_frame_equal(
            original_stats, nulls_stats, check_column_type=False, check_like=True, check_dtype=False
        )


def test_bin_stats_point_thresholds():
    # Get the base scenario data
    y_true = np.array([1, 1, 0])
    y_prob = np.array([0.1, 0.5, 0.1])

    # Get results with not_point_thresholds=True (original behavior)
    base_stats = undertest.calculate_bin_stats(y_true, y_prob, not_point_thresholds=True)

    # Get results with default point thresholds
    actual = undertest.calculate_bin_stats(y_true, np.array(y_prob))

    # In the original test, the rows were duplicated based on threshold gaps
    expected = []
    for dup, row in zip([50, 40, 11], base_stats.iterrows()):
        expected.extend([row[1]] * dup)
    expected = pd.DataFrame(expected)
    expected.Threshold = np.arange(100, -1, -1)
    expected = expected.reset_index(drop=True)

    # Verify accuracy of threshold point-wise
    # Some columns don't work with the duplication
    threshold_dependent_columns = ["NetBenefitScore"]
    expected = expected.drop(columns=threshold_dependent_columns)
    actual = actual.drop(columns=threshold_dependent_columns)

    pd.testing.assert_frame_equal(actual, expected[actual.columns], check_dtype=False)


def test_threshold_precision_increases_rows():
    y_true = np.array([1, 0])
    y_prob = np.array([0.5, 0.3])

    # Use default threshold precision (0): 101 rows (0–100)
    default_stats = undertest.calculate_bin_stats(y_true, y_prob)
    assert len(default_stats) == 101

    # Now try precision=2 → 10,001 thresholds
    high_precision_stats = undertest.calculate_bin_stats(y_true, y_prob, threshold_precision=2)
    assert len(high_precision_stats) == 100 * 10**2 + 1


def test_thresholds_are_rounded_correctly():
    y_true = np.array([1, 0])
    y_prob = np.array([0.6, 0.3])
    threshold_precision = 2

    stats = undertest.calculate_bin_stats(y_true, y_prob, threshold_precision=threshold_precision)
    thresholds = stats["Threshold"].to_numpy()

    step = 1 / 10**threshold_precision
    assert np.allclose(np.diff(thresholds[::-1]), step, atol=1e-6)


def test_all_metrics_valid_with_high_threshold_precision():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.7, 0.4, 0.2])

    stats = undertest.calculate_bin_stats(y_true, y_prob, threshold_precision=2)

    all_metrics = undertest.STATNAMES + [f"NNT@{undertest.DEFAULT_RHO:0.3n}"]

    for metric in all_metrics:
        assert metric in stats.columns
        values = stats[metric].dropna()

        if metric in {"Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Flag Rate"}:
            assert (0.0 <= values).all() and (values <= 1.0).all()

        elif metric in {"TP", "FP", "TN", "FN"}:
            assert (values >= 0).all()
            assert pd.api.types.is_integer_dtype(values)

        elif metric in {"LR+", "NetBenefitScore", "NNE", f"NNT@{undertest.DEFAULT_RHO:0.3n}"}:
            finite_values = values[np.isfinite(values)]
            assert (finite_values >= 0).all()

        else:
            assert pd.api.types.is_numeric_dtype(values)


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
