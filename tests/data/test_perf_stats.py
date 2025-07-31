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
# (scenario_id, threshold, metric_name): expected_value
EXPECTED_VALUES = {
    # Base scenario expected values
    ("base", 100, "TP"): 0,
    ("base", 100, "FP"): 0,
    ("base", 100, "TN"): 1,
    ("base", 100, "FN"): 2,
    ("base", 100, "Accuracy"): 1 / 3,
    ("base", 100, "Sensitivity"): 0,
    ("base", 100, "Specificity"): 1,
    ("base", 100, "PPV"): 1,
    ("base", 100, "NPV"): 1 / 3,
    ("base", 100, "Flag Rate"): 0,
    ("base", 100, "LR+"): np.nan,
    ("base", 100, "NNE"): 1,
    ("base", 100, "NetBenefitScore"): np.nan,
    ("base", 100, "NNT@0.333"): 3,
    ("base", 50, "TP"): 1,
    ("base", 50, "FP"): 0,
    ("base", 50, "TN"): 1,
    ("base", 50, "FN"): 1,
    ("base", 50, "Accuracy"): 2 / 3,
    ("base", 50, "Sensitivity"): 0.5,
    ("base", 50, "Specificity"): 1,
    ("base", 50, "PPV"): 1,
    ("base", 50, "NPV"): 0.5,
    ("base", 50, "Flag Rate"): 1 / 3,
    ("base", 50, "LR+"): np.inf,
    ("base", 50, "NNE"): 1,
    ("base", 50, "NetBenefitScore"): 1 / 3,
    ("base", 50, "NNT@0.333"): 3,
    ("base", 10, "TP"): 2,
    ("base", 10, "FP"): 1,
    ("base", 10, "TN"): 0,
    ("base", 10, "FN"): 0,
    ("base", 10, "Accuracy"): 2 / 3,
    ("base", 10, "Sensitivity"): 1,
    ("base", 10, "Specificity"): 0,
    ("base", 10, "PPV"): 2 / 3,
    ("base", 10, "NPV"): 1,
    ("base", 10, "Flag Rate"): 1,
    ("base", 10, "LR+"): 1,
    ("base", 10, "NNE"): 1.5,
    ("base", 10, "NetBenefitScore"): 17 / 27,
    ("base", 10, "NNT@0.333"): 4.5,
    ("base", 0, "TP"): 2,
    ("base", 0, "FP"): 1,
    ("base", 0, "TN"): 0,
    ("base", 0, "FN"): 0,
    ("base", 0, "Accuracy"): 2 / 3,
    ("base", 0, "Sensitivity"): 1,
    ("base", 0, "Specificity"): 0,
    ("base", 0, "PPV"): 2 / 3,
    ("base", 0, "NPV"): 1,
    ("base", 0, "Flag Rate"): 1,
    ("base", 0, "LR+"): 1,
    ("base", 0, "NNE"): 1.5,
    ("base", 0, "NetBenefitScore"): 2 / 3,
    ("base", 0, "NNT@0.333"): 4.5,
    # 0-pred scenario
    ("0-pred", 100, "TP"): 0,
    ("0-pred", 100, "FP"): 0,
    ("0-pred", 100, "TN"): 2,
    ("0-pred", 100, "FN"): 2,
    ("0-pred", 100, "Accuracy"): 0.5,
    ("0-pred", 100, "Sensitivity"): 0,
    ("0-pred", 100, "Specificity"): 1,
    ("0-pred", 100, "PPV"): 1,
    ("0-pred", 100, "NPV"): 0.5,
    ("0-pred", 100, "Flag Rate"): 0,
    ("0-pred", 100, "LR+"): np.nan,
    ("0-pred", 100, "NNE"): 1,
    ("0-pred", 100, "NetBenefitScore"): np.nan,
    ("0-pred", 100, "NNT@0.333"): 3,
    ("0-pred", 50, "TP"): 1,
    ("0-pred", 50, "FP"): 0,
    ("0-pred", 50, "TN"): 2,
    ("0-pred", 50, "FN"): 1,
    ("0-pred", 50, "Accuracy"): 0.75,
    ("0-pred", 50, "Sensitivity"): 0.5,
    ("0-pred", 50, "Specificity"): 1,
    ("0-pred", 50, "PPV"): 1,
    ("0-pred", 50, "NPV"): 2 / 3,
    ("0-pred", 50, "Flag Rate"): 0.25,
    ("0-pred", 50, "LR+"): np.inf,
    ("0-pred", 50, "NNE"): 1,
    ("0-pred", 50, "NetBenefitScore"): 1 / 4,
    ("0-pred", 50, "NNT@0.333"): 3,
    ("0-pred", 10, "TP"): 2,
    ("0-pred", 10, "FP"): 1,
    ("0-pred", 10, "TN"): 1,
    ("0-pred", 10, "FN"): 0,
    ("0-pred", 10, "Accuracy"): 0.75,
    ("0-pred", 10, "Sensitivity"): 1,
    ("0-pred", 10, "Specificity"): 0.5,
    ("0-pred", 10, "PPV"): 2 / 3,
    ("0-pred", 10, "NPV"): 1,
    ("0-pred", 10, "Flag Rate"): 0.75,
    ("0-pred", 10, "LR+"): 2,
    ("0-pred", 10, "NNE"): 1.5,
    ("0-pred", 10, "NetBenefitScore"): 17 / 36,
    ("0-pred", 10, "NNT@0.333"): 4.5,
    ("0-pred", 0, "TP"): 2,
    ("0-pred", 0, "FP"): 2,
    ("0-pred", 0, "TN"): 0,
    ("0-pred", 0, "FN"): 0,
    ("0-pred", 0, "Accuracy"): 0.5,
    ("0-pred", 0, "Sensitivity"): 1,
    ("0-pred", 0, "Specificity"): 0,
    ("0-pred", 0, "PPV"): 0.5,
    ("0-pred", 0, "NPV"): 1,
    ("0-pred", 0, "Flag Rate"): 1,
    ("0-pred", 0, "LR+"): 1,
    ("0-pred", 0, "NNE"): 2,
    ("0-pred", 0, "NetBenefitScore"): 1 / 2,
    ("0-pred", 0, "NNT@0.333"): 6,
    # 1-pred scenario
    ("1-pred", 100, "TP"): 1,
    ("1-pred", 100, "FP"): 0,
    ("1-pred", 100, "TN"): 1,
    ("1-pred", 100, "FN"): 2,
    ("1-pred", 100, "Accuracy"): 0.5,
    ("1-pred", 100, "Sensitivity"): 1 / 3,
    ("1-pred", 100, "Specificity"): 1,
    ("1-pred", 100, "PPV"): 1,
    ("1-pred", 100, "NPV"): 1 / 3,
    ("1-pred", 100, "Flag Rate"): 0.25,
    ("1-pred", 100, "LR+"): np.inf,
    ("1-pred", 100, "NNE"): 1,
    ("1-pred", 100, "NetBenefitScore"): np.nan,
    ("1-pred", 100, "NNT@0.333"): 3,
    ("1-pred", 50, "TP"): 2,
    ("1-pred", 50, "FP"): 0,
    ("1-pred", 50, "TN"): 1,
    ("1-pred", 50, "FN"): 1,
    ("1-pred", 50, "Accuracy"): 0.75,
    ("1-pred", 50, "Sensitivity"): 2 / 3,
    ("1-pred", 50, "Specificity"): 1,
    ("1-pred", 50, "PPV"): 1,
    ("1-pred", 50, "NPV"): 0.5,
    ("1-pred", 50, "Flag Rate"): 0.5,
    ("1-pred", 50, "LR+"): np.inf,
    ("1-pred", 50, "NNE"): 1,
    ("1-pred", 50, "NetBenefitScore"): 1 / 2,
    ("1-pred", 50, "NNT@0.333"): 3,
    ("1-pred", 10, "TP"): 3,
    ("1-pred", 10, "FP"): 1,
    ("1-pred", 10, "TN"): 0,
    ("1-pred", 10, "FN"): 0,
    ("1-pred", 10, "Accuracy"): 0.75,
    ("1-pred", 10, "Sensitivity"): 1,
    ("1-pred", 10, "Specificity"): 0,
    ("1-pred", 10, "PPV"): 0.75,
    ("1-pred", 10, "NPV"): 1,
    ("1-pred", 10, "Flag Rate"): 1,
    ("1-pred", 10, "LR+"): 1,
    ("1-pred", 10, "NNE"): 4 / 3,
    ("1-pred", 10, "NetBenefitScore"): 13 / 18,
    ("1-pred", 10, "NNT@0.333"): 4,
    ("1-pred", 0, "TP"): 3,
    ("1-pred", 0, "FP"): 1,
    ("1-pred", 0, "TN"): 0,
    ("1-pred", 0, "FN"): 0,
    ("1-pred", 0, "Accuracy"): 0.75,
    ("1-pred", 0, "Sensitivity"): 1,
    ("1-pred", 0, "Specificity"): 0,
    ("1-pred", 0, "PPV"): 0.75,
    ("1-pred", 0, "NPV"): 1,
    ("1-pred", 0, "Flag Rate"): 1,
    ("1-pred", 0, "LR+"): 1,
    ("1-pred", 0, "NNE"): 4 / 3,
    ("1-pred", 0, "NetBenefitScore"): 3 / 4,
    ("1-pred", 0, "NNT@0.333"): 4,
    # 1 and 0 preds
    ("1 and 0 preds", 100, "TP"): 1,
    ("1 and 0 preds", 100, "FP"): 0,
    ("1 and 0 preds", 100, "TN"): 2,
    ("1 and 0 preds", 100, "FN"): 2,
    ("1 and 0 preds", 100, "Accuracy"): 0.6,
    ("1 and 0 preds", 100, "Sensitivity"): 1 / 3,
    ("1 and 0 preds", 100, "Specificity"): 1,
    ("1 and 0 preds", 100, "PPV"): 1,
    ("1 and 0 preds", 100, "NPV"): 0.5,
    ("1 and 0 preds", 100, "Flag Rate"): 0.2,
    ("1 and 0 preds", 100, "LR+"): np.inf,
    ("1 and 0 preds", 100, "NNE"): 1,
    ("1 and 0 preds", 100, "NetBenefitScore"): np.nan,
    ("1 and 0 preds", 100, "NNT@0.333"): 3,
    ("1 and 0 preds", 50, "TP"): 2,
    ("1 and 0 preds", 50, "FP"): 0,
    ("1 and 0 preds", 50, "TN"): 2,
    ("1 and 0 preds", 50, "FN"): 1,
    ("1 and 0 preds", 50, "Accuracy"): 0.8,
    ("1 and 0 preds", 50, "Sensitivity"): 2 / 3,
    ("1 and 0 preds", 50, "Specificity"): 1,
    ("1 and 0 preds", 50, "PPV"): 1,
    ("1 and 0 preds", 50, "NPV"): 2 / 3,
    ("1 and 0 preds", 50, "Flag Rate"): 0.4,
    ("1 and 0 preds", 50, "LR+"): np.inf,
    ("1 and 0 preds", 50, "NNE"): 1,
    ("1 and 0 preds", 50, "NetBenefitScore"): 2 / 5,
    ("1 and 0 preds", 50, "NNT@0.333"): 3,
    ("1 and 0 preds", 10, "TP"): 3,
    ("1 and 0 preds", 10, "FP"): 1,
    ("1 and 0 preds", 10, "TN"): 1,
    ("1 and 0 preds", 10, "FN"): 0,
    ("1 and 0 preds", 10, "Accuracy"): 0.8,
    ("1 and 0 preds", 10, "Sensitivity"): 1,
    ("1 and 0 preds", 10, "Specificity"): 0.5,
    ("1 and 0 preds", 10, "PPV"): 0.75,
    ("1 and 0 preds", 10, "NPV"): 1,
    ("1 and 0 preds", 10, "Flag Rate"): 0.8,
    ("1 and 0 preds", 10, "LR+"): 2,
    ("1 and 0 preds", 10, "NNE"): 4 / 3,
    ("1 and 0 preds", 10, "NetBenefitScore"): 26 / 45,
    ("1 and 0 preds", 10, "NNT@0.333"): 4,
    ("1 and 0 preds", 0, "TP"): 3,
    ("1 and 0 preds", 0, "FP"): 2,
    ("1 and 0 preds", 0, "TN"): 0,
    ("1 and 0 preds", 0, "FN"): 0,
    ("1 and 0 preds", 0, "Accuracy"): 0.6,
    ("1 and 0 preds", 0, "Sensitivity"): 1,
    ("1 and 0 preds", 0, "Specificity"): 0,
    ("1 and 0 preds", 0, "PPV"): 0.6,
    ("1 and 0 preds", 0, "NPV"): 1,
    ("1 and 0 preds", 0, "Flag Rate"): 1,
    ("1 and 0 preds", 0, "LR+"): 1,
    ("1 and 0 preds", 0, "NNE"): 5 / 3,
    ("1 and 0 preds", 0, "NetBenefitScore"): 3 / 5,
    ("1 and 0 preds", 0, "NNT@0.333"): 5,
    # 0 preds scenario
    ("0 preds", 100, "TP"): 0,
    ("0 preds", 100, "FP"): 0,
    ("0 preds", 100, "TN"): 2,
    ("0 preds", 100, "FN"): 2,
    ("0 preds", 100, "Accuracy"): 0.5,
    ("0 preds", 100, "Sensitivity"): 0,
    ("0 preds", 100, "Specificity"): 1,
    ("0 preds", 100, "PPV"): 1,
    ("0 preds", 100, "NPV"): 0.5,
    ("0 preds", 100, "Flag Rate"): 0,
    ("0 preds", 100, "LR+"): np.nan,
    ("0 preds", 100, "NNE"): 1,
    ("0 preds", 100, "NetBenefitScore"): np.nan,
    ("0 preds", 100, "NNT@0.333"): 3,
    ("0 preds", 75, "TP"): 1,
    ("0 preds", 75, "FP"): 0,
    ("0 preds", 75, "TN"): 2,
    ("0 preds", 75, "FN"): 1,
    ("0 preds", 75, "Accuracy"): 0.75,
    ("0 preds", 75, "Sensitivity"): 0.5,
    ("0 preds", 75, "Specificity"): 1,
    ("0 preds", 75, "PPV"): 1,
    ("0 preds", 75, "NPV"): 2 / 3,
    ("0 preds", 75, "Flag Rate"): 0.25,
    ("0 preds", 75, "LR+"): np.inf,
    ("0 preds", 75, "NNE"): 1,
    ("0 preds", 75, "NetBenefitScore"): 1 / 4,
    ("0 preds", 75, "NNT@0.333"): 3,
    ("0 preds", 50, "TP"): 2,
    ("0 preds", 50, "FP"): 0,
    ("0 preds", 50, "TN"): 2,
    ("0 preds", 50, "FN"): 0,
    ("0 preds", 50, "Accuracy"): 1,
    ("0 preds", 50, "Sensitivity"): 1,
    ("0 preds", 50, "Specificity"): 1,
    ("0 preds", 50, "PPV"): 1,
    ("0 preds", 50, "NPV"): 1,
    ("0 preds", 50, "Flag Rate"): 0.5,
    ("0 preds", 50, "LR+"): np.inf,
    ("0 preds", 50, "NNE"): 1,
    ("0 preds", 50, "NetBenefitScore"): 1 / 2,
    ("0 preds", 50, "NNT@0.333"): 3,
    ("0 preds", 25, "TP"): 2,
    ("0 preds", 25, "FP"): 1,
    ("0 preds", 25, "TN"): 1,
    ("0 preds", 25, "FN"): 0,
    ("0 preds", 25, "Accuracy"): 0.75,
    ("0 preds", 25, "Sensitivity"): 1,
    ("0 preds", 25, "Specificity"): 0.5,
    ("0 preds", 25, "PPV"): 2 / 3,
    ("0 preds", 25, "NPV"): 1,
    ("0 preds", 25, "Flag Rate"): 0.75,
    ("0 preds", 25, "LR+"): 2,
    ("0 preds", 25, "NNE"): 1.5,
    ("0 preds", 25, "NetBenefitScore"): 5 / 12,
    ("0 preds", 25, "NNT@0.333"): 4.5,
    ("0 preds", 0, "TP"): 2,
    ("0 preds", 0, "FP"): 2,
    ("0 preds", 0, "TN"): 0,
    ("0 preds", 0, "FN"): 0,
    ("0 preds", 0, "Accuracy"): 0.5,
    ("0 preds", 0, "Sensitivity"): 1,
    ("0 preds", 0, "Specificity"): 0,
    ("0 preds", 0, "PPV"): 0.5,
    ("0 preds", 0, "NPV"): 1,
    ("0 preds", 0, "Flag Rate"): 1,
    ("0 preds", 0, "LR+"): 1,
    ("0 preds", 0, "NNE"): 2,
    ("0 preds", 0, "NetBenefitScore"): 1 / 2,
    ("0 preds", 0, "NNT@0.333"): 6,
}


def get_scenario_params():
    """Generate parameters for pytest.mark.parametrize"""
    params = []
    ids = []

    for y_true, y_prob, thresholds, scenario_id in SCENARIOS:
        for threshold in thresholds:
            # For each metric
            for metric in undertest.STATNAMES + ["NNT@0.333"]:
                expected_value = EXPECTED_VALUES.get((scenario_id, threshold, metric))
                if expected_value is not None:  # Skip if no expected value defined
                    params.append((y_true, y_prob, threshold, metric, expected_value))
                    ids.append(f"{scenario_id}-t{threshold}-{metric}")

    return params, ids


# Generate all the parameter combinations
METRIC_PARAMS, METRIC_IDS = get_scenario_params()


@pytest.mark.parametrize("y_true,y_prob,threshold,metric,expected_value", METRIC_PARAMS, ids=METRIC_IDS)
def test_individual_metric_values(y_true, y_prob, threshold, metric, expected_value):
    """Test individual metric values for various scenarios and thresholds"""
    stats = undertest.calculate_bin_stats(y_true, y_prob, not_point_thresholds=True)

    # Find the row with the matching threshold
    row = stats.loc[stats["Threshold"] == threshold]

    if len(row) != 1:
        pytest.fail(f"Expected exactly one row with threshold={threshold}, got {len(row)}")

    actual_value = row[metric].iloc[0]

    # For nan values, check that both are nan
    if np.isnan(expected_value):
        assert np.isnan(actual_value), f"Expected {metric} to be nan, got {actual_value}"
    else:
        assert actual_value == pytest.approx(
            expected_value, rel=0.01
        ), f"Expected {metric}={expected_value}, got {actual_value}"


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
