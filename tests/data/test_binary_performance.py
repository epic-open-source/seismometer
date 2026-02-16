from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from seismometer.configuration import ConfigProvider
from seismometer.configuration.model import Cohort, Event
from seismometer.data.binary_performance import calculate_stats, generate_analytics_data
from seismometer.data.loader import SeismogramLoader
from seismometer.seismogram import Seismogram


def get_test_config(tmp_path):
    mock_config = Mock(autospec=ConfigProvider)
    mock_config.output_dir.return_value
    mock_config.events = {
        "event1": Event(source="event1", display_name="event1", window_hr=1),
        "event2": Event(source="event2", display_name="event2", window_hr=2, aggregation_method="min"),
        "event3": Event(source="event3", display_name="event3", window_hr=1),
    }
    mock_config.target = "event1"
    mock_config.entity_keys = ["entity"]
    mock_config.predict_time = "time"
    mock_config.cohorts = [Cohort(source=name) for name in ["cohort1"]]
    mock_config.features = ["one"]
    mock_config.config_dir = tmp_path / "config"
    mock_config.censor_min_count = 0
    mock_config.targets = ["event1", "event2", "event3"]
    mock_config.output_list = ["prediction", "score1", "score2"]

    return mock_config


def get_test_loader(config):
    mock_loader = Mock(autospec=SeismogramLoader)
    mock_loader.config = config

    return mock_loader


def get_test_data():
    return pd.DataFrame(
        {
            "entity": ["A", "A", "B", "C"],
            "prediction": [1, 2, 3, 4],
            "time": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "event1_Value": [0, 1, 0, 1],
            "event1_Time": ["2022-01-01", "2022-01-02", "2022-01-03", "2021-12-31"],
            "event2_Value": [1, 0, 0, 1],
            "event2_Time": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "event3_Value": [0, 1, 1, 1],
            "event3_Time": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "cohort1": ["A", "A", "A", "B"],
            "score1": [0.1, 0.4, 0.35, 0.8],
            "score2": [0.2, 0.5, 0.3, 0.7],
            "target1": [0, 1, 0, 1],
            "target2": [1, 0, 1, 0],
            "target3": [1, 1, 1, 0],
        }
    )


@pytest.fixture
def fake_seismo(tmp_path):
    config = get_test_config(tmp_path)
    loader = get_test_loader(config)
    sg = Seismogram(config, loader)
    sg.dataframe = get_test_data()
    yield sg

    Seismogram.kill()


class TestCalculateStats:
    @pytest.mark.parametrize("metric", ["Sensitivity", "Specificity", "Flag Rate"])
    def test_basic(self, metric):
        df = pd.DataFrame(
            {"target": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0], "score": [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5]}
        )
        metric_values = [0.5, 0.7]
        metrics = ["Sensitivity", "Specificity", "Flag\u00A0Rate", "PPV", "Accuracy"]
        metrics.remove(metric.replace("Flag Rate", "Flag\u00A0Rate"))
        stats = calculate_stats(df, "target", "score", metric, metric_values)
        assert stats["AUROC"] == roc_auc_score(df["target"], df["score"])
        precision, recall, _ = precision_recall_curve(df["target"], df["score"])
        assert np.allclose(stats["AUPRC"], auc(recall, precision), rtol=0.01)
        assert all(f"{val}_{metric}" in stats for val in metric_values for metric in metrics)
        assert all(col in stats for col in ["Positives", "Prevalence"])

    def test_invalid_metric(self):
        df = pd.DataFrame(
            {"target": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0], "score": [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5]}
        )
        metric_values = [0.5, 0.7]
        with pytest.raises(
            ValueError,
            match="Invalid metric name: InvalidMetric. The metric needs to be one of: "
            "\\['Sensitivity', 'Specificity', 'Flag Rate', 'Threshold'\\]",
        ):
            calculate_stats(df, "target", "score", "InvalidMetric", metric_values)

    def test_positives_prevalence(self):
        df = pd.DataFrame(
            {"target": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0], "score": [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5]}
        )
        metric_values = [0.5, 0.7]
        stats = calculate_stats(df, "target", "score", "Sensitivity", metric_values)
        assert stats["Positives"] == np.sum(df["target"])
        assert stats["Prevalence"] == np.mean(df["target"])

    @pytest.mark.parametrize("metric", ["Sensitivity", "Specificity", "Flag Rate"])
    def test_metric_values_decimals(self, metric):
        df = pd.DataFrame(
            {"target": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0], "score": [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5]}
        )
        metric_values = [0.534, 0.7, 0.100032, 0.1 + 0.3, 0.00002]
        expected_metric_values = [0.53, 0.7, 0.1, 0.4, 0]
        metrics = ["Sensitivity", "Specificity", "Flag Rate", "PPV", "Accuracy"]
        metrics.remove(metric)
        stats = calculate_stats(df, "target", "score", metric, metric_values, decimals=2)
        metrics = [val.replace(" ", "\u00A0") for val in metrics]
        assert all(f"{val}_{metric}" in stats for val in expected_metric_values for metric in metrics)

    def test_metrics_to_display(self):
        df = pd.DataFrame(
            {"target": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0], "score": [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5]}
        )
        metric_values = [0.5, 0.7]
        stats = calculate_stats(
            df,
            "target",
            "score",
            "Flag Rate",
            metric_values,
            metrics_to_display=["Sensitivity", "Specificity", "Prevalence"],
        )
        threshold_specific_cols = ["Sensitivity", "Specificity"]
        overall_stats_cols = ["Prevalence"]
        excluded_cols = ["Flag\u00A0Rate", "PPV", "Positives", "AUROC", "AUPRC"]
        assert all(f"{val}_{metric}" in stats for val in metric_values for metric in threshold_specific_cols)
        assert all(f"{val}_{metric}" not in stats for val in metric_values for metric in excluded_cols)
        assert all(col not in stats for col in excluded_cols)
        assert all(col in stats for col in overall_stats_cols)

    @pytest.mark.parametrize(
        "metric, expected_thresholds",
        [
            ("Sensitivity", np.array([100.0, 80.0, 70.0, 40.0])),
            ("Specificity", np.array([10.0, 35.0, 35.0, 100.0])),
            ("Flag Rate", np.array([100.0, 50.0, 35.0, 10.0])),
        ],
    )
    def test_computed_threshold_basic(self, metric, expected_thresholds):
        df = pd.DataFrame(
            {"target": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0], "score": [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5]}
        )
        metric_values = [0, 0.5, 0.7, 1]
        stats = calculate_stats(df, "target", "score", metric, metric_values)
        computed_thresholds = [stats[f"{val}_Threshold"] for val in metric_values]
        assert np.array_equal(computed_thresholds, expected_thresholds)

    @pytest.mark.parametrize(
        "metric, expected_thresholds",
        [
            ("Sensitivity", np.array([100.0, 100.0, 100.0, 100.0])),
            ("Specificity", np.array([10.0, 40.0, 40.0, 100.0])),
            ("Flag Rate", np.array([100.0, 40.0, 30.0, 10.0])),
        ],
    )
    def test_computed_threshold_edge_cases_all_zeroes(self, metric, expected_thresholds):
        df = pd.DataFrame({"target": [0, 0, 0, 0, 0], "score": [0.1, 0.2, 0.3, 0.4, 0.5]})
        metric_values = [0, 0.5, 0.7, 1]
        stats = calculate_stats(df, "target", "score", metric, metric_values, metrics_to_display=["Threshold"])
        computed_thresholds = [stats[f"{val}_Threshold"] for val in metric_values]
        assert np.array_equal(computed_thresholds, expected_thresholds)

    @pytest.mark.parametrize(
        "metric, expected_thresholds",
        [
            ("Sensitivity", np.array([100.0, 40.0, 30.0, 10.0])),
            ("Specificity", np.array([100.0, 100.0, 100.0, 100.0])),
            ("Flag Rate", np.array([100.0, 40.0, 30.0, 10.0])),
        ],
    )
    def test_computed_threshold_edge_cases_all_ones(self, metric, expected_thresholds):
        df = pd.DataFrame({"target": [1, 1, 1, 1, 1], "score": [0.1, 0.2, 0.3, 0.4, 0.5]})
        metric_values = [0, 0.5, 0.7, 1]
        stats = calculate_stats(df, "target", "score", metric, metric_values, metrics_to_display=["Threshold"])
        computed_thresholds = [stats[f"{val}_Threshold"] for val in metric_values]
        assert np.array_equal(computed_thresholds, expected_thresholds)


class TestCalculateStatsErrorHandling:
    """Test error handling and edge cases for calculate_stats()"""

    def test_empty_metric_values_list(self):
        """Test calculate_stats() with empty metric_values list"""
        df = pd.DataFrame(
            {"target": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0], "score": [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5]}
        )
        metric_values = []
        stats = calculate_stats(df, "target", "score", "Sensitivity", metric_values)

        # Should still return overall stats like AUROC, Prevalence, Positives
        assert "AUROC" in stats
        assert "AUPRC" in stats
        assert "Positives" in stats
        assert "Prevalence" in stats
        assert stats["Positives"] == 5
        assert stats["Prevalence"] == 0.5

    def test_invalid_metrics_to_display(self):
        """Test calculate_stats() with invalid metrics_to_display"""
        df = pd.DataFrame(
            {"target": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0], "score": [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5]}
        )
        metric_values = [0.5, 0.7]

        # Invalid metric names should raise KeyError or similar error
        with pytest.raises(Exception):  # Could be KeyError from BinaryClassifierMetricGenerator
            calculate_stats(
                df, "target", "score", "Sensitivity", metric_values, metrics_to_display=["InvalidMetric", "AnotherBad"]
            )

    def test_all_nan_target_column(self):
        """Test calculate_stats() with all NaN target values"""
        df = pd.DataFrame({"target": [np.nan, np.nan, np.nan, np.nan], "score": [0.1, 0.4, 0.35, 0.8]})
        metric_values = [0.5, 0.7]

        # FIXED BUG #5: Now raises helpful ValueError instead of cryptic IndexError
        with pytest.raises(
            ValueError, match="Cannot calculate statistics: all values in target column 'target' are NaN"
        ):
            calculate_stats(df, "target", "score", "Sensitivity", metric_values)

    def test_all_nan_score_column(self):
        """Test calculate_stats() with all NaN score values"""
        df = pd.DataFrame({"target": [0, 1, 0, 1], "score": [np.nan, np.nan, np.nan, np.nan]})
        metric_values = [0.5, 0.7]

        # FIXED BUG #5: Now raises helpful ValueError instead of cryptic IndexError
        with pytest.raises(
            ValueError, match="Cannot calculate statistics: all values in score column 'score' are NaN"
        ):
            calculate_stats(df, "target", "score", "Sensitivity", metric_values)

    def test_no_valid_paired_rows(self):
        """Test calculate_stats() when no valid paired rows remain after filtering NaN."""
        # Each row has at least one NaN, so after filtering, zero valid rows remain
        df = pd.DataFrame({"target": [1, np.nan, 0, np.nan], "score": [np.nan, 0.5, np.nan, 0.8]})
        metric_values = [0.5]

        # ENHANCED BUG #5 FIX: Also catches when filtering leaves zero valid rows
        with pytest.raises(
            ValueError, match="Cannot calculate statistics: no valid rows remain after removing NaN values"
        ):
            calculate_stats(df, "target", "score", "Sensitivity", metric_values)

    def test_mixed_nan_values(self):
        """Test calculate_stats() with mixed NaN values (some valid data)"""
        df = pd.DataFrame(
            {
                "target": [0, 1, np.nan, 1, 1, 0, 1, 0, np.nan, 0],
                "score": [0.1, 0.4, 0.35, np.nan, 0.7, 0.2, np.nan, 0.3, 0.6, 0.5],
            }
        )
        metric_values = [0.5]

        # With mixed NaN values, behavior depends on implementation
        # Either it should work (dropping NaNs) or fail cleanly
        try:
            stats = calculate_stats(df, "target", "score", "Sensitivity", metric_values)
            # If it succeeds, validate the stats are reasonable
            assert "AUROC" in stats
            assert 0 <= stats["AUROC"] <= 1
        except (ValueError, RuntimeError):
            # Or it fails cleanly with sklearn error
            pass


class TestGenerateAnalyticsData:
    def test_censor_threshold_below(self, fake_seismo):
        # Seismogram().dataframe has fewer rows than the censor_threshold
        result = generate_analytics_data(
            score_columns=["score1"],
            target_columns=["target1"],
            metric="Sensitivity",
            metric_values=[0.5, 0.7],
            censor_threshold=5,
        )
        # Expected None when data rows are below censor_threshold
        assert result is None

    def test_censor_threshold_above(self, fake_seismo):
        # Seismogram().dataframe has more rows than the censor_threshold
        result = generate_analytics_data(
            score_columns=["score1"],
            target_columns=["target1"],
            metric="Sensitivity",
            metric_values=[0.5, 0.7],
            censor_threshold=3,
        )
        assert result is not None
        assert not result.empty

    def test_censor_threshold_exact(self, fake_seismo):
        # Seismogram().dataframe has exactly the same number of rows as the censor_threshold
        result = generate_analytics_data(
            score_columns=["score1"],
            target_columns=["target1"],
            metric="Sensitivity",
            metric_values=[0.5, 0.7],
            censor_threshold=4,
        )
        assert result is None

    def test_per_context_does_not_leak_state_across_iterations(self, fake_seismo):
        # Ensure generate_analytics_data works correctly across multiple (score, target) pairs
        # without interference between iterations due to in-place data mutation.
        fake_seismo.event_aggregation_method = lambda target: "max"
        result = generate_analytics_data(
            score_columns=["score1", "score2"],
            target_columns=["event1_Value", "event2_Value"],
            metric="Sensitivity",
            metric_values=[0.5],
            per_context=True,
            censor_threshold=3,
        )

        # Should return a DataFrame with 4 rows (2 scores x 2 targets)
        assert result is not None
        assert len(result) == 4

        # Combining scores for each (score, target) pair should have two positives
        assert result["Positives"].tolist() == [2, 2, 2, 2]

    def test_generate_analytics_data_metric_differs_but_is_close(self, fake_seismo):
        fake_seismo.event_aggregation_method = lambda target: "max"

        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=1000)  # Balanced 0s and 1s
        y_score = rng.uniform(0, 1, size=1000)  # Uniform prediction scores between 0 and 1
        df = pd.DataFrame(
            {
                "target1": y_true,
                "score1": y_score,
            }
        )

        fake_seismo.dataframe = df

        result_low = generate_analytics_data(
            score_columns=["score1"],
            target_columns=["target1"],
            metric="Sensitivity",
            metric_values=[0.5],
            decimals=3,
            censor_threshold=1,
        )
        result_high = generate_analytics_data(
            score_columns=["score1"],
            target_columns=["target1"],
            metric="Sensitivity",
            metric_values=[0.5],
            decimals=5,
            censor_threshold=1,
        )

        assert result_low is not None and result_high is not None

        # Ensure both versions yield same structure
        assert list(result_low.columns) == list(result_high.columns)

        for col, atol in [("0.5_Threshold", 0.1), ("AUROC", 0.001)]:
            assert col in result_low.columns and col in result_high.columns

            val_low = result_low[col].iloc[0]
            val_high = result_high[col].iloc[0]

            # Values differ (better precision affects threshold resolution)
            assert (
                val_low != val_high
            ), "Differing precision expected, test seed was chosen to verify these two values are not equal"

            # But still close enough (numerically stable)
            assert np.isclose(val_low, val_high, atol=atol)

    def test_per_context_missing_columns(self, fake_seismo):
        """Test generate_analytics_data() with per_context=True but missing required columns"""
        # Remove entity_keys column to trigger error
        fake_seismo.dataframe = fake_seismo.dataframe.drop(columns=["entity"])

        # This should fail because entity_keys column is missing
        with pytest.raises((KeyError, ValueError)):
            generate_analytics_data(
                score_columns=["score1"],
                target_columns=["target1"],
                metric="Sensitivity",
                metric_values=[0.5],
                per_context=True,
                censor_threshold=1,
            )

    def test_invalid_cohort_dict_keys(self, fake_seismo):
        """Test generate_analytics_data() with invalid cohort_dict keys (non-existent columns)"""
        # Use a cohort column that doesn't exist in the dataframe
        invalid_cohort_dict = {"NonExistentColumn": ("A",)}

        # This should fail because the cohort column doesn't exist
        with pytest.raises((KeyError, ValueError)):
            generate_analytics_data(
                score_columns=["score1"],
                target_columns=["target1"],
                metric="Sensitivity",
                metric_values=[0.5],
                cohort_dict=invalid_cohort_dict,
                censor_threshold=1,
            )

    def test_empty_score_columns(self, fake_seismo):
        """Test generate_analytics_data() with empty score_columns list"""
        result = generate_analytics_data(
            score_columns=[],
            target_columns=["target1"],
            metric="Sensitivity",
            metric_values=[0.5],
            censor_threshold=1,
        )

        # Empty score_columns should result in empty or None result
        assert result is None or result.empty

    def test_empty_target_columns(self, fake_seismo):
        """Test generate_analytics_data() with empty target_columns list"""
        result = generate_analytics_data(
            score_columns=["score1"],
            target_columns=[],
            metric="Sensitivity",
            metric_values=[0.5],
            censor_threshold=1,
        )

        # Empty target_columns should result in empty or None result
        assert result is None or result.empty

    def test_both_empty_score_and_target_columns(self, fake_seismo):
        """Test generate_analytics_data() with both empty score_columns and target_columns"""
        result = generate_analytics_data(
            score_columns=[],
            target_columns=[],
            metric="Sensitivity",
            metric_values=[0.5],
            censor_threshold=1,
        )

        # Both empty should result in empty or None result
        assert result is None or result.empty
