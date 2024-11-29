import numpy as np
import pytest
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from seismometer.data.binary_performance import calculate_stats


class TestCalculateStats:
    @pytest.mark.parametrize("metric", ["Sensitivity", "Specificity", "Flagged"])
    def test_basic(self, metric):
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5])
        metric_values = [0.5, 0.7]
        metrics = ["Sensitivity", "Specificity", "Flag\u00A0Rate", "PPV", "Accuracy"]
        metrics.remove(metric.replace("Flagged", "Flag\u00A0Rate"))
        stats = calculate_stats(y_true, y_pred, metric, metric_values)
        assert stats["AUROC"] == roc_auc_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        assert np.allclose(stats["AUPRC"], auc(recall, precision), rtol=0.01)
        assert all(f"{val}_{metric}" in stats for val in metric_values for metric in metrics)
        assert all(col in stats for col in ["Positives", "Prevalence"])

    def test_invalid_metric(self):
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5])
        metric_values = [0.5, 0.7]
        with pytest.raises(
            ValueError,
            match="Invalid metric name: InvalidMetric. The metric needs to be one of: "
            "\\['Sensitivity', 'Specificity', 'Flagged', 'Threshold'\\]",
        ):
            calculate_stats(y_true, y_pred, "InvalidMetric", metric_values)

    def test_positives_prevalence(self):
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5])
        metric_values = [0.5, 0.7]
        stats = calculate_stats(y_true, y_pred, "sensitivity", metric_values)
        assert stats["Positives"] == np.sum(y_true)
        assert stats["Prevalence"] == np.mean(y_true)

    @pytest.mark.parametrize("metric", ["Sensitivity", "Specificity", "Flagged"])
    def test_metric_values_decimals(self, metric):
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5])
        metric_values = [0.534, 0.7, 0.100032, 0.1 + 0.3, 0.00002]
        expected_metric_values = [0.53, 0.7, 0.1, 0.4, 0]
        metrics = ["Sensitivity", "Specificity", "Flag\u00A0Rate", "PPV", "Accuracy"]
        metrics.remove(metric.replace("Flagged", "Flag\u00A0Rate"))
        stats = calculate_stats(y_true, y_pred, metric, metric_values, decimals=2)
        assert all(f"{val}_{metric}" in stats for val in expected_metric_values for metric in metrics)

    def test_metrics_to_display(self):
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5])
        metric_values = [0.5, 0.7]
        stats = calculate_stats(
            y_true, y_pred, "Flagged", metric_values, metrics_to_display=["Sensitivity", "Specificity", "Prevalence"]
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
            ("Flagged", np.array([100.0, 50.0, 35.0, 10.0])),
        ],
    )
    def test_computed_threshold_basic(self, metric, expected_thresholds):
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5])
        metric_values = [0, 0.5, 0.7, 1]
        stats = calculate_stats(y_true, y_pred, metric, metric_values)
        computed_thresholds = [stats[f"{val}_Threshold"] for val in metric_values]
        assert np.array_equal(computed_thresholds, expected_thresholds)

    @pytest.mark.parametrize(
        "metric, expected_thresholds",
        [
            ("Sensitivity", np.array([100.0, 100.0, 100.0, 100.0])),
            ("Specificity", np.array([10.0, 40.0, 40.0, 100.0])),
            ("Flagged", np.array([100.0, 40.0, 30.0, 10.0])),
        ],
    )
    def test_computed_threshold_edge_cases_all_zeroes(self, metric, expected_thresholds):
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        metric_values = [0, 0.5, 0.7, 1]
        stats = calculate_stats(y_true, y_pred, metric, metric_values, metrics_to_display=["Threshold"])
        computed_thresholds = [stats[f"{val}_Threshold"] for val in metric_values]
        assert np.array_equal(computed_thresholds, expected_thresholds)

    @pytest.mark.parametrize(
        "metric, expected_thresholds",
        [
            ("Sensitivity", np.array([100.0, 40.0, 30.0, 10.0])),
            ("Specificity", np.array([100.0, 100.0, 100.0, 100.0])),
            ("Flagged", np.array([100.0, 40.0, 30.0, 10.0])),
        ],
    )
    def test_computed_threshold_edge_cases_all_ones(self, metric, expected_thresholds):
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        metric_values = [0, 0.5, 0.7, 1]
        stats = calculate_stats(y_true, y_pred, metric, metric_values, metrics_to_display=["Threshold"])
        computed_thresholds = [stats[f"{val}_Threshold"] for val in metric_values]
        assert np.array_equal(computed_thresholds, expected_thresholds)
