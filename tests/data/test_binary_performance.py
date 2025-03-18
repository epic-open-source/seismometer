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
            "event2_Value": [0, 1, 0, 1],
            "event2_Time": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "event3_Value": [0, 2, 5, 1],
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
