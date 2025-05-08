from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import seismometer.data.performance as undertest
from seismometer.data.confidence.calculations import ValueWithCI, _RocRegionResults
from seismometer.data.confidence.parameters import _SUPPORTED_PLOTS

ALL_STATS = [undertest.THRESHOLD] + undertest.STATNAMES + undertest.PERCENTS + ["NNT@0.333"]


class TestAssertValidPerf:
    def test_no_stats_fails(self):
        df = pd.DataFrame({"col1": [1, 2, 3]})
        with pytest.raises(ValueError):
            undertest.assert_valid_performance_metrics_df(df)

    @pytest.mark.parametrize(
        "col_list",
        [
            pytest.param(ALL_STATS, id="All stats"),
            pytest.param(["Threshold", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV"], id="Six required"),
            pytest.param(["another"] + ALL_STATS, id="Extra column"),
        ],
    )
    def test_default_allstats_passes(self, col_list):
        df = pd.DataFrame(columns=col_list)
        try:
            undertest.assert_valid_performance_metrics_df(df)
        except BaseException:
            pytest.fail("Assertion did not pass")

    def test_missing_stats_fails(self):
        only_two = ["Threshold", "Accuracy"]
        with pytest.raises(ValueError):
            undertest.assert_valid_performance_metrics_df(pd.DataFrame(columns=only_two))

    def test_custom_stats_prioritized(self):
        only_two = ["Threshold", "Accuracy"]
        try:
            undertest.assert_valid_performance_metrics_df(pd.DataFrame(columns=only_two), needs_columns=only_two)
        except BaseException:
            pytest.fail("Assertion did not pass")


def ci_testcase0():
    # Inputs
    model = pd.DataFrame({"truth": [0, 1, 0, 1], "output": [0.4, 0.5, 0.6, 0.7]})

    stats = pd.DataFrame(
        data=np.vstack(
            (
                # TP,FP,TN,FN, Sens, PPV
                [
                    [
                        0,
                        0,
                        2,
                        2,
                        0.0,
                        0.0,
                    ]
                ]
                * 30,
                [
                    [
                        1,
                        0,
                        2,
                        1,
                        0.5,
                        1,
                    ]
                ]
                * 10,
                [
                    [
                        1,
                        1,
                        1,
                        1,
                        0.5,
                        0.5,
                    ]
                ]
                * 10,
                [
                    [
                        2,
                        1,
                        1,
                        0,
                        1,
                        2 / 3,
                    ]
                ]
                * 10,
                [
                    [
                        2,
                        2,
                        0,
                        0,
                        1,
                        0.5,
                    ]
                ]
                * 41,
            )
        ),
        columns=["TP", "FP", "TN", "FN", "Sensitivity", "PPV"],
    )
    stats["Threshold"] = np.arange(100, -1, -1)

    # Outputs
    a = 0.03967721
    b = 0.96032279
    roc = {
        "Threshold": [np.inf, 0.7, 0.6, 0.5, 0.4],
        "interval": ValueWithCI(0.75, 0.20847047, 1),
        "TPR": [0, 0.5, 0.5, 1, 1],
        "FPR": [0, 0, 0.5, 0.5, 1],
        "region": _RocRegionResults([0, 0, 0, a, a], [b, b, 1, 1, 1], [b, 1, 1, 1, 1], [0, 0, 0, 0, a]),
    }
    pr = {"interval": ValueWithCI(0.54166666, 0.14188138, 0.89415081)}
    return (stats, model.truth.values, model.output.values, roc, pr)


@pytest.mark.parametrize("force", [True, False])
class TestCalCi:
    @pytest.mark.parametrize(
        "key",
        [
            "Threshold",
            "TPR",
            "FPR",
            "region",
            "interval",
        ],
    )
    @pytest.mark.parametrize("input_percentages", [True, False])
    def test_calc_ci_maps_roc_values(self, key, force, input_percentages):
        stats, truth, output, expected, _ = ci_testcase0()
        if input_percentages:  # convert the input argument to percentages
            output = output * 100

        actual = undertest.calculate_eval_ci(stats, truth, output, force_percentages=force)

        # sklearn update 1.3.0 changed 0,0 threshold from 1+max to inf
        if key == "Threshold":
            assert actual["roc"][key][0] > 1
            actual["roc"][key][0] = np.inf

        if force or input_percentages:  # expect 0-100 thresholds
            expected["Threshold"] = [np.inf, 70, 60, 50, 40]

        for actual_val, expected_val in zip(actual["roc"][key], expected[key]):
            assert np.isclose(actual_val, expected_val).all()

    def test_calc_ci_maps_pr(self, force):
        key = "interval"
        stats, truth, output, _, expected = ci_testcase0()

        actual = undertest.calculate_eval_ci(stats, truth, output, force_percentages=force)

        for actual_val, expected_val in zip(actual["pr"][key], expected[key]):
            assert np.isclose(actual_val, expected_val).all()

    @pytest.mark.parametrize("conf_val", [0.99, 0.95, 0.90])
    def test_calc_ci_returns_conf(self, conf_val, force):
        stats, truth, output, _, _ = ci_testcase0()
        actual = undertest.calculate_eval_ci(stats, truth, output, conf_val, force_percentages=force)
        conf = actual["conf"]
        for plot in _SUPPORTED_PLOTS:
            assert "level" in conf[plot] and conf[plot]["level"] == conf_val

    def test_no_truth_returns_conf(self, force):
        stats, truth, output, _, _ = ci_testcase0()
        actual = undertest.calculate_eval_ci(stats, None, output, 0.7, force_percentages=force)

        for plot in _SUPPORTED_PLOTS:
            assert "level" in actual[0][plot] and actual[0][plot]["level"] == 0.7

        for i in [1, 2, 3]:
            assert actual[i] is None

    def test_no_output_returns_conf(self, force):
        stats, truth, output, _, _ = ci_testcase0()
        actual = undertest.calculate_eval_ci(stats, truth, None, 0.7, force_percentages=force)

        for plot in _SUPPORTED_PLOTS:
            assert "level" in actual[0][plot] and actual[0][plot]["level"] == 0.7

        for i in [1, 2, 3]:
            assert actual[i] is None


class TestMetricGenerator:
    @pytest.mark.parametrize(
        "errorType,errorStr,args",
        [
            pytest.param(TypeError, "metric_fn", [["test_metric"]], id="No function"),
            pytest.param(ValueError, "metric_names", [[], lambda x: x], id="No metrics"),
            pytest.param(ValueError, "metric_fn", [["metric"], "not_callable"], id="Not Callable"),
            pytest.param(ValueError, "Reserved metric", [["metric", "Count"], lambda x: x], id="Count is reserved"),
        ],
    )
    def test_generate_metrics_init_fails(self, errorType, errorStr, args):
        with pytest.raises(errorType) as error:
            undertest.MetricGenerator(*args)
        assert errorStr in str(error.value)

    def test_generate_metrics_init_correctly(self):
        metric = undertest.MetricGenerator(["test_metric"], lambda data, names: {"test_metric": 1})
        assert metric.metric_names == ["test_metric"]
        assert metric(ci_testcase0()) == {"test_metric": 1}

    def test_generate_metrics_empty_dataframe(self):
        metric = undertest.MetricGenerator(["test_metric"], lambda data, names: {"test_metric": 1})
        assert metric.metric_names == ["test_metric"]
        assert metric(pd.DataFrame()) == {"test_metric": np.nan}

    def test_generate_named_metrics(self):
        metric = undertest.MetricGenerator(["metric1", "metric2"], lambda data, names: {name: 1 for name in names})
        assert metric(ci_testcase0(), ["metric2"]) == {"metric2": 1}
        with pytest.raises(ValueError) as error:
            metric(ci_testcase0(), ["metric3"])
        assert "metric3" in str(error.value)

    def test_generate_metrics_with_kwargs(self):
        def metric_fn(data, metric_names: list[str], *, special: int = 2):
            return {name: special for name in metric_names}

        metric = undertest.MetricGenerator(["metric1", "metric2"], metric_fn=metric_fn)
        assert metric(ci_testcase0(), ["metric1"], special=3) == {"metric1": 3}


class TestBinaryMetricGenerator:
    def test_binary_metric_generator_init(self):
        metrics = undertest.BinaryClassifierMetricGenerator()
        assert metrics.rho == 1 / 3
        assert metrics.metric_names == ALL_STATS[1:]

    @mock.patch.object(
        undertest.BinaryClassifierMetricGenerator,
        "calculate_binary_stats",
        return_value=(
            pd.DataFrame.from_records(
                [{"Accuracy": 0.8, "Sensitivity": 0.7, "Specificity": 0.6, "PPV": 0.5, "NPV": 0.4}], index=[20]
            ),
            None,
        ),
    )
    def test_binary_metric_generation(self, mock_stats):
        metrics = undertest.BinaryClassifierMetricGenerator(rho=0.01)
        data = pd.DataFrame({"fake": [0, 1, 0, 1], "data": [0.4, 0.5, 0.6, 0.7]})
        result = metrics(data, ["Accuracy", "PPV"], target_col="TARGET", score_col="SCORE", score_threshold=0.2)
        assert result == {"Accuracy": 0.8, "PPV": 0.5}
        mock_stats.assert_called_once_with(data, "TARGET", "SCORE", ["Accuracy", "PPV"])

    def test_calc_ci_maps_roc_values(self):
        metrics = undertest.BinaryClassifierMetricGenerator(rho=0.2)
        _, truth, output, _, _ = ci_testcase0()
        data = pd.DataFrame({"truth": truth, "output": output})
        expected = {
            "Flag Rate": 0.25,
            "Accuracy": 0.75,
            "Sensitivity": 0.5,
            "Specificity": 1.0,
            "PPV": 1.0,
            "NPV": 0.66667,
            "LR+": np.inf,
            "NetBenefitScore": 0.25,
            "TP": 1.0,
            "FP": 0.0,
            "TN": 2.0,
            "FN": 1.0,
            "NNE": 1.0,
            "NNT@0.2": 5.0,
            "TP (%)": 25.0,
            "FP (%)": 0.0,
            "TN (%)": 50.0,
            "FN (%)": 25.0,
        }
        actual = metrics(
            data, metric_names=metrics.metric_names, target_col="truth", score_col="output", score_threshold=0.7
        )

        assert actual == expected

    def test_binary_metric_generator_with_threshold_precision(self):
        metrics = undertest.BinaryClassifierMetricGenerator()
        y_true = np.array([1, 0])
        y_prob = np.array([0.9, 0.2])
        df = pd.DataFrame({"truth": y_true, "score": y_prob})

        # Low precision: coarse thresholds
        low_precision = metrics.calculate_binary_stats(df, "truth", "score", ["Accuracy"], threshold_precision=0)[0]

        # High precision: fine-grained thresholds
        high_precision = metrics.calculate_binary_stats(df, "truth", "score", ["Accuracy"], threshold_precision=2)[0]

        assert len(high_precision) > len(low_precision)
        assert "Accuracy" in high_precision.columns

    def test_score_threshold_row_lookup(self):
        metrics = undertest.BinaryClassifierMetricGenerator()
        y_true = np.array([1, 0])
        y_prob = np.array([0.55, 0.25])
        df = pd.DataFrame({"truth": y_true, "score": y_prob})

        stats, _ = metrics.calculate_binary_stats(df, "truth", "score", ["Accuracy"])
        assert 55 in stats.index  # score_threshold=0.55 → index 55

    def test_overall_stats_include_auc_with_precision(self):
        metrics = undertest.BinaryClassifierMetricGenerator()
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.7, 0.4, 0.1])
        df = pd.DataFrame({"truth": y_true, "score": y_prob})

        _, overall = metrics.calculate_binary_stats(df, "truth", "score", ["Accuracy"], threshold_precision=2)

        assert "AUROC" in overall
        assert 0.0 <= overall["AUROC"] <= 1.0
        assert "AUPRC" in overall
        assert 0.0 <= overall["AUPRC"] <= 1.0

    def test_auc_precision_effect_on_larger_data(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=1000)
        y_prob = rng.uniform(0, 1, size=1000)

        true_auc = roc_auc_score(y_true, y_prob)

        # Coarse threshold grid
        stats_coarse = undertest.calculate_bin_stats(y_true, y_prob, threshold_precision=0)
        auc_coarse = undertest.auc(1 - stats_coarse["Specificity"], stats_coarse["Sensitivity"])

        # Fine threshold grid
        stats_fine = undertest.calculate_bin_stats(y_true, y_prob, threshold_precision=1)
        auc_fine = undertest.auc(1 - stats_fine["Specificity"], stats_fine["Sensitivity"])

        # Assertions
        assert abs(auc_fine - true_auc) < abs(auc_coarse - true_auc)
        assert abs(auc_fine - true_auc) < 0.001
        assert abs(auc_coarse - true_auc) < 0.01
