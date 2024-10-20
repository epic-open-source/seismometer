from unittest import mock

import numpy as np
import pandas as pd
import pytest

import seismometer.data.performance as undertest
from seismometer.data.confidence.calculations import ValueWithCI, _RocRegionResults
from seismometer.data.confidence.parameters import _SUPPORTED_PLOTS

ALL_STATS = [undertest.THRESHOLD] + undertest.STATNAMES + ["NNT@0.333"]


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
    def test_calc_ci_maps_roc_values(self, key):
        stats, truth, output, expected, _ = ci_testcase0()
        actual = undertest.calculate_eval_ci(stats, truth, output)

        # sklearn update 1.3.0 changed 0,0 threshold from 1+max to inf
        if key == "Threshold":
            assert actual["roc"][key][0] > 1
            actual["roc"][key][0] = np.inf

        for actual_val, expected_val in zip(actual["roc"][key], expected[key]):
            assert np.isclose(actual_val, expected_val).all()

    def test_calc_ci_maps_pr(self):
        key = "interval"
        stats, truth, output, _, expected = ci_testcase0()
        actual = undertest.calculate_eval_ci(stats, truth, output)

        for actual_val, expected_val in zip(actual["pr"][key], expected[key]):
            assert np.isclose(actual_val, expected_val).all()

    @pytest.mark.parametrize("conf_val", [0.99, 0.95, 0.90])
    def test_calc_ci_returns_conf(self, conf_val):
        stats, truth, output, _, _ = ci_testcase0()
        actual = undertest.calculate_eval_ci(stats, truth, output, conf_val)
        conf = actual["conf"]
        for plot in _SUPPORTED_PLOTS:
            assert "level" in conf[plot] and conf[plot]["level"] == conf_val

    def test_no_truth_returns_conf(self):
        stats, truth, output, _, _ = ci_testcase0()
        actual = undertest.calculate_eval_ci(stats, None, output, 0.7)

        for plot in _SUPPORTED_PLOTS:
            assert "level" in actual[0][plot] and actual[0][plot]["level"] == 0.7

        for i in [1, 2, 3]:
            assert actual[i] is None

    def test_no_output_returns_conf(self):
        stats, truth, output, _, _ = ci_testcase0()
        actual = undertest.calculate_eval_ci(stats, truth, None, 0.7)

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
        assert metric(pd.DataFrame()) == {"test_metric": np.NaN}

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

    @mock.patch(
        "seismometer.data.performance.calculate_binary_stats",
        return_value={"Accuracy": 0.8, "Sensitivity": 0.7, "Specificity": 0.6, "PPV": 0.5, "NPV": 0.4},
    )
    def test_binary_metric_generation(self, mock_stats):
        metrics = undertest.BinaryClassifierMetricGenerator(rho=0.01)
        data = pd.DataFrame({"fake": [0, 1, 0, 1], "data": [0.4, 0.5, 0.6, 0.7]})
        result = metrics(data, ["Accuracy", "PPV"], target_col="TARGET", score_col="SCORE", score_threshold=0.2)
        assert result == {"Accuracy": 0.8, "PPV": 0.5}
        mock_stats.assert_called_once_with(data, "TARGET", "SCORE", 0.2, rho=0.01)


class TestBinaryStats:
    def test_calc_ci_maps_roc_values(self):
        _, truth, output, _, _ = ci_testcase0()
        data = pd.DataFrame({"truth": truth, "output": output})
        expected = {
            "Threshold": 70.0,
            "TP": 1.0,
            "FP": 0.0,
            "TN": 2.0,
            "FN": 1.0,
            "Accuracy": 0.75,
            "Sensitivity": 0.5,
            "Specificity": 1.0,
            "PPV": 1.0,
            "NPV": 0.66667,
            "Flagged": 0.25,
            "LR+": np.inf,
            "NetBenefitScore": 0.25,
            "NNT@0.2": 3.0,
        }
        actual = undertest.calculate_binary_stats(data, "truth", "output", 0.7, 0.2)

        assert actual == expected
