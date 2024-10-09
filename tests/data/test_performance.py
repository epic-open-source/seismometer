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
