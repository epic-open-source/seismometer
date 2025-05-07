from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from ipywidgets import HTML

import seismometer.table.fairness as undertest
from seismometer.data.performance import MetricGenerator

# ---- Fixtures ----


def sample_data():
    return pd.DataFrame(
        {
            "Cohort": ["Last", "First", "Middle", "Last", "First", "Middle", "Last", "First", "Middle"],
            "Class": ["L1", "Fn", "M3", "L4", "F5", "M?", "L7", "F8", "M9"],
            "Count": [1, np.nan, 3, 4, 5, undertest.FairnessIcons.UNKNOWN.value, 7, 8, 9],
        }
    )


def large_dataset_data():
    return pd.DataFrame(
        {
            "Cohort": ["Last", "First", "Middle"] * 10 + ["First", "Middle", "Middle"] * 10 + ["Middle"] * 30,
            "Category": ["L1", "F3", "M3", "L4", "F5", "M8", "L7", "F8", "M9"] * 10,
            "Number": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 10,
        }
    )


# ---- Test Classes ----


class TestSortKeys:
    def test_sort_keys(self):
        data = sample_data()
        data = undertest.sort_fairness_table(data, ["First", "Middle", "Last"])
        assert data["Cohort"].tolist() == [
            "First",
            "First",
            "First",
            "Middle",
            "Middle",
            "Middle",
            "Last",
            "Last",
            "Last",
        ]
        assert data["Class"].tolist() == ["F8", "F5", "Fn", "M9", "M3", "M?", "L7", "L4", "L1"]
        assert data["Count"].tolist() == [8, 5, np.nan, 9, 3, "❔", 7, 4, 1]


class TestFairnessTable:
    def test_fairness_table_filters_values_small(self):
        data = sample_data()
        fake_metrics = MetricGenerator(["M1", "M2", "M3"], lambda x, names: {"M1": 1, "M2": 2, "M3": 3})
        table = undertest.fairness_table(data, fake_metrics, ["M1", "M2"], 0.1, {"Cohort": ["First", "Middle"]})
        assert "Last" not in table.value
        assert "M3" not in table.value

    def test_fairness_table_filters_values_large(self):
        data = large_dataset_data()
        fake_metrics = MetricGenerator(
            ["M1", "M2", "M3"],
            lambda x, names: {"M1": x.Number.mean(), "M2": x.Category.str.count("F8").sum(), "M3": 3},
        )
        table = undertest.fairness_table(data, fake_metrics, ["M1", "M2"], 0.1, {"Cohort": ["First", "Middle"]})
        assert "Last" not in table.value
        assert "M3" not in table.value
        assert "60" in table.value
        assert "🔹  5.43" in table.value
        assert "🔻  3.00" in table.value
        assert "🔻  4.35" in table.value
        assert "🔹  7.00" in table.value


class TestFairnessIcons:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.5, undertest.FairnessIcons.CRITICAL_LOW),
            (0.75, undertest.FairnessIcons.WARNING_LOW),
            (0.9, undertest.FairnessIcons.GOOD),
            (1.0, undertest.FairnessIcons.DEFAULT),
            (1.1, undertest.FairnessIcons.GOOD),
            (1.26, undertest.FairnessIcons.WARNING_HIGH),
            (1.71, undertest.FairnessIcons.CRITICAL_HIGH),
            (None, undertest.FairnessIcons.UNKNOWN),
        ],
    )
    def test_values(self, value, expected):
        assert undertest.FairnessIcons.get_fairness_icon(value) == expected


class TestFairnessLegend:
    def test_get_fairness_legend_contains_expected_text(self):
        legend = undertest.FairnessIcons.get_fairness_legend(limit=0.2, open=False, censor_threshold=5)
        legend_str = str(legend)
        assert "Within 20.00%" in legend_str
        assert "fewer than 5 observations" in legend_str


class TestFairnessTableValidation:
    def test_fairness_table_raises_value_error_on_bad_inputs(self):
        df = pd.DataFrame({"group": ["A", "B"], "value": [1, 2]})
        dummy_metric = MetricGenerator(["M1"], lambda x, names: {"M1": 1})

        with pytest.raises(ValueError, match="Fairness ratio must be greater than 0"):
            undertest.fairness_table(df, dummy_metric, ["M1"], 0.0, {"group": ("A", "B")})

        with pytest.raises(ValueError, match="No cohorts provided"):
            undertest.fairness_table(df, dummy_metric, ["M1"], 0.25, None)

    def test_fairness_table_censors_small_groups(self):
        df = pd.DataFrame(
            {
                "group": ["A", "B", "A", "B"],
                "val": [1, 2, 1, 2],
            }
        )
        metric_fn = MetricGenerator(["M1"], lambda x, names: {"M1": 1.0})
        cohort_dict = {"group": ("A", "B")}

        result: HTML = undertest.fairness_table(df, metric_fn, ["M1"], 0.25, cohort_dict, censor_threshold=10)
        assert "❔" in result.value or "--" in result.value


class TestFairnessWrappers:
    def test_binary_metrics_fairness_table_runs(self, monkeypatch):
        sg_mock = Mock()
        sg_mock.dataframe = pd.DataFrame({"group": ["A", "B"], "val": [1, 2]})
        sg_mock.entity_keys = []
        sg_mock.predict_time = None
        sg_mock.censor_threshold = 10
        sg_mock.event_aggregation_method.return_value = "mean"
        monkeypatch.setattr("seismometer.seismogram.Seismogram", lambda: sg_mock)

        gen = MetricGenerator(["M1"], lambda x, names: {"M1": 1})
        html_result = undertest.binary_metrics_fairness_table(
            gen, ["M1"], {"group": ("A", "B")}, 0.25, "target", "score", 0.5
        )
        assert isinstance(html_result, HTML)

    def test_custom_metrics_fairness_table_runs(self, monkeypatch):
        sg_mock = Mock()
        sg_mock.dataframe = pd.DataFrame({"group": ["A", "B"], "val": [1, 2]})
        sg_mock.available_cohort_groups = {"group": ("A", "B")}
        sg_mock.censor_threshold = 10
        monkeypatch.setattr("seismometer.seismogram.Seismogram", lambda: sg_mock)

        gen = MetricGenerator(["M1"], lambda x, names: {"M1": 1})
        html_result = undertest.custom_metrics_fairness_table(gen, ["M1"], None, 0.25)
        assert isinstance(html_result, HTML)


class TestFairnessOptionsWidget:
    def test_fairness_options_widget_value_behavior(self):
        metric_names = ("M1", "M2")
        cohort_dict = {"group": ("A", "B")}
        widget = undertest.FairnessOptionsWidget(metric_names, cohort_dict, fairness_ratio=0.3)

        # Initial state: cohort_list is empty
        val = widget.value
        assert list(val["metric_list"]) == list(metric_names)
        assert val["cohort_list"] == {}  # no selection yet
        assert widget.cohorts == cohort_dict  # fallback works

        # Simulate user selecting both values
        widget.cohort_list.value = {"group": ("A", "B")}
        widget._on_value_changed()

        updated_val = widget.value
        assert updated_val["cohort_list"] == {"group": ("A", "B")}  # now it reflects user input

        # Test enabling/disabling
        widget.disabled = True
        assert widget.metric_list.disabled
        assert widget.cohort_list.disabled
        assert widget.fairness_slider.disabled

        widget.disabled = False
        assert not widget.metric_list.disabled
