import numpy as np
import pandas as pd
import pytest

import seismometer.table.fairness as undertest
from seismometer.data.performance import MetricGenerator


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
    def test_fairness_table_filters_values(self):
        data = sample_data()
        fake_metrics = MetricGenerator(["M1", "M2", "M3"], lambda x, names: {"M1": 1, "M2": 2, "M3": 3})
        table = undertest.fairness_table(data, fake_metrics, ["M1", "M2"], 0.1, {"Cohort": ["First", "Middle"]})
        assert "Last" not in table.value
        assert "M3" not in table.value

    def test_fairness_table_filters_values(self):
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
