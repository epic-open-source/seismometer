import numpy as np
import pandas as pd
import pytest

import seismometer.report.fairness as undertest


class TestSortKeys:
    def test_sort_keys(self):
        data = pd.DataFrame(
            {
                "Cohort": ["Last", "First", "Middle", "Last", "First", "Middle", "Last", "First", "Middle"],
                "Class": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
                "Count": [1, np.nan, 3, 4, 5, undertest.FairnessIcons.UNKNOWN.value, 7, 8, 9],
            }
        )
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


class TestFairnessTable:
    @pytest.mark.parametrize(
        "data,output",
        [
            ("1", "2"),
            ("2", "3"),
        ],
    )
    def test_fairness_table(self, data, output):
        # placeholder
        assert data + 1 == output
