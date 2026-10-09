import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import seismometer.report.profile_report as undertest
from seismometer.data.profile import AlertType, profile_dataframe

sys.path.insert(0, str(Path(__file__).parents[1] / "data"))
from profile_cases import make_profile_frame  # noqa: E402

GOLDEN = Path(__file__).parents[1] / "resources" / "profile" / "ydata_golden.json"


@pytest.fixture(scope="module")
def profile():
    return profile_dataframe(make_profile_frame(), title="Profile")


def plain_text(markup: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", markup))


class TestFormatting:
    @pytest.mark.parametrize(
        "value,expected", [(0.0004, "< 0.1%"), (0.9996, "> 99.9%"), (0.5, "50.0%"), (0, "0.0%"), (1, "100.0%")]
    )
    def test_fmt_percent(self, value, expected):
        assert undertest.fmt_percent(value) == expected

    @pytest.mark.parametrize(
        "value,precision,expected",
        [
            (4.4123, 2, "4.4"),
            (441234.5, 2, "4.4 × 10<sup>5</sup>"),
            (0.000123, 2, "0.00012"),
            (1.2e-7, 2, "1.2 × 10<sup>-7</sup>"),
            (1.686407099123, 10, "1.686407099"),
            (None, 2, "N/A"),
        ],
    )
    def test_fmt_numeric(self, value, precision, expected):
        assert undertest.fmt_numeric(value, precision) == expected

    def test_fmt_only_formats_python_numbers(self):
        assert undertest.fmt(1.234567890123) == "1.23456789"
        assert undertest.fmt(np.int64(12345678901)) == "12345678901"
        assert undertest.fmt("<b>") == "&lt;b&gt;"

    @pytest.mark.parametrize("value,expected", [(512, "512.0 B"), (17487804, "16.7 MiB"), (2048, "2.0 KiB")])
    def test_fmt_bytesize(self, value, expected):
        assert undertest.fmt_bytesize(value) == expected

    @pytest.mark.parametrize(
        "seconds,expected", [(1.2, "1.2 second"), (0.5, "0.5 seconds"), (75, "1 minute and 15 seconds")]
    )
    def test_fmt_timespan(self, seconds, expected):
        assert undertest.fmt_timespan(seconds) == expected


class TestAlerts:
    def test_wording_matches_ydata_profiling(self, profile):
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        expected = [a["text"] for a in golden["alerts"] if a["text"] is not None]
        actual = [plain_text(undertest.alert_html(a)) for a in profile.alerts if a.alert_type != AlertType.REJECTED]
        assert actual == expected

    def test_column_names_are_escaped(self):
        profile = profile_dataframe(pd.DataFrame({"<b>x": [0] * 9 + [1]}))
        rendered = undertest.alert_html(profile.alerts[0])
        assert "&lt;b&gt;x" in rendered
        assert "<b>x" not in rendered


class TestProfileReport:
    def test_single_report(self, profile):
        page = undertest.ProfileReport([profile], title="Feature Report").to_html()
        for text in [
            "Feature Report",
            "Dataset statistics",
            "Number of observations",
            "Variable types",
            "Alerts (21)",
            "Real number (&Ropf;)",
            "Quantile statistics",
            "Common values",
            "Extreme values",
            "Common Values (Plot)",
            "Most occurring characters",
            "Other values (",
            "(Missing)",
            "float_mixed",
        ]:
            assert text in page, text
        assert page.count('class="variable"') == len(profile.variables)

    def test_values_are_escaped(self):
        profile = profile_dataframe(pd.DataFrame({"text": ["<script>alert(1)</script>", "b"]}))
        page = undertest.ProfileReport([profile], title="T").to_html()
        assert "<script>alert(1)" not in page
        assert "&lt;script&gt;" in page

    def test_does_not_modify_profile(self, profile):
        _ = undertest.ProfileReport([profile], title="T").to_html()
        assert all("alert_fields" not in description for description in profile.variables.values())

    def test_comparison(self):
        df = make_profile_frame()
        left, right = profile_dataframe(df.iloc[:600], "Left"), profile_dataframe(df.iloc[600:], "Right")
        page = undertest.ProfileReport([left, right], title="Compare", labels=["Target=0", "Target=1"]).to_html()
        assert page.count("Target=0") > 10
        assert "Target=1" in page
        assert page.count('class="variable"') == df.shape[1]

    def test_comparison_with_different_types(self):
        left = profile_dataframe(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), "Left")
        right = profile_dataframe(pd.DataFrame({"x": [np.nan, np.nan]}), "Right")
        page = undertest.ProfileReport([left, right], title="Compare").to_html()
        assert "Left: Real number (&Ropf;)" in page
        assert "Right: Unsupported" in page

    def test_to_file(self, profile, tmp_path):
        path = tmp_path / "report.html"
        undertest.ProfileReport([profile], title="T").to_file(path)
        assert path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_rejects_more_than_two_profiles(self, profile):
        with pytest.raises(ValueError):
            undertest.ProfileReport([profile] * 3, title="T")
