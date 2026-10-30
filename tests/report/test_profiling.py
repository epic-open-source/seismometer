import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from IPython.display import HTML as IPyHTML
from IPython.display import IFrame

from seismometer.core.io import slugify
from seismometer.data.profile import Alert, AlertType
from seismometer.report.alerting import ParsedAlert, ParsedAlertList
from seismometer.report.profiling import (
    ComparisonReportWrapper,
    ReportWrapper,
    SingleReportWrapper,
    filter_unsupported_columns,
)


class DummyReportWrapper(ReportWrapper):
    def __init__(self):
        self._html_path = Path("fake/path/report.html")
        self._title = "My Report"
        self._report = MagicMock()


@pytest.fixture
def dummy_wrapper():
    return DummyReportWrapper()


@pytest.fixture
def example_dfs():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    return df, df.copy()


@pytest.fixture
def small_df():
    return pd.DataFrame(
        {
            "text": ["a", "b"],
            "number": [1, 2],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )


class TestFilterUnsupportedColumns:
    def test_removes_datetime_columns(self):
        df = pd.DataFrame(
            {
                "string_col": ["a", "b"],
                "int_col": [1, 2],
                "float_col": [1.1, 2.2],
                "date_col": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            }
        )

        result = filter_unsupported_columns(df)

        assert "date_col" not in result.columns
        assert set(result.columns) == {"string_col", "int_col", "float_col"}

    def test_all_supported_columns_remain(self):
        df = pd.DataFrame(
            {
                "col1": ["x", "y"],
                "col2": [10, 20],
                "col3": [3.14, 2.71],
            }
        )

        result = filter_unsupported_columns(df)
        pd.testing.assert_frame_equal(result, df)

    def test_removes_only_supported_datetime_types(self):
        df = pd.DataFrame(
            {
                "datetime64_col": pd.date_range("2020-01-01", periods=3),
                "object_col": ["2020-01-01", "2020-01-02", "2020-01-03"],  # remains
            }
        )

        result = filter_unsupported_columns(df)

        assert "datetime64_col" not in result.columns
        assert "object_col" in result.columns


class TestReportWrapper:
    def test_display_report_inline(self, dummy_wrapper):
        with patch("seismometer.report.profiling.display") as mock_display:
            dummy_wrapper.display_report(inline=True)
            mock_display.assert_called_once()
            iframe = mock_display.call_args[0][0]
            assert isinstance(iframe, IFrame)
            assert re.search(r"fake[\\/]+path[\\/]+report\.html$", iframe.src)

    def test_display_report_link(self, dummy_wrapper):
        with patch("seismometer.report.profiling.display") as mock_display:
            dummy_wrapper.display_report(inline=False)
            mock_display.assert_called_once()
            link_html = mock_display.call_args[0][0]
            assert isinstance(link_html, IPyHTML)
            assert "Open My Report" in link_html.data

    def test_save_report_calls_to_file(self, dummy_wrapper):
        dummy_wrapper.save_report()
        dummy_wrapper._report.to_file.assert_called_once_with(dummy_wrapper._html_path)


class TestComparisonReportWrapper:
    @patch("seismometer.report.profiling.Path.is_file", return_value=False)
    @patch("seismometer.report.profiling.ComparisonReportWrapper.save_report")
    @patch("seismometer.report.profiling.ComparisonReportWrapper.generate_report")
    def test_init_generates_report_if_file_missing(
        self, mock_generate, mock_save, mock_is_file, example_dfs, tmp_path
    ):
        l_df, r_df = example_dfs
        wrapper = ComparisonReportWrapper(
            l_df=l_df,
            r_df=r_df,
            output_path=tmp_path,
            l_title="L",
            r_title="R",
            exclude_cols=[],
        )
        assert "L" in wrapper._title
        assert "R" in wrapper._title
        assert wrapper._html_path.name.endswith(".html")
        mock_generate.assert_called_once()
        mock_save.assert_called_once()

    @patch("seismometer.report.profiling.Path.is_file", return_value=True)
    @patch("seismometer.report.profiling.ComparisonReportWrapper.save_report")
    @patch("seismometer.report.profiling.ComparisonReportWrapper.generate_report")
    def test_init_skips_generation_if_file_exists(self, mock_generate, mock_save, mock_is_file, example_dfs, tmp_path):
        l_df, r_df = example_dfs
        _ = ComparisonReportWrapper(
            l_df=l_df,
            r_df=r_df,
            output_path=tmp_path,
        )
        mock_generate.assert_not_called()
        mock_save.assert_not_called()

    def test_generate_report_excludes_columns(self, example_dfs, tmp_path):
        l_df, r_df = example_dfs

        wrapper = ComparisonReportWrapper(
            l_df=l_df,
            r_df=r_df,
            output_path=tmp_path,
            l_title="left",
            r_title="right",
            exclude_cols=["b"],
        )

        assert [list(p.variables) for p in wrapper._report.profiles] == [["a"], ["a"]]
        assert wrapper._report.labels == ["left", "right"]
        assert wrapper._html_path.parent == tmp_path
        # titled after the datasets, as ydata-profiling titles comparisons
        assert "<em>Comparing</em> left <em>and</em> right" in wrapper._html_path.read_text(encoding="utf-8")


class TestSingleReportWrapper:
    @patch("seismometer.report.profiling.SingleReportWrapper._serialize_alerts")
    @patch("seismometer.report.profiling.SingleReportWrapper._parse_alerts")
    @patch("seismometer.report.profiling.SingleReportWrapper.save_report")
    @patch("seismometer.report.profiling.SingleReportWrapper.generate_report")
    def test_initializes_and_creates_report(
        self,
        mock_serialize,  # for _serialize_alerts
        mock_parse,  # for _parse_alerts
        mock_save,  # for save_report
        mock_generate,  # for generate_report
        tmp_path,
        small_df,
    ):
        _ = SingleReportWrapper(
            df=small_df,
            output_path=tmp_path,
            exclude_cols=[],
            title="report",
            alert_config=MagicMock(),
        )

        assert mock_generate.call_count == 1
        assert mock_save.call_count == 1
        assert mock_parse.call_count == 1
        assert mock_serialize.call_count == 1

    def test_deserialize_alerts(self, tmp_path, small_df):
        # Prepare a mock alert JSON file
        alert_data = {"alerts": [{"name": "test", "severity": "#FF0000", "display_html": "<b>Test</b>"}]}
        title = "test alert report"
        slug = slugify(title)
        alert_path = tmp_path / f"{slug}_alerts.json"
        alert_path.write_text(json.dumps(alert_data))

        # Create dummy report file to skip generation
        report_path = tmp_path / f"{slug}.html"
        report_path.write_text("dummy report")

        # Patch the internal methods that would invoke ydata_profiling
        with patch.object(SingleReportWrapper, "generate_report"), patch.object(
            SingleReportWrapper, "save_report"
        ), patch.object(SingleReportWrapper, "_parse_alerts"), patch.object(SingleReportWrapper, "_serialize_alerts"):
            wrapper = SingleReportWrapper(
                df=small_df, output_path=tmp_path, exclude_cols=[], title=title, alert_config=MagicMock()
            )

            # Check that alerts are loaded from the file
            assert len(wrapper._parsed_alerts.alerts) == 1
            alert = wrapper._parsed_alerts.alerts[0]
            assert alert.name == "test"
            assert alert.severity == "#FF0000"
            assert alert.display_html == "<b>Test</b>"

    def test_serialize_alerts(self, tmp_path, small_df):
        title = "test"
        slug = title.lower()
        alert_path = tmp_path / f"{slug}_alerts.json"
        report_path = tmp_path / f"{slug}.html"

        # Ensure dummy report file exists to skip generation
        report_path.write_text("dummy")

        with patch.object(SingleReportWrapper, "generate_report"), patch.object(
            SingleReportWrapper, "save_report"
        ), patch.object(SingleReportWrapper, "_parse_alerts"), patch.object(SingleReportWrapper, "_serialize_alerts"):
            wrapper = SingleReportWrapper(
                df=small_df, output_path=tmp_path, exclude_cols=[], title=title, alert_config=MagicMock()
            )

        # assign parsed alerts and test the method itself
        wrapper._parsed_alerts = ParsedAlertList(alerts=[])
        wrapper._serialize_alerts()

        assert alert_path.exists()
        assert "alerts" in alert_path.read_text()

    def test_generate_report_success(self):
        df = pd.DataFrame({"col1": [1, 2, 3], "drop": [4, 5, 6]})
        wrapper = SingleReportWrapper.__new__(SingleReportWrapper)  # avoid __init__
        wrapper._df = df
        wrapper._title = "TestReport"
        wrapper._exclude_cols = ["drop"]

        wrapper.generate_report()

        assert wrapper._report.title == "TestReport"
        assert list(wrapper._report.profiles[0].variables) == ["col1"]
        pd.testing.assert_frame_equal(wrapper._df, df.drop(columns=["drop"]))

    def test_end_to_end_report_and_alerts(self, tmp_path):
        df = pd.DataFrame({"id": range(100), "zeros": [0] * 60 + [1] * 40})
        alert_config = MagicMock()
        alert_config.parse_alert.return_value = (True, "#FF0000")

        wrapper = SingleReportWrapper(
            df=df, output_path=tmp_path, exclude_cols=["id"], title="Report", alert_config=alert_config
        )

        assert (tmp_path / "report.html").is_file()
        parsed = [(a.name, a.variable, a.percentage) for a in wrapper._parsed_alerts.alerts]
        assert parsed == [("Zeros", "zeros", 0.6)]
        saved = json.loads((tmp_path / "report_alerts.json").read_text())
        assert saved["alerts"][0]["display_html"].endswith("has 60 (60.0%) zeros")

    @patch.object(SingleReportWrapper, "_serialize_alerts")
    def test_parse_alerts_empty(self, mock_serialize):
        wrapper = SingleReportWrapper.__new__(SingleReportWrapper)
        wrapper._report = MagicMock()
        wrapper._report.alerts = []
        wrapper._alert_config = MagicMock()

        wrapper._parse_alerts()

        assert isinstance(wrapper._parsed_alerts, ParsedAlertList)
        assert wrapper._parsed_alerts.alerts == []
        mock_serialize.assert_not_called()

    @patch.object(SingleReportWrapper, "_serialize_alerts")
    def test_parse_alerts_skips_rejected(self, mock_serialize):
        alert = Alert(AlertType.REJECTED, column_name="unsupported")

        wrapper = SingleReportWrapper.__new__(SingleReportWrapper)
        wrapper._report = MagicMock()
        wrapper._report.alerts = [alert]
        wrapper._alert_config = MagicMock()

        wrapper._parse_alerts()

        assert wrapper._parsed_alerts.alerts == []
        mock_serialize.assert_called_once()

    @patch.object(SingleReportWrapper, "_serialize_alerts")
    def test_parse_alerts_valid_alert(self, mock_serialize):
        alert = Alert(AlertType.MISSING, {"n_missing": 3, "p_missing": 0.3}, column_name="test_column")

        alert_config = MagicMock()
        alert_config.parse_alert.return_value = (True, "#FF0000")

        wrapper = SingleReportWrapper.__new__(SingleReportWrapper)
        wrapper._report = MagicMock()
        wrapper._report.alerts = [alert]
        wrapper._alert_config = alert_config

        wrapper._parse_alerts()

        parsed = wrapper._parsed_alerts.alerts
        assert len(parsed) == 1
        assert parsed[0].name == "Missing"
        assert parsed[0].severity == "#FF0000"
        assert parsed[0].variable == "test_column"
        assert parsed[0].percentage == 0.3
        assert parsed[0].display_html == (
            '<a href="#pp_var_test_column"><code>test_column</code></a> has 3 (30.0%) missing values'
        )
        mock_serialize.assert_called_once()

    @patch("seismometer.report.profiling.display")
    @patch("seismometer.report.profiling.HTML")
    def test_display_alerts_no_alerts(self, mock_html, mock_display):
        wrapper = SingleReportWrapper.__new__(SingleReportWrapper)
        wrapper._parsed_alerts = ParsedAlertList(alerts=[])

        wrapper.display_alerts()

        html_calls = [call for call in mock_html.call_args_list if "<p>No alerts to display</p>" in str(call)]
        assert len(html_calls) == 1, "Expected one HTML call with fallback message"
        assert mock_display.call_count >= 1

    @patch("seismometer.report.profiling.GridBox")
    @patch("seismometer.report.profiling.display")
    @patch("seismometer.report.profiling.HTML")
    def test_display_alerts_with_alerts(self, mock_html, mock_display, mock_gridbox):
        alert = ParsedAlert(name="TestAlert", severity="#FF0000", display_html="<b>Danger</b>")
        wrapper = SingleReportWrapper.__new__(SingleReportWrapper)
        wrapper._parsed_alerts = ParsedAlertList(alerts=[alert])

        wrapper.display_alerts()

        # Expect at least 2 HTML displays: style block and alert content
        assert mock_html.call_count >= 2
        assert mock_display.call_count == 2
        mock_gridbox.assert_called_once()
