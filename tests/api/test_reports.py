from unittest.mock import MagicMock, Mock, patch

import pytest

from seismometer.api.reports import cohort_comparison_report, feature_alerts, feature_summary, target_feature_summary
from seismometer.controls.cohort_comparison import ComparisonReportGenerator


@pytest.fixture
def mock_seismogram():
    sg_mock = Mock()
    sg_mock.dataframe = Mock()
    sg_mock.config.output_dir = "/tmp/output"
    sg_mock.entity_keys = ["id"]
    sg_mock.alert_config = {}
    sg_mock.output_path = "/tmp/output"
    sg_mock.target = "target_col"
    sg_mock.available_cohort_groups = {"group": ("A", "B")}
    return sg_mock


class TestReportFunctions:
    @patch("seismometer.api.reports.Seismogram")
    @patch("seismometer.api.reports.SingleReportWrapper")
    def test_feature_alerts(self, mock_wrapper, mock_seismogram, mock_seismogram_obj=Mock()):
        mock_seismogram.return_value = mock_seismogram_obj
        mock_seismogram_obj.entity_keys = ["id"]
        feature_alerts()
        mock_wrapper.assert_called_once()
        mock_wrapper.return_value.display_alerts.assert_called_once()

    @patch("seismometer.api.reports.Seismogram")
    @patch("seismometer.api.reports.SingleReportWrapper")
    def test_feature_summary(self, mock_wrapper, mock_seismogram, mock_seismogram_obj=Mock()):
        mock_seismogram.return_value = mock_seismogram_obj
        mock_seismogram_obj.entity_keys = ["id"]
        feature_summary(inline=True)
        mock_wrapper.assert_called_once()
        mock_wrapper.return_value.display_report.assert_called_once_with(True)

    @patch("seismometer.api.reports.Seismogram")
    @patch("seismometer.api.reports.ComparisonReportWrapper")
    def test_target_feature_summary(self, mock_wrapper, mock_seismogram, mock_seismogram_obj=Mock()):
        df_mock = Mock()
        df_mock.empty = False
        filter_rule_mock = MagicMock()
        filter_rule_mock.filter.side_effect = [df_mock, df_mock]
        filter_rule_mock.__invert__.return_value = filter_rule_mock  # handle ~mock

        with patch("seismometer.api.reports.FilterRule.eq", return_value=filter_rule_mock):
            mock_seismogram.return_value = mock_seismogram_obj
            mock_seismogram_obj.dataframe = df_mock
            mock_seismogram_obj.target = "target_col"
            mock_seismogram_obj.output_path = "/tmp"
            mock_seismogram_obj.entity_keys = ["id"]

            target_feature_summary(inline=True)

            mock_wrapper.assert_called_once()
            mock_wrapper.return_value.display_report.assert_called_once_with(True)

    @patch("seismometer.api.reports.Seismogram")
    @patch("seismometer.controls.cohort_comparison.ComparisonReportGenerator")
    def test_cohort_comparison_report(self, mock_generator, mock_seismogram, mock_seismogram_obj=Mock()):
        mock_seismogram.return_value = mock_seismogram_obj
        mock_seismogram_obj.available_cohort_groups = {"group": ("A", "B")}
        cohort_comparison_report()
        mock_generator.assert_called_once()
        mock_generator.return_value.show.assert_called_once()

    @pytest.mark.parametrize(
        "neg_empty, pos_empty, expected_log",
        [
            (True, False, "negative target has no data to profile"),
            (False, True, "positive target has no data to profile"),
        ],
    )
    @patch("seismometer.api.reports.Seismogram")
    @patch("seismometer.api.reports.ComparisonReportWrapper")
    def test_target_feature_summary_missing_cohort_data(
        self, mock_wrapper, mock_seismogram, neg_empty, pos_empty, expected_log, caplog
    ):
        from seismometer.api.reports import target_feature_summary

        df_mock = Mock()
        neg_df = Mock()
        pos_df = Mock()
        neg_df.empty = neg_empty
        pos_df.empty = pos_empty

        filter_rule_mock = MagicMock()
        filter_rule_mock.filter.side_effect = [neg_df, pos_df]
        filter_rule_mock.__invert__.return_value = filter_rule_mock

        with patch("seismometer.api.reports.FilterRule.eq", return_value=filter_rule_mock):
            mock_sg = Mock()
            mock_sg.dataframe = df_mock
            mock_sg.target = "target_col"
            mock_sg.output_path = "/tmp"
            mock_sg.entity_keys = ["id"]
            mock_seismogram.return_value = mock_sg

            with caplog.at_level("WARNING"):
                target_feature_summary(inline=True)

            assert expected_log in caplog.text
            mock_wrapper.return_value.display_report.assert_not_called()

    @pytest.mark.parametrize(
        "l_filter, r_filter, l_empty, r_empty, expected_display",
        [
            (None, Mock(), False, False, False),  # l_cohort is None
            (Mock(), None, False, False, False),  # r_cohort is None
            (Mock(), Mock(), True, False, False),  # l_df is empty
            (Mock(), Mock(), False, True, False),  # r_df is empty
            (Mock(), Mock(), False, False, True),  # both valid
        ],
    )
    @patch("seismometer.seismogram.Seismogram")
    @patch("seismometer.controls.cohort_comparison.filter_rule_from_cohort_dictionary")
    @patch("seismometer.report.profiling.ComparisonReportWrapper")
    def test_generate_comparison_report_cases(
        self,
        mock_wrapper,
        mock_filter_rule,
        mock_seismogram,
        l_filter,
        r_filter,
        l_empty,
        r_empty,
        expected_display,
    ):
        sg = Mock()
        sg.entity_keys = []
        sg.output_path = "/tmp"
        sg.dataframe = Mock()
        sg.cohort_hierarchies = None
        mock_seismogram.return_value = sg

        def wrap_filter(mock_filter, is_empty):
            mock_df = Mock()
            mock_df.empty = is_empty
            mock_filter.filter.return_value = mock_df
            return mock_filter

        if l_filter is not None:
            l_filter = wrap_filter(l_filter, l_empty)
        if r_filter is not None:
            r_filter = wrap_filter(r_filter, r_empty)

        mock_filter_rule.side_effect = [l_filter, r_filter]

        crg = ComparisonReportGenerator(selections={"group": ("A", "B")})
        crg.output = Mock()
        crg.output.__enter__ = lambda s: s
        crg.output.__exit__ = lambda s, a, b, c: None
        crg.button = Mock()

        for i in range(2):
            selector = Mock()
            selector.value = {"group": ("A",)}
            selector.get_selection_text.return_value = f"Selection {i}"
            crg.selectors[i] = selector

        crg._generate_comparison_report()

        assert crg.button.disabled is False
        if expected_display:
            mock_wrapper.return_value.display_report.assert_called_once_with(inline=False)
        else:
            mock_wrapper.return_value.display_report.assert_not_called()
