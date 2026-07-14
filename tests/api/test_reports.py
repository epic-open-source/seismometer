from unittest.mock import MagicMock, Mock, patch

import pytest

from seismometer.api.reports import (
    ExploreAnalyticsTable,
    ExploreCohortOrdinalMetrics,
    ExploreFairnessAudit,
    ExploreOrdinalMetrics,
    cohort_comparison_report,
    feature_alerts,
    feature_summary,
    target_feature_summary,
)
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


# ============================================================================
# WIDGET CLASS TESTS
# ============================================================================


class TestWidgetClassDefinitions:
    """Test that widget classes are properly defined and importable."""

    def test_explore_fairness_audit_class_exists(self):
        """Test that ExploreFairnessAudit class is defined."""
        assert ExploreFairnessAudit is not None
        assert hasattr(ExploreFairnessAudit, "__init__")

    def test_explore_analytics_table_class_exists(self):
        """Test that ExploreAnalyticsTable class is defined."""
        assert ExploreAnalyticsTable is not None
        assert hasattr(ExploreAnalyticsTable, "__init__")

    def test_explore_ordinal_metrics_class_exists(self):
        """Test that ExploreOrdinalMetrics class is defined."""
        assert ExploreOrdinalMetrics is not None
        assert hasattr(ExploreOrdinalMetrics, "__init__")

    def test_explore_cohort_ordinal_metrics_class_exists(self):
        """Test that ExploreCohortOrdinalMetrics class is defined."""
        assert ExploreCohortOrdinalMetrics is not None
        assert hasattr(ExploreCohortOrdinalMetrics, "__init__")


# ============================================================================
# ADDITIONAL ERROR HANDLING AND EDGE CASE TESTS
# ============================================================================


class TestFeatureAlertsEdgeCases:
    """Test edge cases and error handling for feature_alerts function."""

    @patch("seismometer.api.reports.Seismogram")
    @patch("seismometer.api.reports.SingleReportWrapper")
    def test_feature_alerts_with_custom_exclude_cols(self, mock_wrapper, mock_seismogram):
        """Test feature_alerts with custom exclude_cols."""
        mock_sg = Mock()
        mock_sg.entity_keys = ["id"]
        mock_sg.dataframe = Mock()
        mock_sg.config.output_dir = "/tmp/output"
        mock_sg.alert_config = {}
        mock_seismogram.return_value = mock_sg

        exclude_cols = ["col1", "col2", "col3"]
        feature_alerts(exclude_cols=exclude_cols)

        mock_wrapper.assert_called_once()
        call_kwargs = mock_wrapper.call_args[1]
        assert call_kwargs["exclude_cols"] == exclude_cols

    @patch("seismometer.api.reports.Seismogram")
    @patch("seismometer.api.reports.SingleReportWrapper")
    def test_feature_alerts_with_empty_exclude_cols(self, mock_wrapper, mock_seismogram):
        """Test feature_alerts with empty exclude_cols list.

        Note: Empty list is falsy, so `exclude_cols or sg.entity_keys` will use entity_keys.
        """
        mock_sg = Mock()
        mock_sg.entity_keys = ["id"]
        mock_sg.dataframe = Mock()
        mock_sg.config.output_dir = "/tmp/output"
        mock_sg.alert_config = {}
        mock_seismogram.return_value = mock_sg

        feature_alerts(exclude_cols=[])

        mock_wrapper.assert_called_once()
        call_kwargs = mock_wrapper.call_args[1]
        # Empty list is falsy, so defaults to entity_keys
        assert call_kwargs["exclude_cols"] == ["id"]

    @patch("seismometer.api.reports.Seismogram")
    @patch("seismometer.api.reports.SingleReportWrapper")
    def test_feature_alerts_exception_in_display(self, mock_wrapper, mock_seismogram):
        """Test feature_alerts when display_alerts raises an exception."""
        mock_sg = Mock()
        mock_sg.entity_keys = ["id"]
        mock_sg.dataframe = Mock()
        mock_sg.config.output_dir = "/tmp/output"
        mock_sg.alert_config = {}
        mock_seismogram.return_value = mock_sg

        mock_wrapper.return_value.display_alerts.side_effect = RuntimeError("Display failed")

        with pytest.raises(RuntimeError, match="Display failed"):
            feature_alerts()


class TestFeatureSummaryEdgeCases:
    """Test edge cases for feature_summary function."""

    @patch("seismometer.api.reports.Seismogram")
    @patch("seismometer.api.reports.SingleReportWrapper")
    @pytest.mark.parametrize("inline", [True, False])
    def test_feature_summary_inline_parameter(self, mock_wrapper, mock_seismogram, inline):
        """Test feature_summary with both inline parameter values."""
        mock_sg = Mock()
        mock_sg.entity_keys = ["id"]
        mock_sg.dataframe = Mock()
        mock_sg.config.output_dir = "/tmp/output"
        mock_sg.alert_config = {}
        mock_seismogram.return_value = mock_sg

        feature_summary(inline=inline)

        mock_wrapper.assert_called_once()
        mock_wrapper.return_value.display_report.assert_called_once_with(inline)

    @patch("seismometer.api.reports.Seismogram")
    @patch("seismometer.api.reports.SingleReportWrapper")
    def test_feature_summary_with_large_exclude_list(self, mock_wrapper, mock_seismogram):
        """Test feature_summary with a large exclude_cols list."""
        mock_sg = Mock()
        mock_sg.entity_keys = ["id"]
        mock_sg.dataframe = Mock()
        mock_sg.config.output_dir = "/tmp/output"
        mock_sg.alert_config = {}
        mock_seismogram.return_value = mock_sg

        # Create a large list of columns to exclude
        exclude_cols = [f"col_{i}" for i in range(100)]
        feature_summary(exclude_cols=exclude_cols, inline=True)

        mock_wrapper.assert_called_once()
        call_kwargs = mock_wrapper.call_args[1]
        assert call_kwargs["exclude_cols"] == exclude_cols


class TestTargetFeatureSummaryEdgeCases:
    """Test edge cases for target_feature_summary function."""

    @patch("seismometer.api.reports.Seismogram")
    @patch("seismometer.api.reports.ComparisonReportWrapper")
    def test_target_feature_summary_with_empty_exclude_cols(self, mock_wrapper, mock_seismogram):
        """Test target_feature_summary with empty exclude_cols.

        Note: Empty list is falsy, so `exclude_cols or sg.entity_keys` will use entity_keys.
        """
        df_mock = Mock()
        df_mock.empty = False
        filter_rule_mock = MagicMock()
        filter_rule_mock.filter.side_effect = [df_mock, df_mock]
        filter_rule_mock.__invert__.return_value = filter_rule_mock

        with patch("seismometer.api.reports.FilterRule.eq", return_value=filter_rule_mock):
            mock_sg = Mock()
            mock_sg.dataframe = df_mock
            mock_sg.target = "target_col"
            mock_sg.output_path = "/tmp"
            mock_sg.entity_keys = ["id"]
            mock_seismogram.return_value = mock_sg

            target_feature_summary(exclude_cols=[], inline=True)

            mock_wrapper.assert_called_once()
            call_kwargs = mock_wrapper.call_args[1]
            # Empty list is falsy, so defaults to entity_keys
            assert call_kwargs["exclude_cols"] == ["id"]

    @patch("seismometer.api.reports.Seismogram")
    @patch("seismometer.api.reports.ComparisonReportWrapper")
    def test_target_feature_summary_both_targets_empty(self, mock_wrapper, mock_seismogram, caplog):
        """Test target_feature_summary when both positive and negative targets are empty."""
        df_mock = Mock()
        neg_df = Mock()
        pos_df = Mock()
        neg_df.empty = True
        pos_df.empty = True

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

            # Should log warning about negative target first
            assert "negative target has no data to profile" in caplog.text
            mock_wrapper.return_value.display_report.assert_not_called()


class TestCohortComparisonReportEdgeCases:
    """Test edge cases for cohort_comparison_report function."""

    @patch("seismometer.api.reports.Seismogram")
    @patch("seismometer.controls.cohort_comparison.ComparisonReportGenerator")
    def test_cohort_comparison_report_with_custom_exclude_cols(self, mock_generator, mock_seismogram):
        """Test cohort_comparison_report with custom exclude_cols."""
        mock_sg = Mock()
        mock_sg.available_cohort_groups = {"group": ("A", "B")}
        mock_sg.cohort_hierarchies = None
        mock_sg.cohort_hierarchy_combinations = None
        mock_seismogram.return_value = mock_sg

        exclude_cols = ["col1", "col2"]
        cohort_comparison_report(exclude_cols=exclude_cols)

        mock_generator.assert_called_once()
        call_kwargs = mock_generator.call_args[1]
        assert call_kwargs["exclude_cols"] == exclude_cols

    @patch("seismometer.api.reports.Seismogram")
    @patch("seismometer.controls.cohort_comparison.ComparisonReportGenerator")
    def test_cohort_comparison_report_with_empty_cohorts(self, mock_generator, mock_seismogram):
        """Test cohort_comparison_report with empty cohort groups."""
        mock_sg = Mock()
        mock_sg.available_cohort_groups = {}
        mock_sg.cohort_hierarchies = None
        mock_sg.cohort_hierarchy_combinations = None
        mock_seismogram.return_value = mock_sg

        cohort_comparison_report()

        mock_generator.assert_called_once()
        # Should still create generator even with empty cohorts
        assert mock_generator.call_args[0][0] == {}

    @patch("seismometer.api.reports.Seismogram")
    @patch("seismometer.controls.cohort_comparison.ComparisonReportGenerator")
    def test_cohort_comparison_report_with_hierarchies(self, mock_generator, mock_seismogram):
        """Test cohort_comparison_report with cohort hierarchies."""
        mock_sg = Mock()
        mock_sg.available_cohort_groups = {"group": ("A", "B")}
        mock_sg.cohort_hierarchies = {"group": ["subgroup1", "subgroup2"]}
        mock_sg.cohort_hierarchy_combinations = [("group", "subgroup1")]
        mock_seismogram.return_value = mock_sg

        cohort_comparison_report()

        mock_generator.assert_called_once()
        call_kwargs = mock_generator.call_args[1]
        assert call_kwargs["hierarchies"] == mock_sg.cohort_hierarchies
        assert call_kwargs["hierarchy_combinations"] == mock_sg.cohort_hierarchy_combinations
