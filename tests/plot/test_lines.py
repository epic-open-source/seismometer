from unittest.mock import Mock

import numpy as np
import pandas as pd

import seismometer.plot.mpl._lines as undertest


class TestVerticalThresholdLines:
    """Test vertical_threshold_lines() function."""

    def test_basic_functionality(self):
        """Test vertical_threshold_lines() plots vertical lines at thresholds."""
        axis = Mock()
        highlight = [0.3, 0.7]

        undertest.vertical_threshold_lines(axis, highlight, color_alerts=False, plot=True)

        # Should plot 2 vertical lines
        assert axis.axvline.call_count == 2
        axis.axvline.assert_any_call(x=0.7, color=None, linestyle="--")
        axis.axvline.assert_any_call(x=0.3, color=None, linestyle="--")

    def test_with_color_alerts(self):
        """Test vertical_threshold_lines() with color_alerts enabled."""
        axis = Mock()
        highlight = [0.5]

        undertest.vertical_threshold_lines(axis, highlight, color_alerts=True, plot=True)

        # Should plot with color from alert_colors
        assert axis.axvline.call_count == 1
        call_kwargs = axis.axvline.call_args[1]
        assert call_kwargs["color"] is not None  # Should have a color

    def test_with_legend(self):
        """Test vertical_threshold_lines() adds legend when position specified."""
        axis = Mock()
        highlight = [0.5]

        undertest.vertical_threshold_lines(axis, highlight, legend_position="upper left", plot=True)

        # Should add legend
        axis.legend.assert_called_once()
        axis.add_artist.assert_called_once()

    def test_plot_false_skips_plotting(self):
        """Test vertical_threshold_lines() with plot=False doesn't draw lines."""
        axis = Mock()
        highlight = [0.5]

        undertest.vertical_threshold_lines(axis, highlight, plot=False)

        # Should not plot vertical lines
        axis.axvline.assert_not_called()

    def test_show_marker_false(self):
        """Test vertical_threshold_lines() with show_marker=False."""
        axis = Mock()
        highlight = [0.5]

        undertest.vertical_threshold_lines(axis, highlight, show_marker=False, legend_position="upper left")

        # Legend should be added but without markers
        axis.legend.assert_called_once()


class TestRocPlot:
    """Test roc_plot() function."""

    def test_plots_roc_curve(self):
        """Test roc_plot() creates ROC curve."""
        axis = Mock()
        fpr = [0.0, 0.2, 0.4, 1.0]
        tpr = [0.0, 0.6, 0.8, 1.0]

        undertest.roc_plot(axis, fpr, tpr, label="Test")

        # Should plot the curve (plus diagonal line is plotted first)
        assert axis.plot.call_count >= 1
        axis.legend.assert_called_once_with(loc="lower right")
        axis.set_xlim.assert_called_once_with(0, 1.01)
        axis.set_ylim.assert_called_once_with(0, 1.01)
        axis.set_xlabel.assert_called_once_with("1 - Specificity")
        axis.set_ylabel.assert_called_once_with("Sensitivity")

    def test_without_label(self):
        """Test roc_plot() without label."""
        axis = Mock()
        fpr = [0.0, 1.0]
        tpr = [0.0, 1.0]

        undertest.roc_plot(axis, fpr, tpr)

        # plot_diagonal is called, then the actual curve
        assert axis.plot.call_count >= 1


class TestReliabilityPlot:
    """Test reliability_plot() function."""

    def test_plots_calibration_curve(self):
        """Test reliability_plot() creates calibration curve."""
        axis = Mock()
        mean_predicted = [0.1, 0.3, 0.5, 0.7, 0.9]
        fraction_positive = [0.15, 0.35, 0.5, 0.65, 0.85]

        undertest.reliability_plot(axis, mean_predicted, fraction_positive, label="Model A")

        # Should plot with 'x-' style (plot_diagonal called first)
        assert axis.plot.call_count >= 1
        axis.set_xlim.assert_called_once_with(0, 1.01)
        axis.set_ylim.assert_called_once_with(0, 1.01)
        axis.set_xlabel.assert_called_once_with("Predicted Probability")
        axis.set_ylabel.assert_called_once_with("Observed Rate")


class TestHistStacked:
    """Test hist_stacked() function."""

    def test_creates_stacked_histogram(self):
        """Test hist_stacked() creates stacked histogram."""
        axis = Mock()
        probabilities = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        labels = ["Class 0", "Class 1"]

        undertest.hist_stacked(axis, probabilities, labels, show_legend=True, bins=10)

        # Should create histogram
        axis.hist.assert_called_once_with(probabilities, bins=10, label=labels, stacked=True)
        axis.legend.assert_called_once_with(loc="lower right")
        axis.set_xlim.assert_called_once_with([0, 1.01])
        axis.set_xlabel.assert_called_once_with("Predicted Probability")
        axis.set_ylabel.assert_called_once_with("Count")

    def test_without_legend(self):
        """Test hist_stacked() with show_legend=False."""
        axis = Mock()
        probabilities = [[0.1, 0.2]]
        labels = ["Class 0"]

        undertest.hist_stacked(axis, probabilities, labels, show_legend=False)

        axis.legend.assert_not_called()


class TestHistSingle:
    """Test hist_single() function."""

    def test_creates_single_histogram(self):
        """Test hist_single() creates histogram with step and fill."""
        axis = Mock()
        # Mock the return value of axis.step to be subscriptable
        mock_line = Mock()
        mock_line.get_color.return_value = "blue"
        axis.step.return_value = [mock_line]

        data_series = pd.Series([0.1, 0.3, 0.5, 0.7, 0.9])

        result = undertest.hist_single(axis, data_series, label="Test", bins=5, scale=1)

        # Should create step plot and fill_between
        axis.step.assert_called_once()
        axis.fill_between.assert_called_once()
        axis.set_xlabel.assert_called_once_with("Predicted Probability")
        axis.set_ylabel.assert_called_once_with("Count")
        assert result is not None  # Returns y_data


class TestPpvSensitivityCurve:
    """Test ppv_sensitivity_curve() function."""

    def test_plots_precision_recall_curve(self):
        """Test ppv_sensitivity_curve() creates precision-recall curve."""
        axis = Mock()
        recall = [0.0, 0.5, 0.8, 1.0]
        precision = [1.0, 0.8, 0.7, 0.6]

        undertest.ppv_sensitivity_curve(axis, recall, precision, label="Model")

        axis.step.assert_called_once_with(recall, precision, where="post", label="Model")
        axis.legend.assert_called_once_with(loc="upper left")
        axis.set_xlim.assert_called_once_with([0, 1.01])
        axis.set_ylim.assert_called_once_with([0, 1.01])
        axis.set_xlabel.assert_called_once_with("Sensitivity")
        axis.set_ylabel.assert_called_once_with("PPV")


class TestPerformanceMetricsPlot:
    """Test performance_metrics_plot() function."""

    def test_plots_multiple_metrics(self):
        """Test performance_metrics_plot() plots multiple performance metrics."""
        axis = Mock()
        sensitivity = np.array([0.9, 0.8, 0.7])
        specificity = np.array([0.7, 0.8, 0.9])
        ppv = np.array([0.6, 0.7, 0.8])
        thresholds = np.array([0.3, 0.5, 0.7])

        undertest.performance_metrics_plot(axis, sensitivity, specificity, ppv, thresholds)

        # Should plot 3 lines (sensitivity, specificity, ppv)
        assert axis.plot.call_count == 3
        axis.legend.assert_called_once_with(loc="lower right")
        axis.set_xlim.assert_called_once_with([0, 1.01])
        axis.set_ylim.assert_called_once_with([0, 1.01])
        axis.set_xlabel.assert_called_once_with("Threshold")
        axis.set_ylabel.assert_called_once_with("Metric")


class TestPerformanceConfidence:
    """Test performance_confidence() function."""

    def test_plots_confidence_intervals(self):
        """Test performance_confidence() plots confidence intervals."""
        axis = Mock()
        perf_stats = pd.DataFrame(
            {
                "Threshold": [0.3, 0.5, 0.7],
                "Sensitivity": [0.9, 0.8, 0.7],
                "TP": [90, 80, 70],
                "FN": [10, 20, 30],
            }
        )

        undertest.performance_confidence(axis, perf_stats, conf=0.95, metric="Sensitivity")

        # Should plot with fill_between (called internally)
        axis.fill_between.assert_called_once()


class TestGetLastLineColor:
    """Test get_last_line_color() function."""

    def test_returns_color_from_last_line(self):
        """Test get_last_line_color() returns color from last plotted line."""
        axis = Mock()
        mock_line = Mock()
        mock_line.get_color.return_value = "blue"
        axis.get_lines.return_value = [mock_line]

        color = undertest.get_last_line_color(axis)

        assert color == "blue"

    def test_empty_axis_returns_none(self):
        """Test get_last_line_color() with no lines returns None."""
        axis = Mock()
        axis.get_lines.return_value = []

        color = undertest.get_last_line_color(axis)

        assert color is None


class TestRadialAnnotations:
    """Test _radial_annotations() function."""

    def test_quadrant_1(self):
        """Test _radial_annotations() for quadrant 1."""
        x, y = 1.0, 1.0
        dx, dy = undertest._radial_annotations(x, y, Q=1)
        assert dx > 0  # Should offset to the right
        assert dy > 0  # Should offset upward

    def test_quadrant_2(self):
        """Test _radial_annotations() for quadrant 2."""
        x, y = -1.0, 1.0
        dx, dy = undertest._radial_annotations(x, y, Q=2)
        assert dx < 0  # Should offset to the left

    def test_quadrant_3(self):
        """Test _radial_annotations() for quadrant 3."""
        x, y = -1.0, -1.0
        dx, dy = undertest._radial_annotations(x, y, Q=3)
        assert dx < 0
        assert dy < 0

    def test_quadrant_4(self):
        """Test _radial_annotations() for quadrant 4."""
        x, y = 1.0, -1.0
        dx, dy = undertest._radial_annotations(x, y, Q=4)
        assert dx > 0
        assert dy < 0


class TestRecallConditionPlot:
    """Test recall_condition_plot() function."""

    def test_basic_plot(self):
        """Test recall_condition_plot() without reference."""
        axis = Mock()
        ppcr = [0.0, 0.3, 0.5, 1.0]
        recall = [0.0, 0.7, 0.8, 1.0]

        undertest.recall_condition_plot(axis, ppcr, recall, prevalence=0.2)

        axis.plot.assert_called_once_with(ppcr, recall)
        axis.set_xlim.assert_called_once_with(0, 1.01)
        axis.set_ylim.assert_called_once_with(0, 1.01)
        axis.set_xlabel.assert_called_once_with("Flag Rate")
        axis.set_ylabel.assert_called_once_with("Sensitivity")

    def test_with_reference(self):
        """Test recall_condition_plot() with reference shading."""
        axis = Mock()
        ppcr = [0.0, 0.3, 0.5, 1.0]
        recall = [0.0, 0.7, 0.8, 1.0]

        undertest.recall_condition_plot(axis, ppcr, recall, prevalence=0.2, show_reference=True)

        # plot_polygon is called, then the actual plot
        assert axis.plot.call_count >= 1


class TestSinglePpv:
    """Test single_ppv() function."""

    def test_without_threshold_line(self):
        """Test single_ppv() without precision threshold."""
        axis = Mock()
        thresholds = np.array([0.3, 0.5, 0.7])
        precision = np.array([0.8, 0.7, 0.6])

        undertest.single_ppv(axis, thresholds, precision, precision_threshold=None)

        axis.plot.assert_called_once()
        axis.set_xlim.assert_called_once_with([0, 1.01])
        axis.set_ylim.assert_called_once_with([0, 1.01])
        axis.set_xlabel.assert_called_once_with("Threshold")
        axis.set_ylabel.assert_called_once_with("PPV")

    def test_with_threshold_line(self):
        """Test single_ppv() with precision threshold."""
        axis = Mock()
        thresholds = np.array([0.3, 0.5, 0.7])
        precision = np.array([0.8, 0.7, 0.6])

        undertest.single_ppv(axis, thresholds, precision, precision_threshold=0.75)

        # plot_horizontal is called in addition to plot (uses axis.plot for horizontal line)
        assert axis.plot.call_count >= 2


class TestMetricVsThresholdCurve:
    """Test metric_vs_threshold_curve() function."""

    def test_plots_metric_curve(self):
        """Test metric_vs_threshold_curve() creates metric curve."""
        axis = Mock()
        metric = np.array([0.9, 0.8, 0.7])
        thresholds = np.array([0.3, 0.5, 0.7])

        undertest.metric_vs_threshold_curve(axis, metric, thresholds, label="Accuracy")

        axis.plot.assert_called_once()
        axis.set_xlim.assert_called_once_with([0, 1.01])
        axis.set_ylim.assert_called_once_with([0, 1.01])
        axis.set_xlabel.assert_called_once_with("Threshold")
        axis.set_ylabel.assert_called_once_with("Accuracy")


class TestRocRegionPlot:
    """Test roc_region_plot() function."""

    def test_plots_confidence_region(self):
        """Test roc_region_plot() fills ROC confidence region."""
        axis = Mock()
        lower_x = np.array([0.0, 0.2, 0.4])
        lower_y = np.array([0.0, 0.5, 0.7])
        upper_x = np.array([0.0, 0.3, 0.5])
        upper_y = np.array([0.0, 0.6, 0.8])

        undertest.roc_region_plot(axis, lower_x, lower_y, upper_x, upper_y)

        axis.fill.assert_called_once()


class TestPerformanceRegionPlot:
    """Test performance_region_plot() function."""

    def test_plots_performance_region(self):
        """Test performance_region_plot() fills performance region."""
        axis = Mock()
        lower = np.array([0.7, 0.6, 0.5])
        upper = np.array([0.9, 0.8, 0.7])
        thresholds = np.array([0.3, 0.5, 0.7])

        undertest.performance_region_plot(axis, lower, upper, thresholds)

        axis.fill_between.assert_called_once()


class TestAddRadialScoreThresholds:
    """Test _add_radial_score_thresholds() function."""

    def test_adds_threshold_annotations(self):
        """Test _add_radial_score_thresholds() adds threshold markers."""
        axis = Mock()
        x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = np.array([0.0, 0.5, 0.7, 0.8, 0.9, 1.0])
        labels = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        thresholds = [0.3, 0.7]

        undertest._add_radial_score_thresholds(axis, x, y, labels, thresholds, Q=1)

        # Should plot threshold markers and add annotations
        assert axis.plot.call_count >= 1
        assert axis.annotate.call_count == 2

    def test_returns_early_if_no_labels(self):
        """Test _add_radial_score_thresholds() returns early with None labels."""
        axis = Mock()
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])

        undertest._add_radial_score_thresholds(axis, x, y, None, [0.5])

        # Should return early without plotting
        axis.plot.assert_not_called()
        axis.annotate.assert_not_called()


class TestAddRadialScoreLabels:
    """Test _add_radial_score_labels() function."""

    def test_adds_score_labels(self):
        """Test _add_radial_score_labels() adds score annotations."""
        axis = Mock()
        x = np.array([0.0, 0.3, 0.6, 0.9])
        y = np.array([0.0, 0.5, 0.8, 1.0])
        labels = [0.1, 0.4, 0.7, 0.9]

        undertest._add_radial_score_labels(axis, x, y, labels, n_scores=4)

        # Should plot markers and add annotations
        axis.plot.assert_called_once()
        axis.legend.assert_called_once_with(loc="lower right")
        assert axis.annotate.call_count == 4

    def test_delegates_to_thresholds_with_highlight(self):
        """Test _add_radial_score_labels() delegates to thresholds when highlight set."""
        axis = Mock()
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.7, 1.0])
        labels = [0.1, 0.5, 0.9]

        undertest._add_radial_score_labels(axis, x, y, labels, highlight=[0.5])

        # Should delegate to _add_radial_score_thresholds
        assert axis.plot.call_count >= 1


class TestFindThresholds:
    def test_thresholds_increasing_labels(self):
        labels = [0.1, 0.3, 0.5, 0.7, 0.9]
        thresholds = [0.2, 0.6]
        expected_output = [0, 2]

        actual = undertest._find_thresholds(labels, thresholds)

        assert actual == expected_output

    def test_thresholds_decreasing_labels(self):
        labels = [0.9, 0.7, 0.5, 0.3, 0.1]
        thresholds = [0.2, 0.6]
        expected_output = [3, 1]

        actual = undertest._find_thresholds(labels, thresholds)

        assert actual == expected_output

    def test_thresholds_ascending_out_of_range(self):
        labels = [0.1, 0.3, 0.5, 0.7, 0.9]
        thresholds = [-0.1, 1.0]
        expected_output = [0, 4]

        actual = undertest._find_thresholds(labels, thresholds)

        assert actual == expected_output

    def test_thresholds_descending_out_of_range(self):
        labels = [0.9, 0.7, 0.5, 0.3, 0.1]
        thresholds = [-0.1, 1.0]
        expected_output = [4, 0]

        actual = undertest._find_thresholds(labels, thresholds)

        assert actual == expected_output
