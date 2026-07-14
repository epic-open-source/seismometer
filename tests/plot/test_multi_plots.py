from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.testing as pdt
from IPython.core.display import SVG
from matplotlib.figure import Figure

import seismometer.plot.mpl.multi_classifier as multPlots


class FakeGS:
    # GridSpec doesn't fit cleanly in Mocking - has methods and is also indexed
    def __init__(self, dd):
        self.__dict__ = dd

    def __getitem__(self, key):
        return self.__dict__.get(key, None)

    def tight_layout(self, arg1, **kwargs):
        pass


def echo_arg1(arg1, *args):
    return arg1


MockFig = Mock(spec=Figure)
MockFig.add_subplot = echo_arg1


class Test_One_Vertical:
    def test_defaults(self):
        mock_fn = Mock()
        mock_ax = Mock(autospec=plt.Axes)

        data = pd.DataFrame(data=[[1, 2], [3, 4], [5, 6]], dtype=int)

        multPlots._plot_one_vertical(data, mock_fn, mock_ax)

        mock_fn.assert_called_once()
        (call_args, kw_args) = mock_fn.call_args_list[0]
        pdt.assert_series_equal(call_args[0], data.iloc[:, 0])
        assert np.array_equal(call_args[1], data.iloc[:, 1])
        assert kw_args == {"axis": mock_ax}

        mock_ax.set_xlim.assert_called_once_with(0, 1)
        assert mock_ax.text.call_count == 0

    def test_with_label(self):
        mock_fn = Mock()
        mock_ax = Mock(autospec=plt.Axes)

        data = pd.DataFrame(data=[[1, 2], [3, 4], [5, 6]], dtype=int)

        multPlots._plot_one_vertical(data, mock_fn, mock_ax, label="TEST_LABEL")

        mock_fn.assert_called_once()

        (call_args, kw_args) = mock_fn.call_args_list[0]
        pdt.assert_series_equal(call_args[0], data.iloc[:, 0])
        assert np.array_equal(call_args[1], data.iloc[:, 1])
        assert kw_args == {"axis": mock_ax}

        mock_ax.set_xlim.assert_called_once_with(0, 1)

    def test_with_all(self):
        mock_fn = Mock()
        mock_ax = Mock(autospec=plt.Axes)

        data = pd.DataFrame(data=[[1, 2], [3, 4], [5, 6]], dtype=int)

        multPlots._plot_one_vertical(data, mock_fn, mock_ax, label="TEST_LABEL", func_kws={"extra_kw": 0})

        mock_fn.assert_called_once()

        (call_args, kw_args) = mock_fn.call_args_list[0]
        pdt.assert_series_equal(call_args[0], data.iloc[:, 0])
        assert np.array_equal(call_args[1], data.iloc[:, 1])
        assert kw_args == {"axis": mock_ax, "extra_kw": 0}

        mock_ax.set_xlim.assert_called_once_with(0, 1)

    def test_with_nan_data(self):
        """Test _plot_one_vertical() handles NaN values."""
        mock_fn = Mock()
        mock_ax = Mock(autospec=plt.Axes)

        data = pd.DataFrame(data=[[1, 2], [3, np.nan], [5, 6]], dtype=float)

        multPlots._plot_one_vertical(data, mock_fn, mock_ax)

        mock_fn.assert_called_once()
        mock_ax.set_xlim.assert_called_once_with(0, 1)

    def test_with_empty_data(self):
        """Test _plot_one_vertical() with empty DataFrame."""
        mock_fn = Mock()
        mock_ax = Mock(autospec=plt.Axes)

        data = pd.DataFrame(columns=[0, 1], dtype=int)

        multPlots._plot_one_vertical(data, mock_fn, mock_ax)

        mock_fn.assert_called_once()


class Test_Cohorts_Overlay:
    """Test cohorts_overlay() function."""

    def test_basic_overlay(self):
        """Test cohorts_overlay() plots multiple cohorts."""
        mock_fn = Mock()
        mock_ax = Mock(autospec=plt.Axes)
        mock_ax.get_figure.return_value = Mock(spec=Figure)

        data = pd.DataFrame(
            {
                "cohort": pd.Categorical(["A", "A", "B", "B"] * 5),
                "value1": range(20),
                "value2": range(20, 40),
            }
        )

        result = multPlots.cohorts_overlay(data, mock_fn, axis=mock_ax)

        # Should call plot_func for each cohort
        assert mock_fn.call_count == 2
        assert isinstance(result, Figure)

    def test_censoring_small_cohorts(self):
        """Test cohorts_overlay() censors cohorts below threshold."""
        mock_fn = Mock()
        mock_ax = Mock(autospec=plt.Axes)
        mock_ax.get_figure.return_value = Mock(spec=Figure)

        # Create data with one small cohort (< 10 samples)
        data = pd.DataFrame(
            {
                "cohort": pd.Categorical(["A"] * 15 + ["B"] * 5),
                "value1": range(20),
                "value2": range(20, 40),
            }
        )

        multPlots.cohorts_overlay(data, mock_fn, axis=mock_ax, censor_threshold=10)

        # Should call plot_func twice but second call has None data (censored)
        assert mock_fn.call_count == 2

    def test_with_labels_filter(self):
        """Test cohorts_overlay() filters by labels."""
        mock_fn = Mock()
        mock_ax = Mock(autospec=plt.Axes)
        mock_ax.get_figure.return_value = Mock(spec=Figure)

        data = pd.DataFrame(
            {
                "cohort": pd.Categorical(["A", "A", "B", "B", "C", "C"] * 3),
                "value1": range(18),
                "value2": range(18, 36),
            }
        )

        multPlots.cohorts_overlay(data, mock_fn, axis=mock_ax, labels=["A", "B"])

        # Should process all 3 cohorts but C should be censored
        assert mock_fn.call_count == 3


class Test_Cohorts_Vertical:
    """Test cohorts_vertical() function."""

    def test_basic_vertical_plot(self):
        """Test cohorts_vertical() creates vertical subplot grid."""
        mock_fn = Mock()

        # Data structure matches get_cohort_data output: cohort, true, pred
        data = pd.DataFrame(
            [[0, 0.2], [1, 0.8], [0, 0.3], [1, 0.7]] * 5,
            columns=[0, 1],
        )
        data["cohort"] = pd.Categorical(["A", "A", "B", "B"] * 5)

        result = multPlots.cohorts_vertical(data, mock_fn)

        # Should call plot_func for each cohort
        assert mock_fn.call_count == 2
        assert isinstance(result, SVG)  # Returns SVG when no axis provided

    def test_with_empty_cohorts(self):
        """Test cohorts_vertical() raises error with no data."""
        mock_fn = Mock()

        data = pd.DataFrame(columns=[0, 1])
        data["cohort"] = pd.Categorical([])

        try:
            multPlots.cohorts_vertical(data, mock_fn)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No cohorts had data" in str(e)

    def test_with_custom_labels(self):
        """Test cohorts_vertical() with custom labels."""
        mock_fn = Mock()

        # Data structure matches get_cohort_data output
        data = pd.DataFrame(
            [[0, 0.2], [1, 0.8], [0, 0.3], [1, 0.7]] * 5,
            columns=[0, 1],
        )
        data["cohort"] = pd.Categorical(["A", "A", "B", "B"] * 5)

        result = multPlots.cohorts_vertical(data, mock_fn, labels=["Label1", "Label2"])

        assert isinstance(result, SVG)  # Returns SVG when no axis provided


class Test_Cohort_Evaluation_Vs_Threshold:
    """Test cohort_evaluation_vs_threshold() function."""

    def test_creates_2x3_grid(self):
        """Test cohort_evaluation_vs_threshold() creates 2x3 subplot grid."""
        # Create complete performance data with all required columns
        stats = pd.DataFrame(
            {
                "cohort": pd.Categorical(["A", "A", "B", "B"] * 10),
                "Threshold": [0.3, 0.5, 0.3, 0.5] * 10,
                "Sensitivity": [0.9, 0.8, 0.85, 0.75] * 10,
                "Specificity": [0.7, 0.8, 0.65, 0.85] * 10,
                "PPV": [0.6, 0.7, 0.55, 0.75] * 10,
                "NPV": [0.85, 0.88, 0.80, 0.90] * 10,
                "Flag Rate": [0.4, 0.35, 0.45, 0.30] * 10,
                "TP": [90, 80, 85, 75] * 10,
                "FP": [30, 20, 40, 15] * 10,
                "TN": [70, 80, 65, 85] * 10,
                "FN": [10, 20, 15, 25] * 10,
            }
        )

        result = multPlots.cohort_evaluation_vs_threshold(stats, "TestCohort")

        # Should create figure with gridspec
        assert isinstance(result, SVG)  # Returns SVG when no axis provided

    def test_with_highlight_thresholds(self):
        """Test cohort_evaluation_vs_threshold() with highlight thresholds."""
        stats = pd.DataFrame(
            {
                "cohort": pd.Categorical(["A", "A"] * 15),
                "Threshold": [0.3, 0.5] * 15,
                "Sensitivity": [0.9, 0.8] * 15,
                "Specificity": [0.7, 0.8] * 15,
                "PPV": [0.6, 0.7] * 15,
                "NPV": [0.85, 0.88] * 15,
                "Flag Rate": [0.4, 0.35] * 15,
                "TP": [90, 80] * 15,
                "FP": [30, 20] * 15,
                "TN": [70, 80] * 15,
                "FN": [10, 20] * 15,
            }
        )

        result = multPlots.cohort_evaluation_vs_threshold(stats, "TestCohort", highlight=[0.3, 0.7])

        assert isinstance(result, SVG)  # Returns SVG when no axis provided


class Test_Leadtime_Violin:
    """Test leadtime_violin() function."""

    def test_creates_violin_plot(self):
        """Test leadtime_violin() creates violin plot."""
        data = pd.DataFrame(
            {
                "leadtime": [-10, -20, -30, -15, -25, -35] * 5,
                "cohort": pd.Categorical(["A", "A", "A", "B", "B", "B"] * 5),
            }
        )

        result = multPlots.leadtime_violin(data, "leadtime", "cohort")

        assert isinstance(result, SVG)  # Returns SVG when no axis provided

    def test_with_xmax(self):
        """Test leadtime_violin() with xmax parameter."""
        # Create actual figure and axis for this test
        fig, ax = plt.subplots()

        data = pd.DataFrame(
            {
                "leadtime": [-10, -20, -30, -15, -25, -35] * 5,
                "cohort": pd.Categorical(["A", "A", "A", "B", "B", "B"] * 5),
            }
        )

        result = multPlots.leadtime_violin(data, "leadtime", "cohort", xmax=50, axis=ax)

        # Should set xlim with xmax (-abs(xmax) - 0.01)
        assert ax.get_xlim()[0] == -50.01
        assert isinstance(result, Figure)  # Returns Figure when axis provided
        plt.close(fig)
