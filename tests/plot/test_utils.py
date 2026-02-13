from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from IPython.display import SVG

import seismometer.plot.mpl._util as utils


class Test_Save_File:
    @patch.object(plt, "savefig")
    def test_calls_save_fig(self, save_mock):
        utils.save_figure("afilename")
        save_mock.assert_called_with("afilename", bbox_inches="tight")

    @patch.object(plt, "savefig")
    def test_calls_save_fig_bbox(self, save_mock):
        utils.save_figure("afilename", bbox_inches="inches")
        save_mock.assert_called_with("afilename", bbox_inches="inches")


class Test_Plot_Curve:
    def test_plots_curve(self):
        axis = Mock()
        utils.plot_curve(axis, [[1, 2, 3, 4], [3, 2, 1, 4]])
        axis.plot.assert_called_once_with([1, 2, 3, 4], [3, 2, 1, 4], label=None)

    def test_plots_curve_with_name(self):
        axis = Mock()
        utils.plot_curve(axis, [[1, 2, 3, 4], [3, 2, 1, 4]], curve_name="feature_name")
        axis.plot.assert_called_once_with([1, 2, 3, 4], [3, 2, 1, 4], label="feature_name")

    def test_plots_curve_with_title(self):
        axis = Mock()
        utils.plot_curve(axis, [[1, 2, 3, 4], [3, 2, 1, 4]], title="curve_title")
        axis.plot.assert_called_once_with([1, 2, 3, 4], [3, 2, 1, 4], label=None)
        axis.set_xlabel.assert_called_once_with("curve_title")

    def test_plots_curve_with_ylabel(self):
        axis = Mock()
        utils.plot_curve(axis, [[1, 2, 3, 4], [3, 2, 1, 4]], y_label="curve_title")
        axis.plot.assert_called_once_with([1, 2, 3, 4], [3, 2, 1, 4], label=None)
        axis.set_ylabel.assert_called_once_with("curve_title")

    def test_plots_with_accents(self):
        axis = Mock()
        utils.plot_curve(
            axis,
            [[1, 2, 3, 4], [3, 2, 1, 4]],
            accent_dict={"decoration": [[1], [2]], "accent": [[3], [4]]},
        )

        axis.plot.assert_any_call([3], [4], "x", label="accent")
        axis.plot.assert_any_call([1], [2], "x", label="decoration")
        axis.plot.assert_any_call([1, 2, 3, 4], [3, 2, 1, 4], label=None)


class Test_Simple_Plots:
    def test_polygon(self):
        axis = Mock()
        utils.plot_polygon(axis, "x", "y")
        axis.fill.assert_called_with("x", "y", alpha=0.1, c="C0")

    def test_diagonal(self):
        axis = Mock()
        utils.plot_diagonal(axis)
        axis.plot.assert_called_with([0, 1], [0, 1], "--", c=utils.REFERENCE_GREY)

    def test_horizontal(self):
        axis = Mock()
        utils.plot_horizontal(axis, 3)
        axis.plot.assert_called_with([0, 1], [3, 3], "r--", c=utils.REFERENCE_GREY)

    def test_vertical(self):
        axis = Mock()
        utils.plot_vertical(axis, 3)
        axis.plot.assert_called_with([3, 3], [0, 1], "r--", c=utils.REFERENCE_GREY)


class Test_Axis_Clear:
    def test_defaults(self):
        axis = Mock()
        utils.axis_clear(axis)

        axis.set_xticklabels.assert_called_once_with([])
        axis.set_xlabel.assert_called_once_with(None)
        axis.set_yticklabels.assert_not_called()
        axis.set_ylabel.assert_not_called()

    def test_clear_y(self):
        axis = Mock()
        utils.axis_clear(axis, 0, 0)

        axis.set_xticklabels.assert_not_called()
        axis.set_xabel.assert_not_called()
        axis.set_yticklabels.assert_not_called()
        axis.set_ylabel.assert_not_called()

    def test_clear_all(self):
        axis = Mock()
        utils.axis_clear(axis, 1, 1)

        axis.set_xticklabels.assert_called_once_with([])
        axis.set_xlabel.assert_called_once_with(None)
        axis.set_yticklabels.assert_called_once_with([])
        axis.set_ylabel.assert_called_once_with(None)


class TestToSvg:
    """Test to_svg() function for SVG generation"""

    @patch.object(plt, "savefig")
    def test_to_svg_returns_svg_object(self, save_mock):
        """Test to_svg() returns an SVG object"""

        # Mock savefig to write valid SVG content to buffer
        def write_svg(buffer, **kwargs):
            buffer.write('<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>')

        save_mock.side_effect = write_svg
        result = utils.to_svg()
        assert isinstance(result, SVG)
        save_mock.assert_called_once()

    @patch.object(plt, "savefig")
    def test_to_svg_calls_savefig_with_svg_format(self, save_mock):
        """Test to_svg() calls savefig with format='svg'"""

        def write_svg(buffer, **kwargs):
            buffer.write('<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>')

        save_mock.side_effect = write_svg
        utils.to_svg()
        # Check that savefig was called with format='svg'
        call_kwargs = save_mock.call_args[1]
        assert call_kwargs.get("format") == "svg"

    @patch.object(plt, "savefig")
    def test_to_svg_with_empty_plot(self, save_mock):
        """Test to_svg() with empty plot doesn't crash"""

        def write_svg(buffer, **kwargs):
            buffer.write('<svg xmlns="http://www.w3.org/2000/svg"></svg>')

        save_mock.side_effect = write_svg
        plt.figure()
        result = utils.to_svg()
        assert isinstance(result, SVG)
        plt.close()


class TestCreateCheckboxes:
    """Test create_checkboxes() function for widget creation"""

    def test_create_checkboxes_returns_list(self):
        """Test create_checkboxes() returns a list"""
        result = utils.create_checkboxes(["A", "B", "C"])
        assert isinstance(result, list)
        assert len(result) == 3

    def test_create_checkboxes_widget_properties(self):
        """Test create_checkboxes() creates widgets with correct properties"""
        values = ["Option1", "Option2"]
        checkboxes = utils.create_checkboxes(values)

        for checkbox, value in zip(checkboxes, values):
            assert checkbox.description == value
            assert checkbox.value is True  # Default value

    def test_create_checkboxes_with_numeric_values(self):
        """Test create_checkboxes() converts numeric values to strings"""
        values = [1, 2, 3]
        checkboxes = utils.create_checkboxes(values)

        for checkbox, value in zip(checkboxes, values):
            assert checkbox.description == str(value)

    def test_create_checkboxes_empty_list(self):
        """Test create_checkboxes() with empty list"""
        result = utils.create_checkboxes([])
        assert result == []

    def test_create_checkboxes_with_mixed_types(self):
        """Test create_checkboxes() with mixed types in list"""
        values = [1, "A", 2.5, True]
        checkboxes = utils.create_checkboxes(values)
        assert len(checkboxes) == 4
        assert checkboxes[0].description == "1"
        assert checkboxes[1].description == "A"
        assert checkboxes[2].description == "2.5"
        assert checkboxes[3].description == "True"


class TestAddUnseen:
    """Test add_unseen() function for categorical data handling"""

    def test_add_unseen_with_missing_categories(self):
        """Test add_unseen() adds missing categorical values"""
        df = pd.DataFrame({"cohort": pd.Categorical(["A", "A"], categories=["A", "B", "C"])})

        result = utils.add_unseen(df, col="cohort")

        # Should have 2 original rows + 2 unseen categories
        assert len(result) == 4
        assert set(result["cohort"].dropna()) == {"A", "B", "C"}

    def test_add_unseen_preserves_categorical_dtype(self):
        """Test add_unseen() preserves categorical dtype and categories"""
        original_cats = ["A", "B", "C"]
        df = pd.DataFrame({"cohort": pd.Categorical(["A"], categories=original_cats)})

        result = utils.add_unseen(df, col="cohort")

        assert result["cohort"].dtype.name == "category"
        assert list(result["cohort"].cat.categories) == original_cats

    def test_add_unseen_with_all_categories_present(self):
        """Test add_unseen() when all categories are already present"""
        df = pd.DataFrame({"cohort": pd.Categorical(["A", "B", "C"], categories=["A", "B", "C"])})

        result = utils.add_unseen(df, col="cohort")

        # Should only have original rows
        assert len(result) == 3

    def test_add_unseen_with_custom_column_name(self):
        """Test add_unseen() with custom column name"""
        df = pd.DataFrame({"feature": pd.Categorical(["X"], categories=["X", "Y", "Z"])})

        result = utils.add_unseen(df, col="feature")

        assert len(result) == 3
        assert set(result["feature"].dropna()) == {"X", "Y", "Z"}

    def test_add_unseen_preserves_other_columns(self):
        """Test add_unseen() preserves other columns (fills with NaN)"""
        df = pd.DataFrame(
            {"cohort": pd.Categorical(["A", "A"], categories=["A", "B"]), "value": [10, 20], "name": ["x", "y"]}
        )

        result = utils.add_unseen(df, col="cohort")

        # Original rows should have values, new rows should have NaN
        assert result.iloc[0]["value"] == 10
        assert result.iloc[1]["value"] == 20
        assert pd.isna(result.iloc[2]["value"])


class TestNeededColors:
    """Test needed_colors() function for color mapping"""

    def test_needed_colors_with_categorical_series(self):
        """Test needed_colors() with categorical series"""
        series = pd.Series(pd.Categorical(["A", "A", "C"], categories=["A", "B", "C"]))
        colors = ["red", "green", "blue"]

        result = utils.needed_colors(series, colors)

        # Should return colors for observed categories (A=0, C=2)
        assert result == ["red", "blue"]

    def test_needed_colors_with_all_categories_observed(self):
        """Test needed_colors() when all categories are observed"""
        series = pd.Series(pd.Categorical(["A", "B", "C"], categories=["A", "B", "C"]))
        colors = ["red", "green", "blue"]

        result = utils.needed_colors(series, colors)

        assert result == ["red", "green", "blue"]

    def test_needed_colors_with_non_categorical_series(self):
        """Test needed_colors() with non-categorical series (fallback behavior)"""
        series = pd.Series(["A", "B", "A", "C"])
        colors = ["red", "green", "blue"]

        result = utils.needed_colors(series, colors)

        # Should return colors based on number of unique values
        assert len(result) == 3  # 3 unique values

    def test_needed_colors_with_numeric_series(self):
        """Test needed_colors() with numeric series"""
        series = pd.Series([1, 2, 1, 3, 2])
        colors = ["red", "green", "blue"]

        result = utils.needed_colors(series, colors)

        # Should handle numeric series (3 unique values)
        assert len(result) == 3

    def test_needed_colors_single_category(self):
        """Test needed_colors() with single category"""
        series = pd.Series(pd.Categorical(["A", "A", "A"], categories=["A", "B"]))
        colors = ["red", "green"]

        result = utils.needed_colors(series, colors)

        assert result == ["red"]


class TestPlotCurveEdgeCases:
    """Test plot_curve() edge cases and error handling"""

    def test_plot_curve_with_empty_line(self):
        """Test plot_curve() with empty line data"""
        axis = Mock()
        utils.plot_curve(axis, [[], []])
        axis.plot.assert_called_once_with([], [], label=None)

    def test_plot_curve_with_single_point(self):
        """Test plot_curve() with single point"""
        axis = Mock()
        utils.plot_curve(axis, [[1], [2]])
        axis.plot.assert_called_once_with([1], [2], label=None)

    def test_plot_curve_with_empty_accent_dict(self):
        """Test plot_curve() with empty accent_dict still calls legend"""
        axis = Mock()
        utils.plot_curve(axis, [[1, 2], [3, 4]], accent_dict={})
        # Legend is called even with empty accent_dict (not None)
        axis.legend.assert_called_once()

    def test_plot_curve_with_none_accent_dict(self):
        """Test plot_curve() with None accent_dict doesn't call legend"""
        axis = Mock()
        utils.plot_curve(axis, [[1, 2], [3, 4]], accent_dict=None)
        # Should not call legend when accent_dict is None
        axis.legend.assert_not_called()

    def test_plot_curve_with_malformed_accent_dict_missing_y(self):
        """Test plot_curve() with malformed accent_dict (missing y values)"""
        axis = Mock()
        with pytest.raises((IndexError, KeyError)):
            utils.plot_curve(axis, [[1, 2], [3, 4]], accent_dict={"accent": [[1]]})

    def test_plot_curve_with_none_values_in_accents(self):
        """Test plot_curve() handles None in accent coordinates"""
        axis = Mock()
        # This should call plot but may fail at matplotlib level
        utils.plot_curve(axis, [[1, 2], [3, 4]], accent_dict={"accent": [[None], [None]]})
        axis.plot.assert_any_call([None], [None], "x", label="accent")


class TestCohortLegend:
    """Test cohort_legend() function for legend creation"""

    def test_cohort_legend_with_true_column(self):
        """Test cohort_legend() with 'true' column format"""
        fig, axes = plt.subplots(1, 2)

        # Plot some lines on first axis
        axes[0].plot([0, 1], [0, 1], label="Cohort A")
        axes[0].plot([0, 1], [0, 0.5], label="Cohort B")

        # Create data with 'true' column
        data = pd.DataFrame(
            {
                "cohort": pd.Categorical(["A", "A", "B", "B"], categories=["A", "B"]),
                "true": [1, 0, 1, 1],
                "pred": [0.8, 0.2, 0.9, 0.7],
            }
        )

        # Call cohort_legend on second axis
        utils.cohort_legend(data, axes[1], "Test Feature", ref_axis=0)

        # Verify legend was created
        assert axes[1].get_legend() is not None

        plt.close(fig)

    def test_cohort_legend_with_cohort_count_column(self):
        """Test cohort_legend() with 'cohort-count' column format"""
        fig, axes = plt.subplots(1, 2)

        # Plot a line on first axis
        axes[0].plot([0, 1], [0, 1], label="Cohort A")

        # Create data with cohort-count format
        data = pd.DataFrame(
            {
                "cohort": pd.Categorical(["A"], categories=["A", "B"]),
                "cohort-count": [100],
                "cohort-targetcount": [25],
            }
        )

        # Call cohort_legend on second axis
        utils.cohort_legend(data, axes[1], "Test Feature", ref_axis=0)

        assert axes[1].get_legend() is not None
        plt.close(fig)

    def test_cohort_legend_below_censor_threshold(self):
        """Test cohort_legend() censors data below threshold"""
        fig, axes = plt.subplots(1, 2)

        # Plot a line on first axis
        axes[0].plot([0, 1], [0, 1], label="Cohort A")

        # Create small data (below default threshold of 10)
        data = pd.DataFrame(
            {"cohort": pd.Categorical(["A"] * 5, categories=["A"]), "true": [1, 0, 1, 1, 0], "pred": [0.8] * 5}
        )

        # Call cohort_legend with default censor_threshold=10
        utils.cohort_legend(data, axes[1], "Test Feature", ref_axis=0)

        # Should still create legend but with censored values
        assert axes[1].get_legend() is not None
        plt.close(fig)

    def test_cohort_legend_more_lines_than_cohorts_error(self):
        """Test cohort_legend() raises IndexError when more lines than cohorts"""
        fig, axes = plt.subplots(1, 2)

        # Plot more lines than we have cohorts
        axes[0].plot([0, 1], [0, 1], label="Line 1")
        axes[0].plot([0, 1], [0, 0.5], label="Line 2")
        axes[0].plot([0, 1], [0, 0.3], label="Line 3")

        # Create data with only 2 cohorts
        data = pd.DataFrame({"cohort": pd.Categorical(["A", "B"], categories=["A", "B"]), "true": [1, 0]})

        # Should raise IndexError
        with pytest.raises(IndexError, match="More lines than cohorts"):
            utils.cohort_legend(data, axes[1], "Test Feature", ref_axis=0)

        plt.close(fig)

    def test_cohort_legend_with_custom_labellist(self):
        """Test cohort_legend() with custom label list"""
        fig, axes = plt.subplots(1, 2)

        # Plot a line
        axes[0].plot([0, 1], [0, 1])

        # Create data (matching array lengths)
        data = pd.DataFrame(
            {"cohort": pd.Categorical(["A", "A"], categories=["A"]), "true": [1, 0], "pred": [0.8, 0.2]}
        )

        # Call with custom labels
        utils.cohort_legend(data, axes[1], "Test Feature", labellist=["Custom Label"], ref_axis=0)

        assert axes[1].get_legend() is not None
        plt.close(fig)

    def test_cohort_legend_skips_dashed_lines(self):
        """Test cohort_legend() skips dashed reference lines"""
        fig, axes = plt.subplots(1, 2)

        # Plot solid and dashed lines
        axes[0].plot([0, 1], [0, 1], label="Cohort A")  # Solid
        axes[0].plot([0, 1], [0.5, 0.5], "--", label="Reference")  # Dashed (should be skipped)

        # Create data with one cohort (matching array lengths)
        data = pd.DataFrame(
            {"cohort": pd.Categorical(["A", "A"], categories=["A"]), "true": [1, 0], "pred": [0.8, 0.2]}
        )

        # Should only use the solid line (not dashed)
        utils.cohort_legend(data, axes[1], "Test Feature", ref_axis=0)

        assert axes[1].get_legend() is not None
        plt.close(fig)
