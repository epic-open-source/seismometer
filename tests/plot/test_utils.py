from unittest.mock import Mock, patch

import matplotlib.pyplot as plt

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
