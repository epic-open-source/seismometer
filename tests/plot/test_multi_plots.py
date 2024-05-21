from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.testing as pdt
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
        assert mock_ax.text.call_count == 1

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
        assert mock_ax.text.call_count == 1
