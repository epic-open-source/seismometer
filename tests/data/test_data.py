import numpy as np
import pandas as pd
import pytest

import seismometer.data.cohorts as undertest


class Test_Resolve_Col:
    df = pd.DataFrame({"col1": np.random.rand(10), "col2": np.random.rand(10)})

    def test_col_np1d(self):
        data = np.random.rand(10)
        actual = undertest.resolve_col_data(self.df, data)
        assert np.array_equal(data, actual)

    def test_col_np2d(self):
        data = np.random.rand(10, 2)
        actual = undertest.resolve_col_data(self.df, data)
        assert np.array_equal(data[:, 1], actual)

    def test_col_list(self):
        data = list(range(10))
        with pytest.raises(TypeError):
            undertest.resolve_col_data(self.df, data)

    def test_col_name(self):
        actual1 = undertest.resolve_col_data(self.df, "col1")
        actual2 = undertest.resolve_col_data(self.df, "col2")

        assert self.df["col1"].equals(actual1)
        assert self.df["col2"].equals(actual2)

        with pytest.raises(KeyError):
            undertest.resolve_col_data(self.df, "col3")
