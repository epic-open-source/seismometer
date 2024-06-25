from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seismometer.core.decorators import DiskCachedFunction, export


def foo(arg1, kwarg1=None):
    if kwarg1 is None:
        return arg1 + 1
    else:
        return kwarg1 + 1


class Test_Export:
    def test_export(self):
        """
        Tests the export decorator in three ways:
           First that the function works with required and key-word arguments
           Second that __all__ was updated
           Finally, that calling it again causes the naming collision error
        """
        global __all__
        __all__ = []

        new_fn = export(foo)

        assert new_fn(1) == 2
        assert new_fn(1, 5) == 6
        assert __all__ == ["foo"]

        with pytest.raises(ImportError):
            export(foo)

    def test_mod_none(self):
        """
        Tests the export decorator in three ways:
           First that the function works with required and key-word arguments
           Second that __all__ was updated
           Finally, that calling it again causes the naming collision error
        """
        global __all__
        __all__ = []

        foo.__module__ = None
        new_fn = export(foo)

        assert new_fn(1) == 2
        assert new_fn(1, 5) == 6
        assert __all__ == []


class BadHash:
    def __init__(self, string):
        self.string = string
        self.unobserved = string[::-1]

    def __str__(self):
        return self.string

    def __repr__(self):
        return f"BadHash({self.string})"

    def __hash__(self):
        return hash(self.string)


@pytest.fixture
def disk_cached_str(tmp_path):
    def save_fn(string, filepath: Path):
        filepath.write_text(string)

    def load_fn(filepath: Path):
        return filepath.read_text()

    cache_decorator = DiskCachedFunction(cache_name="string", save_fn=save_fn, load_fn=load_fn, return_type=str)
    cache_decorator.SEISMOMETER_CACHE_DIR = tmp_path
    cache_decorator.enable()
    yield cache_decorator
    cache_decorator.clear_all()


@pytest.fixture
def large_dataframes():
    """two dataframes with the same data at the beginning and end, but different in the middle"""
    size = 2000
    df1 = pd.DataFrame(
        {
            "Val": np.random.normal(20, 10, size=size),
            "Cat": np.random.choice(["A", "B", "C", "D"], size=size, replace=True, p=[2 / 5, 1 / 2, 3 / 40, 1 / 40]),
            "T/F": np.random.choice([0, 1, np.nan], size=size, replace=True, p=[4 / 7, 2 / 7, 1 / 7]),
        }
    )

    # make the last row the different from the first
    df1.at[1999, "Val"] = -df1.at[0, "Val"]
    df1.at[1999, "Cat"] = "A" if df1.at[0, "Cat"] == "B" else "B"
    df1.at[1999, "T/F"] = 1 if df1.at[0, "T/F"] == 0 else 0
    df2 = df1.copy(deep=True)  # copy the first dataframe

    # replace the middle third of the second dataframe with new data
    start, stop = size // 3, 2 * size // 3
    middle_rows = df1[start:stop].copy(deep=True)
    middle_rows["Val"] = -middle_rows["Val"]
    middle_rows["Cat"] = middle_rows["Cat"].apply(lambda x: "A" if x == "B" else "B")
    middle_rows["T/F"] = middle_rows["T/F"].apply(lambda x: 1 if x == 0 else 0)
    df2[start:stop] = middle_rows
    return df1, df2


class Test_DiskCachedFunction:
    def test_errors_for_wrong_annotation(self, disk_cached_str):
        with pytest.raises(TypeError) as error:

            @disk_cached_str
            def foo(x: BadHash) -> int:
                return len(x.unobserved)

        assert "foo" in str(error.value)
        assert "must return 'str' object" in str(error.value)

    def test_errors_for_no_annotation(self, disk_cached_str):
        with pytest.raises(TypeError) as error:

            @disk_cached_str
            def foo(x: BadHash):
                return len(x.unobserved)

        assert "foo" in str(error.value)
        assert "must return 'str' object" in str(error.value)

    def test_errors_for_good_annotation_but_bad_return(self, disk_cached_str):
        @disk_cached_str
        def foo(x: BadHash) -> str:
            return len(x.unobserved)

        with pytest.raises(ValueError) as error:
            foo(BadHash("browns"))
        assert "'foo' did not return the expected type 'str" in str(error.value)

    def test_caches_results(self, disk_cached_str):
        @disk_cached_str
        def foo(x: BadHash) -> str:
            return x.unobserved

        treat = BadHash("treat")
        treat.unobserved = "trick"
        assert foo(treat) == "trick"
        treat.unobserved = "treat?"
        assert foo(treat) == "trick"

    def test_caches_unordered_kwargs(self, disk_cached_str):
        global count
        count = 0

        @disk_cached_str
        def foo(x: str, *, y: str, z: int) -> str:
            global count
            count += 1
            return f"{x}-{y}-{z}"

        assert foo("a", y="b", z=1) == "a-b-1"
        assert count == 1
        assert foo("a", z=1, y="b") == "a-b-1"
        assert count == 1  # same cache hit
        assert foo("a", z=2, y="b") == "a-b-2"
        assert count == 2  # new cache hit

    def test_can_clear_cached_results(self, disk_cached_str):
        @disk_cached_str
        def foo(x: BadHash) -> str:
            return x.unobserved

        treat = BadHash("treat")
        treat.unobserved = "trick"

        assert foo(treat) == "trick"
        disk_cached_str.clear()
        assert disk_cached_str.cache_dir.exists() is False

        treat.unobserved = "treat!"
        assert foo(treat) == "treat!"

    def test_can_ignore_cached_results(self, disk_cached_str):
        @disk_cached_str
        def foo(x: BadHash) -> str:
            return x.unobserved

        treat = BadHash("treat")
        treat.unobserved = "trick"

        assert foo(treat) == "trick"
        disk_cached_str.disable()
        assert disk_cached_str.cache_dir.exists() is True
        treat.unobserved = "treat!"
        assert foo(treat) == "treat!"

    def test_can_skip_caching_results(self, disk_cached_str):
        @disk_cached_str
        def foo(x: BadHash) -> str:
            return x.unobserved

        treat = BadHash("treat")
        treat.unobserved = "trick"
        assert disk_cached_str.cache_dir.exists() is False
        disk_cached_str.disable()

        assert foo(treat) == "trick"
        assert disk_cached_str.cache_dir.exists() is False
        treat.unobserved = "treat!"
        assert foo(treat) == "treat!"


class Test_DiskCachedFunction_With_Pandas:
    def test_supports_dataframe_cachable(self, disk_cached_str):
        global count
        count = 0

        @disk_cached_str
        def foo(x: pd.DataFrame) -> str:
            global count
            buffer = StringIO()
            count += 1
            x.to_csv(buffer)
            return buffer.getvalue().replace("\r", "")

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert foo(df) == ",a,b\n0,1,4\n1,2,5\n2,3,6\n"
        assert count == 1
        assert foo(df) == ",a,b\n0,1,4\n1,2,5\n2,3,6\n"
        assert count == 1

    def test_supports_dataframe_reordered_as_cachable(self, disk_cached_str):
        global count
        count = 0

        @disk_cached_str
        def foo(x: pd.DataFrame) -> str:
            global count
            buffer = StringIO()
            count += 1
            x.to_csv(buffer)
            return buffer.getvalue().replace("\r", "")

        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[2, 1, 0])
        assert foo(df1) == ",a,b\n2,1,4\n1,2,5\n0,3,6\n"
        assert count == 1
        df2 = df1.reindex()  # reverses row order
        assert foo(df2) == ",a,b\n2,1,4\n1,2,5\n0,3,6\n"
        assert count == 1

    def test_supports_dataframe_can_skip_cachable(self, disk_cached_str):
        global count
        count = 0

        @disk_cached_str
        def foo(x: pd.DataFrame) -> str:
            global count
            buffer = StringIO()
            count += 1
            x.to_csv(buffer)
            return buffer.getvalue().replace("\r", "")

        disk_cached_str.disable()
        assert disk_cached_str.is_enabled() is False
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert foo(df) == ",a,b\n0,1,4\n1,2,5\n2,3,6\n"
        assert count == 1
        assert foo(df) == ",a,b\n0,1,4\n1,2,5\n2,3,6\n"
        assert count == 2

    def test_can_differentiate_similar_dataframes(self, disk_cached_str, large_dataframes):
        """
        This test is to assure that if we swap some of the data that is not visible to the
        stringified dataframe (which includes start and end rows only), that the cache will
        still be able to differentiate between the two dataframes, via a valid hash
        """
        global count
        count = 0

        @disk_cached_str
        def foo(x: pd.DataFrame) -> str:
            global count
            count += 1
            # return the middle row as a sttring
            return str(x.iloc[len(x) // 2])

        df1, df2 = large_dataframes
        mid1 = str(df1.iloc[len(df1) // 2])
        mid2 = str(df2.iloc[len(df2) // 2])

        assert str(df1) == str(df2)  # fixture used in test should make sure that these are the same

        assert foo(df1) == mid1
        assert count == 1
        assert foo(df2) == mid2  # different data
        assert count == 2

    def test_can_match_large_dataframes(self, disk_cached_str, large_dataframes):
        """
        This test is to assure that if we reorder some of the data, that the cache will
        still be able to match two dataframes with the same content, via a valid hash
        """
        global count
        count = 0

        @disk_cached_str
        def foo(x: pd.DataFrame) -> str:
            global count
            count += 1
            # return the first row as a sttring
            x = str(x.iloc[0])
            return x

        df1, _ = large_dataframes
        df2 = df1.sort_index(ascending=False).sort_index(axis=1, ascending=True)  # reverse rows, resort columns
        first1 = str(df1.iloc[0])
        first2 = str(df2.iloc[0])
        assert first1 != first2  # fixture used in test should make sure that these are different

        assert foo(df1) == first1
        assert count == 1

        assert foo(df2) == first1
        assert foo(df2) != first2
        assert count == 1

    def test_can_match_pandas_index(self, disk_cached_str):
        """Tests that we can differentiate between pandas Index objects"""

        global count
        count = 0

        @disk_cached_str
        def first_value(x: pd.Index) -> str:
            global count
            count += 1
            return str(x[0])

        assert first_value(pd.Index([1, 2, 3])) == "1"
        assert count == 1
        assert first_value(pd.RangeIndex(1, 4)) == "1"
        assert count == 1
        assert first_value(pd.Index([3, 2, 1])) == "3"
        assert count == 2

    def test_can_match_pandas_series(self, disk_cached_str):
        """Tests that we can differentiate between pandas Index objects"""

        global count
        count = 0

        @disk_cached_str
        def first_value(x: pd.Series) -> str:
            global count
            count += 1
            return str(x.loc[1])

        assert first_value(pd.Series(["A", "B", "C"], index=pd.Index([1, 2, 3]))) == "A"
        assert count == 1
        assert first_value(pd.Series(["C", "B", "A"], index=pd.Index([3, 2, 1]))) == "A"
        assert count == 1
        assert first_value(pd.Series(["C", "B", "A"], index=pd.Index([1, 2, 3]))) == "C"
        assert count == 2
