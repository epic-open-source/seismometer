import builtins
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seismometer.core.decorators import DiskCachedFunction, export


@pytest.fixture(autouse=True)
def restore_builtins_print():
    """Ensure builtins.print is restored after each test.

    This is needed because indented_function decorator modifies builtins.print,
    and if an exception is raised, it may not be restored (decorator has no try/finally).
    """
    original_print = builtins.print
    yield
    builtins.print = original_print


def get_test_function():
    def foo(arg1, kwarg1=None):
        if kwarg1 is None:
            return arg1 + 1
        else:
            return kwarg1 + 1

    return foo


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

        test_fn = get_test_function()
        new_fn = export(test_fn)

        assert new_fn(1) == 2
        assert new_fn(1, 5) == 6
        assert __all__ == ["foo"]

        with pytest.raises(ImportError):
            export(test_fn)

    def test_mod_none(self):
        """
        Tests the export decorator in three ways:
           First that the function works with required and key-word arguments
           Second that __all__ was updated
           Finally, that calling it again causes the naming collision error
        """
        global __all__
        __all__ = []

        test_fn = get_test_function()
        test_fn.__module__ = None
        new_fn = export(test_fn)

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


# ============================================================================
# ADDITIONAL EDGE CASE TESTS
# ============================================================================


class TestDiskCachedFunctionCacheManagement:
    """Test DiskCachedFunction cache file management."""

    def test_cache_files_created_in_subdirectory(self, disk_cached_str):
        """Test that cache files are created in function-specific subdirectories."""

        @disk_cached_str
        def foo(x: str) -> str:
            return x.upper()

        # First call creates cache
        assert foo("hello") == "HELLO"

        # Cache structure: cache_dir/function_name/hash
        cache_files = list(disk_cached_str.cache_dir.glob("**/*"))
        cache_files = [f for f in cache_files if f.is_file()]
        assert len(cache_files) >= 1

    def test_cache_deleted_file_causes_recomputation(self, disk_cached_str):
        """Test that deleted cache file causes re-computation."""
        global call_count
        call_count = 0

        @disk_cached_str
        def foo(x: str) -> str:
            global call_count
            call_count += 1
            return x.upper()

        # First call creates cache
        assert foo("world") == "WORLD"
        assert call_count == 1

        # Delete the cache file
        cache_files = list(disk_cached_str.cache_dir.glob("**/*"))
        cache_files = [f for f in cache_files if f.is_file()]
        assert len(cache_files) >= 1
        cache_files[0].unlink()

        # Should re-compute since cache is missing
        result = foo("world")
        assert result == "WORLD"
        assert call_count == 2


class TestDiskCachedFunctionConcurrentAccess:
    """Test DiskCachedFunction with simulated concurrent access."""

    def test_multiple_calls_same_args(self, disk_cached_str):
        """Test multiple calls with same arguments hit cache."""
        global call_count
        call_count = 0

        @disk_cached_str
        def foo(x: str) -> str:
            global call_count
            call_count += 1
            return x.upper()

        # Multiple calls with same args
        results = [foo("test") for _ in range(10)]

        # All should return same result
        assert all(r == "TEST" for r in results)
        # Only computed once (cached for rest)
        assert call_count == 1

    def test_interleaved_calls_different_args(self, disk_cached_str):
        """Test interleaved calls with different arguments."""
        global call_count
        call_count = 0

        @disk_cached_str
        def foo(x: str) -> str:
            global call_count
            call_count += 1
            return x.upper()

        # Interleaved calls
        results = []
        for i in range(5):
            results.append(foo("a"))
            results.append(foo("b"))

        # Should have correct results
        assert results == ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
        # Only 2 unique computations (a and b)
        assert call_count == 2

    def test_cache_survives_multiple_function_calls(self, disk_cached_str):
        """Test that cache persists across multiple function invocations."""
        global call_count
        call_count = 0

        @disk_cached_str
        def foo(x: str) -> str:
            global call_count
            call_count += 1
            return x.upper()

        # First batch
        for _ in range(3):
            foo("test")

        first_count = call_count

        # Second batch (should use cache)
        for _ in range(3):
            foo("test")

        # Call count should not increase
        assert call_count == first_count


class TestDiskCachedFunctionSymlinkHandling:
    """Test DiskCachedFunction with symlinks."""

    @pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="Symlinks not supported on this platform")
    def test_cache_dir_as_symlink(self, tmp_path):
        """Test that cache works when cache_dir is a symlink."""
        # Create actual directory
        actual_dir = tmp_path / "actual_cache"
        actual_dir.mkdir()

        # Create symlink to it
        symlink_dir = tmp_path / "symlink_cache"
        symlink_dir.symlink_to(actual_dir)

        # Create cache with symlink path
        def save_fn(string, filepath: Path):
            filepath.write_text(string)

        def load_fn(filepath: Path):
            return filepath.read_text()

        cache_decorator = DiskCachedFunction(cache_name="test", save_fn=save_fn, load_fn=load_fn, return_type=str)
        cache_decorator.SEISMOMETER_CACHE_DIR = symlink_dir
        cache_decorator.enable()

        @cache_decorator
        def foo(x: str) -> str:
            return x.upper()

        # Should work through symlink
        result = foo("hello")
        assert result == "HELLO"

        # Cache file should exist in actual directory (cache files have no extension)
        cache_files = [f for f in actual_dir.glob("**/*") if f.is_file()]
        assert len(cache_files) >= 1

        cache_decorator.clear_all()


class TestIndentedFunctionDecorator:
    """Test indented_function() decorator (completely untested)."""

    def test_indented_function_basic_usage(self, capsys):
        """Test that indented_function adds indentation to print statements."""
        from seismometer.core.decorators import indented_function

        @indented_function
        def foo():
            print("Hello")
            print("World")

        foo()

        captured = capsys.readouterr()
        # Output should be indented with ">  " prefix
        assert ">  Hello" in captured.out
        assert ">  World" in captured.out

    def test_indented_function_with_arguments(self, capsys):
        """Test indented_function with function arguments."""
        from seismometer.core.decorators import indented_function

        @indented_function
        def foo(name, greeting="Hello"):
            print(f"{greeting}, {name}!")

        foo("Alice")

        captured = capsys.readouterr()
        assert ">  Hello, Alice!" in captured.out

    def test_indented_function_with_return_value(self):
        """Test indented_function preserves return values."""
        from seismometer.core.decorators import indented_function

        @indented_function
        def foo(x):
            print(f"Processing {x}")
            return x * 2

        result = foo(5)
        assert result == 10

    def test_indented_function_with_no_prints(self):
        """Test indented_function with function that doesn't print."""
        from seismometer.core.decorators import indented_function

        @indented_function
        def foo(x, y):
            return x + y

        result = foo(3, 4)
        assert result == 7

    def test_indented_function_nested_calls(self, capsys):
        """Test indented_function with nested function calls."""
        from seismometer.core.decorators import indented_function

        @indented_function
        def inner():
            print("Inner function")

        @indented_function
        def outer():
            print("Outer function")
            inner()

        outer()

        captured = capsys.readouterr()
        # Both should be indented
        assert ">  Outer function" in captured.out
        assert ">  Inner function" in captured.out

    def test_indented_function_preserves_function_name(self):
        """Test that indented_function preserves function metadata."""
        from seismometer.core.decorators import indented_function

        @indented_function
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_indented_function_with_exception(self, capsys):
        """Test indented_function when function raises exception."""
        from seismometer.core.decorators import indented_function

        @indented_function
        def foo():
            print("Before error")
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            foo()

        captured = capsys.readouterr()
        # Print before error should still be indented
        assert ">  Before error" in captured.out

    def test_indented_function_multiple_decorators(self):
        """Test indented_function combined with other decorators."""
        from seismometer.core.decorators import indented_function

        def multiply_by_two(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs) * 2

            return wrapper

        @multiply_by_two
        @indented_function
        def foo(x):
            return x + 1

        result = foo(5)
        assert result == 12  # (5 + 1) * 2


class TestExportDecoratorEdgeCases:
    """Test export decorator with additional edge cases."""

    def test_export_with_lambda(self):
        """Test export decorator with lambda function."""
        global __all__
        __all__ = []

        # Lambdas don't have proper __name__, so this should handle gracefully
        lambda_fn = lambda x: x + 1  # noqa: E731
        exported_fn = export(lambda_fn)

        assert exported_fn(5) == 6

    def test_export_preserves_docstring(self):
        """Test that export decorator preserves docstring."""
        global __all__
        __all__ = []

        def documented_function():
            """This is a docstring."""
            return 42

        exported_fn = export(documented_function)

        assert exported_fn.__doc__ == "This is a docstring."
        assert exported_fn() == 42

    def test_export_with_class_method(self):
        """Test export decorator with class methods."""
        global __all__
        __all__ = []

        class MyClass:
            @staticmethod
            def my_method():
                return "method result"

        exported_method = export(MyClass.my_method)

        assert exported_method() == "method result"


class TestDiskCachedFunctionEdgeCases:
    """Test DiskCachedFunction with additional edge cases."""

    def test_cache_with_none_argument(self, disk_cached_str):
        """Test caching with None as argument."""

        @disk_cached_str
        def foo(x) -> str:
            return str(x)

        result = foo(None)
        assert result == "None"

        # Should cache None argument
        result2 = foo(None)
        assert result2 == "None"

    def test_cache_with_empty_string_argument(self, disk_cached_str):
        """Test caching with empty string argument."""

        @disk_cached_str
        def foo(x: str) -> str:
            return f"[{x}]"

        result = foo("")
        assert result == "[]"

        # Should cache empty string
        result2 = foo("")
        assert result2 == "[]"

    def test_cache_recreated_after_manual_deletion(self, disk_cached_str):
        """Test that cache works after manual file deletion."""
        global call_count
        call_count = 0

        @disk_cached_str
        def foo(x: str) -> str:
            global call_count
            call_count += 1
            return x.upper()

        # Create cache
        result1 = foo("test")
        assert result1 == "TEST"
        assert call_count == 1

        # Manually delete cache files (simulating corruption or cleanup)
        cache_files = list(disk_cached_str.cache_dir.glob("**/*"))
        cache_files = [f for f in cache_files if f.is_file()]
        for f in cache_files:
            f.unlink()

        # Should re-compute after cache deletion
        result2 = foo("test")
        assert result2 == "TEST"
        assert call_count == 2
