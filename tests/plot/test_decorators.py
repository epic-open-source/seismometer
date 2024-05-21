import shutil
from unittest.mock import patch

import numpy as np
import pytest

from seismometer.plot.mpl.decorators import disk_cached_function, export


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


def dummy_func():
    return "You dummy"


@disk_cached_function(".decorator_cache")
def a_pickled_cache(badrev):
    return f"I am a pickled {badrev.reverse}!"


@disk_cached_function(".decorator_cache", cache_type="npz")
def a_pickled_numpy_cache(num):
    return np.array(range(num))


class BadReverse:
    def __init__(self, string):
        self.string = string
        self.reverse = string[::-1]

    def __str__(self):
        return self.string

    def __repr__(self):
        return f"BadReverse({self.string})"


class Test_Cached_Function:
    def teardown(self):
        shutil.rmtree(".decorator_cache", ignore_errors=True)

    def test_caches_call_correctly(self):
        reverse = BadReverse("pumpkin")
        assert a_pickled_cache(reverse) == "I am a pickled nikpmup!"
        reverse.reverse = "melon"
        assert a_pickled_cache(reverse) == "I am a pickled nikpmup!"
        assert a_pickled_cache(BadReverse("melon")) == "I am a pickled nolem!"

    def test_skips_cache_call_correctly(self):
        reverse = BadReverse("nikpmup")
        assert a_pickled_cache(reverse) == "I am a pickled pumpkin!"
        reverse.reverse = "melon"
        assert a_pickled_cache(reverse, cache_policy="skip") == "I am a pickled melon!"

    def test_caches_numpy_correctly(self):
        assert (a_pickled_numpy_cache(4) == np.array((0, 1, 2, 3))).all()
        assert (a_pickled_numpy_cache(4) == np.array((0, 1, 2, 3))).all()
        assert (a_pickled_numpy_cache(5) == np.array((0, 1, 2, 3, 4))).all()

    def test_handles_bad_cache_type(self):
        with pytest.raises(ValueError):
            disk_cached_function(".decorator_cache", cache_type="Not A Cache Type")(dummy_func)()

    def test_handles_bad_cache_policy(self):
        with pytest.raises(ValueError):
            disk_cached_function(".decorator_cache")(dummy_func)(cache_policy="Bad_Policy")

    def test_handles_bad_cache_file(self):
        def bad_open(*args, **kwargs):
            if args[1] == "rb":
                raise EOFError  # this error gets skipped because the cache file doesn't end up existing
            if args[1] == "wb":
                raise OverflowError

        with patch("builtins.open", bad_open):
            reverse = BadReverse("pumpkin")
            assert a_pickled_cache(reverse) == "I am a pickled nikpmup!"
            reverse.reverse = "melon"
            assert a_pickled_cache(reverse) == "I am a pickled melon!"
            assert a_pickled_cache(BadReverse("melon")) == "I am a pickled nolem!"
            shutil.rmtree(".decorator_cache")
