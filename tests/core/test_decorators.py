from pathlib import Path

import pytest

from seismometer.core._decorators import DiskCachedFunction, export


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
