import pytest
from IPython.display import HTML

import seismometer.controls.decorators as under_test


class BadReverse:
    def __init__(self, string):
        self.string = string
        self.reverse = string[::-1]

    def __str__(self):
        return self.string

    def __repr__(self):
        return f"BadReverse({self.string})"


class Test_Cached_Function:
    def teardown_method(self):
        under_test.disk_cached_html_segment.clear_cache()
        under_test.SEISMOMETER_CACHE_ENABLED = False

    def test_caches_call_correctly(self):
        under_test.SEISMOMETER_CACHE_ENABLED = True

        @under_test.disk_cached_html_segment
        def an_html_cache(badrev) -> HTML:
            return HTML(f"I am a pickled {badrev.reverse}!")

        reverse = BadReverse("pumpkin")
        assert an_html_cache(reverse).data == "I am a pickled nikpmup!"
        reverse.reverse = "melon"
        assert an_html_cache(reverse).data == "I am a pickled nikpmup!"
        assert an_html_cache(BadReverse("melon")).data == "I am a pickled nolem!"

    def test_clears_cache_call_correctly(self):
        under_test.SEISMOMETER_CACHE_ENABLED = True

        @under_test.disk_cached_html_segment
        def an_html_cache(badrev) -> HTML:
            return HTML(f"I am a pickled {badrev.reverse}!")

        reverse = BadReverse("nikpmup")
        assert an_html_cache(reverse).data == "I am a pickled pumpkin!"
        under_test.disk_cached_html_segment.clear_cache()
        reverse.reverse = "melon"
        assert an_html_cache(reverse).data == "I am a pickled melon!"

    def tet_no_caching_if_disabled(self):
        under_test.SEISMOMETER_CACHE_ENABLED = False

        @under_test.disk_cached_html_segment
        def an_html_cache(badrev) -> HTML:
            return HTML(f"I am a pickled {badrev.reverse}!")

        reverse = BadReverse("pumpkin")
        assert an_html_cache(reverse).data == "I am a pickled nikpmup!"
        reverse.reverse = "melon"
        assert an_html_cache(reverse).data == "I am a pickled melon!"

    def test_does_not_allow_non_HTML_returns(self):
        under_test.SEISMOMETER_CACHE_ENABLED = True

        @under_test.disk_cached_html_segment
        def an_html_cache(badrev) -> HTML:
            return f"I am a pickled {badrev.reverse}!"

        reverse = BadReverse("pumpkin")
        with pytest.raises(TypeError):
            an_html_cache(reverse)

    def test_does_not_allow_mising_HTML_annotation(self):
        under_test.SEISMOMETER_CACHE_ENABLED = True

        with pytest.raises(TypeError):

            @under_test.disk_cached_html_segment
            def an_html_cache(badrev):
                return HTML(f"I am a pickled {badrev.reverse}!")
