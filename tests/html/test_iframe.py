import os
from pathlib import Path
from unittest.mock import patch

import pytest
from IPython.display import HTML, IFrame

import seismometer.html.iframe as undertest

TEST_HTML_CONTENT = "<html>This is a local webpage</html>"


@pytest.fixture
def html_path(tmp_path):
    prev_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        html_path = tmp_path / "test.html"
        html_path.write_text(TEST_HTML_CONTENT)
        yield html_path
    finally:
        os.chdir(prev_cwd)


class TestIFrameSupport:
    def test_host_supports_iframe(self, html_path):
        with patch.object(undertest.NotebookHost, "supports_iframe", return_value=True):
            might_be_iframe = undertest.load_as_iframe(html_path)

            assert isinstance(might_be_iframe, IFrame)
            assert might_be_iframe.src == html_path.relative_to(Path.cwd())

    def test_host_does_not_support_iframe(self, html_path):
        with patch.object(undertest.NotebookHost, "supports_iframe", return_value=False):
            might_be_iframe = undertest.load_as_iframe(html_path)

            assert isinstance(might_be_iframe, HTML)
            assert TEST_HTML_CONTENT in might_be_iframe.data
