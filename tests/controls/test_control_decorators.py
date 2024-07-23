from IPython.display import HTML

import seismometer.controls.decorators as undertest


class TestCachedHTML:
    def test_save_load_html(self, tmp_path):
        html_path = tmp_path / "test.html"
        undertest.html_save(HTML("<html></html>"), html_path)
        html = undertest.html_load(html_path)
        assert html.data == "<html></html>"
