import seismometer.controls.styles as undertest


class TestHelperFunctions:
    def test_html_title(self):
        title = "Test Title"
        html = undertest.html_title(title)
        assert title in html.value
        assert "h4" in html.value
