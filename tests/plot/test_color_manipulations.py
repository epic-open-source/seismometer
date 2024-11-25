import pytest

from seismometer.plot.mpl.color_manipulation import color_to_rgb, create_bar, lighten_color


class TestColorManipulations:
    def test_color_to_rgb(self):
        assert color_to_rgb("#FF0000") == (255, 0, 0)
        assert color_to_rgb("red") == (255, 0, 0)
        with pytest.raises(ValueError):
            color_to_rgb("invalid_color")

    def test_lighten_color(self):
        assert lighten_color("#FF0000") == "#f97878"
        assert lighten_color("red") == "#f97878"
        assert lighten_color("red", n_colors=5, position=-1) == "#ff0000"
        assert lighten_color("blue", n_colors=5, position=2) == "#7878f9"
        with pytest.raises(ValueError):
            lighten_color("invalid_color")

    def test_create_bar(self):
        bar_html = create_bar(0.5, 100, 20, "red", "lightgray", 0.5)
        assert "width:50.0px" in bar_html
        assert "background-color:rgba(255, 0, 0, 0.5)" in bar_html
        assert "height:20px" in bar_html
        assert "width: 100px" in bar_html
