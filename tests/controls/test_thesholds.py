import seismometer.controls.thresholds as undertest


class TestPercentSliderWidget:
    def test_init(self):
        widget = undertest.PercentSliderListWidget(("a", "b"))
        assert widget.names == ("a", "b")
        assert widget.value == (0, 0)
        assert len(widget.sliders) == 2
        assert widget.sliders[0].description == "a"
        assert widget.sliders[1].description == "b"

    def test_init_with_values(self):
        widget = undertest.PercentSliderListWidget(("a", "b"), (0.1, 0.2))
        assert widget.sliders[0].value == 0.1
        assert widget.sliders[1].value == 0.2

    def test_slider_change(self):
        widget = undertest.PercentSliderListWidget(("a", "b"))
        widget.sliders[0].value = 0.1
        assert widget.value == (0.1, 0)

    def test_value_change(self):
        widget = undertest.PercentSliderListWidget(("a", "b"))
        widget.value = (0.1, 0.2)
        assert widget.sliders[0].value == 0.1
        assert widget.sliders[1].value == 0.2


class TestMonotonicPercentSliderWidget:
    def test_init(self):
        widget = undertest.MonotonicPercentSliderListWidget(("a", "b"))
        assert widget.names == ("a", "b")
        assert widget.value == (0, 0)
        assert len(widget.sliders) == 2
        assert widget.sliders[0].description == "a"
        assert widget.sliders[1].description == "b"

    def test_init_with_values(self):
        widget = undertest.MonotonicPercentSliderListWidget(("a", "b"), (0.1, 0.2), increasing=True)
        assert widget.value == (0.1, 0.2)
        assert widget.sliders[0].value == 0.1
        assert widget.sliders[1].value == 0.2

    def test_init_with_values_inceasing(self):
        widget = undertest.MonotonicPercentSliderListWidget(("a", "b"), (0.1, 0.2), increasing=True)
        assert widget.value == (0.1, 0.2)
        widget.sliders[0].value = 0.4
        assert widget.sliders[0].value == 0.4
        assert widget.sliders[1].value == 0.4
        widget.sliders[1].value = 0.3
        assert widget.sliders[0].value == 0.3
        assert widget.sliders[1].value == 0.3

    def test_init_with_values_decreasing(self):
        widget = undertest.MonotonicPercentSliderListWidget(("a", "b"), (0.2, 0.1), increasing=False)
        # the order of the sliders is reveresed, but the final result stays in accending order
        assert widget.value == (0.2, 0.1)
        widget.sliders[0].value = 0.4
        assert widget.sliders[0].value == 0.4
        assert widget.sliders[1].value == 0.1
        widget.sliders[1].value = 0.5
        assert widget.sliders[0].value == 0.5
        assert widget.sliders[1].value == 0.5

    def test_slider_change(self):
        widget = undertest.MonotonicPercentSliderListWidget(("a", "b"))
        widget.sliders[0].value = 0.1
        assert widget.value == (0.1, 0.1)

    def test_value_change(self):
        widget = undertest.MonotonicPercentSliderListWidget(("a", "b"))
        widget.value = (0.1, 0.2)
        assert widget.sliders[0].value == 0.1
        assert widget.sliders[1].value == 0.2
