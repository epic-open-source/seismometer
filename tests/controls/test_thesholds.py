import pytest

import seismometer.controls.thresholds as undertest


class TestPercentSliderWidget:
    def test_init(self):
        widget = undertest.PercentSliderListWidget(("a", "b"))
        assert widget.value == {"a": 0, "b": 0}
        assert len(widget.sliders) == 2
        assert widget.sliders["a"].description == "a"
        assert widget.sliders["b"].description == "b"

    def test_init_with_values(self):
        widget = undertest.PercentSliderListWidget(("a", "b"), (0.1, 0.2))
        assert widget.value == {"a": 0.1, "b": 0.2}
        assert widget.sliders["a"].value == 0.1
        assert widget.sliders["b"].value == 0.2

    def test_slider_change(self):
        widget = undertest.PercentSliderListWidget(("a", "b"))
        widget.sliders["a"].value = 0.1
        widget.sliders["b"].value = 0
        assert widget.value == {"a": 0.1, "b": 0}

    def test_value_change(self):
        widget = undertest.PercentSliderListWidget(("a", "b"))
        widget.value = {"a": 0.1, "b": 0.2}
        assert widget.sliders["a"].value == 0.1
        assert widget.sliders["b"].value == 0.2

    def test_clamps_value(self):
        widget = undertest.PercentSliderListWidget(("a", "b"), (-0.1, 20))
        assert widget.value == {"a": 0, "b": 1}
        assert widget.sliders["a"].value == 0.0
        assert widget.sliders["b"].value == 1.0

    def test_mismatched_names_and_value(self):
        with pytest.raises(ValueError):
            undertest.PercentSliderListWidget(("a", "b", "C"), (-0.1, 20))

    def test_disabled(self):
        widget = undertest.PercentSliderListWidget(("a", "b"))
        assert not widget.disabled
        widget.disabled = True
        assert widget.disabled
        assert widget.sliders["a"].disabled
        assert widget.sliders["b"].disabled


class TestMonotonicPercentSliderWidget:
    def test_init(self):
        widget = undertest.MonotonicPercentSliderListWidget(("a", "b"))
        assert widget.names == ("a", "b")
        assert widget.value == {"a": 0, "b": 0}
        assert len(widget.sliders) == 2
        assert widget.sliders["a"].description == "a"
        assert widget.sliders["b"].description == "b"

    def test_init_with_values(self):
        widget = undertest.MonotonicPercentSliderListWidget(("a", "b"), (0.1, 0.2), increasing=True)
        assert widget.value == {"a": 0.1, "b": 0.2}
        assert widget.sliders["a"].value == 0.1
        assert widget.sliders["b"].value == 0.2

    def test_init_with_values_inceasing(self):
        widget = undertest.MonotonicPercentSliderListWidget(("a", "b"), (0.1, 0.2), increasing=True)
        assert widget.value == {"a": 0.1, "b": 0.2}
        widget.sliders["a"].value = 0.4
        assert widget.sliders["a"].value == 0.4
        assert widget.sliders["b"].value == 0.4
        assert widget.value == {"a": 0.4, "b": 0.4}
        widget.sliders["b"].value = 0.3
        assert widget.sliders["a"].value == 0.3
        assert widget.sliders["b"].value == 0.3
        assert widget.value == {"a": 0.3, "b": 0.3}

    def test_init_with_values_decreasing(self):
        widget = undertest.MonotonicPercentSliderListWidget(("a", "b"), (0.2, 0.1), increasing=False)
        # the order of the sliders is reveresed
        assert widget.value == {"a": 0.2, "b": 0.1}
        widget.sliders["a"].value = 0.4
        assert widget.sliders["a"].value == 0.4
        assert widget.sliders["b"].value == 0.1
        assert widget.value == {"a": 0.4, "b": 0.1}
        widget.sliders["b"].value = 0.5
        assert widget.sliders["a"].value == 0.5
        assert widget.sliders["b"].value == 0.5
        assert widget.value == {"a": 0.5, "b": 0.5}
        widget.sliders["a"].value = 0.05
        assert widget.sliders["a"].value == 0.05
        assert widget.sliders["b"].value == 0.05
        assert widget.value == {"a": 0.05, "b": 0.05}

    def test_slider_change(self):
        widget = undertest.MonotonicPercentSliderListWidget(("a", "b"))
        assert widget.value == {"a": 0, "b": 0}
        widget.sliders["b"].value = 0.1
        assert widget.value == {"a": 0, "b": 0.1}

    def test_value_change(self):
        widget = undertest.MonotonicPercentSliderListWidget(("a", "b"))
        widget.value = {"a": 0.1, "b": 0.2}
        assert widget.sliders["a"].value == 0.1
        assert widget.sliders["b"].value == 0.2

    def test_clamps_value(self):
        widget = undertest.MonotonicPercentSliderListWidget(("a", "b"), (-0.1, 20))
        assert widget.value == {"a": 0, "b": 1}

    def test_value_out_of_order_raises_value_error(self):
        with pytest.raises(ValueError):
            undertest.MonotonicPercentSliderListWidget(("a", "b"), (-0.1, 20), increasing=False)
