import logging

import traitlets
from ipywidgets import FloatSlider, Layout, ValueWidget, VBox

from seismometer.plot.mpl._ux import alert_colors

logger = logging.getLogger("seismometer")


class PercentSliderListWidget(ValueWidget, VBox):
    """
    Vertical list of sliders
    """

    value = traitlets.Tuple(help="The selected values for the slider list")

    def __init__(self, names: tuple[str], value: tuple[int] = None):
        """A vertical list of sliders

        Parameters
        ----------
        names : tuple[str]
            Slider names
        value : Optional[tuple[str]], optional
            Slider start values, by default None, starts all sliders at zero.
        """
        super().__init__()
        self.names = tuple(names)  # make immutable
        self.value = value or tuple(0 for _ in names)  # set initial value
        self.sliders = []
        for name, val in zip(self.names, self.value):
            sub_slider = FloatSlider(
                value=val,
                min=0,
                max=1.0,
                step=0.01,
                description=name,
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format=".2f",
            )
            self.sliders.append(sub_slider)
            sub_slider.observe(self._on_slider_change, "value")
        self.layout = Layout(width="max-content", min_width="300px")
        self.children = self.sliders
        self.observe(self._on_value_change, "value")

    def _on_slider_change(self, change=None):
        """Slider has changed, update the value tuple"""
        self.value = tuple(slider.value for slider in self.sliders)

    def _on_value_change(self, change=None):
        """Bubble up changes to sliders"""
        for slider, val in zip(self.sliders, self.value):
            slider.value = val


class MonotonicPercentSliderListWidget(ValueWidget, VBox):
    """
    Vertical list of sliders, with increasing values
    """

    value = traitlets.Tuple(help="The selected values for the slider list")

    def __init__(self, names: tuple[str], value: tuple[int] = None, increasing: bool = True):
        """A vertical list of sliders

        Parameters
        ----------
        names : tuple[str]
            Slider names
        value : Optional[tuple[str]], optional
            Slider start values, by default None, starts all sliders at zero.
        """
        super().__init__()
        self.names = tuple(names) if names else []  # make immutable
        self.increasing = increasing
        self.value = value or tuple(0 for _ in names)  # set initial value
        self.sliders = []
        for name, val in zip(self.names, self.value):
            sub_slider = FloatSlider(
                value=val,
                min=0,
                max=1.0,
                step=0.01,
                description=name,
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format=".2f",
            )
            if increasing:
                sub_slider.style.handle_color = alert_colors[
                    (len(self.value) - len(self.sliders) - 1) % len(alert_colors)
                ]
            else:
                sub_slider.style.handle_color = alert_colors[len(self.sliders) % len(alert_colors)]
            self.sliders.append(sub_slider)
            sub_slider.observe(self._on_slider_change, "value")

        self.children = self.sliders
        self.observe(self._on_value_change, "value")

    def _on_slider_change(self, change=None):
        """Slider has changed, update the value tuple, making sure values are increating"""
        try:
            slider_index = self.sliders.index(change["owner"])
        except ValueError:  # Only the sliders can change values
            return
        new = change["new"]
        old = change["old"]
        if self.increasing:  # monotonic increasing
            if new > old:  # increase in value, increase all sliders after this one
                new_tuple = [slider.value for slider in self.sliders[:slider_index]] + [
                    max(new, slider.value) for slider in self.sliders[slider_index:]
                ]
            else:  # decrease in value, decrease all sliders before this one
                new_tuple = [min(new, slider.value) for slider in self.sliders[:slider_index]] + [
                    slider.value for slider in self.sliders[slider_index:]
                ]
        else:  # monotonic decreasing
            if new > old:  # increase in value, increase all sliders before this one
                new_tuple = [max(new, slider.value) for slider in self.sliders[:slider_index]] + [
                    slider.value for slider in self.sliders[slider_index:]
                ]
            else:  # decrease in value, decrease all sliders after this one
                new_tuple = [slider.value for slider in self.sliders[:slider_index]] + [
                    min(new, slider.value) for slider in self.sliders[slider_index:]
                ]
        self.value = new_tuple

    def _on_value_change(self, change=None):
        """Bubble up changes to sliders"""
        for slider, val in zip(self.sliders, self.value):
            slider.value = val
