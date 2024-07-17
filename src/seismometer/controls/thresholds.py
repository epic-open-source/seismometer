import logging
from collections.abc import Iterable

import traitlets
from ipywidgets import FloatSlider, Layout, ValueWidget, VBox

from seismometer.plot.mpl._ux import alert_colors

from .styles import WIDE_LABEL_STYLE

logger = logging.getLogger("seismometer")


class PercentSliderListWidget(ValueWidget, VBox):
    """
    Vertical list of sliders
    """

    value = traitlets.Dict(help="The names and values for the slider list")

    def __init__(self, names: Iterable[str], value: Iterable[int] = None):
        """A vertical list of sliders

        Parameters
        ----------
        names : tuple[str]
            Slider names
        value : Optional[tuple[str]], optional
            Slider start values, by default None, starts all sliders at zero.
        """
        self.names = tuple(names)  # cast to static tuple
        if value and len(value) != len(names):
            raise ValueError(f"Value length {len(value)} does not match names length {len(names)}")
        values = tuple(0 for _ in names) if not value else tuple(value)  # set initial values
        values = tuple(min(1.0, max(0, val)) for val in values)  # clamp values to [0, 1]
        self.value = {k: v for k, v in zip(self.names, values)}
        self.sliders = {}
        for name, val in self.value.items():
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
                style=WIDE_LABEL_STYLE,
            )
            self.sliders[name] = sub_slider
            sub_slider.observe(self._on_slider_change, "value")

        self.value_update_in_progress = False

        super().__init__(children=list(self.sliders.values()), layout=Layout(width="max-content", min_width="300px"))
        self.observe(self._on_value_change, "value")
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        for slider in self.sliders.values():
            slider.disabled = disabled

    def _on_slider_change(self, change=None):
        """Slider has changed, update the value tuple"""
        if self.value_update_in_progress:
            return
        if change:
            if (slider := change["owner"]) in self.sliders.values():
                self.value[slider.description] = slider.value

    def _on_value_change(self, change=None):
        """Bubble up changes to sliders"""
        self.value_update_in_progress = True
        if change:
            if val := change["new"]:
                for key, value in val.items():
                    self.sliders[key].value = value
        self.value_update_in_progress = False


class MonotonicPercentSliderListWidget(PercentSliderListWidget):
    """
    Vertical list of sliders, with increasing values
    """

    def __init__(self, names: tuple[str], value: tuple[int] = None, increasing: bool = True):
        """A vertical list of sliders

        Parameters
        ----------
        names : tuple[str]
            Slider names
        value : Optional[tuple[str]], optional
            Slider start values, by default None, starts all sliders at zero.
        increasing : bool, optional = True
            Forces sliders to be increasing, else decreasing.
            If initial values are not sorted, raise an ValueError.
        """
        super().__init__(names, value)

        if tuple(sorted(self.value.values(), reverse=not increasing)) != tuple(self.value.values()):
            raise ValueError("Initial values are not sorted")

        self.increasing = increasing
        for index, sub_slider in enumerate(self.sliders.values()):
            if increasing:
                sub_slider.style.handle_color = alert_colors[(len(self.value) - index) % len(alert_colors)]
            else:
                sub_slider.style.handle_color = alert_colors[index % len(alert_colors)]

    def _on_slider_change(self, change=None):
        """Slider has changed, update the value tuple, making sure values are increating"""
        sliders = list(self.sliders.values())
        if self.value_update_in_progress:
            return
        try:
            slider_index = sliders.index(change["owner"])
        except ValueError:  # Only the sliders can change values
            return
        new = change["new"]
        old = change["old"]
        if self.increasing:  # monotonic increasing
            if new > old:  # increase in value, increase all sliders after this one
                new_tuple = [slider.value for slider in sliders[:slider_index]] + [
                    max(new, slider.value) for slider in sliders[slider_index:]
                ]
            else:  # decrease in value, decrease all sliders before this one
                new_tuple = [min(new, slider.value) for slider in sliders[:slider_index]] + [
                    slider.value for slider in sliders[slider_index:]
                ]
        else:  # monotonic decreasing
            if new > old:  # increase in value, increase all sliders before this one
                new_tuple = [max(new, slider.value) for slider in sliders[:slider_index]] + [
                    slider.value for slider in sliders[slider_index:]
                ]
            else:  # decrease in value, decrease all sliders after this one
                new_tuple = [slider.value for slider in sliders[:slider_index]] + [
                    min(new, slider.value) for slider in sliders[slider_index:]
                ]
        self.value = {k: v for k, v in zip(self.sliders.keys(), new_tuple)}
