import logging
import operator
from collections.abc import Iterable
from typing import Optional

import traitlets
from ipywidgets import FloatSlider, Layout, ValueWidget, VBox

from seismometer.plot.mpl._ux import alert_colors

from .styles import WIDE_LABEL_STYLE

logger = logging.getLogger("seismometer")


class ProbabilitySliderListWidget(ValueWidget, VBox):
    """
    Vertical list of sliders, bounded between 0 and 1 with a dynamic step size based on specified decimal precision.
    """

    value = traitlets.Dict(help="The names and values for the slider list")

    def __init__(self, names: Iterable[str], value: Optional[Iterable[float]] = None, decimals: int = 2):
        """
        Vertical list of sliders, bounded between 0 and 1 with a dynamic step size
        based on specified decimal precision.

        Parameters
        ----------
        names : Iterable[str]
            Slider names
        value : Optional[Iterable[float]], optional
            Slider start values, by default None, starts all sliders at zero.
        decimals : int, optional
            Number of decimal places (determines slider step size, e.g., 2 → 0.01), by default 2.
        """
        self.names = tuple(names)  # cast to static tuple
        if value and len(value) != len(names):
            raise ValueError(f"Value length {len(value)} does not match names length {len(names)}")
        self.decimals = decimals
        step_size = 1 / 10**decimals
        readout_fmt = f".{decimals}f"
        values = tuple(0 for _ in names) if not value else tuple(value)  # set initial values
        values = tuple(min(1.0, max(0, val)) for val in values)  # clamp values to [0, 1]
        self.value = {k: round(v, self.decimals) for k, v in zip(self.names, values)}
        self.sliders = {}
        for name, val in self.value.items():
            sub_slider = FloatSlider(
                value=round(val, self.decimals),
                min=0,
                max=1.0,
                step=step_size,
                description=name,
                tooltip=name,
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format=readout_fmt,
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
                rounded_val = round(slider.value, self.decimals)
                self.value[slider.description] = rounded_val

    def _on_value_change(self, change=None):
        """Bubble up changes to sliders"""
        self.value_update_in_progress = True
        if change:
            if val := change["new"]:
                for key, value in val.items():
                    rounded_val = round(value, self.decimals)
                    self.sliders[key].value = rounded_val
        self.value_update_in_progress = False


class MonotonicProbabilitySliderListWidget(ProbabilitySliderListWidget):
    """
    A vertical list of probability sliders, each bounded between 0 and 1, with dynamic step size
    based on the specified decimal precision.

    Monotonicity is maintained between the sliders so they are always ascending or descending in value.

    Supports up to 6 sliders.
    """

    def __init__(
        self, names: Iterable[str], value: Optional[Iterable[float]] = None, ascending: bool = True, decimals: int = 2
    ):
        """
        A vertical list of sliders, bounded between 0 and 1 with dynamic step size
        based on the specified decimal precision.

        Parameters
        ----------
        names : Iterable[str]
            Slider names
        value : Optional[Iterable[float]], optional
            Slider start values, by default None, starts all sliders at zero.
        ascending : bool, optional = True
            Forces sliders to be ascending, else decreasing.
            If initial values are not sorted, raise an ValueError.
        decimals : int, optional
            Number of decimal places for slider precision (affects step size and rounding), by default 2.
        """
        if len(names) > 6:
            raise ValueError("MonotonicProbabilitySliderListWidget only supports up to 6 sliders")

        super().__init__(names, value, decimals)

        thresholds = list(self.value.values())
        op = operator.le if ascending else operator.ge
        if not all(op(thresholds[i], thresholds[i + 1]) for i in range(len(thresholds) - 1)):
            # check if sorted
            direction = "ascending" if ascending else "descending"
            raise ValueError(f"Initial values are not sorted, expected {direction}")

        self.ascending = ascending
        for index, sub_slider in enumerate(self.sliders.values()):
            if ascending:
                # line them up so that last slider is the first alert color
                sub_slider.style.handle_color = alert_colors[(len(self.value) - index - 1) % len(alert_colors)]
            else:
                sub_slider.style.handle_color = alert_colors[index % len(alert_colors)]

    def _on_slider_change(self, change=None):
        """Slider has changed, update the value tuple, making sure values are increasing"""
        sliders = list(self.sliders.values())
        if self.value_update_in_progress:
            return
        try:
            slider_index = sliders.index(change["owner"])
            new = change["new"]
            old = change["old"]
        except ValueError:  # Only the sliders can change values
            return

        if self.ascending:  # monotonic increasing
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
        new_tuple = [round(min(1.0, max(0.0, v)), self.decimals) for v in new_tuple]
        self.value = {k: v for k, v in zip(self.sliders.keys(), new_tuple)}
