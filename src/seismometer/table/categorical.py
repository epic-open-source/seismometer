import base64
import logging
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import plot_likert

# import ipywidgets as widgets
import traitlets
from ipywidgets import HTML, Box, Checkbox, Layout, ValueWidget, VBox

from seismometer.controls.explore import ExplorationWidget
from seismometer.controls.selection import MultiselectDropdownWidget, MultiSelectionListWidget
from seismometer.controls.styles import BOX_GRID_LAYOUT, WIDE_LABEL_STYLE, html_title

# from seismometer.data import pandas_helpers as pdh
from seismometer.data.filter import FilterRule
from seismometer.seismogram import Seismogram

logger = logging.getLogger("seismometer")


class OrdinalCategoricalPlot:
    def __init__(
        self,
        metrics,
        plot_type="Likert Plot",
        compute_percentages=False,
        bar_labels=False,
        cohort_dict=None,
        title=None,
    ):
        self.metrics = metrics
        self.plot_type = plot_type
        self.title = title
        self.compute_percentages = compute_percentages
        self.bar_labels = bar_labels
        self.plot_functions = self.initialize_plot_functions()

        sg = Seismogram()
        cohort_filter = FilterRule.from_cohort_dictionary(cohort_dict)
        self.dataframe = cohort_filter.filter(sg.dataframe)

        self.values = None
        self._extract_metric_values()

    def _extract_metric_values(self):
        sg = Seismogram()
        for metric_col in self.metrics:
            if metric_col in sg.metrics:
                metric = sg.metrics[metric_col]
                if metric.metric_details.values is not None:
                    self.values = metric.metric_details.values
                    return
        self.values = sorted(self.dataframe[self.metrics[0]].unique())
        return

    @classmethod
    def initialize_plot_functions(cls):
        return {
            "Likert Plot": cls.plot_likert,
        }

    def plot_likert(self):
        fig, ax = plt.subplots()
        plt.close(fig)
        plot_likert.plot_likert(
            self.dataframe[self.metrics],
            plot_scale=self.values,
            plot_percentage=self.compute_percentages,
            bar_labels=self.bar_labels,
            bar_labels_color="snow",
            ax=ax,
        )
        return fig

    def fig_to_html(self, fig):
        """
        Converts a Matplotlib figure to an HTML string.

        Parameters
        ----------
        fig: matplotlib.figure.Figure
            Matplotlib figure object.

        Returns
        -------
        str
            HTML string of the figure.
        """
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        html_str = f'<img src="data:image/png;base64,{img_str}" alt="Plot">'
        return html_str

    def generate_plot(self):
        if self.plot_type not in self.plot_functions:
            raise ValueError(f"Unknown plot type: {self.plot_type}")

        return self.plot_functions[self.plot_type](self)


# region Plots Wrapper


def ordinal_categorical_plot(
    metrics: str,
    compute_percentages,
    bar_labels,
    cohort_dict,
    # cohort_col,
    *,
    title: str = None,
) -> HTML:
    """ """
    sg = Seismogram
    cohort_dict = cohort_dict or sg.available_cohort_groups
    plot = OrdinalCategoricalPlot(
        metrics,
        plot_type="Likert Plot",
        compute_percentages=compute_percentages,
        cohort_dict=cohort_dict,
        bar_labels=bar_labels,
        title=title,
    )
    return plot.generate_plot()


# endregion
# region Plot Controls


class ExploreCategoricalPlots(ExplorationWidget):
    def __init__(self, title: Optional[str] = None):
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.title = title

        super().__init__(
            title="Plot Metrics",
            option_widget=CategoricalFeedbackOptionsWidget(
                list(sg.metric_groups.keys()),
                cohort_dict=sg.available_cohort_groups,
                compute_percentages=False,
                bar_labels=False,
                title=title,
            ),
            plot_function=ordinal_categorical_plot,
            initial_plot=False,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the analytics table."""
        args = (
            list(self.option_widget.metrics),
            self.option_widget.compute_percentages,
            self.option_widget.bar_labels,
            self.option_widget.cohort_dict,
        )
        kwargs = {"title": self.title}
        return args, kwargs


class CategoricalFeedbackOptionsWidget(Box, ValueWidget, traitlets.HasTraits):
    value = traitlets.Dict(help="The selected values for the ordinal categorical options.")

    def __init__(
        self,
        metric_groups,
        cohort_dict: dict[str, list[str]],
        compute_percentages,
        bar_labels,
        *,
        model_options_widget=None,
        title: str = None,
    ):
        """ """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        metric_groups = metric_groups or list(sg.metric_groups.keys())
        self.model_options_widget = model_options_widget
        self.title = title
        self.all_cohorts = cohort_dict

        self._metric_groups = MultiselectDropdownWidget(
            options=metric_groups,
            value=metric_groups,
            title="Metric Groups",
        )

        all_metrics = list(set(metric for group in metric_groups for metric in sg.metric_groups[group]))
        self._metrics = MultiselectDropdownWidget(
            options=all_metrics,
            value=all_metrics,
            title="Metrics",
        )

        self._compute_percentages = Checkbox(
            value=compute_percentages,
            description="Show as percentages?",
            tooltip="Show values as percentages",
            style=WIDE_LABEL_STYLE,
        )

        self._bar_labels = Checkbox(
            value=bar_labels,
            description="Show values?",
            tooltip="Show values on the plot.",
            style=WIDE_LABEL_STYLE,
        )

        self._cohort_dict = MultiSelectionListWidget(sg.available_cohort_groups, title="Cohorts")

        self._metric_groups.observe(self._on_value_changed, names="value")
        self._metrics.observe(self._on_value_changed, names="value")
        self._compute_percentages.observe(self._on_value_changed, names="value")
        self._bar_labels.observe(self._on_value_changed, names="value")
        self._cohort_dict.observe(self._on_value_changed, names="value")

        v_children = [
            html_title("Plot Options"),
            self._metric_groups,
            self._metrics,
            self._compute_percentages,
            self._bar_labels,
        ]
        if model_options_widget:
            v_children.insert(0, model_options_widget)
            self.model_options_widget.observe(self._on_value_changed, names="value")

        vbox_layout = Layout(align_items="flex-end", flex="0 0 auto")

        vbox = VBox(children=v_children, layout=vbox_layout)

        super().__init__(
            children=[vbox, self._cohort_dict],
            layout=BOX_GRID_LAYOUT,
        )

        self._on_value_changed()
        self._disabled = False

    @property
    def disabled(self):
        return self._disabled

    @disabled.setter
    def disabled(self, value):
        self._disabled = value
        self._metric_groups.disabled = value
        self._metrics.disabled = value
        self._compute_percentages.disabled = value
        self._bar_labels.disabled = value
        self._cohort_dict.disabled = value
        if self.model_options_widget:
            self.model_options_widget.disabled = value

    def _update_disabled_state(self):
        self._metrics.disabled = len(self._metrics.options) == 0

    def _on_value_changed(self, change=None):
        sg = Seismogram()
        metric_groups = self._metric_groups.value
        metrics_set = set(metric for metric_group in metric_groups for metric in sg.metric_groups[metric_group])
        self._metrics.options = list(metrics_set)
        self._metrics.value = list(set(self._metrics.value) & metrics_set)
        self._update_disabled_state()

        new_value = {
            "metric_groups": self._metric_groups.value,
            "metrics": self._metrics.value,
            "compute_percentages": self._compute_percentages.value,
            "bar_labels": self._bar_labels.value,
            "cohort_dict": self._cohort_dict.value,
        }
        if self.model_options_widget:
            new_value["model_options"] = self.model_options_widget.value
        self.value = new_value

    @property
    def metric_groups(self):
        return self._metric_groups.value

    @property
    def metrics(self):
        return self._metrics.value

    @property
    def compute_percentages(self):
        return self._compute_percentages.value

    @property
    def bar_labels(self):
        return self._bar_labels.value

    @property
    def cohort_dict(self):
        return self._cohort_dict.value or self.all_cohorts


# endregion
