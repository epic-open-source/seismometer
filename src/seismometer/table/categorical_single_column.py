import base64
import logging
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import plot_likert
import traitlets
from ipywidgets import HTML, Box, Checkbox, Dropdown, Layout, ValueWidget, VBox

from seismometer.controls.explore import ExplorationWidget
from seismometer.controls.selection import DisjointSelectionListsWidget
from seismometer.controls.styles import BOX_GRID_LAYOUT, WIDE_LABEL_STYLE, html_title

# from seismometer.data import pandas_helpers as pdh
from seismometer.data.filter import FilterRule
from seismometer.seismogram import Seismogram

logger = logging.getLogger("seismometer")


class OrdinalCategoricalSinglePlot:
    def __init__(
        self,
        metric_col,
        plot_type="Likert Plot",
        compute_percentages=False,
        bar_labels=False,
        cohort_dict: dict = None,
        title=None,
    ):
        self.metric_col = metric_col
        self.plot_type = plot_type
        self.title = title
        self.compute_percentages = compute_percentages
        self.bar_labels = bar_labels
        self.cohort_col = next(iter(cohort_dict))
        self.cohort_values = list(cohort_dict[self.cohort_col])

        sg = Seismogram()
        cohort_filter = FilterRule.from_cohort_dictionary(cohort_dict)
        self.dataframe = cohort_filter.filter(sg.dataframe)

        self.plot_functions = self.initialize_plot_functions()

        self.values = None
        self._extract_metric_values()

    def _extract_metric_values(self):
        sg = Seismogram()
        if self.metric_col in sg.metrics:
            self.values = sg.metrics[self.metric_col].metric_details.values
        self.values = self.values or sorted(self.dataframe[self.metric_col].unique())

    @classmethod
    def initialize_plot_functions(cls):
        return {
            "Likert Plot": cls.plot_likert,
        }

    def plot_likert(self):
        fig, ax = plt.subplots()
        plt.close(fig)
        df = (
            self.dataframe.groupby([self.cohort_col, self.metric_col], observed=False).size().reset_index(name="count")
        )
        df = df.pivot(index=self.cohort_col, columns=self.metric_col, values="count").fillna(0)
        df = df.loc[self.cohort_values]
        plot_likert.plot_counts(
            df,
            scale=self.values,
            compute_percentages=self.compute_percentages,
            bar_labels=self.bar_labels,
            bar_labels_color="snow",
            ax=ax,
        )
        plt.tight_layout()
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


def ordinal_categorical_single_col_plot(
    metric_col: str,
    compute_percentages,
    bar_labels,
    cohort_dict,
    *,
    title: str = None,
) -> HTML:
    """ """
    plot = OrdinalCategoricalSinglePlot(
        metric_col,
        plot_type="Likert Plot",
        compute_percentages=compute_percentages,
        bar_labels=bar_labels,
        cohort_dict=cohort_dict,
        title=title,
    )
    return plot.generate_plot()


# endregion
# region Plot Controls


class ExploreSingleCategoricalPlots(ExplorationWidget):
    def __init__(self, title: Optional[str] = None):
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.title = title

        super().__init__(
            title="Plot Cohort Distribution",
            option_widget=CategoricalFeedbackSingleColumnOptionsWidget(
                list(set(metric for metric_group in sg.metric_groups for metric in sg.metric_groups[metric_group])),
                cohort_groups=sg.available_cohort_groups,
                compute_percentages=False,
                bar_labels=False,
                title=title,
            ),
            plot_function=ordinal_categorical_single_col_plot,
            initial_plot=False,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the analytics table."""
        cohort_dict = {self.option_widget.cohort_list[0]: tuple(self.option_widget.cohort_list[1])}
        args = (
            self.option_widget.metric_col,
            self.option_widget.compute_percentages,
            self.option_widget.bar_labels,
            cohort_dict,
        )
        kwargs = {"title": self.title}
        return args, kwargs


class CategoricalFeedbackSingleColumnOptionsWidget(Box, ValueWidget, traitlets.HasTraits):
    value = traitlets.Dict(help="The selected values for the ordinal categorical options.")

    def __init__(
        self,
        metrics,
        cohort_groups,
        compute_percentages,
        bar_labels,
        *,
        model_options_widget=None,
        title: str = None,
    ):
        """ """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.model_options_widget = model_options_widget
        self.title = title

        metric_options = metrics or list(sg.metrics.keys())
        self._metric_col = Dropdown(
            options=metric_options,
            value=metric_options[0],
            description="Metric",
            style=WIDE_LABEL_STYLE,
        )

        self._compute_percentages = Checkbox(
            value=compute_percentages,
            description="Show as percentages?",
            tooltip="Show values as percentages.",
            style=WIDE_LABEL_STYLE,
        )

        self._bar_labels = Checkbox(
            value=bar_labels,
            description="Show values?",
            tooltip="Show values on the plot.",
            style=WIDE_LABEL_STYLE,
        )
        self._cohort_list = DisjointSelectionListsWidget(options=cohort_groups, title="Cohort Filter", select_all=True)

        self._metric_col.observe(self._on_value_changed, names="value")
        self._compute_percentages.observe(self._on_value_changed, names="value")
        self._bar_labels.observe(self._on_value_changed, names="value")
        self._cohort_list.observe(self._on_value_changed, names="value")

        v_children = [
            html_title("Plot Options"),
            self._metric_col,
            self._compute_percentages,
            self._bar_labels,
        ]
        if model_options_widget:
            v_children.insert(0, model_options_widget)
            self.model_options_widget.observe(self._on_value_changed, names="value")

        vbox_layout = Layout(align_items="flex-end", flex="0 0 auto")

        vbox = VBox(children=v_children, layout=vbox_layout)

        super().__init__(
            children=[vbox, self._cohort_list],
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
        self._metric_col.disabled = value
        self._compute_percentages.disabled = value
        self._bar_labels.disabled = value
        self._cohort_list.disabled = value
        if self.model_options_widget:
            self.model_options_widget.disabled = value

    def _on_value_changed(self, change=None):
        new_value = {
            "metric_col": self._metric_col.value,
            "compute_percentages": self._compute_percentages.value,
            "bar_labels": self._bar_labels.value,
            "cohort_list": self._cohort_list.value,
        }
        if self.model_options_widget:
            new_value["model_options"] = self.model_options_widget.value
        self.value = new_value

    @property
    def metric_col(self):
        return self._metric_col.value

    @property
    def compute_percentages(self):
        return self._compute_percentages.value

    @property
    def bar_labels(self):
        return self._bar_labels.value

    @property
    def cohort_list(self):
        return self._cohort_list.value


# endregion
