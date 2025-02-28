import base64
import logging
from io import BytesIO
from typing import Optional

import traitlets
from ipywidgets import HTML, Box, Dropdown, Layout, ValueWidget, VBox

from seismometer.controls.explore import ExplorationWidget
from seismometer.controls.selection import DisjointSelectionListsWidget
from seismometer.controls.styles import BOX_GRID_LAYOUT, WIDE_LABEL_STYLE, html_title

# from seismometer.data import pandas_helpers as pdh
from seismometer.data.filter import FilterRule
from seismometer.plot.mpl.likert import likert_plot
from seismometer.seismogram import Seismogram

logger = logging.getLogger("seismometer")


class OrdinalCategoricalSinglePlot:
    def __init__(
        self,
        metric_col,
        plot_type="Likert Plot",
        cohort_dict: dict = None,
        title=None,
    ):
        self.metric_col = metric_col
        self.plot_type = plot_type
        self.title = title
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
        df = self._count_cohort_group_values()
        return likert_plot(df=df, include_counts_plot=True)

    def _count_cohort_group_values(self):
        df = (
            self.dataframe.groupby([self.cohort_col, self.metric_col], observed=False).size().reset_index(name="count")
        )
        df = df.pivot(index=self.cohort_col, columns=self.metric_col, values="count").fillna(0)
        df = df.loc[self.cohort_values]
        return df

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
    cohort_dict,
    *,
    title: str = None,
) -> HTML:
    """ """
    plot = OrdinalCategoricalSinglePlot(
        metric_col,
        plot_type="Likert Plot",
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

        self._cohort_list = DisjointSelectionListsWidget(options=cohort_groups, title="Cohort Filter", select_all=True)

        self._metric_col.observe(self._on_value_changed, names="value")
        self._cohort_list.observe(self._on_value_changed, names="value")

        v_children = [
            html_title("Plot Options"),
            self._metric_col,
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
        self._cohort_list.disabled = value
        if self.model_options_widget:
            self.model_options_widget.disabled = value

    def _on_value_changed(self, change=None):
        new_value = {
            "metric_col": self._metric_col.value,
            "cohort_list": self._cohort_list.value,
        }
        if self.model_options_widget:
            new_value["model_options"] = self.model_options_widget.value
        self.value = new_value

    @property
    def metric_col(self):
        return self._metric_col.value

    @property
    def cohort_list(self):
        return self._cohort_list.value


# endregion
