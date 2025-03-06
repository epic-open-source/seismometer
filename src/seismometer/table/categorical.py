import base64
import logging
from io import BytesIO
from typing import Optional

import pandas as pd

# import ipywidgets as widgets
import traitlets
from ipywidgets import HTML, Box, Layout, ValueWidget, VBox
from matplotlib.figure import Figure

from seismometer.controls.explore import ExplorationWidget
from seismometer.controls.selection import MultiselectDropdownWidget, MultiSelectionListWidget
from seismometer.controls.styles import BOX_GRID_LAYOUT, html_title

# from seismometer.data import pandas_helpers as pdh
from seismometer.data.filter import FilterRule
from seismometer.plot.mpl.likert import likert_plot
from seismometer.seismogram import Seismogram

logger = logging.getLogger("seismometer")


class OrdinalCategoricalPlot:
    def __init__(
        self,
        metrics: list[str],
        plot_type: str = "Likert Plot",
        cohort_dict: Optional[dict[str, tuple]] = None,
        title: Optional[str] = None,
    ):
        """
        Initializes the OrdinalCategoricalPlot class.

        Parameters
        ----------
        metrics : list[str]
            List of metrics (columns) to be plotted.
        plot_type : str, optional
            Type of plot to generate, by default "Likert Plot".
        cohort_dict : Optional[dict[str, tuple]], optional
            Dictionary defining the cohort filter, by default None.
        title : Optional[str], optional
            Title of the plot, by default None.
        """
        self.metrics = metrics
        self.plot_type = plot_type
        self.title = title
        self.plot_functions = self.initialize_plot_functions()

        sg = Seismogram()
        cohort_filter = FilterRule.from_cohort_dictionary(cohort_dict)
        self.dataframe = cohort_filter.filter(sg.dataframe)

        self.values = None
        self._extract_metric_values()

    def _extract_metric_values(self):
        """
        Extracts the unique metric values for the provided metrics (columns in sg.datagframe).
        """
        sg = Seismogram()
        for metric_col in self.metrics:
            if metric_col in sg.metrics:
                metric = sg.metrics[metric_col]
                if metric.metric_details.values is not None:
                    self.values = metric.metric_details.values
                    return
        self.values = sorted(pd.unique(self.dataframe[self.metrics].values.ravel()))

    @classmethod
    def initialize_plot_functions(cls):
        """
        Initializes the plot functions.

        Returns
        -------
        dict[str, Callable]
            Dictionary mapping plot types to their corresponding functions.
        """
        return {
            "Likert Plot": cls.plot_likert,
        }

    def plot_likert(self):
        """
        Generates a Likert plot to show the distribution of values across provided metrics.

        Returns
        -------
        Figure
            The generated Likert plot figure.
        """
        return likert_plot(df=self._count_values_in_columns())

    def _count_values_in_columns(self):
        """
        Counts occurrences of each unique value in each metric column.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the counts of each unique value in each metric column.
        """
        # Create a dictionary to store the counts
        data = {"Feedback Metrics": self.metrics}

        # Count occurrences of each unique value in each metric column
        for value in self.values:
            data[value] = [self.dataframe[col].value_counts().get(value, 0) for col in self.metrics]

        # Create a new DataFrame from the dictionary and set "Feedback Metrics" as index
        counts_df = pd.DataFrame(data)
        counts_df.set_index("Feedback Metrics", inplace=True)

        return counts_df

    def fig_to_html(self, fig: Figure):
        """
        Converts a Matplotlib figure to an HTML string.

        Parameters
        ----------
        fig: Figure
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
        return HTML(html_str)

    def generate_plot(self):
        """
        Generates the plot based on the specified plot type.

        Returns
        -------
        Figure
            The generated plot figure.

        Raises
        ------
        ValueError
            If the specified plot type is unknown.
        """
        if self.plot_type not in self.plot_functions:
            raise ValueError(f"Unknown plot type: {self.plot_type}")

        fig = self.plot_functions[self.plot_type](self)
        return fig


# region Plots Wrapper


def ordinal_categorical_plot(
    metrics: list[str],
    cohort_dict: dict[str, tuple],
    *,
    title: Optional[str] = None,
) -> Figure:
    """
    Generates a likert plot for the provided list of ordinal categorical metric columns.

    Parameters
    ----------
    metrics : list[str]
        Metric columns to be plotted.
    cohort_dict : dict[str, tuple]
        Dictionary defining the cohort filter.
    title : Optional[str], optional
        Title of the plot, by default None.

    Returns
    -------
    Figure
        Figure object generated by the plot.
    """
    sg = Seismogram
    cohort_dict = cohort_dict or sg.available_cohort_groups
    plot = OrdinalCategoricalPlot(
        metrics,
        plot_type="Likert Plot",
        cohort_dict=cohort_dict,
        title=title,
    )
    return plot.generate_plot()


# endregion
# region Plot Controls


class ExploreCategoricalPlots(ExplorationWidget):
    def __init__(self, title: Optional[str] = None):
        """
        Initializes the ExploreCategoricalPlots class.

        Parameters
        ----------
        title : str, optional
            Title of the plot, by default None.
        """
        sg = Seismogram()
        self.title = title

        super().__init__(
            title="Plot Metrics",
            option_widget=CategoricalFeedbackOptionsWidget(
                list(sg.metric_groups.keys()),
                cohort_dict=sg.available_cohort_groups,
                title=title,
            ),
            plot_function=ordinal_categorical_plot,
            initial_plot=False,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the ordinal categorical plot."""
        args = (
            list(self.option_widget.metrics),
            self.option_widget.cohort_dict,
        )
        kwargs = {"title": self.title}
        return args, kwargs


class CategoricalFeedbackOptionsWidget(Box, ValueWidget, traitlets.HasTraits):
    value = traitlets.Dict(help="The selected values for the ordinal categorical options.")

    def __init__(
        self,
        metric_groups: list[str],
        cohort_dict: dict[str, list[str]],
        *,
        model_options_widget=None,
        title: str = None,
    ):
        """
        Initializes the CategoricalFeedbackOptionsWidget class.

        Parameters
        ----------
        metric_groups : list[str]
            List of metric groups.
        cohort_dict : dict[str, list[str]]
            Dictionary defining the cohort filter.
        model_options_widget : Optional[widget], optional
            Additional widget options if needed, by default None.
        title : str, optional
            Title of the plot, by default None.
        """
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

        self._cohort_dict = MultiSelectionListWidget(sg.available_cohort_groups, title="Cohorts")

        self._metric_groups.observe(self._on_value_changed, names="value")
        self._metrics.observe(self._on_value_changed, names="value")
        self._cohort_dict.observe(self._on_value_changed, names="value")

        v_children = [
            html_title("Plot Options"),
            self._metric_groups,
            self._metrics,
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
    def cohort_dict(self):
        return self._cohort_dict.value or self.all_cohorts


# endregion
