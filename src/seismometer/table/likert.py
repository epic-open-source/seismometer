import base64
import logging
import warnings
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_likert
import seaborn as sns

# import ipywidgets as widgets
import traitlets
from ipywidgets import HTML, Box, Checkbox, Dropdown, GridBox, Layout

from seismometer.controls.explore import ExplorationWidget, _combine_scores_checkbox

# from seismometer.controls.selection import MultiselectDropdownWidget
from seismometer.controls.styles import BOX_GRID_LAYOUT, WIDE_LABEL_STYLE

# from seismometer.data import pandas_helpers as pdh
from seismometer.seismogram import Seismogram

logger = logging.getLogger("seismometer")

ORDINAL_CATEGORICAL_PLOT_TYPES = ["bar", "pie", "likert"]


class OrdinalCategoricalPlot:
    def __init__(self, target_cols, plot_type="bar", order_by_count=False, title="", per_context=False):
        sg = Seismogram()
        self.target_cols = target_cols or sg.get_ordinal_categorical_targets()
        self.order_by_count = order_by_count
        if plot_type not in {"bar", "pie", "likert"}:
            raise ValueError(f"Invalid plot_type: {plot_type}. Choose from 'bar', 'pie', 'likert'.")
        self.plot_type = plot_type
        self.title = title
        self.per_context = per_context

    def plot_bar(self, possibilities, counts):
        """
        Produces a bar plot.

        Parameters
        ----------
        possibilities: list
            List of possible categories.
        counts: list
            List of counts corresponding to each category.
        order_by_count: bool
            Whether to order the possibilities by count.

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot as a Matplotlib figure.
        """
        data = pd.DataFrame({"Possibilities": possibilities, "Counts": counts})
        if self.order_by_count:
            data = data.sort_values(by="Counts", ascending=False)

        fig, ax = plt.subplots()
        plt.close(fig)
        sns.barplot(x="Possibilities", y="Counts", data=data, ax=ax)
        ax.set_title("Bar Plot of Ordinal Categories")
        return fig

    def plot_pie(self, possibilities, counts):
        """
        Produces a pie chart.

        Parameters
        ----------
        possibilities: list
            List of possible categories.
        counts: list
            List of counts corresponding to each category.
        order_by_count: bool
            Whether to order the possibilities by count.

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot as a Matplotlib figure.
        """
        data = pd.DataFrame({"Possibilities": possibilities, "Counts": counts})
        if self.order_by_count:
            data = data.sort_values(by="Counts", ascending=False)

        fig, ax = plt.subplots()
        plt.close(fig)
        ax.pie(data["Counts"], labels=data["Possibilities"], autopct="%1.1f%%")
        ax.set_title("Pie Chart of Ordinal Categories")
        return fig

    def plot_likerts(self, possibilities, counts):
        """
        Produces a Likert plot.

        Parameters
        ----------
        possibilities: list
            List of possible categories.
        counts: list
            List of counts corresponding to each category.
        order_by_count: bool
            Whether to order the possibilities by count.

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot as a Matplotlib figure.
        """
        # if self.order_by_count:
        #     data = data.sort_values(by="Counts", ascending=False)
        # Suppress FutureWarning messages from plot_likert
        warnings.filterwarnings("ignore", category=FutureWarning, module="plot_likert")
        sg = Seismogram()
        data = sg.dataframe
        plot_scale = possibilities
        fig, ax = plt.subplots()
        plt.close(fig)
        plot_likert.plot_likert(data[self.target_cols], plot_scale=plot_scale, plot_percentage=True, ax=ax)
        ax.set_title("Likert Plot of Survey Responses")
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
        """
        Generates the corresponding plot for each type: bar, pie, likert.

        Parameters
        ----------
        possibilities: list
            List of possible categories.
        counts: list
            List of counts corresponding to each category.
        plot_type: str
            Type of plot to generate ('bar', 'pie', 'likert').
        order_by_count: bool
            Whether to order the possibilities by count.

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot as a Matplotlib figure.

        Raises
        ------
        ValueError
            If an invalid plot_type is provided.
        """
        plots = []
        sg = Seismogram()
        for col in self.target_cols:
            value_counts = sg.dataframe[col].value_counts()
            possibilities = value_counts.index.tolist()
            counts = value_counts.tolist()

            if self.plot_type == "bar":
                plots.append(self.plot_bar(possibilities, counts))
            elif self.plot_type == "pie":
                plots.append(self.plot_pie(possibilities, counts))
            elif self.plot_type == "likert":
                plots.append(self.plot_likerts(possibilities, counts))
        return plots


# region Analytics Table Wrapper


def ordinal_categorical_plot(
    target_cols: list[str],
    plot_type,
    order_by_count,
    *,
    title: str = None,
    per_context: bool = False,
) -> HTML:
    """
    Binary fairness metrics table

    Parameters
    ----------
    target_cols : list[str]
        A list of column names corresponding to a subset of (binary) targets in sg.target_cols.
    score_cols : list[str]
        A list of column names corresponding to a subset of scores in sg.output_list.
    metric : str
        Performance metrics will be presented for the provided values of this metric.
    metric_values : list[float]
        Values for the specified metric to derive detailed performance statistics.
    metrics_to_display : list[str]
        List of performance metrics to include in the table.
    group_by : str
        The primary grouping category in the performance table. It could be "Score" or "Target".
    title : str, optional
        The title for the performance statistics table, by default None.
    per_context : bool, optional
        If scores should be grouped by context, by default False.

    Returns
    -------
    HTML
        The HTML table for the fairness evaluation.
    """
    plot = OrdinalCategoricalPlot(target_cols, plot_type, order_by_count, title=title, per_context=per_context)
    return plot.generate_plot()[0]  # TODO: Also support multi plots


# endregion
# region Analytics Table Controls


class ExploreOrdinalCategoricalPlots(ExplorationWidget):
    def __init__(self, title: Optional[str] = None):
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.title = title

        super().__init__(
            title="Model Performance Comparison",
            option_widget=OrdinalCategoricalOptionsWidget(
                sg.get_ordinal_categorical_targets(),
                plot_type="bar",
                order_by_count=False,
                title=title,
            ),
            plot_function=ordinal_categorical_plot,
            initial_plot=False,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the analytics table."""
        args = (
            self.option_widget.target_cols
            if isinstance(self.option_widget.target_cols, list)
            else [self.option_widget.target_cols],  # Updated to use target_columns
            self.option_widget.plot_type,  # Updated to use plot_types
            self.option_widget.order_by_count,  # Updated to use order_by_count
        )
        kwargs = {"title": self.title, "per_context": self.option_widget.group_scores}
        return args, kwargs


class OrdinalCategoricalOptionsWidget(Box, traitlets.HasTraits):
    value = traitlets.Dict(help="The selected values for the ordinal categorical options.")

    def __init__(
        self,
        target_cols: tuple[str],
        plot_type,
        order_by_count,
        *,
        model_options_widget=None,
        title: str = None,
    ):
        """
        Widget for selecting analytics table options

        Parameters
        ----------
        target_cols : tuple[str]
            Available (binary) target columns.
        score_cols : tuple[str]
            Available score columns.
        metric : str
            Available metrics. Performance metrics will be presented for the provided values of this metric.
        model_options_widget : Optional[widget]
            Additional model options widget, by default None.
        metric_values : Optional[list[float]]
            Metric values for which performance metric will be computed, by default None.
        metrics_to_display : Optional[tuple[str]]
            Metrics to include in the analytics table, by default None.
        title : Optional[str]
            Title of the widget, by default None.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.model_options_widget = model_options_widget
        self.title = title

        # Multiple select dropdowns for targets and scores
        self._target_cols = Dropdown(
            options=target_cols or sg.get_ordinal_categorical_targets(),
            value=sg.get_ordinal_categorical_targets()[0],
            description="Target",
            style=WIDE_LABEL_STYLE,
        )
        self._plot_type = Dropdown(
            options=ORDINAL_CATEGORICAL_PLOT_TYPES,
            value=plot_type,
            description="Plot type",
            style=WIDE_LABEL_STYLE,
        )
        self._order_by_count = Checkbox(
            value=order_by_count,
            description="Order by count?",
            disabled=False,
            tooltip="Order the counts",
            style=WIDE_LABEL_STYLE,
        )
        self.per_context_checkbox = _combine_scores_checkbox(per_context=False)

        self._target_cols.observe(self._on_value_changed, names="value")
        self._plot_type.observe(self._on_value_changed, names="value")
        self._order_by_count.observe(self._on_value_changed, names="value")
        self.per_context_checkbox.observe(self._on_value_changed, names="value")

        v_children = [
            self._target_cols,
            self._plot_type,
            self._order_by_count,
            self.per_context_checkbox,
        ]
        if model_options_widget:
            v_children.insert(0, model_options_widget)
            self.model_options_widget.observe(self._on_value_changed, names="value")

        grid_layout = Layout(
            width="80%", grid_template_columns="repeat(3, 1fr)", justify_items="flex-end", grid_gap="10px"
        )  # Three columns

        # Create a GridBox with the specified layout
        grid_box = GridBox(children=v_children, layout=grid_layout)

        super().__init__(
            children=[grid_box],
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
        self._target_cols.disabled = value
        self._plot_type.disabled = value
        self._order_by_count.disabled = value
        self.per_context_checkbox.disabled = value
        if self.model_options_widget:
            self.model_options_widget.disabled = value

    def _on_value_changed(self, change=None):
        new_value = {
            "target_cols": self._target_cols.value,
            "plot_type": self._plot_type.value,
            "order_by_count": self._order_by_count.value,
            "group_scores": self.per_context_checkbox.value,
        }
        if self.model_options_widget:
            new_value["model_options"] = self.model_options_widget.value
        self.value = new_value

    @property
    def target_cols(self):
        return self._target_cols.value

    @property
    def plot_type(self):
        return self._plot_type.value

    @property
    def order_by_count(self):
        return self._order_by_count.value

    @property
    def group_scores(self):
        return self.per_context_checkbox.value


# endregion
