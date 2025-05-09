import logging
from typing import Optional, Union

import pandas as pd

# import ipywidgets as widgets
import traitlets
from IPython.display import HTML
from ipywidgets import Box, Layout, ValueWidget, VBox

from seismometer.controls.decorators import disk_cached_html_segment
from seismometer.controls.explore import ExplorationWidget
from seismometer.controls.selection import MultiselectDropdownWidget, MultiSelectionListWidget
from seismometer.controls.styles import BOX_GRID_LAYOUT, html_title
from seismometer.data.filter import FilterRule
from seismometer.html import template
from seismometer.plot.mpl._ux import MAX_CATEGORY_SIZE
from seismometer.plot.mpl.likert import likert_plot
from seismometer.seismogram import Seismogram

logger = logging.getLogger("seismometer")

""" The maximum number of categories allowed in a categorical column. """


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
        from seismometer.seismogram import Seismogram

        self.metrics = metrics
        self.plot_type = plot_type
        self.title = title
        self.plot_functions = self.initialize_plot_functions()

        sg = Seismogram()
        cohort_filter = FilterRule.from_cohort_dictionary(cohort_dict)
        self.dataframe = cohort_filter.filter(sg.dataframe)
        self.censor_threshold = sg.censor_threshold

        self.values = self._extract_metric_values() if self.metrics else []

    def _extract_metric_values(self):
        """
        Extracts the ordered set of values from all selected metrics.

        Raises
        ------
        ValueError
            If values are inconsistent or cannot be merged safely.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        value_lists = []

        for metric in self.metrics:
            if metric not in sg.metrics:
                raise ValueError(f"Metric {metric} is not a valid metric.")

            values = sg.metrics[metric].metric_details.values
            if values is None:
                raise ValueError(
                    f"Metric values for metric {metric} are not provided. Please update "
                    + "metric details in usage_config with expected metric values."
                )

            value_lists.append(list(values))

        first = value_lists[0]
        for other in value_lists[1:]:
            if first != other:
                raise ValueError("Inconsistent metric values provided across selected metrics.")

        if len(first) > MAX_CATEGORY_SIZE:
            raise ValueError(f"Total number of values ({len(first)}) exceeds MAX_CATEGORY_SIZE ({MAX_CATEGORY_SIZE}).")

        return first

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
        Optional[SVG]
            The SVG object corresponding to the generated Likert plot.
        """
        df = self._count_values_in_columns()
        return likert_plot(df=df) if not df.empty else None

    def _count_values_in_columns(self):
        """
        Counts occurrences of each unique value in each metric column.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the counts of each unique value in each metric column.
        """
        sg = Seismogram()
        # Create a dictionary to store the counts
        data = {"Feedback Metrics": [sg.metrics[metric].display_name for metric in self.metrics]}

        # Count occurrences of each unique value in each metric column
        col_counts = {col: self.dataframe[col].value_counts() for col in self.metrics}
        for value in self.values:
            data[value] = [col_counts[col].get(value, 0) for col in self.metrics]

        # Create a new DataFrame from the dictionary and set "Feedback Metrics" as index
        counts_df = pd.DataFrame(data)
        counts_df.set_index("Feedback Metrics", inplace=True)
        counts_df = counts_df[counts_df.sum(axis=1) >= self.censor_threshold]

        return counts_df

    def generate_plot(self):
        """
        Generates the plot based on the specified plot type.

        Returns
        -------
        HTML
            The HTML object representing the generated plot figure.

        Raises
        ------
        ValueError
            If the specified plot type is unknown.
        """
        if self.plot_type not in self.plot_functions:
            raise ValueError(f"Unknown plot type: {self.plot_type}")
        if len(self.dataframe) < self.censor_threshold:
            return template.render_censored_plot_message(self.censor_threshold)
        svg = self.plot_functions[self.plot_type](self)
        return (
            template.render_title_with_image(self.title, svg)
            if svg is not None
            else template.render_censored_plot_message(self.censor_threshold)
        )


# region Plots Wrapper


@disk_cached_html_segment
def ordinal_categorical_plot(
    metrics: list[str],
    cohort_dict: dict[str, tuple],
    *,
    title: Optional[str] = None,
) -> HTML:
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
    HTML
        HTML object corresponding to the figure generated by the plot.
    """
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
    def __init__(self, group_key: Optional[str] = None, title: Optional[str] = None):
        """
        Initializes the ExploreCategoricalPlots class.

        Parameters
        ----------
        title : str, optional
            Title of the plot, by default None.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.title = title

        super().__init__(
            title="Plot Metrics",
            option_widget=CategoricalOptionsWidget(
                group_key or sg.get_ordinal_categorical_groups(MAX_CATEGORY_SIZE),
                cohort_dict=sg.available_cohort_groups,
                title=title,
            ),
            plot_function=ordinal_categorical_plot,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the ordinal categorical plot."""
        args = (
            list(self.option_widget.metrics),
            self.option_widget.cohort_dict,
        )
        kwargs = {"title": self.option_widget.title}
        return args, kwargs


class CategoricalOptionsWidget(Box, ValueWidget, traitlets.HasTraits):
    value = traitlets.Dict(help="The selected values for the ordinal categorical options.")

    def __init__(
        self,
        metric_groups: Union[str, list[str]],
        cohort_dict: dict[str, list[str]],
        *,
        model_options_widget=None,
        title: str = None,
    ):
        """
        Initializes the CategoricalOptionsWidget class.

        Parameters
        ----------
        metric_groups : Union[str,list[str]]
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
        metric_groups = metric_groups or sg.get_ordinal_categorical_groups(MAX_CATEGORY_SIZE)
        self.model_options_widget = model_options_widget
        self.title = title
        self.include_groups = isinstance(metric_groups, list)
        if not self.include_groups:
            self.metric_group = metric_groups
            metric_groups = [metric_groups]

        self.metric_display_name_to_source = {
            sg.metrics[metric].display_name: metric
            for group in metric_groups
            for metric in sg.metric_groups[group]
            if metric in sg.get_ordinal_categorical_metrics(MAX_CATEGORY_SIZE)
        }
        self.all_metrics = set(self.metric_display_name_to_source.keys())
        all_metrics_list = sorted(list(self.all_metrics))

        self._metrics = MultiselectDropdownWidget(
            options=all_metrics_list,
            value=all_metrics_list,
            title="Metrics",
        )

        self._cohort_dict = MultiSelectionListWidget(cohort_dict, title="Cohorts")
        v_children = [html_title("Plot Options")]

        if self.include_groups:
            self._metric_groups = MultiselectDropdownWidget(
                options=metric_groups,
                value=metric_groups,
                title="Metric Groups",
            )
            v_children.append(self._metric_groups)
            self._metric_groups.observe(self._on_value_changed, names="value")
        self._metrics.observe(self._on_value_changed, names="value")
        self._cohort_dict.observe(self._on_value_changed, names="value")

        v_children.append(self._metrics)
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
        self._metrics.disabled = value
        self._cohort_dict.disabled = value
        if self.model_options_widget:
            self.model_options_widget.disabled = value
        if self.include_groups:
            self._metric_groups.disabled = value

    def _update_disabled_state(self):
        self._metrics.disabled = len(self._metrics.options) == 0

    def _on_value_changed(self, change=None):
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        new_value = {}
        if self.include_groups:
            metric_groups = self._metric_groups.value
            metrics_set = set(
                sg.metrics[metric].display_name
                for metric_group in metric_groups
                for metric in sg.metric_groups[metric_group]
            )
            metrics_set = metrics_set & self.all_metrics
            self._metrics._update_options(sorted(metrics_set))
            self._metrics.value = sorted(list(set(self._metrics.value) & metrics_set))
        self._update_disabled_state()

        new_value = {
            "metrics": self._metrics.value,
            "cohort_dict": self._cohort_dict.value,
        }
        if self.include_groups:
            new_value["metric_groups"] = self._metric_groups.value
        if self.model_options_widget:
            new_value["model_options"] = self.model_options_widget.value
        self.value = new_value

    @property
    def metric_groups(self):
        return self._metric_groups.value if self.include_groups else self.metric_group

    @property
    def metrics(self):
        return (self.metric_display_name_to_source[metric_name] for metric_name in self._metrics.value)

    @property
    def cohort_dict(self):
        return self._cohort_dict.value


# endregion
