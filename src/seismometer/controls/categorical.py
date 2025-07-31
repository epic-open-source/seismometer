import logging
from collections import defaultdict
from typing import List, Optional, Union

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

        self.values = self._extract_metric_values()

    def _extract_metric_values(self):
        """
        Extracts the ordered set of values from all selected metrics.

        For each metric, uses `metric.metric_details.values` if available;
        otherwise, raises an error — order must be explicitly defined.

        Combines value lists if they are consistent and impose a unique order.
        Raises a ValueError if a merge is ambiguous or inconsistent.

        Raises
        ------
        ValueError
            If values are inconsistent or cannot be merged safely.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        metric_to_values = {}

        for metric in self.metrics:
            if metric not in sg.metrics:
                raise ValueError(f"Metric {metric} is not a valid metric.")

            values = sg.metrics[metric].metric_details.values
            if values is None:
                raise ValueError(
                    f"Metric values for metric {metric} are not provided. Please update "
                    + "metric details in usage_config with expected metric values."
                )

            metric_to_values[metric] = list(values)

        try:
            result = self._merge_ordered_lists(metric_to_values=metric_to_values)
        except ValueError as e:
            debug_info = "\n".join(f"{metric}: {metric_to_values[metric]}" for metric in metric_to_values)
            raise ValueError(f"{str(e)}\n\nThe following metric value orders were provided:\n{debug_info}") from e

        if len(result) > MAX_CATEGORY_SIZE:
            raise ValueError(
                f"Total number of values ({len(result)}) exceeds MAX_CATEGORY_SIZE ({MAX_CATEGORY_SIZE})."
            )

        return result

    def _merge_ordered_lists(self, metric_to_values: dict[str, List[str]]) -> List[str]:
        """
        Merges multiple ordered lists into a single linear order.

        Only succeeds if there is a unique valid next element at every step,
        based on the in-degree of the graph built from pairwise relationships.

        Parameters
        ----------
        metric_to_values : dict[str, list[str]]
            Mapping from metric names to their user-defined ordered value lists.

        Returns
        -------
        list of str
            Merged, globally consistent (unique) linear order.

        Raises
        ------
        ValueError
            If there are conflicting or ambiguous orderings.
        """
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        all_nodes = set()

        # Step 1: Build directed graph from the order of each list
        for metric, lst in metric_to_values.items():
            all_nodes.update(lst)
            for a, b in zip(lst, lst[1:]):
                if b not in graph[a]:
                    graph[a].add(b)
                    in_degree[b] += 1
        result = []

        # Step 2: Find total ordering by returning a sequence of directed graph sources (the least element in
        #         each ordered graph)
        while len(result) < len(all_nodes):
            candidates = [n for n in all_nodes if in_degree[n] == 0 and n not in result]

            if len(candidates) != 1:
                self._handle_ordering_error(candidates, graph, metric_to_values, result)

            node = candidates[0]
            result.append(node)

            # remove the node from the graph
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
            graph.pop(node, None)
            in_degree.pop(node, None)

        return result

    def _handle_ordering_error(
        self,
        candidates: list[str],
        graph: dict[str, set[str]],
        metric_to_values: dict[str, list[str]],
        partial_order: list[str],
    ) -> None:
        """
        Dispatches error handling based on the number of valid candidates during
        topological ordering. Delegates to either cycle or ambiguity error reporting.

        Parameters
        ----------
        candidates : list[str]
            The list of current candidate values (with in-degree 0 and not yet processed).
        graph : dict[str, set[str]]
            A directed graph representing value orderings.
        metric_to_values : dict[str, list[str]]
            The original ordered lists of metric values.
        partial_order : list[str]
            The current partially constructed global order.

        Raises
        ------
        ValueError
            With a detailed explanation of the inconsistency or ambiguity in ordering.
        """
        if len(candidates) == 0:
            self._report_cycle_error(graph, metric_to_values)
        else:
            self._report_ambiguous_error(candidates, metric_to_values, partial_order)

    def _report_cycle_error(self, graph: dict[str, set[str]], metric_to_values: dict[str, list[str]]) -> None:
        """
        Finds and reports a cycle in the ordering graph, including source metrics
        that contributed to each conflicting edge in the cycle.

        Parameters
        ----------
        graph : dict[str, set[str]]
            A directed graph representing value orderings.
        metric_to_values : dict[str, list[str]]
            The original metric value orderings from which the graph was built.

        Raises
        ------
        ValueError
            If a cycle is found, including an annotated path showing which metrics
            contributed to each edge in the cycle.
        """
        cycle = self._find_any_cycle(graph)
        if not cycle:
            raise ValueError("Inconsistent ordering: a cycle is expected but could not be reconstructed.")

        edge_to_metrics = defaultdict(list)
        for metric, lst in metric_to_values.items():
            for a, b in zip(lst, lst[1:]):
                edge_to_metrics[(a, b)].append(metric)

        cycle_lines = []
        for i in range(len(cycle) - 1):
            a, b = cycle[i], cycle[i + 1]
            metrics = ", ".join(sorted(edge_to_metrics.get((a, b), [])))
            cycle_lines.append(f" - {a} → {b}   [from: {metrics}]")

        cycle_msg = "\n".join(cycle_lines)

        raise ValueError(
            f"Inconsistent ordering: a cycle was detected in the value ordering graph.\n" f"Cycle path:\n{cycle_msg}"
        )

    def _report_ambiguous_error(
        self,
        candidates: list[str],
        metric_to_values: dict[str, list[str]],
        partial_order: list[str],
    ) -> None:
        """
        Reports an ambiguous ordering situation in which multiple candidates are valid next elements
        given the current partial_order. Annotates each candidate with the metrics where it appears.

        Parameters
        ----------
        candidates : list[str]
            The list of conflicting candidate values with in-degree 0.
        metric_to_values : dict[str, list[str]]
            The original metric value orderings.
        partial_order : list[str]
            The current partially constructed global order.

        Raises
        ------
        ValueError
            Including each ambiguous candidate and the source metrics that contributed to it.
        """
        candidates_metrics = {
            candidate: [metric for metric in metric_to_values if candidate in metric_to_values[metric]]
            for candidate in candidates
        }
        candidates_metrics = {candidate: ", ".join(metrics) for candidate, metrics in candidates_metrics.items()}
        candidate_lines = "\n".join(
            f" - {candidate}: {metrics}" for candidate, metrics in sorted(candidates_metrics.items())
        )

        raise ValueError(
            f"Ambiguous ordering: multiple values could be the next in sequence {partial_order}.\n"
            f"Conflicting candidates and their metric sources:\n{candidate_lines}"
        )

    def _find_any_cycle(self, graph: dict[str, set[str]]) -> list[str]:
        """
        Finds and returns one cycle in the given directed graph.

        Parameters
        ----------
        graph : dict[str, set[str]]
            A directed graph represented as an adjacency list.

        Returns
        -------
        list[str]
            A list of nodes forming a cycle, with the start node repeated at the end to close the loop.
            Returns an empty list if no cycle is found (should not happen if a cycle is guaranteed).
        """
        visited = set()

        def dfs(node: str, path: list[str]) -> Optional[list[str]]:
            """Performs depth-first search to find and return a cycle path starting from the given node."""
            if node in path:
                idx = path.index(node)
                return path[idx:] + [node]

            if node in visited:
                return None

            visited.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                result = dfs(neighbor, path)
                if result:
                    return result

            path.pop()
            return None

        for node in graph:
            result = dfs(node, [])
            if result:
                return result

        return []

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
        missing = [
            value
            for value in self.values
            if all(value not in sg.metrics[metric].metric_details.values for metric in self.metrics)
        ]
        if missing:
            logger.warning(f"The following metric values are missing from all the metrics: {missing}")
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
