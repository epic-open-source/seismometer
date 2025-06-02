from enum import Enum
from typing import Any, List, Optional

import pandas as pd

# import ipywidgets as widgets
import traitlets
from great_tables import GT, loc, style
from ipywidgets import HTML, Dropdown, GridBox, Layout, VBox
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from seismometer.controls.explore import ExplorationWidget, _combine_scores_checkbox
from seismometer.controls.selection import MultiselectDropdownWidget, MultiSelectionListWidget
from seismometer.controls.styles import BOX_GRID_LAYOUT, html_title
from seismometer.controls.thresholds import MonotonicProbabilitySliderListWidget
from seismometer.data import pandas_helpers as pdh
from seismometer.data.binary_performance import GENERATED_COLUMNS, generate_analytics_data
from seismometer.data.performance import MONOTONIC_METRICS, OVERALL_PERFORMANCE, STATNAMES, THRESHOLD
from seismometer.html import template
from seismometer.seismogram import Seismogram

from .analytics_table_config import COLORING_CONFIG_DEFAULT, AnalyticsTableConfig

# region Analytics Table


class TopLevel(Enum):
    """
    Enumeration for available values for top_level parameter in AnalyticsTable class.
    """

    Score = "score"
    Target = "target"


class AnalyticsTable:
    """
    A class to provide a table that displays overall performance statistics and threshold-specific statistics
    across a list of models (scores) and targets.
    """

    # The validations for some of property setters behave differently on class initialization
    _initializing = True

    _get_second_level = {"Target": "Score", "Score": "Target"}

    def __init__(
        self,
        score_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        *,
        metric: str = "threshold",
        metric_values: List[float] = [0.1, 0.2],
        metrics_to_display: Optional[List[str]] = None,
        title: str = "Model Performance Statistics",
        top_level: str = "Score",
        cohort_dict: Optional[dict[str, tuple]] = None,
        table_config: Optional[AnalyticsTableConfig] = AnalyticsTableConfig(),
        statistics_data: Optional[pd.DataFrame] = None,
        per_context: bool = False,
        censor_threshold: int = 10,
    ):
        """
        Initializes the AnalyticsTable object with the necessary data and parameters.

        Parameters
        ----------
        score_columns: Optional[List[str]], optional
            A list of column names corresponding to model prediction scores, by default None.
        target_columns: Optional[List[str]], optional
            A list of column names corresponding to (binary) targets, by default None.
        metric: str, optional
            Performance metrics will be presented for the provided values of this metric, by default "Threshold".
        metric_values: List[float], optional
            Values for the specified metric to derive detailed performance statistics, by default [0.1, 0.2].
        metrics_to_display: Optional[List[str]], optional
            List of metrics to include in the table, by default None. The default behavior is to show all columns
            in GENERATED_COLUMNS.
        title: str, optional
            The title for the performance statistics table, by default "Model Performance Statistics".
        top_level: str, optional
            The primary grouping category in the performance table, by default 'Score'.
        cohort_dict : Optional[dict[str,tuple]], optional
            dictionary of cohort columns and values used to subselect a population for evaluation, by default None.
        decimals: int, optional
            The number of decimal places for rounding numerical results, by default 3.
        table_config: Optional[AnalyticsTableConfig], optional
            Configuration for the analytics table, including formatting and display options.
        statistics_data: Optional[pd.DataFrame], optional
            Additional performance metrics statistics, will be joined with the statistics data generated
            by the code, by default None.
        per_context : bool, optional
            If scores should be grouped by context, by default False.
        censor_threshold : int, optional
            Minimum number of rows required in the cohort data to enable the generation of an analytics table,
            by default 10.

        Raises
        ------
        ValueError
            If neither "sg.dataframe" nor "statistics_data" has data.
            If "sg.dataframe" has data but either "score_columns" or "target_columns" is not provided.
            If the provided metric name is not recognized.
            If the provided top_level name is not "Score" or "Target".
        """
        sg = Seismogram()
        self.df = sg.dataframe
        self.score_columns = score_columns or sg.output_list
        self.target_columns = target_columns
        if sg.dataframe is not None and sg.target_cols:
            self.target_columns = self.target_columns or sg.get_binary_targets()
        self.statistics_data = statistics_data
        if self.df is None and self.statistics_data is None:
            raise ValueError("At least one of 'df' or 'statistics_data' needs to be provided.")
        if self.df is not None:
            if not self.score_columns or not self.target_columns:
                raise ValueError(
                    "When df is provided, both 'score_columns' and 'target_columns' need to be provided as well."
                )

        self.decimals = table_config.decimals
        self.metric = metric
        self.metric_values = list(set(metric_values))
        self.metrics_to_display = metrics_to_display or GENERATED_COLUMNS
        self.title = title
        self.top_level = top_level
        self.cohort_dict = cohort_dict
        self.columns_show_percentages = table_config.columns_show_percentages

        self.percentages_decimals = table_config.percentages_decimals

        self._initializing = False
        self.per_context = per_context
        self.censor_threshold = censor_threshold

    def _validate_df_statistics_data(self):
        if not self._initializing:  # Skip validation during initial setup
            if self.df is None and self.statistics_data is None:
                raise ValueError("At least one of 'df' or 'statistics_data' needs to be provided.")
            if self.df is not None:
                if not self.score_columns or not self.target_columns:
                    raise ValueError(
                        "When df is provided, both 'score_columns' and 'target_columns' need to be provided as well."
                    )

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        self._df = value
        self._validate_df_statistics_data()

    @property
    def statistics_data(self):
        return self._statistics_data

    @statistics_data.setter
    def statistics_data(self, value):
        self._statistics_data = value
        self._validate_df_statistics_data()

    @property
    def score_columns(self):
        return self._score_columns

    @score_columns.setter
    def score_columns(self, value):
        self._score_columns = value
        self._validate_df_statistics_data()

    @property
    def target_columns(self):
        return self._target_columns

    @target_columns.setter
    def target_columns(self, value):
        self._target_columns = value
        self._validate_df_statistics_data()

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        if value not in MONOTONIC_METRICS + [THRESHOLD]:
            raise ValueError(
                f"Invalid metric name: {value}. The metric needs to be one of: " f"{MONOTONIC_METRICS + [THRESHOLD]}"
            )
        self._metric = value

    @property
    def metrics_to_display(self):
        return self._metrics_to_display

    @metrics_to_display.setter
    def metrics_to_display(self, value):
        for metric in value:
            if metric not in GENERATED_COLUMNS and metric not in [THRESHOLD] + STATNAMES + OVERALL_PERFORMANCE:
                raise ValueError(f"Invalid metric name: {value}. The metric needs to be one of: {GENERATED_COLUMNS}")
        self._metrics_to_display = value if value else GENERATED_COLUMNS

    @property
    def metric_values(self):
        return self._metric_values

    @metric_values.setter
    def metric_values(self, value):
        self._metric_values = sorted([round(num, self.decimals) for num in value]) if self.df is not None else []
        self._metric_values = [0 if val == 0.0 else val for val in self._metric_values]

    @property
    def top_level(self):
        return self._top_level

    @top_level.setter
    def top_level(self, value):
        try:
            self._top_level = TopLevel(value.lower()).name
        except ValueError as e:
            raise ValueError(
                f"Invalid top_level name: {value}. The top_level needs to be one of: "
                f"{[member.value for member in TopLevel]}"
            ) from e

    def generate_initial_table(self, data) -> GT:
        """
        Generates the initial table with formatted headers, stubs, and numeric/percentage formatting.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the data to be displayed in the table.

        Returns
        -------
        gt : GT
            The formatted GT (from great_tables) object.
        """
        gt = GT(data).tab_header(title=self.title)
        gt = (
            gt.tab_stub(rowname_col=self._get_second_level[self.top_level], groupname_col=self.top_level)
            .fmt_number(
                columns=[
                    col
                    for col in data.columns
                    if is_numeric_dtype(data[col].dtype) and not is_integer_dtype(data[col].dtype)
                ],
                decimals=self.decimals,
            )
            .fmt_number(
                columns=[col for col in data.columns if col.endswith(f"_{THRESHOLD}")],
                decimals=self.decimals - 2,
            )
            .fmt_percent(columns=self.columns_show_percentages, decimals=self.percentages_decimals)
            .tab_style(
                style=[
                    style.text(align="center"),
                ],
                locations=loc.title(),
            )
        )
        return gt

    def group_columns_by_metric_value(self, gt, columns, value) -> GT:
        """
        Groups columns by a specified metric value and adds borders.

        Parameters
        ----------
        gt : GT
            The table object to which the column grouping will be added.
        columns : List[str]
            The list of columns to be grouped.
        value : str
            The metric value used for grouping columns.

        Returns
        -------
        gt : GT
            The table object with grouped columns and added borders.
        """
        value = round(value * 100, max(0, self.decimals - 2)) if self.metric == THRESHOLD else value
        gt = gt.tab_spanner(label=f"{self.metric}={value}", columns=columns).cols_label(
            **{col: "_".join(col.split("_")[1:]) for col in columns}
        )
        return self.add_borders(gt, columns[0], columns[-1])

    def add_borders(self, gt, left_column, right_column) -> GT:
        """
        Adds borders to the left of left_column and right of right_column in the table.

        Parameters
        ----------
        gt : GT
            The table object to which the borders will be added.
        left_column : str
            The name of the left column to which the border will be added on the left.
        right_column : str
            The name of the right column to which the border will be added on the right.

        Returns
        -------
        gt : GT
            The table object with added borders.
        """
        gt = gt.tab_style(
            style=style.borders(sides=["left"], weight="1px", color="#D3D3D3"),
            locations=loc.body(columns=[left_column]),
        ).tab_style(
            style=style.borders(sides=["right"], weight="1px", color="#D3D3D3"),
            locations=loc.body(columns=[right_column]),
        )
        return gt

    def analytics_table(self) -> HTML:
        """
        Generates an analytics table based on calculated performance statistics.

        Returns
        -------
        HTML
            An HTML object representing the formatted analytics table.
        """
        data = self._generate_table_data()
        if data is None:
            return template.render_censored_plot_message(self.censor_threshold)

        gt = self.generate_initial_table(data)

        # Group columns of the form value_*** together
        for value in self.metric_values:
            columns = [column for column in data.columns if column.startswith(f"{value}_")]
            if columns:
                gt = self.group_columns_by_metric_value(gt, columns, value)

        gt = (
            gt.opt_horizontal_padding(scale=3)
            .tab_options(row_group_font_weight="bold")
            .tab_style(
                style=[
                    style.text(align="center"),
                ],
                locations=loc.column_header(),
            )
            .tab_style(
                style=[
                    style.text(align="center"),
                ],
                locations=loc.body(),
            )
        )

        return HTML(gt.as_raw_html(), layout=Layout(max_height="800px"))

    def _generate_table_data(self) -> Optional[pd.DataFrame]:
        """
        Generates a DataFrame containing calculated statistics for each combination of scores and targets.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the calculated statistics for each combination of scores and targets.
            This data will be used to generate the GT (from great_tables) object.
        """
        data = generate_analytics_data(
            self.score_columns,
            self.target_columns,
            self.metric,
            self.metric_values,
            top_level=self.top_level,
            cohort_dict=self.cohort_dict,
            per_context=self.per_context,
            metrics_to_display=self.metrics_to_display,
            decimals=self.decimals,
            censor_threshold=self.censor_threshold,
        )
        if data is None:
            return None

        # Update target columns name to drop possible _Value at the end
        data["Target"] = data["Target"].apply(pdh.event_name)

        # Add statistics_data if provided.
        if self.statistics_data is not None:
            self.statistics_data["Score"] = pd.Categorical(
                self.statistics_data["Score"], categories=self.score_columns, ordered=True
            )
            self.statistics_data["Target"] = pd.Categorical(
                self.statistics_data["Target"], categories=self.target_columns, ordered=True
            )
            self.statistics_data = self.statistics_data.sort_values(
                by=[self.top_level, self._get_second_level[self.top_level]]
            )
            data = (
                pd.merge(data, self.statistics_data, how="inner", on=["Score", "Target"])
                if data is not None
                else self.statistics_data
            )

        return data


# endregion
# region Analytics Table Wrapper


def binary_analytics_table(
    target_cols: list[str],
    score_cols: list[str],
    metric: str,
    metric_values: list[float],
    metrics_to_display: list[str],
    group_by: str,
    cohort_dict: dict[str, tuple[Any]],
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
    cohort_dict : dict[str, tuple[Any]]
        dictionary of cohort columns and values used to subselect a population for evaluation.
    title : str, optional
        The title for the performance statistics table, by default None.
    per_context : bool, optional
        If scores should be grouped by context, by default False.

    Returns
    -------
    HTML
        The HTML table representing the corresponding analytics table.
    """
    sg = Seismogram()
    table_config = AnalyticsTableConfig(**COLORING_CONFIG_DEFAULT)
    performance_metrics = AnalyticsTable(
        score_columns=score_cols,
        target_columns=target_cols,
        metric=metric,
        metric_values=metric_values,
        metrics_to_display=metrics_to_display,
        title=title or "Model Performance Statistics",
        top_level=group_by,
        cohort_dict=cohort_dict,
        table_config=table_config,
        per_context=per_context,
        censor_threshold=sg.censor_threshold,
    )
    return performance_metrics.analytics_table()


# endregion
# region Analytics Table Controls


class ExploreBinaryModelAnalytics(ExplorationWidget):
    def __init__(self, title: Optional[str] = None):
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.title = title

        super().__init__(
            title="Model Performance Comparison",
            option_widget=AnalyticsTableOptionsWidget(
                tuple(map(pdh.event_name, sg.get_binary_targets())),
                sg.output_list,
                metric=THRESHOLD,
                metric_values=None,
                metrics_to_display=None,
                cohort_dict=sg.available_cohort_groups,
                title=title,
            ),
            plot_function=binary_analytics_table,
            initial_plot=False,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the analytics table."""
        args = (
            tuple(map(pdh.event_value, self.option_widget.target_cols)),  # Updated to use target_cols
            self.option_widget.score_cols,  # Updated to use score_cols
            self.option_widget.metric,  # Updated to use metric
            self.option_widget.metric_values,  # Updated to use metric_values
            list(self.option_widget.metrics_to_display),  # Updated to use metrics_to_display
            self.option_widget.group_by,  # Updated to use group_by
            self.option_widget.cohort_dict,
        )
        kwargs = {"title": self.title, "per_context": self.option_widget.group_scores}
        return args, kwargs


class AnalyticsTableOptionsWidget(VBox, traitlets.HasTraits):
    value = traitlets.Dict(help="The selected values for the analytics table options.")

    def __init__(
        self,
        target_cols: tuple[str],
        score_cols: tuple[str],
        metric: str,
        *,
        model_options_widget=None,
        metric_values=None,
        metrics_to_display: Optional[tuple[str]] = None,
        cohort_dict: Optional[dict[str, tuple[Any]]] = None,
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
        cohort_dict : Optional[dict[str, tuple[Any]]]
            dictionary of cohort columns and values used to subselect a population for evaluation, by default None.
        title : Optional[str]
            Title of the widget, by default None.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.model_options_widget = model_options_widget
        self.title = title

        # Multiple select dropdowns for targets and scores
        self._target_cols = MultiselectDropdownWidget(
            tuple(map(pdh.event_name, sg.get_binary_targets())),
            value=target_cols or tuple(map(pdh.event_name, sg.get_binary_targets())),
            title="Targets",
        )
        self._score_cols = MultiselectDropdownWidget(
            sg.output_list,
            value=score_cols,
            title="Scores",
        )
        self._metric = Dropdown(
            options=[THRESHOLD] + MONOTONIC_METRICS,
            value=metric,
            description="Metric",
            style={"description_width": "min-content"},
            layout=Layout(width="calc(max(max-content, var(--jp-widgets-inline-width-short)))", min_width="200px"),
        )
        self._metrics_to_display = MultiselectDropdownWidget(
            options=[THRESHOLD] + STATNAMES + OVERALL_PERFORMANCE,
            value=metrics_to_display or GENERATED_COLUMNS,
            title="Performance Metrics to Display",
        )
        metric_values = metric_values or (0.8, 0.2)
        self._metric_values = MonotonicProbabilitySliderListWidget(
            names=("Metric Value 1", "Metric Value 2"),
            value=tuple(metric_values),
            ascending=False,
            decimals=AnalyticsTableConfig().decimals,
        )
        for slider in self._metric_values.sliders.values():
            slider.style.description_width = "min-content"
        self._group_by = Dropdown(
            options=["Score", "Target"],
            value="Score",
            description="Group By",
            style={"description_width": "min-content"},
            layout=Layout(width="calc(max(max-content, var(--jp-widgets-inline-width-short)))", min_width="200px"),
        )
        self._cohort_dict = MultiSelectionListWidget(cohort_dict or sg.available_cohort_groups, title="Cohort Filter")
        self.per_context_checkbox = _combine_scores_checkbox(per_context=False)

        self._target_cols.observe(self._on_value_changed, names="value")
        self._score_cols.observe(self._on_value_changed, names="value")
        self._metric.observe(self._on_value_changed, names="value")
        self._metric_values.observe(self._on_value_changed, "value")
        self._metrics_to_display.observe(self._on_value_changed, names="value")
        self._group_by.observe(self._on_value_changed, names="value")
        self.per_context_checkbox.observe(self._on_value_changed, names="value")
        self._cohort_dict.observe(self._on_value_changed, names="value")

        v_children = [
            self._target_cols,
            self._score_cols,
            self._metrics_to_display,
            self._metric,
            self._metric_values,
            self._group_by,
            self.per_context_checkbox,
        ]
        if model_options_widget:
            v_children.insert(0, model_options_widget)
            self.model_options_widget.observe(self._on_value_changed, names="value")

        grid_layout = Layout(
            width="100%", grid_template_columns="repeat(3, 1fr)", justify_items="flex-start", grid_gap="10px"
        )  # Three columns

        # Create a GridBox with the specified layout
        grid_box = GridBox(children=v_children, layout=grid_layout)

        # Add title
        grid_with_title = VBox(
            children=[
                html_title("Analytics Table Options"),
                grid_box,
            ],
            layout=Layout(
                align_items="flex-start",
            ),
        )

        super().__init__(
            children=[self._cohort_dict, grid_with_title],
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
        self._score_cols.disabled = value
        self._metric.disabled = value
        self._metric_values.disabled = value
        self._metrics_to_display.disabled = value
        self._group_by.disabled = value
        self._cohort_dict.disabled = value
        self.per_context_checkbox.disabled = value
        if self.model_options_widget:
            self.model_options_widget.disabled = value

    def _on_value_changed(self, change=None):
        new_value = {
            "target_cols": self._target_cols.value,
            "score_cols": self._score_cols.value,
            "metric": self._metric.value,
            "metric_values": self._metric_values.value,
            "metrics_to_display": self._metrics_to_display.value,
            "group_by": self._group_by.value,
            "cohort_dict": self._cohort_dict.value,
            "group_scores": self.per_context_checkbox.value,
        }
        if self.model_options_widget:
            new_value["model_options"] = self.model_options_widget.value
        self.value = new_value

    @property
    def target_cols(self):
        return self._target_cols.value

    @property
    def score_cols(self):
        return self._score_cols.value

    @property
    def metric(self):
        return self._metric.value

    @property
    def metric_values(self):
        return tuple(self._metric_values.value.values())

    @property
    def metrics_to_display(self):
        return self._metrics_to_display.value

    @property
    def group_by(self):
        return self._group_by.value

    @property
    def cohort_dict(self):
        return self._cohort_dict.value

    @property
    def group_scores(self):
        return self.per_context_checkbox.value


# endregion
