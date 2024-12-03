import itertools
from enum import Enum
from typing import Any, List, Optional

import pandas as pd

# import ipywidgets as widgets
import traitlets
from great_tables import GT, loc, nanoplot_options, style
from ipywidgets import HTML, Box, Dropdown, FloatRangeSlider, GridBox, Layout
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from seismometer.controls.explore import ExplorationWidget
from seismometer.controls.selection import MultiselectDropdownWidget
from seismometer.controls.styles import BOX_GRID_LAYOUT, WIDE_LABEL_STYLE
from seismometer.data import pandas_helpers as pdh
from seismometer.data.binary_performance import GENERATED_COLUMNS, Metric, calculate_stats, is_binary_array
from seismometer.data.performance import (  # MetricGenerator,
    OVERALL_PERFORMANCE,
    STATNAMES,
    THRESHOLD,
    BinaryClassifierMetricGenerator,
)
from seismometer.seismogram import Seismogram

from .analytics_table_config import COLORING_CONFIG_DEFAULT, AnalyticsTableConfig


class TopLevel(Enum):
    """
    Enumeration for available values for top_level parameter in PerformanceMetrics class.
    """

    Score = "score"
    Target = "target"


class PerformanceMetrics:
    """
    A class to calculate and provide overall performance statistics and threshold-specific statistics
    across a list of models (scores) and targets.
    """

    # The validations for some of property setters behave differently on class initialization
    _initializing = True

    _get_second_level = {"Target": "Score", "Score": "Target"}

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        score_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        *,
        metric: str = "threshold",
        metric_values: List[float] = [0.1, 0.2],
        metrics_to_display: Optional[List[str]] = None,
        title: str = "Model Performance Statistics",
        top_level: str = "Score",
        table_config: AnalyticsTableConfig = AnalyticsTableConfig(),
        statistics_data: Optional[pd.DataFrame] = None,
        per_context: bool = False,
    ):
        """
        Initializes the PerformanceMetrics object with the necessary data and parameters.

        Parameters
        ----------
        df: Optional[pd.DataFrame]
            The DataFrame containing the model's predictions and targets, by default None.
        score_columns: Optional[List[str]]
            A list of column names corresponding to model prediction scores to be evaluated, by default None.
        target_columns: Optional[List[str]]
            A list of column names corresponding to targets, by default None.
        metric: str
            Performance metrics will be presented for the provided values of this metric.
        metric_values: List[float]
            Values for the specified metric to derive detailed performance statistics, by default [0.1, 0.2].
        metrics_to_display: Optional[List[str]]
            List of metrics to include in the table, by default None. The default behavior is to shows all columns
            in GENERATED_COLUMNS, which is a dictionary mapping metric names to their corresponding column names.
        title: str
            The title for the performance statistics report, by default "Model Performance Statistics".
        top_level: str
            The primary grouping category in the performance report, by default 'Score'.
        decimals: int
            The number of decimal places for rounding numerical results, by default 3.
        columns_show_percentages: Union[str, List[str]]
            Columns that will display values as percentages in the report, by default "Prevalence".
        columns_show_bar: Optional[Dict[str, str]]
            Columns mapped to specific colors for bar representation in the report, by default None.
        percentages_decimals: int
            The number of decimal places for percentage values in the report, by default 0.
        statistics_data: Optional[pd.DataFrame]
            Additional performance metrics statistics, will be joined with the statistics data generated
            by the code, by default None.

        Raises
        ------
        ValueError
            If any of the provided spanner colors are not recognized as valid color names.
            If neither "df" nor "statistics_data" is provided.
            If "df" is provided but either "score_columns" or "target_columns" is not provided.
            If the provided metric name is not recognized.
            If the provided top_level name is not "Score" or "Target".
        """
        sg = Seismogram()
        self.df = df if df is not None else sg.dataframe
        self.score_columns = score_columns if score_columns else sg.output_list
        self.target_columns = target_columns
        if sg.dataframe is not None:
            self.target_columns = (
                self.target_columns
                if self.target_columns
                else [
                    pdh.event_value(target_col)
                    for target_col in sg.target_cols
                    if is_binary_array(sg.dataframe[[pdh.event_value(target_col)]])
                ]
            )
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
        self.metric_values = metric_values
        self.metrics_to_display = metrics_to_display if metrics_to_display else list(GENERATED_COLUMNS.keys())
        self.title = title
        self.top_level = top_level
        self.columns_show_percentages = table_config.columns_show_percentages
        self.columns_show_bar = table_config.columns_show_bar

        self.percentages_decimals = table_config.percentages_decimals
        self.data_bar_stroke_width = table_config.data_bar_stroke_width

        self._initializing = False
        self.rows_group_length = len(self.target_columns) if self.top_level == "Score" else len(self.score_columns)
        self.num_of_rows = len(self.score_columns) * len(self.target_columns)
        self.per_context = per_context

        # If polars package is not installed, overwrite is_na function in great_tables package to treat Agnostic
        # as pandas dataframe.
        try:
            import polars as pl

            # Use 'pl' to avoid the F401 error
            _ = pl.DataFrame()
        except ImportError:
            from great_tables._tbl_data import Agnostic, PdDataFrame, is_na

            @is_na.register(Agnostic)
            def _(df: PdDataFrame, x: Any) -> bool:
                return pd.isna(x)

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
        try:
            self._metric = Metric(value.lower()).name
        except ValueError as e:
            raise ValueError(
                f"Invalid metric name: {value}. The metric needs to be one of: "
                f"{[member.value for member in Metric]}"
            ) from e

    @property
    def metrics_to_display(self):
        return self._metrcis_to_display

    @metrics_to_display.setter
    def metrics_to_display(self, value):
        for metric in value:
            if metric.lower() not in GENERATED_COLUMNS and metric not in [THRESHOLD] + STATNAMES + OVERALL_PERFORMANCE:
                raise ValueError(
                    f"Invalid metric name: {value}. The metric needs to be one of: {GENERATED_COLUMNS.keys()}"
                )
        self._metrcis_to_display = value if value else list(GENERATED_COLUMNS.keys())

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

    def generate_initial_table(self, data):
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
            .fmt_percent(columns=self.columns_show_percentages, decimals=self.percentages_decimals)
        )
        return gt

    def generate_color_bar(self, gt, columns):
        """
        Adds color bars corresponding to the specified columns in the table.

        Parameters
        ----------
        gt : GT
            The table object to which the color bars will be added.
        columns : List[str]
            The list of columns to which the color bars will be applied.

        Returns
        -------
        gt : GT
            The table object with color bars added.
        """
        for data_col, col in itertools.product(columns, self.columns_show_bar):
            if data_col.endswith(f"_{col}_bar") or data_col == f"{col}_bar":
                gt = gt.fmt_nanoplot(
                    f"{data_col}",
                    plot_type="bar",
                    options=nanoplot_options(
                        data_bar_fill_color=self.columns_show_bar[col],
                        data_bar_stroke_width=self.data_bar_stroke_width,
                    ),
                )
            # If col and col_bar are not grouped under a metric value, group them together.
            if data_col == f"{col}_bar":
                gt = gt.tab_spanner(label=col, columns=[col, f"{col}_bar"])
        return gt

    def group_columns_by_metric_value(self, gt, columns, value):
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
        gt = gt.tab_spanner(label=f"{self.metric}={value}", columns=columns).cols_label(
            **{col: "_".join(col.split("_")[1:]) for col in columns}
        )
        return self.add_borders(gt, columns[0], columns[-1])

    def add_borders(self, gt, left_column, right_column):
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
            style=style.borders(sides=["left"], weight="1px", color="lightgray"),
            locations=loc.body(columns=[left_column]),
        ).tab_style(
            style=style.borders(sides=["right"], weight="1px", color="lightgray"),
            locations=loc.body(columns=[right_column]),
        )
        return gt

    def analytics_table(self):
        """
        Generates an analytics table based on calculated performance statistics.

        Returns
        -------
        GT
            A `GT` (from great_tables package) object representing the formatted analytics table.
        """
        data = self._generate_table_data()
        data = self._prepare_data(data)
        self.rows_group_length = len(self.target_columns) if self.top_level == "Score" else len(self.score_columns)
        self.num_of_rows = len(self.score_columns) * len(self.target_columns)

        gt = self.generate_initial_table(data)

        gt = self.generate_color_bar(gt, columns=data.columns)

        # Group columns of the form value_*** together
        grouped_columns = []
        for value in self.metric_values:
            columns = [column for column in data.columns if column.startswith(f"{value}_")]
            gt = self.group_columns_by_metric_value(gt, columns, value)
            grouped_columns.extend(columns)

        # If a column is not yet grouped and col_bar has been generated, group them together
        for col in self.columns_show_bar:
            if col not in grouped_columns and f"{col}_bar" in data.columns:
                gt = self.add_borders(gt, col, f"{col}_bar")

        gt = gt.opt_horizontal_padding(scale=3).tab_options(row_group_font_weight="bold")

        return HTML(gt.as_raw_html())

    def _prepare_data(self, data):
        """
        Prepares the data for displaying in the analytics table.

        Parameters
        ----------
        data: pd.DataFrame
            The input DataFrame containing relevant statistics data.

        Returns
        -------
        pd.DataFrame
            The modified DataFrame with additional columns for bar plots (if applicable).
        """
        for data_col, col in itertools.product(data.columns, self.columns_show_bar):
            if data_col.endswith(f"_{col}") or data_col == col:
                data.insert(data.columns.get_loc(data_col) + 1, data_col + "_bar", data[data_col])
        return data

    def _generate_table_data(self):
        """
        Generates a DataFrame containing calculated statistics for each combination of scores and targets.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the calculated statistics for each combination of scores and targets.
            This data will be used to generate the GT (from great_tables) object.
        """
        data = None
        if self.df is not None:
            rows_list = []
            product = (
                itertools.product(self.score_columns, self.target_columns)
                if self.top_level == "Score"
                else itertools.product(self.target_columns, self.score_columns)
            )
            for first, second in product:
                current_row = {self.top_level: first, self._get_second_level[self.top_level]: second}
                (score, target) = (first, second) if self.top_level == "Score" else (second, first)
                if self.per_context:
                    sg = Seismogram()
                    data = pdh.event_score(
                        sg.dataframe,
                        sg.entity_keys,
                        score=score,
                        ref_event=sg.predict_time,
                        aggregation_method=sg.event_aggregation_method(target),
                    )
                else:
                    data = self.df
                current_row.update(
                    calculate_stats(
                        data[target],
                        data[score],
                        self.metric,
                        self.metric_values,
                        self.metrics_to_display,
                        self.decimals,
                    )
                )
                rows_list.append(current_row)
            # Create a DataFrame from the rows data
            data = pd.DataFrame(rows_list)

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


def binary_analytics_table(
    target_cols: list[str],
    score_cols: list[str],
    metric: str,
    metric_values: list[float],
    metrics_to_display: list[str],
    group_by: str,
    *,
    title: str = None,
    per_context=False,
) -> HTML:
    """
    Binary fairness metrics table

    Parameters
    ----------


    Returns
    -------
    HTML
        The HTML table for the fairness evaluation.
    """
    from seismometer.seismogram import Seismogram

    sg = Seismogram()
    target_cols = [
        pdh.event_value(target_col)
        for target_col in target_cols
        if is_binary_array(sg.dataframe[[pdh.event_value(target_col)]])
    ]

    table_config = AnalyticsTableConfig(**COLORING_CONFIG_DEFAULT)
    performance_metrics = PerformanceMetrics(
        df=None,
        score_columns=score_cols,
        target_columns=target_cols,
        metric=metric,
        metric_values=metric_values,
        metrics_to_display=metrics_to_display,
        title=title if title else "Model Performance Statistics",
        top_level=group_by,
        table_config=table_config,
        per_context=per_context,
    )
    return performance_metrics.analytics_table()


class ExploreAnalyticsTable(ExplorationWidget):
    def __init__(self, title: Optional[str] = None, *, per_context: bool = False):
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.metric_generator = BinaryClassifierMetricGenerator()
        self.title = title
        self.per_context = per_context

        super().__init__(
            title="Model Performance Comparison",
            option_widget=AnalyticsTableOptionsWidget(
                sg.target_cols,
                sg.output_list,
                metric="Threshold",
                metric_values=None,
                metrics_to_display=None,
                title=title,
            ),
            plot_function=binary_analytics_table,
            initial_plot=False,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the analytics table."""
        args = (
            self.option_widget.target_cols,  # Updated to use target_columns
            self.option_widget.score_cols,  # Updated to use score_columns
            self.option_widget.metric,  # Updated to use metric
            self.option_widget.metric_values,  # Updated to use metric_values
            list(self.option_widget.metrics_to_display),  # Updated to use metrics_to_display
            self.option_widget.group_by,  # Updated to use group_by
        )
        kwargs = {"title": self.title, "per_context": self.per_context}
        return args, kwargs


class AnalyticsTableOptionsWidget(Box, traitlets.HasTraits):
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
        title: str = None,
    ):
        """
        Widget for selecting analytics table options

        Parameters
        ----------
        target_cols : tuple[str]
            Available target columns.
        score_cols : tuple[str]
            Available score columns.
        metric : str
            Default metric.
        model_options_widget : widget, optional
            Additional model options widget.
        metric_values : list[float], optional
            Default metric values for the slider.
        metrics_to_display : tuple[str], optional
            Metrics to show.
        title : str, optional
            Title of the widget.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.model_options_widget = model_options_widget
        self.title = title
        self.binary_targets = [
            target_col for target_col in target_cols if is_binary_array(sg.dataframe[[pdh.event_value(target_col)]])
        ]
        # Multiple select dropdowns for targets and scores
        self._target_cols = MultiselectDropdownWidget(
            self.binary_targets,
            value=self.binary_targets,
            title="Targets",
        )
        self._score_cols = MultiselectDropdownWidget(
            sg.output_list,
            value=score_cols,
            title="Scores",
        )
        self._metric = Dropdown(
            options=["Threshold"] + [val.name for val in Metric],
            value=metric,
            description="Metric",
            style=WIDE_LABEL_STYLE,
        )
        self._metrics_to_display = MultiselectDropdownWidget(
            options=[THRESHOLD] + STATNAMES + OVERALL_PERFORMANCE,
            value=metrics_to_display if metrics_to_display else list(GENERATED_COLUMNS.values()),
            title="Performance Metrics to Display",
        )
        self._metric_values_slider = FloatRangeSlider(
            min=0.01,
            max=1.00,
            step=0.01,
            value=metric_values if metric_values else [0.2, 0.8],
            description="Metric Values",
            style=WIDE_LABEL_STYLE,
        )
        self._group_by = Dropdown(
            options=["Score", "Target"],
            value="Score",
            description="Group By",
            style=WIDE_LABEL_STYLE,
        )

        self._target_cols.observe(self._on_value_changed, names="value")
        self._score_cols.observe(self._on_value_changed, names="value")
        self._metric.observe(self._on_value_changed, names="value")
        self._metric_values_slider.observe(self._on_value_changed, names="value")
        self._metrics_to_display.observe(self._on_value_changed, names="value")
        self._group_by.observe(self._on_value_changed, names="value")

        v_children = [
            self._target_cols,
            self._score_cols,
            self._metrics_to_display,
            self._metric,
            self._metric_values_slider,
            self._group_by,
        ]
        if model_options_widget:
            v_children.insert(0, model_options_widget)
            self.model_options_widget.observe(self._on_value_changed, names="value")

        grid_layout = Layout(width="100%", grid_template_columns="repeat(3, 1fr)", grid_gap="10px")  # Four columns

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
        self._score_cols.disabled = value
        self._metric.disabled = value
        self._metric_values_slider.disabled = value
        self._metrics_to_display.disabled = value
        self._group_by.disabled = value
        if self.model_options_widget:
            self.model_options_widget.disabled = value

    def _on_value_changed(self, change=None):
        new_value = {
            "target_cols": self._target_cols.value,
            "score_cols": self._score_cols.value,
            "metric": self._metric.value,
            "metric_values": self._metric_values_slider.value,
            "metrics_to_display": self._metrics_to_display.value,
            "group_by": self._group_by.value,
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
        return self._metric_values_slider.value

    @property
    def metrics_to_display(self):
        return self._metrics_to_display.value

    @property
    def group_by(self):
        return self._group_by.value
