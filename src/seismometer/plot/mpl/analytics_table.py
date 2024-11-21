import itertools
from enum import Enum
from typing import Any, List, Optional

import pandas as pd
from great_tables import GT, loc, nanoplot_options, style
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from ...seismogram import Seismogram
from .analytics_table_config import AnalyticsTableConfig
from .color_manipulation import create_bar, lighten_color
from .metric_to_threshold import calculate_stats


class Metric(Enum):
    """
    Enumeration for available values for metric parameter in PerformanceMetrics class.
    """

    Sensitivity = "sensitivity"
    Specificity = "specificity"
    PPV = "ppv"
    Flagged = "flagrate"
    Threshold = "threshold"


class TopLevel(Enum):
    """
    Enumeration for available values for top_level parameter in PerformanceMetrics class.
    """

    Score = "score"
    Target = "target"


class ColorBarStyle(Enum):
    """
    Enumeration for different color bar styles available in PerformanceMetrics class.
    """

    Style1 = 1
    Style2 = 2
    # Add more styles as needed


class TableStyle(Enum):
    """
    Enumeration for different table styles available in great_tables package.
    """

    Style1 = 1
    Style2 = 2
    Style3 = 3
    Style4 = 4
    Style5 = 5
    Style6 = 6


GENERATED_COLUMNS = {
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "ppv": "PPV",
    "flagrate": "Flagged",
    "threshold": "Threshold",
    "positives": "Positives",
    "prevalence": "Prevalence",
    "auroc": "AUROC",
    "auprc": "AUPRC",
}


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
        title: str = "Model Performance Statistics",
        top_level: str = "Score",
        table_config: AnalyticsTableConfig,
        statistics_data: Optional[pd.DataFrame] = None,
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
            Values for the specified metric to derive detailed performance statistics, by default [0.7, 0.8].
        title: str
            The title for the performance statistics report, by default "Model Performance Statistics".
        top_level: str
            The primary grouping category in the performance report, by default 'Score'.
        decimals: int
            The number of decimal places for rounding numerical results, by default 3.
        spanner_colors: Optional[List[str]]
            Colors used for highlighting spanner elements in the report, by default None.
        columns_show_percentages: Union[str, List[str]]
            Columns that will display values as percentages in the report, by default "Prevalence".
        columns_show_bar: Optional[Dict[str, str]]
            Columns mapped to specific colors for bar representation in the report, by default None.
        color_bar_style: int
            The visual style of the color bars used in the report, by default 1.
        style: int
            The overall style index applied to the report's visual representation, by default 1.
        opacity: float
            The opacity level for color bars, affecting their transparency, by default 0.5.
        percentages_decimals: int
            The number of decimal places for percentage values in the report, by default 0.
        alternating_row_colors: bool
            A flag to specify if row colors should alternate, by default True.
        data_bar_stroke_width: int
            The stroke width of the data bars, affecting their thickness, by default 4.
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
        self.target_columns = (
            target_columns if target_columns else [target_col + "_Value" for target_col in sg.target_cols]
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
        self.title = title
        self.top_level = top_level
        self.spanner_colors = table_config.spanner_colors
        self.columns_show_percentages = table_config.columns_show_percentages
        self.columns_show_bar = table_config.columns_show_bar

        self.color_bar_style = ColorBarStyle(table_config.color_bar_style).value
        self.opacity = table_config.opacity
        self.style = TableStyle(table_config.style).value
        self.percentages_decimals = table_config.percentages_decimals
        self.alternating_row_colors = table_config.alternating_row_colors
        self.data_bar_stroke_width = table_config.data_bar_stroke_width

        self._initializing = False
        self.spanner_color_index = 0
        self.rows_group_length = len(self.target_columns) if self.top_level == "Score" else len(self.score_columns)
        self.num_of_rows = len(self.score_columns) * len(self.target_columns)

        # If polars package is not installed, overwrite is_na function in great_tables package to treat Agnostic
        # as pandas dataframe.
        try:
            import polars as pl

            # Use 'pl' in some way to avoid the F401 error
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
        return gt

    def add_coloring_parity(self, gt, columns=None, even_color="#F2F2F2", odd_color="white"):
        """
        Adds alternating row colors to the specified columns in the table.

        Parameters
        ----------
        gt : GT
            The table object to which the alternating row colors will be added.
        columns : Optional[List[str]], optional
            The list of columns to which the alternating row colors will be applied, by default None.
            If None, all columns are considered.
        even_color : str, optional
            The color for even rows, by default "#F2F2F2" (light gray).
        odd_color : str, optional
            The color for odd rows, by default "white".

        Returns
        -------
        gt : GT
            The table object with alternating row colors.
        """
        gt = gt.tab_style(
            style=style.fill(color=even_color),
            locations=loc.body(
                columns=columns,
                rows=[row for row in range(self.num_of_rows) if (row % self.rows_group_length) % 2 == 0],
            ),
        ).tab_style(
            style=style.fill(color=odd_color),
            locations=loc.body(
                columns=columns,
                rows=[row for row in range(self.num_of_rows) if (row % self.rows_group_length) % 2 == 1],
            ),
        )
        return gt

    def group_columns_by_metric_value(self, gt, columns, value):
        gt = (
            gt.tab_spanner(label=f"{self.metric}={value}", columns=columns)
            .cols_label(**{col: "_".join(col.split("_")[1:]) for col in columns})
            .tab_style(
                style=style.borders(sides=["left"], weight="2px", color="black"),
                locations=loc.body(columns=[columns[0]]),
            )
            .tab_style(
                style=style.borders(sides=["right"], weight="2px", color="black"),
                locations=loc.body(columns=[columns[-1]]),
            )
        )
        gt = self.color_group_of_columns(gt, columns)

        return gt

    def color_group_of_columns(self, gt, columns):
        if not self.spanner_colors:
            return gt
        elif self.alternating_row_colors:
            gt = self.add_coloring_parity(
                gt,
                columns,
                even_color=lighten_color(self.spanner_colors[self.spanner_color_index]),
                odd_color=self.spanner_colors[self.spanner_color_index],
            )
        else:
            gt = gt.tab_style(
                style=style.fill(color=self.spanner_colors[self.spanner_color_index]),
                locations=loc.body(columns=columns),
            )
        # increase the spanner_color_index after using spanner colors.
        self.spanner_color_index = (self.spanner_color_index + 1) % len(self.spanner_colors)

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
        self.spanner_color_index = 0
        self.rows_group_length = len(self.target_columns) if self.top_level == "Score" else len(self.score_columns)
        self.num_of_rows = len(self.score_columns) * len(self.target_columns)

        gt = self.generate_initial_table(data)

        # Light gray/white alternating pattern needs to be corrected only if there are even many
        # rows in each row-group.
        if self.rows_group_length % 2 == 0:
            gt = self.add_coloring_parity(gt)

        if self.color_bar_style == 1:
            gt = self.generate_color_bar(gt, columns=data.columns)

        colored_columns = []
        for col in data.columns:
            if col in colored_columns:
                continue

            matching_values = [value for value in self.metric_values if col.startswith(f"{value}_")]
            if matching_values:
                value = matching_values[0]
                columns = [column for column in data.columns if column.startswith(f"{value}_")]
                gt = self.group_columns_by_metric_value(gt, columns, value)
                colored_columns.extend(columns)
            elif col in self.columns_show_bar and f"{col}_bar" in data.columns:
                gt = (
                    gt.tab_spanner(label=col, columns=[col, f"{col}_bar"])
                    .tab_style(
                        style=style.borders(sides=["left"], weight="2px", color="black"),
                        locations=loc.body(columns=[col]),
                    )
                    .tab_style(
                        style=style.borders(sides=["right"], weight="2px", color="black"),
                        locations=loc.body(columns=[f"{col}_bar"]),
                    )
                )
                gt = self.color_group_of_columns(gt, columns=[col, f"{col}_bar"])
                colored_columns.extend([col, f"{col}_bar"])

        gt = gt.opt_stylize(style=self.style)
        return gt

    def _prepare_data(self, data):
        """
        Prepares the data for display in the analytics table.

        Parameters
        ----------
        data: pd.DataFrame
            The input DataFrame containing relevant statistics data.

        Returns
        -------
        pd.DataFrame
            The modified DataFrame with additional columns for bar plots (if applicable).
        """
        if self.color_bar_style == 1:
            for data_col, col in itertools.product(data.columns, self.columns_show_bar):
                if data_col.endswith(f"_{col}") or data_col == col:
                    data.insert(data.columns.get_loc(data_col) + 1, data_col + "_bar", data[data_col])
        elif self.color_bar_style == 2:
            data = data.round(self.decimals)
            for col in data.columns:
                if col in self.columns_show_bar:
                    data[col] = data[col].apply(
                        lambda x: create_bar(x, max_width=75, height=20, color=self.columns_show_bar[col])
                    )
                elif any(col.endswith(f"_{val}") for val in self.columns_show_bar):
                    data[col] = data[col].apply(
                        lambda x: create_bar(
                            x, max_width=75, height=20, color=self.columns_show_bar[col.split("_")[-1]]
                        )
                    )
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
                current_row.update(
                    calculate_stats(self.df[target], self.df[score], self.metric, self.metric_values, self.decimals)
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
