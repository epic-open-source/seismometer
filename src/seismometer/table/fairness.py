import logging
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import traitlets
from great_tables import GT, html, loc, style
from ipywidgets import HTML, Box, FloatSlider, Layout, ValueWidget, VBox

from seismometer.controls.explore import ExplorationWidget, ModelOptionsWidget
from seismometer.controls.selection import MultiselectDropdownWidget, MultiSelectionListWidget
from seismometer.controls.styles import BOX_GRID_LAYOUT, WIDE_LABEL_STYLE, html_title
from seismometer.data import pandas_helpers as pdh
from seismometer.data.filter import FilterRule
from seismometer.data.performance import BinaryClassifierMetricGenerator, MetricGenerator

logger = logging.getLogger("seismometer")

COUNT = "Count"
COHORT = "Cohort"
CLASS = "Class"


# region Fairness Icons
class FairnessIcons(Enum):
    """
    Enum for fairness icons
    """

    DEFAULT = "🔹"
    GOOD = "🟢"
    UNKNOWN = "❔"
    WARNING_HIGH = "🔼"
    WARNING_LOW = "🔽"
    CRITICAL_HIGH = "🔺"
    CRITICAL_LOW = "🔻"

    @classmethod
    def get_fairness_legend(cls, limit: float = 0.25, *, open: bool = True, censor_threshold: int = 10) -> str:
        return html(
            f"""
<details {'open' if open else ''}><summary><span style="font-size: 100%; font-weight: bold;">Legend</span></summary>
<table>
<tr style="background: none;">
<td style="text-align: left;">{cls.DEFAULT.value} The default cohort for the category.</td>
<td style="text-align: left;">{cls.GOOD.value} Within {limit:.2%} of the default cohort.</td>
</tr>
<tr style="background: none;">
<td style="text-align: left;">{cls.WARNING_LOW.value} Within {2*limit:.2%} lower than the default cohort.</td>
<td style="text-align: left;">{cls.WARNING_HIGH.value} Within {2*limit:.2%} greater than the default cohort.</td>
</tr>
<tr style="background: none;">
<td style="text-align: left;">{cls.CRITICAL_LOW.value} More than {2*limit:.2%} lower than the default cohort.</td>
<td style="text-align: left;">{cls.CRITICAL_HIGH.value} More than {2*limit:.2%} greater than the default cohort.</td>
</tr>
<tr style="background: none;">
<td style="text-align: left;">{cls.UNKNOWN.value} Censored, fewer than {censor_threshold} observations.</td>
</tr>
</details>"""
        )

    @classmethod
    def get_fairness_icon(cls, ratio, limit: float = 0.25) -> "FairnessIcons":
        """
        Icon for fairness ratio
        If fairness ratio is 0.25 (25%) we want to show a warning if we are outside this range and
        a critical warning if we are 2x outside this range

        We are looking at 1 / (1 + limit) < ratio < 1 + limit

        For a limit of 0.25 we are looking at 0.80 < ratio < 1.25 (25% bigger, or 20% smaller)
        Alternatively for a limit of 0.50 we are looking at 0.67 < ratio < 1.50 (50% bigger, or 33% smaller)
        For a limit of 1.0 we are looking at 0.5 < ratio < 2.0 (100% bigger, or 50% smaller) which allows a 2x
        difference between a one group and another.

        The extended upper bounds are at 1 + limit and 1 + 2*limit.

        Parameters
        ----------
        ratio : float
            Ratio of the cohort to the largest cohort
        limit : float, optional
            Allowed percentage difference by cohort, by default 0.25, measured from the smaller metric to the larger.

        Returns
        -------
        FairnessIcons
            Icon for the ratio based on the limit.
        """
        upper_limit, twice_upper_limit = 1 + limit, 1 + 2 * limit
        lower_limit, twice_lower_limit = 1 / upper_limit, 1 / twice_upper_limit

        if ratio is None or np.isnan(ratio):
            return FairnessIcons.UNKNOWN
        if ratio > twice_upper_limit:
            return FairnessIcons.CRITICAL_HIGH
        if ratio < twice_lower_limit:
            return FairnessIcons.CRITICAL_LOW
        if ratio > upper_limit:
            return FairnessIcons.WARNING_HIGH
        if ratio < lower_limit:
            return FairnessIcons.WARNING_LOW
        if ratio == 1:
            return FairnessIcons.DEFAULT
        return FairnessIcons.GOOD


# endregion
# region Fairness Table


def sort_fairness_table(dataframe: pd.DataFrame, cohort_groups: tuple[str]):
    """
    Generates a sort key for the fairness table based on Cohort group name and Count

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame to sort
    cohort_groups : tuple[str]
        Cohort group names for sorting.
    """

    def fairness_sort_key(key: pd.Series) -> pd.Series:
        match key.name:
            case "Count":
                return key.apply(lambda x: -x if x not in [FairnessIcons.UNKNOWN.value, "--"] else 0)
            case "Cohort":
                return key.apply(lambda x: cohort_groups.index(x))
            case _:
                return key

    return dataframe.sort_values(by=[COHORT, COUNT], key=fairness_sort_key)


def fairness_table(
    dataframe: pd.DataFrame,
    metric_fn: Callable[..., dict[str, float]],
    metric_list: list[str] = None,
    fairness_ratio: float = 0.25,
    cohort_dict: dict[str, tuple[Any]] = None,
    *,
    censor_threshold: int = 10,
    **kwargs,
) -> HTML:
    """
    Fairness table for evaluating metrics across cohorts, found by taking the largest subgroup within each cohort
    as the default cohort and taking the ratio between a subgroup's metric and the default group's metric.

    For example if if a cohort has three classes A, B, and C with counts of 10, 20, and 30 respectively, the default
    cohort would be C. For a metric M, we would calculate M(A), M(B) and M(C) and then calculate the ratios
    M(A)/M(C) and its reciprocal M(C)/M(A).

    If M(A)/M(C) > 1 + limit, then cohort A will be flagged as higher than the default.
    If M(A)/M(C) > 1 + 2 * limit, then cohort A will be flagged as critically higher than the default.

    If M(C)/M(A) > 1 + limit, then cohort A will be flagged as lower than the default.
    If M(C)/M(A) > 1 + 2 * limit, then cohort A will be flagged as critically lower than the default.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Source data to generate a fairness table for
    metric_fn : Callable[..., dict[str, float]]
        Metric function to generate raw metrics, which MUST be positive values.
    metric_list : list[str]
        List of metrics to use from the metric function
    fairness_ratio : float
        Ratio of acceptable difference between cohorts, 20% is 0.2, 200% is 2.0.
        Bound is multiplicatively symmetric around 1, so 200% means up to 3x larger or 3x smaller (1/3 the original).
        A typical bound is 0.25 (1.25x larger or 0.8x smaller)
    cohort_dict : dict[str, tuple[Any]]
        collection of cohort groups to loop over
    censor_threshold : int, optional
        Limit at which a cohort group will be removed from the table if not enough observations are found,
        by default 10.

    Returns
    -------
    HTML
        The HTML table for the fairness evaluation
    """
    fairness_groups = []
    metric_groups = []
    icon_groups = []

    if fairness_ratio <= 0:
        raise ValueError("Fairness ratio must be greater than 0")

    if not cohort_dict:
        raise ValueError("No cohorts provided for fairness evaluation")

    for cohort_column in cohort_dict:
        cohort_indices = []
        cohort_values = []
        for cohort_class in cohort_dict[cohort_column]:
            cohort_indices.append((cohort_column, cohort_class))

            cohort_filter = FilterRule.eq(cohort_column, cohort_class)
            cohort_dataframe = cohort_filter.filter(dataframe)

            index_value = {COUNT: len(cohort_dataframe)}
            metrics = metric_fn(cohort_dataframe, metric_list, **kwargs)
            index_value.update(metrics)

            cohort_values.append(index_value)
        cohort_data = pd.DataFrame.from_records(
            cohort_values, index=pd.MultiIndex.from_tuples(cohort_indices, names=[COHORT, CLASS])
        )
        cohort_ratios = cohort_data.div(cohort_data.loc[cohort_data[COUNT].idxmax()], axis=1)

        cohort_icons = cohort_ratios.drop(COUNT, axis=1).map(
            lambda ratio: FairnessIcons.get_fairness_icon(ratio, fairness_ratio)
        )
        cohort_icons[COUNT] = cohort_data[COUNT]

        fairness_groups.append(cohort_ratios)
        icon_groups.append(cohort_icons)
        metric_groups.append(cohort_data)

    fairness_data = pd.concat(fairness_groups)
    metric_data = pd.concat(metric_groups)
    fairness_icons = pd.concat(icon_groups).astype(object)

    for cohort_column, cohort_class in metric_data.index:
        if metric_data.loc[(cohort_column, cohort_class), COUNT] < censor_threshold:
            fairness_icons.loc[(cohort_column, cohort_class), COUNT] = FairnessIcons.UNKNOWN.value
            for metric in metric_list:
                fairness_icons.loc[(cohort_column, cohort_class), metric] = FairnessIcons.UNKNOWN
                metric_data.loc[(cohort_column, cohort_class), metric] = np.nan
                fairness_data.loc[(cohort_column, cohort_class), metric] = np.nan

    fairness_icons[metric_list] = fairness_icons[metric_list].map(
        lambda x: x.value if x != FairnessIcons.UNKNOWN else "--"
    )
    fairness_icons[metric_list] = (
        fairness_icons[metric_list]
        + metric_data[metric_list].map(lambda x: f"  {x:.2f}  " if not np.isnan(x) else "")
        + fairness_data[metric_list].map(lambda x: f"  ({x-1:.2%})  " if (np.isfinite(x) and x != 1.0) else "")
    )

    legend = FairnessIcons.get_fairness_legend(fairness_ratio, censor_threshold=censor_threshold)

    table_data = fairness_icons.reset_index()[[COHORT, CLASS, COUNT] + metric_list]
    table_data = sort_fairness_table(table_data, list(cohort_dict.keys()))

    table_html = (
        GT(table_data)
        .tab_stub(groupname_col=COHORT, rowname_col=CLASS)
        .tab_style(
            style=style.text(align="center"),
            locations=loc.column_header(),
        )
        .tab_style(
            style=style.borders(sides=["right"], weight="1px", color="#D3D3D3"),
            locations=loc.body(columns=[COUNT] + metric_list),
        )
        .tab_source_note(source_note=legend)
        .opt_horizontal_padding(scale=3)
        .tab_options(row_group_font_weight="bold")
        .cols_align(align="left")
        .cols_align(align="right", columns=[COUNT])
    ).as_raw_html()
    return HTML(table_html, layout=Layout(max_height="800px"))


# endregion
# region Fairness Table Wrapper


def binary_metrics_fairness_table(
    metric_generator: BinaryClassifierMetricGenerator,
    metric_list: list[str],
    cohort_dict: dict[str, tuple[Any]],
    fairness_ratio: float,
    target: str,
    score: str,
    threshold: float,
    *,
    per_context=False,
) -> HTML:
    """
    Binary fairness metrics table

    Parameters
    ----------
    metric_generator : The BinaryClassifierMetricGenerator that determines rho.
    metric_list : list[str]
        List of metrics to evaluate.
    cohort_dict : dict[str, tuple[Any]]
        Collection of cohort groups to loop over.
    fairness_ratio : float
        Ratio of acceptable difference between cohorts, 20% is 0.2, 200% is 2.0.
    target : str
        The target descriptor for the binary classifier.
    score : str
        The score descriptor for the binary classifier.
    threshold : float
        The threshold for the binary classifier.
    per_context : bool, optional
        Whether to group scores by context, by default False.

    Returns
    -------
    HTML
        The HTML table for the fairness evaluation.
    """
    from seismometer.seismogram import Seismogram

    sg = Seismogram()
    target_column = pdh.event_value(target)
    data = (
        pdh.event_score(
            sg.dataframe,
            sg.entity_keys,
            score=score,
            ref_time=sg.predict_time,
            ref_event=target,
            aggregation_method=sg.event_aggregation_method(target),
        )
        if per_context
        else sg.dataframe
    )
    return fairness_table(
        data,
        metric_generator,
        metric_list,
        fairness_ratio,
        cohort_dict,
        censor_threshold=sg.censor_threshold,
        target_col=target_column,
        score_col=score,
        score_threshold=threshold,
    )


def custom_metrics_fairness_table(metric_generator, metric_list, cohort_dict, fairness_ratio) -> HTML:
    """
    For use by fairness tables that need custom metric generators.

    Parameters
    ----------
    metric_generator : MetricGenerator
        Metric generator to use for the fairness table.
    metric_list : list[str]
        List of metrics to evaluate.
    cohort_dict : dict[str, tuple[Any]]
        Collection of cohort groups to loop over.
    fairness_ratio : float
        Ratio of acceptable difference between cohorts, 20% is 0.2, 200% is 2.0.

    Returns
    -------
    HTML
        The HTML table for the fairness evaluation.
    """
    from seismometer.seismogram import Seismogram

    sg = Seismogram()
    dataframe = sg.dataframe
    if not cohort_dict:
        cohort_dict = sg.available_cohort_groups
    return fairness_table(
        dataframe, metric_generator, metric_list, fairness_ratio, cohort_dict, censor_threshold=sg.censor_threshold
    )


# endregion

# region Fairness Controls


class FairnessOptionsWidget(Box, ValueWidget):
    value = traitlets.Dict(help="The selected values for the model options.")

    def __init__(
        self,
        metric_names: tuple[str],
        cohort_dict: dict[str, tuple[Any]],
        fairness_ratio: float = 0.2,
        *,
        model_options_widget=None,
        default_metrics=None,
    ):
        """
        Widget for selecting fairness options

        Parameters
        ----------
        metric_names : tuple[str]
            Metrics that can be evaluated for fairness.
        cohort_dict : dict[str, tuple[Any]]
            Dictionary of cohort groups.
        fairness_ratio : float, optional
            Allowed difference by cohort, by default 0.2.
        model_options_widget : Optional[widget], optional
            Additional model options if needed, will appear before fairness options, by default None.
        default_metrics : Optional[tuple[str]], optional
            Default list of metrics to select initially for fairness evaluation, by default None.
        """
        self.model_options_widget = model_options_widget
        default_metrics = default_metrics or metric_names
        self.metric_list = MultiselectDropdownWidget(metric_names, value=default_metrics, title="Fairness Metrics")
        self.cohort_list = MultiSelectionListWidget(cohort_dict, title="Cohorts")
        self.fairness_slider = FloatSlider(
            min=0.01,
            max=1.00,
            step=0.01,
            value=fairness_ratio,
            description="Threshold",
            style=WIDE_LABEL_STYLE,
        )
        self.all_cohorts = cohort_dict
        self.metric_list.observe(self._on_value_changed, names="value")
        self.cohort_list.observe(self._on_value_changed, names="value")
        self.fairness_slider.observe(self._on_value_changed, names="value")

        v_children = [
            html_title("Fairness Options"),
            self.fairness_slider,
            self.metric_list,
        ]
        if model_options_widget:
            v_children.insert(0, model_options_widget)
            self.model_options_widget.observe(self._on_value_changed, names="value")

        super().__init__(
            children=[
                VBox(children=v_children, layout=Layout(align_items="flex-end", flex="0 0 auto")),
                self.cohort_list,
            ],
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
        self.metric_list.disabled = value
        self.cohort_list.disabled = value
        self.fairness_slider.disabled = value
        if self.model_options_widget:
            self.model_options_widget.disabled = value

    def _on_value_changed(self, change=None):
        new_value = {
            "metric_list": self.metric_list.value,
            "cohort_list": self.cohort_list.value,
            "fairness_ratio": self.fairness_slider.value,
        }
        if self.model_options_widget:
            new_value["model_options"] = self.model_options_widget.value
        self.value = new_value

    @property
    def metrics(self):
        return self.metric_list.value

    @property
    def cohorts(self):
        return self.cohort_list.value or self.all_cohorts

    @property
    def fairness_ratio(self):
        return self.fairness_slider.value

    @property
    def model_options(self):
        return self.model_options_widget if self.model_options_widget else None


class ExplorationFairnessWidget(ExplorationWidget):
    """
    A widget for exploring model fairness across cohorts
    """

    def __init__(self, metrics: MetricGenerator):
        """
        Exploration widget for model fairness evaluation based on cohort selection.
        Only works for global model metrics, not metrics that rely on parameters.

        Parameters
        ----------
        metrics : list[MetricGenerator] or MetricGenerator
            list of metric functions to evaluate for fairness
        """

        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.metrics_generator = metrics
        metric_names = [name for name in metrics.metric_names]

        super().__init__(
            title="Fairness Audit",
            option_widget=FairnessOptionsWidget(metric_names, sg.available_cohort_groups, fairness_ratio=0.2),
            plot_function=custom_metrics_fairness_table,
            initial_plot=False,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the model evaluation plot"""
        args = (
            self.metrics_generator,
            list(self.option_widget.metrics),
            self.option_widget.cohorts,
            self.option_widget.fairness_ratio,
        )
        return args, {}


class ExploreBinaryModelFairness(ExplorationWidget):
    """
    A widget for exploring model fairness across cohorts for a binary classifier
    """

    def __init__(self, rho: Optional[float] = None):
        """
        Exploration widget for model evaluation, showing a plot for a given target,
        score, threshold, and cohort selection.

        Parameters
        ----------
        rho : Optional[float], between 0 and 1
            treatment efficacy as a probability of positive result.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.metric_generator = BinaryClassifierMetricGenerator(rho=rho)
        metric_names = tuple(self.metric_generator.metric_names)
        model_options_widget = ModelOptionsWidget(
            sg.target_cols, sg.output_list, {"Score Threshold": max(sg.thresholds)}, per_context=False
        )

        super().__init__(
            title="Binary Classifier Fairness Audit",
            option_widget=FairnessOptionsWidget(
                metric_names,
                sg.available_cohort_groups,
                fairness_ratio=0.2,
                model_options_widget=model_options_widget,
                default_metrics=["Accuracy", "Sensitivity", "Specificity", "PPV"],
            ),
            plot_function=binary_metrics_fairness_table,
            initial_plot=False,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the model evaluation plot"""
        args = (
            self.metric_generator,
            list(self.option_widget.metrics),
            self.option_widget.cohorts,
            self.option_widget.fairness_ratio,
            self.option_widget.model_options.target,
            self.option_widget.model_options.score,
            self.option_widget.model_options.thresholds["Score Threshold"],
        )
        kwargs = {"per_context": self.option_widget.model_options.group_scores}
        return args, kwargs


# endregion
