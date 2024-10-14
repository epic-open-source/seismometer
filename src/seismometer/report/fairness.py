from enum import Enum
from typing import Any, Callable

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
    def get_fairness_legend(cls, limit: float = 0.2, open: bool = False, censor_threshold: int = 10) -> str:
        return html(
            f"""
<details {'open' if open else ''}><summary>Legend</summary>
<table>
<tr style="background: none;">
<td style="text-align: left;">{cls.DEFAULT.value} the largest cohort for the category</td>
<td style="text-align: left;">{cls.GOOD.value} within {limit:.2%} of the largest cohort</td>
</tr>
<tr style="background: none;">
<td style="text-align: left;">{cls.WARNING_LOW.value} within {2*limit:.2%} lower than the largest cohort</td>
<td style="text-align: left;">{cls.WARNING_HIGH.value} within {2*limit:.2%} greater than the largest cohort</td>
</tr>
<tr style="background: none;">
<td style="text-align: left;">{cls.CRITICAL_LOW.value} more than {2*limit:.2%} lower than the largest cohort</td>
<td style="text-align: left;">{cls.CRITICAL_HIGH.value} more than {2*limit:.2%} greater than the largest cohort</td>
</tr>
<tr style="background: none;">
<td style="text-align: left;">{cls.UNKNOWN.value} fewer than {censor_threshold} samples, data was censored</td>
</tr>
</details>"""
        )

    @classmethod
    def get_fairness_icon(cls, ratio, limit: float = 0.2) -> "FairnessIcons":
        """
        Icon for fairness ratio
        If fairness ratio is 0.2 (20%) we want to show a warning if we are outside this range and
        a critical warning if we are 2x outside this range

        4/5 - 5/4 is symmetric around 1

        So we are looking at 1-ratio and 1/(1-ratio)

        the limit is required to be strictly between 0 and 0.5.
        """
        lower_limit, upper_limit = 1 - limit, 1 / (1 - limit)
        twice_lower_limit, twice_upper_limit = 1 - 2 * limit, 1 / (1 - 2 * limit)
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
        Dataframe to sort
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

    return dataframe.sort_values(by=["Cohort", "Count"], key=fairness_sort_key)


def fairness_table(
    dataframe: pd.DataFrame,
    metric_fn: Callable[..., dict[str, float]],
    metric_list: list[str],
    fairness_ratio: float,
    cohort_dict: dict[str, tuple[Any]],
    *,
    censor_threshold: int = 10,
    **kwargs,
) -> HTML:
    fairness_data = pd.DataFrame()
    metric_data = pd.DataFrame()
    fairness_icons = pd.DataFrame()
    footnotes = []

    if fairness_ratio > 1:  # backwards compatibility for limits greater than 1
        fairness_ratio = (fairness_ratio - 1) / fairness_ratio  # (1.25 - 1 ) / 1.25 = 0.25/1.25 = 0.2

    if fairness_ratio <= 0 or fairness_ratio >= 0.5:
        raise ValueError("Fairness ratio must be between 0 and 0.5")

    if not cohort_dict:
        raise ValueError("No cohorts provided for fairness evaluation")

    for cohort_column in cohort_dict:
        cohort_indices = []
        cohort_values = []
        for cohort_class in cohort_dict[cohort_column]:
            cohort_indices.append((cohort_column, cohort_class))

            cohort_filter = FilterRule.eq(cohort_column, cohort_class)
            cohort_dataframe = cohort_filter.filter(dataframe)

            index_value = {"Count": len(cohort_dataframe)}
            metrics = metric_fn(cohort_dataframe, [x for x in metric_list if x in metric_fn.metric_names], **kwargs)
            index_value.update(metrics)

            cohort_values.append(index_value)
        cohort_data = pd.DataFrame.from_records(
            cohort_values, index=pd.MultiIndex.from_tuples(cohort_indices, names=["Cohort", "Class"])
        )
        cohort_ratios = cohort_data.div(cohort_data.loc[cohort_data["Count"].idxmax()], axis=1)

        fairness_data = pd.concat([fairness_data, cohort_ratios])
        cohort_icons = cohort_ratios.drop("Count", axis=1).applymap(
            lambda ratio: FairnessIcons.get_fairness_icon(ratio, fairness_ratio)
        )
        cohort_icons["Count"] = cohort_data["Count"]
        fairness_icons = pd.concat([fairness_icons, cohort_icons])
        metric_data = pd.concat([metric_data, cohort_data])
        for cohort_column, cohort_class in cohort_indices:
            if cohort_data.loc[(cohort_column, cohort_class), "Count"] < censor_threshold:
                fairness_icons.loc[(cohort_column, cohort_class), "Count"] = FairnessIcons.UNKNOWN.value
                for metric in metric_list:
                    fairness_icons.loc[(cohort_column, cohort_class), metric] = FairnessIcons.UNKNOWN
                    metric_data.loc[(cohort_column, cohort_class), metric] = np.nan
                    fairness_data.loc[(cohort_column, cohort_class), metric] = np.nan
                warning = f'not enough samples with "{cohort_column}" == "{cohort_class}"'
                footnotes.append(f"{FairnessIcons.UNKNOWN.value} **{cohort_column}** - {warning}")

    fairness_icons[metric_list] = fairness_icons[metric_list].applymap(
        lambda x: x.value if x != FairnessIcons.UNKNOWN else "--"
    )
    fairness_icons[metric_list] = (
        fairness_icons[metric_list]
        + metric_data[metric_list].applymap(lambda x: f"  {x:.2f}  " if not np.isnan(x) else "")
        + fairness_data[metric_list].applymap(lambda x: f"  ({1-x:.2%})  " if not (np.isnan(x) or x == 1.0) else "")
    )

    legend = FairnessIcons.get_fairness_legend(fairness_ratio, censor_threshold=censor_threshold)

    table_data = fairness_icons.reset_index()[["Cohort", "Class", "Count"] + metric_list]
    table_data = sort_fairness_table(table_data, list(cohort_dict.keys()))

    table_html = (
        GT(table_data)
        .tab_stub(groupname_col="Cohort", rowname_col="Class")
        .tab_style(
            style=style.borders(sides=["right"], weight="1px", color="#D3D3D3"),
            locations=loc.body(columns=metric_list),
        )
        .cols_align(align="left")
        .cols_align(align="right", columns=["Count"])
        .tab_source_note(source_note=legend)
        .opt_horizontal_padding(scale=3)
        .tab_options(row_group_font_weight="bold")
    ).as_raw_html()
    return HTML(table_html)


# region Fairness Table Wrapper


def binary_metrics_fairness_table(
    metric_generator, metric_list, cohort_dict, fairness_ratio, target, score, threshold, *, per_context=False
) -> HTML:
    from seismometer.seismogram import Seismogram

    sg = Seismogram()
    target_column = pdh.event_value(target)
    data = (
        pdh.event_score(
            sg.data(),
            sg.entity_keys,
            score=score,
            ref_event=sg.predict_time,
            aggregation_method=sg.event_aggregation_method(target_column),
        )
        if per_context
        else sg.data()
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


class FarinessOptionsWidget(Box, ValueWidget):
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
            Metrics that can be evaluated for fairness
        cohort_dict : dict[str, tuple[Any]]
            Dictionary of cohort groups.
        fairness_ratio : float, optional
            Allowed difference by cohort, by default 0.2
        model_options_widget : _type_, optional
            Additional model options if needed, will appear before fairness options, by default None
        """
        self.model_options_widget = model_options_widget
        default_metrics = default_metrics or metric_names
        self.metric_list = MultiselectDropdownWidget(metric_names, value=default_metrics, title="Fairness Metrics")
        self.cohort_list = MultiSelectionListWidget(cohort_dict, title="Cohorts")
        self.fairness_slider = FloatSlider(
            min=0.01,
            max=0.49,
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
            option_widget=FarinessOptionsWidget(metric_names, sg.available_cohort_groups, fairness_ratio=0.2),
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

    def __init__(self):
        """
        Exploration widget for model evaluation, showing a plot for a given target,
        score, threshold, and cohort selection.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.metric_generator = BinaryClassifierMetricGenerator()
        metric_names = tuple(self.metric_generator.metric_names)
        model_options_widget = ModelOptionsWidget(
            sg.target_cols, sg.output_list, {"Score Threshold": max(sg.thresholds)}, per_context=False
        )

        super().__init__(
            title="Binary Classifier Fairness Audit",
            option_widget=FarinessOptionsWidget(
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
