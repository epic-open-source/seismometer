from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
import traitlets
from great_tables import GT, html, loc, style
from ipywidgets import HTML, Box, FloatSlider, Layout, ValueWidget, VBox

from seismometer.controls.explore import ExplorationWidget
from seismometer.controls.selection import MultiselectDropdownWidget, MultiSelectionListWidget
from seismometer.controls.styles import BOX_GRID_LAYOUT, WIDE_LABEL_STYLE, html_title
from seismometer.data.filter import FilterRule


def funcToMethod(func, clas, method_name=None):
    setattr(clas, method_name or func.__name__, func)


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
    def fairness_legend(cls, limit: float = 0.2, open: bool = False, censor_threshold: int = 10) -> str:
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
    def fairness_icon(cls, ratio, limit: float = 0.2) -> "FairnessIcons":
        """
        Icon for fairness ratio
        If fairness ratio is 0.2 (20%) we want to show a warning if we are outside this range and
        a critical warning if we are 2x outside this range

        4/5 - 5/4 is symmetric around 1

        So we are looing at 1-ratio and 1/(1-ratio)
        """
        if ratio > 1 / (1 - 2 * limit):
            return FairnessIcons.CRITICAL_HIGH
        if ratio < 1 - 2 * limit:
            return FairnessIcons.CRITICAL_LOW
        if ratio > 1 / (1 - limit):
            return FairnessIcons.WARNING_HIGH
        if ratio < 1 - limit:
            return FairnessIcons.WARNING_LOW
        if ratio == 1:
            return FairnessIcons.DEFAULT
        return FairnessIcons.GOOD


def fairness_sort_key(key: pd.Series) -> pd.Series:
    return key.apply(lambda x: x if x not in [FairnessIcons.UNKNOWN.value, "--"] else 0)


class MetricFunction:
    def __init__(self, metric_names: list[str], metric_fn: Callable[[pd.DataFrame], dict[str, float]]):
        self.metric_names = metric_names
        self.metric_fn = metric_fn

    def __call__(self, dataframe: pd.DataFrame) -> dict[str, float]:
        return self.metric_fn(dataframe)


def fairness_table(
    dataframe: pd.DataFrame,
    metric_fns: list[MetricFunction],
    metric_list: list[str],
    fairness_ratio: float,
    cohort_dict: dict[str, tuple[Any]],
    *,
    censor_threshold: int = 10,
) -> HTML:
    fairness_data = pd.DataFrame()
    metric_data = pd.DataFrame()
    fairness_icons = pd.DataFrame()
    footnotes = []
    metric_list = list(metric_list)

    for cohort_column in cohort_dict:
        cohort_indices = []
        cohort_values = []
        for cohort_class in cohort_dict[cohort_column]:
            cohort_indices.append((cohort_column, cohort_class))

            cohort_filter = FilterRule.eq(cohort_column, cohort_class)
            cohort_dataframe = cohort_filter.filter(dataframe)

            index_value = {"Count": len(cohort_dataframe)}
            for metric_fn in metric_fns:
                metrics = metric_fn(cohort_dataframe)
                index_value.update(metrics)

            cohort_values.append(index_value)
        cohort_data = pd.DataFrame.from_records(
            cohort_values, index=pd.MultiIndex.from_tuples(cohort_indices, names=["Cohort", "Class"])
        )
        cohort_ratios = cohort_data.div(cohort_data.loc[cohort_data["Count"].idxmax()], axis=1)

        fairness_data = pd.concat([fairness_data, cohort_ratios])
        cohort_icons = cohort_ratios.drop("Count", axis=1).applymap(
            lambda ratio: FairnessIcons.fairness_icon(ratio, fairness_ratio)
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

    legend = FairnessIcons.fairness_legend(fairness_ratio, censor_threshold=censor_threshold)

    table_data = fairness_icons.reset_index()[["Cohort", "Class", "Count"] + metric_list]
    table_data = table_data.sort_values(
        by=["Count", "Cohort", "Class"], ascending=[False, True, True], key=fairness_sort_key
    )

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


class FarinessOptionsWidget(Box, ValueWidget):
    value = traitlets.Dict(help="The selected values for the model options.")

    def __init__(self, metric_names: tuple[str], cohort_dict: dict[str, tuple[Any]], fairness_ratio: float = 0.2):
        self.metric_list = MultiselectDropdownWidget(metric_names, value=metric_names, title="Fairness Metrics")
        self.cohort_list = MultiSelectionListWidget(cohort_dict, show_all=True, title="Cohorts")
        self.fairness_slider = FloatSlider(
            min=0.01,
            max=0.49,
            step=0.01,
            value=fairness_ratio,
            description="Threshold",
            style=WIDE_LABEL_STYLE,
        )

        self.metric_list.observe(self._on_value_changed, names="value")
        self.cohort_list.observe(self._on_value_changed, names="value")
        self.fairness_slider.observe(self._on_value_changed, names="value")

        v_children = [
            html_title("Fairness Options"),
            self.fairness_slider,
            self.metric_list,
        ]
        super().__init__(
            children=[VBox(children=v_children, layout=Layout(align_items="flex-end")), self.cohort_list],
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

    def _on_value_changed(self, change=None):
        self.value = {
            "metric_list": self.metric_list.value,
            "cohort_list": self.cohort_list.value,
            "fairness_ratio": self.fairness_slider.value,
        }

    @property
    def metrics(self):
        return self.metric_list.value

    @property
    def cohorts(self):
        return self.cohort_list.value

    @property
    def fairness_ratio(self):
        return self.fairness_slider.value


class ExplorationFairnessWidget(ExplorationWidget):
    """
    A widget for exploring model fairness across cohorts
    """

    def __init__(self, metrics: list[MetricFunction] | MetricFunction = None):
        """
        Exploration widget for model evaluation, showing a plot for a given target,
        score, threshold, and cohort selection.

        Parameters
        ----------
        title : str
            title of the control
        metrics : list[MetricFunction] or MetricFunction
            list of metric functions to evaluate for fairness
        """

        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        if not isinstance(metrics, list):
            metrics = [metrics]
        self.metric_functions = metrics
        metric_names = [item for metric_fn in metrics for item in metric_fn.metric_names]

        super().__init__(
            title="Fairness Audit",
            option_widget=FarinessOptionsWidget(metric_names, sg.available_cohort_groups, fairness_ratio=0.2),
            plot_function=table_wrapper_function,
            initial_plot=False,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the model evaluation plot"""
        args = (
            self.metric_functions,
            self.option_widget.cohorts,
            self.option_widget.fairness_ratio,
            self.option_widget.metrics,
        )
        return args, {}


def table_wrapper_function(metric_functions, cohort_dict, fairness_ratio, metric_list):
    from seismometer.seismogram import Seismogram

    sg = Seismogram()
    dataframe = sg.dataframe
    if not cohort_dict:
        cohort_dict = sg.available_cohort_groups
    return fairness_table(
        dataframe, metric_functions, metric_list, fairness_ratio, cohort_dict, censor_threshold=sg.censor_threshold
    )
