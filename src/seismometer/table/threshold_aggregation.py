"""
Threshold Aggregation Exploration Widget

Provides an interactive interface for exploring threshold-specific aggregation methods
(e.g., 'first_above_threshold'). Generates a formatted AnalyticsTable-style summary
after applying the selected aggregation to model predictions.
"""

from typing import Any, Optional

import traitlets
from ipywidgets import Dropdown, GridBox, Layout, VBox

from seismometer.controls.explore import ExplorationWidget, _combine_scores_checkbox
from seismometer.controls.selection import MultiselectDropdownWidget, MultiSelectionListWidget
from seismometer.controls.styles import BOX_GRID_LAYOUT, html_title
from seismometer.controls.thresholds import MonotonicProbabilitySliderListWidget
from seismometer.data import pandas_helpers as pdh
from seismometer.data.binary_performance import GENERATED_COLUMNS
from seismometer.data.filter import filter_rule_from_cohort_dictionary
from seismometer.data.performance import THRESHOLD
from seismometer.table.analytics_table import AnalyticsTable

# region Options Widget ---------------------------------------------------------


class ThresholdAggregationOptionsWidget(VBox, traitlets.HasTraits):
    """
    Widget for selecting options for threshold-specific aggregation.

    Provides controls to select a target, score, aggregation method, and a fixed threshold.
    """

    value = traitlets.Dict(help="The selected values for the threshold aggregation options.")

    def __init__(
        self,
        target_cols: tuple[str],
        score_cols: tuple[str],
        cohort_dict: Optional[dict[str, tuple[Any]]] = None,
        *,
        aggregation_methods: Optional[tuple[str]] = None,
        metrics_to_display: Optional[tuple[str]] = None,
    ):
        """
        Initializes the threshold aggregation options widget.

        Parameters
        ----------
        target_cols : tuple[str]
            List of target columns to select from.
        score_cols : tuple[str]
            List of model score columns to select from.
        cohort_dict : dict[str, tuple[Any]], optional
            Dictionary of cohort columns and values for filtering, by default None.
        aggregation_methods : tuple[str], optional
            Supported threshold-based aggregation methods, by default:
            ('first_above_threshold')
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        aggregation_methods = aggregation_methods or ("first_above_threshold",)
        metrics_to_display = metrics_to_display or GENERATED_COLUMNS

        self.title = html_title("Threshold Aggregation Options")

        # Cohort Filter
        self._cohort_dict = MultiSelectionListWidget(cohort_dict or sg.available_cohort_groups, title="Cohort Filter")

        self._target_cols = MultiselectDropdownWidget(
            options=tuple(map(pdh.event_name, target_cols)),
            value=target_cols[:2] if len(target_cols) > 1 else target_cols,
            title="Targets",
        )

        self._score_cols = MultiselectDropdownWidget(
            options=score_cols,
            value=score_cols[:2] if len(score_cols) > 1 else score_cols,
            title="Scores",
        )

        # Aggregation Method Selector
        self._aggregation_method = Dropdown(
            options=aggregation_methods,
            value=aggregation_methods[0],
            description="Aggregation",
            style={"description_width": "min-content"},
            layout=Layout(width="250px"),
        )

        # Threshold Slider
        self._threshold = MonotonicProbabilitySliderListWidget(
            names=("Threshold",),
            value=(0.5,),
            ascending=False,
        )

        # Metrics to display
        self._metrics_to_display = MultiselectDropdownWidget(
            options=GENERATED_COLUMNS,
            value=metrics_to_display,
            title="Performance Metrics to Display",
        )

        # Group By
        self._group_by = Dropdown(
            options=["Score", "Target"],
            value="Score",
            description="Group By",
            style={"description_width": "min-content"},
            layout=Layout(width="250px"),
        )

        # Combine Scores Checkbox
        self.per_context_checkbox = _combine_scores_checkbox(per_context=False)

        # Observe all widgets for updates
        for w in [
            self._cohort_dict,
            self._target_cols,
            self._score_cols,
            self._aggregation_method,
            self._threshold,
            self._metrics_to_display,
            self._group_by,
            self.per_context_checkbox,
        ]:
            w.observe(self._on_value_changed, names="value")

        # Layout
        grid_layout = Layout(
            width="100%", grid_template_columns="repeat(3, 1fr)", justify_items="flex-start", grid_gap="10px"
        )

        # Create a 3-column grid of main controls
        grid_box = GridBox(
            children=[
                self._target_cols,
                self._score_cols,
                self._metrics_to_display,
                self._aggregation_method,
                self._threshold,
                self._group_by,
                self.per_context_checkbox,
            ],
            layout=grid_layout,
        )

        # Combine with title and cohort filter above
        grid_with_title = VBox(
            children=[self.title, grid_box],
            layout=Layout(align_items="flex-start"),
        )

        super().__init__(children=[self._cohort_dict, grid_with_title], layout=BOX_GRID_LAYOUT)
        self._on_value_changed()
        self._disabled = False

    # region Properties

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, value: bool):
        self._disabled = value
        self._cohort_dict.disabled = value
        self._target_cols.disabled = value
        self._score_cols.disabled = value
        self._aggregation_method.disabled = value
        self._threshold.disabled = value
        self._metrics_to_display.disabled = value
        self._group_by.disabled = value
        self.per_context_checkbox.disabled = value

    def _on_value_changed(self, change=None):
        """Update internal dictionary when any option changes."""
        self.value = {
            "cohort_dict": self._cohort_dict.value,
            "target_cols": self._target_cols.value,
            "score_cols": self._score_cols.value,
            "aggregation_method": self._aggregation_method.value,
            "threshold": list(self._threshold.value.values())[0],
            "metrics_to_display": self._metrics_to_display.value,
            "group_by": self._group_by.value,
            "group_scores": self.per_context_checkbox.value,
        }

    @property
    def cohort_dict(self) -> dict[str, tuple[Any]]:
        return self._cohort_dict.value

    @property
    def target_cols(self) -> tuple[str]:
        return self._target_cols.value

    @property
    def score_cols(self) -> tuple[str]:
        return self._score_cols.value

    @property
    def aggregation_method(self) -> str:
        return self._aggregation_method.value

    @property
    def metrics_to_display(self):
        return self._metrics_to_display.value

    @property
    def group_by(self) -> str:
        return self._group_by.value

    @property
    def threshold(self) -> float:
        return list(self._threshold.value.values())[0]

    @property
    def group_scores(self) -> bool:
        return self.per_context_checkbox.value

    # endregion


# endregion
# region Explore Widget ---------------------------------------------------------


class ExploreThresholdAggregation(ExplorationWidget):
    """
    Exploration widget for threshold-specific aggregation methods.

    Applies a fixed threshold and aggregation strategy (e.g., 'first_above_threshold'),
    then generates a formatted AnalyticsTable-style summary of results.
    """

    def __init__(self, title: Optional[str] = None):
        """
        Initializes the threshold aggregation exploration widget.

        Parameters
        ----------
        title : str, optional
            The title displayed above the control, by default "Threshold Aggregation Explorer".
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        self.title = title or "Threshold Aggregation Explorer"

        super().__init__(
            title=self.title,
            option_widget=ThresholdAggregationOptionsWidget(
                target_cols=tuple(map(pdh.event_name, sg.get_binary_targets())),
                score_cols=sg.output_list,
                cohort_dict=sg.available_cohort_groups,
            ),
            plot_function=self._plot_threshold_aggregation,
            initial_plot=False,
        )

    def _plot_threshold_aggregation(
        self,
        cohort_dict: dict[str, tuple[Any]],
        target_cols: list[str],
        score_cols: list[str],
        aggregation_method: str,
        threshold: float,
        metrics_to_display: list[str],
        group_by: str,
        per_context: bool,
    ):
        """
        Applies a threshold-based aggregation and renders an AnalyticsTable-style summary.

        Parameters
        ----------
        cohort_dict : dict[str, tuple[Any]]
            Cohort filter to apply before aggregation.
        target_cols : list[str]
            Target columns to aggregate over.
        score_cols : list[str]
            Score columns used for thresholding.
        aggregation_method : str
            Aggregation strategy to apply (e.g., 'first_above_threshold').
        threshold : float
            The score threshold to use.
        metrics_to_display : list[str]
            Metrics to include in the output table.
        group_by : str
            Whether to group by "Score" or "Target".
        per_context : bool
            Whether to aggregate per context instead of globally.

        Returns
        -------
        HTML
            Rendered AnalyticsTable summary for the aggregated data.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        df = sg.dataframe

        if cohort_dict:
            df = filter_rule_from_cohort_dictionary(cohort_dict).filter(df)

        # Build AnalyticsTable-style summary using the existing class
        summary_table = AnalyticsTable(
            score_columns=score_cols,
            target_columns=target_cols,
            metric=THRESHOLD,
            metric_values=[threshold],
            metrics_to_display=metrics_to_display,
            title="Threshold Specific Aggregation",
            top_level=group_by,
            cohort_dict=cohort_dict,
            per_context=per_context,
            censor_threshold=sg.censor_threshold,
            aggregation_method=aggregation_method,
        )

        return summary_table.analytics_table()

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """
        Generates arguments for the plot_function.

        Returns
        -------
        tuple[tuple, dict]
            Positional and keyword arguments to be passed to the plot_function.
        """
        opts = self.option_widget
        args = (
            opts.cohort_dict,
            tuple(map(pdh.event_value, opts.target_cols)),
            opts.score_cols,
            opts.aggregation_method,
            opts.threshold,
            opts.metrics_to_display,
            opts.group_by,
            opts.group_scores,
        )
        kwargs = {}
        return args, kwargs


# endregion
