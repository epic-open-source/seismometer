import logging
import os
from functools import wraps
from typing import Any, Callable, Literal, Optional

import traitlets
from IPython import get_ipython
from IPython.display import display
from ipywidgets import HTML, Box, Button, Checkbox, Dropdown, Layout, Output, ValueWidget, VBox

from seismometer.data.performance import MetricGenerator

from .selection import DisjointSelectionListsWidget, MultiselectDropdownWidget, MultiSelectionListWidget
from .styles import BOX_GRID_LAYOUT, WIDE_LABEL_STYLE, html_title
from .thresholds import MonotonicProbabilitySliderListWidget

logger = logging.getLogger("seismometer")


# region Model Evaluation Header Controls
def _combine_scores_checkbox(per_context):
    """Returns checkbox for if scores should be combined."""
    return Checkbox(
        value=per_context,
        description="Combine scores?",
        disabled=False,
        tooltip="Combine scores by taking a representative score from the target window.",
        style=WIDE_LABEL_STYLE,
    )


class UpdatePlotWidget(Box):
    """Widget for updating plots and showing code behind the plot call."""

    UPDATE_PLOTS = "Update"
    UPDATING_PLOTS = "Updating ..."
    SHOW_CODE = "Show code?"
    SHOW_DATA = "Show raw data?"

    def __init__(self):
        self.code_checkbox = Checkbox(
            value=False,
            description=self.SHOW_CODE,
            disabled=False,
            indent=False,
            tooltip="Show the code used to generate the plot.",
            layout=Layout(margin="var(--jp-widgets-margin) var(--jp-widgets-margin) var(--jp-widgets-margin) 10px;"),
        )
        self.data_checkbox = Checkbox(
            value=False,
            description=self.SHOW_DATA,
            disabled=False,
            indent=False,
            tooltip="Show the raw data used to generate the plot.",
            layout=Layout(margin="var(--jp-widgets-margin) var(--jp-widgets-margin) var(--jp-widgets-margin) 10px;"),
        )

        self.plot_button = Button(description=self.UPDATE_PLOTS, button_style="primary")
        layout = Layout(align_items="flex-start")
        children = [self.plot_button, self.code_checkbox, self.data_checkbox]
        super().__init__(layout=layout, children=children)

    @property
    def show_code(self) -> bool:
        return self.code_checkbox.value

    @show_code.setter
    def show_code(self, show_code: bool) -> bool:
        self.code_checkbox.value = show_code

    @property
    def show_data(self) -> bool:
        return self.data_checkbox.value

    @show_data.setter
    def show_data(self, show_data: bool) -> bool:
        self.data_checkbox.value = show_data

    @property
    def disabled(self) -> bool:
        return self.plot_button.disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        if not disabled:
            self.plot_button.description = self.UPDATE_PLOTS
        self.plot_button.disabled = disabled

    def on_click(self, callback):
        @wraps(callback)
        def callback_wrapper(button):
            button.description = self.UPDATING_PLOTS
            button.disabled = True
            callback(button)
            button.description = self.UPDATE_PLOTS

        self.plot_button.on_click(callback_wrapper)

    def on_toggle_code(self, callback):
        @wraps(callback)
        def callback_wrapper(change):
            callback(self.code_checkbox.value)

        self.code_checkbox.observe(callback_wrapper, "value")

    def on_toggle_data(self, callback):
        @wraps(callback)
        def callback_wrapper(change):
            callback(self.data_checkbox.value)

        self.data_checkbox.observe(callback_wrapper, "value")


class ModelOptionsWidget(VBox, ValueWidget):
    value = traitlets.Dict(help="The selected values for the model options.")

    def __init__(
        self,
        target_names: tuple[Any],
        score_names: tuple[Any],
        thresholds: Optional[dict[str, float]] = None,
        per_context: Optional[bool] = None,
    ):
        """
        Widget for model options.

        Parameters
        ----------
        target_names : tuple[Any]
            List of target column names.
        score_names : tuple[Any]
            List of model score names.
        thresholds : dict[str, float]
            List of thresholds for the model scores, will be sorted into decreasing order.
        per_context : bool, optional
            If scores should be grouped by context, by default None, in which case this checkbox is not shown.
        """
        self.title = html_title("Model Options")
        self.target_list = Dropdown(
            options=target_names,
            value=target_names[0],
            description="Target Column",
            style=WIDE_LABEL_STYLE,
            disabled=len(target_names) == 1,
        )
        self.score_list = Dropdown(
            options=score_names,
            value=score_names[0],
            description="Score Column",
            style=WIDE_LABEL_STYLE,
            disabled=len(score_names) == 1,
        )

        self.target_list.observe(self._on_value_change, "value")
        self.score_list.observe(self._on_value_change, "value")
        children = [self.title, self.target_list, self.score_list]

        if thresholds:
            thresholds = {
                k: v for k, v in sorted(thresholds.items(), key=lambda x: x[1], reverse=True)
            }  # decreasing order
            self.threshold_list = MonotonicProbabilitySliderListWidget(
                names=tuple(thresholds.keys()), value=tuple(thresholds.values()), ascending=False
            )
            children.append(self.threshold_list)
            self.threshold_list.observe(self._on_value_change, "value")
        else:
            self.threshold_list = None

        if per_context is not None:
            self.per_context_checkbox = _combine_scores_checkbox(per_context)
            children.append(self.per_context_checkbox)
            self.per_context_checkbox.observe(self._on_value_change, "value")
        else:
            self.per_context_checkbox = None

        super().__init__(
            children=children,
            layout=Layout(align_items="flex-start", flex="0 0 auto"),
        )
        self._on_value_change()
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        self.target_list.disabled = len(self.target_list.options) == 1 or disabled
        self.score_list.disabled = len(self.score_list.options) == 1 or disabled
        if self.threshold_list:
            self.threshold_list.disabled = disabled
        if self.per_context_checkbox:
            self.per_context_checkbox.disabled = disabled

    def _on_value_change(self, change=None):
        self.value = {
            "target": self.target,
            "score": self.score,
            "thresholds": self.thresholds,
            "group_scores": self.group_scores,
        }

    @property
    def target(self) -> str:
        """Target column descriptor"""
        return self.target_list.value

    @property
    def score(self) -> str:
        """Score column descriptor"""
        return self.score_list.value

    @property
    def thresholds(self) -> tuple[float]:
        """Thresholds for the score"""
        if self.threshold_list:
            return self.threshold_list.value

    @property
    def group_scores(self) -> bool:
        """If scores should be grouped by context."""
        if self.per_context_checkbox:
            return self.per_context_checkbox.value


class ModelScoreComparisonOptionsWidget(VBox, ValueWidget):
    value = traitlets.Dict(help="The selected values for the model options and scores to compare.")

    def __init__(
        self,
        target_names: tuple[Any],
        score_names: tuple[Any],
        per_context: Optional[bool] = None,
    ):
        """
        Widget for model based options, including scores to compare.

        Parameters
        ----------
        target_names : tuple[Any]
            List of target column names.
        score_names : tuple[Any]
            List of model score names.
        per_context : bool, optional
            If scores should be grouped by context, by default None, in which case this checkbox is not shown.
        """
        self.title = html_title("Model Options")
        self.target_list = Dropdown(
            options=target_names,
            value=target_names[0],
            description="Target Column",
            style=WIDE_LABEL_STYLE,
        )
        self.score_list = MultiselectDropdownWidget(
            options=score_names, value=score_names[0:2], title="Compare Scores"
        )

        self.target_list.observe(self._on_value_change, "value")
        self.score_list.observe(self._on_value_change, "value")
        children = [
            self.title,
            VBox(children=[self.target_list, self.score_list], layout=Layout(align_items="flex-end")),
        ]

        if per_context is not None:
            self.per_context_checkbox = _combine_scores_checkbox(per_context)
            children.append(self.per_context_checkbox)
            self.per_context_checkbox.observe(self._on_value_change, "value")
        else:
            self.per_context_checkbox = None

        super().__init__(
            children=children,
            layout=Layout(align_items="flex-start", flex="0 0 auto"),
        )
        self._on_value_change()
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        self.target_list.disabled = len(self.target_list.options) == 1 or disabled
        self.score_list.disabled = disabled
        if self.per_context_checkbox:
            self.per_context_checkbox.disabled = disabled

    def _on_value_change(self, change=None):
        self.value = {
            "target": self.target,
            "scores": self.scores,
            "group_scores": self.group_scores,
        }

    @property
    def target(self) -> str:
        """Target column descriptor"""
        return self.target_list.value

    @property
    def scores(self) -> tuple[str]:
        """Score column descriptor"""
        return self.score_list.value

    @property
    def group_scores(self) -> bool:
        """If scores should be grouped by context."""
        if self.per_context_checkbox:
            return self.per_context_checkbox.value


class ModelScoreComparisonAndCohortsWidget(Box, ValueWidget):
    value = traitlets.Dict(help="The selected values for the cohorts and model options.")

    def __init__(
        self,
        cohort_groups: dict[str, tuple[Any]],
        target_names: tuple[Any],
        score_names: tuple[Any],
        per_context: bool = False,
    ):
        """
        Widget for model based options and cohort selection, including scores to compare.

        Parameters
        ----------
        cohort_groups : dict[str, tuple[Any]]
            cohort columns and groupings
        target_names : tuple[Any]
            model target columns
        score_names : tuple[Any]
            model score columns
        per_context : bool, optional
            if scores should be grouped by context, by default False
        """
        self.cohort_list = MultiSelectionListWidget(options=cohort_groups, title="Cohort Filter")
        self.model_options = ModelScoreComparisonOptionsWidget(target_names, score_names, per_context)
        self.cohort_list.observe(self._on_value_change, "value")
        self.model_options.observe(self._on_value_change, "value")

        super().__init__(children=[self.model_options, self.cohort_list], layout=BOX_GRID_LAYOUT)

        self._on_value_change()
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        self.cohort_list.disabled = disabled
        self.model_options.disabled = disabled

    def _on_value_change(self, change=None):
        self.value = {
            "cohorts": self.cohort_list.value,
            "model_options": self.model_options.value,
        }

    @property
    def cohorts(self) -> dict[str, tuple[str]]:
        """Selected cohorts"""
        return self.cohort_list.value

    @property
    def target(self) -> str:
        """Target column descriptor"""
        return self.model_options.target

    @property
    def scores(self) -> tuple[str]:
        """Score column descriptors"""
        return self.model_options.scores

    @property
    def group_scores(self) -> bool:
        """If scores should be grouped by context."""
        return self.model_options.group_scores


class ModelTargetComparisonOptionsWidget(VBox, ValueWidget):
    value = traitlets.Dict(help="The selected values for the model options and targets to evaluate.")

    def __init__(
        self,
        target_names: tuple[Any],
        score_names: tuple[Any],
        per_context: Optional[bool] = None,
    ):
        """
        Widget for model based options, including multiple targets.

        Parameters
        ----------
        target_names : tuple[Any]
            List of target names
        score_names : tuple[Any]
            List of model score names
        per_context : bool, optional
            If scores should be grouped by context, by default None, in which case this checkbox is not shown.
        """
        self.title = html_title("Model Options")
        self.target_list = MultiselectDropdownWidget(
            options=target_names, value=target_names[0:2], title="Compare Targets"
        )
        self.score_list = Dropdown(
            options=score_names,
            value=score_names[0],
            description="Score Column",
            style=WIDE_LABEL_STYLE,
        )
        self.target_list.observe(self._on_value_change, "value")
        self.score_list.observe(self._on_value_change, "value")
        children = [
            self.title,
            VBox(children=[self.target_list, self.score_list], layout=Layout(align_items="flex-end")),
        ]

        if per_context is not None:
            self.per_context_checkbox = _combine_scores_checkbox(per_context)
            children.append(self.per_context_checkbox)
            self.per_context_checkbox.observe(self._on_value_change, "value")
        else:
            self.per_context_checkbox = None

        super().__init__(
            children=children,
            layout=Layout(align_items="flex-start", flex="0 0 auto"),
        )
        self._on_value_change()
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        self.target_list.disabled = disabled
        self.score_list.disabled = len(self.score_list.options) == 1 or disabled
        if self.per_context_checkbox:
            self.per_context_checkbox.disabled = disabled

    def _on_value_change(self, change=None):
        self.value = {
            "targets": self.targets,
            "score": self.score,
            "group_scores": self.group_scores,
        }

    @property
    def targets(self) -> str:
        """Target column descriptors"""
        return self.target_list.value

    @property
    def score(self) -> tuple[str]:
        """Score column descriptor"""
        return self.score_list.value

    @property
    def group_scores(self) -> bool:
        """If scores should be grouped by context."""
        if self.per_context_checkbox:
            return self.per_context_checkbox.value


class ModelTargetComparisonAndCohortsWidget(Box, ValueWidget):
    value = traitlets.Dict(help="The selected values for the cohorts and model options and targets.")

    def __init__(
        self,
        cohort_groups: dict[str, tuple[Any]],
        target_names: tuple[Any],
        score_names: tuple[Any],
        per_context: bool = False,
    ):
        """
        Widget for model based options and cohort selection, including multiple targets.

        Parameters
        ----------
        cohort_groups : dict[str, tuple[Any]]
            Cohort columns and groupings
        target_names : tuple[Any]
            List of model target descriptors
        score_names : tuple[Any]
            List of model score names
        per_context : bool, optional
            If scores should be grouped by context, by default False.
        """
        self.cohort_list = MultiSelectionListWidget(options=cohort_groups, title="Cohort Filter")
        self.model_options = ModelTargetComparisonOptionsWidget(target_names, score_names, per_context)
        self.cohort_list.observe(self._on_value_change, "value")
        self.model_options.observe(self._on_value_change, "value")

        super().__init__(children=[self.model_options, self.cohort_list], layout=BOX_GRID_LAYOUT)

        self._on_value_change()
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        self.cohort_list.disabled = disabled
        self.model_options.disabled = disabled

    def _on_value_change(self, change=None):
        self.value = {
            "cohorts": self.cohort_list.value,
            "model_options": self.model_options.value,
        }

    @property
    def cohorts(self) -> dict[str, tuple[str]]:
        """Selected cohorts"""
        return self.cohort_list.value

    @property
    def targets(self) -> str:
        """Target column descriptors"""
        return self.model_options.targets

    @property
    def score(self) -> tuple[str]:
        """Score column descriptor"""
        return self.model_options.score

    @property
    def group_scores(self) -> bool:
        """If scores should be grouped by context."""
        return self.model_options.group_scores


class ModelOptionsAndCohortsWidget(Box, ValueWidget):
    value = traitlets.Dict(help="The selected values for the cohorts and model options.")

    def __init__(
        self,
        cohort_groups: dict[str, tuple[Any]],
        target_names: tuple[Any],
        score_names: tuple[Any],
        thresholds: dict[str, float],
        per_context: bool = False,
    ):
        """
        Widget for model based options and cohort selection.

        Parameters
        ----------
        cohort_groups : dict[str, tuple[Any]]
            cohort columns and groupings
        target_names : tuple[Any]
            model target columns
        score_names : tuple[Any]
            model score columns
        thresholds : dict[str, float]
            model thresholds
        per_context : bool, optional
            If scores should be grouped by context, by default False.
        """
        self.cohort_list = MultiSelectionListWidget(options=cohort_groups, title="Cohort Filter")
        self.model_options = ModelOptionsWidget(target_names, score_names, thresholds, per_context)
        self.cohort_list.observe(self._on_value_change, "value")
        self.model_options.observe(self._on_value_change, "value")

        super().__init__(children=[self.model_options, self.cohort_list], layout=BOX_GRID_LAYOUT)

        self._on_value_change()
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        self.cohort_list.disabled = disabled
        self.model_options.disabled = disabled

    def _on_value_change(self, change=None):
        self.value = {
            "cohorts": self.cohort_list.value,
            "model_options": self.model_options.value,
        }

    @property
    def cohorts(self) -> dict[str, tuple[str]]:
        """Selected cohorts"""
        return self.cohort_list.value

    @property
    def target(self) -> str:
        """Target column descriptor"""
        return self.model_options.target

    @property
    def score(self) -> str:
        """Score column descriptor"""
        return self.model_options.score

    @property
    def thresholds(self) -> tuple[float]:
        """Score thresholds"""
        return self.model_options.thresholds

    @property
    def group_scores(self) -> bool:
        """If scores should be grouped by context."""
        return self.model_options.group_scores


class ModelOptionsAndCohortGroupWidget(Box, ValueWidget):
    value = traitlets.Dict(help="The selected values for the cohorts and model options.")

    def __init__(
        self,
        cohort_groups: dict[str, tuple[Any]],
        target_names: tuple[Any],
        score_names: tuple[Any],
        thresholds: dict[str, float],
        per_context: bool = False,
    ):
        """
        Selection widget for model options and cohort groups, when displaying results for
        individual values within a cohort attribute.

        Parameters
        ----------
        cohort_groups : dict[str, tuple[Any]]
            groups of cohort columns and values
        target_names : tuple[Any]
            model target columns
        score_names : tuple[Any]
            model score columns
        thresholds : dict[str, float]
            thresholds for the model scores
        per_context : bool, optional
            If scores should be grouped by context, by default False.
        """
        self.cohort_list = DisjointSelectionListsWidget(options=cohort_groups, title="Cohort Filter", select_all=True)
        self.model_options = ModelOptionsWidget(target_names, score_names, thresholds, per_context)
        self.cohort_list.observe(self._on_value_change, "value")
        self.model_options.observe(self._on_value_change, "value")

        super().__init__(children=[self.model_options, self.cohort_list], layout=BOX_GRID_LAYOUT)

        self._on_value_change()
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        self.cohort_list.disabled = disabled
        self.model_options.disabled = disabled

    def _on_value_change(self, change=None):
        self.value = {
            "cohorts": self.cohort_list.value,
            "model_options": self.model_options.value,
        }

    @property
    def cohort(self) -> str:
        """Cohort column descriptor"""
        return self.cohort_list.value[0]

    @property
    def cohort_groups(self) -> tuple[Any]:
        """Cohort groups"""
        return self.cohort_list.value[1]

    @property
    def target(self) -> str:
        """Target column descriptor"""
        return self.model_options.target

    @property
    def score(self) -> str:
        """Score column descriptor"""
        return self.model_options.score

    @property
    def thresholds(self) -> tuple[float]:
        """Score thresholds"""
        return self.model_options.thresholds

    @property
    def group_scores(self) -> bool:
        """If scores should be grouped by context."""
        return self.model_options.group_scores


class ModelInterventionOptionsWidget(VBox, ValueWidget):
    value = traitlets.Dict(help="The selected values for the intervention options")

    def __init__(
        self,
        outcome_names: tuple[Any] = None,
        intervention_names: tuple[Any] = None,
        reference_time_names: tuple[Any] = None,
    ):
        """
        Widget for selecting an intervention and outcome for a model implementation.

        Parameters
        ----------
        outcome_names : tuple[Any], optional
            Names of outcome columns, by default None.
        intervention_names : tuple[Any], optional
            Names of intervention columns, by default None.
        reference_time_names : tuple[Any], optional
            Name for the reference time to align patients, by default None.
        """
        self.title = html_title("Model Options")
        self.outcome_list = Dropdown(options=outcome_names, value=outcome_names[0], description="Outcome")
        self.intervention_list = Dropdown(
            options=intervention_names, value=intervention_names[0], description="Intervention"
        )
        self.reference_time_list = Dropdown(
            options=reference_time_names, value=reference_time_names[0], description="Reference Time"
        )

        children = [self.title, self.outcome_list, self.intervention_list, self.reference_time_list]

        super().__init__(
            children=children,
            layout=Layout(align_items="flex-start", flex="0 0 auto"),
        )

        self.outcome_list.observe(self._on_value_change, "value")
        self.intervention_list.observe(self._on_value_change, "value")
        self.reference_time_list.observe(self._on_value_change, "value")

        self._on_value_change()
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        self.outcome_list.disabled = disabled
        self.intervention_list.disabled = disabled
        self.reference_time_list.disabled = disabled

    def _on_value_change(self, change=None):
        self.value = {
            "outcome": self.outcome,
            "intervention": self.intervention,
            "reference_time": self.reference_time,
        }

    @property
    def outcome(self) -> str:
        """Outcome column descriptor"""
        return self.outcome_list.value

    @property
    def intervention(self) -> str:
        """Intervention column descriptor"""
        return self.intervention_list.value

    @property
    def reference_time(self) -> str:
        """Reference time column descriptor"""
        return self.reference_time_list.value


class ModelInterventionAndCohortGroupWidget(Box, ValueWidget):
    value = traitlets.Dict(help="The selected values for the cohorts and model options")

    def __init__(
        self,
        cohort_groups: dict[str, tuple[Any]],
        outcome_names: tuple[Any] = None,
        intervention_names: tuple[Any] = None,
        reference_time_names: tuple[Any] = None,
    ):
        """
        Widget for selecting interventions and outcomes across categories in a cohort group.

        Parameters
        ----------
        cohort_groups : dict[str, tuple[Any]]
            Cohort names and category values.
        outcome_names : tuple[Any], optional
            Outcome descriptors, by default None.
        intervention_names : tuple[Any], optional
            Intervention descriptors, by default None.
        reference_time_names : tuple[Any], optional
            Reference time descriptors, by default None.
        """
        self.cohort_list = DisjointSelectionListsWidget(options=cohort_groups, title="Cohort Filter", select_all=True)
        self.model_options = ModelInterventionOptionsWidget(outcome_names, intervention_names, reference_time_names)
        self.cohort_list.observe(self._on_value_change, "value")
        self.model_options.observe(self._on_value_change, "value")

        super().__init__(children=[self.model_options, self.cohort_list], layout=BOX_GRID_LAYOUT)

        self._on_value_change()
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        self.model_options.disabled = disabled
        self.cohort_list.disabled = disabled

    def _on_value_change(self, change=None):
        self.value = {
            "cohorts": self.cohort_list.value,
            "model_options": self.model_options.value,
        }

    @property
    def cohort(self) -> str:
        """Cohort column descriptor"""
        return self.cohort_list.value[0]

    @property
    def cohort_groups(self) -> tuple[Any]:
        """Cohort category values"""
        return self.cohort_list.value[1]

    @property
    def outcome(self) -> str:
        """Outcome column descriptor"""
        return self.model_options.outcome

    @property
    def intervention(self) -> str:
        """Intervention column descriptor"""
        return self.model_options.intervention

    @property
    def reference_time(self) -> str:
        """Reference time column descriptor"""
        return self.model_options.reference_time


# endregion
# region Exploration Widgets


class ExplorationWidget(VBox):
    """
    Parent class for model exploration widgets.
    """

    NO_CODE_STRING = "No plot generated."

    def __init__(
        self, title: str, option_widget: ValueWidget, plot_function: Callable[..., Any], initial_plot: bool = True
    ):
        """
        Parent class for a plot exploration widget.

        Parameters
        ----------
        title : str
            Widget title.
        option_widget : ValueWidget
            Widget that contains the options the plot_function.
        plot_function : Callable[..., Any]
            A function that generates content for display within the control.
        """
        layout = Layout(
            width="100%",
            height="min-content",
            border="solid 1px var(--jp-border-color1)",
            padding="var(--jp-cell-padding)",
        )
        title = HTML(value=f"""<h3 style="text-align: left; margin-top: 0px;">{title}</h3>""")
        self.center = Output(layout=Layout(height="max-content", max_width="1200px"))
        self.code_output = Output(layout=Layout(height="max-content", max_width="1200px"))
        self.data_output = Output(layout=Layout(height="max-content", max_width="1200px"))
        self.option_widget = option_widget
        self.plot_function = plot_function
        self.update_plot_widget = UpdatePlotWidget()
        super().__init__(
            children=[
                title,
                self.option_widget,
                self.update_plot_widget,
                self.code_output,
                self.data_output,
                self.center,
            ],
            layout=layout,
        )

        # show initial plot
        if initial_plot:
            self.update_plot(initial=True)
        else:
            self.current_plot_code = self.NO_CODE_STRING
            self.current_plot_data = None

        # attache event handlers
        self.option_widget.observe(self._on_option_change, "value")
        self.update_plot_widget.on_click(self._on_plot_button_click)
        self.update_plot_widget.on_toggle_code(self._on_toggle_code)
        self.update_plot_widget.on_toggle_data(self._on_toggle_data)

    @property
    def disabled(self) -> bool:
        """
        If the widget is disabled.
        """
        return False

    @property
    def show_code(self) -> bool:
        """
        If the widget should show the plot's code.
        """
        return self.update_plot_widget.show_code

    @show_code.setter
    def show_code(self, show_code: bool):
        self.update_plot_widget.show_code = show_code

    @property
    def show_data(self) -> bool:
        """
        If the widget should show the plot's data.
        """
        return self.update_plot_widget.show_data

    @show_data.setter
    def show_data(self, show_data: bool):
        self.update_plot_widget.show_data = show_data

    @staticmethod
    def _is_interactive_notebook() -> bool:
        ip = get_ipython()
        if ip is None:
            return False
        if os.environ.get("BUILDING_DOCS") == "1":
            return False
        return True

    def update_plot(self, initial: bool = False):
        if not initial:
            self.center.clear_output()

        if not initial or self._is_interactive_notebook():
            with self.center:
                plot = self._try_generate_plot()
                display(plot)
        else:
            plot = self._try_generate_plot()
            self.center.append_display_data(plot)
        self.update_plot_widget.disabled = True

    def _try_generate_plot(self) -> Any:
        """Attempt to generate the plot. Displaying the error as HTML"""
        try:
            plot_args, plot_kwargs = self.generate_plot_args()
            self.current_plot_code = self.generate_plot_code(plot_args, plot_kwargs)
            # The plot function will now return a tuple (plot, data)
            plot, self.current_plot_data = self.plot_function(*plot_args, **plot_kwargs)
        except Exception as e:
            import traceback

            self.current_plot_data = None
            plot = HTML(f"<div><h3>Exception: <pre>{e}</pre> </h3><pre>{traceback.format_exc()}</pre></div>")
        return plot

    def _on_plot_button_click(self, button=None):
        """Handle for the update plot button."""
        self.option_widget.disabled = True
        self.update_plot()
        self._on_toggle_code(self.show_code)
        self._on_toggle_data(self.show_data)
        self.option_widget.disabled = False

    def generate_plot_code(self, plot_args: tuple = None, plot_kwargs: dict = None) -> str:
        """String representation of the plot's code."""
        plot_module = self.plot_function.__module__
        plot_method = self.plot_function.__name__

        match plot_module:
            case "__main__":
                method_string = plot_method
            case "seismometer.api":
                method_string = f"sm.{plot_method}"
            case _:
                method_string = f"{plot_module}.{plot_method}"

        args_string = ""
        if plot_args:
            args_string += ", ".join([repr(x) for x in plot_args])
        if plot_kwargs:
            if args_string:
                args_string += ", "
            args_string += ", ".join([f"{k}={repr(v)}" for k, v in plot_kwargs.items()])

        return f"{method_string}({args_string})"

    def _on_toggle_code(self, show_code: bool):
        """Handle for the toggle code checkbox."""
        self.code_output.clear_output()
        if not show_code:
            return

        highlighted_code = HTML(
            f"<span style='user-select: none;'>Plot code: </span><code>{self.current_plot_code}</code>"
        )
        highlighted_code.add_class("jp-RenderedHTMLCommon")
        with self.code_output:
            display(highlighted_code)

    def _on_toggle_data(self, show_data: bool):
        """Handle for the toggle data checkbox."""
        self.data_output.clear_output()
        if not show_data:
            return

        if self.current_plot_data is not None:
            with self.data_output:
                display(self.current_plot_data)

    def _on_option_change(self, change=None):
        """Enable the plot to be updated."""
        self.update_plot_widget.disabled = self.disabled

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Override this method with code to show a plot."""
        raise NotImplementedError("Subclasses must implement this method")


class ExplorationSubpopulationWidget(ExplorationWidget):
    """
    A widget for exploring subpopulations based on cohort selection.
    """

    def __init__(
        self,
        title: str,
        plot_function: Callable[..., Any],
    ):
        """
        Exploration widget for a subpopulation defined by a cohort selection.

        Parameters
        ----------
        title : str
            Title of the control..
        plot_function : Callable[..., Any]
            Expected to have the following signature:

            .. code:: python

                def plot_function(cohort_dict: dict[str,tuple[Any]]) -> Any
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        cohort_groups = sg.available_cohort_groups
        option_widget = MultiSelectionListWidget(options=cohort_groups, title="Cohort Filter")
        option_widget.layout = BOX_GRID_LAYOUT
        super().__init__(
            title=title,
            option_widget=option_widget,
            plot_function=plot_function,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the model evaluation plot."""
        args = (self.option_widget.value,)
        kwargs = {
            # empty dictionary
        }
        return args, kwargs


class ExplorationModelSubgroupEvaluationWidget(ExplorationWidget):
    """
    A widget for exploring the model performance of a cohort.
    """

    def __init__(
        self,
        title: str,
        plot_function: Callable[..., Any],
    ):
        """
        Exploration widget for model evaluation, showing a plot for a given target,
        score, threshold, and cohort selection.

        Parameters
        ----------
        title : str
            Title of the control.
        plot_function : Callable[..., Any]
            Expected to have the following signature:

            .. code:: python

                def plot_function(
                    cohort_dict: dict[str,tuple[Any]],
                    target: str,
                    score: str,
                    thresholds: tuple[float],
                    *, per_context: bool) -> Any
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        thresholds = {f"Threshold {k}": v for k, v in enumerate(sorted(sg.thresholds, reverse=True), 1)}
        super().__init__(
            title=title,
            option_widget=ModelOptionsAndCohortsWidget(
                sg.available_cohort_groups, sg.target_cols, sg.output_list, thresholds, per_context=False
            ),
            plot_function=plot_function,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the model evaluation plot."""
        args = (
            self.option_widget.cohorts,
            self.option_widget.target,
            self.option_widget.score,
            list(self.option_widget.thresholds.values()),
        )
        kwargs = {
            "per_context": self.option_widget.group_scores,
        }
        return args, kwargs


class ExplorationCohortSubclassEvaluationWidget(ExplorationWidget):
    """
    A widget for exploring the model performance based on the subgroups of a cohort column.
    """

    def __init__(
        self,
        title: str,
        plot_function: Callable[..., Any],
        *,
        ignore_grouping: bool = False,
        threshold_handling: Literal["all", "max", "min", None] = "all",
    ):
        """
        Exploration widget for model evaluation, showing a plot for a given target,
        score, threshold, broken down across labels in a cohort column.

        Parameters
        ----------
        title : str
            Title of the control.
        plot_function : Callable[..., Any]
            Expected to have the following signature:

            .. code:: python

                def plot_function(
                    cohorts_col: str
                    cohort_subgroups: tuple[Any]
                    target: str,
                    score: str,
                    thresholds: tuple[float],
                    *, per_context: bool) -> Any
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        match threshold_handling:
            case "all":
                self.thresholds = {f"Threshold {k}": v for k, v in enumerate(sorted(sg.thresholds, reverse=True), 1)}
            case "max":
                self.thresholds = {"Threshold": max(sg.thresholds)}
            case "min":
                self.thresholds = {"Threshold": min(sg.thresholds)}
            case _:
                self.thresholds = None

        self.threshold_handling = threshold_handling
        self.ignore_grouping = ignore_grouping

        option_widget = ModelOptionsAndCohortGroupWidget(
            sg.available_cohort_groups,
            sg.target_cols,
            sg.output_list,
            thresholds=self.thresholds,
            per_context=False if not self.ignore_grouping else None,
        )
        super().__init__(title=title, option_widget=option_widget, plot_function=plot_function)

    @property
    def disabled(self):
        return not self.option_widget.cohort_groups

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the model evaluation plot."""

        args = [
            self.option_widget.cohort,
            self.option_widget.cohort_groups,
            self.option_widget.target,
            self.option_widget.score,
        ]
        match self.threshold_handling:
            case "all":
                args.append(list(self.option_widget.thresholds.values()))
            case "max" | "min":
                args.append(self.option_widget.thresholds["Threshold"])

        kwargs = {}
        if not self.ignore_grouping:
            kwargs["per_context"] = self.option_widget.group_scores
        return args, kwargs


class ExplorationCohortOutcomeInterventionEvaluationWidget(ExplorationWidget):
    """
    A widget for exploring the model outcomes and interventions based on the subgroups of a cohort column.
    """

    def __init__(
        self,
        title: str,
        plot_function: Callable[..., Any],
    ):
        """
        Exploration widget for plotting of interventions and outcomes across categories in a cohort group.

        Parameters
        ----------
        title : str
            Title of the control.
        plot_function : Callable[..., Any]
            Expected to have the following signature:

            .. code:: python

                def plot_function(
                    cohorts_col: str
                    cohort_subgroups: tuple[Any]
                    outcome: str,
                    intervention: str,
                    reference_time: str) -> Any
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        reference_times = [sg.predict_time]
        if sg.comparison_time and sg.comparison_time != sg.predict_time:
            reference_times.append(sg.comparison_time)

        super().__init__(
            title,
            option_widget=ModelInterventionAndCohortGroupWidget(
                sg.available_cohort_groups,
                tuple(sg.config.outcomes.keys()),
                tuple(sg.config.interventions.keys()),
                reference_times,
            ),
            plot_function=plot_function,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the model evaluation plot."""
        args = [
            self.option_widget.cohort,
            self.option_widget.cohort_groups,
            self.option_widget.outcome,
            self.option_widget.intervention,
            self.option_widget.reference_time,
        ]
        kwargs = {}
        return args, kwargs


class ExplorationScoreComparisonByCohortWidget(ExplorationWidget):
    """
    A widget to explore different model scores based on a cohort selection.
    """

    def __init__(self, title: str, plot_function: Callable[..., Any]):
        """
        Exploration widget for model score comparison, showing a plot for a given target
        and cohort selection, across different scores.

        Parameters
        ----------
        title : str
            Title of the control.
        plot_function : Callable[..., Any]
            Expected to have the following signature:

            .. code:: python

                def plot_function(
                    cohort_dict: dict[str,tuple[Any]],
                    target: str,
                    scores: tuple[str],
                    *, per_context: bool) -> Any
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        super().__init__(
            title,
            option_widget=ModelScoreComparisonAndCohortsWidget(
                sg.available_cohort_groups, sg.target_cols, sg.output_list, per_context=False
            ),
            plot_function=plot_function,
        )

    @property
    def disabled(self):
        return len(self.option_widget.scores) < 2

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the model evaluation plot."""
        args = [
            self.option_widget.cohorts,
            self.option_widget.target,
            self.option_widget.scores,
        ]
        kwargs = {"per_context": self.option_widget.group_scores}
        return args, kwargs


class ExplorationTargetComparisonByCohortWidget(ExplorationWidget):
    """
    A widget to explore different model targets based on a cohort selection.
    """

    def __init__(self, title: str, plot_function: Callable[..., Any]):
        """
        Exploration widget for model target comparison, showing a plot for a given scores
        and cohort selection, across different targets.

        Parameters
        ----------
        title : str
            Title of the control.
        plot_function : Callable[..., Any]
            Expected to have the following signature:

            .. code:: python

                def plot_function(
                    cohort_dict: dict[str,tuple[Any]],
                    targets: tuple[str],
                    score: str,
                    *, per_context: bool) -> Any
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        super().__init__(
            title,
            option_widget=ModelTargetComparisonAndCohortsWidget(
                sg.available_cohort_groups, sg.target_cols, sg.output_list, per_context=False
            ),
            plot_function=plot_function,
        )

    @property
    def disabled(self):
        return len(self.option_widget.targets) < 2

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the model evaluation plot."""
        args = [
            self.option_widget.cohorts,
            self.option_widget.targets,
            self.option_widget.score,
        ]
        kwargs = {"per_context": self.option_widget.group_scores}
        return args, kwargs


# endregion


# region Binary Model Metric Exploration Widget


class BinaryModelMetricOptions(Box, ValueWidget):
    value = traitlets.Dict(help="The selected values for the audit and model options")

    def __init__(
        self,
        metrics_generator: MetricGenerator,
        cohort_groups: dict[str, tuple[Any]],
        target_names: tuple[Any],
        score_names: tuple[Any],
        per_context: bool = True,
        default_metrics: Optional[tuple[str]] = None,
    ):
        """
        Widget for selecting interventions and outcomes across categories in a cohort group.

        Parameters
        ----------
        metrics_generator: MetricGenerator
            class to generate metrics of interest
        cohort_groups : dict[str, tuple[Any]]
            cohort columns and groupings
        target_names : tuple[Any]
            model target columns
        score_names : tuple[Any]
            model score columns
        per_context : bool, optional
            if scores should be grouped by context, by default True
        default_metrics : tuple[str], optional
            list of fairness metrics to display, if None (default) will use those from metric_generator
        """
        metrics = metrics_generator.metric_names
        default_metrics = default_metrics or metrics_generator.default_metrics
        self.metric_list = MultiselectDropdownWidget(options=metrics, value=default_metrics, title="Metrics")
        self.metric_list.layout = Layout(width="min-content", align_self="flex-end")

        self.model_options = ModelOptionsWidget(target_names, score_names, per_context=per_context)
        self.model_options.children = list(self.model_options.children) + [self.metric_list]
        self.cohort_list = MultiSelectionListWidget(options=cohort_groups, title="Cohort Filter")

        super().__init__(children=[self.model_options, self.cohort_list], layout=BOX_GRID_LAYOUT)

        self.metric_list.observe(self._on_value_change, "value")
        self.model_options.observe(self._on_value_change, "value")
        self.cohort_list.observe(self._on_value_change, "value")

        self._on_value_change()
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        self.metric_list.disabled = disabled
        self.cohort_list.disabled = disabled
        self.model_options.disabled = disabled

    def _on_value_change(self, change=None):
        self.value = {
            "metrics": self.metric_list.value,
            "cohort": self.cohort_list.value,
            "model_options": self.model_options.value,
        }

    @property
    def metrics(self) -> tuple[str]:
        """selected metrics"""
        return self.metric_list.value

    @property
    def cohorts(self) -> tuple[str]:
        """selected cohorts"""
        return self.cohort_list.value

    @property
    def target(self) -> str:
        """target column descriptor"""
        return self.model_options.target

    @property
    def score(self) -> str:
        """Score column descriptor"""
        return self.model_options.score

    @property
    def group_scores(self) -> bool:
        """If scores should be grouped by context"""
        return self.model_options.group_scores


class ExplorationMetricWidget(ExplorationWidget):
    """
    A widget to explore different model metrics for a score, target, and cohort
    """

    def __init__(
        self,
        title: str,
        metric_generator: MetricGenerator,
        plot_function: Callable[..., Any],
        *,
        default_metrics: Optional[tuple[str]] = None,
    ):
        """
        Exploration widget for binary model metrics, showing a plot for a given target/score
        and cohort selection.

        Parameters
        ----------
        title : str
            title of the control
        metric_generator: MetricGenerator
            used to select metrics and compute them.
        plot_function : Callable[..., Any]
            Expected to have the following signature:

            .. code:: python

                def plot_function(
                    metric_generator: MetricGenerator,
                    metrics: tuple[str],
                    cohort_dict: dict[str,tuple[Any]],
                    target: tuple[str],
                    score: str,
                    *, per_context: bool) -> Any
        default_metrics : tuple[str], optional
            list of fairness metrics to display, if None (default) will use those from metric_generator
        """
        from seismometer.seismogram import Seismogram

        self.metric_generator = metric_generator
        sg = Seismogram()
        super().__init__(
            title,
            option_widget=BinaryModelMetricOptions(
                metric_generator,
                sg.available_cohort_groups,
                sg.target_cols,
                sg.output_list,
                per_context=False,
                default_metrics=default_metrics,
            ),
            plot_function=plot_function,
        )

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """Generates the plot arguments for the model evaluation plot"""
        args = [
            self.metric_generator,
            self.option_widget.metrics,
            self.option_widget.cohorts,
            self.option_widget.target,
            self.option_widget.score,
        ]
        kwargs = {"per_context": self.option_widget.group_scores}
        return args, kwargs


# endregion
