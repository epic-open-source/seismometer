import logging
from functools import wraps
from typing import Any

import traitlets
from IPython.display import display
from ipywidgets import HTML, Box, Button, Checkbox, Dropdown, Layout, Output, ValueWidget, VBox

from seismometer.core.decorators import export

from .selection import DisjointSelectionListsWidget, MultiSelectionListWidget
from .styles import BOX_GRID_LAYOUT, WIDE_LABEL_STYLE
from .thresholds import MonotonicPercentSliderListWidget

logger = logging.getLogger("seismometer")


class UpdatePlotWidget(Box):
    UPDATE_PLOTS = "Update Plots"
    UPDATING_PLOTS = "Updating ..."

    def __init__(self):
        self.code_checkbox = Checkbox(
            value=False,
            description="show code",
            disabled=False,
            indent=False,
            tooltip="Show the code used to generate the plot",
            layout=Layout(margin="var(--jp-widgets-margin) var(--jp-widgets-margin) var(--jp-widgets-margin) 10px;"),
        )

        self.plot_button = Button(description=self.UPDATE_PLOTS, button_style="primary", width="max-content")
        layout = Layout(align_items="flex-start")
        children = [self.plot_button, self.code_checkbox]
        super().__init__(layout=layout, children=children)

    @property
    def show_code(self):
        return self.code_checkbox.value

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

    def trigger(self):
        self.plot_button.click()


class ExlorationWidget(VBox):
    def __init__(
        self,
        title: str,
        option_widget: ValueWidget,
    ):
        layout = Layout(
            width="100%",
            height="min-content",
            border="solid 1px var(--jp-border-color1)",
            padding="var(--jp-cell-padding)",
        )
        title = HTML(value=f"""<h3 style="text-align: left; margin-top: 0px;">{title}</h3>""")
        self.center = Output(layout=Layout(height="max-content", max_width="2000px"))
        self.option_widget = option_widget
        self.update_plot_widget = UpdatePlotWidget()
        super().__init__(children=[title, self.option_widget, self.update_plot_widget, self.center], layout=layout)

        # attach button handler and show initial plot
        self.update_plot_widget.on_click(self._on_plot_button_click)
        self.update_plot_widget.trigger()
        self.option_widget.observe(self._on_option_change, "value")

    @property
    def disabled(self):
        return False

    @property
    def show_code(self) -> bool:
        return self.update_plot_widget.show_code

    def _on_plot_button_click(self, button=None):
        self.center.clear_output()
        with self.center:
            self.update_plot()

    def _on_option_change(self, change=None):
        self.update_plot_widget.disabled = self.disabled

    def update_plot(self):
        raise NotImplementedError("Subclasses must implement this method")


class ModelOptionsWidget(VBox, ValueWidget):
    value = traitlets.Dict(help="The selected values for the slider list")

    def __init__(
        self,
        target_names: tuple[Any],
        score_names: tuple[Any],
        thresholds: dict[str, float],
        per_context: bool = False,
    ):
        self.title = HTML('<h4 style="text-align: left; margin: 0px;">Model Options</h4>')
        self.target_list = Dropdown(
            options=target_names,
            value=target_names[0],
            description="Target Column",
            style=WIDE_LABEL_STYLE,
        )
        self.score_list = Dropdown(
            options=score_names, value=score_names[0], description="Score Column", style=WIDE_LABEL_STYLE
        )

        self.target_list.observe(self._on_value_change, "value")
        self.score_list.observe(self._on_value_change, "value")
        children = [self.title, self.target_list, self.score_list]

        if thresholds:
            self.threshold_list = MonotonicPercentSliderListWidget(
                names=tuple(thresholds.keys()), value=tuple(thresholds.values()), increasing=False
            )
            children.append(self.threshold_list)
            self.threshold_list.observe(self._on_value_change, "value")
        else:
            self.threshold_list = None

        if per_context:
            self.per_context_checkbox = Checkbox(
                value=False,
                description="combine scores",
                disabled=False,
                tooltip="Combine scores by taking the maximum score in the target window",
                style=WIDE_LABEL_STYLE,
            )
            children.append(self.per_context_checkbox)
            self.per_context_checkbox.observe(self._on_value_change, "value")
        else:
            self.per_context_checkbox = None

        super().__init__(
            children=children,
            layout=Layout(align_items="flex-start", flex="0 0 auto"),
        )

    def _on_value_change(self, change=None):
        self.value = {
            "target": self.target,
            "score": self.score,
            "thresholds": self.thresholds,
            "group_scores": self.group_scores,
        }

    @property
    def target(self):
        return self.target_list.value

    @property
    def score(self):
        return self.score_list.value

    @property
    def thresholds(self):
        if self.threshold_list:
            return self.threshold_list.value

    @property
    def group_scores(self):
        if self.per_context_checkbox:
            return self.per_context_checkbox.value


class ModelOptionsAndCohortsWidget(Box, ValueWidget):
    value = traitlets.Dict(help="The selected values for the cohorts and moedel options")

    def __init__(
        self,
        cohort_groups: dict[str, tuple[Any]],
        target_names: tuple[Any],
        score_names: tuple[Any],
        thresholds: dict[str, float],
        per_context: bool = False,
    ):
        self.cohort_list = MultiSelectionListWidget(options=cohort_groups, title="Cohort Filter")
        self.model_options = ModelOptionsWidget(target_names, score_names, thresholds, per_context)
        self.cohort_list.observe(self._on_value_change, "value")
        self.model_options.observe(self._on_value_change, "value")

        super().__init__(children=[self.model_options, self.cohort_list], layout=BOX_GRID_LAYOUT)

    def _on_value_change(self, change=None):
        self.value = {
            "cohorts": self.cohort_list.value,
            "model_options": self.model_options.value,
        }

    @property
    def cohorts(self):
        return self.cohort_list.value

    @property
    def target(self):
        return self.model_options.target

    @property
    def score(self):
        return self.model_options.score

    @property
    def thresholds(self):
        return self.model_options.thresholds

    @property
    def group_scores(self):
        return self.model_options.group_scores


class ModelOptionsAndCohortGroupWidget(Box, ValueWidget):
    value = traitlets.Dict(help="The selected values for the cohorts and moedel options")

    def __init__(
        self,
        cohort_groups: dict[str, tuple[Any]],
        target_names: tuple[Any],
        score_names: tuple[Any],
        thresholds: dict[str, float],
        per_context: bool = False,
    ):
        self.cohort_list = DisjointSelectionListsWidget(options=cohort_groups, title="Cohort Filter", select_all=True)
        self.model_options = ModelOptionsWidget(target_names, score_names, thresholds, per_context)
        self.cohort_list.observe(self._on_value_change, "value")
        self.model_options.observe(self._on_value_change, "value")

        super().__init__(children=[self.model_options, self.cohort_list], layout=BOX_GRID_LAYOUT)

    def _on_value_change(self, change=None):
        self.value = {
            "cohorts": self.cohort_list.value,
            "model_options": self.model_options.value,
        }

    @property
    def cohort(self):
        return self.cohort_list.value[0]

    @property
    def cohort_groups(self):
        return self.cohort_list.value[1]

    @property
    def target(self):
        return self.model_options.target

    @property
    def score(self):
        return self.model_options.score

    @property
    def thresholds(self):
        return self.model_options.thresholds

    @property
    def group_scores(self):
        return self.model_options.group_scores


class ExplorationModelEvaluationWidget(ExlorationWidget):
    """
    A widget for exploring the model performance of a cohort.
    """

    def __init__(
        self,
        title: str,
        cohort_groups: dict[str, tuple[Any]],
        target_names: tuple[Any],
        score_names: tuple[Any],
        thresholds: dict[str, float],
        per_context: bool = False,
    ):
        """
        A widget for exploring the results of a cohort.

        Parameters
        ----------
        cohort : seismometer.data.cohort.Cohort
            The cohort to explore.
        plot_fn : Callable
            A function that takes a cohort and returns a plot.
        filter_fn : Optional[Callable], optional
            A function that takes a cohort and returns a filter, by default None.
        """
        super().__init__(
            title=title,
            option_widget=ModelOptionsAndCohortsWidget(
                cohort_groups, target_names, score_names, thresholds, per_context
            ),
        )


class ExplorationCohortSubclassEvaluationWidget(ExlorationWidget):
    """
    A widget for exploring the model performance based on the subgroups of a cohort column.
    """

    def __init__(
        self,
        title: str,
        cohort_groups: dict[str, tuple[Any]],
        target_names: tuple[Any],
        score_names: tuple[Any],
        thresholds: dict[str, float],
        per_context: bool = False,
    ):
        """
        A widget for exploring the results of a cohort's subgroups.

        Parameters
        ----------
        cohort : seismometer.data.cohort.Cohort
            The cohort to explore.
        plot_fn : Callable
            A function that takes a cohort and returns a plot.
        filter_fn : Optional[Callable], optional
            A function that takes a cohort and returns a filter, by default None.
        """
        option_widget = ModelOptionsAndCohortGroupWidget(
            cohort_groups, target_names, score_names, thresholds, per_context
        )
        super().__init__(title=title, option_widget=option_widget)

    @property
    def disabled(self):
        return not self.option_widget.cohort_groups


@export
class ExploreModelEvaluation(ExplorationModelEvaluationWidget):
    """
    A widget for exploring the model performance of a cohort.
    """

    def __init__(self):
        """
        A widget for exploring the results of a cohort.

        Parameters
        ----------
        cohort : seismometer.data.cohort.Cohort
            The cohort to explore.
        plot_fn : Callable
            A function that takes a cohort and returns a plot.
        filter_fn : Optional[Callable], optional
            A function that takes a cohort and returns a filter, by default None.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        thresholds = {f"Threshold {k}": v for k, v in enumerate(sorted(sg.thresholds), 1)}
        super().__init__(
            "Model Performance",
            sg.available_cohort_groups,
            sg.target_cols,
            sg.output_list,
            thresholds=thresholds,
            per_context=True,
        )

    def update_plot(self):
        from seismometer._api import plot_model_evaluation

        display(
            plot_model_evaluation(
                self.option_widget.cohorts,
                self.option_widget.target,
                self.option_widget.score,
                self.option_widget.thresholds,
                per_context=self.option_widget.group_scores,
            )
        )
        if self.show_code:
            display(self.plot_code())

    def plot_code(self):
        args = ", ".join(
            [
                repr(x)
                for x in [
                    self.option_widget.cohorts,
                    self.option_widget.target,
                    self.option_widget.score,
                    self.option_widget.thresholds,
                ]
            ]
        )
        help_text = HTML(
            f"Plot code: <code>sm.plot_model_evaluation({args}, per_context={self.option_widget.group_scores})</code>"
        )
        help_text.add_class("jp-RenderedHTMLCommon")
        return help_text


@export
class ExploreCohortEvaluation(ExplorationCohortSubclassEvaluationWidget):
    """
    A widget for exploring the model performance of a cohort.
    """

    def __init__(self):
        """
        A widget for exploring the results of a cohort.

        Parameters
        ----------
        cohort : seismometer.data.cohort.Cohort
            The cohort to explore.
        plot_fn : Callable
            A function that takes a cohort and returns a plot.
        filter_fn : Optional[Callable], optional
            A function that takes a cohort and returns a filter, by default None.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        thresholds = {f"Threshold {k}": v for k, v in enumerate(sorted(sg.thresholds), 1)}
        super().__init__(
            "Cohort Group Performance",
            sg.available_cohort_groups,
            sg.target_cols,
            sg.output_list,
            thresholds=thresholds,
            per_context=True,
        )

    def update_plot(self):
        from seismometer._api import plot_cohort_evaluation

        display(
            plot_cohort_evaluation(
                self.option_widget.cohort,
                self.option_widget.cohort_groups,
                self.option_widget.target,
                self.option_widget.score,
                self.option_widget.thresholds,
                per_context=self.option_widget.group_scores,
            )
        )
        if self.show_code:
            display(self.plot_code())

    def plot_code(self):
        args = ", ".join(
            [
                repr(x)
                for x in [
                    self.option_widget.cohort,
                    self.option_widget.cohort_groups,
                    self.option_widget.target,
                    self.option_widget.score,
                    self.option_widget.thresholds,
                ]
            ]
        )
        help_text = HTML(
            f"Plot code: <code>sm.plot_cohort_evaluation({args}, per_context={self.option_widget.group_scores})</code>"
        )
        help_text.add_class("jp-RenderedHTMLCommon")
        return help_text


@export
class ExploreCohortHistograms(ExplorationCohortSubclassEvaluationWidget):
    """
    A widget for exploring the model performance of a cohort.
    """

    def __init__(self):
        """
        A widget for exploring the results of a cohort.

        Parameters
        ----------
        cohort : seismometer.data.cohort.Cohort
            The cohort to explore.
        plot_fn : Callable
            A function that takes a cohort and returns a plot.
        filter_fn : Optional[Callable], optional
            A function that takes a cohort and returns a filter, by default None.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        super().__init__(
            "Cohort Group Score Histograms",
            sg.available_cohort_groups,
            sg.target_cols,
            sg.output_list,
            thresholds=None,
            per_context=False,
        )

    def update_plot(self):
        from seismometer._api import plot_cohort_group_histograms

        display(
            plot_cohort_group_histograms(
                self.option_widget.cohort,
                self.option_widget.cohort_groups,
                self.option_widget.target,
                self.option_widget.score,
            )
        )
        if self.show_code:
            display(self.plot_code())

    def plot_code(self):
        args = ", ".join(
            [
                repr(x)
                for x in [
                    self.option_widget.cohort,
                    self.option_widget.cohort_groups,
                    self.option_widget.target,
                    self.option_widget.score,
                ]
            ]
        )
        help_text = HTML(f"Plot code: <code>sm.plot_cohort_group_histograms({args})</code>")
        help_text.add_class("jp-RenderedHTMLCommon")
        return help_text


@export
class ExploreCohortLeadTime(ExplorationCohortSubclassEvaluationWidget):
    """
    A widget for exploring the model performance of a cohort.
    """

    def __init__(self):
        """
        A widget for exploring the results of a cohort.

        Parameters
        ----------
        cohort : seismometer.data.cohort.Cohort
            The cohort to explore.
        plot_fn : Callable
            A function that takes a cohort and returns a plot.
        filter_fn : Optional[Callable], optional
            A function that takes a cohort and returns a filter, by default None.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        thresholds = {"Score Threshold": min(sg.thresholds)}
        super().__init__(
            "Leadtime Analysis", sg.available_cohort_groups, sg.target_cols, sg.output_list, thresholds=thresholds
        )

    def update_plot(self):
        from seismometer._api import plot_cohort_lead_time

        display(
            plot_cohort_lead_time(
                self.option_widget.cohort,
                self.option_widget.cohort_groups,
                self.option_widget.target,
                self.option_widget.score,
                self.option_widget.thresholds[0],
            )
        )
        if self.show_code:
            display(self.plot_code())

    def plot_code(self):
        args = ", ".join(
            [
                repr(x)
                for x in [
                    self.option_widget.cohort,
                    self.option_widget.cohort_groups,
                    self.option_widget.target,
                    self.option_widget.score,
                    self.option_widget.thresholds[0],
                ]
            ]
        )
        help_text = HTML(f"Plot code: <code>sm.plot_cohort_lead_time({args})</code>")
        help_text.add_class("jp-RenderedHTMLCommon")
        return help_text


class ModelInterventionOptionsWidget(VBox, ValueWidget):
    value = traitlets.Dict(help="The selected values for the slider list")

    def __init__(
        self,
        outcome_names: tuple[Any] = None,
        intervention_names: tuple[Any] = None,
        reference_time_names: tuple[Any] = None,
    ):
        self.title = HTML('<h4 style="text-align: left; margin: 0px;">Model Options</h4>')
        self.outcome_list = Dropdown(options=outcome_names, value=outcome_names[0], description="Outcome")
        self.intervention_list = Dropdown(
            options=intervention_names, value=intervention_names[0], description="Intervention"
        )
        self.ref_time_list = Dropdown(
            options=reference_time_names, value=reference_time_names[0], description="Reference Time"
        )

        children = [self.title, self.outcome_list, self.intervention_list, self.ref_time_list]

        super().__init__(
            children=children,
            layout=Layout(align_items="flex-start", flex="0 0 auto"),
        )

        self.outcome_list.observe(self._on_value_change, "value")
        self.intervention_list.observe(self._on_value_change, "value")
        self.ref_time_list.observe(self._on_value_change, "value")

    def _on_value_change(self, change=None):
        self.value = {
            "outcome": self.outcome,
            "intervention": self.intervention,
            "reference_time": self.reference_time,
        }

    @property
    def outcome(self):
        return self.outcome_list.value

    @property
    def intervention(self):
        return self.intervention_list.value

    @property
    def reference_time(self):
        return self.ref_time_list.value


class ModelInterventionAndCohortGroupWidget(Box, ValueWidget):
    value = traitlets.Dict(help="The selected values for the cohorts and moedel options")

    def __init__(
        self,
        cohort_groups: dict[str, tuple[Any]],
        outcome_names: tuple[Any] = None,
        intervention_names: tuple[Any] = None,
        reference_time_names: tuple[Any] = None,
    ):
        self.cohort_list = DisjointSelectionListsWidget(options=cohort_groups, title="Cohort Filter", select_all=True)
        self.model_options = ModelInterventionOptionsWidget(outcome_names, intervention_names, reference_time_names)
        self.cohort_list.observe(self._on_value_change, "value")
        self.model_options.observe(self._on_value_change, "value")

        super().__init__(children=[self.model_options, self.cohort_list], layout=BOX_GRID_LAYOUT)

    def _on_value_change(self, change=None):
        self.value = {
            "cohorts": self.cohort_list.value,
            "model_options": self.model_options.value,
        }

    @property
    def cohort(self):
        return self.cohort_list.value[0]

    @property
    def cohort_groups(self):
        return self.cohort_list.value[1]

    @property
    def outcome(self):
        return self.model_options.outcome

    @property
    def intervention(self):
        return self.model_options.intervention

    @property
    def reference_time(self):
        return self.model_options.reference_time


class ExplorationCohortInterventionEvaluationWidget(ExlorationWidget):
    """
    A widget for exploring the model performance based on the subgroups of a cohort column.
    """

    def __init__(
        self,
        title: str,
        cohort_groups: dict[str, tuple[Any]],
        outcome_names: tuple[Any],
        intervention_names: tuple[Any],
        reference_time_names: tuple[Any],
    ):
        """
        A widget for exploring the results of a cohort's subgroups based on scores and intervention.

        Parameters
        ----------
        cohort : seismometer.data.cohort.Cohort
            The cohort to explore.
        plot_fn : Callable
            A function that takes a cohort and returns a plot.
        filter_fn : Optional[Callable], optional
            A function that takes a cohort and returns a filter, by default None.
        """
        super().__init__(
            title,
            option_widget=ModelInterventionAndCohortGroupWidget(
                cohort_groups,
                outcome_names,
                intervention_names,
                reference_time_names,
            ),
        )


@export
class ExploreCohortInterventionTimes(ExplorationCohortInterventionEvaluationWidget):
    """
    A widget for exploring the model performance of a cohort.
    """

    def __init__(self):
        """
        A widget for exploring the results of a cohort.

        Parameters
        ----------
        cohort : seismometer.data.cohort.Cohort
            The cohort to explore.
        plot_fn : Callable
            A function that takes a cohort and returns a plot.
        filter_fn : Optional[Callable], optional
            A function that takes a cohort and returns a filter, by default None.
        """
        from seismometer.seismogram import Seismogram

        sg = Seismogram()
        reference_times = [sg.predict_time]
        if sg.comparison_time and sg.comparison_time != sg.predict_time:
            reference_times.append(sg.comparison_time)

        super().__init__(
            "Outcome / Intervention Analysis",
            sg.available_cohort_groups,
            sg.config.outcomes,
            sg.config.interventions,
            reference_times,
        )

    def update_plot(self):
        from seismometer._api import plot_intervention_outcome_timeseries

        display(
            plot_intervention_outcome_timeseries(
                self.option_widget.outcome,
                self.option_widget.intervention,
                self.option_widget.reference_time,
                self.option_widget.cohort,
                self.option_widget.cohort_groups,
            )
        )
        if self.show_code:
            display(self.plot_code())

    def plot_code(self):
        args = ", ".join(
            [
                repr(x)
                for x in [
                    self.option_widget.outcome,
                    self.option_widget.intervention,
                    self.option_widget.reference_time,
                    self.option_widget.cohort,
                    self.option_widget.cohort_groups,
                ]
            ]
        )
        help_text = HTML(f"Plot code: <code>sm.plot_intervention_outcome_timeseries({args})</code>")
        help_text.add_class("jp-RenderedHTMLCommon")
        return help_text
