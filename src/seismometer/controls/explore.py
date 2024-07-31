import logging
from functools import wraps
from typing import Any, Callable, Literal, Optional

import traitlets
from IPython.display import display
from ipywidgets import HTML, Box, Button, Checkbox, Dropdown, FloatSlider, HBox, Layout, Output, ValueWidget, VBox

from .selection import DisjointSelectionListsWidget, MultiSelectionListWidget, SelectionListWidget
from .styles import BOX_GRID_LAYOUT, WIDE_LABEL_STYLE, html_title
from .thresholds import MonotonicProbabilitySliderListWidget

logger = logging.getLogger("seismometer")

# region Model Evaluation Header Controls


class UpdatePlotWidget(Box):
    """Widget for updating plots and showing code behind the plot call."""

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

        self.plot_button = Button(description=self.UPDATE_PLOTS, button_style="primary")
        layout = Layout(align_items="flex-start")
        children = [self.plot_button, self.code_checkbox]
        super().__init__(layout=layout, children=children)

    @property
    def show_code(self) -> bool:
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

    def on_toggle_code(self, callback):
        @wraps(callback)
        def callback_wrapper(change):
            callback(self.code_checkbox.value)

        self.code_checkbox.observe(callback_wrapper, "value")


class ModelOptionsWidget(VBox, ValueWidget):
    value = traitlets.Dict(help="The selected values for the model options")

    def __init__(
        self,
        target_names: tuple[Any],
        score_names: tuple[Any],
        thresholds: Optional[dict[str, float]] = None,
        per_context: Optional[bool] = None,
    ):
        """Widget for model based options

        Parameters
        ----------
        target_names : tuple[Any]
            list of target column names
        score_names : tuple[Any]
            list of model score names
        thresholds : dict[str, float]
            list of thresholds for the model scores, will be sorted into decreasing order
        per_context : bool, optional
            if scores should be grouped by context, by default None, in which case this checkbox is not shown.
        """
        self.title = html_title("Model Options")
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
            self.per_context_checkbox = Checkbox(
                value=per_context,
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
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        self.target_list.disabled = disabled
        self.score_list.disabled = disabled
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
        """target column descriptor"""
        return self.target_list.value

    @property
    def score(self) -> str:
        """score column descriptor"""
        return self.score_list.value

    @property
    def thresholds(self) -> tuple[float]:
        """thresholds for the score"""
        if self.threshold_list:
            return self.threshold_list.value

    @property
    def group_scores(self) -> bool:
        """if the scores should be grouped"""
        if self.per_context_checkbox:
            return self.per_context_checkbox.value


class ModelOptionsAndCohortsWidget(Box, ValueWidget):
    value = traitlets.Dict(help="The selected values for the cohorts and model options")

    def __init__(
        self,
        cohort_groups: dict[str, tuple[Any]],
        target_names: tuple[Any],
        score_names: tuple[Any],
        thresholds: dict[str, float],
        per_context: bool = False,
    ):
        """
        Widget for model based options and cohort selection

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
            if scores should be grouped by context, by default False
        """
        self.cohort_list = MultiSelectionListWidget(options=cohort_groups, title="Cohort Filter")
        self.model_options = ModelOptionsWidget(target_names, score_names, thresholds, per_context)
        self.cohort_list.observe(self._on_value_change, "value")
        self.model_options.observe(self._on_value_change, "value")

        super().__init__(children=[self.model_options, self.cohort_list], layout=BOX_GRID_LAYOUT)

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
    def thresholds(self) -> tuple[float]:
        """Score thresholds"""
        return self.model_options.thresholds

    @property
    def group_scores(self) -> bool:
        """If scores should be grouped by context"""
        return self.model_options.group_scores


class ModelOptionsAndCohortGroupWidget(Box, ValueWidget):
    value = traitlets.Dict(help="The selected values for the cohorts and model options")

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
        individual values within a cohort attribute

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
            if scores should be grouped by context, by default False
        """
        self.cohort_list = DisjointSelectionListsWidget(options=cohort_groups, title="Cohort Filter", select_all=True)
        self.model_options = ModelOptionsWidget(target_names, score_names, thresholds, per_context)
        self.cohort_list.observe(self._on_value_change, "value")
        self.model_options.observe(self._on_value_change, "value")

        super().__init__(children=[self.model_options, self.cohort_list], layout=BOX_GRID_LAYOUT)
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
        """cohort column descriptor"""
        return self.cohort_list.value[0]

    @property
    def cohort_groups(self) -> tuple[Any]:
        """cohort groups"""
        return self.cohort_list.value[1]

    @property
    def target(self) -> str:
        """target column descriptor"""
        return self.model_options.target

    @property
    def score(self) -> str:
        """score column descriptor"""
        return self.model_options.score

    @property
    def thresholds(self) -> tuple[float]:
        """score thresholds"""
        return self.model_options.thresholds

    @property
    def group_scores(self) -> bool:
        """if scores should be grouped by context"""
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
        Widget for selecting an intervention and outcome for a model implementation

        Parameters
        ----------
        outcome_names : tuple[Any], optional
            names of outcome columns, by default None
        intervention_names : tuple[Any], optional
            names of intervention columns, by default None
        reference_time_names : tuple[Any], optional
            name for the reference time to align patients, by default None
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
        """outcome column descriptor"""
        return self.outcome_list.value

    @property
    def intervention(self) -> str:
        """intervention column descriptor"""
        return self.intervention_list.value

    @property
    def reference_time(self) -> str:
        """reference time column descriptor"""
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
            cohort names and category values
        outcome_names : tuple[Any], optional
            outcome descriptors, by default None
        intervention_names : tuple[Any], optional
            intervention descriptors, by default None
        reference_time_names : tuple[Any], optional
            reference time descriptors, by default None
        """
        self.cohort_list = DisjointSelectionListsWidget(options=cohort_groups, title="Cohort Filter", select_all=True)
        self.model_options = ModelInterventionOptionsWidget(outcome_names, intervention_names, reference_time_names)
        self.cohort_list.observe(self._on_value_change, "value")
        self.model_options.observe(self._on_value_change, "value")

        super().__init__(children=[self.model_options, self.cohort_list], layout=BOX_GRID_LAYOUT)
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
        """cohort column descriptor"""
        return self.cohort_list.value[0]

    @property
    def cohort_groups(self) -> tuple[Any]:
        """cohort category values"""
        return self.cohort_list.value[1]

    @property
    def outcome(self) -> str:
        """outcome column descriptor"""
        return self.model_options.outcome

    @property
    def intervention(self) -> str:
        """intervention column descriptor"""
        return self.model_options.intervention

    @property
    def reference_time(self) -> str:
        """reference time column descriptor"""
        return self.model_options.reference_time


class ModelFairnessAuditOptions(Box, ValueWidget):
    value = traitlets.Dict(help="The selected values for the audit and model options")

    def __init__(
        self,
        target_names: tuple[Any],
        score_names: tuple[Any],
        score_threshold: float,
        per_context: bool = True,
        fairness_metrics: tuple[str] = None,
        fairness_threshold: float = 1.25,
    ):
        """
        Widget for selecting interventions and outcomes across categories in a cohort group.

        Parameters
        ----------
        target_names : tuple[Any]
            model target columns
        score_names : tuple[Any]
            model score columns
        score_threshold : float
            main threshold for the model score
        per_context : bool, optional
            if scores should be grouped by context, by default True
        fairness_metrics : tuple[str], optional
            list of fairness metrics to display, if None (default) will use ["tpr", "fpr", "pprev"]
        fairness_threshold : float, optional
            threshold for fairness metrics, by default 1.25
        """
        all_metrics = ["pprev", "tpr", "fpr", "fnr", "ppr", "precision"]
        fairness_metrics = fairness_metrics or ["pprev", "tpr", "fpr"]
        self.fairness_slider = FloatSlider(
            description="Threshold",
            value=fairness_threshold,
            min=1.0,
            max=2.0,
            step=0.01,
            tooltip="Threshold for fairness metrics",
            style=WIDE_LABEL_STYLE,
        )
        thresholds = {"Score Threshold": score_threshold}
        self.fairness_list = SelectionListWidget(options=all_metrics, value=fairness_metrics, title="Metrics")
        fairness_section = VBox(
            children=[html_title("Audit Options"), HBox(children=[self.fairness_list, self.fairness_slider])]
        )
        self.model_options = ModelOptionsWidget(target_names, score_names, thresholds, per_context)

        super().__init__(children=[self.model_options, fairness_section], layout=BOX_GRID_LAYOUT)

        self.fairness_list.observe(self._on_value_change, "value")
        self.fairness_slider.observe(self._on_value_change, "value")
        self.model_options.observe(self._on_value_change, "value")
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        self.fairness_list.disabled = disabled
        self.fairness_slider.disabled = disabled
        self.model_options.disabled = disabled

    def _on_value_change(self, change=None):
        self.value = {
            "fairness_metrics": self.fairness_list.value,
            "fairness_threshold": self.fairness_slider.value,
            "model_options": self.model_options.value,
        }

    @property
    def metrics(self) -> tuple[str]:
        """selected cohorts"""
        return self.fairness_list.value

    @property
    def fairness_threshold(self) -> tuple[str]:
        """selected cohorts"""
        return self.fairness_slider.value

    @property
    def target(self) -> str:
        """target column descriptor"""
        return self.model_options.target

    @property
    def score(self) -> str:
        """Score column descriptor"""
        return self.model_options.score

    @property
    def score_threshold(self) -> tuple[float]:
        """Score thresholds"""
        return self.model_options.thresholds["Score Threshold"]

    @property
    def group_scores(self) -> bool:
        """If scores should be grouped by context"""
        return self.model_options.group_scores


# endregion
# region Exploration Widgets


class ExplorationWidget(VBox):
    """
    Parent class for model exploration widgets.
    """

    def __init__(self, title: str, option_widget: ValueWidget, plot_function: Callable[..., Any]):
        """Parent class for a plot exploration widget.

        Parameters
        ----------
        title : str
            Widget title
        option_widget : ValueWidget
            widget that contains the options the plot_function
        plot_function : Callable[..., Any]
            a function that generates content for display within the control.
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
        self.option_widget = option_widget
        self.plot_function = plot_function
        self.update_plot_widget = UpdatePlotWidget()
        super().__init__(
            children=[title, self.option_widget, self.update_plot_widget, self.code_output, self.center], layout=layout
        )

        # show initial plot
        self.update_plot(initial=True)

        # attache event handlers
        self.option_widget.observe(self._on_option_change, "value")
        self.update_plot_widget.on_click(self._on_plot_button_click)
        self.update_plot_widget.on_toggle_code(self._on_toggle_code)

    @property
    def disabled(self) -> bool:
        """
        If the widget is disabled.
        """
        return False

    @property
    def show_code(self) -> bool:
        """
        If the widget should show the plot's code
        """
        return self.update_plot_widget.show_code

    def update_plot(self, initial: bool = False):
        plot_args, plot_kwargs = self.generate_plot_args()
        self.current_plot_code = self.generate_plot_code(plot_args, plot_kwargs)
        plot = self.plot_function(*plot_args, **plot_kwargs)
        if not initial:
            self.center.clear_output()
            with self.center:
                display(plot)
        else:
            self.center.append_display_data(plot)

    def _on_plot_button_click(self, button=None):
        """handle for the update plot button"""
        self.option_widget.disabled = True
        self.update_plot()
        self._on_toggle_code(self.show_code)
        self.option_widget.disabled = False

    def generate_plot_code(self, plot_args: tuple = None, plot_kwargs: dict = None) -> str:
        """String representation of the plot's code"""
        plot_module = self.plot_function.__module__
        plot_method = self.plot_function.__name__

        match plot_module:
            case "__main__":
                method_string = plot_method
            case "seismometer._api":
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
        """handle for the toggle code checkbox"""
        self.code_output.clear_output()
        if not show_code:
            return

        highlighted_code = HTML(
            f"<span style='user-select: none;'>Plot code: </span><code>{self.current_plot_code}</code>"
        )
        highlighted_code.add_class("jp-RenderedHTMLCommon")
        with self.code_output:
            display(highlighted_code)

    def _on_option_change(self, change=None):
        """enable the plot to be updated"""
        self.update_plot_widget.disabled = self.disabled

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """override this method with code to show a plot"""
        raise NotImplementedError("Subclasses must implement this method")


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
            title of the control
        plot_function : Callable[..., Any]
            Expected to have the following signature:

            .. code:: python

                def plot_function(
                    cohorts: dict[str,tuple[Any]]
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
        """Generates the plot arguments for the model evaluation plot"""
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
            title of the control
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
        """Generates the plot arguments for the model evaluation plot"""

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
            title of the control
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
        """Generates the plot arguments for the model evaluation plot"""
        args = [
            self.option_widget.cohort,
            self.option_widget.cohort_groups,
            self.option_widget.outcome,
            self.option_widget.intervention,
            self.option_widget.reference_time,
        ]
        kwargs = {}
        return args, kwargs


# endregion
