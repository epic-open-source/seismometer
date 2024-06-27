import logging
from typing import Any

from IPython.display import display
from ipywidgets import HTML, Box, Button, Checkbox, Dropdown, Layout, Output, ValueWidget, VBox

from seismometer.controls.selection import MultiSelectionListWidget
from seismometer.controls.thresholds import MonotonicPercentSliderListWidget

logger = logging.getLogger("seismometer")


class ExplorationModelEvaluationWidget(VBox):
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
        **kwargs,
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
        layout = Layout(
            width="100%",
            height="min-content",
            border="solid 1px var(--jp-border-color1)",
            padding="var(--jp-cell-padding)",
        )
        self.center = Output(layout=Layout(height="max-content", max_width="2000px"))

        self.cohort_list = MultiSelectionListWidget(options=cohort_groups, title="Cohort Filter")

        self.model_options = ModelThresholdsWidget(target_names, score_names, thresholds)

        title = HTML(value=f"""<h3 style="text-align: left; margin-top: 0px;">{title}</h3>""")

        controls = Box(
            children=[self.model_options, self.cohort_list], layout=Layout(align_items="flex-start", grid_gap="20px")
        )

        self.show_code = Checkbox(
            value=False,
            description="show code",
            disabled=False,
            indent=False,
            tooltip="Show the code used to generate the plot",
        )
        self.plot_button = Button(description="Update Plots", button_style="primary", width="max-content")
        plot_controls = Box(layout=Layout(align_items="flex-start"), children=[self.plot_button, self.show_code])
        super().__init__(children=[title, controls, plot_controls, self.center], layout=layout)

        self.plot_button.on_click(self._on_plot_button_click)
        # show default plot
        self._on_plot_button_click()

    def _on_plot_button_click(self, button=None):
        self.center.clear_output()
        with self.center:
            self.update_plot()

    def update_plot(self):
        raise NotImplementedError("Subclasses must implement this method")


class ModelThresholdsWidget(VBox, ValueWidget):
    def __init__(self, target_names: tuple[Any], score_names: tuple[Any], thresholds: dict[str, float]):
        self.title = HTML('<h4 style="text-align: left; margin: 0px;">Model Options</h4>')
        self.target_list = Dropdown(options=target_names, value=target_names[0], description="Target Column")
        self.score_list = Dropdown(options=score_names, value=score_names[0], description="Score Column")
        self.threshold_list = MonotonicPercentSliderListWidget(
            names=tuple(thresholds.keys()), value=tuple(thresholds.values())
        )

        self.per_context_checkbox = Checkbox(
            value=False,
            description="combine scores",
            disabled=False,
            tooltip="Combine scores by taking the maximum score in the target window",
        )
        super().__init__(
            children=[self.title, self.target_list, self.score_list, self.threshold_list, self.per_context_checkbox],
            layout=Layout(align_items="flex-start", style={"flex-shrink": "0"}),
        )

    @property
    def target(self):
        return self.target_list.value

    @property
    def score(self):
        return self.score_list.value

    @property
    def thresholds(self):
        return self.threshold_list.value

    @property
    def group_scores(self):
        return self.per_context_checkbox.value


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
        thresholds = {f"Threshold {k}": v for k, v in enumerate(sg.thresholds, 1)}
        super().__init__(
            "Model Performance",
            sg.available_cohort_groups,
            sg.target_cols,
            sg.output_list,
            thresholds=thresholds,
        )

    def update_plot(self):
        from seismometer._api import plot_model_evaluation

        display(
            plot_model_evaluation(
                self.cohort_list.value,
                self.model_options.target,
                self.model_options.score,
                self.model_options.thresholds,
                per_context=self.model_options.group_scores,
            )
        )
        if self.show_code.value:
            display(self.plot_code())

    def plot_code(self):
        args = ", ".join(
            [
                repr(x)
                for x in [
                    self.cohort_list.value,
                    self.model_options.target,
                    self.model_options.score,
                    self.model_options.thresholds,
                ]
            ]
        )
        help_text = HTML(
            f"Plot code: <code>sm.plot_model_evaluation({args}, per_context={self.model_options.group_scores})</code>"
        )
        help_text.add_class("jp-RenderedHTMLCommon")
        return help_text
