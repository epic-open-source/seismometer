from typing import Any

from IPython.display import HTML, display

from seismometer.controls.decorators import disk_cached_html_segment
from seismometer.controls.explore import (
    ExplorationCohortOutcomeInterventionEvaluationWidget,
    ExplorationCohortSubclassEvaluationWidget,
    ExplorationModelSubgroupEvaluationWidget,
    ExplorationScoreComparisonByCohortWidget,
    ExplorationSubpopulationWidget,
    ExplorationTargetComparisonByCohortWidget,
    ExplorationWidget,
    ModelFairnessAuditOptions,
)
from seismometer.core.decorators import export
from seismometer.data import pandas_helpers as pdh
from seismometer.html import template
from seismometer.seismogram import Seismogram

from .plots import (
    plot_cohort_evaluation,
    plot_cohort_group_histograms,
    plot_cohort_lead_time,
    plot_intervention_outcome_timeseries,
    plot_model_evaluation,
    plot_model_score_comparison,
    plot_model_target_comparison,
)
from .reports import generate_fairness_audit

# region Exploration Widgets


@export
class ExploreSubgroups(ExplorationSubpopulationWidget):
    """
    Explore the models base statistics based on a selected subpopulation.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__("Subpopulation Statistics", cohort_list_details)


@export
class ExploreModelEvaluation(ExplorationModelSubgroupEvaluationWidget):
    """
    Exploration widget for model evaluation, showing model performance for a specific subpopulation.

    This includes the ROC, recall vs predicted condition prevalence, calibration,
    PPV vs sensitivity, sensitivity/specificity/ppv, and a histogram.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__("Model Performance", plot_model_evaluation)


@export
class ExploreModelScoreComparison(ExplorationScoreComparisonByCohortWidget):
    """
    Exploration widget for model evaluation, showing model performance for a specific subpopulation.

    This includes the ROC, recall vs predicted condition prevalence, calibration,
    PPV vs sensitivity, sensitivity/specificity/ppv, and a histogram.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__("Model Score Comparison", plot_model_score_comparison)


@export
class ExploreModelTargetComparison(ExplorationTargetComparisonByCohortWidget):
    """
    Exploration widget for model target evaluation, showing model performance for a specific subpopulation.

    This includes the ROC, recall vs predicted condition prevalence, calibration,
    PPV vs sensitivity, sensitivity/specificity/PPV, and a histogram.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__("Model Target Comparison", plot_model_target_comparison)


@export
class ExploreCohortEvaluation(ExplorationCohortSubclassEvaluationWidget):
    """
    Exploration widget for cohort evaluation, showing model performance across thresholds and cohort subgroups.

    Creates a 2x3 grid of individual performance metrics across cohorts.

    Plots include Sensitivity, Flagged, PPV, Specificity, NPV vs Thresholds.
    Includes a legend with cohort size.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__("Cohort Group Performance", plot_cohort_evaluation)


@export
class ExploreCohortHistograms(ExplorationCohortSubclassEvaluationWidget):
    """
    Exploration widget to show the true positives and negative by model score.

    Shows a distribution of scores for each category in a cohort group.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__(
            "Cohort Group Score Histograms",
            plot_cohort_group_histograms,
            threshold_handling=None,
            ignore_grouping=True,
        )


@export
class ExploreCohortLeadTime(ExplorationCohortSubclassEvaluationWidget):
    """
    Exploration widget for the lead time between a model prediction and an event of interest.

    Shows the amount of lead time for each category in the cohort group.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__(
            "Leadtime Analysis",
            plot_cohort_lead_time,
            threshold_handling="min",
            ignore_grouping=True,
        )


@export
class ExploreCohortOutcomeInterventionTimes(ExplorationCohortOutcomeInterventionEvaluationWidget):
    """
    Exploration widget for viewing rates of interventions and outcomes across categories in a cohort group.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__("Outcome / Intervention Analysis", plot_intervention_outcome_timeseries)


@export
class ExploreFairnessAudit(ExplorationWidget):
    """
    Exploration widget for Aequitas model fairness metrics, showing details for a given target, score, and threshold.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        title = "Fairness Audit"
        sg = Seismogram()
        self.cohort_columns = sg.cohort_cols

        option_widget = ModelFairnessAuditOptions(
            sg.target_cols,
            sg.output_list,
            score_threshold=max(sg.thresholds),
            per_context=True,
        )
        super().__init__(title=title, option_widget=option_widget, plot_function=generate_fairness_audit)

    def generate_plot_args(self) -> tuple[tuple, dict]:
        """
        Returns plot function args from the option widget

        Returns
        -------
        tuple[tuple, dict]
            args, and kwargs for the plot function.
        """
        args = (
            self.cohort_columns,
            self.option_widget.target,
            self.option_widget.score,
            self.option_widget.score_threshold,
            self.option_widget.group_scores,
            list(self.option_widget.metrics),
            self.option_widget.fairness_threshold,
        )
        return args, {}


# endregion


@export
def cohort_list():
    """
    Displays an exhaustive list of available cohorts for analysis.
    """
    sg = Seismogram()
    from ipywidgets import Output, VBox

    from seismometer.controls.selection import MultiSelectionListWidget
    from seismometer.controls.styles import BOX_GRID_LAYOUT

    options = sg.available_cohort_groups

    comparison_selections = MultiSelectionListWidget(options, title="Cohort", show_all=True)
    output = Output()

    def on_widget_value_changed(*args):
        with output:
            display("Recalculating...", clear=True)
            html = cohort_list_details(comparison_selections.value)
            display(html, clear=True)

    comparison_selections.observe(on_widget_value_changed, "value")

    # get initial value
    on_widget_value_changed()

    return VBox(children=[comparison_selections, output], layout=BOX_GRID_LAYOUT)


@disk_cached_html_segment
@export
def cohort_list_details(cohort_dict: dict[str, tuple[Any]]) -> HTML:
    """
    Generates a HTML table of cohort details.

    Parameters
    ----------
    cohort_dict : dict[str, tuple[Any]]
        dictionary of cohort columns and values used to subselect a population for evaluation


    Returns
    -------
    HTML
        able indexed by targets, with counts of unique entities, and mean values of the output columns.
    """
    from seismometer.data.filter import filter_rule_from_cohort_dictionary

    sg = Seismogram()
    cfg = sg.config
    target_cols = [pdh.event_value(x) for x in cfg.targets]
    intervention_cols = [pdh.event_value(x) for x in cfg.interventions]
    outcome_cols = [pdh.event_value(x) for x in cfg.outcomes]

    rule = filter_rule_from_cohort_dictionary(cohort_dict)
    data = rule.filter(sg.dataframe)[
        cfg.entity_keys + cfg.output_list + intervention_cols + outcome_cols + target_cols
    ]
    cohort_count = data[sg.entity_keys[0]].nunique()
    if cohort_count < sg.censor_threshold:
        return template.render_censored_plot_message(sg.censor_threshold)

    groups = data.groupby(target_cols)
    float_cols = list(data[intervention_cols + outcome_cols].select_dtypes(include=float))

    stat_dict = {k: ["mean"] for k in float_cols}
    stat_dict.update({cfg.entity_id: ["nunique", "count"], cfg.context_id: ["nunique"]})

    groupstats = groups.agg(stat_dict)
    groupstats.columns = [pdh.event_name(x) for x in float_cols] + [
        f"Unique {cfg.entity_id}",
        f"{cfg.entity_id} Count",
        f"Unique {cfg.context_id}",
    ]
    new_names = [pdh.event_name(x) for x in target_cols]
    if len(new_names) == 1:
        new_names = new_names[0]  # because pandas Index only accepts a string for rename.
    groupstats.index.rename(new_names, inplace=True)
    html_table = groupstats.to_html()
    title = "Summary"
    return template.render_title_message(title, html_table)


# endregion
