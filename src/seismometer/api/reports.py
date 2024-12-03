import logging
from typing import Any, Optional

from IPython.display import HTML, IFrame

from seismometer.core.decorators import export
from seismometer.core.exceptions import CensoredResultException
from seismometer.core.io import slugify
from seismometer.core.nbhost import NotebookHost
from seismometer.data import pandas_helpers as pdh
from seismometer.data.filter import FilterRule
from seismometer.html import template
from seismometer.html.iframe import load_as_iframe
from seismometer.report.fairness import ExploreBinaryModelFairness
from seismometer.report.profiling import ComparisonReportWrapper, SingleReportWrapper
from seismometer.seismogram import Seismogram

logger = logging.getLogger("seismometer")


@export
class ExploreFairnessAudit(ExploreBinaryModelFairness):
    """
    Exploration widget for model fairness across cohorts for a binary classifier.

    .. versionchanged:: 0.3.0
       Uses explore controls instead of aequitas report.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__()


# region Reports
@export
def feature_alerts(exclude_cols: Optional[list[str]] = None):
    """
    Generates (or loads from disk) the `ydata-profiling` report feature quality alerts.

    Note: Does not regenerate the report if one exists on disk.

    Parameters
    ----------
    exclude_cols : Optional[list[str]], optional
        Columns to exclude from profiling. If None, defaults to excluding the identifiers in the dataset,
        by default None.
    """
    sg = Seismogram()
    exclude_cols = exclude_cols or sg.entity_keys

    SingleReportWrapper(
        df=sg.dataframe,
        output_path=sg.config.output_dir,
        exclude_cols=exclude_cols,
        title="Feature Report",
        alert_config=sg.alert_config,
    ).display_alerts()


@export
def feature_summary(exclude_cols: Optional[list[str]] = None, inline: bool = False):
    """
    Generates (or loads from disk) the `ydata-profiling` report.

    Note: Does not regenerate the report if one exists on disk.

    Parameters
    ----------
    exclude_cols : Optional[list[str]], optional
        Columns to exclude from profiling. If None, defaults to excluding the identifiers in the dataset,
        by default None.
    inline : bool, optional
        If True, shows the `ydata-profiling` report inline, by default False; displaying a link instead.
    """
    sg = Seismogram()
    exclude_cols = exclude_cols or sg.entity_keys

    SingleReportWrapper(
        df=sg.dataframe,
        output_path=sg.config.output_dir,
        exclude_cols=exclude_cols,
        title="Feature Report",
        alert_config=sg.alert_config,
    ).display_report(inline)


@export
def cohort_comparison_report(exclude_cols: list[str] = None):
    """
    Generates (or loads from disk) the `ydata-profiling` report stratified by the selected cohort variable.

    Note: Does not regenerate the report if one exists on disk.

    Parameters
    ----------
    exclude_cols : Optional[list[str]], optional
        Columns to exclude from profiling. If None, defaults to excluding the identifiers in the dataset,
        by default None.
    """
    sg = Seismogram()
    from seismometer.controls.cohort_comparison import ComparisonReportGenerator

    comparison_selections = ComparisonReportGenerator(sg.available_cohort_groups, exclude_cols=exclude_cols)
    comparison_selections.show()


@export
def target_feature_summary(exclude_cols: list[str] = None, inline=False):
    """
    Generates (or loads from disk) the `ydata-profiling` report stratified by the target variable.

    Note: Does not regenerate the report if one exists on disk.

    Parameters
    ----------
    exclude_cols : Optional[list[str]], optional
        Columns to exclude from profiling. If None, defaults to excluding the identifiers in the dataset,
        by default None.
    inline : bool, optional
        True to show the `ydata-profiling` report inline, by default False; displaying a link instead.
    """
    sg = Seismogram()

    exclude_cols = exclude_cols or sg.entity_keys
    positive_target = FilterRule.eq(sg.target, 1)
    negative_target = ~positive_target

    negative_target_df = negative_target.filter(sg.dataframe)
    positive_target_df = positive_target.filter(sg.dataframe)

    if negative_target_df.empty:
        logger.warning("No comparison report generated. The negative target has no data to profile.")
        return

    if positive_target_df.empty:
        logger.warning("No comparison report generated. The positive target has no data to profile.")
        return

    wrapper = ComparisonReportWrapper(
        l_df=negative_target_df,
        r_df=positive_target_df,
        output_path=sg.output_path,
        l_title=f"{sg.target}=0",
        r_title=f"{sg.target}=1",
        exclude_cols=exclude_cols,
        base_title="Target Comparison Report",
    )

    wrapper.display_report(inline)


@export
def generate_fairness_audit(
    cohort_columns: list[str],
    target_column: str,
    score_column: str,
    score_threshold: float,
    per_context: bool = False,
    metric_list: Optional[list[str]] = None,
    fairness_threshold: float = 1.25,
) -> HTML | IFrame | Any:
    """
    Generates the Aequitas fairness audit for a set of sensitive groups and metrics.

    Parameters
    ----------
    cohort_columns : list[str]
        cohort columns to investigate
    target_column : str
        target column to use
    score_column : str
        score column to use
    score_threshold : float
        threshold at which a score predicts a positive event
    per_context : bool, optional
        if scores should be grouped within a context, by default False
    metric_list : Optional[list[str]], optional
        list of metrics to evaluate, by default None
    fairness_threshold : float, optional
        allowed deviation from the default class within a cohort, by default 1.25

    Returns
    -------
    HTML | IFrame | Altair Chart
        IFrame holding the HTML of the audit
    """

    sg = Seismogram()

    path = "aequitas_{cohorts}_with_{target}_and_{score}_gt_{threshold}_metrics_{metrics}_ratio_{ratio}".format(
        cohorts="_".join(cohort_columns),
        target=target_column,
        score=score_column,
        threshold=score_threshold,
        metrics="_".join(metric_list),
        ratio=fairness_threshold,
    )
    if per_context:
        path += "_grouped"
    fairness_path = sg.config.output_dir / (slugify(path) + ".html")
    height = 200 + 100 * len(metric_list)

    if NotebookHost.supports_iframe() and fairness_path.exists():
        return load_as_iframe(fairness_path, height=height)
    data = (
        pdh.event_score(
            sg.data(),
            sg.entity_keys,
            score=score_column,
            ref_event=sg.predict_time,
            aggregation_method=sg.event_aggregation_method(target_column),
        )
        if per_context
        else sg.data()
    )

    target = pdh.event_value(target_column)
    data = data[[target, score_column] + cohort_columns]
    data = FilterRule.isin(target, (0, 1)).filter(data)
    positive_samples = data[target].sum()
    if min(positive_samples, len(data) - positive_samples) < sg.censor_threshold:
        return template.render_censored_plot_message(sg.censor_threshold)

    try:
        altair_plot = fairness_audit_altair(
            data, cohort_columns, score_column, target, score_threshold, metric_list, fairness_threshold
        )
    except CensoredResultException as error:
        return template.render_censored_data_message(str(error))

    if NotebookHost.supports_iframe():
        altair_plot.save(fairness_path, format="html")
        return load_as_iframe(fairness_path, height=height)

    return altair_plot


# endregion
