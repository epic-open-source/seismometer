import logging
from typing import Optional

import numpy as np
import pandas as pd
from IPython.display import HTML, SVG, display
from pandas.io.formats.style import Styler

import seismometer.plot as plot

from .controls.decorators import disk_cached_html_segment, display_cached_widget
from .data import (
    assert_valid_performance_metrics_df,
    calculate_bin_stats,
    calculate_eval_ci,
    default_cohort_summaries,
    get_cohort_data,
    get_cohort_performance_data,
)
from .data import pandas_helpers as pdh
from .data import score_target_cohort_summaries
from .data.decorators import export
from .data.filter import FilterRule
from .data.timeseries import create_metric_timeseries
from .html import template
from .report.auditing import display_fairness_audit
from .report.profiling import ComparisonReportWrapper, SingleReportWrapper
from .seismogram import Seismogram

logger = logging.getLogger("seismometer")


# region Reports
@export
def feature_alerts(exclude_cols: list[str] = None):
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


def feature_summary(exclude_cols: list[str] = None, inline: bool = False):
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
    from .controls.cohort_comparison import ComparisonReportGenerator

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
def fairness_audit(metric_list: Optional[list[str]] = None, fairness_threshold=1.25) -> None:
    """
    Displays the Aequitas fairness audit for a set of sensitive groups and metrics.

    Parameters
    ----------
    sensitive_groups : list[str]
        A list of columns that correspond to the cohorts to stratify by.
    metric_list : list[str]
        The list of metrics to use in Aequitas. Chosen from:
            "tpr",
            "tnr",
            "for",
            "fdr",
            "fpr",
            "fnr",
            "npv",
            "ppr",
            "precision",
            "pprev".
    fairness_threshold : float
        The maximum ratio between sensitive groups before differential performance is considered a 'failure'.
    """
    sg = Seismogram()

    sensitive_groups = sg.cohort_cols
    metric_list = metric_list or ["tpr", "fpr", "pprev"]

    df = pdh.event_score(
        sg.data(),
        sg.entity_keys,
        score=sg.output,
        ref_event=sg.predict_time,
        aggregation_method=sg.event_aggregation_method(sg.target),
    )[[sg.target, sg.output] + sensitive_groups]

    display_fairness_audit(
        df, sensitive_groups, sg.output, sg.target, sg.thresholds[0], metric_list, fairness_threshold
    )


# endregion

# region notebook IPWidgets # TODO - move to .controls submodule


@export
@display_cached_widget
def cohort_selector():
    """
    Display an IPyWidget selector to control cohorts used in visualizations.
    """
    from seismometer.controls.selection import DisjointSelectionListsWidget

    sg = Seismogram()
    options = sg.available_cohort_groups
    widget = DisjointSelectionListsWidget(
        options=options, value=sg.selected_cohort, title="Select Subgroups of Interest", select_all=True
    )

    def on_widget_value_changed(*args):
        sg.selected_cohort = widget.value

    widget.observe(on_widget_value_changed, "value")

    # get intial value
    on_widget_value_changed()
    return widget


@export
@display_cached_widget
def cohort_list():
    """
    Displays an exhaustive list of available cohorts for analysis.
    """
    sg = Seismogram()
    from ipywidgets import HBox, Output

    from .controls.selection import MultiSelectionListWidget
    from .data.filter import filter_rule_from_cohort_dictionary

    options = sg.available_cohort_groups

    comparison_selections = MultiSelectionListWidget(
        options, ghost_text="Select cohort groups", title="Available Cohorts"
    )
    output = Output()

    def on_widget_value_changed(*args):
        output.clear_output()
        with output:
            if comparison_selections.value:
                rule = filter_rule_from_cohort_dictionary(comparison_selections.value)
                cohort_count = sum(rule.mask(sg.dataframe))
                if cohort_count < 10:
                    print("Selected Cohort is empty, or has fewer than 10 predictions.")
                else:
                    print(f"Selected Cohort has {sum(rule.mask(sg.dataframe))} predictions.")
            else:
                print(f"No Cohort Selected: {len(sg.dataframe)} predictions total.")

    comparison_selections.observe(on_widget_value_changed, "value")

    # get intial value
    on_widget_value_changed()

    return HBox(children=[comparison_selections, output])


# endregion


# region plot accessors
@export
def plot_cohort_hist():
    """Display a histogram plot of predicted probabilities for all cohorts in the selected attribute."""
    sg = Seismogram()
    cohort_col = sg.selected_cohort[0]
    subgroups = sg.selected_cohort[1]
    censor_threshold = sg.censor_threshold
    return _plot_cohort_hist(sg.data(), sg.target_event, sg.target, sg.output, cohort_col, subgroups, censor_threshold)


@disk_cached_html_segment
def _plot_cohort_hist(
    dataframe: pd.DataFrame,
    target_event: str,
    target: str,
    output: str,
    cohort_col: str,
    subgroups: list[str],
    censor_threshold: int = 10,
) -> HTML:
    """
    Creates an HTML segment displaying a histogram of predicted probabilities for each cohort.

    Parameters
    ----------
    dataframe : pd.DataFrame
        data source
    target_event : str
        event time for the target
    target : str
        event value for the target
    output : str
        score column
    cohort_col : str
        column for cohort splits
    subgroups : list[str]
        column values to split by
    censor_threshold : int
        minimum rows to allow in a plot, by default 10.

    Returns
    -------
    HTML
        A stacked set of histograms for each selected subgroup in the cohort.
    """

    cData = get_cohort_data(dataframe, cohort_col, proba=output, true=target, splits=subgroups)

    # filter by group size
    cCount = cData["cohort"].value_counts()
    good_groups = cCount.loc[cCount > censor_threshold].index
    cData = cData.loc[cData["cohort"].isin(good_groups)]

    if len(cData.index) == 0:
        logger.error(
            "No groups were left uncensored; timeframe is too small in combination with frequency of "
            + f"{target_event} and the number of cohorts"
        )
        return

    bins = np.histogram_bin_edges(cData["pred"], bins=20)
    svg1 = plot.cohorts_vertical(cData, plot.histogram_stacked, func_kws={"show_legend": False, "bins": bins})
    title = f"Predicted Probabilities by {cohort_col}"
    return HTML(f"""<div style="width: max-content;"><h3 style="text-align: center;">{title}</h3>{svg1.data}</div>""")


@export
def plot_leadtime_enc(score=None, ref_time=None, target_event=None, max_hours=8):
    """Displays the amount of time that a high prediction gives before an event of interest.

    Parameters
    ----------
    score : Optional[str], optional
        The name of the score column to use, by default None; uses sg.output.
    ref_time : Optional[str], optional
        The reference time used in the visualization, by default None; uses sg.predict_time.
    target_event : Optional[str], optional
        The name of the target column to use, by default None; uses sg.target.
    """
    sg = Seismogram()
    # Assume sg.dataframe has scores/events with appropriate windowed event
    cohort_col = sg.selected_cohort[0]
    subgroups = sg.selected_cohort[1]
    censor_threshold = sg.censor_threshold
    x_label = "Lead Time (hours)"

    score = score or sg.output
    ref_time = ref_time or sg.predict_time
    target_event = pdh.event_value(target_event) or sg.target
    target_zero = pdh.event_time(target_event) or sg.time_zero
    threshold = sg.thresholds[0]

    if target_event not in sg.dataframe:
        logger.error(f"Target event ({target_event}) not found in dataset. Cannot plot leadtime.")
        return

    if target_zero not in sg.dataframe:
        logger.error(f"Target event time-zero ({target_zero}) not found in dataset. Cannot plot leadtime.")
        return

    return _plot_leadtime_enc(
        sg.dataframe,
        target_event,
        target_zero,
        threshold,
        score,
        ref_time,
        sg.entity_keys,
        cohort_col,
        subgroups,
        max_hours,
        x_label,
        censor_threshold,
    )


@disk_cached_html_segment
def _plot_leadtime_enc(
    dataframe: pd.DataFrame,
    target_event: str,
    target_zero: str,
    threshold: list[float],
    score: str,
    ref_time: str,
    entity_keys: str,
    cohort_col: str,
    subgroups: list[any],
    max_hours: int,
    x_label: str,
    censor_threshold: int = 10,
) -> HTML:
    """
    HTML Plot of time between prediction and target event.

    Parameters
    ----------
    dataframe : pd.DataFrame
        source data
    target_event : str
        event column
    target_zero : str
        event value
    threshold : str
        score thresholds
    score : str
        score column
    ref_time : str
        prediction time
    entity_keys : str
        entity key column
    cohort_col : str
        cohort column name
    subgroups : list[any]
        cohort groups from the cohort column
    x_label : str
        label for the x axis of the plot
    max_hours : _type_
        max horizon time
    censor_threshold : int
        minimum rows to allow in a plot.

    Returns
    -------
    HTML
        Lead time plot
    """
    summary_data = dataframe[dataframe[target_event] == 1]
    if len(summary_data.index) == 0:
        logger.error(f"No positive events ({target_event}=1) were found")
        return

    cohort_mask = summary_data[cohort_col].isin(subgroups)
    threshold_mask = summary_data[score] > threshold

    # summarize to first score
    summary_data = pdh.event_score(
        summary_data[cohort_mask & threshold_mask],
        entity_keys,
        score=score,
        ref_event=target_zero,
        aggregation_method="first",
    )[[target_zero, ref_time, cohort_col]]

    # filter by group size
    counts = summary_data[cohort_col].value_counts()
    good_groups = counts.loc[counts > censor_threshold].index
    summary_data = summary_data.loc[summary_data[cohort_col].isin(good_groups)]

    if len(summary_data.index) == 0:
        logger.error(
            "No groups were left uncensored; timeframe is too small "
            + f"in combination with frequency of {target_event} and the number of cohorts"
        )
        return

    # Truncate to minute but plot hour
    summary_data[x_label] = (summary_data[ref_time] - summary_data[target_zero]).dt.total_seconds() // 60 / 60

    title = f'Lead Time from {score.replace("_", " ")} to {(target_zero).replace("_", " ")}'
    rows = summary_data[cohort_col].nunique()
    svg1 = plot.leadtime_violin(summary_data, x_label, cohort_col, xmax=max_hours, figsize=(9, 1 + rows))

    return HTML(f"""<div style="width: max-content;"><h3 style="text-align: center;">{title}</h3>{svg1.data}</div>""")


@export
def cohort_evaluation(per_context_id=False):
    """Displays model performance metrics on cohort attribute across thresholds.

    Parameters
    ----------
    per_context_id : bool, optional
        If True, limits data to one row per context_id, by default False.
    """

    sg = Seismogram()

    cohort_col = sg.selected_cohort[0]
    subgroups = sg.selected_cohort[1]
    censor_threshold = sg.censor_threshold
    return _plot_cohort_evaluation(
        sg.data(),
        sg.entity_keys,
        sg.target,
        sg.output,
        sg.thresholds,
        cohort_col,
        subgroups,
        censor_threshold,
        per_context_id,
        sg.event_aggregation_method(sg.target),
        sg.predict_time,
    )


@disk_cached_html_segment
def _plot_cohort_evaluation(
    dataframe: pd.DataFrame,
    entity_keys: list[str],
    target: str,
    output: str,
    thresholds: list[float],
    cohort_col: str,
    subgroups: list[str],
    censor_threshold: int = 10,
    per_context_id: bool = False,
    aggregation_method: str = "max",
    ref_time: str = None,
) -> HTML:
    """
    Plots model performance metrics split by on a cohort attribute.

    Parameters
    ----------
    dataframe : pd.DataFrame
        source data
    entity_keys : list[str]
        columns to use for aggregation
    target : str
        target value
    output : str
        score column
    thresholds : list[float]
        model thresholds
    cohort_col : str
        cohort column name
    subgroups : list[str]
        subgroups of interest in the cohort column
    censor_threshold : int
        minimum rows to allow in a plot, by default 10
    per_context_id : bool, optional
        if true, aggregate scores for each context, by default False
    aggregation_method : str, optional
        method to reduce multiple scores into a single value before calculation of performance, by default "max"
        ignored if per_context_id is False
    ref_time : str, optional
        reference time column used for aggregation when per_context_id is True and aggregation_method is time-based

    Returns
    -------
    HTML
        _description_
    """
    data = (
        pdh.event_score(
            dataframe, entity_keys, score=output, ref_event=ref_time, aggregation_method=aggregation_method
        )
        if per_context_id
        else dataframe
    )

    plot_data = get_cohort_performance_data(
        data, cohort_col, proba=output, true=target, splits=subgroups, censor_threshold=censor_threshold
    )
    try:
        assert_valid_performance_metrics_df(plot_data)
    except ValueError:
        # Log instead of raise so can continue
        logger.error(f"Insufficient data; likely censoring all cohorts with threshold of {censor_threshold}.")
        return

    svg1 = plot.cohort_evaluation_vs_threshold(plot_data, cohort_feature=cohort_col, highlight=thresholds)
    title = f"Model Performance Metrics on {cohort_col} across Thresholds"
    return HTML(f"""<div style="width: max-content;"><h3 style="text-align: center;">{title}</h3>{svg1.data}</div>""")


@export
def model_evaluation(per_context_id=False):
    """Displays overall performance of the model.

    Parameters
    ----------
    per_context_id : bool, optional
        If True, limits data to one row per context_id, by default False.
    """
    sg = Seismogram()
    return _model_evaluation(
        sg.dataframe,
        sg.entity_keys,
        sg.target_event,
        sg.target,
        sg.output,
        sg.thresholds,
        per_context_id,
        sg.event_aggregation_method(sg.target),
        sg.predict_time,
    )


@disk_cached_html_segment
def _model_evaluation(
    dataframe: pd.DataFrame,
    entity_keys: list[str],
    target_event: str,
    target: str,
    output: str,
    thresholds: Optional[list[float]],
    per_context_id: bool = False,
    aggregation_method: str = "max",
    ref_time: str = None,
) -> HTML:
    """
    plots common model evaluation metrics

    Parameters
    ----------
    dataframe : pd.DataFrame
        source data
    entity_keys : list[str]
        columns to use for aggregation
    target_event : str
        target event name
    target : str
        target column
    output : str
        score column
    thresholds : Optional[list[float]]
        model thresholds
    per_context_id : bool, optional
        report only the max score for a given entity context, by default False
    aggregation_method : str, optional
        method to reduce multiple scores into a single value before calculation of performance, by default "max"
        ignored if per_context_id is False
    ref_time : str, optional
        reference time column used for aggregation when per_context_id is True and aggregation_method is time-based

    Returns
    -------
    HTML
        Plot of model evaluation metrics
    """
    data = (
        pdh.event_score(
            dataframe, entity_keys, score=output, ref_event=ref_time, aggregation_method=aggregation_method
        )
        if per_context_id
        else dataframe
    )

    # Validate
    data = data.loc[data[target].notna() & data[output].notna()]
    if len(data.index) == 0:
        logger.error(f"No rows have {target_event} and {output}.")
        return
    if (lcount := data[target].nunique()) != 2:
        logger.error(f"Evaluation Issue: Expected exactly two classes but found {lcount}")
        return
    stats = calculate_bin_stats(data[target], data[output])
    ci_data = calculate_eval_ci(stats, data[target], data[output], conf=0.95)
    title = f"Overall Performance for {target_event} (Per {'Encounter' if per_context_id else 'Observation'})"
    svg1 = plot.evaluation(
        stats,
        ci_data=ci_data,
        truth=data[target],
        output=data[output].values,
        show_thresholds=True,
        highlight=thresholds,
    )

    return HTML(f"""<div style="width: max-content;"><h3 style="text-align: center;">{title}</h3>{svg1.data}</div>""")


def plot_trend_intervention_outcome() -> HTML:
    """
    Plots two timeseries based on selectors; an outcome and then an intervention.

    Makes use of the cohort selectors as well intervention and outcome selectors for which data to use.
    Uses the configuration for comparison_time as the reference time for both plots.
    """
    sg = Seismogram()
    return _plot_trend_intervention_outcome(
        sg.dataframe,
        sg.entity_keys,
        sg.comparison_time or sg.predict_time,
        sg.outcome,
        sg.intervention,
        sg.selected_cohort[0],
        sg.selected_cohort[1],
        sg.censor_threshold,
    )


@disk_cached_html_segment
def _plot_trend_intervention_outcome(
    dataframe: pd.DataFrame,
    entity_keys: list[str],
    reftime: str,
    outcome: str,
    intervention: str,
    cohort_col: str,
    subgroups: list[str],
    censor_threshold: int = 10,
) -> HTML:
    """
    Plots two timeseries based on selectors; an outcome and then an intervention.

    Parameters
    ----------
    dataframe : pd.DataFrame
        data source
    entity_keys : list[str]
        columns to use for aggregation
    reftime : str
        reference time column for alignment
    outcome : str
        model score
    intervention : str
        intervention event time column
    cohort_col : str
        column name for the cohort to split on
    subgroups : list[str]
        values of interest in the cohort column
    censor_threshold : int, optional
        minimum rows to allow in a plot, by default 10

    Returns
    -------
    HTML
        Plot of two timeseries
    """
    time_bounds = (dataframe[reftime].min(), dataframe[reftime].max())  # Use the full time range
    show_legend = True

    try:
        outcome_col = pdh.event_value(outcome)
        svg1 = _plot_ts_cohort(
            dataframe,
            entity_keys,
            outcome_col,
            cohort_col,
            subgroups,
            reftime=reftime,
            time_bounds=time_bounds,
            show_legend=show_legend,
            censor_threshold=censor_threshold,
        )
        show_legend = False
    except IndexError:
        logger.warning("No outcome timeseries plotted; needs one event with configured usage of `outcome`.")

    try:
        intervention_col = pdh.event_value(intervention)
        svg2 = _plot_ts_cohort(
            dataframe,
            entity_keys,
            intervention_col,
            cohort_col,
            subgroups,
            reftime=reftime,
            time_bounds=time_bounds,
            boolean_event=True,
            show_legend=show_legend,
            censor_threshold=censor_threshold,
        )
    except IndexError:
        logger.warning(
            "No intervention timeseries plotted; " "needs one event with configured usage of `intervention`."
        )
    return HTML(
        f"""<div style="width: max-content;"><h3 style="text-align: center;">Outcome</h3>{svg1.data}
        <h3 style="text-align: center;">Intervention</h3>{svg2.data}</div>"""
    )


def _plot_ts_cohort(
    dataframe: pd.DataFrame,
    entity_keys: list[str],
    event_col: str,
    cohort_col: str,
    subgroups: list[str],
    reftime: str,
    *,
    time_bounds: Optional[str] = None,
    boolean_event: bool = False,
    plot_counts: bool = False,
    show_legend: bool = False,
    ylabel: str = None,
    censor_threshold: int = 10,
) -> SVG:
    """
    Plot a single timeseries given a full dataframe and parameters.

    Given a dataframe like sg.dataframe and the relevant column names, will filter data to the values,
    cohorts, and timeframe before plotting.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input data.
    entity_keys : list[str]
        A list of column names to use for summarizing the data.
    event_col : str
        The column name of value, used as y-axis.
    cohort_col : str
        The column name of the cohort, used to style separate lines.
    subgroups : list[str]
        The specific cohorts to plot, restricts cohort_col.
    reftime : str
        The column name of the reference time.
    time_bounds : Optional[str], optional
        An optional tuple (min, max) of times to bound the data to, by default None.
        If present the data is reduced to times within bounds prior to summarization.
    boolean_event : bool, optional
        A flag when set indicates the event is boolean so negative values are filtered out, by default False.
    plot_counts : bool, optional
        A flag when set will add an additional axis showing count of data at each point, by default False.
    show_legend : bool, optional
        A flag when set will show the legend on the plot, by default False.
    save : bool, optional
        A flag when set will save the plot to disk, by default False.
    ylabel : Optional[str], optional
        The label for the y-axis, by default None; will derive the label from the column name.
    censor_threshold : int, optional
        The minimum number of values for a given time that are needed to not be filtered, by default 10.
    """

    keep_columns = [cohort_col, reftime, event_col]

    cohort_msk = dataframe[cohort_col].isin(subgroups)
    plotdata = create_metric_timeseries(
        dataframe[cohort_msk][entity_keys + keep_columns],
        reftime,
        event_col,
        entity_keys,
        cohort_col,
        time_bounds=time_bounds,
        boolean_event=boolean_event,
        censor_threshold=censor_threshold,
    )

    if plotdata.empty:
        logger.error(
            "Insufficient data to plot; " + f"likely censoring all cohorts with threshold of {censor_threshold}."
        )
        return

    counts = None
    if plot_counts:
        counts = plotdata.groupby([reftime, cohort_col]).count()

    if ylabel is None:
        ylabel = event_col[:-6] if event_col.endswith("_Value") else event_col
    plotdata = plotdata.rename(columns={event_col: ylabel})

    # plot
    return plot.compare_series(
        plotdata,
        cohort_col=cohort_col,
        ref_str=reftime,
        event_col=event_col,
        ylabel=ylabel,
        counts=counts,
        show_legend=show_legend,
    )


# endregion

# region Templates


def _get_info_dict(plot_help: bool) -> dict[str, str | list[str]]:
    """
    Gets the required data dictionary for the info template.

    Parameters
    ----------
    plot_help : bool
        If True, displays additional information about available plotting utilities, by default False.

    Returns
    -------
    dict[str, str | list[str]]
        The data dictionary.
    """
    sg = Seismogram()

    info_vals = {
        "tables": [
            {
                "name": "predictions",
                "description": "Scores, features, configured demographics, and merged events for each prediction",
                "num_rows": sg.prediction_count,
                "num_cols": sg.feature_count,
            }
        ],
        "num_predictions": sg.prediction_count,
        "num_entities": sg.entity_count,
        "start_date": sg.start_time.strftime("%Y-%m-%d"),
        "end_date": sg.end_time.strftime("%Y-%m-%d"),
        "plot_help": plot_help,
    }

    return info_vals


@export
def show_info(plot_help: bool = False):
    """
    Displays information about the dataset

    Parameters
    ----------
    plot_help : bool, optional
        If True, displays additional information about available plotting utilities, by default False.
    """

    info_vals = _get_info_dict(plot_help)

    display(template.render_info_template(info_vals))


def _style_cohort_summaries(df: pd.DataFrame, attribute: str) -> Styler:
    """
    Adds required styling to a cohort dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The output of default_cohort_summaries().
    coattributehort : str
        The display name of the cohort.

    Returns
    -------
    Styler
        Styled dataframe.
    """
    df.index = df.index.rename("Cohort")
    style = df.style.format(precision=2)
    style = style.format_index(precision=2)
    return style.set_caption(f"Counts by {attribute}")


def _score_target_levels_and_index(
    selected_attribute: str, by_target: bool, by_score: bool
) -> tuple[list[str], list[str], list[str]]:
    """
    Gets the summary levels for the cohort summary tables.

    Parameters
    ----------
    selected_attribute : str
        The name of the current attribute to generate summary levels for.
    by_target : bool
        If True, adds an additional summary table to break down the population by target prevalence.
    by_score : bool
        If True, adds an additional summary table to break down the population by model output.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        groupby_groups: The levels in the dataframe to groupby when summarizing
        grab_groups: The columns in the dataframe to grab to summarize
        index_rename: The display names for the indices
    """
    sg = Seismogram()

    score_bins = sg.score_bins()
    cut_bins = pd.cut(sg.dataframe[sg.output], score_bins)

    groupby_groups = [selected_attribute]
    grab_groups = [selected_attribute]
    index_rename = ["Cohort"]

    if by_score:
        groupby_groups.append(cut_bins)
        grab_groups.append(sg.output)
        index_rename.append(sg.output)

    if by_target:
        groupby_groups.append(sg.target)
        grab_groups.append(sg.target)
        index_rename.append(sg.target.strip("_Value"))

    return groupby_groups, grab_groups, index_rename


def _style_score_target_cohort_summaries(df: pd.DataFrame, index_rename: list[str], cohort: str) -> Styler:
    """
    Adds required styling to a multiple summary level cohort summary dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The output of score_target_cohort_summaries.
    index_rename : list[str]
        The display names to put in the index from _score_target_levels_and_index().
    cohort : str
        The display name of the cohort.

    Returns
    -------
    Styler
        Styled dataframe.
    """
    df.index = df.index.rename(index_rename)
    style = df.style.format(precision=2)
    style = style.format_index(precision=2)
    return style.set_caption(f"Counts by {cohort}")


def _get_cohort_summary_dataframes(by_target: bool, by_score: bool) -> dict[str, list[str]]:
    """
    Gets the formatted summary cohort dataframes to display in the cohort summary template.

    Parameters
    ----------
    by_target : bool
        If True, adds an additional summary table to break down the population by target prevalence.
    by_score : bool
        If True, adds an additional summary table to break down the population by model output.

    Returns
    -------
    dict[str, list[str]]
        The dictionary, indexed by cohort attribute (e.g. Race), of summary dataframes.
    """
    sg = Seismogram()

    dfs: dict[str, list[str]] = {}

    available_cohort_groups = sg.available_cohort_groups

    for attribute, options in available_cohort_groups.items():
        df = default_cohort_summaries(sg.dataframe, attribute, options, sg.config.entity_id)
        styled = _style_cohort_summaries(df, attribute)

        dfs[attribute] = [styled.to_html()]

        if by_score or by_target:
            groupby_groups, grab_groups, index_rename = _score_target_levels_and_index(attribute, by_target, by_score)

            results = score_target_cohort_summaries(sg.dataframe, groupby_groups, grab_groups, sg.config.entity_id)
            results_styled = _style_score_target_cohort_summaries(results, index_rename, attribute)

            dfs[attribute].append(results_styled.to_html())

    return dfs


@export
def show_cohort_summaries(by_target: bool = False, by_score: bool = False):
    """
    Displays a table of selectable attributes and their associated counts.
    Use `by_target` and `by_score` to add additional summary levels to the tables.

    Parameters
    ----------
    by_target : bool, optional
        If True, adds an additional summary table to break down the population by target prevalence, by default False.
    by_score : bool, optional
        If True, adds an additional summary table to break down the population by model output, by default False.
    """
    dfs = _get_cohort_summary_dataframes(by_target, by_score)

    display(template.render_cohort_summary_template(dfs))


# endregion
