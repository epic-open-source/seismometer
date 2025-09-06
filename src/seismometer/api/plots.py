import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from IPython.display import HTML, SVG

import seismometer.plot as plot
from seismometer.controls.decorators import disk_cached_html_and_df_segment
from seismometer.core.autometrics import AutomationManager, store_call_parameters
from seismometer.core.decorators import export
from seismometer.data import get_cohort_data, get_cohort_performance_data, metric_apis
from seismometer.data import pandas_helpers as pdh
from seismometer.data.filter import FilterRule
from seismometer.data.performance import (
    STATNAMES,
    BinaryClassifierMetricGenerator,
    assert_valid_performance_metrics_df,
    calculate_bin_stats,
    calculate_eval_ci,
)
from seismometer.data.timeseries import create_metric_timeseries
from seismometer.html import template
from seismometer.seismogram import Seismogram

logger = logging.getLogger("seismometer")


@export
def plot_cohort_hist():
    """Display a histogram plot of predicted probabilities for all cohorts in the selected attribute."""
    sg = Seismogram()
    cohort_col = sg.selected_cohort[0]
    subgroups = sg.selected_cohort[1]
    censor_threshold = sg.censor_threshold
    return _plot_cohort_hist(sg.dataframe, sg.target, sg.output, cohort_col, subgroups, censor_threshold)


@disk_cached_html_and_df_segment
@export
def plot_cohort_group_histograms(
    cohort_col: str, subgroups: list[str], target_column: str, score_column: str
) -> tuple[HTML, pd.DataFrame]:
    """
    Generate a histogram plot of predicted probabilities for each subgroup in a cohort.

    Parameters
    ----------
    cohort_col : str
        column for cohort splits
    subgroups : list[str]
        column values to split by
    target_column : str
        target column
    score_column : str
        score column

    Returns
    -------
    tuple[HTML, pd.DataFrame]
        html visualization of the histogram and the data used to generate it
    """
    sg = Seismogram()
    target_column = pdh.event_value(target_column)
    return _plot_cohort_hist(sg.dataframe, target_column, score_column, cohort_col, subgroups, sg.censor_threshold)


def _plot_cohort_hist(
    dataframe: pd.DataFrame,
    target: str,
    output: str,
    cohort_col: str,
    subgroups: list[str],
    censor_threshold: int = 10,
) -> tuple[HTML, pd.DataFrame]:
    """
    Creates an HTML segment displaying a histogram of predicted probabilities for each cohort.

    Parameters
    ----------
    dataframe : pd.DataFrame
        data source
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

    # We use this to display probabilities.
    dataframe = FilterRule.isin(target, (0, 1)).filter(dataframe)

    cData = get_cohort_data(dataframe, cohort_col, proba=output, true=target, splits=subgroups)

    # filter by groups by size
    cCount = cData["cohort"].value_counts()
    good_groups = cCount.loc[cCount > censor_threshold].index
    cData = cData.loc[cData["cohort"].isin(good_groups)]

    if len(cData.index) == 0:
        return template.render_censored_plot_message(censor_threshold), cData

    bin_count = 20
    bins = np.histogram_bin_edges(cData["pred"], bins=bin_count)

    try:
        svg = plot.cohorts_vertical(cData, plot.histogram_stacked, func_kws={"show_legend": False, "bins": bins})
        title = f"Predicted Probabilities by {cohort_col}"
        return template.render_title_with_image(title, svg), cData
    except Exception as error:
        return template.render_title_message("Error", f"Error: {error}"), pd.DataFrame()


@export
def plot_leadtime_enc(score=None, ref_time=None, target_event=None):
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
    target_event = pdh.event_value(target_event or sg.target)
    target_zero = pdh.event_time(target_event or sg.time_zero)
    max_hours = sg.event_aggregation_window_hours(target_event)
    threshold = sg.thresholds[0]
    target_data = FilterRule.isin(target_event, (0, 1)).filter(sg.dataframe)

    return _plot_leadtime_enc(
        target_data,
        sg.entity_keys,
        target_event,
        target_zero,
        score,
        threshold,
        ref_time,
        cohort_col,
        subgroups,
        max_hours,
        x_label,
        censor_threshold,
    )


@store_call_parameters(cohort_col="cohort_col", subgroups="subgroups")
@disk_cached_html_and_df_segment
@export
def plot_cohort_lead_time(
    cohort_col: str, subgroups: list[str], event_column: str, score_column: str, threshold: float
) -> tuple[HTML, pd.DataFrame]:
    """
    Plots a lead times between the first positive prediction give an threshold and an event.

    Parameters
    ----------
    cohort_col : str
        cohort column name
    subgroups : list[str]
        subgroups of interest in the cohort column
    event_column : str
        event column name
    score_column : str
        score column name
    threshold : float
        _description_

    Returns
    -------
    HTML
        _description_
    """
    sg = Seismogram()
    x_label = "Lead Time (hours)"
    event_value = pdh.event_value(event_column)
    event_time = pdh.event_time(event_column)
    target_data = FilterRule.isin(event_value, (0, 1)).filter(sg.dataframe)
    max_hours = sg.event_aggregation_window_hours(event_value)

    return _plot_leadtime_enc(
        target_data,
        sg.entity_keys,
        event_value,
        event_time,
        score_column,
        threshold,
        sg.predict_time,
        cohort_col,
        subgroups,
        max_hours,
        x_label,
        sg.censor_threshold,
    )


def _plot_leadtime_enc(
    dataframe: pd.DataFrame,
    entity_keys: list[str],
    target_event: str,
    target_zero: str,
    score: str,
    threshold: list[float],
    ref_time: str,
    cohort_col: str,
    subgroups: list[any],
    max_hours: int,
    x_label: str,
    censor_threshold: int = 10,
) -> tuple[HTML, pd.DataFrame]:
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
    entity_keys : list[str]
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
    tuple[HTML, pd.DataFrame]
        Lead time plot and the data used to generate it
    """
    if target_event not in dataframe:
        msg = f"Target event ({target_event}) not found in dataset. Cannot plot leadtime."
        logger.error(msg)
        return template.render_title_message("Error", msg), pd.DataFrame()

    if target_zero not in dataframe:
        msg = f"Target event time-zero ({target_zero}) not found in dataset. Cannot plot leadtime."
        logger.error(msg)
        return template.render_title_message("Error", msg), pd.DataFrame()

    summary_data = dataframe[dataframe[target_event] == 1]
    if len(summary_data.index) == 0:
        msg = f"No positive events ({target_event}=1) were found"
        logger.error(msg)
        return template.render_title_message("Error", msg), pd.DataFrame()

    cohort_mask = summary_data[cohort_col].isin(subgroups)
    threshold_mask = summary_data[score] > threshold

    # summarize to first score
    summary_data = pdh.event_score(
        summary_data[cohort_mask & threshold_mask],
        entity_keys,
        score=score,
        ref_time=target_zero,
        aggregation_method="first",
    )
    if summary_data is not None and len(summary_data) > censor_threshold:
        summary_data = summary_data[[target_zero, ref_time, cohort_col]]
    else:
        return template.render_censored_plot_message(censor_threshold), pd.DataFrame()

    # filter by group size
    counts = summary_data[cohort_col].value_counts()
    good_groups = counts.loc[counts > censor_threshold].index
    summary_data = summary_data.loc[summary_data[cohort_col].isin(good_groups)]

    metric_apis.log_quantiles(
        summary_data, "Time Lead", score, target_zero, ref_time, cohort_col, good_groups, threshold
    )

    if len(summary_data.index) == 0:
        return template.render_censored_plot_message(censor_threshold), summary_data

    # Truncate to minute but plot hour
    summary_data[x_label] = (summary_data[ref_time] - summary_data[target_zero]).dt.total_seconds() // 60 / 60

    title = f'Lead Time from {score.replace("_", " ")} to {(target_zero).replace("_", " ")}'
    rows = summary_data[cohort_col].nunique()
    svg = plot.leadtime_violin(summary_data, x_label, cohort_col, xmax=max_hours, figsize=(9, 1 + rows))
    return template.render_title_with_image(title, svg), summary_data


@store_call_parameters(cohort_col="cohort_col", subgroups="subgroups")
@disk_cached_html_and_df_segment
@export
def plot_cohort_evaluation(
    cohort_col: str,
    subgroups: list[str],
    target_column: str,
    score_column: str,
    thresholds: list[float],
    per_context: bool = False,
) -> tuple[HTML, pd.DataFrame]:
    """
    Plots model performance metrics split by on a cohort attribute.

    Parameters
    ----------
    cohort_col : str
        cohort column name
    subgroups : list[str]
        subgroups of interest in the cohort column
    target_column : str
        target column
    score_column : str
        score column
    thresholds : list[float]
        thresholds to highlight
    per_context : bool, optional
        if scores should be grouped, by default False

    Returns
    -------
    HTML
        an html visualization of the model performance metrics
    """
    sg = Seismogram()
    target_event = pdh.event_value(target_column)
    target_data = FilterRule.isin(target_event, (0, 1)).filter(sg.dataframe)
    return _plot_cohort_evaluation(
        target_data,
        sg.entity_keys,
        target_event,
        score_column,
        thresholds,
        cohort_col,
        subgroups,
        sg.censor_threshold,
        per_context,
    )


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
    threshold_col: str = "Threshold",
    ref_time: str = None,
) -> tuple[HTML, pd.DataFrame]:
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
    threshold_col: str, optional
        Which column should be called the threshold column.
    ref_time : str, optional
        reference time column used for aggregation when per_context_id is True and aggregation_method is time-based
    Returns
    -------
    HTML
        _description_
    """
    data = pdh.get_model_scores(
        dataframe,
        entity_keys,
        score_col=output,
        ref_time=ref_time,
        ref_event=target,
        aggregation_method=aggregation_method,
        per_context_id=per_context_id,
    )

    plot_data = get_cohort_performance_data(
        data, cohort_col, proba=output, true=target, splits=subgroups, censor_threshold=censor_threshold
    )
    recorder = metric_apis.OpenTelemetryRecorder(metric_names=STATNAMES, name=f"Performance split by {cohort_col}")
    base_attributes = {"target": target, "score": output}
    # Go through all cohort values, by means of:
    cohort_categories = list(set(plot_data["cohort"]))
    recorder.log_by_column(
        df=plot_data,
        col_name=threshold_col,
        cohorts={"cohort": cohort_categories},
        base_attributes=base_attributes,
        col_values=[t * 100 for t in thresholds],
    )
    try:
        assert_valid_performance_metrics_df(plot_data)
    except ValueError:
        return template.render_censored_plot_message(censor_threshold), plot_data
    svg = plot.cohort_evaluation_vs_threshold(plot_data, cohort_feature=cohort_col, highlight=thresholds)
    title = f"Model Performance Metrics on {cohort_col} across Thresholds"
    return template.render_title_with_image(title, svg), plot_data


@store_call_parameters(cohort_dict="cohort_dict")
@disk_cached_html_and_df_segment
@export
def plot_model_evaluation(
    cohort_dict: dict[str, tuple[Any]],
    target_column: str,
    score_column: str,
    thresholds: list[float],
    per_context: bool = False,
) -> tuple[HTML, pd.DataFrame]:
    """
    Generates a 2x3 plot showing the performance of a model.

    This includes the ROC, recall vs predicted condition prevalence, calibration,
    PPV vs sensitivity, sensitivity/specificity/ppv, and a histogram.

    Parameters
    ----------
    cohort_dict : dict[str, tuple[Any]]
        dictionary of cohort columns and values used to subselect a population for evaluation
    target_column : str
        target column
    score_column : str
        score column
    thresholds : list[float]
        thresholds to highlight
    per_context : bool, optional
        if scores should be grouped, by default False

    Returns
    -------
    HTML
        an html visualization of the model evaluation metrics
    """
    sg = Seismogram()
    cohort_filter = FilterRule.from_cohort_dictionary(cohort_dict)
    data = cohort_filter.filter(sg.dataframe)
    target_event = pdh.event_value(target_column)
    target_data = FilterRule.isin(target_event, (0, 1)).filter(data)
    aggregation_method = sg.event_aggregation_method(sg.target)
    ref_time = sg.predict_time
    return _model_evaluation(
        target_data,
        sg.entity_keys,
        target_column,
        target_event,
        score_column,
        thresholds,
        sg.censor_threshold,
        per_context,
        aggregation_method,
        ref_time,
        cohort=cohort_dict,
    )


def _model_evaluation(
    dataframe: pd.DataFrame,
    entity_keys: list[str],
    target_event: str,
    target: str,
    score_col: str,
    thresholds: Optional[list[float]],
    censor_threshold: int = 10,
    per_context_id: bool = False,
    aggregation_method: str = "max",
    ref_time: Optional[str] = None,
    cohort: dict = {},
) -> tuple[HTML, pd.DataFrame]:
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
    score_col : str
        score column
    thresholds : Optional[list[float]]
        model thresholds
    censor_threshold : int, optional
        minimum rows to allow in a plot, by default 10
    per_context_id : bool, optional
        report only the max score for a given entity context, by default False
    aggregation_method : str, optional
        method to reduce multiple scores into a single value before calculation of performance, by default "max"
        ignored if per_context_id is False
    ref_time : Optional[str], optional
        reference time column used for aggregation when per_context_id is True and aggregation_method is time-based

    Returns
    -------
    HTML
        Plot of model evaluation metrics
    """
    data = pdh.get_model_scores(
        dataframe,
        entity_keys,
        score_col=score_col,
        ref_time=ref_time,
        ref_event=target_event,
        aggregation_method=aggregation_method,
        per_context_id=per_context_id,
    )

    # Validate
    requirements = FilterRule.isin(target, (0, 1)) & FilterRule.notna(score_col)
    data = requirements.filter(data)
    if len(data.index) < censor_threshold:
        return template.render_censored_plot_message(censor_threshold), data
    if (lcount := data[target].nunique()) != 2:
        return (
            template.render_title_message(
                "Evaluation Error", f"Model Evaluation requires exactly two classes but found {lcount}"
            ),
            data,
        )

    # stats and ci handle percentile/percentage independently - evaluation wants 0-100 for displays
    stats = calculate_bin_stats(data[target], data[score_col])
    ci_data = calculate_eval_ci(stats, data[target], data[score_col], conf=0.95, force_percentages=True)
    recorder = metric_apis.OpenTelemetryRecorder(metric_names=STATNAMES, name="Model Performance")
    params = {"target_column": target, "score_column": score_col}
    for t in thresholds:
        recorder.populate_metrics(
            attributes=params | cohort | {"threshold": t}, metrics=stats[stats["Threshold"] == t * 100]
        )
    am = AutomationManager()
    for metric in stats.columns:
        log_all = am.get_metric_config(metric)["log_all"]
        if log_all:
            recorder.populate_metrics(
                attributes=params | cohort,
                metrics={metric: stats[[metric, "Threshold"]].set_index("Threshold").to_dict()},
            )
    title = f"Overall Performance for {target_event} (Per {'Encounter' if per_context_id else 'Observation'})"
    svg = plot.evaluation(
        stats,
        ci_data=ci_data,
        truth=data[target],
        output=data[score_col].values,
        show_thresholds=True,
        highlight=thresholds,
    )
    return template.render_title_with_image(title, svg), data


@store_call_parameters
@export
def plot_trend_intervention_outcome() -> HTML:
    """
    Plots two timeseries based on selectors; an outcome and then an intervention.

    Makes use of the cohort selectors as well intervention and outcome selectors for which data to use.
    Uses the configuration for comparison_time as the reference time for both plots.
    """
    sg = Seismogram()
    if not sg.outcome or not sg.intervention:
        return HTML("No outcome or intervention configured.")
    return _plot_trend_intervention_outcome(
        sg.dataframe,
        sg.entity_keys,
        sg.selected_cohort[0],
        sg.selected_cohort[1],
        sg.outcome,
        sg.intervention,
        sg.comparison_time or sg.predict_time,
        sg.censor_threshold,
    )


@disk_cached_html_and_df_segment
@export
def plot_intervention_outcome_timeseries(
    cohort_col: str,
    subgroups: list[str],
    outcome: str,
    intervention: str,
    reference_time_col: str,
    censor_threshold: int = 10,
) -> tuple[HTML, pd.DataFrame]:
    """
    Plots two timeseries based on an outcome and an intervention.

    cohort_col : str
        column name for the cohort to split on
    subgroups : list[str]
        values of interest in the cohort column
    outcome : str
        outcome event time column
    intervention : str
        intervention event time column
    reference_time_col : str
        reference time column for alignment
    censor_threshold : int, optional
        minimum rows to allow in a plot, by default 10

    Returns
    -------
    HTML
        Plot of two timeseries
    """
    sg = Seismogram()
    return _plot_trend_intervention_outcome(
        sg.dataframe,
        sg.entity_keys,
        cohort_col,
        subgroups,
        outcome,
        intervention,
        reference_time_col,
        censor_threshold,
    )


def _plot_trend_intervention_outcome(
    dataframe: pd.DataFrame,
    entity_keys: list[str],
    cohort_col: str,
    subgroups: list[str],
    outcome: str,
    intervention: str,
    reftime: str,
    censor_threshold: int = 10,
) -> tuple[HTML, pd.DataFrame]:
    """
    Plots two timeseries based on selectors; an outcome and then an intervention.

    Parameters
    ----------
    dataframe : pd.DataFrame
        data source
    entity_keys : list[str]
        columns to use for aggregation
    cohort_col : str
        column name for the cohort to split on
    subgroups : list[str]
        values of interest in the cohort column
    outcome : str
        model score
    intervention : str
        intervention event time column
    reftime : str
        reference time column for alignment
    censor_threshold : int, optional
        minimum rows to allow in a plot, by default 10

    Returns
    -------
    HTML
        Plot of two timeseries
    """
    time_bounds = (dataframe[reftime].min(), dataframe[reftime].max())  # Use the full time range
    show_legend = True
    outcome_plot, intervention_plot = None, None

    try:
        intervention_col = pdh.event_value(intervention)
        intervention_svg = _plot_ts_cohort(
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
        intervention_plot = template.render_title_with_image("Intervention: " + intervention, intervention_svg)
        show_legend = False
    except IndexError:
        intervention_plot = template.render_title_message(
            "Missing Intervention", f"No intervention timeseries plotted; No events with name {intervention}."
        )

    try:
        outcome_col = pdh.event_value(outcome)
        outcome_svg = _plot_ts_cohort(
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
        outcome_plot = template.render_title_with_image("Outcome: " + outcome, outcome_svg)
    except IndexError:
        outcome_plot = template.render_title_message(
            "Missing Outcome", f"No outcome timeseries plotted; No events with name {outcome}."
        )

    return HTML(outcome_plot.data + intervention_plot.data), dataframe


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
        # all groups have been censored.
        return template.render_censored_plot_message(censor_threshold)

    counts = None
    if plot_counts:
        counts = plotdata.groupby([reftime, cohort_col]).count()

    if ylabel is None:
        ylabel = pdh.event_name(event_col)
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


@store_call_parameters(cohort_dict="cohort_dict")
@disk_cached_html_and_df_segment
@export
def plot_model_score_comparison(
    cohort_dict: dict[str, tuple[Any]], target: str, scores: tuple[str], *, per_context: bool
) -> tuple[HTML, pd.DataFrame]:
    """
    Plots a comparison of model scores for a given subpopulation.

    Parameters
    ----------
    cohort_dict : dict[str,tuple[Any]]
        The cohort values that comprise the subpopulation of interest.
    target : str
        The target column.
    scores : tuple[str]
        The score columns to compare.
    per_context : bool
        If True, limits data to one row per context_id.
    """
    sg = Seismogram()

    cohort_filter = FilterRule.from_cohort_dictionary(cohort_dict)
    data = cohort_filter.filter(sg.dataframe)
    target_event = pdh.event_value(target)
    dataframe = FilterRule.isin(target_event, (0, 1)).filter(data)

    # Need a dataframe with three columns, ScoreName, Score, and Target
    # Index - one copy of the index for each score name (to allow three columns of real scores.)
    # ScoreName - the cohort column (which score was used)
    # Score - the score value
    # Target - the target value

    data = []
    for score in scores:
        if per_context:
            one_score_data = pdh.event_score(
                dataframe, sg.entity_keys, score=score, ref_event=target_event, aggregation_method="max"
            )[[score, target_event]]
        else:
            one_score_data = dataframe[[score, target_event]].copy()
        one_score_data["ScoreName"] = score
        one_score_data.rename(columns={score: "Score"}, inplace=True)
        data.append(one_score_data)
    data = pd.concat(data, axis=0, ignore_index=True)
    data["ScoreName"] = data["ScoreName"].astype("category")

    plot_data = get_cohort_performance_data(
        data,
        "ScoreName",
        proba=data["Score"],
        true=data[target_event],
        splits=list(scores),
        censor_threshold=sg.censor_threshold,
    )
    recorder = metric_apis.OpenTelemetryRecorder(metric_names=STATNAMES, name="Model Score Comparison")
    recorder.log_by_column(
        df=plot_data, col_name="Threshold", cohorts={"cohort": scores}, base_attributes={"Target Column": target}
    )
    try:
        assert_valid_performance_metrics_df(plot_data)
    except ValueError:
        return template.render_censored_plot_message(sg.censor_threshold), plot_data
    svg = plot.cohort_evaluation_vs_threshold(plot_data, cohort_feature="ScoreName")
    title = f"Model Metrics: {', '.join(scores)} vs {target}"
    return template.render_title_with_image(title, svg), plot_data


@disk_cached_html_and_df_segment
@export
def plot_model_target_comparison(
    cohort_dict: dict[str, tuple[Any]], targets: tuple[str], score: str, *, per_context: bool
) -> tuple[HTML, pd.DataFrame]:
    """
    Plots a comparison of model targets for a given score and subpopulation.

    Parameters
    ----------
    cohort_dict : dict[str,tuple[Any]]
        The cohort values that comprise the subpopulation of interest.
    targets : tuple[str]
        The target columns to compare.
    scores: str
        The score column
    per_context : bool
        If True, limits data to one row per context_id.
    """
    sg = Seismogram()

    cohort_filter = FilterRule.from_cohort_dictionary(cohort_dict)
    source_data = cohort_filter.filter(sg.dataframe)

    # Need a dataframe with three columns, ScoreName, Score, and Target
    # Index - one copy of the index for each score name (to allow three columns of real scores.)
    # TargetName - the cohort column (which target was used)
    # Score - the score value
    # Target - the target value

    data = []
    for target in targets:
        target_event = pdh.event_value(target)
        dataframe = FilterRule.isin(target_event, (0, 1)).filter(source_data)
        if per_context:
            one_score_data = pdh.event_score(
                dataframe, sg.entity_keys, score=score, ref_event=target_event, aggregation_method="max"
            )[[score, target_event]]
        else:
            one_score_data = dataframe[[score, target_event]].copy()
        one_score_data["TargetName"] = target
        one_score_data.rename(columns={target_event: "Target"}, inplace=True)
        data.append(one_score_data)
    data = pd.concat(data, axis=0, ignore_index=True)
    data["TargetName"] = data["TargetName"].astype("category")

    plot_data = get_cohort_performance_data(
        data,
        "TargetName",
        proba=data[score],
        true=data["Target"],
        splits=list(targets),
        censor_threshold=sg.censor_threshold,
    )
    recorder = metric_apis.OpenTelemetryRecorder(metric_names=STATNAMES, name="Model Score Comparison")
    recorder.log_by_column(
        df=plot_data, col_name="Threshold", cohorts={"cohort": targets}, base_attributes={"Score Column": score}
    )
    try:
        assert_valid_performance_metrics_df(plot_data)
    except ValueError:
        return template.render_censored_plot_message(sg.censor_threshold), plot_data
    svg = plot.cohort_evaluation_vs_threshold(plot_data, cohort_feature="ScoreName")
    title = f"Model Metrics: {', '.join(targets)} vs {score}"
    return template.render_title_with_image(title, svg), plot_data


# region Explore Any Metric (NNT, etc)
@store_call_parameters(cohort_dict="cohort_dict")
@disk_cached_html_and_df_segment
@export
def plot_binary_classifier_metrics(
    metric_generator: BinaryClassifierMetricGenerator,
    metrics: str | list[str],
    cohort_dict: dict[str, tuple[Any]],
    target: str,
    score_column: str,
    *,
    per_context: bool = False,
    table_only: bool = False,
) -> tuple[HTML, pd.DataFrame]:
    """
    Generates a plot with model metrics.

    Parameters
    ----------
    metric_generator: BinaryClassifierMetricGenerator
        class that creates metrics for a model
    metrics: str | list[str]
        subset of metrics to display
    cohort_dict : dict[str, tuple[Any]]
        dictionary of cohort columns and values used to subselect a population for evaluation
    target : str
        name of the target
    score_column : str
        score column
    per_context : bool, optional
        if scores should be grouped, by default False
    table_only : bool, optional
        if only the table should be displayed, by default False

    Returns
    -------
    HTML
        an html visualization of the model evaluation metrics
    """
    sg = Seismogram()
    cohort_filter = FilterRule.from_cohort_dictionary(cohort_dict)
    data = cohort_filter.filter(sg.dataframe)
    target_event = pdh.event_value(target)
    target_data = FilterRule.isin(target_event, (0, 1)).filter(data)
    return binary_classifier_metric_evaluation(
        metric_generator,
        metrics,
        target_data,
        sg.entity_keys,
        target_event,
        score_column,
        sg.censor_threshold,
        per_context,
        sg.event_aggregation_method(target),
        sg.predict_time,
        table_only=table_only,
    )


def _autometric_plot_binary_classifier_metrics(
    metric_generator: float,
    metrics: str | list[str],
    cohort_dict: dict[str, tuple[Any]],
    target: str,
    score_column: str,
    *,
    per_context: bool = False,
    table_only: bool = False,
):
    """Serves only as a wrapper of plot_binary_classifier_metrics so that
    we don't have to serialize a metric generator object.

    Parameters
    ----------
    metric_generator: float between 0 and 1
        Probability of a treatment being effective. This is named metric_generator
        instead of rho because it is an internal method and having the object be
        the same name as what it is replacing in the real method makes
        serialization much easier.
    """
    bcmg = BinaryClassifierMetricGenerator(rho=metric_generator)
    plot_binary_classifier_metrics(
        bcmg, metrics, cohort_dict, target, score_column, per_context=per_context, table_only=table_only
    )


def binary_classifier_metric_evaluation(
    metric_generator: BinaryClassifierMetricGenerator,
    metrics: str | list[str],
    dataframe: pd.DataFrame,
    entity_keys: list[str],
    target: str,
    score_col: str,
    censor_threshold: int = 10,
    per_context_id: bool = False,
    aggregation_method: str = "max",
    ref_time: str = None,
    table_only: bool = False,
) -> tuple[HTML, pd.DataFrame]:
    """
    plots common model evaluation metrics

    Parameters
    ----------
    dataframe : pd.DataFrame
        source data
    entity_keys : list[str]
        columns to use for aggregation
    target : str
        target column
    score_col : str
        score column
    censor_threshold : int, optional
        minimum rows to allow in a plot, by default 10
    per_context_id : bool, optional
        report only the max score for a given entity context, by default False
    aggregation_method : str, optional
        method to reduce multiple scores into a single value before calculation of performance, by default "max"
        ignored if per_context_id is False
    ref_time : str, optional
        reference time column used for aggregation when per_context_id is True and aggregation_method is time-based
    table_only : bool, optional
        if only the table should be displayed, by default False

    Returns
    -------
    HTML
        Plot of model evaluation metrics
    """
    data = pdh.get_model_scores(
        dataframe,
        entity_keys,
        score_col=score_col,
        ref_time=ref_time,
        ref_event=target,
        aggregation_method=aggregation_method,
        per_context_id=per_context_id,
    )
    # Validate
    requirements = FilterRule.isin(target, (0, 1)) & FilterRule.notna(score_col)
    data = requirements.filter(data)
    if len(data.index) < censor_threshold:
        return template.render_censored_plot_message(censor_threshold), data
    if (lcount := data[target].nunique()) != 2:
        return (
            template.render_title_message(
                "Evaluation Error", f"Model Evaluation requires exactly two classes but found {lcount}"
            ),
            data,
        )
    if isinstance(metrics, str):
        metrics = [metrics]
    stats = metric_generator.calculate_binary_stats(data, target, score_col, metrics)[0]
    recorder = metric_apis.OpenTelemetryRecorder(name="Binary Classifier Evaluations", metric_names=metrics)
    attributes = {"score_col": score_col, "target": target}
    am = AutomationManager()
    for metric in metrics:
        log_all = am.get_metric_config(metric)["log_all"]
        if log_all:
            recorder.populate_metrics(attributes=attributes, metrics={metric: stats[metric].to_dict()})
    if table_only:
        return HTML(stats[metrics].T.to_html()), data
    return plot.binary_classifier.plot_metric_list(stats, metrics), data


# endregion
