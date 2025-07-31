# -*- coding: utf-8 -*-
from numbers import Number
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

import seismometer.plot.mpl._lines as lines
from seismometer.data.performance import as_probabilities

from .decorators import can_draw_empty_perf, export, model_plot

SeriesOrArray = pd.Series | np.ndarray
DataFrameOrArray = pd.DataFrame | np.ndarray
Axes = plt.Axes | plt.Subplot


@export
@model_plot
def evaluation(
    stats: pd.DataFrame,
    *,
    ci_data: dict,
    truth: Optional[pd.Series] = None,
    output: Optional[pd.Series] = None,
    show_thresholds: Optional[bool] = True,
    highlight: Optional[list] = None,
) -> plt.Figure:
    """
    Generates a 2x3 plot showing the performance of a model.

    This includes the ROC, recall vs predicted condition prevalence, calibration,
    PPV vs sensitivity, sensitivity/specificity/ppv, and a histogram.

    Parameters
    ----------
    stats : pd.DataFrame
        Table of performance metrics, of the form given by calculate_bin_stats.
    ci_data : dict
        A required dictionary of the confidence interval information for containing plots.
        Expects keys 'roc', 'pr', and 'conf'.
    truth : Optional[pd.Series], optional
        A series of the true labels, needed for calibration plot, by default None.
    output : Optional[pd.Series], optional
        A series of the model output, needed for calibration plot, by default None.
    show_thresholds : Optional[bool], optional
        If True, shows thresholds on the ROC and PPV vs Sensitivity plots, by default None.
    highlight : Optional[list[float]], optional
        An optional list of thresholds to highlight on the plots, by default None.

    Returns
    -------
    The matplotlib figure.
    """
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    prevalence = stats.loc[0, ["TP", "FN"]].sum()
    if prevalence != 0:
        prevalence = prevalence / stats.loc[0, ["TP", "FP", "TN", "FN"]].sum()

    roc_data = ci_data["roc"]
    singleROC(
        roc_data["TPR"],
        roc_data["FPR"],
        roc_data["Threshold"],
        conf_region=roc_data["region"],
        conf_interval=roc_data["interval"],
        axis=ax1,
        annotate=show_thresholds,
        highlight=highlight,
    )
    recall_condition(
        stats["Flag Rate"],
        stats["Sensitivity"],
        stats["Threshold"],
        prevalence=prevalence,
        axis=ax2,
        annotate=show_thresholds,
        show_reference=True,
        highlight=highlight,
    )
    calibration(truth, output, axis=ax3, highlight=highlight)
    ppv_vs_sensitivity(
        stats["PPV"],
        stats["Sensitivity"],
        stats["Threshold"],
        conf_interval=ci_data["pr"]["interval"],
        axis=ax4,
        highlight=highlight,
    )
    performance_metrics(stats, conf=ci_data["conf"]["metric"], axis=ax5, highlight=highlight)
    histogram_stacked(truth, output, axis=ax6, highlight=highlight)
    plt.subplots_adjust(bottom=0.1, top=0.96, left=0.1, right=0.95, wspace=0.3, hspace=0.3)
    return fig


##########################################################################
# Classification Plots
##########################################################################


@export
@model_plot
def singleROC(
    tpr: pd.Series,
    fpr: pd.Series,
    thresholds: pd.Series,
    conf_region: Optional["_RocRegionResults"] = None,
    conf_interval: Optional["ValueWithCI"] = None,
    *,
    annotate: bool = False,
    highlight: Optional[list[Number]] = None,
    axis: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Creates an ROC plot.

    Has the option to include a confidence region visually and interval in the legend.
    Can have annotations for thresholds generally or highlight specific ones.

    Parameters
    ----------
    tpr : pd.Series
        Series of true positive rates.
    fpr : pd.Series
        Series of false positive rates.
    thresholds : pd.Series
        Series of thresholds, used for annotations.
    conf_region : Optional[_RocRegionResults], optional
        When specified, will display a confidence region on the curve, by default None.
    conf_interval : Optional[ValueWithCI], optional
        When specified, will display the confidence interval on the legend, by default None.
    annotate : bool, optional
        A flag to add annotations to the plot, by default False.
    highlight : Optional[list[Number]], optional
        A list of thresholds to highlight on the plot, by default None.
    axis : Optional[plt.Axes], optional
        The matplotlib axis to draw, by default None; creates a new figure.
    """
    if conf_interval is None:
        modelLabel = f"AUROC = {metrics.auc(fpr, tpr):0.2f}"
    else:
        modelLabel = f"AUROC = {conf_interval.value:0.2f} ({conf_interval.lower:0.2f}, {conf_interval.upper:0.2f})"

    if conf_region is not None:
        lines.roc_region_plot(
            axis, conf_region.lower_fpr, conf_region.lower_tpr, conf_region.upper_fpr, conf_region.upper_tpr
        )
    lines.roc_plot(axis, fpr, tpr, label=modelLabel)

    if annotate:
        lines._add_radial_score_labels(axis, fpr, tpr, thresholds, highlight=highlight)
    return axis.get_figure()


@export
@model_plot
def recall_condition(
    ppcr: pd.Series,
    recall: pd.Series,
    thresholds: pd.Series,
    prevalence: Number,
    *,
    show_reference: bool = False,
    annotate: bool = False,
    highlight: Optional[list[Number]] = None,
    axis: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plots the recall of a model against the predicted condition rate.

    Parameters
    ----------
    ppcr : pd.Series
        The predicted positive condition rate.
    recall : pd.Series
        The recall (sensitivity) of the model.
    thresholds : pd.Series
        The thresholds of the model, if annotations are desired.
    prevalence : Number
        The prevalence of the condition in the dataset, plotted as a reference line.
    show_reference : bool, optional
        A flag to show the prevalence as a reference line, by default False.
    annotate : bool, optional
        A flag to add annotations to the plot, by default False.
    highlight : Optional[list[Number]], optional
        A list of thresholds to highlight on the plot, by default None.
    axis : Optional[plt.Axes], optional
        The matplotlib axis to draw, by default None; creates a new figure.
    """
    lines.recall_condition_plot(axis, ppcr, recall, prevalence, show_reference)

    if annotate:
        lines._add_radial_score_labels(axis, ppcr, recall, thresholds, highlight=highlight)
    return axis.get_figure()


@export
@model_plot
def calibration(
    truth: pd.Series, output: pd.Series, *, highlight: Optional[list[Number]] = None, axis: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plots the calibration curve for the model.

    Parameters
    ----------
    truth : pd.Series
        The true labels.
    output : pd.Series
        The model output predictions.
    highlight : Optional[list[Number]], optional
        A list of thresholds to highlight on the plot, by default None.
    axis : Optional[plt.Axes], optional
        The matplotlib axis to draw, by default None; creates a new figure.
    """
    # Normalize scores for binary classfiers
    if (output > 1).any():
        output = output / 100.0

    from sklearn.calibration import calibration_curve

    fraction_positive, mean_predicted = calibration_curve(truth, output, n_bins=10)
    lines.reliability_plot(axis, mean_predicted, fraction_positive)
    if highlight is not None:
        lines.vertical_threshold_lines(axis, highlight, color_alerts=True, legend_position="upper right")
    return axis.get_figure()


@export
@model_plot
def histogram_stacked(
    y_label: pd.Series,
    output: pd.Series,
    *,
    highlight: Optional[list[Number]] = None,
    bins: int | Iterable = 20,
    axis: Optional[plt.Axes] = None,
    show_legend: bool = True,
) -> plt.Figure:
    """
    Plots a stacked histogram of the model output by class.

    Parameters
    ----------
    y_label : pd.Series
        The groundtruth or class label.
    output : pd.Series
        The models output predictions.
    highlight : Optional[list[Number]], optional
        A list of thresholds to highlight on the plot, by default None.
    axis : Optional[plt.Axes], optional
        The matplotlib axis to draw, by default None; creates a new figure.
    show_legend : bool, optional
        A flag to show the legend, by default True.

    """
    # Normalize scores for binary classfiers
    if (output > 1).any():
        output = output / 100.0

    # split into samples
    data = pd.DataFrame({"predicted": output, "real": y_label})

    samples = []
    labels = []
    for value in sorted(data["real"].unique().tolist()):
        data_group = data[data["real"] == value]["predicted"]
        samples.append(data_group)
        labels.append("Actual {} ({})".format(value, len(data_group)))

    lines.hist_stacked(axis, samples, labels, show_legend, bins=bins)
    if highlight is not None:
        lines.vertical_threshold_lines(axis, highlight, color_alerts=True)
    return axis.get_figure()


@export
@can_draw_empty_perf
@model_plot
def ppv_vs_sensitivity(
    ppv: pd.Series,
    sensitivity: pd.Series,
    thresholds: pd.Series,
    *,
    conf_interval: Optional["ValueWithCI"] = None,
    highlight: Optional[list[Number]] = None,
    axis: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plots the PPV vs Sensitivity (precision-recall curve).

    Will optionally highlight thresholds on the curve.

    Parameters
    ----------
    ppv : pd.Series
        The positive predictive value (precision).
    sensitivity : pd.Series
        The sensitivity (recall).
    thresholds : pd.Series
        The thresholds of the model, used for annotations.
    conf_interval : Optional['ValueWithCI'], optional
        The confidence interval for the AUPRC, by default None.
    highlight : Optional[list[Number]], optional
        A list of thresholds to highlight on the plot, by default None.
    axis : Optional[plt.Axes], optional
        The matplotlib axis to draw, by default None; creates a new figure.
    """
    auprc = f"AUPRC = {metrics.auc(sensitivity, ppv):0.2f}"

    if conf_interval is not None:
        auprc = "AUPRC = %0.2f (%0.2f, %0.2f)" % (
            metrics.auc(sensitivity, ppv),
            conf_interval.lower,
            conf_interval.upper,
        )

    lines.ppv_sensitivity_curve(axis, sensitivity, ppv, label=auprc)

    if highlight is not None:
        lines._add_radial_score_thresholds(axis, sensitivity, ppv, thresholds, thresholds=highlight, Q=4)
    return axis.get_figure()


@export
@can_draw_empty_perf
@model_plot
def metric_vs_threshold(
    stats: pd.DataFrame,
    metric: str,
    *,
    conf: Number = 0.95,
    highlight: Optional[list[Number]] = None,
    axis: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plots a metric vs threshold curve.

    Parameters
    ----------
    stats : pd.DataFrame
        The table of performance metrics of the form given by calculate_bin_stats.
    metric : str
        The performance metric to plot, must be a column in the stats dataframe.
    conf : Number, optional
        The confidence level for the performance metric, by default 0.95.
    highlight : Optional[list[Number]], optional
        A list of thresholds to highlight on the plot, by default None.
    axis : Optional[plt.Axes], optional
        The matplotlib axis to draw, by default None; creates a new figure.
    """
    lines.metric_vs_threshold_curve(axis, stats[metric], stats["Threshold"], label=metric)

    if conf is not None:
        lines.performance_confidence(axis, stats, conf, metric, color=lines.get_last_line_color(axis))

    if highlight is not None:
        lines.vertical_threshold_lines(axis, highlight, color_alerts=True)
    return axis.get_figure()


@export
@model_plot
def performance_metrics(
    stats: pd.DataFrame,
    *,
    conf: Number = 0.95,
    highlight: Optional[list[Number]] = None,
    axis: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Single plot of sensitivity, specificity, and PPV.

    Parameters
    ----------
    stats : pd.DataFrame
        The table of performance metrics of the form given by calculate_bin_stats.
    conf : Number, optional
        The confidence level for the performance metric, by default 0.95.
    highlight : Optional[list[Number]], optional
        A list of thresholds to highlight on the plot, by default None.
    axis : Optional[plt.Axes], optional
        The matplotlib axis to draw, by default None; creates a new figure.
    """

    lines.performance_metrics_plot(axis, stats.Sensitivity, stats.Specificity, stats.PPV, stats.Threshold)

    if conf is not None:
        # Assume labels are metric names - True for performance_metrics_plot lines
        handles, metrics = axis.get_legend_handles_labels()
        for handle, metric in zip(handles, metrics):
            lines.performance_confidence(axis, stats, conf, metric, color=handle.get_color())

    if highlight is not None:
        lines.vertical_threshold_lines(axis, highlight, color_alerts=True)
    return axis.get_figure()


@export
@model_plot
def plot_metric_list(
    stats: pd.DataFrame,
    metrics: list[str],
) -> plt.Figure:
    """
    Plots a list of metrics vs threshold.

    Parameters
    ----------
    stats : pd.DataFrame
        The table of performance metrics with the index being threshold percentiles.
    metrics : list[str]
        The performance metrics to plot, must be columns in the stats dataframe.

    Returns
    -------
    plt.Figure
        The figure object containing the plot.
    """
    fig = plt.figure(figsize=(6, 4))
    axis = fig.gca()

    thresholds = as_probabilities(stats.index)
    for metric in metrics:
        axis.plot(thresholds, stats[metric], label=metric)

    axis.legend(loc="lower right")
    axis.set_xlim([0, 1.01])
    axis.set_xlabel("Threshold")

    return fig
