import itertools
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from seismometer.data import pandas_helpers as pdh
from seismometer.data.filter import FilterRule
from seismometer.seismogram import Seismogram

from . import BinaryClassifierMetricGenerator
from .performance import MONOTONIC_METRICS, THRESHOLD

GENERATED_COLUMNS = [
    "Positives",
    "Prevalence",
    "AUROC",
    "AUPRC",
    "Accuracy",
    "PPV",
    "Sensitivity",
    "Specificity",
    "Flag Rate",
    "Threshold",
]


def calculate_stats(
    df: pd.DataFrame,
    target_col: str,
    score_col: str,
    metric: str,
    metric_values: List[str],
    metrics_to_display: Optional[List[str]] = None,
    decimals: int = 3,
) -> dict:
    """
    Calculates overall performance statistics such as AUROC and threshold-specific statistics
    such as sensitivity and specificity for the provided list of metric values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    target_col : str
        Column name corresponding to the (binary) target.
    score_col : str
        Column name corresponding to the score.
    metric : str
        The metric ('Flag Rate', 'Sensitivity', 'Specificity', 'Threshold') for which statistics are calculated.
    metric_values : List[str]
        A list of metric values for which corresponding statistics are calculated.
    metrics_to_display : Optional[List[str]]
        List of metrics to include in the table, by default None. The default behavior is to show all columns
        in GENERATED_COLUMNS.
    decimals: int
        The number of decimal places for rounding numerical results, by default 3.

    Returns
    -------
    dict
        A dictionary containing performance metrics. A subset of:
            - 'Positives': Total positive samples.
            - 'Prevalence': Prevalence of positive samples.
            - 'AUROC': Area under the receiver operating characteristic curve.
            - 'AUPRC': Area under the precision-recall curve.
            - Additional metrics (PPV, Flag Rate, Sensitivity, Specificity, Threshold, etc.).
    """
    # Check if metric is a valid name.
    if metric not in MONOTONIC_METRICS + [THRESHOLD]:
        raise ValueError(
            f"Invalid metric name: {metric}. The metric needs to be one of: {MONOTONIC_METRICS + [THRESHOLD]}"
        )

    # Initializing row data, to be populated with data specified in metrics_to_display.
    row_data = {}
    metrics_to_display = metrics_to_display or GENERATED_COLUMNS
    metrics_to_display = metrics_to_display if metric in metrics_to_display else metrics_to_display + [metric]

    stats, overall_stats = BinaryClassifierMetricGenerator().calculate_binary_stats(
        dataframe=df,
        target_col=target_col,
        score_col=score_col,
        metrics=metrics_to_display,
        threshold_precision=decimals - 2,
    )
    stats = stats.reset_index()

    # Add overall stats that should be displayed in the table.
    overall_stats = dict((stat, overall_stats[stat]) for stat in overall_stats if stat in metrics_to_display)
    row_data.update(overall_stats)

    # Order/round metric values
    metric_values = sorted([round(num, decimals) for num in metric_values])
    metric_values = [0 if val == 0.0 else val for val in metric_values]

    metric_data = stats[metric].to_numpy()
    thresholds = stats["Threshold"].to_numpy()

    if metric != "Threshold":
        indices = np.argmin(np.abs(metric_data[:, None] - metric_values), axis=0)
        computed_thresholds = thresholds[indices]
    else:
        computed_thresholds = np.array(metric_values) * 100

    # Find indices corresponding to the provided metric values
    threshold_indices = np.argmin(np.abs(thresholds[:, None] - computed_thresholds), axis=0)

    for metric_to_display in metrics_to_display:
        if metric_to_display != metric and metric_to_display not in overall_stats:
            metric_data = stats[metric_to_display].to_numpy()[threshold_indices]
            metric_to_display = metric_to_display.replace(" ", "\u00A0")
            row_data.update(
                {
                    f"{metric_value}_{metric_to_display}": value
                    for metric_value, value in zip(metric_values, metric_data)
                }
            )
    return row_data


def generate_analytics_data(
    score_columns: List[str],
    target_columns: List[str],
    metric: str,
    metric_values: List[float],
    *,
    top_level: str = "Score",
    cohort_dict: Optional[dict[str, tuple[Any]]] = None,
    per_context: bool = False,
    metrics_to_display: Optional[List[str]] = None,
    decimals: int = 3,
    censor_threshold: int = 10,
) -> Optional[pd.DataFrame]:
    """
    Generates a DataFrame containing calculated statistics for each combination of scores and targets.

    Parameters
    ----------
    score_columns : List[str]
        A list of column names corresponding to model prediction scores.
    target_columns : List[str]
        A list of column names corresponding to (binary) targets, by default None.
    metric : str
        The metric ('Flag Rate', 'Sensitivity', 'Specificity', 'Threshold') for which statistics are calculated.
    metric_values : List[float]
        A list of metric values for which corresponding statistics are calculated.
    top_level : str, optional
        The primary grouping category in the performance table, by default "Score".
    cohort_dict : Optional[dict[str, tuple[Any]]], optional
        dictionary of cohort columns and values used to subselect a population for evaluation, by default None.
    per_context : bool
        If scores should be grouped by context, by default False.
    metrics_to_display : Optional[List[str]], optional
        List of metrics to include in the table, by default None. The default behavior is to show all columns
        in GENERATED_COLUMNS.
    decimals : int, optional
        The number of decimal places for rounding numerical results, by default 3.
    censor_threshold : int, optional
        Minimum rows required to generate analytics data, by default 10.

    Returns
    -------
    Optional[pd.DataFrame]
        A DataFrame containing the calculated statistics for each combination of scores and targets.
    """
    rows_list = []
    product = (
        itertools.product(score_columns, target_columns)
        if top_level == "Score"
        else itertools.product(target_columns, score_columns)
    )
    second_level = "Target" if top_level == "Score" else "Score"
    sg = Seismogram()
    cohort_dict = cohort_dict or {}
    cohort_filter = FilterRule.from_cohort_dictionary(cohort_dict)
    cohort_filter.MIN_ROWS = censor_threshold
    data = cohort_filter.filter(sg.dataframe)
    if len(data) <= censor_threshold:
        return None
    for first, second in product:
        current_row = {top_level: first, second_level: second}
        (score, target) = (first, second) if top_level == "Score" else (second, first)
        per_context_data = (
            pdh.event_score(
                data,
                sg.entity_keys,
                score=score,
                ref_time=sg.predict_time,
                ref_event=target,
                aggregation_method=sg.event_aggregation_method(target),
            )
            if per_context
            else data
        )
        current_row.update(
            calculate_stats(
                per_context_data[[target, score]],
                target_col=target,
                score_col=score,
                metric=metric,
                metric_values=metric_values,
                metrics_to_display=metrics_to_display,
                decimals=decimals,
            )
        )
        rows_list.append(current_row)
    # Create a DataFrame from the rows data
    analytics_data = pd.DataFrame(rows_list)
    return analytics_data
