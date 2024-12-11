from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import auc

from . import calculate_bin_stats
from .performance import MonotonicMetric

GENERATED_COLUMNS = {
    "positives": "Positives",
    "prevalence": "Prevalence",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "accuracy": "Accuracy",
    "ppv": "PPV",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "flagrate": "Flag Rate",
    "threshold": "Threshold",
}


def calculate_stats(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric: str,
    metric_values: List[str],
    metrics_to_display: Optional[List[str]] = None,
    decimals: int = 3,
):
    """
    Calculates overall performance statistics such as AUROC and threshold-specific statistics
    such as sensitivity and specificity for the provided list of metric values.

    Parameters
    ----------
    y_true : array_like
        True binary labels (ground truth).
    y_pred : array_like
        Predicted probabilities or scores.
    metric : str
        The metric ('FlagRate', 'Sensitivity', 'Specificity', 'Threshold') for which statistics are
        calculated.
    metric_values : List[str]
        A list of metric values for which corresponding statistics are calculated.
    metrics_to_display : Optional[List[str]]
        List of metrics to include in the table, by default None. The default behavior is to shows all columns
        in GENERATED_COLUMNS, which is a dictionary mapping metric names to their corresponding column names.
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
    try:
        _ = MonotonicMetric(metric.lower())
    except ValueError:
        raise ValueError(
            f"Invalid metric name: {metric}. The metric needs to be one of: {list(MonotonicMetric.__members__.keys())}"
        )

    # Initializing row data, to be populated with data specified in metrics_to_display.
    row_data = {}
    metric = metric.lower()
    metrics_to_display = metrics_to_display or list(GENERATED_COLUMNS.keys())
    _metrics_to_display_lower = [metric_to_display.lower() for metric_to_display in metrics_to_display]

    stats = calculate_bin_stats(y_true, y_pred)

    # Calculate overall statistics
    if "positives" in _metrics_to_display_lower:
        row_data["Positives"] = stats["TP"].iloc[-1]
    if "prevalence" in _metrics_to_display_lower:
        row_data["Prevalence"] = stats["TP"].iloc[-1] / len(y_true)
    if "auroc" in _metrics_to_display_lower:
        row_data["AUROC"] = auc(1 - stats["Specificity"], stats["Sensitivity"])
    if "auprc" in _metrics_to_display_lower:
        row_data["AUPRC"] = auc(stats["Sensitivity"], stats["PPV"])

    # Order/round metric values
    metric_values = sorted([round(num, decimals) for num in metric_values])
    metric_values = [0 if val == 0.0 else val for val in metric_values]

    metric_data = stats[GENERATED_COLUMNS[metric]].to_numpy()
    thresholds = stats["Threshold"].to_numpy()

    if metric != "threshold":
        indices = np.argmin(np.abs(metric_data[:, None] - metric_values), axis=0)
        computed_thresholds = thresholds[indices]
    else:
        computed_thresholds = np.array(metric_values) * 100
        computed_thresholds = computed_thresholds.astype(int)

    # Find indices corresponding to the provided metric values
    threshold_indices = np.argmin(np.abs(thresholds[:, None] - computed_thresholds), axis=0)

    for metric_to_display in metrics_to_display:
        column_name = GENERATED_COLUMNS.get(metric_to_display.lower(), metric_to_display)
        if metric_to_display.lower() != metric and column_name not in row_data:
            metric_data = stats[column_name].to_numpy()[threshold_indices]
            column_name = column_name.replace(" ", "\u00A0")
            row_data.update(
                {f"{metric_value}_{column_name}": value for metric_value, value in zip(metric_values, metric_data)}
            )

    return row_data
