from typing import List
from numpy.typing import ArrayLike

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from seismometer.data import calculate_bin_stats


def compute_thresholds(metric : str, metric_values : List[str], ppv : ArrayLike, sensitivity : ArrayLike, 
                       specificity : ArrayLike, flagged : ArrayLike, thresholds : ArrayLike):
    """
    Computes the thresholds that correspond to the provided list of metric (sensitivity, specificity,
    ppv, flag rate) values.

    Parameters
    ----------
    metric : str
        The metric for which thresholds are computed ('Sensitivity', 'Specificity', 'PPV', 'Flagged').
    metric_values : List[str]
        A list of metric values for which corresponding thresholds are computed.
    ppv : array_like
        ppv values corresponding to thresholds.
    sensitivity : array_like
        sensitivity values corresponding to thresholds.
    specificity : array_like
        Specificity values corresponding to thresholds.
    flagged : array_like
        Flag rate values corresponding to thresholds.
    thresholds : array_like
        List of threshold values.

    Returns
    -------
    np.ndarray
        Computed thresholds corresponding to the specified metric values.
    """
    computed_thresholds = []

    # Find the closest threshold for each metric value
    if metric == 'Sensitivity':
        indices = np.where(sensitivity[:, None] >= metric_values, np.arange(sensitivity.size)[:, None], -1)
        indices = np.max(indices, axis=0)
    elif metric == 'Specificity':
        indices = np.where(specificity[:, None] <= metric_values, np.arange(specificity.size)[:, None], 0)
        indices = np.max(indices, axis=0)
    elif metric == 'Flagged':
        indices = np.where(flagged[:, None] >= metric_values, np.arange(flagged.size)[:, None], -1)
        indices = np.max(indices, axis=0)
    elif metric == 'PPV':
        indices = np.argmin(np.abs(ppv[:, None] - metric_values), axis=0)
    else:
        raise ValueError(f"Invalid metric name: {metric}. The metric needs to be one of: 'Sensitivity', 'Specificity', 'PPV', 'Flagged'.")

    computed_thresholds = thresholds[indices]
    return np.array(computed_thresholds)

def calculate_stats(y_true : ArrayLike, y_pred : ArrayLike, metric : str, metric_values : List[str], decimals : int = 3):
    """
    Calculates overall performance statistics such as AUROC and threshold-specific statistics such as sensitivity and specificity.

    Parameters
    ----------
    y_true : array_like
        True binary labels (ground truth).
    y_pred : array_like
        Predicted probabilities or scores.
    metric : str
        The metric ('PPV', 'Flagged', 'Sensitivity', 'Specificity', 'Threshold') for which statistics are calculated.
    metric_values : List[str]
        A list of metric values for which corresponding statistics are calculated.
    decimals: int
        The number of decimal places for rounding numerical results, by default 3.

    Returns
    -------
    dict
        A dictionary containing performance metrics:
            - 'Positives': Total positive samples.
            - 'Prevalence': Prevalence of positive samples.
            - 'AUROC': Area under the receiver operating characteristic curve.
            - 'AUPRC': Area under the precision-recall curve.
            - Additional metrics based on the specified metric type (PPV, Flagged, Sensitivity, Specificity, Threshold).
    """
    # Calculate AUROC and AUPRC
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)

    stats = calculate_bin_stats(y_true, y_pred)

    ppv = stats["PPV"]
    flagged = stats["Flagged"]
    sensitivity = stats["Sensitivity"]
    specificity = stats["Specificity"]
    threshold = stats["Threshold"]

    # Calculate Positives
    positives = sum(y_true)
    
    # Calculate Prevalence
    prevalence = positives / len(y_true)

    row_data = {'Positives': positives,
                'Prevalence': prevalence,
                'AUROC': auroc,
                'AUPRC': auprc}

    if metric != 'Threshold':
        computed_thresholds = compute_thresholds(metric, metric_values, ppv, sensitivity, specificity, flagged, threshold)
    else:
        computed_thresholds = np.array(metric_values)

    if metric != 'PPV':
        row_data.update({str(metric_value)+'_PPV': value for metric_value,value in zip(metric_values, ppv)})
    if metric != 'Flagged':
        row_data.update({str(metric_value)+'_Flagged': value for metric_value,value in zip(metric_values, flagged)})
    if metric != 'Sensitivity':
        row_data.update({str(metric_value)+'_Sensitivity': value for metric_value,value in zip(metric_values, sensitivity)})
    if metric != 'Specificity':
        row_data.update({str(metric_value)+'_Specificity': value for metric_value,value in zip(metric_values, specificity)})
    if metric != 'Threshold':
        row_data.update({str(metric_value)+'_Threshold': value for metric_value,value in zip(metric_values, computed_thresholds)})

    return pd.Dataframe(row_data)