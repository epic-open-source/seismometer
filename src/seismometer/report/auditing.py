import logging
import warnings
from io import StringIO

import pandas as pd
from IPython.display import HTML

logger = logging.getLogger("seismometer")

allowed_metrics = ["tpr", "tnr", "for", "fdr", "fpr", "fnr", "npv", "ppr", "precision", "pprev"]


def to_html(summary_plot) -> HTML:
    buffer = StringIO()
    summary_plot.save(buffer, format="html")
    return HTML(buffer.getvalue())


def fairness_audit_as_html(
    df: pd.DataFrame,
    sensitive_groups: list[str],
    score_column: str,
    target_column: str,
    score_threshold: float,
    metric_list: list[str],
    fairness_threshold: float,
) -> HTML:
    """
    Displays the Aequitas fairness audit for a set of sensitive groups and metrics

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing scores, targets, and demographics.
    sensitive_groups : list[str]
        A list of columns that correspond to the cohorts to stratify by.
    score_column : str
        The score column.
    target_column : str
        The target column.
    score_threshold : float
        The threshold above which a score is 'positive'.
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
        For example, a PPV of 0.5 for group A and a PPV of 0.75 (or 0.33) for group B would be considered a failure
        for any fairness_threshold < 1.5.
    """
    try:
        from aequitas import Audit
        from aequitas.plot.commons.validators import METRICS_LIST
    except ImportError:
        raise ImportError(
            "Error: aequitas or one of its required packages is not installed. Install with `pip install aequitas`."
        )

    for metric in metric_list:
        if metric not in METRICS_LIST:
            raise Exception(f"Unknown metric {metric}.")

    df = df.copy()

    # Need str(category) for aequitas
    for col in sensitive_groups:
        df[col] = df[col].astype(str)
    df["score"] = df[score_column] > score_threshold
    df.drop(score_column, axis=1, inplace=True)

    if df["score"].nunique() != 2:
        logger.error("Audit requires exactly two target classes to be present")
        return

    # Do NOT pass list of sensitive attributes; reducing frame gets desired behavior
    audit = Audit(df, score_column="score", label_column=target_column)
    audit.audit()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        return to_html(audit.summary_plot(metrics=metric_list, fairness_threshold=fairness_threshold))
