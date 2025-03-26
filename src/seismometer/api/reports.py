import logging
from typing import Optional

from seismometer.controls.categorical import ExploreCategoricalPlots
from seismometer.controls.categorical_single_column import ExploreSingleCategoricalPlots
from seismometer.core.decorators import export
from seismometer.data.filter import FilterRule
from seismometer.report.profiling import ComparisonReportWrapper, SingleReportWrapper
from seismometer.seismogram import Seismogram
from seismometer.table.analytics_table import ExploreBinaryModelAnalytics
from seismometer.table.fairness import ExploreBinaryModelFairness

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


@export
class ExploreAnalyticsTable(ExploreBinaryModelAnalytics):
    """
    Exploration widget for model comparison across cohorts / binary classifiers / targets.
    """

    def __init__(self):
        """
        Passes the plot function to the superclass.
        """
        super().__init__()


@export
class ExploreOrdinalMetrics(ExploreCategoricalPlots):
    """
    Exploration widget for ordinal categorical metrics, summarizing multiple metrics for a model.
    """

    def __init__(self, group_key=None, title="Metrics distribution"):
        """
        Passes the plot function to the superclass.
        """
        super().__init__(group_key, title)


@export
class ExploreCohortOrdinalMetrics(ExploreSingleCategoricalPlots):
    """
    Exploration widget for ordinal categorical metrics, summarizing an ordinal metric across cohort subclasses.
    """

    def __init__(self, group_key=None, title="Metric distribution"):
        """
        Passes the plot function to the superclass.
        """
        super().__init__(group_key, title)


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


# endregion
