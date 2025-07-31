from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .decorators import export
from .performance import calculate_bin_stats

SeriesOrArray = Union[pd.Series, np.ndarray]

# region Stats


@export
def get_cohort_data(
    df: pd.DataFrame,
    cohort_feature: str,
    *,
    proba: Union[str, SeriesOrArray],
    true: Union[str, SeriesOrArray] = "TARGET",
    splits: Optional[List] = None,
) -> pd.DataFrame:
    """
    Convenience function to create and format data for use in the cohort plots.
    Takes in information about the class, predictions, and true labels to output a dataset and corresponding labels.

    In the case that multiple columns are used, predictions from each column are appended to the result
    so that rows sharing a cohort group are disjoint, and rows with different cohort columns potentially overlap.

    Currently supports cohort_features of type Categorical (splits all categories) and Numeric (splits on specified
    values or at mean).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of observations to use for plotting, must contain the column specified in cohort_feature.
        Additionally, must contain columns specified by proba and true if using strings and not arrays.
    cohort_feature : str
        string specification of the dataframe column to split.
        Currently supports numeric and categorical columns.
    proba : Union[str, SeriesOrArray]
        The predictions made by the model under review.

        - If string - must be a column in the dataframe.
        - If series or array - must be the same length as the dataframe.

    true : Union[str, SeriesOrArray]
        The true label associated with a prediction, by default "TARGET".

        - If string - must be a column in the dataframe.
        - If series or array - must be the same length as the dataframe and int values.

    splits : Optional[List]
        The numeric values to split cohorts or category values to include, treats each category value as its own
        split, by default None.
        If None, will create a dichotomy for numeric values split at the mean.

    Returns
    -------
    pd.DataFrame
        Data - ingestible by plot_cohort_* functions.
    """
    # Find data
    proba_series = resolve_col_data(df, proba).astype(float)
    true_series = resolve_col_data(df, true).astype(float)  # handle nan but require numeric

    cohort_series = resolve_cohorts(df[cohort_feature], splits)

    # Create standard DataFrame
    rv = pd.concat([true_series, proba_series, cohort_series], axis="columns")
    rv.columns = ["true", "pred", "cohort"]  # Force column names to be consistent

    return rv.dropna(axis="index", how="any")


@export
def get_cohort_performance_data(
    df: pd.DataFrame,
    cohort_feature: str,
    *,
    proba: Union[str, SeriesOrArray],
    true: Union[str, SeriesOrArray] = "TARGET",
    splits: Optional[List] = None,
    censor_threshold: int = 10,
) -> pd.DataFrame:
    """
    Generates a dataframe with particular performance metrics (accuracy, sensitivity,
    specificity, ppv, npv, and flag rate (predicted positive condition rate)) for
    particular threshold values and cohort.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of observations to use, must contain the column specified in cohort_feature.
        Additionally, must contain columns specified by proba and true if using strings and not arrays.
    cohort_feature : str
        String specification of the dataframe column to split.
        Currently supports numeric and categorical columns.
    proba : Union[str, SeriesOrArray]
        The predictions made by the model under review.

        - If string - must be a column in the dataframe.
        - If series or array - must be the same length as the dataframe.

    true : Union[str, SeriesOrArray], default="TARGET"
        The true label being predicted.

        - If string - must be a column in the dataframe.
        - If series or array - must be the same length as the dataframe and int values.

    splits : Optional[List], default=None
        Optional - the numeric values to split cohorts or category values to include, treats each category value as its
        own split.
        If None, will create a dichotomy for numeric values split at the mean.
    censor_threshold : int, default=10
        Minimum number of observations in a cohort to calculate performance metrics.

    Returns
    -------
    pd.DataFrame
        Performance statistics for particular threshold values by cohort.
    """
    data = get_cohort_data(df, cohort_feature, proba=proba, true=true, splits=splits)

    cohort_perf_stats = []
    observed = set()
    data["true"] = data["true"].astype(int)
    for label, group_data in data.groupby("cohort", observed=True):
        if (group_data.true.sum() == 0) or (group_data.true.count() < censor_threshold):
            continue
        ind_perf_stats = calculate_bin_stats(group_data.true, group_data.pred)
        ind_perf_stats["cohort"] = label
        ind_perf_stats["cohort-count"] = len(group_data)
        ind_perf_stats["cohort-targetcount"] = group_data.true.sum()

        cohort_perf_stats.append(ind_perf_stats)
        observed.add(label)

    # Add empty cohorts
    for label in set(data["cohort"].cat.categories) - observed:
        cohort_perf_stats.append(
            pd.DataFrame({"cohort": label, "cohort-count": 0, "cohort-targetcount": 0}, index=[0])
        )

    if not cohort_perf_stats:
        return pd.DataFrame()

    frame = pd.concat(cohort_perf_stats, ignore_index=True)
    frame["cohort"] = frame["cohort"].astype(pd.CategoricalDtype(data["cohort"].cat.categories))

    return frame


def resolve_col_data(df: pd.DataFrame, feature: Union[str, pd.Series]) -> pd.Series:
    """
    Handles resolving feature from either being a series or specifying a series in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Containing a column of name feature if feature is passed in as a string.
    feature : Union[str, pd.Series]
        Either a pandas.Series or a column name in the dataframe.

    Returns
    -------
    pd.Series.
    """

    if isinstance(feature, str):
        if feature in df.columns:
            return df[feature].copy()
        else:
            raise KeyError(f"Feature {feature} was not found in dataframe")
    elif hasattr(feature, "ndim"):
        if feature.ndim > 1:  # probas from sklearn is nx2 with second column being the positive predictions
            return feature[:, 1]
        else:
            return feature
    else:
        raise TypeError("Feature must be a string or pandas.Series, was given a ", type(feature))


# endregion
# region Labels


@export
def resolve_cohorts(series: SeriesOrArray, splits: Optional[List] = None) -> pd.Series:
    """
    Bin a series of data based on the defined splits if defined.
    Only handles numeric and categorical data.

    Parameters
    ----------
    series : SeriesOrArray
        pandas series of data to bin.
    splits : Optional[List], optional
        List of splits to define inner bins (default: None).

    Returns
    -------
    pd.Series
        Categorical series with labels.
    """
    if hasattr(series, "cat"):  # is categorical series
        return label_cohorts_categorical(series, splits)
    # Treat everything else like continuous - can raise errors with unexpected data types
    return label_cohorts_numeric(series, splits)


def label_cohorts_numeric(series: SeriesOrArray, splits: Optional[List] = None) -> pd.Series:
    """
    Bin a continuous numeric series of data, based on thresholds of inner bin edges.

    Parameters
    ----------
    series : SeriesOrArray
        pandas series of data to bin.
    splits : Optional[List], optional
        List of splits to define inner bins (default: None-> series.mean()).

    Returns:
    --------
    pd.Series
        Categorical series with labels.
    """
    bins = find_bin_edges(series, splits)
    bin_ixs = np.digitize(series, bins, right=False)
    has_good_binning(bin_ixs, bins)

    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)] + [f">={bins[-1]}"]
    labels[0] = f"<{bins[1]}"
    cat = pd.Categorical.from_codes(bin_ixs - 1, labels)
    return pd.Series(cat)


def has_good_binning(bin_ixs: List, bin_edges: List) -> None:
    """
    Verifies that the binning is sound by making sure lists are equal length.

       - If there are fewer realized ix than edges then a bin is empty.
       - If there are more realized ix than edges then the edge array got out of sync somehow.

    Parameters
    ----------
    bin_ixs : List
        List of ix for binned values; output of np.digitize using bin_edges.
    bin_edges : List
        List of bin edges to split on.

    Raises
    ------
    IndexError
        The list of indexes does not align with the bin edge list.
    """
    if len(bin_edges) != len(np.unique(bin_ixs)):
        raise IndexError("Splits provided contain some empty bins.")


def label_cohorts_categorical(series: SeriesOrArray, cat_values: Optional[list] = None) -> Tuple[pd.Series, list[str]]:
    """
    Bin a categorical series of data, reduced to a set of category values.

    Parameters
    ----------
    series : SeriesOrArray
        pandas series of data to bin.
    cat_values : Optional[list], optional
        List of categories to reduce to (default: None-> all observed categories).

    Returns
    -------
        np.array of ints indicating the bin index for each value in the input series.
        List of string labels for each bin; which is the list of categories.
    """
    series.name = "cohort"
    series.cat._name = "cohort"  # CategoricalAccessors have a different name..

    # If no splits specified, restrict to observed values
    if cat_values is None:
        return series.cat.remove_unused_categories()

    # If the series has exactly the request categories, return it
    if set(cat_values) == set(series.cat.categories):
        return series

    if cat_values is not None:
        return series.where(series.isin(cat_values), np.nan)


def find_bin_edges(series: SeriesOrArray, thresholds: Optional[list] = None) -> list[float]:
    """
    Creates list of bin edges from a series of continuous numeric data and list of inner thresholds.
    Contains lower bound but does not need upper bound due to numpy handling already understanding greater than max.

    Parameters
    ----------
    series : SeriesOrArray
        pandas series of data to bin.
    thresholds : Optional[list], optional
        List of thresholds indicating inner bin edges (default: None-> series.mean()).

    Returns
    -------
    list[float]
        Sorted list of bin edges.
    """
    if not thresholds:
        thresholds = [np.mean(series)]
    # Ensure list-like; handle float
    if not hasattr(thresholds, "insert"):
        thresholds = [thresholds]
    # Ensure sorted and unique entries
    bins = sorted(set(thresholds))

    ymin = min(series)
    if bins[0] > ymin:
        bins.insert(0, ymin)

    return bins


# endregion
