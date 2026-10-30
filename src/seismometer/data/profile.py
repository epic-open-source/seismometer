"""
Column profiles and data quality alerts for the feature reports.

Reproduces the variable types, statistics and alert rules of ydata-profiling 4.18 (MIT license) under the settings
seismometer used with it, so the reports and alert_config.yml thresholds carry over. Types come from the dtypes, as
with ydata's ``infer_dtypes: false``. Correlations, interactions, missing-value diagrams, samples and duplicate
detection were turned off in those settings and are not reproduced, nor are time-series detection, Unicode script and
block names, and word clouds.
"""
import string
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, auto
from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from pandas.core.arrays.integer import IntegerDtype

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
HISTOGRAM_BINS = 50
MAX_HISTOGRAM_BINS = 250
CARDINALITY_THRESHOLD = 50
SKEWNESS_THRESHOLD = 20
IMBALANCE_THRESHOLD = 0.5
ALERT_FRACTION = 0.01  # missing, zero and infinite values alert above this fraction
_PANDAS_2 = int(pd.__version__.split(".")[0]) < 3


class AlertType(Enum):
    """Data quality alerts; the lowercase names are the keys of alert_config.yml."""

    CONSTANT = auto()
    EMPTY = auto()
    HIGH_CARDINALITY = auto()
    IMBALANCE = auto()
    INFINITE = auto()
    MISSING = auto()
    REJECTED = auto()
    SKEWED = auto()
    UNIQUE = auto()
    UNSUPPORTED = auto()
    ZEROS = auto()


@dataclass
class Alert:
    """An alert on a column, carrying that column's description as its values."""

    alert_type: AlertType
    values: dict = field(default_factory=dict, repr=False)  # keeps column data out of logs
    column_name: Optional[str] = None

    @property
    def alert_type_name(self) -> str:
        """The alert type as ydata-profiling names it, e.g. "High cardinality"."""
        return self.alert_type.name.replace("_", " ").capitalize()


@dataclass
class DataProfile:
    """Profile of a dataframe: table statistics, a description per column, and alerts."""

    title: str
    table: dict
    variables: dict[str, dict]
    alerts: list[Alert]
    started: datetime
    finished: datetime


def profile_dataframe(dataframe: pd.DataFrame, title: str = "Profiling Report") -> DataProfile:
    """
    Describes every column of a dataframe and checks it for data quality alerts.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The data to profile; it is not modified.
    title : str, optional
        The title of the profile, by default "Profiling Report".

    Returns
    -------
    DataProfile
        The table statistics, column descriptions and alerts.
    """
    started = datetime.now()
    dataframe = dataframe.copy(deep=False)
    dataframe.columns = dataframe.columns.map(lambda col: "df_index" if col == "index" else str(col))

    variables = {col: describe_series(dataframe[col]) for col in dataframe.columns}
    table = _table_stats(dataframe, variables)

    alerts = [Alert(AlertType.EMPTY, values=table)] if table["n"] == 0 else []
    for col, description in variables.items():
        alerts += _variable_alerts(col, description)
    alerts.sort(key=lambda alert: alert.alert_type.name)

    return DataProfile(title, table, variables, alerts, started, datetime.now())


def describe_series(series: pd.Series) -> dict:
    """
    Describes one column: its type, counts, and the statistics for that type.

    Parameters
    ----------
    series : pd.Series
        The column to describe.

    Returns
    -------
    dict
        The description, keyed like ydata-profiling's (e.g. n_missing, p_distinct, imbalance, 25%).
    """
    # pandas 3 never downcasts silently, and deprecates the option
    no_downcasting = pd.option_context("future.no_silent_downcasting", True) if _PANDAS_2 else nullcontext()
    with no_downcasting:
        series = series.fillna(np.nan).infer_objects()

    summary = {"type": detect_type(series)}
    _describe_counts(series, summary)
    _describe_generic(series, summary)
    if not summary["hashable"]:
        return summary

    _describe_supported(summary)
    present = series.dropna() if series.hasnans else series
    describers = {
        "Numeric": _describe_numeric,
        "Categorical": _describe_categorical,
        "Text": _describe_text,
        "DateTime": _describe_date,
    }
    if summary["type"] == "Boolean":
        _describe_boolean(summary)  # unlike the others, sees the missing values too
    elif summary["type"] in describers:
        with np.errstate(invalid="ignore"):  # infinite values give nan sums and deviations, as in ydata
            describers[summary["type"]](present, summary)
    return summary


def detect_type(series: pd.Series) -> str:
    """
    Finds the variable type of a column, as ydata-profiling does.

    Parameters
    ----------
    series : pd.Series
        The column.

    Returns
    -------
    str
        Boolean, Categorical, Numeric, DateTime, Text, or Unsupported (including all-missing columns).
    """
    present = series.dropna()
    if present.empty:
        return "Unsupported"

    if pdt.is_object_dtype(present):
        try:
            is_boolean = bool(present.isin({True, False}).all())
        except Exception:  # unhashable values
            is_boolean = False
    else:
        is_boolean = pdt.is_bool_dtype(present)
    if is_boolean:
        return "Boolean"
    if isinstance(present.dtype, pd.CategoricalDtype):
        return "Categorical"
    if pdt.is_numeric_dtype(present):
        return "Numeric"
    if pdt.is_string_dtype(present) and _all_strings(present):  # checked before the slower date check
        return "Text"
    if pdt.is_datetime64_any_dtype(present) or present.map(type).isin([date, datetime]).all():
        return "DateTime"
    return "Unsupported"


def _all_strings(series: pd.Series) -> bool:
    if isinstance(series.dtype, pd.StringDtype):
        return True
    if not all(isinstance(value, str) for value in series.iloc[:5]):
        return False
    try:
        return bool((series.astype(str).to_numpy() == series.to_numpy()).all())
    except (TypeError, ValueError):
        return False


def _describe_counts(series: pd.Series, summary: dict) -> None:
    try:
        counts = series.value_counts(dropna=False)
        _ = set(counts.index)
        summary["hashable"] = True
    except Exception:  # unhashable values
        summary["hashable"] = False

    summary["ordering"] = False
    if not summary["hashable"]:
        summary["n_missing"] = series.isna().sum()
        return

    counts = counts[counts > 0]  # drops unused categories
    missing = counts.index.isnull()
    summary["n_missing"] = counts[missing].sum() if missing.any() else 0
    summary["value_counts_without_nan"] = counts[~missing] if missing.any() else counts
    try:
        summary["value_counts_index_sorted"] = summary["value_counts_without_nan"].sort_index(ascending=True)
        summary["ordering"] = True
    except TypeError:
        pass


def _describe_generic(series: pd.Series, summary: dict) -> None:
    n = len(series)
    summary.update(
        {
            "n": n,
            "p_missing": summary["n_missing"] / n if n > 0 else 0,
            "count": n - summary["n_missing"],
            "memory_size": series.memory_usage(deep=False),
        }
    )


def _describe_supported(summary: dict) -> None:
    count = summary["count"]
    counts = summary["value_counts_without_nan"]
    n_distinct = len(counts)
    n_unique = counts.where(counts == 1).count()
    summary.update(
        {
            "n_distinct": n_distinct,
            "p_distinct": n_distinct / count if count > 0 else 0,
            "is_unique": n_unique == count and count > 0,
            "n_unique": n_unique,
            "p_unique": n_unique / count if count > 0 else 0,
        }
    )


def _shape(series: pd.Series, counts: pd.Series) -> dict:
    """Kurtosis and skewness, 0 for a constant column as in ydata; pandas 3.1 returns nan."""
    constant = len(counts) == 1 and np.isfinite(counts.index[0])
    return {
        "kurtosis": 0.0 if constant and len(series) >= 4 else series.kurt(),
        "skewness": 0.0 if constant and len(series) >= 3 else series.skew(),
    }


def _describe_numeric(series: pd.Series, summary: dict) -> None:
    counts = summary["value_counts_without_nan"]
    n = summary["n"]

    summary["n_negative"] = counts.loc[counts.index < 0].sum()
    summary["p_negative"] = summary["n_negative"] / n
    infinite = counts.index.isin([np.inf, -np.inf])
    summary["n_infinite"] = counts.loc[infinite].sum()
    summary["n_zeros"] = counts.loc[0] if 0 in counts.index else 0

    if isinstance(series.dtype, IntegerDtype):
        summary.update(
            {
                "mean": series.mean(),
                "std": series.std(),
                "variance": series.var(),
                "min": series.min(),
                "max": series.max(),
                **_shape(series, counts),
                "sum": series.sum(),
            }
        )
        present = series.astype(str(series.dtype).lower())
    else:
        present = series.to_numpy()
        values = counts.index.to_numpy()
        if len(values):
            summary.update(
                {
                    "mean": np.average(values, weights=counts.to_numpy()),
                    "std": np.std(present, ddof=1),
                    "variance": np.var(present, ddof=1),
                    "min": np.min(values),
                    "max": np.max(values),
                    **_shape(series, counts),
                    "sum": np.dot(values, counts.to_numpy()),
                }
            )
        else:
            summary.update(
                {
                    "mean": np.nan,
                    "std": 0.0,
                    "variance": 0.0,
                    "min": np.nan,
                    "max": np.nan,
                    "kurtosis": 0.0,
                    "skewness": 0.0,
                    "sum": 0,
                }
            )

    summary["mad"] = np.median(np.abs(present - np.median(present)))
    summary["range"] = summary["max"] - summary["min"]
    # pandas 3 gives NA instead of nan for nullable quantiles of infinite values
    summary.update({f"{q:.0%}": value for q, value in series.quantile(QUANTILES).astype(float).to_dict().items()})
    summary["iqr"] = summary["75%"] - summary["25%"]
    summary["cv"] = summary["std"] / summary["mean"] if summary["mean"] else np.nan
    summary["p_zeros"] = summary["n_zeros"] / n
    summary["p_infinite"] = summary["n_infinite"] / n

    increasing, decreasing, unique = series.is_monotonic_increasing, series.is_monotonic_decreasing, series.is_unique
    summary.update(
        {
            "monotonic_increase": increasing,
            "monotonic_decrease": decreasing,
            "monotonic_increase_strict": increasing and unique,
            "monotonic_decrease_strict": decreasing and unique,
        }
    )
    summary["monotonic"] = (
        2 if increasing and unique else -2 if decreasing and unique else 1 if increasing else -1 if decreasing else 0
    )

    finite = counts[~infinite]
    if len(finite):
        summary["histogram"] = _histogram(finite.index.to_numpy(), summary["n_distinct"], weights=finite.to_numpy())


def _describe_categorical(series: pd.Series, summary: dict) -> None:
    counts = summary["value_counts_without_nan"]
    counts.index = counts.index.astype(str)
    summary["imbalance"] = imbalance_score(counts)
    summary["first_rows"] = series.astype(str).head(5)


def _describe_text(series: pd.Series, summary: dict) -> None:
    counts = summary["value_counts_without_nan"]
    counts.index = counts.index.astype(str)
    summary["first_rows"] = series.astype(str).head(5)

    unique = pd.Series(counts.index, index=counts.to_numpy(), dtype=object)
    lengths = pd.Series(unique.index, index=unique.str.len()).groupby(level=0, sort=False).sum()
    lengths = lengths.sort_values(ascending=False)
    summary.update(
        {
            "max_length": np.max(lengths.index),
            "mean_length": np.average(lengths.index, weights=lengths.to_numpy()) if len(lengths) else np.nan,
            "median_length": _weighted_median(lengths.index.to_numpy(), lengths.to_numpy())
            if len(lengths)
            else np.nan,
            "min_length": np.min(lengths.index),
            "length_histogram": lengths,
        }
    )
    summary["histogram_length"] = _histogram(lengths.index.to_numpy(), len(lengths), weights=lengths.to_numpy())

    exploded = unique[unique != ""].apply(list).explode()
    character_counts = pd.Series(exploded.index, index=exploded).dropna()
    if len(character_counts) > 0:
        character_counts = character_counts.groupby(level=0, sort=False).sum().sort_values(ascending=False)
        character_counts = character_counts[character_counts.index.str.len() > 0]
    summary.update(
        {
            "n_characters_distinct": len(character_counts),
            "n_characters": int(character_counts.sum()),
            "character_counts": character_counts,
        }
    )

    words = unique.str.lower().str.split().explode().str.strip(string.punctuation + string.whitespace)
    word_counts = pd.Series(words.index, index=words)
    word_counts = word_counts[word_counts.index.notnull()].groupby(level=0, sort=False).sum()
    if not word_counts.empty:
        summary["word_counts"] = word_counts.sort_values(ascending=False)


def _describe_boolean(summary: dict) -> None:
    counts = summary["value_counts_without_nan"]
    if counts.empty:
        summary.update({"top": np.nan, "freq": 0, "imbalance": 0})
    else:
        summary.update({"top": counts.index[0], "freq": counts.iloc[0], "imbalance": imbalance_score(counts)})


def _describe_date(series: pd.Series, summary: dict) -> None:
    dates = pd.to_datetime(series, format="mixed", errors="coerce")
    invalid = series[dates.isna()]
    dates = dates.dropna()
    if summary["value_counts_without_nan"].empty:
        values = dates.to_numpy()
        summary.update({"min": pd.NaT, "max": pd.NaT, "range": 0})
    else:
        summary.update({"min": dates.min().to_pydatetime(), "max": dates.max().to_pydatetime()})
        summary["range"] = summary["max"] - summary["min"]
        values = dates.to_numpy().astype("datetime64[s]").astype(np.int64)
    summary["histogram"] = _histogram(values, dates.nunique())
    summary.update(
        {
            "invalid_dates": invalid.nunique(),
            "n_invalid_dates": len(invalid),
            "p_invalid_dates": len(invalid) / summary["n"],
        }
    )


def imbalance_score(counts: pd.Series) -> float:
    """
    One minus the entropy of the value counts over the largest possible entropy for that many classes.

    Parameters
    ----------
    counts : pd.Series
        The count of each distinct value.

    Returns
    -------
    float
        0 for evenly spread values (or a single value), 1 when nearly all rows share one value.
    """
    if len(counts) <= 1:
        return 0
    p = counts.to_numpy(dtype=float)
    p = p / p.sum()
    p = p[p > 0]
    return 1 - (-(p * np.log2(p)).sum()) / np.log2(len(counts))


def _histogram(values: np.ndarray, n_distinct: int, weights: Optional[np.ndarray] = None) -> tuple:
    """Fixed-width histogram, with at most HISTOGRAM_BINS bins and no more bins than distinct values."""
    if len(values) == 0:
        return ()
    finite = values[np.isfinite(values)]
    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if vmax == vmin:
        eps = 0.5 if vmin == 0 else abs(vmin) * 0.1
        bins = np.array([vmin - eps, vmin + eps])
    else:
        bins = np.linspace(vmin, vmax, max(1, min(HISTOGRAM_BINS, n_distinct, MAX_HISTOGRAM_BINS)) + 1)
    return np.histogram(values, bins=bins, weights=weights)


def _weighted_median(data: np.ndarray, weights: np.ndarray) -> Any:
    # Sorts data and weights separately, as ydata-profiling does, so median lengths match its reports
    s_data, s_weights = np.sort(data), np.sort(weights)
    midpoint = 0.5 * np.sum(s_weights)
    if s_weights[-1] > midpoint:
        return data[weights == np.max(weights)][0]
    cumulative = np.cumsum(s_weights)
    idx = np.where(cumulative <= midpoint)[0][-1]
    if cumulative[idx] == midpoint:
        end = idx + 2
        return np.mean(s_data[idx:end])
    return s_data[idx + 1]


def _table_stats(dataframe: pd.DataFrame, variables: dict[str, dict]) -> dict:
    n = len(dataframe)
    memory_size = dataframe.memory_usage(deep=False).sum()
    table = {
        "n": n,
        "n_var": len(dataframe.columns),
        "memory_size": memory_size,
        "record_size": float(memory_size) / n if n > 0 else 0,
        "n_cells_missing": 0,
        "n_vars_with_missing": 0,
        "n_vars_all_missing": 0,
    }
    for description in variables.values():
        if description["n_missing"] > 0:
            table["n_vars_with_missing"] += 1
            table["n_cells_missing"] += description["n_missing"]
            if description["n_missing"] == n:
                table["n_vars_all_missing"] += 1
    cells = table["n"] * table["n_var"]
    table["p_cells_missing"] = table["n_cells_missing"] / cells if cells > 0 else 0
    table["types"] = dict(Counter(description["type"] for description in variables.values()))
    return table


def _over_alert_fraction(value: float) -> bool:
    return not pd.isna(value) and value > ALERT_FRACTION


def _variable_alerts(column: str, description: dict) -> list[Alert]:
    types = []
    if _over_alert_fraction(description["p_missing"]):
        types.append(AlertType.MISSING)

    if description["type"] == "Unsupported":
        types += [AlertType.UNSUPPORTED, AlertType.REJECTED]
    else:
        n_distinct = description.get("n_distinct", np.nan)
        if n_distinct == description["n"]:
            types.append(AlertType.UNIQUE)
        if n_distinct == 1:
            types.append(AlertType.CONSTANT)

        if description["type"] == "Categorical":
            if n_distinct > CARDINALITY_THRESHOLD:
                types.append(AlertType.HIGH_CARDINALITY)
            if description.get("imbalance", 0) > IMBALANCE_THRESHOLD:
                types.append(AlertType.IMBALANCE)
        elif description["type"] == "Numeric":
            skewness = description["skewness"]
            if not pd.isna(skewness) and abs(skewness) > SKEWNESS_THRESHOLD:
                types.append(AlertType.SKEWED)
            if _over_alert_fraction(description["p_infinite"]):
                types.append(AlertType.INFINITE)
            if _over_alert_fraction(description["p_zeros"]):
                types.append(AlertType.ZEROS)
        elif description["type"] == "Boolean":
            if description.get("imbalance", 0) > IMBALANCE_THRESHOLD:
                types.append(AlertType.IMBALANCE)

    return [Alert(alert_type, values=description, column_name=column) for alert_type in types]
