"""
HTML feature reports for one dataframe profile, or a side-by-side comparison of two.

Keeps the sections, labels, number formatting, alert wording and plots of the ydata-profiling reports they replace.
"""
import base64
import math
import re
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup, escape
from matplotlib import rc_context, rcParams
from matplotlib.figure import Figure

import seismometer
from seismometer.core.io import slugify
from seismometer.data.profile import Alert, AlertType, DataProfile

COLORS = ["#0085f2", "#dd299d"]  # the first and second dataset
PRECISION = 2  # significant digits for statistics
N_FREQUENT = 5  # rows in value frequency tables
N_EXTREME = 5  # rows in the minimum and maximum value tables
N_FREQUENT_SMALL = {"Categorical": 5, "Boolean": 3}
N_COMMON_WORDS = 10
MAX_PLOTTED_CATEGORIES = 10

TYPE_LABELS = {
    "Numeric": "Real number (&Ropf;)",
    "Categorical": "Categorical",
    "Text": "Text",
    "Boolean": "Boolean",
    "DateTime": "Date",
    "Unsupported": "Unsupported",
}
ALERT_STYLES = {
    "constant": "warning",
    "empty": "info",
    "high_cardinality": "danger",
    "imbalance": "primary",
    "infinite": "info",
    "missing": "info",
    "rejected": "danger",
    "skewed": "info",
    "unique": "danger",
    "unsupported": "warning",
    "zeros": "info",
}
UNIQUE_HELP = "The number of unique values (all values that occur exactly once in the dataset)."
# The ydata-profiling plot style, independent of seismometer's own matplotlib style:
# ydata's settings with seaborn's white style on top, as ydata applies them
PLOT_STYLE = {
    "patch.facecolor": "#348ABD",
    "patch.antialiased": True,
    "font.size": 10.0,
    "figure.edgecolor": "0.50",
    "figure.facecolor": "white",
    "text.color": ".15",
    "axes.labelcolor": ".15",
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": ".15",
    "ytick.color": ".15",
    "axes.axisbelow": True,
    "image.cmap": "Greys",
    "font.family": ["sans-serif"],
    "font.sans-serif": ["Arial", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"],
    "grid.linestyle": "-",
    "lines.solid_capstyle": "round",
    "axes.grid": True,
    "axes.facecolor": "#EAEAF2",
    "axes.edgecolor": "white",
    "axes.linewidth": 0,
    "grid.color": "white",
    "figure.figsize": [8.0, 5.5],
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "grid.linewidth": 1,
    "lines.linewidth": 1.75,
    "patch.linewidth": 0.3,
    "lines.markersize": 7,
    "lines.markeredgewidth": 0,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.pad": 7,
    "ytick.major.pad": 7,
    **sns.axes_style("white"),
}

_env = Environment(
    loader=PackageLoader("seismometer", "html/resources"),
    autoescape=select_autoescape(default=True),
    lstrip_blocks=True,
    trim_blocks=True,
)


# region formatting
def fmt_percent(value: float, edge_cases: bool = True) -> str:
    """
    Formats a ratio as a percentage with one decimal.

    Parameters
    ----------
    value : float
        The ratio, from 0 to 1.
    edge_cases : bool, optional
        Whether ratios that would round to 0% or 100% show as "< 0.1%" and "> 99.9%", by default True.

    Returns
    -------
    str
        The percentage, e.g. "50.0%".
    """
    if edge_cases and round(value, 3) == 0 and value > 0:
        return "< 0.1%"
    if edge_cases and round(value, 3) == 1 and value < 1:
        return "> 99.9%"
    return f"{value * 100:2.1f}%"


def fmt_numeric(value: Any, precision: int = 10) -> Markup:
    """
    Formats a number with the given significant digits.

    Parameters
    ----------
    value : Any
        The number, or None.
    precision : int, optional
        The number of significant digits, by default 10.

    Returns
    -------
    Markup
        The number, with exponents written as × 10<sup>n</sup>, or "N/A" for None.
    """
    if value is None:
        return Markup("N/A")
    text = f"{{:.{precision}g}}".format(value)
    for exponent in ["e+", "e-"]:
        if exponent in text:
            sign = "-" if exponent == "e-" else ""
            text = text.replace(exponent, " × 10<sup>") + "</sup>"
            text = text.replace("<sup>0", "<sup>").replace("<sup>", f"<sup>{sign}")
    return Markup(text)


def fmt(value: Any) -> Markup:
    """
    Formats a value for the report.

    Parameters
    ----------
    value : Any
        The value.

    Returns
    -------
    Markup
        Python floats and ints formatted by fmt_numeric, anything else as escaped text.
    """
    if type(value) in [float, int]:
        return fmt_numeric(value)
    return escape(value)


def fmt_bytesize(num: float, suffix: str = "B") -> str:
    """
    Formats a size with binary units.

    Parameters
    ----------
    num : float
        The size, in bytes.
    suffix : str, optional
        The unit after the binary prefix, by default "B".

    Returns
    -------
    str
        The size, e.g. "16.7 MiB".
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


def fmt_monotonic(value: int) -> str:
    """
    Names a column's monotonicity.

    Parameters
    ----------
    value : int
        2 or 1 for strictly increasing or increasing, 0 for neither, -1 or -2 for decreasing or strictly decreasing.

    Returns
    -------
    str
        The name, e.g. "Strictly increasing".
    """
    return {
        2: "Strictly increasing",
        1: "Increasing",
        0: "Not monotonic",
        -1: "Decreasing",
        -2: "Strictly decreasing",
    }[value]


def fmt_timespan(seconds: float) -> str:
    """
    Formats a duration.

    Parameters
    ----------
    seconds : float
        The duration, in seconds.

    Returns
    -------
    str
        The duration, e.g. "1.2 seconds" or "3 minutes and 4.5 seconds".
    """
    if seconds < 60:
        text = re.sub(r"\.?0+$", "", f"{seconds:.2f}")
        return f"{text} second{'' if math.floor(seconds) == 1 else 's'}"
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)} minute{'' if int(minutes) == 1 else 's'} and {fmt_timespan(seconds)}"


_env.filters.update(fmt=fmt, fmt_percent=fmt_percent, fmt_numeric=fmt_numeric)


# endregion
# region alerts
_ALERT_TEMPLATES = {
    AlertType.CONSTANT: '{{ name }} has constant value "{{ values["value_counts_without_nan"].index[0] }}"',
    AlertType.EMPTY: "Dataset is empty",
    AlertType.HIGH_CARDINALITY: "{{ name }} has a high cardinality: {{ values['n_distinct'] }} distinct values",
    AlertType.IMBALANCE: "{{ name }} is highly imbalanced ({{ values['imbalance'] | fmt_percent }})",
    AlertType.INFINITE: (
        "{{ name }} has {{ values['n_infinite'] }} ({{ values['p_infinite'] | fmt_percent }}) infinite values"
    ),
    AlertType.MISSING: (
        "{{ name }} has {{ values['n_missing'] }} ({{ values['p_missing'] | fmt_percent }}) missing values"
    ),
    AlertType.SKEWED: "{{ name }} is highly skewed (γ1 = {{ values['skewness'] | fmt_numeric }})",
    AlertType.UNIQUE: "{{ name }} has unique values",
    AlertType.UNSUPPORTED: "{{ name }} is an unsupported type, check if it needs cleaning or further analysis",
    AlertType.ZEROS: "{{ name }} has {{ values['n_zeros'] }} ({{ values['p_zeros'] | fmt_percent }}) zeros",
}


def alert_html(alert: Alert, anchor: Optional[str] = None) -> str:
    """
    Describes an alert in a sentence, worded as in ydata-profiling.

    Parameters
    ----------
    alert : Alert
        The alert.
    anchor : Optional[str], optional
        The anchor of the column's section of the report, by default the one made from the column name.

    Returns
    -------
    str
        HTML for the sentence, with the column name linked to its section of the report.
    """
    anchor = anchor or _anchor(alert.column_name)
    name = Markup('<a href="#pp_var_{}"><code>{}</code></a>').format(anchor, alert.column_name)
    template = _env.from_string(_ALERT_TEMPLATES.get(alert.alert_type, "{{ name }}"))
    return str(template.render(name=name, values=alert.values))


def _alert_badge(alert: Alert) -> Markup:
    style = ALERT_STYLES.get(alert.alert_type.name.lower(), "secondary")
    return Markup('<span class="badge text-bg-{}">{}</span>').format(style, alert.alert_type_name)


# endregion
# region plots
def _svg_image(figure: Figure, **savefig_kwargs) -> Markup:
    buffer = StringIO()
    figure.savefig(buffer, format="svg", **savefig_kwargs)
    encoded = base64.b64encode(buffer.getvalue().encode("utf-8")).decode("ascii")
    return Markup('<img class="plot" src="data:image/svg+xml;base64,{}">').format(encoded)


def _histogram_plot(histograms: list[tuple], mini: bool = False, date: bool = False) -> Optional[Markup]:
    """Bar plot of precomputed (counts, edges) histograms, overlaid when comparing datasets."""
    histograms = [h if len(h) else None for h in histograms]
    if all(h is None or not np.isfinite(np.asarray(h[0], dtype=float)).any() for h in histograms):
        return None
    with rc_context(PLOT_STYLE):
        return _draw_histograms(histograms, mini, date)


def _draw_histograms(histograms: list[Optional[tuple]], mini: bool, date: bool) -> Markup:
    figure = Figure(figsize=(3, 2.25) if mini else (7, 3))
    ax = figure.add_subplot(111)
    for idx in reversed(range(len(histograms))):
        if histograms[idx] is None:
            continue
        counts, edges = histograms[idx]
        widths = np.diff(edges)
        alpha = 0.6 if len(histograms) > 1 else None
        ax.bar(edges[:-1] + widths / 2, counts, widths, facecolor=COLORS[idx], alpha=alpha)
    if date:
        ax.xaxis.set_major_formatter(_date_tick)
    if mini:
        ax.set_facecolor("w")
        ax.yaxis.set_visible(False)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(6 if date else 8)
    else:
        ax.set_ylabel("Frequency")
    ax.xaxis.set_tick_params(rotation=90 if date else 45)
    figure.tight_layout()
    return _svg_image(figure)


def _date_tick(seconds: float, _: Any = None) -> str:
    """Label for a tick in seconds since 1970, of naive dates; fromtimestamp would apply the local time zone."""
    try:
        return (datetime(1970, 1, 1) + timedelta(seconds=seconds)).strftime("%Y-%m-%d")
    except OverflowError:  # matplotlib can place ticks past year 9999
        return ""


def _category_plot(counts: pd.Series) -> Markup:
    """Stacked horizontal bar of category frequencies."""
    with rc_context(PLOT_STYLE):
        return _draw_categories(counts)


def _draw_categories(counts: pd.Series) -> Markup:
    colors = rcParams["axes.prop_cycle"].by_key()["color"]
    colors = (colors * (len(counts) // len(colors) + 1))[: len(counts)]
    figure = Figure(figsize=(7, 2))
    ax = figure.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, np.sum(counts))
    ax.set_ylim(0.4, 1.6)
    start = 0
    for count, label, color in zip(counts, counts.index.astype(str), colors):
        rects = ax.barh(y=1, width=count, height=1, left=start, label=label, color=color)
        r, g, b, _ = rects[0].get_facecolor()
        percent = count / counts.sum() * 100
        if percent > 8:
            ax.bar_label(
                rects,
                labels=[f"{percent:.1f}%\n({count})"],
                label_type="center",
                color="white" if r * g * b < 0.5 else "darkgrey",
                fontsize="x-large",
                fontweight="bold",
            )
        start += count
    legend = ax.legend(ncol=1, bbox_to_anchor=(0, 0), fontsize="xx-large", loc="upper left")
    return _svg_image(figure, bbox_extra_artists=[legend], bbox_inches="tight")


# endregion
# region tables
def _table(rows: list[tuple], name: Optional[str] = None, caption: Optional[str] = None) -> dict:
    """rows are (label, values per dataset[, highlighted, help text])."""
    return {
        "kind": "table",
        "name": name,
        "caption": caption,
        "rows": [
            {
                "name": row[0],
                "values": row[1],
                "alert": len(row) > 2 and row[2],
                "help": row[3] if len(row) > 3 else None,
            }
            for row in rows
        ],
    }


def _frequency_rows(counts: pd.Series, n: int, max_rows: int) -> list[dict]:
    """The most frequent values, then Other values (k) and (Missing) when they outweigh the last value shown."""
    if len(counts) == 0:
        return []
    max_rows = min(max_rows, n)
    if max_rows < len(counts):
        other, min_shown = np.sum(counts.iloc[max_rows:]), counts.to_numpy()[max_rows]
    else:
        other, min_shown = 0, 0
    missing = n - np.sum(counts)
    largest = max(counts.to_numpy()[0], other, missing)
    if largest == 0:
        return []

    rows = [
        {"label": label, "count": count, "width": count / largest, "percentage": float(count) / n, "extra": ""}
        for label, count in counts.iloc[:max_rows].items()
    ]
    if other > min_shown:
        label = f"Other values ({counts.count() - max_rows})"
        rows.append(
            {"label": label, "count": other, "width": other / largest, "percentage": min(other / n, 1.0)}
            | {"extra": "other"}
        )
    if missing > min_shown:
        rows.append(
            {"label": "(Missing)", "count": missing, "width": missing / largest, "percentage": missing / n}
            | {"extra": "missing"}
        )
    return rows


def _extreme_rows(counts: pd.Series, n: int) -> list[dict]:
    shown = counts.iloc[:N_EXTREME]
    largest = shown.max()
    return [
        {"label": label, "count": count, "width": count / largest if largest else 0, "percentage": count / n}
        | {"extra": ""}
        for label, count in shown.items()
    ]


def _freq(rows_per_dataset: list[list[dict]], name: Optional[str] = None, small: bool = False) -> dict:
    return {"kind": "freq", "name": name, "small": small, "tables": rows_per_dataset}


def _image(image: Optional[Markup], name: Optional[str] = None, caption: Optional[Markup] = None) -> Optional[dict]:
    return None if image is None else {"kind": "image", "image": image, "name": name, "caption": caption}


def _images(images: list[Optional[Markup]], name: str) -> Optional[dict]:
    """One image per dataset, shown side by side."""
    if all(image is None for image in images):
        return None
    return {"kind": "images", "images": images, "name": name}


def _row(label: str, summaries: list[dict], key: str, formatter=fmt, alert_field: Optional[str] = None) -> tuple:
    values = [formatter(s[key]) if key in s else "" for s in summaries]
    return (label, values, any(alert_field in s.get("alert_fields", ()) for s in summaries) if alert_field else False)


# endregion
# region variables
def _numeric(summaries: list[dict]) -> tuple[list, list]:
    num = lambda value: fmt_numeric(value, PRECISION)  # noqa: E731
    top = [
        _table(
            [
                _row("Distinct", summaries, "n_distinct", alert_field="n_distinct"),
                _row("Distinct (%)", summaries, "p_distinct", fmt_percent, "p_distinct"),
                _row("Missing", summaries, "n_missing", alert_field="n_missing"),
                _row("Missing (%)", summaries, "p_missing", fmt_percent, "p_missing"),
                _row("Infinite", summaries, "n_infinite", alert_field="n_infinite"),
                _row("Infinite (%)", summaries, "p_infinite", fmt_percent, "p_infinite"),
                _row("Mean", summaries, "mean", num),
            ]
        ),
        _table(
            [
                _row("Minimum", summaries, "min", num),
                _row("Maximum", summaries, "max", num),
                _row("Zeros", summaries, "n_zeros", alert_field="n_zeros"),
                _row("Zeros (%)", summaries, "p_zeros", fmt_percent, "p_zeros"),
                _row("Negative", summaries, "n_negative"),
                _row("Negative (%)", summaries, "p_negative", fmt_percent),
                _row("Memory size", summaries, "memory_size", fmt_bytesize),
            ]
        ),
        _image(_histogram_plot([s.get("histogram", ()) for s in summaries], mini=True)),
    ]
    statistics = [
        _table(
            [
                _row("Minimum", summaries, "min", num),
                _row("5-th percentile", summaries, "5%", num),
                _row("Q1", summaries, "25%", num),
                _row("median", summaries, "50%", num),
                _row("Q3", summaries, "75%", num),
                _row("95-th percentile", summaries, "95%", num),
                _row("Maximum", summaries, "max", num),
                _row("Range", summaries, "range", num),
                _row("Interquartile range (IQR)", summaries, "iqr", num),
            ],
            name="Quantile statistics",
        ),
        _table(
            [
                _row("Standard deviation", summaries, "std", num),
                _row("Coefficient of variation (CV)", summaries, "cv", num),
                _row("Kurtosis", summaries, "kurtosis", num),
                _row("Mean", summaries, "mean", num),
                _row("Median Absolute Deviation (MAD)", summaries, "mad", num),
                _row("Skewness", summaries, "skewness", num, "skewness"),
                _row("Sum", summaries, "sum", num),
                _row("Variance", summaries, "variance", num),
                _row("Monotonicity", summaries, "monotonic", fmt_monotonic),
            ],
            name="Descriptive statistics",
        ),
    ]
    histograms = [s.get("histogram", ()) for s in summaries]
    bins = max((len(h[1]) - 1 for h in histograms if len(h)), default=0)
    caption = Markup("<strong>Histogram with fixed size bins</strong> (bins={})").format(bins)
    tabs = [
        ("Statistics", statistics),
        ("Histogram", [_image(_histogram_plot(histograms), caption=caption)]),
        (
            "Common values",
            [_freq([_frequency_rows(s["value_counts_without_nan"], s["n"], N_FREQUENT) for s in summaries])],
        ),
        (
            "Extreme values",
            [
                _freq(
                    [_extreme_rows(s["value_counts_index_sorted"], s["n"]) for s in summaries],
                    f"Minimum {N_EXTREME} values",
                ),
                _freq(
                    [_extreme_rows(s["value_counts_index_sorted"][::-1], s["n"]) for s in summaries],
                    f"Maximum {N_EXTREME} values",
                ),
            ],
            "tabs",
        ),
    ]
    return top, tabs


def _counts_table(summaries: list[dict]) -> dict:
    return _table(
        [
            _row("Distinct", summaries, "n_distinct", alert_field="n_distinct"),
            _row("Distinct (%)", summaries, "p_distinct", fmt_percent, "p_distinct"),
            _row("Missing", summaries, "n_missing", alert_field="n_missing"),
            _row("Missing (%)", summaries, "p_missing", fmt_percent, "p_missing"),
            _row("Memory size", summaries, "memory_size", fmt_bytesize),
        ]
    )


def _unique_and_sample(summaries: list[dict]) -> list:
    rows = ("1st row", "2nd row", "3rd row", "4th row", "5th row")
    unique = _table(
        [
            ("Unique", [fmt(s["n_unique"]) for s in summaries], False, UNIQUE_HELP),
            ("Unique (%)", [fmt_percent(s["p_unique"]) for s in summaries]),
        ],
        name="Unique",
    )
    samples = [list(s["first_rows"]) for s in summaries]
    sample = _table(
        [(label, [fmt(rows_[i]) if i < len(rows_) else "" for rows_ in samples]) for i, label in enumerate(rows)],
        name="Sample",
    )
    return [unique, sample]


def _categorical(summaries: list[dict]) -> tuple[list, list]:
    kind = summaries[0]["type"]
    small_n = (lambda s: s["n"]) if kind == "Boolean" else (lambda s: s["count"])
    top = [
        _counts_table(summaries),
        _freq(
            [_frequency_rows(s["value_counts_without_nan"], small_n(s), N_FREQUENT_SMALL[kind]) for s in summaries],
            small=True,
        ),
    ]
    common = _freq([_frequency_rows(s["value_counts_without_nan"], s["n"], N_FREQUENT) for s in summaries])
    plots = _images(
        [
            _category_plot(s["value_counts_without_nan"]) if s["n_distinct"] <= MAX_PLOTTED_CATEGORIES else None
            for s in summaries
        ],
        "Common Values (Plot)",
    )
    if kind == "Boolean":
        return top, [("Common Values (Table)", [common]), ("Common Values (Plot)", [plots])]
    common["name"] = "Common Values"
    return top, [("Overview", _unique_and_sample(summaries)), ("Categories", [common, plots], "stack")]


def _text(summaries: list[dict]) -> tuple[list, list]:
    num = lambda value: fmt_numeric(value, PRECISION)  # noqa: E731
    length = _table(
        [
            _row("Max length", summaries, "max_length", num),
            _row("Median length", summaries, "median_length", num),
            _row("Mean length", summaries, "mean_length", num),
            _row("Min length", summaries, "min_length", num),
        ],
        name="Length",
    )
    characters = _table(
        [
            _row("Total characters", summaries, "n_characters"),
            _row("Distinct characters", summaries, "n_characters_distinct"),
        ],
        name="Characters",
    )
    words = [
        _frequency_rows(s["word_counts"], int(s["word_counts"].sum()), N_COMMON_WORDS) if "word_counts" in s else []
        for s in summaries
    ]
    most_characters = [_frequency_rows(s["character_counts"], s["n_characters"], N_FREQUENT) for s in summaries]
    tabs = [
        ("Overview", [length, characters] + _unique_and_sample(summaries)),
        ("Words", [_freq(words, "Common words")]),
        ("Characters", [_freq(most_characters, "Most occurring characters")]),
    ]
    return [_counts_table(summaries)], tabs


def _date(summaries: list[dict]) -> tuple[list, list]:
    top = [
        _counts_table(summaries),
        _table([_row("Minimum", summaries, "min"), _row("Maximum", summaries, "max")]),
        _image(_histogram_plot([s.get("histogram", ()) for s in summaries], mini=True, date=True)),
    ]
    histogram = _image(_histogram_plot([s.get("histogram", ()) for s in summaries], date=True))
    return top, [("Histogram", [histogram])]


def _unsupported(summaries: list[dict]) -> tuple[list, list]:
    table = _table(
        [
            _row("Missing", summaries, "n_missing", alert_field="n_missing"),
            _row("Missing (%)", summaries, "p_missing", fmt_percent, "p_missing"),
            _row("Memory size", summaries, "memory_size", fmt_bytesize),
        ]
    )
    return [table], []


_RENDERERS = {
    "Numeric": _numeric,
    "Categorical": _categorical,
    "Boolean": _categorical,
    "Text": _text,
    "DateTime": _date,
    "Unsupported": _unsupported,
}


def _anchor(column: Optional[str]) -> str:
    try:
        return slugify(str(column))
    except Exception:  # names with no letters or digits
        return f"col-{abs(hash(column))}"


def _unique_anchors(columns: list[str]) -> dict[str, str]:
    """Anchors per column; names with the same slug, like age and Age, get numbered."""
    anchors, used = {}, set()
    for column in columns:
        anchor = base = _anchor(column)
        n = 1
        while anchor in used:
            n += 1
            anchor = f"{base}-{n}"
        anchors[column] = anchor
        used.add(anchor)
    return anchors


def _tab(name: str, items: list, layout: str = "row") -> dict:
    """
    A tab of variable details laid out as in ydata-profiling: its items side by side ("row"), one under another
    ("stack"), or each in a tab of its own named after the item ("tabs").
    """
    items = [item for item in items if item]
    if layout == "tabs":
        items = [{"name": item["name"], "items": [item | {"name": None}]} for item in items]
    return {"name": name, "items": items, "layout": layout}


def _variable(column: str, profiles: list[DataProfile], anchor: str) -> dict:
    alerts = [
        [a for a in p.alerts if a.column_name == column and a.alert_type != AlertType.REJECTED] for p in profiles
    ]
    summaries = [
        None
        if column not in p.variables
        else p.variables[column]
        | {"alert_fields": {key for a in found for key in _ALERT_FIELDS.get(a.alert_type, ())}}
        for p, found in zip(profiles, alerts)
    ]

    present = [s for s in summaries if s is not None]
    types = {s["type"] for s in present}
    if len(types) == 1 and len(present) == len(summaries):
        sections = [_RENDERERS[types.pop()](summaries)]
    else:  # types differ between the datasets: render each separately
        sections = [_RENDERERS[s["type"]]([s]) if s is not None else ([], []) for s in summaries]

    return {
        "name": column,
        "anchor": anchor,
        "types": [Markup(TYPE_LABELS[s["type"]]) if s is not None else "(not present)" for s in summaries],
        "alerts": [[_alert_badge(a) for a in column_alerts] for column_alerts in alerts],
        "sections": [
            {"top": [item for item in top if item], "tabs": [_tab(*tab) for tab in tabs]} for top, tabs in sections
        ],
    }


# Statistics highlighted in the tables when their alert fires
_ALERT_FIELDS = {
    AlertType.CONSTANT: {"n_distinct"},
    AlertType.HIGH_CARDINALITY: {"n_distinct"},
    AlertType.IMBALANCE: {"imbalance"},
    AlertType.INFINITE: {"p_infinite", "n_infinite"},
    AlertType.MISSING: {"p_missing", "n_missing"},
    AlertType.SKEWED: {"skewness"},
    AlertType.UNIQUE: {"n_distinct", "p_distinct", "n_unique", "p_unique"},
    AlertType.ZEROS: {"n_zeros", "p_zeros"},
}

# endregion


class ProfileReport:
    """HTML report for one profile, or for two profiles side by side."""

    def __init__(self, profiles: list[DataProfile], title: str, labels: Optional[list[str]] = None):
        """
        Parameters
        ----------
        profiles : list[DataProfile]
            One profile, or two to compare.
        title : str
            The report title. A comparison is titled after its datasets instead, as in ydata-profiling.
        labels : Optional[list[str]], optional
            Names for the datasets when comparing, by default the profile titles.
        """
        if not 1 <= len(profiles) <= 2:
            raise ValueError("A report shows one profile or compares two.")
        self.profiles = profiles
        self.title = title
        self.labels = labels or [p.title for p in profiles]

    @property
    def alerts(self) -> list[Alert]:
        """Alerts of the first profile."""
        return self.profiles[0].alerts

    def to_html(self) -> str:
        """The report as a standalone HTML page."""
        profiles = self.profiles
        columns = list(dict.fromkeys(col for p in profiles for col in p.variables))
        anchors = _unique_anchors(columns)
        tables = [p.table for p in profiles]
        types = list(dict.fromkeys(t for table in tables for t in table["types"]))
        overview = [
            _table(
                [
                    ("Number of variables", [fmt(t["n_var"]) for t in tables]),
                    ("Number of observations", [fmt(t["n"]) for t in tables]),
                    ("Missing cells", [fmt(t["n_cells_missing"]) for t in tables]),
                    ("Missing cells (%)", [fmt_percent(t["p_cells_missing"]) for t in tables]),
                    ("Total size in memory", [fmt_bytesize(t["memory_size"]) for t in tables]),
                    ("Average record size in memory", [fmt_bytesize(t["record_size"]) for t in tables]),
                ],
                name="Dataset statistics",
            ),
            _table([(t, [fmt(table["types"].get(t, 0)) for table in tables]) for t in types], name="Variable types"),
        ]
        # One row per alert type and column, with what each dataset shows
        alerts = {}
        for idx, profile in enumerate(profiles):
            for alert in profile.alerts:
                if alert.alert_type != AlertType.REJECTED:
                    row = alerts.setdefault((alert.alert_type, alert.column_name), [None] * len(profiles))
                    html = alert_html(alert, anchors.get(alert.column_name))
                    row[idx] = {"badge": _alert_badge(alert), "html": Markup(html)}
        reproduction = _table(
            [
                ("Analysis started", [str(p.started) for p in profiles]),
                ("Analysis finished", [str(p.finished) for p in profiles]),
                ("Duration", [fmt_timespan((p.finished - p.started).total_seconds()) for p in profiles]),
                ("Software version", [f"seismometer v{seismometer.__version__}" for _ in profiles]),
            ],
            name="Reproduction",
        )
        template = _env.get_template("profile_report.html")
        return template.render(
            title=self.title,
            labels=self.labels,
            colors=COLORS,
            overview=overview,
            alerts=list(alerts.values()),
            reproduction=reproduction,
            variables=[_variable(col, profiles, anchors[col]) for col in columns],
        )

    def to_file(self, path: str | Path) -> None:
        """
        Writes the report as an HTML file.

        Parameters
        ----------
        path : str | Path
            The file to write.
        """
        page = self.to_html()  # before opening, so a failed render leaves no empty file
        with open(path, "w", encoding="utf-8") as f:
            f.write(page)
