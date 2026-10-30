"""
Regenerates tests/resources/profile/golden.json, the ydata-profiling results the profile tests compare against.

Needs pandas 2 and ydata-profiling 4.18, which seismometer no longer depends on, plus the report_config.yml that
seismometer passed to ydata (seismometer 0.5.0, src/seismometer/report/report_config.yml):

    python tests/data/make_profile_golden.py path/to/report_config.yml
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from profile_cases import make_profile_frame
from ydata_profiling import ProfileReport, __version__
from ydata_profiling.report.presentation.flavours.html import templates

OUT = Path(__file__).parents[1] / "resources" / "profile" / "golden.json"
SCALARS = [
    "type", "n", "count", "n_missing", "p_missing", "n_distinct", "p_distinct", "n_unique", "p_unique", "is_unique",
    "imbalance", "n_zeros", "p_zeros", "n_infinite", "p_infinite", "n_negative", "p_negative", "mean", "std",
    "variance", "min", "max", "kurtosis", "skewness", "sum", "mad", "range", "iqr", "cv", "monotonic",
    "5%", "25%", "50%", "75%", "95%", "memory_size", "top", "freq", "max_length", "mean_length", "median_length",
    "min_length", "n_characters", "n_characters_distinct", "n_invalid_dates",
]  # fmt: skip


def plain(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value if isinstance(value, (int, float, str, bool, type(None))) else str(value)


def counts(series: pd.Series) -> dict:
    # Every value, since pandas 3 orders tied counts differently than pandas 2
    return {str(label): int(count) for label, count in series.items()}


def main(config_path: str):
    report = ProfileReport(make_profile_frame(), config_file=config_path, title="Profile")
    description = report.get_description()

    variables = {}
    for col, summary in description.variables.items():
        golden = {key: plain(summary[key]) for key in SCALARS if key in summary}
        for key in ["value_counts_without_nan", "character_counts", "word_counts"]:
            if key in summary:
                golden[key] = counts(summary[key])
        if "first_rows" in summary:
            golden["first_rows"] = [str(v) for v in summary["first_rows"]]
        for key in ["histogram", "histogram_length"]:
            if key in summary and len(summary[key]):
                golden[key] = [summary[key][0].tolist(), summary[key][1].tolist()]
        variables[col] = golden

    alerts = []
    for alert in description.alerts:
        text = None
        if alert.alert_type.name != "REJECTED":  # no template; seismometer never shows it
            html = templates.template(f"alerts/alert_{alert.alert_type.name.lower()}.html").render(alert=alert)
            text = re.sub(r"<[^>]+>", "", str(html))
        alerts.append({"type": alert.alert_type.name, "column": alert.column_name, "text": text})

    table = {key: plain(value) for key, value in description.table.items() if key != "types"}
    table["types"] = {key: int(value) for key, value in description.table["types"].items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    golden = {"ydata_profiling": __version__, "pandas": pd.__version__, "table": table}
    golden |= {"variables": variables, "alerts": alerts}
    OUT.write_text(json.dumps(golden, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
