import warnings

import pandas as pd
from IPython.display import SVG
from pandas.errors import SettingWithCopyWarning

import seismometer.plot as plot
from seismometer.api.plots import _plot_leadtime_enc


def test_plot_leadtime_enc_no_settingwithcopywarning(monkeypatch):
    monkeypatch.setattr(plot, "leadtime_violin", lambda *a, **k: SVG("<svg></svg>"))

    df = pd.DataFrame(
        {
            "cohort": ["A", "A", "B", "B"],
            "event": [1, 1, 1, 1],
            "time_zero": pd.to_datetime(["2025-01-01 00:00:00"] * 4),
            "pred_time": pd.to_datetime(
                ["2025-01-01 02:00:00", "2025-01-01 03:00:00", "2025-01-01 01:00:00", "2025-01-01 04:00:00"]
            ),
            "score": [0.9, 0.8, 0.95, 0.7],
            "entity_id": [1, 1, 2, 2],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SettingWithCopyWarning)

        html = _plot_leadtime_enc(
            dataframe=df,
            entity_keys=["entity_id"],
            target_event="event",
            target_zero="time_zero",
            score="score",
            threshold=0.75,
            ref_time="pred_time",
            cohort_col="cohort",
            subgroups=["A", "B"],
            max_hours=24,
            x_label="Lead Time (hours)",
            censor_threshold=0,
        )

    assert html is not None
    assert hasattr(html, "data")
    assert not any(
        isinstance(w.message, SettingWithCopyWarning) for w in caught
    ), "Expected no SettingWithCopyWarning, but one was emitted."
