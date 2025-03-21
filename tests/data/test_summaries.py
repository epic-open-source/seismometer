from unittest.mock import Mock, patch

import pandas as pd
import pytest
from conftest import res  # noqa: F401

from seismometer import seismogram
from seismometer.data import summaries as undertest


@pytest.fixture
def prediction_data(res):
    file = res / "summaries/input_predictions.tsv"

    return pd.read_csv(file, sep="\t")


@pytest.fixture
def expected_default_summary(res):
    file = res / "summaries/expected_default_summary.tsv"

    return pd.read_csv(file, sep="\t", index_col=0)


@pytest.fixture
def expected_score_target_summary(res):
    file = res / "summaries/expected_score_target_summary.tsv"

    df = pd.read_csv(file, sep="\t", index_col=[0])
    df["Score"] = (
        df["Score"]
        .str.strip("()[]")
        .str.split(",", expand=True)
        .astype(float)
        .apply(lambda x: pd.Interval(x[0], x[1]), axis=1)
        .astype(pd.CategoricalDtype([pd.Interval(0, 0.5), pd.Interval(0.5, 1)], ordered=True))
    )
    df["Entities"] = df["Entities"].astype("Int64")
    df = df.set_index("Score", append=True)

    return df


@pytest.fixture
def expected_score_target_summary_cuts(res):
    file = res / "summaries/input_predictions.tsv"

    df = pd.read_csv(file, sep="\t")

    return pd.cut(df["Score"], [0, 0.5, 1])


class Test_Summaries:
    def test_default_summaries(self, prediction_data, expected_default_summary):
        actual = undertest.default_cohort_summaries(prediction_data, "Has_ECG", [1, 2, 3, 4, 5], "ID")
        pd.testing.assert_frame_equal(actual, expected_default_summary, check_names=False)

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_score_target_summaries(
        self, mock_seismo, prediction_data, expected_score_target_summary, expected_score_target_summary_cuts
    ):
        fake_seismo = mock_seismo()
        fake_seismo.output = "Score"
        fake_seismo.target = "Target"
        fake_seismo.event_aggregation_method = lambda x: "max"
        groupby_groups = ["Has_ECG", expected_score_target_summary_cuts]
        grab_groups = ["Has_ECG", "Score"]
        pd.testing.assert_frame_equal(
            undertest.score_target_cohort_summaries(prediction_data, groupby_groups, grab_groups, "ID"),
            expected_score_target_summary,
        )
