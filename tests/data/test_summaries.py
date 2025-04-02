from unittest.mock import Mock, patch

import pandas as pd
import pytest
from conftest import res  # noqa: F401

from seismometer import seismogram
from seismometer.data import summaries as undertest
from seismometer.data.pandas_helpers import event_score


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
    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_default_summaries(self, mock_seismo, prediction_data, expected_default_summary):
        fake_seismo = mock_seismo()
        fake_seismo.output = "Score"
        fake_seismo.target = "Target"
        fake_seismo.event_aggregation_method = lambda x: "max"
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

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    @pytest.mark.parametrize("aggregation_method", ["min", "max", "first", "last"])
    def test_event_score_match_score_target_summaries(
        self, mock_seismo, aggregation_method, prediction_data, expected_score_target_summary_cuts
    ):
        fake_seismo = mock_seismo()
        fake_seismo.output = "Score"
        fake_seismo.target = "Target"
        fake_seismo.event_aggregation_method = lambda x: aggregation_method

        # Calculate score target cohort summary using event_score
        ref_event = "Target"
        df_aggregated = event_score(prediction_data, ["ID"], "Score", ref_event, aggregation_method)
        bins = [0, 0.5, 1.0]
        labels = ["(0,0.5]", "(0.5,1]"]
        # Create a new column for binned scores
        df_aggregated["Score_Binned"] = pd.cut(
            df_aggregated["Score"], bins=bins, labels=labels, include_lowest=False, right=True
        )
        # Group by the desired columns and count
        entities_event_score = (
            df_aggregated.groupby(["Has_ECG", "Target_Value", "Score_Binned"], observed=False)
            .size()
            .reset_index(name="Count")["Count"]
        )

        # Calculate score target cohort summary using score_target_cohort_summaries
        groupby_groups = ["Has_ECG", "Target_Value", expected_score_target_summary_cuts]
        grab_groups = ["Has_ECG", "Score", "Target_Value"]
        entities_summary = undertest.score_target_cohort_summaries(prediction_data, groupby_groups, grab_groups, "ID")[
            "Entities"
        ]

        # Ensuring they produce the same number of entities for each score-target-cohort group
        assert entities_event_score.tolist() == entities_summary.tolist()
