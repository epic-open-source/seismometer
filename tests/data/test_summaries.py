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
        fake_seismo.predict_time = "Target"
        fake_seismo.event_aggregation_method = lambda x: aggregation_method

        # Calculate score target cohort summary using event_score
        ref_time = "Target"
        ref_event = "Target"
        df_aggregated = event_score(prediction_data, ["ID"], "Score", ref_time, ref_event, aggregation_method)
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


class TestDefaultCohortSummariesErrorHandling:
    """Test error handling for default_cohort_summaries()."""

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_missing_entity_id_col(self, mock_seismo, prediction_data):
        """Test default_cohort_summaries with missing entity_id_col."""
        fake_seismo = mock_seismo()
        fake_seismo.output = "Score"
        fake_seismo.target = "Target"
        fake_seismo.predict_time = "Target"
        fake_seismo.event_aggregation_method = lambda x: "max"

        # Missing entity_id_col causes ValueError in pandas drop_duplicates
        with pytest.raises(ValueError):
            undertest.default_cohort_summaries(prediction_data, "Has_ECG", [1, 2, 3], "ID_MISSING")

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_invalid_attribute_column(self, mock_seismo, prediction_data):
        """Test default_cohort_summaries with invalid attribute column."""
        fake_seismo = mock_seismo()
        fake_seismo.output = "Score"
        fake_seismo.target = "Target"
        fake_seismo.event_aggregation_method = lambda x: "max"

        with pytest.raises(KeyError, match="INVALID_ATTR"):
            undertest.default_cohort_summaries(prediction_data, "INVALID_ATTR", [1, 2, 3], "ID")

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_empty_dataframe(self, mock_seismo):
        """Test default_cohort_summaries with empty dataframe."""
        fake_seismo = mock_seismo()
        fake_seismo.output = "Score"
        fake_seismo.target = "Target"
        fake_seismo.predict_time = "Target_Time"
        fake_seismo.event_aggregation_method = lambda x: "max"

        empty_df = pd.DataFrame(
            {"ID": [], "Has_ECG": [], "Score": [], "Target": [], "Target_Time": [], "Target_Value": []}
        )
        result = undertest.default_cohort_summaries(empty_df, "Has_ECG", [1, 2, 3], "ID")

        # Should return a dataframe with options as index but NaN values
        assert len(result) == 3
        assert result.index.tolist() == [1, 2, 3]

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_seismogram_none_values(self, mock_seismo, prediction_data):
        """Test default_cohort_summaries with sg.output/sg.target = None."""
        fake_seismo = mock_seismo()
        fake_seismo.output = None  # This will cause AttributeError in event_value()
        fake_seismo.target = "Target"
        fake_seismo.predict_time = "Target_Time"
        fake_seismo.event_aggregation_method = lambda x: "max"

        # event_score will fail when trying to call .endswith() on None
        with pytest.raises(AttributeError):
            undertest.default_cohort_summaries(prediction_data, "Has_ECG", [1, 2, 3], "ID")


class TestScoreTargetCohortSummariesErrorHandling:
    """Test error handling for score_target_cohort_summaries()."""

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_misaligned_groups(self, mock_seismo, prediction_data):
        """Test score_target_cohort_summaries with misaligned groupby and grab groups."""
        fake_seismo = mock_seismo()
        fake_seismo.output = "Score"
        fake_seismo.target = "Target"
        fake_seismo.predict_time = "Target"
        fake_seismo.event_aggregation_method = lambda x: "max"

        # groupby_groups contains column that's not in grab_groups
        groupby_groups = ["Has_ECG", "Target_Value"]
        grab_groups = ["Has_ECG"]  # Missing Target_Value

        # This should fail because groupby references columns not in grab
        with pytest.raises((KeyError, ValueError)):
            undertest.score_target_cohort_summaries(prediction_data, groupby_groups, grab_groups, "ID")

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_missing_columns(self, mock_seismo, prediction_data):
        """Test score_target_cohort_summaries with missing columns in dataframe."""
        fake_seismo = mock_seismo()
        fake_seismo.output = "Score"
        fake_seismo.target = "Target"
        fake_seismo.predict_time = "Target"
        fake_seismo.event_aggregation_method = lambda x: "max"

        groupby_groups = ["MISSING_COL"]
        grab_groups = ["MISSING_COL"]

        with pytest.raises(KeyError, match="MISSING_COL"):
            undertest.score_target_cohort_summaries(prediction_data, groupby_groups, grab_groups, "ID")
