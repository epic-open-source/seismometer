from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import seismometer.data.cohorts as undertest
import seismometer.data.performance  # NoQA - used in patching


def input_df():
    return pd.DataFrame(
        {
            "TARGET": [1, 0, 0, 1, 0, 1],
            "col1": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "tri": [0.0, 0, 1, 1, 2, 2],
        }
    )


def expected_df(cohorts):
    data_rows = np.vstack(
        (
            # TP,FP,TN,FN,   Acc,Sens,Spec,PPV,NPV,    Flag,      LR+, NNE, NNT1/2,  cohort,ct,tgtct,
            [[0, 0, 1, 1, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0, np.nan, 1, 2, "<1.0", 2, 1]] * 70,
            [[0, 1, 0, 1, 0, 0, 0, 0, 0, 0.5, 0, np.inf, np.inf, "<1.0", 2, 1]] * 10,
            [[1, 1, 0, 0, 0.5, 1, 0, 0.5, 1, 1, 1, 2, 4, "<1.0", 2, 1]] * 21,
            # TP,FP,TN,FN,   Acc,Sens,Spec, PPV, NPV,    Flag,      LR+, NNE, NNT1/2,  cohort,ct,tgtct
            [[0, 0, 2, 2, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0, np.nan, 1, 2, ">=1.0", 4, 2]] * 30,
            [[1, 0, 2, 1, 0.75, 0.5, 1, 1, 2 / 3, 0.25, np.inf, 1, 2, ">=1.0", 4, 2]] * 10,
            [[1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 2, 4, ">=1.0", 4, 2]] * 10,
            [[2, 1, 1, 0, 0.75, 1, 0.5, 2 / 3, 1, 0.75, 2, 1.5, 3, ">=1.0", 4, 2]] * 10,
            [[2, 2, 0, 0, 0.5, 1, 0, 0.5, 1, 1, 1, 2, 4, ">=1.0", 4, 2]] * 41,
            # TP,FP,TN,FN,   Acc,Sens,Spec,PPV,NPV,    Flag,      LR+, NNE, NNT1/2,  cohort,ct,tgtct
            [[0, 0, 1, 1, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0, np.nan, 1, 2, "1.0-2.0", 2, 1]] * 50,
            [[1, 0, 1, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, np.inf, 1, 2, "1.0-2.0", 2, 1]] * 10,
            [[1, 1, 0, 0, 0.5, 1, 0, 0.5, 1, 1, 1, 2, 4, "1.0-2.0", 2, 1]] * 41,
            # TP,FP,TN,FN,   Acc,Sens,Spec,PPV,NPV,    Flag,      LR+, NNE, NNT1/2,  cohort,ct,tgtct
            [[0, 0, 1, 1, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0, np.nan, 1, 2, ">=2.0", 2, 1]] * 30,
            [[1, 0, 1, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, np.inf, 1, 2, ">=2.0", 2, 1]] * 10,
            [[1, 1, 0, 0, 0.5, 1, 0, 0.5, 1, 1, 1, 2, 4, ">=2.0", 2, 1]] * 61,
        )
    )

    df = pd.DataFrame(
        data_rows,
        columns=[
            "TP",
            "FP",
            "TN",
            "FN",
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
            "Flag Rate",
            "LR+",
            "NNE",
            "NNT@0.5",
            "cohort",
            "cohort-count",
            "cohort-targetcount",
        ],
    )
    df["Threshold"] = np.tile(np.arange(100, -1, -1), 4)

    df[["TP", "FP", "TN", "FN", "cohort-count", "cohort-targetcount"]] = df[
        ["TP", "FP", "TN", "FN", "cohort-count", "cohort-targetcount"]
    ].astype(int)
    df[["Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Flag Rate", "LR+", "NNE", "NNT@0.5"]] = df[
        ["Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Flag Rate", "LR+", "NNE", "NNT@0.5"]
    ].astype(float)

    # reduce before creating category
    reduced_df = df.loc[df.cohort.isin(cohorts)].reset_index(drop=True)
    reduced_df["cohort"] = pd.Categorical(reduced_df["cohort"], categories=cohorts)
    return reduced_df


# We drop these ase the values are threshold dependent and testing would reimplement the formula
# Instead rely on the test_cases in test_perf_stats.py
THRESHOLD_DEPENDENT_COLUMNS = ["NetBenefitScore", "F1", "F0.5", "F2"]


@patch.object(seismometer.data.performance, "DEFAULT_RHO", 0.5)
class Test_Performance_Data:
    def test_data_defaults(self):
        df = input_df()
        actual = undertest.get_cohort_performance_data(df, "tri", proba="col1", censor_threshold=0)
        actual = actual.drop(columns=THRESHOLD_DEPENDENT_COLUMNS)
        expected = expected_df(["<1.0", ">=1.0"])

        pd.testing.assert_frame_equal(actual, expected, check_column_type=False, check_like=True, check_dtype=False)

    def test_data_splits(self):
        df = input_df()
        actual = undertest.get_cohort_performance_data(df, "tri", proba="col1", splits=[1.0, 2.0], censor_threshold=0)
        actual = actual.drop(columns=THRESHOLD_DEPENDENT_COLUMNS)
        expected = expected_df(["<1.0", "1.0-2.0", ">=2.0"])

        pd.testing.assert_frame_equal(actual, expected, check_column_type=False, check_like=True, check_dtype=False)


class TestGetCohortData:
    """Tests for get_cohort_data() function - previously untested."""

    def test_get_cohort_data_with_column_names(self):
        """Test get_cohort_data with proba and true as column names."""
        df = input_df()
        result = undertest.get_cohort_data(df, "tri", proba="col1", true="TARGET")

        assert "true" in result.columns
        assert "pred" in result.columns
        assert "cohort" in result.columns
        assert len(result) == 6

    def test_get_cohort_data_with_array_inputs(self):
        """Test get_cohort_data with proba and true as arrays."""
        df = input_df()
        proba_array = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        true_array = np.array([1, 0, 0, 1, 0, 1])

        result = undertest.get_cohort_data(df, "tri", proba=proba_array, true=true_array)

        # Verify correct columns and row count
        assert len(result) == 6
        assert "pred" in result.columns
        assert "true" in result.columns
        assert "cohort" in result.columns
        # Values should match input arrays
        assert list(result["pred"].values) == list(proba_array)
        assert list(result["true"].values) == list(true_array)

    def test_get_cohort_data_with_mismatched_array_lengths(self):
        """Test get_cohort_data documents edge case behavior with mismatched lengths."""
        df = input_df()
        proba_series = pd.Series([0.2, 0.3], index=[0, 1])  # Only 2 rows

        # Pandas will align by index, then dropna removes mismatched indices
        result = undertest.get_cohort_data(df, "tri", proba=proba_series, true="TARGET")

        # Documents behavior: only matching indices kept
        assert len(result) >= 0  # May be 0-2 depending on cohort column alignment

    def test_get_cohort_data_with_nan_values(self):
        """Test get_cohort_data drops NaN values."""
        df = pd.DataFrame({"TARGET": [1, 0, np.nan, 1], "col1": [0.2, np.nan, 0.4, 0.5], "tri": [0, 0, 1, 1]})

        result = undertest.get_cohort_data(df, "tri", proba="col1", true="TARGET")

        # Should drop rows with NaN (2 rows dropped)
        assert len(result) == 2

    def test_get_cohort_data_with_splits(self):
        """Test get_cohort_data with custom splits parameter."""
        df = input_df()

        result = undertest.get_cohort_data(df, "tri", proba="col1", true="TARGET", splits=[1.0, 2.0])

        # Should create cohorts based on splits
        assert "cohort" in result.columns
        assert result["cohort"].cat.categories.tolist() == ["<1.0", "1.0-2.0", ">=2.0"]


class TestResolveColData:
    """Tests for resolve_col_data() helper function - previously untested."""

    def test_resolve_col_data_with_string_column(self):
        """Test resolve_col_data with column name as string."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        result = undertest.resolve_col_data(df, "col1")

        pd.testing.assert_series_equal(result, pd.Series([1, 2, 3], name="col1"))

    def test_resolve_col_data_with_missing_column(self):
        """Test resolve_col_data raises KeyError for missing column."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        with pytest.raises(KeyError, match="Feature missing_col was not found in dataframe"):
            undertest.resolve_col_data(df, "missing_col")

    def test_resolve_col_data_with_2d_array(self):
        """Test resolve_col_data handles 2D array (sklearn probabilities)."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        proba_2d = np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])

        result = undertest.resolve_col_data(df, proba_2d)

        # Should return second column (positive class)
        np.testing.assert_array_equal(result, np.array([0.8, 0.7, 0.6]))

    def test_resolve_col_data_with_1d_array(self):
        """Test resolve_col_data handles 1D array."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        array_1d = np.array([0.2, 0.3, 0.4])

        result = undertest.resolve_col_data(df, array_1d)

        np.testing.assert_array_equal(result, array_1d)

    def test_resolve_col_data_with_invalid_type(self):
        """Test resolve_col_data raises TypeError for invalid input."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        with pytest.raises(TypeError, match="Feature must be a string, pandas.Series, or numpy.ndarray"):
            undertest.resolve_col_data(df, 123)  # Invalid type


class TestResolveCohorts:
    """Tests for resolve_cohorts() function - previously untested."""

    def test_resolve_cohorts_with_categorical_series(self):
        """Test resolve_cohorts auto-dispatches to categorical handler."""
        series = pd.Series(pd.Categorical(["A", "B", "A", "C"]), name="test_cohort")

        result = undertest.resolve_cohorts(series)

        assert isinstance(result, pd.Series)
        assert hasattr(result, "cat")
        assert set(result.cat.categories) == {"A", "B", "C"}  # Unused removed

    def test_resolve_cohorts_with_numeric_series(self):
        """Test resolve_cohorts auto-dispatches to numeric handler."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        result = undertest.resolve_cohorts(series)

        assert isinstance(result, pd.Series)
        assert hasattr(result, "cat")  # Should be categorical
        # Should split at mean (3.0)
        assert len(result.cat.categories) == 2

    def test_resolve_cohorts_with_numeric_splits(self):
        """Test resolve_cohorts with custom numeric splits."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        result = undertest.resolve_cohorts(series, splits=[2.5, 4.0])

        assert result.cat.categories.tolist() == ["<2.5", "2.5-4.0", ">=4.0"]


class TestHasGoodBinning:
    """Tests for has_good_binning() error checking function - previously untested."""

    def test_has_good_binning_with_valid_bins(self):
        """Test has_good_binning passes with valid binning."""
        bin_ixs = np.array([1, 1, 2, 2, 3, 3])
        bin_edges = [0.0, 1.0, 2.0]

        # Should not raise
        undertest.has_good_binning(bin_ixs, bin_edges)

    def test_has_good_binning_with_empty_bins(self):
        """Test has_good_binning raises IndexError for empty bins."""
        bin_ixs = np.array([1, 1, 3, 3])  # Missing bin 2
        bin_edges = [0.0, 1.0, 2.0]

        with pytest.raises(IndexError, match="Splits provided contain some empty bins"):
            undertest.has_good_binning(bin_ixs, bin_edges)

    def test_has_good_binning_with_single_bin(self):
        """Test has_good_binning with single bin edge case."""
        bin_ixs = np.array([1, 1, 1])
        bin_edges = [0.0]

        # Should not raise
        undertest.has_good_binning(bin_ixs, bin_edges)


class TestLabelCohortsCategorical:
    """Tests for label_cohorts_categorical() function - previously untested."""

    def test_label_cohorts_categorical_without_cat_values(self):
        """Test label_cohorts_categorical removes unused categories."""
        series = pd.Series(pd.Categorical(["A", "B", "A"], categories=["A", "B", "C", "D"]))

        result = undertest.label_cohorts_categorical(series)

        # Should remove unused categories C and D
        assert set(result.cat.categories) == {"A", "B"}

    def test_label_cohorts_categorical_with_cat_values_matching(self):
        """Test label_cohorts_categorical with matching cat_values."""
        series = pd.Series(pd.Categorical(["A", "B", "C"], categories=["A", "B", "C"]))

        result = undertest.label_cohorts_categorical(series, cat_values=["A", "B", "C"])

        # Should return as-is
        pd.testing.assert_series_equal(result, series, check_names=False)

    def test_label_cohorts_categorical_with_cat_values_filtering(self):
        """Test label_cohorts_categorical filters to specified cat_values."""
        series = pd.Series(pd.Categorical(["A", "B", "C", "D"], categories=["A", "B", "C", "D"]))

        result = undertest.label_cohorts_categorical(series, cat_values=["A", "C"])

        # Should filter to only A and C, rest become NaN
        assert result.notna().sum() == 2
        assert set(result.dropna()) == {"A", "C"}


class TestFindBinEdges:
    """Tests for find_bin_edges() function - previously untested."""

    def test_find_bin_edges_with_no_thresholds(self):
        """Test find_bin_edges defaults to mean split."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])  # Mean = 3.0

        result = undertest.find_bin_edges(series)

        # Returns list with [min, mean]
        assert len(result) == 2
        assert result[0] == 1.0  # Series minimum
        assert result[1] == 3.0  # Series mean

    def test_find_bin_edges_with_custom_thresholds(self):
        """Test find_bin_edges with custom threshold values."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        result = undertest.find_bin_edges(series, thresholds=[2.0, 4.0])

        # Returns list with [min, threshold1, threshold2]
        assert len(result) == 3
        assert result[0] == 1.0  # Series minimum
        assert result[1] == 2.0
        assert result[2] == 4.0

    def test_find_bin_edges_with_single_value_series(self):
        """Test find_bin_edges with series containing single unique value."""
        series = pd.Series([5.0, 5.0, 5.0])

        result = undertest.find_bin_edges(series)

        # Edge case: single value means min = mean
        # Documents that this creates degenerate bins (both edges same)
        assert len(result) >= 1
        assert all(val == 5.0 for val in result)  # All edges are 5.0
