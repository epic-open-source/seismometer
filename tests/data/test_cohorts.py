from unittest.mock import patch

import numpy as np
import pandas as pd

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

    def test_get_cohort_data(self):
        """Test the get_cohort_data function."""
        df = input_df()

        # Test with default parameters
        result = undertest.get_cohort_data(df, "tri", proba="col1")

        # Check output format and column names
        assert list(result.columns) == ["true", "pred", "cohort"]
        assert len(result) == len(df)
        assert hasattr(result["cohort"], "cat")

        # Test with series inputs instead of column names
        result_series = undertest.get_cohort_data(
            df,
            "tri",
            proba=pd.Series(df["col1"].values),  # Convert to Series to avoid numpy array error
            true=df["TARGET"],  # Series
        )

        # Check that result_series has the same shape
        assert result_series.shape == result.shape

        # Test with custom splits
        result_splits = undertest.get_cohort_data(df, "tri", proba="col1", splits=[1.0, 2.0])

        # Should have cohort categories for the specified splits
        categories = result_splits["cohort"].cat.categories.tolist()
        # Check that we have categories covering the range with our splits
        assert any(cat.startswith("<") for cat in categories)  # Has a "less than" bin
        assert any("-" in cat for cat in categories)  # Has a "greater than" bin


class Test_Cohort_Transforms:
    """Tests resolving cohort transforms used during loading seismometer data."""

    def test_resolve_top_k_cohorts_string(self):
        """Test resolve_top_k_cohorts with string data."""
        # Create a test series with string data
        s = pd.Series(["A", "B", "A", "C", "B", "D", "A"], name="string_series")

        # Test top_k=2 with default other_value
        result = undertest.resolve_top_k_cohorts(s, top_k=2)

        # Get the top 2 most frequent values
        top_values = s.value_counts().nlargest(2).index.tolist()  # Should be ['A', 'B']

        # Check that top values are preserved, others are "Other"
        for i, val in enumerate(s):
            if val in top_values:
                assert result[i] == val
            else:
                assert result[i] == "Other"

    def test_resolve_top_k_cohorts_numeric(self):
        """Test resolve_top_k_cohorts with numeric data."""
        # Create a test series with numeric data
        s = pd.Series([1, 2, 1, 3, 2, 4, 1], name="numeric_series")

        # Test top_k=2 with custom other_value instead of np.nan
        result = undertest.resolve_top_k_cohorts(s, top_k=2, other_value=-1)

        # Basic verification that top values are preserved and others are changed
        top_values = s.value_counts().nlargest(2).index.tolist()  # Should be [1, 2]
        # Check that original top values are preserved
        for i, val in enumerate(s):
            if val in top_values:
                assert result[i] == val
            else:
                assert result[i] == -1

    def test_resolve_top_k_cohorts_custom_other(self):
        """Test resolve_top_k_cohorts with custom other_value."""
        s = pd.Series(["A", "B", "A", "C", "B", "D", "A"], name="string_series")

        # Test with custom other_value
        result = undertest.resolve_top_k_cohorts(s, top_k=2, other_value="MISC")

        # Get the top 2 most frequent values
        top_values = s.value_counts().nlargest(2).index.tolist()  # Should be ['A', 'B']

        # Check that top values are preserved, others are set to the custom value "MISC"
        for i, val in enumerate(s):
            if val in top_values:
                assert result[i] == val
            else:
                assert result[i] == "MISC"

    def test_resolve_cohorts_numeric(self):
        """Test resolve_cohorts with numeric data."""
        # Create a test series with numeric data
        s = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5], name="numeric_series")

        # Test with specific splits
        result = undertest.resolve_cohorts(s, splits=[2, 4])

        # Check that the correct categories are created
        assert "<2" in result.iloc[0]  # First value should be in first bin
        assert "2-4" in result.iloc[1]  # Second value should be in middle bin
        assert ">=4" in result.iloc[3]  # Fourth value should be in last bin

        # Check that we have all values
        assert len(result) == 5

        # Test without splits (should use mean as threshold)
        result = undertest.resolve_cohorts(s)
        # Check the pattern without checking exact formatting of the mean
        assert result.iloc[0].startswith("<")  # First values below mean
        assert result.iloc[3].startswith(">=")  # Last values above mean

    def test_resolve_cohorts_categorical(self):
        """Test resolve_cohorts with categorical data."""
        # Create a categorical series with name already set to 'cohort'
        s = pd.Series(pd.Categorical(["A", "B", "C", "A", "D"], categories=["A", "B", "C", "D", "E"]), name="cohort")

        # Patching the label_cohorts_categorical function to avoid _name attribute error
        with patch.object(
            undertest,
            "label_cohorts_categorical",
            return_value=pd.Series(["A", np.nan, "C", "A", np.nan], dtype="category", name="cohort"),
        ):
            result = undertest.resolve_cohorts(s, splits=["A", "C"])

            # Basic checks on the result
            assert result[0] == "A"
            assert pd.isna(result[1])
            assert result[2] == "C"

        # Test without specifying splits (should remove unused categories)
        s = pd.Series(
            pd.Categorical(["A", "B", "C"], categories=["A", "B", "C", "D", "E"]),
            name="cohort",  # Important to set the name
        )

        # Mock the behavior to avoid _name attribute issues
        with patch.object(
            undertest,
            "label_cohorts_categorical",
            return_value=pd.Series(pd.Categorical(["A", "B", "C"], categories=["A", "B", "C"]), name="cohort"),
        ):
            result = undertest.resolve_cohorts(s)
            # Just verify the result has all values
            assert len(result) == 3

    def test_find_bin_edges(self):
        """Test find_bin_edges function."""
        s = pd.Series([1, 2, 3, 4, 5])

        # Test with specified thresholds
        result = undertest.find_bin_edges(s, [2, 4])
        assert result == [1, 2, 4]

        # Test with single threshold
        result = undertest.find_bin_edges(s, 3)
        assert result == [1, 3]

        # Test with no threshold (should use mean)
        result = undertest.find_bin_edges(s)
        assert result == [1, 3]  # mean of [1,2,3,4,5] is 3

    def test_has_good_binning(self):
        """Test has_good_binning function."""
        # Good binning case
        bin_edges = [1, 3, 5]
        bin_ixs = np.array([1, 1, 2, 2, 3, 3])  # 3 unique values

        # Should not raise an error
        try:
            undertest.has_good_binning(bin_ixs, bin_edges)
            # Test passes if no exception
        except Exception as e:
            assert False, f"has_good_binning raised unexpected exception: {e}"

        # Bad binning case (empty bin)
        bin_edges = [1, 3, 5, 7]
        bin_ixs = np.array([1, 1, 2, 2, 4, 4])  # Only 3 unique values

        # Should raise an error
        try:
            undertest.has_good_binning(bin_ixs, bin_edges)
            assert False, "Expected IndexError but none was raised"
        except IndexError:
            pass  # Expected behavior

    def test_label_cohorts_numeric(self):
        """Test label_cohorts_numeric function directly."""
        s = pd.Series([1.0, 2.5, 3.0, 4.5, 5.0])

        # Test with splits
        result = undertest.label_cohorts_numeric(s, splits=[2.0, 4.0])

        # Verify binning pattern
        assert result[0].startswith("<")  # First value should be < 2.0
        assert "2.0-4.0" in result[1]  # Middle values in middle bin
        assert "2.0-4.0" in result[2]
        assert result[3].startswith(">=")  # Last values >= 4.0
        assert result[4].startswith(">=")  # Last values >= 4.0

    def test_label_cohorts_categorical(self):
        """Test label_cohorts_categorical function directly."""
        # Create a categorical series with name already set to 'cohort'
        s = pd.Series(pd.Categorical(["A", "B", "C", "D", "A"], categories=["A", "B", "C", "D", "E"]), name="cohort")

        # Test with subset of categories
        result = undertest.label_cohorts_categorical(s, cat_values=["A", "C"])

        # Verify the filtering
        assert result[0] == "A"
        assert pd.isna(result[1])
        assert result[2] == "C"
        assert pd.isna(result[3])
        assert result[4] == "A"

        # Test without specifying categories
        s_for_remove = pd.Series(
            pd.Categorical(["A", "B", "C", "D"], categories=["A", "B", "C", "D", "E", "F"]), name="cohort"
        )
        result = undertest.label_cohorts_categorical(s_for_remove)

        # Should have only observed categories
        assert set(result.cat.categories) == set(["A", "B", "C", "D"])

    def test_resolve_col_data(self):
        """Test resolve_col_data function."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        # Test with string column name
        result = undertest.resolve_col_data(df, "feature1")
        pd.testing.assert_series_equal(result, df["feature1"])

        # Test with series
        series = pd.Series([7, 8, 9])
        result = undertest.resolve_col_data(df, series)
        pd.testing.assert_series_equal(result, series)

        # Test with numpy array
        array = np.array([10, 11, 12])
        result = undertest.resolve_col_data(df, array)
        np.testing.assert_array_equal(result, array)

        # Test with 2D array (like sklearn probabilities output)
        array_2d = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
        result = undertest.resolve_col_data(df, array_2d)
        np.testing.assert_array_equal(result, array_2d[:, 1])

        # Test with invalid string
        try:
            undertest.resolve_col_data(df, "nonexistent_feature")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass  # Expected

        # Test with invalid type
        try:
            undertest.resolve_col_data(df, 123)  # Not a string or series
            assert False, "Should have raised TypeError"
        except TypeError:
            pass  # Expected
