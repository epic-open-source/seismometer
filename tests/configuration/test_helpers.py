from unittest.mock import patch

import pandas as pd
import pytest

import seismometer.configuration.config_helpers as undertest
from seismometer.core.io import load_yaml


def frame_case():
    df = pd.DataFrame({"Type": ["A", "B", "C", "B"], "IntCol": [1, 2, 3, 4]})
    df.Type = df.Type.astype("category")
    df.IntCol = df.IntCol.astype("int32")

    return df


@pytest.mark.usefixtures("tmp_as_current")
@patch.object(pd, "read_parquet", return_value=frame_case())
class TestGenerateDict:
    def test_generate_events(self, mock_read):
        outfile = "out.yml"
        expected = {
            "events": [
                {
                    "definition": "Placeholder description for A",
                    "display_name": "A",
                    "dtype": "string",
                    "name": "A",
                },
                {
                    "definition": "Placeholder description for B",
                    "display_name": "B",
                    "dtype": "string",
                    "name": "B",
                },
                {
                    "definition": "Placeholder description for C",
                    "display_name": "C",
                    "dtype": "string",
                    "name": "C",
                },
            ]
        }

        undertest.generate_dictionary_from_parquet("TESTIN", outfile, section="events")
        actual = load_yaml(outfile)

        mock_read.assert_called_once_with("TESTIN")
        assert actual == expected

    def test_generate_predictions(self, mock_read):
        outfile = "out.yml"
        expected = {
            "predictions": [
                {
                    "definition": "Placeholder description for Type",
                    "display_name": "Type",
                    "dtype": "category",
                    "name": "Type",
                },
                {
                    "definition": "Placeholder description for IntCol",
                    "display_name": "IntCol",
                    "dtype": "int32",
                    "name": "IntCol",
                },
            ]
        }

        undertest.generate_dictionary_from_parquet("TESTIN", outfile, section="predictions")
        actual = load_yaml(outfile)

        mock_read.assert_called_once_with("TESTIN")
        assert actual == expected

    def test_generate_dict_invalid_section(self, mock_read):
        section_type = "invalid"
        with pytest.raises(ValueError, match="not recognized") as err:
            undertest.generate_dictionary_from_parquet("TESTIN", "out.yml", section=section_type)

        assert section_type in str(err.value)


# This is intentionally outside the class to NOT use the shared pd patch
@pytest.mark.parametrize("section_type", ["events", "predictions"])
@patch.object(undertest.pd, "read_parquet", return_value=pd.DataFrame())
def test_generate_dict_no_data_raises_error(mock_read, section_type, tmp_as_current):
    with pytest.raises(ValueError, match="No data loaded"):
        undertest.generate_dictionary_from_parquet("TESTIN", "out.yml", section=section_type)


# ============================================================================
# ADDITIONAL EDGE CASE TESTS
# ============================================================================


class TestGenerateEventDictionaryWithMissingColumn:
    """Test _generate_event_dictionary with missing column."""

    def test_missing_column_returns_empty_events_list(self):
        """Test that missing column results in empty events list."""
        df = pd.DataFrame({"ActualColumn": ["A", "B", "C"]})

        result = undertest._generate_event_dictionary(df, column="NonExistentColumn")

        assert isinstance(result, undertest.EventDictionary)
        assert result.events == []

    def test_missing_column_with_empty_dataframe(self):
        """Test missing column with empty DataFrame."""
        df = pd.DataFrame()

        result = undertest._generate_event_dictionary(df, column="MissingColumn")

        assert isinstance(result, undertest.EventDictionary)
        assert result.events == []

    def test_column_exists_but_empty(self):
        """Test column exists but has no data."""
        df = pd.DataFrame({"Type": []})

        result = undertest._generate_event_dictionary(df, column="Type")

        assert isinstance(result, undertest.EventDictionary)
        assert result.events == []


class TestGeneratePredictionDictionaryWithRealDtypes:
    """Test _generate_prediction_dictionary with real DataFrame dtypes (not mocked)."""

    def test_with_numeric_dtypes(self):
        """Test with int, float, and uint dtypes."""
        df = pd.DataFrame(
            {
                "int_col": pd.array([1, 2, 3], dtype="int64"),
                "float_col": pd.array([1.1, 2.2, 3.3], dtype="float64"),
                "uint_col": pd.array([1, 2, 3], dtype="uint32"),
            }
        )

        result = undertest._generate_prediction_dictionary(df)

        assert len(result.predictions) == 3
        assert result.predictions[0].name == "int_col"
        assert result.predictions[0].dtype == "int64"
        assert result.predictions[1].dtype == "float64"
        assert result.predictions[2].dtype == "uint32"

    def test_with_string_and_category_dtypes(self):
        """Test with string and category dtypes."""
        df = pd.DataFrame(
            {
                "str_col": pd.array(["a", "b", "c"], dtype="string"),
                "cat_col": pd.Categorical(["x", "y", "z"]),
            }
        )

        result = undertest._generate_prediction_dictionary(df)

        assert len(result.predictions) == 2
        assert "string" in result.predictions[0].dtype
        assert "category" in result.predictions[1].dtype

    def test_with_datetime_dtype(self):
        """Test with datetime dtype."""
        df = pd.DataFrame(
            {
                "date_col": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            }
        )

        result = undertest._generate_prediction_dictionary(df)

        assert len(result.predictions) == 1
        assert "datetime64" in result.predictions[0].dtype

    def test_with_boolean_dtype(self):
        """Test with boolean dtype."""
        df = pd.DataFrame(
            {
                "bool_col": [True, False, True],
            }
        )

        result = undertest._generate_prediction_dictionary(df)

        assert len(result.predictions) == 1
        assert "bool" in result.predictions[0].dtype

    def test_with_mixed_dtypes(self):
        """Test with multiple different dtypes."""
        df = pd.DataFrame(
            {
                "int": [1, 2],
                "float": [1.5, 2.5],
                "str": ["a", "b"],
                "bool": [True, False],
                "cat": pd.Categorical(["x", "y"]),
            }
        )

        result = undertest._generate_prediction_dictionary(df)

        assert len(result.predictions) == 5
        # Each column should have its dtype correctly captured
        names = [p.name for p in result.predictions]
        assert set(names) == {"int", "float", "str", "bool", "cat"}


class TestEmptyColumnNameHandling:
    """Test handling of empty column names."""

    def test_predictions_with_empty_column_name(self):
        """Test _generate_prediction_dictionary with empty string column name."""
        df = pd.DataFrame({"": [1, 2, 3], "normal_col": [4, 5, 6]})

        result = undertest._generate_prediction_dictionary(df)

        assert len(result.predictions) == 2
        # Empty string should be captured as a column name
        names = [p.name for p in result.predictions]
        assert "" in names
        assert "normal_col" in names

    def test_events_with_empty_string_values(self):
        """Test _generate_event_dictionary with empty string in values."""
        df = pd.DataFrame({"Type": ["", "A", "B", ""]})

        result = undertest._generate_event_dictionary(df, column="Type")

        # Empty string should be included in unique values
        names = [e.name for e in result.events]
        assert "" in names
        assert "A" in names
        assert "B" in names


class TestNonExistentFilePath:
    """Test handling of non-existent file paths (not mocked)."""

    @pytest.mark.usefixtures("tmp_as_current")
    def test_nonexistent_parquet_file_raises_error(self):
        """Test that non-existent file raises FileNotFoundError."""
        nonexistent_path = "/nonexistent/path/to/file.parquet"

        with pytest.raises((FileNotFoundError, OSError)):
            undertest.generate_dictionary_from_parquet(nonexistent_path, "out.yml")

    @pytest.mark.usefixtures("tmp_as_current")
    def test_invalid_parquet_path_raises_error(self, tmp_path):
        """Test that invalid path raises appropriate error."""
        invalid_path = tmp_path / "nonexistent_dir" / "file.parquet"

        with pytest.raises((FileNotFoundError, OSError)):
            undertest.generate_dictionary_from_parquet(invalid_path, "out.yml")


class TestDataFrameWithNaNValues:
    """Test handling of DataFrames with NaN/null values."""

    def test_predictions_with_nan_values(self):
        """Test _generate_prediction_dictionary with NaN values in columns."""
        df = pd.DataFrame(
            {
                "col_with_nan": [1.0, float("nan"), 3.0, float("nan")],
                "col_no_nan": [1, 2, 3, 4],
            }
        )

        result = undertest._generate_prediction_dictionary(df)

        # Should create entries for all columns regardless of NaN
        assert len(result.predictions) == 2
        names = [p.name for p in result.predictions]
        assert "col_with_nan" in names
        assert "col_no_nan" in names

    def test_events_with_nan_values_raises_validation_error(self):
        """Test _generate_event_dictionary with NaN values raises ValidationError.

        NaN values in event type column are invalid and should be rejected.
        Pydantic validation for DictionaryItem.name requires a string.
        """
        df = pd.DataFrame({"Type": ["A", float("nan"), "B", "A", float("nan")]})

        # NaN cannot be used as event name (must be string)
        with pytest.raises(Exception):  # pydantic ValidationError
            undertest._generate_event_dictionary(df, column="Type")

    def test_events_with_only_valid_strings(self):
        """Test _generate_event_dictionary with only valid string values."""
        df = pd.DataFrame({"Type": ["A", "B", "C", "A"]})

        result = undertest._generate_event_dictionary(df, column="Type")

        assert len(result.events) == 3  # A, B, C (unique)
        names = [e.name for e in result.events]
        assert "A" in names
        assert "B" in names
        assert "C" in names

    def test_predictions_with_all_nan_column(self):
        """Test _generate_prediction_dictionary with all-NaN column."""
        df = pd.DataFrame(
            {
                "all_nan": [float("nan"), float("nan"), float("nan")],
                "normal": [1, 2, 3],
            }
        )

        result = undertest._generate_prediction_dictionary(df)

        # All-NaN column should still be included
        assert len(result.predictions) == 2
        names = [p.name for p in result.predictions]
        assert "all_nan" in names

    def test_events_with_none_values_raises_validation_error(self):
        """Test _generate_event_dictionary with None values raises ValidationError.

        None values in event type column are invalid and should be rejected.
        Pydantic validation for DictionaryItem.name requires a string.
        """
        df = pd.DataFrame({"Type": ["A", None, "B", "A", None]})

        # None cannot be used as event name (must be string)
        with pytest.raises(Exception):  # pydantic ValidationError
            undertest._generate_event_dictionary(df, column="Type")


class TestSpecialCharactersInColumnNames:
    """Test handling of special characters in column names."""

    def test_predictions_with_special_characters(self):
        """Test _generate_prediction_dictionary with special characters in column names."""
        df = pd.DataFrame(
            {
                "col-with-dash": [1, 2, 3],
                "col.with.dots": [4, 5, 6],
                "col with spaces": [7, 8, 9],
                "col$with$dollar": [10, 11, 12],
                "col@with@at": [13, 14, 15],
            }
        )

        result = undertest._generate_prediction_dictionary(df)

        assert len(result.predictions) == 5
        names = [p.name for p in result.predictions]
        assert "col-with-dash" in names
        assert "col.with.dots" in names
        assert "col with spaces" in names
        assert "col$with$dollar" in names
        assert "col@with@at" in names

    def test_predictions_with_unicode_characters(self):
        """Test _generate_prediction_dictionary with Unicode characters."""
        df = pd.DataFrame(
            {
                "col_with_émojis": [1, 2, 3],
                "col_with_中文": [4, 5, 6],
                "col_with_ñ": [7, 8, 9],
            }
        )

        result = undertest._generate_prediction_dictionary(df)

        assert len(result.predictions) == 3
        names = [p.name for p in result.predictions]
        assert "col_with_émojis" in names
        assert "col_with_中文" in names
        assert "col_with_ñ" in names

    def test_events_with_special_characters_in_values(self):
        """Test _generate_event_dictionary with special characters in values."""
        df = pd.DataFrame(
            {
                "Type": [
                    "event-with-dash",
                    "event.with.dots",
                    "event with spaces",
                    "event$special",
                    "event@sign",
                ]
            }
        )

        result = undertest._generate_event_dictionary(df, column="Type")

        assert len(result.events) == 5
        names = [e.name for e in result.events]
        assert "event-with-dash" in names
        assert "event.with.dots" in names
        assert "event with spaces" in names

    def test_predictions_with_newlines_and_tabs(self):
        """Test _generate_prediction_dictionary with newlines and tabs in column names."""
        df = pd.DataFrame(
            {
                "col\nwith\nnewline": [1, 2, 3],
                "col\twith\ttab": [4, 5, 6],
            }
        )

        result = undertest._generate_prediction_dictionary(df)

        assert len(result.predictions) == 2
        # Column names with special whitespace should be preserved
        names = [p.name for p in result.predictions]
        assert any("\n" in name for name in names)
        assert any("\t" in name for name in names)
