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
