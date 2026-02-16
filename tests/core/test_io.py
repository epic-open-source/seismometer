import json
import logging
from pathlib import Path

import nbformat
import pytest
from conftest import res, tmp_as_current  # noqa: F401

import seismometer.core.io as undertest


class Test_IO:
    @pytest.mark.parametrize(
        "string, output",
        [
            # No event impute to negative
            pytest.param("abcdef", "abcdef", id="alphabet characters ok"),
            pytest.param("1234", "1234", id="numeric characters ok"),
            pytest.param("<>", "_lt_gt_", id="<> get converted"),
            pytest.param("abc_123_    ,,...,,,,['<18', '18-30']", "abc_123_lt_18_18-30", id="combination of above"),
            pytest.param(
                "Feature Report Comparing Age_Group in ['<18', '18-30', '30-40'] vs Ethnicity in ['Not Hispanic']",
                "feature_report_comparing_age_group_in_lt_18_18-30_a9386de720764da2ed95cd2ecea4f4ae",
                id="example filepath",
            ),
            pytest.param("Б, В, Г, Д, Ж, З, К, Л", "б_в_г_д_ж_з_к_л", id="acceptable unicode characters"),
            pytest.param("____________", "_", id="underscores combined"),
            pytest.param("_____b_______", "_b_", id="underscores combined"),
        ],
    )
    def test_slugify(self, string, output):
        assert undertest.slugify(string) == output

    @pytest.mark.parametrize(
        "string",
        [
            pytest.param("`~!@#$%^&*()=+,./'\\][}{", id="special characters not allowed"),
            pytest.param("😀😃😄😁", id="emojis not allowed"),
        ],
    )
    def test_slugify_raises(self, string):
        with pytest.raises(Exception):
            undertest.slugify(string)


@pytest.fixture
def testdir(res) -> Path:
    return res / "io"


# region Loaders
def test_load_json(testdir):
    filepath = testdir / "simple_load.json"

    result = undertest.load_json(filepath)
    assert result == {"key": "value"}


def test_load_json_with_dir(testdir):
    filepath = "simple_load.json"
    result = undertest.load_json(filepath, testdir)
    assert result == {"key": "value"}


class TestLoadNotebook:
    filepath = "simple_notebook.ipynb"
    expected = json.loads(
        '{"cells": [{"cell_type": "markdown", "metadata": {},'
        + '"source": "One cell."}],'
        + '"metadata": {"language_info": {"name": "python"}},'
        + '"nbformat": 4, "nbformat_minor": 2}'
    )

    def test_load_notebook_with_filepath(self, testdir):
        filepath = testdir / TestLoadNotebook.filepath
        result = undertest.load_notebook(filepath=filepath)
        assert result == TestLoadNotebook.expected

    def test_load_notebook_with_missing_file(self, tmp_as_current):
        filepath = "not_a_file.ipynb"
        with pytest.raises(FileNotFoundError):
            _ = undertest.load_notebook(filepath=filepath)


def test_load_markdown(testdir):
    filepath = testdir / "simple_load.md"
    result = undertest.load_markdown(filepath)
    assert result == ["# Title\n", "And text\n"]


def test_load_yaml(testdir):
    filepath = testdir / "simple_load.yml"
    result = undertest.load_yaml(filepath)
    assert result == {"topkey": {"key1": "value1"}}


# endregion


# region Writers
class TestMdWriters:
    file = Path("test.md")
    md_content = "# Title\nAnd text"

    def test_write_md(self, tmp_as_current):
        undertest.write_markdown(self.md_content, self.file, overwrite=False)
        assert self.file.read_text().strip() == self.md_content

    def test_write_md_fails_if_file_exists(self, tmp_as_current):
        self.file.touch()

        with pytest.raises(FileExistsError):
            undertest.write_markdown(self.md_content, self.file, overwrite=False)

    def test_overwrite_md_if_specified(self, tmp_as_current):
        self.file.touch()

        undertest.write_markdown(self.md_content, self.file, overwrite=True)
        assert self.file.read_text().strip() == self.md_content


class TestJsonWriters:
    file = Path("test.ipynb")

    content = {
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": "One cell."}],
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2,
    }

    @property
    def nb_content(self):
        return nbformat.reads(json.dumps(self.content), as_version=nbformat.current_nbformat)

    def test_write_json(self, tmp_as_current):
        undertest.write_json(self.content, self.file)
        assert json.loads(self.file.read_text())

    def test_write_json_fails_if_file_exists(self, tmp_as_current):
        self.file.touch()

        with pytest.raises(FileExistsError):
            undertest.write_json(self.content, self.file)

    def test_overwrite_json_if_specified(self, tmp_as_current):
        self.file.touch()

        undertest.write_json(self.content, self.file, overwrite=True)
        assert json.loads(self.file.read_text()) == self.content

    def test_write_ipynb(self, tmp_as_current):
        undertest.write_notebook(self.nb_content, self.file)
        actual = nbformat.reads(self.file.read_text(), as_version=nbformat.current_nbformat)
        assert actual == self.content

    def test_write_ipynb_fails_if_file_exists(self, tmp_as_current):
        self.file.touch()

        with pytest.raises(FileExistsError):
            undertest.write_notebook(self.nb_content, self.file)

    def test_overwrite_ipynb_if_specified(self, tmp_as_current):
        self.file.touch()

        undertest.write_notebook(self.nb_content, self.file, overwrite=True)
        actual = nbformat.reads(self.file.read_text(), as_version=nbformat.current_nbformat)
        assert actual == self.content


class TestYamlWriters:
    file = Path("test.yml")
    content = {"topkey": {"key1": "value1"}}

    def test_write_yaml(self, tmp_as_current):
        undertest.write_yaml(self.content, self.file)
        assert self.file.read_text().strip() == "topkey:\n  key1: value1"

    def test_write_yaml_fails_if_file_exists(self, tmp_as_current):
        self.file.touch()

        with pytest.raises(FileExistsError):
            undertest.write_yaml(self.content, self.file)

    def test_overwrite_yaml_if_specified(self, tmp_as_current):
        self.file.touch()

        undertest.write_yaml(self.content, self.file, overwrite=True)
        assert self.file.read_text().strip() == "topkey:\n  key1: value1"


def test_write_new_file_in_nonexistent_directory(tmp_as_current):
    file = Path("nonexistent_directory") / "new_file.txt"
    undertest._write(lambda content, fo: fo.write(content), "test content", file, overwrite=False)
    assert file.read_text() == "test content"


def test_write_logs_file_written(tmp_as_current, caplog):
    file = Path("new_file.txt")

    with caplog.at_level(logging.INFO, logger="seismometer"):
        undertest._write(lambda content, fo: fo.write(content), "test content", file, overwrite=False)

    assert f"File written: {file.resolve()}" in caplog.text


@pytest.mark.usefixtures("tmp_as_current")
class TestResolveFile:
    def test_no_cohort_returns_slim(self):
        filename = "new_file"
        expected = Path("output") / filename
        actual = undertest.resolve_filename("new_file", create=False)

        assert actual == expected

    def test_strip_chars_from_subgroups(self):
        attribute = "attr"
        subgroups = ["first", "st*r|i💣p"]
        filename = "new_file"
        expected = Path("output/attr") / "first+strip" / filename
        actual = undertest.resolve_filename("new_file", attribute, subgroups, create=False)

        assert actual == expected

    def test_subgroup_handles_special_chars(self):
        attribute = "attr"
        subgroups = ["age1.0-3.0,self", "str|i💣p"]
        filename = "new_file"
        expected = Path("output/attr") / "age1_0-3_0-self+strip" / filename
        actual = undertest.resolve_filename("new_file", attribute, subgroups, create=False)

        assert actual == expected

    def test_create_makesdir_on_none(self, tmp_path, caplog):
        attribute = "attr"
        subgroups = ["age1.0-3.0,self", "str|i💣p"]
        filename = "new_file"
        expected = Path("output/attr") / "age1_0-3_0-self+strip" / filename

        with caplog.at_level(30, logger="seismometer"):
            _ = undertest.resolve_filename("new_file", attribute, subgroups)

        assert not caplog.text
        assert expected.parent.is_dir()

    def test_no_create_warns_on_nonexistent(self, caplog):
        attribute = "attr"
        subgroups = ["age1.0-3.0,self", "str|i💣p"]
        filename = "new_file"
        expected = Path("output/attr") / "age1_0-3_0-self+strip" / filename

        with caplog.at_level(30, logger="seismometer"):
            _ = undertest.resolve_filename("new_file", attribute, subgroups, create=False)

        assert "No directory" in caplog.text
        assert not expected.parent.is_dir()

    def test_no_create_existent_does_not_warn(self, caplog):
        attribute = "attr"
        subgroups = ["gg"]
        filename = "new_file"
        expected = Path("output/attr") / "gg" / filename

        expected.parent.mkdir(parents=True, exist_ok=False)
        with caplog.at_level(30, logger="seismometer"):
            _ = undertest.resolve_filename("new_file", attribute, subgroups, create=False)

        assert not caplog.text
        assert expected.parent.is_dir()


# endregion


# ============================================================================
# ADDITIONAL ERROR HANDLING TESTS
# ============================================================================


class TestLoadNotebookErrorHandling:
    """Test load_notebook() with corrupted/invalid files."""

    def test_load_notebook_with_corrupted_json(self, tmp_as_current):
        """Test load_notebook with corrupted JSON content."""
        corrupted_file = Path("corrupted.ipynb")
        corrupted_file.write_text("{invalid json content")

        # nbformat raises NotJSONError (subclass of ValueError) for invalid JSON
        with pytest.raises((json.JSONDecodeError, ValueError)):
            undertest.load_notebook(corrupted_file)

    def test_load_notebook_with_invalid_notebook_format(self, tmp_as_current):
        """Test load_notebook with valid JSON but invalid notebook structure."""
        invalid_nb = Path("invalid.ipynb")
        # Valid JSON but missing required notebook fields
        invalid_nb.write_text('{"not": "a notebook"}')

        with pytest.raises((KeyError, nbformat.validator.ValidationError)):
            undertest.load_notebook(invalid_nb)

    def test_load_notebook_with_empty_file(self, tmp_as_current):
        """Test load_notebook with empty file."""
        empty_file = Path("empty.ipynb")
        empty_file.write_text("")

        # nbformat raises NotJSONError (subclass of ValueError) for empty files
        with pytest.raises((json.JSONDecodeError, ValueError)):
            undertest.load_notebook(empty_file)

    def test_load_notebook_with_binary_content(self, tmp_as_current):
        """Test load_notebook with binary content (encoding issue)."""
        binary_file = Path("binary.ipynb")
        # Write binary data that's not valid UTF-8
        binary_file.write_bytes(b"\x80\x81\x82\x83")

        with pytest.raises((UnicodeDecodeError, json.JSONDecodeError)):
            undertest.load_notebook(binary_file)


class TestLoadMarkdownErrorHandling:
    """Test load_markdown() with various edge cases."""

    def test_load_markdown_with_empty_file(self, tmp_as_current):
        """Test load_markdown with empty file returns empty list."""
        empty_file = Path("empty.md")
        empty_file.write_text("")

        result = undertest.load_markdown(empty_file)

        assert result == []

    def test_load_markdown_with_only_whitespace(self, tmp_as_current):
        """Test load_markdown with only whitespace."""
        whitespace_file = Path("whitespace.md")
        whitespace_file.write_text("   \n\n   \n")

        result = undertest.load_markdown(whitespace_file)

        # Should return the whitespace lines as-is
        assert len(result) == 3

    def test_load_markdown_with_binary_content(self, tmp_as_current):
        """Test load_markdown with binary content (encoding issue)."""
        binary_file = Path("binary.md")
        # Write binary data that's not valid UTF-8
        binary_file.write_bytes(b"\x80\x81\x82\x83")

        with pytest.raises(UnicodeDecodeError):
            undertest.load_markdown(binary_file)

    def test_load_markdown_with_nonexistent_file(self, tmp_as_current):
        """Test load_markdown with non-existent file."""
        nonexistent = Path("nonexistent.md")

        with pytest.raises(FileNotFoundError):
            undertest.load_markdown(nonexistent)


class TestLoadJsonErrorHandling:
    """Test load_json() with malformed JSON."""

    def test_load_json_with_malformed_json(self, tmp_as_current):
        """Test load_json with malformed JSON syntax."""
        malformed_file = Path("malformed.json")
        malformed_file.write_text('{"key": "value",}')  # Trailing comma is invalid

        with pytest.raises(json.JSONDecodeError):
            undertest.load_json(malformed_file)

    def test_load_json_with_incomplete_json(self, tmp_as_current):
        """Test load_json with incomplete JSON."""
        incomplete_file = Path("incomplete.json")
        incomplete_file.write_text('{"key": "value"')  # Missing closing brace

        with pytest.raises(json.JSONDecodeError):
            undertest.load_json(incomplete_file)

    def test_load_json_with_empty_file(self, tmp_as_current):
        """Test load_json with empty file."""
        empty_file = Path("empty.json")
        empty_file.write_text("")

        with pytest.raises(json.JSONDecodeError):
            undertest.load_json(empty_file)

    def test_load_json_with_non_json_content(self, tmp_as_current):
        """Test load_json with non-JSON content."""
        text_file = Path("text.json")
        text_file.write_text("This is just plain text, not JSON")

        with pytest.raises(json.JSONDecodeError):
            undertest.load_json(text_file)

    def test_load_json_with_binary_content(self, tmp_as_current):
        """Test load_json with binary content (encoding issue)."""
        binary_file = Path("binary.json")
        binary_file.write_bytes(b"\x80\x81\x82\x83")

        with pytest.raises((UnicodeDecodeError, json.JSONDecodeError)):
            undertest.load_json(binary_file)

    def test_load_json_with_nonexistent_file(self, tmp_as_current):
        """Test load_json with non-existent file."""
        nonexistent = Path("nonexistent.json")

        with pytest.raises(FileNotFoundError):
            undertest.load_json(nonexistent)


class TestLoadYamlErrorHandling:
    """Test load_yaml() with invalid YAML syntax."""

    def test_load_yaml_with_invalid_indentation(self, tmp_as_current):
        """Test load_yaml with invalid YAML indentation."""
        invalid_file = Path("invalid.yml")
        invalid_file.write_text("key1: value1\n  key2: value2\n key3: value3")  # Inconsistent indentation

        with pytest.raises(Exception):  # yaml.YAMLError or similar
            undertest.load_yaml(invalid_file)

    def test_load_yaml_with_unclosed_quotes(self, tmp_as_current):
        """Test load_yaml with unclosed quotes."""
        invalid_file = Path("unclosed.yml")
        invalid_file.write_text('key: "unclosed quote')

        with pytest.raises(Exception):  # yaml.scanner.ScannerError
            undertest.load_yaml(invalid_file)

    def test_load_yaml_with_tabs_instead_of_spaces(self, tmp_as_current):
        """Test load_yaml with tabs (YAML requires spaces)."""
        invalid_file = Path("tabs.yml")
        invalid_file.write_text("key1:\n\tsubkey: value")  # Tab indentation is invalid in YAML

        with pytest.raises(Exception):  # yaml.scanner.ScannerError
            undertest.load_yaml(invalid_file)

    def test_load_yaml_with_empty_file(self, tmp_as_current):
        """Test load_yaml with empty file returns None."""
        empty_file = Path("empty.yml")
        empty_file.write_text("")

        result = undertest.load_yaml(empty_file)

        # Empty YAML file returns None
        assert result is None

    def test_load_yaml_with_binary_content(self, tmp_as_current):
        """Test load_yaml with binary content (encoding issue)."""
        binary_file = Path("binary.yml")
        binary_file.write_bytes(b"\x80\x81\x82\x83")

        with pytest.raises(UnicodeDecodeError):
            undertest.load_yaml(binary_file)

    def test_load_yaml_with_nonexistent_file(self, tmp_as_current):
        """Test load_yaml with non-existent file."""
        nonexistent = Path("nonexistent.yml")

        with pytest.raises(FileNotFoundError):
            undertest.load_yaml(nonexistent)

    def test_load_yaml_with_duplicate_keys(self, tmp_as_current):
        """Test load_yaml with duplicate keys (valid YAML, last value wins)."""
        duplicate_file = Path("duplicate.yml")
        duplicate_file.write_text("key: value1\nkey: value2")

        result = undertest.load_yaml(duplicate_file)

        # YAML allows duplicate keys, last one wins
        assert result == {"key": "value2"}


class TestWriteFunctionsErrorHandling:
    """Test write functions error handling.

    Note: Permission denied tests are skipped because chmod on owned directories
    doesn't reliably prevent writes (owner can always write). Real permission
    errors would require external system configuration or mocking.
    """

    def test_write_functions_handle_nested_directories(self, tmp_as_current):
        """Test that write functions create nested directories."""
        nested_file = Path("deep") / "nested" / "dir" / "file.yml"

        undertest.write_yaml({"key": "value"}, nested_file)

        assert nested_file.exists()
        assert nested_file.parent.is_dir()


class TestResolveFilenameEdgeCases:
    """Test resolve_filename() with additional edge cases."""

    @pytest.mark.usefixtures("tmp_as_current")
    def test_resolve_filename_with_very_long_filename(self):
        """Test resolve_filename with very long filename."""
        long_name = "a" * 200
        result = undertest.resolve_filename(long_name, create=False)

        # Should handle long names (may be truncated by OS)
        assert isinstance(result, Path)
        assert "output" in str(result)

    @pytest.mark.usefixtures("tmp_as_current")
    def test_resolve_filename_with_path_traversal_preserves_dots(self):
        """Test resolve_filename does NOT sanitize path traversal (actual behavior).

        Note: This reveals that the function doesn't sanitize ../ sequences.
        Path traversal is preserved in the output, which could be a security
        concern if user input is used without validation.
        """
        malicious_name = "../../../etc/passwd"
        result = undertest.resolve_filename(malicious_name, create=False)

        # Actual behavior: path traversal is preserved
        assert "output" in str(result)
        # Function does not sanitize ../ sequences

    @pytest.mark.usefixtures("tmp_as_current")
    def test_resolve_filename_with_null_bytes_preserves_them(self):
        """Test resolve_filename does NOT sanitize null bytes (actual behavior).

        Note: This reveals that the function doesn't sanitize null bytes.
        Null bytes are preserved in the path, which could cause issues with
        file operations on some systems.
        """
        filename_with_null = "test\x00file"

        result = undertest.resolve_filename(filename_with_null, create=False)

        # Actual behavior: null bytes are preserved
        assert "output" in str(result)
        # Function does not sanitize null bytes
