import json
import logging
from pathlib import Path

import nbformat
import pytest
from conftest import res, tmp_as_current  # noqa: F401

import seismometer.core.io as undertest
from seismometer.configuration.options import Option


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
                "feature_report_comparing_age_group_in_lt_18_18-30_30-40_vs_ethnicity_in_not_hispanic",
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
    return res / "builder"


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

    def test_load_notebook_with_invalid_input(self):
        with pytest.raises(ValueError):
            _ = undertest.load_notebook()

    def test_load_notebook_with_nb_template(self, testdir):
        file = testdir / TestLoadNotebook.filepath
        nb_template = Option(value=file, name="test")
        result = undertest.load_notebook(nb_template=nb_template)
        assert result == TestLoadNotebook.expected

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
