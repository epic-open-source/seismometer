import os
import shutil

import pytest
from conftest import res, tmp_as_current, working_dir_as  # noqa: F401

from seismometer.builder import main_cli
from seismometer.core.io import load_notebook


@pytest.fixture
def resource_dir(res):
    return res / "builder"


def cp_build(src, dest):
    shutil.copytree(src, dest, dirs_exist_ok=True)


def test_extract_and_build_one_file(tmp_path, resource_dir):
    test_case = resource_dir / "config_one"
    expected = load_notebook(filepath=test_case / "expected_out.ipynb")

    cp_build(test_case / "inputs", tmp_path)
    os.makedirs(tmp_path / "working")

    with working_dir_as(tmp_path / "working"):
        with pytest.raises(SystemExit) as sysexit:
            main_cli("extract -c ../single_config.yml".split(" "))
        assert sysexit.value.code == 0

        with pytest.raises(SystemExit) as sysexit:
            main_cli("build -c ../single_config.yml".split(" "))
        assert sysexit.value.code == 0

    actual = load_notebook(filepath=tmp_path / "gen_classifier_bin.ipynb")
    # No files should be in "working" given config info_dir is ..
    assert len(os.listdir(tmp_path / "working")) == 0
    assert len(os.listdir(tmp_path)) == 15  # 14 files + 1 directory
    # Output is the same; will change with template
    assert actual == expected


def test_build_from_many(tmp_path, resource_dir):
    test_case = resource_dir / "build_split"
    expected = load_notebook(filepath=test_case / "expected_out.ipynb")

    cp_build(test_case / "inputs", tmp_path)

    with working_dir_as(tmp_path), pytest.raises(SystemExit) as sysexit:
        main_cli(["build"])

    # write is to info_dir => "more"
    actual = load_notebook(filepath=tmp_path / "more" / "gen_classifier_bin.ipynb")

    assert sysexit.value.code == 0

    # Output is the same; will change with template
    assert actual == expected


def test_build_insufficient_warns(tmp_path, resource_dir, caplog):
    test_case = resource_dir / "build_split"
    expected = load_notebook(filepath=test_case / "expected_no_td.ipynb")

    cp_build(test_case / "inputs", tmp_path)
    os.remove(tmp_path / "more" / "definitions.md")

    with working_dir_as(tmp_path), pytest.raises(SystemExit) as sysexit:
        main_cli(["build"])

    # write is to info_dir => "more"
    actual = load_notebook(filepath=tmp_path / "more" / "gen_classifier_bin.ipynb")

    assert sysexit.value.code == 0
    assert len(caplog.records) == 1
    assert "definitions" in caplog.records[0].message

    # Output is the same; will change with template
    assert actual == expected
