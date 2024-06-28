import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from pytest import fixture

TEST_ROOT = Path(__file__).parent


@fixture
def res():
    return TEST_ROOT / "resources"


@fixture
def tmp_as_current(tmp_path):
    with working_dir_as(tmp_path):
        yield tmp_path


@contextmanager
def working_dir_as(path: Path) -> Generator:
    """
    Temporarily changes the current working directory
    Useful for testing when the model root is assumed

    Parameters
    ----------
    path : Path
        Directory to treat as working directory
    """
    oldpath = Path().absolute()

    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(oldpath)
