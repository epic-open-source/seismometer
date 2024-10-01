import hashlib
import json
import logging
import os
import re
import unicodedata
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import nbformat
import yaml

Pathlike = str | Path
logger = logging.getLogger("seismometer")

# region io-accessor functions
read_ipynb = partial(nbformat.read, as_version=nbformat.current_nbformat)
dump_json_pretty = partial(json.dump, indent=4)
dump_yaml = yaml.dump


def _read_text(fo):
    return fo.readlines()


def _print_to_file(content, fo):
    print(content, file=fo)


# endregion


def slugify(value: str) -> str:
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    and modified to fit our use. Convert spaces or repeated dashes to single dashes.
    Remove characters that aren't alphanumerics, underscores, or hyphens. Convert
    to lowercase. Also strip leading and trailing whitespace, dashes, and underscores.
    Converts multiple underscores in a row to a single underscore.
    Raises an exception if the resulting slug is empty.

    Parameters
    ----------
    value : str
        The string to slugify.

    Returns
    -------
    str
        The slugified string.

    Raises
    ------
    Exception
        Raises an exception if the resulting slug is empty.
    """
    old_value = value
    value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("<", "_lt_")
    value = value.replace(">", "_gt_")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    value = re.sub(r"[\s]+", "_", value)
    value = re.sub(r"_+", "_", value)
    if len(value) == 0:
        raise Exception(f"Invalid filename to slugify: {old_value}")
    if len(value) > 82:
        hash_str = hashlib.md5(value.encode()).hexdigest()
        value = value[:50] + hash_str
    return value


def resolve_filename(
    filename: Path,
    cohort_attribute: Optional[str] = None,
    subgroups: list[str] = None,
    basedir: str = "output",
    create=True,
) -> Path:
    """
    Construct filepath from filename and cohorts.

    Identifies a path <cohort_attribute>/<included subgroups>/filename to write a file.

    Parameters
    ----------
    filename : Path
        Leaf name of the file.
    cohort_attribute : Optional[str], optional
        Name of the column or cohort attribute, by default None.
    subgroups : list[str], optional
        List of subgroups included, by default None.
    basedir : str, optional
        The directory for all output files, by default "output".
    create : bool, optional
        Flag to indicate whether to preemptively create derived directory, by default True.

    Returns
    -------
    Path
        filepath to write
    """
    basedir = Path(basedir)
    if not cohort_attribute:
        return basedir / filename

    safe_groups = [slugify(g.replace(".", "_").replace(",", "-")) for g in sorted(subgroups) if g]
    subdir = "+".join(safe_groups)

    basedir = basedir / slugify(cohort_attribute) / subdir

    # Create preemptively
    if not basedir.is_dir():
        if not create:
            logger.warning(f"No directory found for group: {basedir}")
        else:
            basedir.mkdir(parents=True, exist_ok=True)
    return basedir / filename


# region load functions


def load_notebook(filepath: Pathlike) -> nbformat.NotebookNode:
    """
    Loads a notebook from a file.

    Parameters
    ----------
    filepath : Pathlike
        The path to a notebook.

    Returns
    -------
    nbformat.NotebookNode
        The loaded notebook.
    """
    return _load(read_ipynb, filepath)


def load_markdown(filepath: Pathlike, resource_dir: Path = None) -> str:
    """
    Loads a markdown file from a path.

    Parameters
    ----------
    filepath : Pathlike
        The path to the markdown file.
    resource_dir : Optional[Path], optional
        A parent directory to the file, by default None.

    Returns
    -------
    str
        The content of the markdown file.

    """
    return _load(_read_text, filepath, resource_dir)


def load_json(filepath: Pathlike, resource_dir: Path = None) -> dict:
    """
    Loads a JSON file from a path.

    Parameters
    ----------
    filepath : Pathlike
        The path to the JSON file.
    resource_dir : Optional[Path], optional
        A parent directory to the file, by default None.

    Returns
    -------
    dict
        The content of the JSON file.

    """
    return _load(json.load, filepath, resource_dir)


def load_yaml(filepath: Pathlike, resource_dir: Path = None) -> dict:
    """
    Loads a YAML file from a path.

    Parameters
    ----------
    filepath : Pathlike
        The path to the YAML file.
    resource_dir : Optional[Path], optional
        A parent directory to the file, by default None.

    Returns
    -------
    dict
        The content of the YAML file.

    """
    return _load(yaml.safe_load, filepath, resource_dir)


def _load(loader: Callable["fileobject", Any], filepath: Path, resource_dir: Path = None) -> Any:
    """
    Loads a file using the specified loader function.

    Generalizes loading a file to have consistent handling on existence and nonexistence.

    Parameters
    ----------
    loader : Callable[fileobject, Any]
        The function that can load a file and takes a single file-object argument.
    filepath : Path
        The path to the file.
    resource_dir : Optional[Path], optional
        A parent directory to the file, by default None.

    Returns
    -------
    Any
        The contents of the file in the format the loader emits.

    """
    full_path = Path(filepath)
    if resource_dir is not None:
        full_path = resource_dir / filepath

    if not (full_path).is_file():
        raise FileNotFoundError(f"Config file not found: {full_path.resolve()}")

    with open(full_path, "r") as fo:
        return loader(fo)


# endregion
# region write functions
def write_yaml(content: dict, filepath: Path, *, overwrite: bool = False) -> None:
    """
    Writes a dictionary to a YAML file.

    Parameters
    ----------
    content : dict
        The dictionary to write.
    filepath : Path
        The location to write.
    overwrite : bool, optional
        Write the file even if one exists, by default False.
    """
    _write(dump_yaml, content, filepath, overwrite)


def write_markdown(content: str, filepath: Path, *, overwrite: bool = False) -> None:
    """
    Writes a string to file as markdown.

    Parameters
    ----------
    content : str
        The content to write.
    filepath : Path
        The location to write.
    overwrite : bool, optional
        Write the file even if one exists, by default False.
    """
    _write(_print_to_file, content, filepath, overwrite)


def write_notebook(notebook: Any, filepath: Path, *, overwrite: bool = False) -> None:
    """
    Writes a notebook to a file.

    Parameters
    ----------
    notebook : Any
        The notebook to write.
    filepath : Path
        The location to write.
    overwrite : bool, optional
        If True, writes the file even if one exists, by default False.
    """
    _write(nbformat.write, notebook, filepath, overwrite)


def write_json(content: Any, filepath: Path, *, overwrite: bool = False) -> None:
    """
    Writes a dictionary to a JSON file.

    Parameters
    ----------
    content : Any
        The dictionary to write.
    filepath : Path
        The location to write.
    overwrite : bool, optional
        If True, writes the file even if one exists, by default False.
    """
    _write(dump_json_pretty, content, filepath, overwrite)


def _write(writer: Callable[[Any, "fileobject"], None], content: Any, file: Path, overwrite: bool) -> None:
    """
    Writes to file using the specified writer function.

    Generalizes writing of files to have consistent handling on existence and nonexistence of file and parent
    directories.

    Parameters
    ----------
    writer : Callable[[Any, fileobject], None]
        The function that can write to a file and takes two arguments, content and a file-object.
    content : Any
        The content to write.
    file : Path
        The location to write.
    overwrite : bool
        If True, writes the file even if one exists.

    """
    file = Path(file)
    if file.is_file() and not overwrite:
        raise FileExistsError(f"File already exists: {file.resolve()}; Specify overwrite to write anyway.")
    if not file.parent.exists():
        os.makedirs(file.parent)

    with open(file, "w") as fo:
        writer(content, fo)
    logger.info(f"File written: {file.resolve()}")
