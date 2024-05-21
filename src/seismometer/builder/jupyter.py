from typing import Any, Iterator


def contrib_cells(notebook: dict, cell_type: list[str] = None) -> Iterator[dict]:
    """
    Iterator for finding contributor cells in a notebook template.

    Searching Jupyter-style notebook objects for cells that are tagged with "sg-contrib" in the metadata.
    Template notebooks with this key are also expected to have sg-id keys for identification.


    Parameters
    ----------
    notebook : dict
        A dictionary representation of a Jupyter notebook.
    cell_type : Optional[list[str]], optional
        An allow-list of values for cell_type, by default None; it allows markdown, code, raw, or heading.

    Yields
    ------
    Iterator[dict]
        A notebook cell dictionary.
    """
    cell_type = cell_type or ["markdown", "code", "raw", "heading"]

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") not in cell_type:
            continue
        if "sg-contrib" in (cell.get("metadata", {}).get("tags", [])):
            yield cell


def get_id(cell: dict, key: str = "sg-id") -> Any:
    """
    Extract the sg-id from a cell's metadata.

    Parameters
    ----------
    cell : dict
        A Jupyter notebook cell.
    key : str, optional
        An override for the metadata key to extract, by default 'sg-id'.

    Returns
    -------
    Any
        The value associated with the cell's key.
    """
    return cell.get("metadata", {}).get(key)


def get_text(cell: dict, strip_highlight=False) -> Any:
    """
    Extract the source content from a cell.

    For the expected markdown cells, returns the markdown content as a string.

    Parameters
    ----------
    cell : dict
        A Jupyter notebook cell.
    strip_highlight : bool, optional
        A flag to strip visual indicators used in templates, by default False.
        NOTE: If True, removes </span> without verifying if it's a highlight.

    Returns
    -------
    Any
        Source content of the cell.
    """
    raw_text = cell.get("source")

    if strip_highlight:  # exact match removals
        for hl_str in ["✨"]:
            raw_text = raw_text.replace(hl_str, "")

    return raw_text
