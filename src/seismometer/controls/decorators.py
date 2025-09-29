# -*- coding: utf-8 -*-
"""
As for all of the utils subpackage, no other custom sub-packages should be referenced.
"""
import hashlib
import logging
import os
import pickle
import shutil
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any, Callable

from IPython.display import HTML
from ipywidgets import Widget

from seismometer.core.decorators import DiskCachedFunction

logger = logging.getLogger("seismometer")


def html_load(filepath) -> HTML:
    return HTML(filepath.read_text())


def html_save(html, filepath) -> None:
    filepath.write_text(html.data)


def html_and_df_save(data, filepath) -> None:
    """
    Saves a tuple of (HTML, pd.DataFrame) to disk.
    """
    html, df = data
    html_path = filepath.with_suffix(".html")
    df_path = filepath.with_suffix(".pkl")
    html_path.write_text(html.data)
    with open(df_path, "wb") as f:
        pickle.dump(df, f)


def html_and_df_load(filepath) -> tuple[HTML, Any]:
    """
    Loads a tuple of (HTML, pd.DataFrame) from disk.
    """
    html_path = filepath.with_suffix(".html")
    df_path = filepath.with_suffix(".pkl")
    html = HTML(html_path.read_text())
    with open(df_path, "rb") as f:
        df = pickle.load(f)
    return html, df


disk_cached_html_segment = DiskCachedFunction("html", save_fn=html_save, load_fn=html_load, return_type=HTML)
disk_cached_html_and_df_segment = DiskCachedFunction("html_and_df", save_fn=html_and_df_save, load_fn=html_and_df_load)
