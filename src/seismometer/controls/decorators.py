# -*- coding: utf-8 -*-
"""
As for all of the utils subpackage, no other custom sub-packages should be referenced.
"""
import hashlib
import logging
import os
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


disk_cached_html_segment = DiskCachedFunction("html", save_fn=html_save, load_fn=html_load, return_type=HTML)
