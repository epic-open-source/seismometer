# -*- coding: utf-8 -*-
"""
As for all of the utils subpackage, no other custom subpackages should be referenced.
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

SEISMOMETER_CACHE_DIR = Path(".seismometer_cache")
SEISMOMETER_CACHE_ENABLED = os.getenv("SEISMOMETER_CACHE_ENABLED", "") != ""


def html_load(filepath) -> HTML:
    return HTML(filepath.read_text())


def html_save(html, filepath) -> None:
    filepath.write_text(html.data)


disk_cached_html_segment = DiskCachedFunction("html", save_fn=html_save, load_fn=html_load, return_type=HTML)


def display_cached_widget(func: Callable[[], Widget | list[Widget]]) -> Callable[[], any]:
    """Decorator that allows display of a cached widget, so that you can
    create and link an ipywidget once, and re-use it in subsequent calls.

    Note display_cached_widget function calls are idempotent!
    Can only be applied to a no-arg function that returns a widget, or list of widgets.

    Returns
    -------
    Callable[..., any]
        A wrapper function that calls the decorated function.
    """
    widgets = {}

    @wraps(func)
    def wrapped_func():
        """
        If a widget is already stored, display it, if not generate the widget by calling the wrapped function.
        """
        if func.__name__ not in widgets:
            widgets[func.__name__] = func()

        res = widgets[func.__name__]
        if isinstance(res, Widget):
            display(res)
        elif isinstance(res, (tuple, list)):
            display(*res)

    return wrapped_func
