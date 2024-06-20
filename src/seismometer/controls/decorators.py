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

logger = logging.getLogger("seismometer")
SEISMOMETER_CACHE_DIR = Path("./.seismometer_cache")
SEISMOMETER_CACHE_ENABLED = os.getenv("SEISMOMETER_CACHE_ENABLED", "") != ""


def disk_cached_html_segment(func: Callable[..., HTML]) -> HTML:
    """
    Caching decorator for HTML data

    Parameters
    ----------
    func : Callable[..., HTML]
        function that generates an HTML component

    Returns
    -------
    HTML
        displayable HTML component

    Raises
    ------
    TypeError
        If called with a function that does not annotate its return type as HTML, or that returns a non-HTML object.
    """
    sig = signature(func)
    if sig.return_annotation != HTML:
        raise TypeError("The function must return {HTML} object.")

    if not SEISMOMETER_CACHE_ENABLED:
        return func

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        arg_str = str((args, frozenset(kwargs.items())))
        hash_object = hashlib.md5(arg_str.encode())
        arg_hash = hash_object.hexdigest()

        file_dir = Path("./.seismometer_cache") / "html" / func.__name__

        filepath = file_dir / f"{arg_hash}.html"
        if filepath and filepath.is_file():
            logger.debug(f"From cache: {filepath}")
            return HTML(filepath.read_text())
        else:
            result = func(*args, **kwargs)
            if not isinstance(result, HTML):
                raise TypeError("The function must return {HTML} object.")
            file_dir.mkdir(parents=True, exist_ok=True)
            filepath.write_text(result.data)
        return result

    return wrapped_func


def _clear_cache_dir() -> None:
    shutil.rmtree(SEISMOMETER_CACHE_DIR / "html", ignore_errors=True)
    logger.debug(f"Cache cleared: {SEISMOMETER_CACHE_DIR}")


disk_cached_html_segment.clear_cache = _clear_cache_dir


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
