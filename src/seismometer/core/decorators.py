# -*- coding: utf-8 -*-
"""
As for all of the utils subpackage, no other custom sub-packages should be referenced.
"""
import builtins
import hashlib
import logging
import os
import shutil
import sys
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("seismometer")


def export(orig_fn):
    """
    This decorator completely passes through any function it is on.
    The difference is that it adds the function name to the module's __all__ list so that
    a wildcard import of the module will include it while being restricted to that list.

    This doc string should be overridden by the source function.
    """
    if orig_fn.__module__ is None:
        return orig_fn
    mod = sys.modules[orig_fn.__module__]

    all_ = mod.__dict__.setdefault("__all__", [])
    fn_name = orig_fn.__name__

    if fn_name not in all_:
        all_.append(fn_name)
        setattr(mod, "__all__", list(sorted(all_)))
    else:
        raise ImportError(f"Naming collision on export of {mod.__name__}.{fn_name}")

    return orig_fn


def indented_function(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that will add an indentation to any print() calls made inside the function.

    Returns
    -------
    Callable[..., Any]
        A wrapper function that calls the decorated function.
    """

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        old = builtins.print
        builtins.print = lambda x, *args, **kwargs: old("> ", x, *args, **kwargs)
        res = func(*args, **kwargs)
        builtins.print = old
        return res

    return wrapped_func


class DiskCachedFunction(object):
    SEISMOMETER_CACHE_DIR = Path(".seismometer_cache")
    SEISMOMETER_CACHE_ENABLED = os.getenv("SEISMOMETER_CACHE_ENABLED", "") != ""

    def __init__(
        self,
        cache_name: str,
        save_fn: Callable[[Any], None],
        load_fn: Callable[[Path], Any],
        return_type: tuple[type] = None,
    ) -> None:
        """
        Creates a new decorator that will store data to a cache folder on disk.

        Cache can be turned on using the SEISMOMETER_CACHE_ENABLED environment variable.

        Parameters
        ----------
        cache_name : str
            subdirectory of the cache folder where the new cache will be stored.
        save_fn : Callable[[Any], None]
            function that saves the cached type to disk.
        load_fun : Callable[[Path], Any]
            function that loads the cached type to disk
        return_type : Optional[type |  str], optional
            required return type to enable caching, by default None
        """
        self.cache_name = cache_name
        self.save_fn = save_fn
        self.load_fn = load_fn
        self.return_type = return_type

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Method that does the wrapping, treated as a callable to make it easy to use as a decorator.

        Parameters
        ----------
        func : Callable[..., Any]
            function to use when caching.

        Returns
        -------
        Callable[..., Any]
            wrapped function that caches the results.

        Raises
        ------
        ValueError
            If the annotated return type is not the same as the required return type, or
            if the runtime return type does not match the annotated type.
        """
        sig = signature(func)
        if self.return_type and (sig.return_annotation != self.return_type):
            raise TypeError(f"The function '{func.__name__}' must return '{self.return_type.__name__}' object.")

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if not self.SEISMOMETER_CACHE_ENABLED:
                # Skip any caching logic, pure pass through
                return func(*args, **kwargs)

            arg_hash = _hash_function_args(*args, **kwargs)
            file_dir = self.cache_dir / func.__name__
            filepath = file_dir / arg_hash
            if filepath and filepath.is_file():
                logger.debug(f"From cache: {filepath}")
                return self.load_fn(filepath)
            else:
                logger.debug(f"Cache not found: {filepath}")
                result = func(*args, **kwargs)
                if self.return_type and not isinstance(result, self.return_type):
                    raise ValueError(
                        f"The function '{func.__name__}' did not return the expected"
                        + f" type '{self.return_type.__name__}'"
                    )
                file_dir.mkdir(parents=True, exist_ok=True)
                self.save_fn(result, filepath)
                logger.debug(f"Saved to cache: {filepath}")
                return result

        return wrapped_func

    @property
    def cache_dir(self):
        """
        Method to return the cache directory.
        """
        return self.SEISMOMETER_CACHE_DIR / self.cache_name

    def clear(self):
        """
        Method to clear the cache directory.
        """
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        logger.debug(f"Cache cleared: {self.cache_name}")

    @classmethod
    def disable(cls):
        """
        Method to disable the cache.
        """
        cls.SEISMOMETER_CACHE_ENABLED = False
        logger.debug("Cache disabled")

    @classmethod
    def enable(cls):
        """
        Method to disable the cache.
        """
        cls.SEISMOMETER_CACHE_ENABLED = True
        logger.debug("Cache enabled")

    @classmethod
    def is_enabled(cls):
        """
        Method to check if the cache is enabled.
        """
        return cls.SEISMOMETER_CACHE_ENABLED

    @classmethod
    def clear_all(cls):
        """
        Method to clear all cache directories.
        """
        shutil.rmtree(cls.SEISMOMETER_CACHE_DIR, ignore_errors=True)
        logger.debug(f"Cache cleared: {cls.SEISMOMETER_CACHE_DIR}")


def _hash_function_args(*args, **kwargs):
    """
    Function to hash the arguments and keyword arguments of a function.

    Parameters
    ----------
    *args : Any
        arguments to hash
    *kwargs : Any
        keyword arguments to hash

    Returns
    -------
    str
        hash of the arguments and keyword arguments
    """
    hash_object = hashlib.md5()
    for arg in args:
        to_hash = _pandas_safe_hash(arg)
        hash_object.update(str(to_hash).encode())
    for key, arg in sorted(kwargs.items()):
        hash_object.update(key.encode())
        to_hash = _pandas_safe_hash(arg)
        hash_object.update(str(to_hash).encode())
    return hash_object.hexdigest()


def _pandas_safe_hash(value):
    """
    Function to get a hash value for a pandas object.
    """
    if isinstance(value, pd.DataFrame):
        value = value.sort_index().sort_index(axis=1)
        value = pd.util.hash_pandas_object(value)
    if isinstance(value, pd.Series):
        value = value.sort_index()
        value = str(hash(tuple(value.items())))
    if isinstance(value, pd.Index):
        value = str(hash(tuple(value)))
    return value
