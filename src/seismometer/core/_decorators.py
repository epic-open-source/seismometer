# -*- coding: utf-8 -*-
"""
As for all of the utils subpackage, no other custom subpackages should be referenced.
"""
import builtins
import hashlib
import os
import pickle
import sys
import time
from functools import wraps
from typing import Any, Callable

import numpy as np


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


def indented_function(func: Callable[..., any]) -> Callable[..., any]:
    """Decorator that will add an indentation to any print() calls made inside the function.

    Returns
    -------
    Callable[..., any]
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


CACHE_POLICY = ["load", "reload", "skip"]
CACHE_TYPE = ["pkl", "npz"]


def disk_cached_function(
    cache_name: str, *, cache_type: str = "pkl"
) -> Callable[[Callable[..., any]], Callable[..., any]]:
    """Creates a decorator function that will store data to a cache folder on disk.

    Notes:

    1.  The cache uses a hash based on function __name__ and inputs.
        If you have two different functions with the same name and inputs, it will hit the SAME cache.
    2.  The cache is a DISK cache, so different processes or runs of the same code will hit the SAME cache,
        this means changing return values requires you to delete the cache folder.

    Parameters
    ----------
    cache_name : str
        Local path to cache directory.
    cache_type : str, optional
        Type of cache method to use, pickle of numpy save, by default 'pkl'.

    Returns
    -------
    Decorator that does the actual work.
    """

    def cached_decorator(func):
        func = indented_function(func)

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            policy = kwargs.pop("cache_policy", "load")
            if policy not in CACHE_POLICY:
                raise ValueError("{func.__name_} only accepts cache options in: {CACHE_POLICY}")

            if cache_type not in CACHE_TYPE:
                raise ValueError("{func.__name_} @cached decorator only accepts cache_type option in: {CACHE_TYPE}")

            arg_str = str((args, frozenset(kwargs.items())))
            hash_object = hashlib.md5(arg_str.encode())
            arg_hash = hash_object.hexdigest()

            cache_path = os.path.join(cache_name, func.__name__)

            os.makedirs(cache_path, exist_ok=True)

            cached_path = os.path.join(cache_path, f"{arg_hash}.{cache_type}")
            res = None
            if policy == "load" and os.path.exists(cached_path):
                try:
                    print(f"Loading result from cache: {cached_path}")
                    file = open(cached_path, "rb")
                    if cache_type == "pkl":
                        res = pickle.load(file)
                    elif cache_type == "npz":
                        res = np.load(file)["res"]
                except EOFError:
                    print("Failed to load cache: EOFError")
                    policy = "reload"
            else:
                print("Failed to load cache: FileNotFound")
                policy = "reload"

            if policy != "load":
                print("Calculating value directly...")
                res = func(*args, **kwargs)

            if not os.path.exists(cached_path) or policy == "reload":
                try:
                    print(f"Saving to cache: {cached_path}")
                    file = open(cached_path, "wb")
                    if cache_type == "pkl":
                        pickle.dump(res, file)
                    if cache_type == "npz":
                        np.savez_compressed(file, res=res)
                except OverflowError:
                    print(f"Could not save to cache: {cached_path}")
            return res

        return wrapped_func

    return cached_decorator
