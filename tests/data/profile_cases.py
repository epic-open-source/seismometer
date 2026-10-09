"""A frame with every variable type and alert, shared by the profile tests and make_profile_golden.py."""
from datetime import date

import numpy as np
import pandas as pd

N = 1000


def make_profile_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)

    float_mixed = rng.normal(2, 3, N)
    float_mixed[rng.choice(N, 50, replace=False)] = np.nan
    float_mixed[:15] = np.inf
    float_mixed[15:20] = -np.inf
    float_mixed[20:50] = 0

    skewed = np.zeros(N)
    skewed[7] = 1e6
    skewed[100:400] = np.arange(300) / 300

    phrases = np.array(
        ["Very good.", "good", "Poor, needs work", "neutral", "excellent!", "Very Poor", "ok - fine"], dtype=object
    )
    text = phrases[rng.choice(len(phrases), N, p=[0.31, 0.22, 0.15, 0.12, 0.1, 0.06, 0.04])]
    text[rng.choice(N, 100, replace=False)] = None

    small = pd.Categorical(
        rng.choice(["A", "B", "C", "D"], N, p=[0.82, 0.1, 0.05, 0.03]), categories=["A", "B", "C", "D", "unused"]
    )
    small[rng.choice(N, 30, replace=False)] = np.nan

    bool_object = np.array([True] * 700 + [False] * 250 + [None] * 50, dtype=object)
    rng.shuffle(bool_object)

    nullable = pd.array(rng.integers(-5, 40, N), dtype="Int64")
    nullable[rng.choice(N, 40, replace=False)] = pd.NA

    return pd.DataFrame(
        {
            "float_mixed": float_mixed,
            "int_counts": rng.poisson(0.4, N),
            "constant": np.full(N, 7),
            "unique_float": np.arange(N) * 1.5,
            "skewed": skewed,
            "bool_col": rng.random(N) < 0.96,
            "bool_object": bool_object,
            "cat_small": small,
            "cat_int": pd.Categorical(rng.choice([3, 1, 2], N, p=[0.5, 0.3, 0.2])),
            "cat_high": pd.Categorical([f"c{i % 80:02d}" for i in rng.permutation(N)]),
            "text": text,
            "text_unique": np.array([f"id-{i:04d} x" for i in range(N)], dtype=object),
            "mixed_object": np.array(["a", 2.5, None, "b"] * (N // 4), dtype=object),
            "all_missing": np.full(N, np.nan),
            "nullable_int": nullable,
            "object_dates": np.array([date(2024, 1, 1 + i % 28) for i in range(N)], dtype=object),
        }
    )
