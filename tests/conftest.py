"""Shared pytest fixtures for the CCC test suite.

Fixtures that are useful to both the CPU (`tests/`) and GPU (`tests/gpu/`)
suites live here. GPU-only helpers (e.g. ``clean_gpu_memory``) live in
``tests/gpu/conftest.py``.
"""

import numpy as np
import pandas as pd
import pytest


def generate_categorical_data(
    n_features,
    n_objects,
    n_categories=3,
    categories=None,
    str_length=None,
    random_state=None,
    feature_names=None,
):
    """Generate a random categorical ``pandas.DataFrame`` of shape (n_objects, n_features).

    Parameters
    ----------
    n_features : int
        Number of features (columns).
    n_objects : int
        Number of objects (rows).
    n_categories : int, optional
        Number of unique categories when ``categories`` is None.
    categories : list or None, optional
        Explicit list of categories to sample from. If None, uses
        ``range(n_categories)`` or random strings when ``str_length`` is given.
    str_length : int or None, optional
        If given (and ``categories`` is None), generate random uppercase-letter
        string categories of this length.
    random_state : int or None, optional
        Seed for reproducibility.
    feature_names : list or None, optional
        Column names; defaults to ``feature_{i}``.
    """
    if random_state is not None:
        np.random.seed(random_state)

    if categories is None:
        if str_length is not None:
            letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            categories = [
                "".join(np.random.choice(letters, size=str_length))
                for _ in range(n_categories)
            ]
        else:
            categories = list(range(n_categories))
    else:
        n_categories = len(categories)

    random_indices = np.random.randint(0, n_categories, size=(n_objects, n_features))
    categorical_data = np.array(
        [[categories[idx] for idx in row] for row in random_indices]
    )

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    return pd.DataFrame(categorical_data, columns=feature_names)


@pytest.fixture
def categorical_data_generator():
    """Return the :func:`generate_categorical_data` helper for use in tests."""
    return generate_categorical_data
