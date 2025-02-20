import functools
import cupy as cp
import pandas as pd
import numpy as np


def clean_gpu_memory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
    return wrapper


def generate_categorical_data(n_features, n_objects, n_categories=3, categories=None, 
                            str_length=None, random_state=None, feature_names=None):
    """
    Generate random categorical data as a pandas DataFrame.

    Parameters:
    -----------
    n_features : int
        Number of features (columns) in the output DataFrame
    n_objects : int
        Number of objects (rows) in the output DataFrame
    n_categories : int, optional (default=3)
        Number of unique categories to generate if categories parameter is None
    categories : list or None, optional (default=None)
        List of categories to sample from. If None, will use range(n_categories)
        or generate random strings if str_length is provided
    str_length : int or None, optional (default=None)
        If provided and categories is None, generate random string categories
        of this length using uppercase letters
    random_state : int or None, optional (default=None)
        Seed for random number generation
    feature_names : list or None, optional (default=None)
        List of column names. If None, will use [f'feature_{i}' for i in range(n_features)]

    Returns:
    --------
    pandas.DataFrame
        DataFrame of shape (n_objects, n_features) containing random categorical data

    Examples:
    --------
    >>> df = generate_categorical_data(2, 3, n_categories=3, str_length=2)
    >>> print(df)
      feature_0 feature_1
    0       XY       AB
    1       PQ       XY
    2       AB       PQ
    """
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Define categories if not provided
    if categories is None:
        if str_length is not None:
            # Generate random string categories
            letters = np.array(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
            categories = []
            for _ in range(n_categories):
                cat = ''.join(np.random.choice(letters, size=str_length))
                categories.append(cat)
        else:
            categories = list(range(n_categories))
    else:
        n_categories = len(categories)

    # Generate random indices
    random_indices = np.random.randint(0, n_categories, size=(n_objects, n_features))

    # Convert indices to categories
    categorical_data = np.array([[categories[idx] for idx in row] for row in random_indices])

    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]

    # Create DataFrame
    df = pd.DataFrame(categorical_data, columns=feature_names)
    
    return df
