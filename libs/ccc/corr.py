"""
Functions to compute different correlation coefficients.

All correlation functions in this module are expected to have the same input and output
structure:

 * The input is a pandas DataFrame with genes in rows (Ensembl IDs) and samples
   in columns. The values are gene expression data normalized with some technique,
   but that should not be relevant for the correlation method. No empty values
   are allowed.

 * The output is a pandas DataFrame, a symmetric correlation matrix with genes
   in rows and columns (Ensembl IDs), and the values are the correlation
   coefficients. Diagonal values are expected to be ones.
"""

from __future__ import annotations

import pandas as pd

import numpy as np
from sklearn.metrics import pairwise_distances


def pearson(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation coefficient.
    """
    corr_mat = 1 - pairwise_distances(data.to_numpy(), metric="correlation", n_jobs=12)

    np.fill_diagonal(corr_mat, 1.0)

    return pd.DataFrame(
        corr_mat,
        index=data.index.copy(),
        columns=data.index.copy(),
    )


def spearman(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Spearman correlation coefficient.
    """
    # compute ranks
    data = data.rank(axis=1)

    corr_mat = 1 - pairwise_distances(data.to_numpy(), metric="correlation", n_jobs=12)

    np.fill_diagonal(corr_mat, 1.0)

    return pd.DataFrame(
        corr_mat,
        index=data.index.copy(),
        columns=data.index.copy(),
    )


def mic(data: pd.DataFrame, estimator="mic_approx", n_jobs=None) -> pd.DataFrame:
    """
    Compute the Maximal Correlation Coefficient (MIC).
    """
    from minepy import pstats

    from ccc.methods import mic as mic_single
    from scipy.spatial.distance import squareform

    if n_jobs is None:
        corr_mat = pstats(
            data.to_numpy(),
            est=estimator,
        )[0]

        corr_mat = squareform(corr_mat)
    else:
        corr_mat = pairwise_distances(data.to_numpy(), metric=mic_single, n_jobs=n_jobs)

    np.fill_diagonal(corr_mat, 1.0)

    return pd.DataFrame(
        corr_mat,
        index=data.index.copy(),
        columns=data.index.copy(),
    )


def ccc(data: pd.DataFrame, internal_n_clusters=None, n_jobs=1) -> pd.DataFrame:
    """
    Compute the Clustermatch Correlation Coefficient (CCC).
    """
    from ccc.coef import ccc
    from scipy.spatial.distance import squareform

    corr_mat = ccc(
        data.to_numpy(),
        internal_n_clusters=internal_n_clusters,
        n_jobs=n_jobs,
    )

    corr_mat = squareform(corr_mat)
    np.fill_diagonal(corr_mat, 1.0)

    return pd.DataFrame(
        corr_mat,
        index=data.index.copy(),
        columns=data.index.copy(),
    )


def ccc_gpu(data: pd.DataFrame, internal_n_clusters=10, n_jobs=24) -> pd.DataFrame:
    """
    Compute the Clustermatch Correlation Coefficient (CCC).
    """
    from ccc.coef.impl_gpu import ccc as ccc_gpu
    from scipy.spatial.distance import squareform

    corr_mat = ccc_gpu(
        data.to_numpy(),
        internal_n_clusters=internal_n_clusters,
        n_jobs=n_jobs,
    )

    corr_mat = squareform(corr_mat)
    np.fill_diagonal(corr_mat, 1.0)

    return pd.DataFrame(
        corr_mat,
        index=data.index.copy(),
        columns=data.index.copy(),
    )
