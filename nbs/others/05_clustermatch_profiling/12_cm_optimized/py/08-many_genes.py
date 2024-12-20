# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# Similar as `06` but with numba disabled to compare with a pure Python implementation.
#
# Here I had to reduce the number of `n_genes`, since it takes too much otherwise.

# %% [markdown] tags=[]
# # Disable numba

# %% tags=[]
# %env NUMBA_DISABLE_JIT=1

# %% [markdown] tags=[]
# # Remove pycache dir

# %% tags=[]
# !echo ${CODE_DIR}

# %% tags=[]
# !find ${CODE_DIR}/libs -regex '^.*\(__pycache__\)$' -print

# %% tags=[]
# !find ${CODE_DIR}/libs -regex '^.*\(__pycache__\)$' -prune -exec rm -rf {} \;

# %% tags=[]
# !find ${CODE_DIR}/libs -regex '^.*\(__pycache__\)$' -print

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import numpy as np

from ccc.coef import ccc

# %% tags=[]
# let numba compile all the code before profiling
ccc(np.random.rand(10), np.random.rand(10))

# %% [markdown] tags=[]
# # Data

# %% tags=[]
n_genes, n_samples = 50, 1000

# %% tags=[]
np.random.seed(0)

# %% tags=[]
data = np.random.rand(n_genes, n_samples)

# %% tags=[]
data.shape


# %% [markdown] tags=[]
# # Profile

# %% tags=[]
def func():
    n_clust = list(range(2, 10 + 1))
    return ccc(data, internal_n_clusters=n_clust)


# %% tags=[]
# %%timeit func()
func()

# %% tags=[]
# %%prun -s cumulative -l 50 -T 08-cm_many_genes.txt
func()

# %% [markdown] tags=[]
# **CONCLUSIONS:** compared with notebook `06` (which has 500 rows (`n_genes`) instead of 50 here), this one would have taken 2.80 hours for 500 rows based on this results. Whereas the numba-compiled version took ~7 minutes.

# %% tags=[]
