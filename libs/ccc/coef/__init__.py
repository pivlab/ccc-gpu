import numpy as np

from ccc.coef.impl import *  # noqa: F403, F401

# Run CCC to initialize/compile its functions with numba
from ccc.coef.impl import ccc

ccc(np.random.rand(10), np.random.rand(10))
