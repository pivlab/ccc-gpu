import pandas as pd
import numpy as np
from ccc.coef.impl import ccc
from ccc.coef.impl_gpu import ccc as ccc_gpu
from utils import clean_gpu_memory


@clean_gpu_memory
def test_ccc():
    # Load the Titanic dataset
    titanic_url = (
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/raw/titanic.csv"
    )
    titanic_df = pd.read_csv(titanic_url)
    titanic_df = titanic_df.dropna(subset=["embarked"]).dropna(axis=1)
    cpu_corrs = ccc(titanic_df)
    gpu_corrs = ccc_gpu(titanic_df)
    assert np.allclose(cpu_corrs, gpu_corrs)
