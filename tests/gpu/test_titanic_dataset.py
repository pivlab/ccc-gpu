import pandas as pd
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
    cpu_corrs = pd.DataFrame(ccc(titanic_df))
    gpu_corrs = pd.DataFrame(ccc_gpu(titanic_df))
    assert pd.testing.assert_frame_equal(cpu_corrs, gpu_corrs, check_exact=False)
