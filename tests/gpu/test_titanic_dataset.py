import pandas as pd

from ccc.coef.impl import ccc
from utils import clean_gpu_memory


@clean_gpu_memory
def test_ccc():
    # Load the Titanic dataset
    titanic_url = (
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/raw/titanic.csv"
    )
    titanic_df = pd.read_csv(titanic_url)
    titanic_df = titanic_df.dropna(subset=["embarked"]).dropna(axis=1)
    print(titanic_df)
    # convert to numpy array
    # titanic_df = titanic_df.to_numpy()
    cpu_corrs, _, cou_parts = ccc(titanic_df, return_parts=True)
    print(cou_parts)

    # gpu_corrs, _, gpu_parts = ccc_gpu(titanic_df, return_parts=True)
    # print(gpu_parts)
    # assert np.alltrue(cou_parts == gpu_parts)
    # print(ccc_corrs_gpu)
