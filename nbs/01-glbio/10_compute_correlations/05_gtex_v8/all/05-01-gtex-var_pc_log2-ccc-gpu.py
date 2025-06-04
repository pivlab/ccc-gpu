import pandas as pd
from time import time
from tqdm import tqdm
from pathlib import Path

from ccc.utils import simplify_string
from ccc.corr import ccc_gpu

GENE_SELECTION_STRATEGY = "var_pc_log2"
TOP_N_GENES = "all"

# select the top 5 tissues (according to sample size, see nbs/05_preprocessing/00-gtex_v8-split_by_tissue.ipynb)
TISSUES = [
    # "Muscle - Skeletal",
    "Whole Blood",
    # "Skin - Sun Exposed (Lower leg)",
    # "Adipose - Subcutaneous",
    # "Artery - Tibial",
]

N_CPU_CORE = 24

CORRELATION_METHOD = lambda x: ccc_gpu(x, n_jobs=N_CPU_CORE)
CORRELATION_METHOD.__name__ = "ccc_gpu"

method_name = CORRELATION_METHOD.__name__
print(method_name)

BENCHMARK_N_TOP_GENE = 10000

DATA_DIR = Path("/mnt/data/proj_data/ccc-gpu/gene_expr/data/gtex_v8")
INPUT_DIR = DATA_DIR / "gene_selection" / "all"
print(INPUT_DIR)

assert INPUT_DIR.exists()

OUTPUT_DIR = DATA_DIR / "similarity_matrices" / TOP_N_GENES
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(OUTPUT_DIR)

tissue_in_file_names = [f"_data_{simplify_string(t.lower())}-" for t in TISSUES]

input_files = sorted(list(INPUT_DIR.glob(f"*-{GENE_SELECTION_STRATEGY}.pkl")))
input_files = [
    f for f in input_files if any(tn in f.name for tn in tissue_in_file_names)
]
print(len(input_files))

assert len(input_files) == len(TISSUES), len(TISSUES)
print(input_files)

print(input_files[0])
test_data = pd.read_pickle(input_files[0])

print(f"test_data.shape: {test_data.shape}")

print(f"test_data.head(): {test_data.head()}")

pbar = tqdm(input_files, ncols=100)

for tissue_data_file in pbar:
    pbar.set_description(tissue_data_file.stem)

    # read
    data = pd.read_pickle(tissue_data_file)
    # data = data.iloc[:BENCHMARK_N_TOP_GENE]
    # compute correlations
    start_time = time()

    data_corrs = CORRELATION_METHOD(data)

    end_time = time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

    # save
    output_filename = f"{tissue_data_file.stem}-{method_name}-{TOP_N_GENES}.pkl"
    data_corrs.to_pickle(path=OUTPUT_DIR / output_filename)
