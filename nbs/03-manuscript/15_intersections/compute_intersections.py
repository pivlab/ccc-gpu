#!/usr/bin/env python
# coding: utf-8

# # Description

# It analyzes how correlation coefficients intersect on different gene pairs. Basically, I take the top gene pairs with the maximum correlation coefficient according to Pearson, Spearman and CCC, and also the equivalent set with the minimum coefficient values, and then compare how these sets intersect each other.
# 
# After identifying different intersection sets, I plot some gene pairs to see what's being captured or not by each coefficient.

# # Modules

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from upsetplot import plot, from_indicators
from pathlib import Path
from ccc.plots import MyUpSet
from ccc import conf

import logging
from datetime import datetime
import shutil


# # Settings

# In[2]:


GTEX_TISSUE = "whole_blood"
GENE_SEL_STRATEGY = "var_pc_log2"
TOP_N_GENES = "all"

DATA_DIR = Path("/mnt/data/proj_data/ccc-gpu/gene_expr/data/gtex_v8")
SIMILARITY_MATRICES_DIR = DATA_DIR / "similarity_matrices" / TOP_N_GENES


# this specificies the threshold to compare coefficients (see below).
# it basically takes the top Q_DIFF coefficient values for gene pairs
# and compare with the bottom Q_DIFF of the other coefficients
Q_DIFF = 0.30


# In[3]:


OUTPUT_DIR = Path("/mnt/data/proj_data/ccc-gpu/results/gene_pair_intersections")
OUTPUT_FIGURE_NAME = "upsetplot_gtex_{GTEX_TISSUE}"
OUTPUT_GENE_PAIR_INTERSECTIONS_NAME = f"gene_pair_intersections-gtex_v8-{GTEX_TISSUE}-{GENE_SEL_STRATEGY}.pkl"

# Create timestamp-based log folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = Path("logs") / timestamp
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOG_DIR / "compute_intersections.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Starting compute_intersections.py with timestamp: {timestamp}")
logger.info(f"Log directory: {LOG_DIR}")
logger.info(f"Configuration:")
logger.info(f"  GTEX_TISSUE: {GTEX_TISSUE}")
logger.info(f"  GENE_SEL_STRATEGY: {GENE_SEL_STRATEGY}")
logger.info(f"  TOP_N_GENES: {TOP_N_GENES}")
logger.info(f"  Q_DIFF: {Q_DIFF}")
logger.info(f"  OUTPUT_DIR: {OUTPUT_DIR}")


# In[4]:


assert OUTPUT_DIR.exists()
assert SIMILARITY_MATRICES_DIR.exists()


# In[5]:


SIMILARITY_MATRIX_FILENAME_TEMPLATE = "gtex_v8_data_{tissue}-{gene_sel_strategy}-{corr_method}.pkl"
INPUT_CORR_FILE_TEMPLATE = SIMILARITY_MATRICES_DIR / SIMILARITY_MATRIX_FILENAME_TEMPLATE
logger.info(f"Input correlation file template: {INPUT_CORR_FILE_TEMPLATE}")


# In[6]:


INPUT_CORR_FILE = SIMILARITY_MATRICES_DIR / str(
    INPUT_CORR_FILE_TEMPLATE
).format(
    tissue=GTEX_TISSUE,
    gene_sel_strategy=GENE_SEL_STRATEGY,
    corr_method="all",
)
logger.info(f"Input correlation file: {INPUT_CORR_FILE}")

assert INPUT_CORR_FILE.exists()


# # Data

# ## Correlation

# In[7]:


logger.info("Loading correlation data...")
df = pd.read_pickle(INPUT_CORR_FILE)


# In[8]:


logger.info(f"Dataframe shape: {df.shape}")


# In[9]:


logger.info("Dataframe head (first 15 rows):")
logger.info(f"\n{df.head(15)}")


# In[17]:


logger.info("Dataframe statistics:")
logger.info(f"\n{df.describe()}")


# In[10]:


# Calculate quantiles from 20% to 100% in 20 steps for each correlation method
# This helps understand the distribution of correlation values and identify
# appropriate thresholds for high/low correlation gene pairs
quantiles_result = df.apply(lambda x: x.quantile(np.linspace(0.20, 1.0, 20)))
logger.info("Quantiles from 20% to 100% in 20 steps:")
logger.info(f"\n{quantiles_result}")


# # Prepare data for plotting

# In[11]:


def get_lower_upper_quantile(method_name, q):
    """Get the lower and upper quantile bounds for a correlation method.
    
    This function calculates the quantile thresholds that will be used to
    identify gene pairs with high and low correlation values. Gene pairs
    with correlation values <= lower quantile are considered "low correlation",
    while those with values >= upper quantile are considered "high correlation".
    
    Args:
        method_name (str): Name of the correlation method column ('ccc', 'pearson', 'spearman')
        q (float): Quantile difference from extremes (e.g., 0.30 means use 30% and 70% quantiles)
    
    Returns:
        pandas.Series: Series with two values [lower_quantile, upper_quantile]
    """
    return df[method_name].quantile([q, 1 - q])


# In[12]:


# Test the get_lower_upper_quantile function with CCC method
# Using 0.20 quantile difference (20% and 80% quantiles)
ccc_quantiles = get_lower_upper_quantile("ccc", 0.20)
logger.info("CCC quantiles test (0.20):")
logger.info(f"{ccc_quantiles}")

# Extract the lower and upper quantile values
ccc_lower_quantile, ccc_upper_quantile = ccc_quantiles
logger.info(f"CCC lower quantile: {ccc_lower_quantile}, upper quantile: {ccc_upper_quantile}")

# Verify that the unpacked values match the series indexing
assert ccc_lower_quantile == ccc_quantiles.iloc[0]  # Lower quantile (20%)
assert ccc_upper_quantile == ccc_quantiles.iloc[1]  # Upper quantile (80%)

# Clean up variables after assertions
del ccc_quantiles, ccc_lower_quantile, ccc_upper_quantile


# In[13]:


clustermatch_lq, clustermatch_hq = get_lower_upper_quantile("ccc", Q_DIFF)
logger.info(f"Clustermatch quantiles: lower={clustermatch_lq}, upper={clustermatch_hq}")

pearson_lq, pearson_hq = get_lower_upper_quantile("pearson", Q_DIFF)
logger.info(f"Pearson quantiles: lower={pearson_lq}, upper={pearson_hq}")

spearman_lq, spearman_hq = get_lower_upper_quantile("spearman", Q_DIFF)
logger.info(f"Spearman quantiles: lower={spearman_lq}, upper={spearman_hq}")


# In[14]:


pearson_higher = df["pearson"] >= pearson_hq
logger.info(f"Pearson higher count: {pearson_higher.sum()}")

pearson_lower = df["pearson"] <= pearson_lq
logger.info(f"Pearson lower count: {pearson_lower.sum()}")


# In[15]:


spearman_higher = df["spearman"] >= spearman_hq
logger.info(f"Spearman higher count: {spearman_higher.sum()}")

spearman_lower = df["spearman"] <= spearman_lq
logger.info(f"Spearman lower count: {spearman_lower.sum()}")


# In[16]:


clustermatch_higher = df["ccc"] >= clustermatch_hq
logger.info(f"Clustermatch higher count: {clustermatch_higher.sum()}")

clustermatch_lower = df["ccc"] <= clustermatch_lq
logger.info(f"Clustermatch lower count: {clustermatch_lower.sum()}")


# # UpSet plot

# In[17]:


# Create a dataframe for plotting with boolean columns indicating whether each gene pair
# falls into the high or low quantile ranges for each correlation method
df_plot = pd.DataFrame(
    {
        "pearson_higher": pearson_higher,  # Gene pairs with Pearson correlation >= high quantile
        "pearson_lower": pearson_lower,    # Gene pairs with Pearson correlation <= low quantile
        "spearman_higher": spearman_higher,  # Gene pairs with Spearman correlation >= high quantile
        "spearman_lower": spearman_lower,    # Gene pairs with Spearman correlation <= low quantile
        "clustermatch_higher": clustermatch_higher,  # Gene pairs with CCC >= high quantile
        "clustermatch_lower": clustermatch_lower,    # Gene pairs with CCC <= low quantile
    }
)


# In[18]:


# Add the original correlation values (ccc, pearson, spearman) to the plot dataframe
df_plot = pd.concat([df_plot, df], axis=1)


# In[19]:


# Clean up
del df
del pearson_higher, pearson_lower, spearman_higher, spearman_lower, clustermatch_higher, clustermatch_lower


# In[20]:


logger.info(f"Plot dataframe shape: {df_plot.shape}")
logger.info(f"Plot dataframe columns: {df_plot.columns.tolist()}")


# In[21]:


assert not df_plot.isna().any().any()


# In[22]:


# Rename columns to more descriptive names for plotting
# This creates cleaner labels for the UpSet plot visualization
df_plot = df_plot.rename(
    columns={
        "pearson_higher": "Pearson (high)",
        "pearson_lower": "Pearson (low)",
        "spearman_higher": "Spearman (high)",
        "spearman_lower": "Spearman (low)",
        "clustermatch_higher": "Clustermatch (high)",
        "clustermatch_lower": "Clustermatch (low)",
    }
)


# In[24]:


# Create sorted list of category names for the UpSet plot
# Filter columns that contain " (" (our boolean indicator columns)
# Sort by: first by threshold level (high/low), then by method name
# This ensures consistent ordering: high thresholds before low thresholds,
# and within each threshold level, methods are alphabetically ordered
categories = sorted(
    [x for x in df_plot.columns if " (" in x],  # Get boolean indicator columns
    reverse=True,  # Reverse to get "high" before "low"
    key=lambda x: x.split(" (")[1] + " (" + x.split(" (")[0],  # Sort by threshold then method
)


# In[25]:


logger.info(f"Categories for upset plot: {categories}")


# ## All subsets (original full plot)

# In[26]:


df_r_data = df_plot


# In[27]:


logger.info(f"Data shape for upset plot: {df_r_data.shape}")


# In[28]:


# Convert the boolean indicator DataFrame to an UpSet-compatible format
# This transforms our boolean columns (categories) into a multi-index structure
# where each gene pair is associated with the set of categories it belongs to

# from_indicators is a function from the upsetplot library that converts
# boolean indicator columns into a multi-index Series suitable for UpSet plots
# It takes the boolean columns (categories) and creates a Series where:
# - The index is a MultiIndex with levels for each category
# - Each gene pair is associated with True/False for each category
# - Only combinations that actually exist in the data are included
gene_pairs_by_cats = from_indicators(categories, data=df_r_data)


# In[29]:


logger.info(f"Gene pairs by categories shape: {gene_pairs_by_cats.shape}")
logger.info("First few entries:")
logger.info(f"\n{gene_pairs_by_cats.head()}")


# In[30]:


logger.info("Creating full upset plot...")
fig = plt.figure(figsize=(18, 5))

g = plot(
    gene_pairs_by_cats,
    show_counts=True,
    sort_categories_by=None,
    element_size=None,
    fig=fig,
)


# In[ ]:


full_svg_path = OUTPUT_DIR / f"{OUTPUT_FIGURE_NAME}_full.svg"
plt.savefig(
    full_svg_path,
    bbox_inches="tight",
    facecolor="white",
)
logger.info(f"Saved full upset plot to: {full_svg_path}")

# Also save to log directory
log_svg_path = LOG_DIR / f"{OUTPUT_FIGURE_NAME}_full.svg"
shutil.copy2(full_svg_path, log_svg_path)
logger.info(f"Copied full upset plot to log directory: {log_svg_path}")


# ## Sort by categories of subsets

# In[31]:


df_r_data = df_plot


# In[32]:


logger.info(f"Data shape for trimmed plot: {df_r_data.shape}")


# In[33]:


gene_pairs_by_cats = from_indicators(categories, data=df_r_data)


# In[34]:


logger.info("Gene pairs by categories for trimmed plot:")
logger.info(f"Shape: {gene_pairs_by_cats.shape}")


# In[35]:


gene_pairs_by_cats = gene_pairs_by_cats.sort_index()


# In[36]:


_tmp_index = gene_pairs_by_cats.index.unique().to_frame(False)
logger.info("Unique index combinations:")
logger.info(f"\n{_tmp_index}")


# In[37]:


combinations_with_3 = _tmp_index[_tmp_index.sum(axis=1) == 3]
logger.info(f"Combinations with exactly 3 True values:")
logger.info(f"\n{combinations_with_3}")


# In[38]:


first_3_zero = _tmp_index.apply(lambda x: x[0:3].sum() == 0, axis=1)
logger.info(f"Number of combinations where first 3 are all False: {first_3_zero.sum()}")


# In[39]:


# agreements on top
agreements_top = _tmp_index.loc[
    _tmp_index[
        _tmp_index.apply(lambda x: x.sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: x[0:3].sum() == 0, axis=1)
        & _tmp_index.apply(lambda x: 3 > x[3:].sum() > 1, axis=1)
    ].index
].apply(tuple, axis=1).to_numpy()
logger.info(f"Agreements on top: {agreements_top}")


# In[40]:


# agreements on bottom
agreements_bottom = _tmp_index.loc[
    _tmp_index[
        _tmp_index.apply(lambda x: x.sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: 3 > x[0:3].sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: x[3:].sum() == 0, axis=1)
    ].index
].apply(tuple, axis=1).to_numpy()
logger.info(f"Agreements on bottom: {agreements_bottom}")


# In[41]:


# diagreements
disagreements = _tmp_index.loc[
    _tmp_index[
        _tmp_index.apply(lambda x: x.sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: x[0:3].sum() > 0, axis=1)
        & _tmp_index.apply(lambda x: x[3:].sum() > 0, axis=1)
    ].index
].apply(tuple, axis=1).to_numpy()
logger.info(f"Disagreements: {disagreements}")


# In[42]:


# order subsets
gene_pairs_by_cats = gene_pairs_by_cats.loc[
    [
        # pairs not included in categories:
        # (False, False, False, False, False, False),
        # full agreements on high:
        (False, False, False, True, True, True),
        # agreements on top
        (False, False, False, False, True, True),
        (False, False, False, True, False, True),
        (False, False, False, True, True, False),
        # agreements on bottom
        (False, True, True, False, False, False),
        (True, False, True, False, False, False),
        (True, True, False, False, False, False),
        # full agreements on low:
        (True, True, True, False, False, False),
        # diagreements
        #   ccc
        (False, True, False, True, False, True),
        (False, True, False, False, False, True),
        (True, False, False, False, False, True),
        (True, True, False, False, False, True),
        #   pearson
        (False, False, True, False, True, False),
        (True, False, False, False, True, False),
        (True, False, True, False, True, False),
        #   spearman
        (False, True, False, True, False, False),
    ]
]


# In[43]:


logger.info("Gene pairs by categories after reordering:")
logger.info(f"\n{gene_pairs_by_cats.head()}")


# In[44]:


gene_pairs_by_cats = gene_pairs_by_cats.rename(
    columns={
        "Clustermatch (high)": "CCC (high)",
        "Clustermatch (low)": "CCC (low)",
    }
)

gene_pairs_by_cats.index.set_names(
    {
        "Clustermatch (high)": "CCC (high)",
        "Clustermatch (low)": "CCC (low)",
    },
    inplace=True,
)


# In[45]:


logger.info("Creating trimmed upset plot...")
fig = plt.figure(figsize=(14, 5))

# g = plot(
g = MyUpSet(
    gene_pairs_by_cats,
    show_counts=True,
    sort_categories_by=None,
    sort_by=None,
    show_percentages=True,
    # min_subset_size=2,
    element_size=None,
    # fig=fig,
).plot(fig)

g["totals"].remove()  # set_visible(False)

# display(fig.get_size_inches())
# fig.set_size_inches(12, 5)

trimmed_svg_path = OUTPUT_DIR / f"{OUTPUT_FIGURE_NAME}_trimmed.svg"
plt.savefig(
    trimmed_svg_path,
    bbox_inches="tight",
    facecolor="white",
)
logger.info(f"Saved trimmed upset plot to: {trimmed_svg_path}")

# Also save to log directory
log_svg_path = LOG_DIR / f"{OUTPUT_FIGURE_NAME}_trimmed.svg"
shutil.copy2(trimmed_svg_path, log_svg_path)
logger.info(f"Copied trimmed upset plot to log directory: {log_svg_path}")

# plt.margins(x=-0.4)


# This plot has the sets that represent agreements on the left, and disagreements on the right.

# The plot shown here is **not the final one for the manuscript**:
# 
# 1. Open the main output svg file (`upsetplot-main.svg`)
# 1. Include the file generated here (`upsetplot.svg`)
# 1. Rearrange the `1e6` at the top, which is overlapping other numbers.
# 1. Add the triangles (red and green). For this I need to move the category names at the left to make space.
# 1. Add a rectangle and clip it to remove the extra space on the left
# 1. Add the "Agreements" and "Disagreements" labels below.
# 1. Automatically resize page to drawing.
# 1. Add a rectangle that covers the entire drawing with white background. And send it to the background.

# # Save groups of gene pairs in each subset

# In[46]:


logger.info(f"Final dataframe shape: {df_plot.shape}")
logger.info("Final dataframe head:")
logger.info(f"\n{df_plot.head()}")


# In[4]:


output_file = (
    OUTPUT_DIR
    / OUTPUT_GENE_PAIR_INTERSECTIONS_NAME
)
logger.info(f"Output file for gene pair intersections: {output_file}")


# In[49]:


logger.info("Saving gene pair intersections data...")
df_plot.to_pickle(output_file)
logger.info(f"Saved gene pair intersections to: {output_file}")

# Also save to log directory
log_pkl_path = LOG_DIR / OUTPUT_GENE_PAIR_INTERSECTIONS_NAME
shutil.copy2(output_file, log_pkl_path)
logger.info(f"Copied gene pair intersections to log directory: {log_pkl_path}")


# In[ ]:


logger.info("Script completed successfully!")
logger.info(f"All outputs saved to log directory: {LOG_DIR}")