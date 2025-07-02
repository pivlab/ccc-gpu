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


# In[4]:


assert OUTPUT_DIR.exists()
assert SIMILARITY_MATRICES_DIR.exists()


# In[5]:


SIMILARITY_MATRIX_FILENAME_TEMPLATE = "gtex_v8_data_{tissue}-{gene_sel_strategy}-{corr_method}.pkl"
INPUT_CORR_FILE_TEMPLATE = SIMILARITY_MATRICES_DIR / SIMILARITY_MATRIX_FILENAME_TEMPLATE
display(INPUT_CORR_FILE_TEMPLATE)


# In[6]:


INPUT_CORR_FILE = SIMILARITY_MATRICES_DIR / str(
    INPUT_CORR_FILE_TEMPLATE
).format(
    tissue=GTEX_TISSUE,
    gene_sel_strategy=GENE_SEL_STRATEGY,
    corr_method="all",
)
display(INPUT_CORR_FILE)

assert INPUT_CORR_FILE.exists()


# # Data

# ## Correlation

# In[7]:


df = pd.read_pickle(INPUT_CORR_FILE)


# In[8]:


df.shape


# In[9]:


df.head(15)


# In[17]:


df.describe()


# In[10]:


# Calculate quantiles from 20% to 100% in 20 steps for each correlation method
# This helps understand the distribution of correlation values and identify
# appropriate thresholds for high/low correlation gene pairs
df.apply(lambda x: x.quantile(np.linspace(0.20, 1.0, 20)))


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
display(ccc_quantiles)

# Extract the lower and upper quantile values
ccc_lower_quantile, ccc_upper_quantile = ccc_quantiles
display((ccc_lower_quantile, ccc_upper_quantile))

# Verify that the unpacked values match the series indexing
assert ccc_lower_quantile == ccc_quantiles.iloc[0]  # Lower quantile (20%)
assert ccc_upper_quantile == ccc_quantiles.iloc[1]  # Upper quantile (80%)

# Clean up variables after assertions
del ccc_quantiles, ccc_lower_quantile, ccc_upper_quantile


# In[13]:


clustermatch_lq, clustermatch_hq = get_lower_upper_quantile("ccc", Q_DIFF)
display((clustermatch_lq, clustermatch_hq))

pearson_lq, pearson_hq = get_lower_upper_quantile("pearson", Q_DIFF)
display((pearson_lq, pearson_hq))

spearman_lq, spearman_hq = get_lower_upper_quantile("spearman", Q_DIFF)
display((spearman_lq, spearman_hq))


# In[14]:


pearson_higher = df["pearson"] >= pearson_hq
display(pearson_higher.sum())

pearson_lower = df["pearson"] <= pearson_lq
display(pearson_lower.sum())


# In[15]:


spearman_higher = df["spearman"] >= spearman_hq
display(spearman_higher.sum())

spearman_lower = df["spearman"] <= spearman_lq
display(spearman_lower.sum())


# In[16]:


clustermatch_higher = df["ccc"] >= clustermatch_hq
display(clustermatch_higher.sum())

clustermatch_lower = df["ccc"] <= clustermatch_lq
display(clustermatch_lower.sum())


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


df_plot


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


categories


# ## All subsets (original full plot)

# In[26]:


df_r_data = df_plot


# In[27]:


df_r_data.shape


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


gene_pairs_by_cats


# In[30]:


fig = plt.figure(figsize=(18, 5))

g = plot(
    gene_pairs_by_cats,
    show_counts=True,
    sort_categories_by=None,
    element_size=None,
    fig=fig,
)


# In[ ]:


plt.savefig(
    OUTPUT_DIR / OUTPUT_FIGURE_NAME + "_full.svg",
    bbox_inches="tight",
    facecolor="white",
)


# ## Sort by categories of subsets

# In[31]:


df_r_data = df_plot


# In[32]:


df_r_data.shape


# In[33]:


gene_pairs_by_cats = from_indicators(categories, data=df_r_data)


# In[34]:


gene_pairs_by_cats


# In[35]:


gene_pairs_by_cats = gene_pairs_by_cats.sort_index()


# In[36]:


_tmp_index = gene_pairs_by_cats.index.unique().to_frame(False)
display(_tmp_index)


# In[37]:


_tmp_index[_tmp_index.sum(axis=1) == 3]


# In[38]:


_tmp_index.apply(lambda x: x[0:3].sum() == 0, axis=1)


# In[39]:


# agreements on top
_tmp_index.loc[
    _tmp_index[
        _tmp_index.apply(lambda x: x.sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: x[0:3].sum() == 0, axis=1)
        & _tmp_index.apply(lambda x: 3 > x[3:].sum() > 1, axis=1)
    ].index
].apply(tuple, axis=1).to_numpy()


# In[40]:


# agreements on bottom
_tmp_index.loc[
    _tmp_index[
        _tmp_index.apply(lambda x: x.sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: 3 > x[0:3].sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: x[3:].sum() == 0, axis=1)
    ].index
].apply(tuple, axis=1).to_numpy()


# In[41]:


# diagreements
_tmp_index.loc[
    _tmp_index[
        _tmp_index.apply(lambda x: x.sum() > 1, axis=1)
        & _tmp_index.apply(lambda x: x[0:3].sum() > 0, axis=1)
        & _tmp_index.apply(lambda x: x[3:].sum() > 0, axis=1)
    ].index
].apply(tuple, axis=1).to_numpy()


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


gene_pairs_by_cats.head()


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

plt.savefig(
    OUTPUT_DIR / OUTPUT_FIGURE_NAME + "_trimmed.svg",
    bbox_inches="tight",
    facecolor="white",
)

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


display(df_plot.shape)
display(df_plot.head())


# In[4]:


output_file = (
    OUTPUT_DIR
    / OUTPUT_GENE_PAIR_INTERSECTIONS_NAME
)
display(output_file)


# In[49]:


df_plot.to_pickle(output_file)


# In[ ]:




