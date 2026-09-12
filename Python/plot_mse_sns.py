"""
Author      : Jie Li
Date        : 2026-09-06 22:46:57
Last Edited : 2026-09-11 14:45:14
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/plot_mse_sns.py
Description :








Copyright (c) 2026, Jie Li, lijie_12@outlook.com
All Rights Reserved.
"""

# %% ========================================================================
# 				import packages
# ===========================================================================
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% ========================================================================
# 				load the data
# ===========================================================================
input_csv = Path("../Results/combined_results_summary.csv")
full_df = pd.read_csv(input_csv)
# Log transforms (add small epsilon)
EPSILON = 1e-16
full_df["log10_mse"] = np.log10(full_df["mse"].astype(float) + EPSILON)
# Ordered categories for consistent display
method_order = ["NW", "CE", "CL", "LL"]
methods = [m for m in method_order if m in full_df["method"].astype(str).unique()]
full_df["method"] = pd.Categorical(
    full_df["method"].astype(str), categories=methods, ordered=True
)
corr_vals = sorted(full_df["corr_fun"].astype(str).dropna().unique().tolist())
kernel_vals = sorted(full_df["kernel"].astype(str).dropna().unique().tolist())
strategy_vals = sorted(
    full_df["sample_strategy"].astype(str).dropna().unique().tolist()
)
n_vals = sorted(full_df["sample_size"].astype(int).dropna().unique().tolist())

full_df["corr_fun"] = pd.Categorical(
    full_df["corr_fun"].astype(str), categories=corr_vals, ordered=True
)
full_df["kernel"] = pd.Categorical(
    full_df["kernel"].astype(str), categories=kernel_vals, ordered=True
)
full_df["sample_strategy"] = pd.Categorical(
    full_df["sample_strategy"].astype(str), categories=strategy_vals, ordered=True
)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
    }
)

# %% ========================================================================
# 			1) mse heatmap (raw seconds, NO log10)
# ===========================================================================
row_index = pd.MultiIndex.from_product(
    [methods, strategy_vals, n_vals], names=["method", "sample_strategy", "sample_size"]
)
col_index = pd.MultiIndex.from_product(
    [corr_vals, kernel_vals], names=["corr_fun", "kernel"]
)

mse_matrix = (
    full_df.groupby(
        ["method", "sample_strategy", "sample_size", "corr_fun", "kernel"],
        observed=True,
    )["log10_mse"]
    .median()
    .unstack(["corr_fun", "kernel"])
    .reindex(index=row_index, columns=col_index)
)
mse_matrix = mse_matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")

# Row / column annotations from the MultiIndex levels
row_annot = pd.DataFrame(
    {
        "Method": [str(x) for x in mse_matrix.index.get_level_values(0)],
        "Strategy": [str(x) for x in mse_matrix.index.get_level_values(1)],
        "SampleSize": [int(x) for x in mse_matrix.index.get_level_values(2)],
    },
    index=mse_matrix.index,
)
col_annot = pd.DataFrame(
    {
        "Corr": [str(x) for x in mse_matrix.columns.get_level_values(0)],
        "Kernel": [str(x) for x in mse_matrix.columns.get_level_values(1)],
    },
    index=mse_matrix.columns,
)

# Pretty labels for heatmap axes
mse_matrix.index = [f"{m} | {s} | n={int(n)}" for m, s, n in mse_matrix.index]
mse_matrix.columns = [f"{a} | {b}" for a, b in mse_matrix.columns]
row_annot.index = mse_matrix.index
col_annot.index = mse_matrix.columns

# Colors
METHOD_COLORS = {
    m: c for m, c in zip(methods, sns.color_palette("tab10", len(methods)))
}
STRATEGY_COLORS = {
    s: c for s, c in zip(strategy_vals, sns.color_palette("Set2", len(strategy_vals)))
}
SAMPLE_COLORS = {
    n: c for n, c in zip(n_vals, sns.color_palette("viridis", len(n_vals)))
}
CORR_COLORS = {
    c: col for c, col in zip(corr_vals, sns.color_palette("tab20", len(corr_vals)))
}
KERNEL_COLORS = {
    k: col for k, col in zip(kernel_vals, sns.color_palette("Dark2", len(kernel_vals)))
}

row_colors = pd.DataFrame(
    {
        "Method": row_annot["Method"].map(METHOD_COLORS),
        "Strategy": row_annot["Strategy"].map(STRATEGY_COLORS),
        "Sample Size": row_annot["SampleSize"].map(SAMPLE_COLORS),
    },
    index=mse_matrix.index,
)
col_colors = pd.DataFrame(
    {
        "Correlation Function": col_annot["Corr"].map(CORR_COLORS),
        "Kernel Function": col_annot["Kernel"].map(KERNEL_COLORS),
    },
    index=mse_matrix.columns,
)

g2 = sns.clustermap(
    mse_matrix,
    row_cluster=False,
    col_cluster=False,
    row_colors=row_colors,
    col_colors=col_colors,
    cmap="crest",
    dendrogram_ratio=(0.01, 0.01),  # minimize empty space for the (hidden) dendrograms
    cbar_kws={"label": "Logarithmic MSE", "shrink": 0.85, "aspect": 25},
    xticklabels=True,
    yticklabels=True,
    figsize=(18, 12),
)
g2.ax_heatmap.set_xlabel(r"Correlation function $\times$ Kernel function", fontsize=12)
g2.ax_heatmap.set_ylabel(
    r"Method $\times$ Sampling Strategy $\times$ Sample Size", fontsize=12
)
g2.ax_heatmap.set_title("Median logarithmic MSE across repeats", fontsize=12)
# elegant colorbar: slim, no box, small ticks, vertically centered on the right
if getattr(g2, "cax", None) is not None:
    g2.cax.tick_params(labelsize=8)
    for spine in g2.cax.spines.values():
        spine.set_visible(False)
    # place the colorbar to the right of the heatmap, vertically centered
    hp = g2.ax_heatmap.get_position()
    cp = g2.cax.get_position()
    gap = 0.09
    g2.cax.set_position(
        [hp.x1 + gap, hp.y0 + (hp.height - cp.height) / 2.0, cp.width, cp.height]
    )
out_heat = Path("../figs/panel_mse_heatmap.png")
out_heat.parent.mkdir(parents=True, exist_ok=True)
g2.savefig(out_heat, dpi=800, bbox_inches="tight")
plt.show()
print("Saved:", out_heat)
