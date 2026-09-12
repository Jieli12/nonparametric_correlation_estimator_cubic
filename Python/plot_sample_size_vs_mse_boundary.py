"""
Author      : Jie Li
Date        : 2026-08-16 22:53:00
Last Edited : 2026-09-11 14:24:30
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/plot_sample_size_vs_mse_boundary.py
Description :

Violin (box-shaped) plots of log10(boundary MSE) vs sample size.

Only the CE and CL methods and the S2 sampling strategy are shown.
For each sample size n (log2 x-axis), one violin per method gives the
distribution of log10(MSE_B) across corr_fun x kernel x 200 replications.


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
out_dir = "../figs/sample_size_vs_mse_boundary.png"

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

# Only CE / CL methods, only the S2 sampling strategy
METHODS = ["NW", "CE", "CL", "LL"]
STRATEGY = "S2"
df = full_df.copy()
df["log10_mse_b"] = np.log10(df["mse_b"].astype(float) + 1e-16)
df = df[df["method"].isin(METHODS) & (df["sample_strategy"] == STRATEGY)]

sample_size_vals = sorted(df["sample_size"].astype(int).unique().tolist())
palette = {
    "NW": "#1f77b4",
    "CE": "#d62728",
    "CL": "#ff7f0e",
    "LL": "#2ca02c",
}  # red, orange, blue

# %% ========================================================================
# 				plot (violin / box-shaped)
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=800)
sns.violinplot(
    data=df,
    x="sample_size",
    y="log10_mse_b",
    hue="method",
    hue_order=METHODS,
    order=sample_size_vals,
    palette=palette,
    cut=0,
    inner="quartile",
    linewidth=1.0,
    ax=ax,
)

# The n values are evenly spaced in log2 (each +1), so the categorical
# violin positions already coincide with the log2 scale.
ax.set_xticks(range(len(sample_size_vals)))
ax.set_xticklabels([f"{int(n)}\n({np.log2(n):.1f})" for n in sample_size_vals])
ax.set_xlabel(r"Sample size $n$ ($\log_2(n)$)")
ax.set_ylabel(r"$\log_{10}(\mathrm{Boundary\ MSE})$")
# ax.set_title("Boundary MSE distribution (CE vs CL, S2)")
ax.legend(title="Method", frameon=False, loc="upper right")


def _style_ax(ax):
    """Remove top/right spines for a clean publication look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


_style_ax(ax)
fig.tight_layout()
fig.savefig(out_dir, dpi=800, bbox_inches="tight")
plt.show()
print("Saved:", out_dir)
