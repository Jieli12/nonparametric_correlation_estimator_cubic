"""
Author      : Jie Li
Date        : 2026-08-16 20:50:11
Last Edited : 2026-09-11 14:18:49
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/plot_sample_size_vs_mse.py
Description :

Density (KDE) curves of log10(MSE) vs sample size (log2 x-axis).

For each sample size n, 8 normal-like density curves are drawn
(4 methods x 2 sample strategies) showing the distribution of
log10(MSE) across corr_fun x kernel x 200 replications.
Color = method; S1 = solid curve, S2 = dashed curve; dot = median.


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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# %% ========================================================================
# 				load the data
# ===========================================================================
input_csv = Path("../Results/combined_results_summary.csv")
full_df = pd.read_csv(input_csv)
out_dir = "../figs/sample_size_vs_mse.png"
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

method_vals = full_df["method"].astype(str).dropna().unique().tolist()
sample_size_vals = sorted(full_df["sample_size"].dropna().unique().tolist())

dfw = full_df.copy()
dfw["log10_mse"] = np.log10(dfw["mse"].astype(float) + 1e-16)
strategy_vals = sorted(dfw["sample_strategy"].dropna().astype(str).unique().tolist())


base_palette = {"NW": "#1f77b4", "LL": "#2ca02c", "CL": "#ff7f0e", "CE": "#d62728"}
palette = {
    m: base_palette.get(m, c)
    for m, c in zip(method_vals, sns.color_palette("tab10", len(method_vals)))
}


# %% ========================================================================
# 				define KDE function
# ===========================================================================
def _kde(y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """1-D Gaussian KDE (Silverman bandwidth) evaluated on ``grid``."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return np.full_like(grid, np.nan)
    std = y.std(ddof=1)
    bw = std * (4.0 / (3.0 * y.size)) ** 0.2
    if not np.isfinite(bw) or bw <= 0:
        bw = 1e-6
    d = np.exp(-0.5 * ((grid[:, None] - y[None, :]) / bw) ** 2).sum(axis=1)
    norm = d.sum() * (grid[1] - grid[0])
    return d / norm if norm > 0 else np.full_like(grid, np.nan)


# %% ========================================================================
#
# ===========================================================================
n_curves = len(method_vals) * len(strategy_vals)
# Keep the 8 curves of the SAME n tightly clustered (small step),
# while group centers stay on the log2 axis, so different-n gaps look large.
step = 0.1  # spacing between adjacent curve baselines within the same n
amp = 0.15  # max horizontal deviation (density -> x)
grid_pts = 100
fig, ax = plt.subplots(figsize=(14, 6), dpi=800)
for n in sample_size_vals:
    center = np.log2(n)
    b = 0
    for m in method_vals:
        for s in strategy_vals:
            sub = dfw[
                (dfw["sample_size"] == n)
                & (dfw["method"] == m)
                & (dfw["sample_strategy"] == s)
            ]
            y = sub["log10_mse"].dropna().to_numpy(dtype=float)
            b += 1
            if y.size < 2:
                continue
            lo, hi = np.percentile(y, [0.5, 99.5])
            pad = (hi - lo) * 0.10
            grid = np.linspace(lo - pad, hi + pad, grid_pts)
            dens = _kde(y, grid)
            offset = (b - 1 - (n_curves - 1) / 2) * step
            xs = center + offset + amp * dens
            ax.plot(
                xs,
                grid,
                color=palette[m],
                linestyle="solid" if s == strategy_vals[0] else "dashed",
                lw=2,
            )
            ax.scatter(
                [center + offset],
                [float(np.median(y))],
                color=palette[m],
                s=14,
                zorder=3,
            )
    ax.axvline(center, color="gray", lw=0.6, ls=":", alpha=0.7)

tick_pos = [np.log2(n) for n in sample_size_vals]
ax.set_xticks(tick_pos)
ax.set_xticklabels([f"{int(n)}\n({np.log2(n):.1f})" for n in sample_size_vals])
ax.set_xlabel(r"Sample size $n$ ($\log_2(n)$)")
ax.set_ylabel(r"$\log_{10}(\mathrm{MSE})$")
ax.set_xlim(min(tick_pos) - 0.5, max(tick_pos) + 0.8)
ax.set_ylim(dfw["log10_mse"].min() + 0.95, dfw["log10_mse"].max() + 0.05)

handles = [
    Patch(facecolor=palette[m], alpha=0.6, edgecolor=palette[m], label=m)
    for m in method_vals
]
handles += [
    Line2D(
        [0], [0], color="black", linestyle="solid", label=f"{strategy_vals[0]} (solid)"
    ),
    Line2D(
        [0],
        [0],
        color="black",
        linestyle="dashed",
        label=f"{strategy_vals[1]} (dashed)",
    ),
]
ax.legend(
    handles=handles, frameon=False, ncol=2, loc="upper right", title="Method / strategy"
)


def _style_ax(ax):
    """Remove top/right spines for a clean publication look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


_style_ax(ax)
fig.tight_layout()
fig.savefig(out_dir, dpi=800, bbox_inches="tight")
# plt.show()
# plt.close(fig)
print("Saved:", out_dir)
