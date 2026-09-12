"""
Author      : Jie Li
Date        : 2026-08-21 12:09:07
Last Edited : 2026-09-11 14:33:11
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/plot_time_sns.py
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

# Total time = bandwidth search + u_test evaluation
full_df["total_time"] = full_df["time_bandwidth"].fillna(0) + full_df[
    "time_u_test"
].fillna(0)

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
# 			1) time heatmap
# ===========================================================================
row_index = pd.MultiIndex.from_product(
    [methods, strategy_vals, n_vals], names=["method", "sample_strategy", "sample_size"]
)
col_index = pd.MultiIndex.from_product(
    [corr_vals, kernel_vals], names=["corr_fun", "kernel"]
)

time_matrix = (
    full_df.groupby(
        ["method", "sample_strategy", "sample_size", "corr_fun", "kernel"],
        observed=True,
    )["total_time"]
    .median()
    .unstack(["corr_fun", "kernel"])
    .reindex(index=row_index, columns=col_index)
)
time_matrix = time_matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")

# Row / column annotations from the MultiIndex levels
row_annot = pd.DataFrame(
    {
        "Method": [str(x) for x in time_matrix.index.get_level_values(0)],
        "Strategy": [str(x) for x in time_matrix.index.get_level_values(1)],
        "SampleSize": [int(x) for x in time_matrix.index.get_level_values(2)],
    },
    index=time_matrix.index,
)
col_annot = pd.DataFrame(
    {
        "Corr": [str(x) for x in time_matrix.columns.get_level_values(0)],
        "Kernel": [str(x) for x in time_matrix.columns.get_level_values(1)],
    },
    index=time_matrix.columns,
)

# Pretty labels for heatmap axes
time_matrix.index = [f"{m} | {s} | n={int(n)}" for m, s, n in time_matrix.index]
time_matrix.columns = [f"{a} | {b}" for a, b in time_matrix.columns]
row_annot.index = time_matrix.index
col_annot.index = time_matrix.columns

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
    index=time_matrix.index,
)
col_colors = pd.DataFrame(
    {
        "Correlation Function": col_annot["Corr"].map(CORR_COLORS),
        "Kernel Function": col_annot["Kernel"].map(KERNEL_COLORS),
    },
    index=time_matrix.columns,
)

g2 = sns.clustermap(
    time_matrix,
    row_cluster=False,
    col_cluster=False,
    row_colors=row_colors,
    col_colors=col_colors,
    cmap="crest",
    dendrogram_ratio=(0.01, 0.01),  # minimize empty space for the (hidden) dendrograms
    cbar_kws={
        "label": "Median time (s)",
        "shrink": 0.85,
        "aspect": 25,
    },  # raw seconds, NOT log10
    xticklabels=True,
    yticklabels=True,
    figsize=(18, 12),
)
g2.ax_heatmap.set_xlabel(r"Correlation function $\times$ Kernel function", fontsize=12)
g2.ax_heatmap.set_ylabel(
    r"Method $\times$ Sampling Strategy $\times$ Sample Size", fontsize=12
)
g2.ax_heatmap.set_title(
    "Median total running time across repeats (seconds)", fontsize=12
)
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
out_heat = Path("../figs/panel_time_heatmap.png")
out_heat.parent.mkdir(parents=True, exist_ok=True)
g2.savefig(out_heat, dpi=800, bbox_inches="tight")
plt.show()
print("Saved:", out_heat)

# %% ========================================================================
# 		2) Running time vs. sample size
# ===========================================================================
agg_df = (
    full_df.groupby(["method", "sample_size", "sample_strategy"], observed=True)[
        "total_time"
    ]
    .agg(
        median="median",
        q025=lambda x: np.percentile(x, 2.5),
        q975=lambda x: np.percentile(x, 97.5),
    )
    .reset_index()
)
agg_df["sample_size_num"] = agg_df["sample_size"].astype(int)

# Broken y-axis: all methods share one log2 x-axis, while the y-axis is
# split into a lower segment (fast methods NW/CE/CL, ~0.4-3 s) and an
# upper segment (slow method LL, ~35 s), so every method stays readable
# on the same figure. S1 = solid, S2 = dashed; the shaded-band boundaries
# (q025/q975) follow the same solid/dashed style.
fast = agg_df[agg_df["method"] != "LL"]
slow = agg_df[agg_df["method"] == "LL"]
y_bottom_top = float(fast["q975"].max()) * 1  # upper limit of the lower segment
y_top_bottom = float(slow["median"].min()) * 0.60  # lower limit of the upper segment
y_top_top = float(slow["q975"].max()) * 1  # upper limit of the upper segment

fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, figsize=(10, 8), dpi=800, sharex=True, gridspec_kw={"height_ratios": [1, 1.2]}
)

for m in methods:
    for s in strategy_vals:
        sub = agg_df[
            (agg_df["method"] == m) & (agg_df["sample_strategy"] == s)
        ].sort_values("sample_size_num")
        color = METHOD_COLORS.get(m, "#333333")
        ls = "-" if s == strategy_vals[0] else "--"
        for ax in (ax_top, ax_bottom):
            # median line: solid (S1) -> square marker, dashed (S2) -> star marker
            ax.plot(
                sub["sample_size_num"],
                sub["median"],
                color=color,
                linestyle=ls,
                marker="s" if s == strategy_vals[0] else "*",
                lw=3,
                label=f"{m}, {s}",
                markersize=12,
            )
            # fill-band boundaries follow the line style (solid / dashed)
            ax.plot(
                sub["sample_size_num"],
                sub["q025"],
                color=color,
                linestyle=ls,
                lw=1.5,
                alpha=0.55,
            )
            ax.plot(
                sub["sample_size_num"],
                sub["q975"],
                color=color,
                linestyle=ls,
                lw=1.5,
                alpha=0.55,
            )
            ax.fill_between(
                sub["sample_size_num"],
                sub["q025"],
                sub["q975"],
                color=color,
                alpha=0.15,
            )

for ax in (ax_top, ax_bottom):
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.spines["right"].set_visible(False)

ax_top.set_ylim(y_top_bottom, y_top_top)
ax_bottom.set_ylim(0.0, y_bottom_top)

# broken-axis cosmetics
ax_top.spines["bottom"].set_visible(False)
ax_bottom.spines["top"].set_visible(False)
ax_top.tick_params(labeltop=False, top=False, labelsize=12)
ax_bottom.tick_params(top=False, labelsize=12)
# diagonal break marks where the two segments meet
d = 0.5
kwargs_break = dict(
    marker=[(-1, -d), (1, d)],
    markersize=10,
    linestyle="none",
    color="k",
    mec="k",
    mew=1,
    clip_on=False,
)
ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs_break)
ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs_break)
# Place ticks exactly at the sample sizes so the labels line up with the data
# (x-data are the n values plotted on a log2 axis).
ax_bottom.set_xticks(n_vals)
ax_bottom.set_xticklabels([f"{int(n)}\n({np.log2(n):.1f})" for n in n_vals])
ax_bottom.set_xlabel(r"Sample size $n$ ($\log_2(n)$)")
fig.supylabel("Median total time (s)", fontsize=14)
# fig.suptitle(
#     "Running time vs. Sample Size (broken y-axis; S1 = solid, S2 = dashed)", fontsize=12
# )
ax_bottom.legend(
    loc="upper left",
    frameon=False,
    ncol=1,
    title="Method, Strategy",
    handlelength=4.0,
    fontsize=12,
)
fig.tight_layout()
out_line = Path("../figs/panel_time_vs_sample_size.png")
fig.savefig(out_line, dpi=800, bbox_inches="tight")
plt.show()
print("Saved:", out_line)

# %%
# Accuracy-speed trade-off (median MSE vs median total time).
# Each method is a line with its points (colored by method); every point is
# annotated with its sample size (e.g. "n=200"); legend order: NW, CE, LL, CL.
trade = (
    full_df.groupby(["method", "sample_size"], observed=True)
    .agg(mse_med=("mse", "median"), time_med=("total_time", "median"))
    .reset_index()
)
trade["sample_size"] = trade["sample_size"].astype(int)

legend_order = ["NW", "CE", "LL", "CL"]

fig, axD = plt.subplots(figsize=(10, 8), dpi=800)
for m in legend_order:
    sub = trade[trade["method"] == m].sort_values("sample_size")
    axD.plot(
        sub["time_med"],
        sub["mse_med"],
        "-o",
        color=METHOD_COLORS.get(m, "#333333"),
        lw=4,
        label=m,
        markersize=10,
    )
    for _, r in sub.iterrows():
        axD.annotate(
            f"n={int(r['sample_size'])}",
            (r["time_med"], r["mse_med"]),
            fontsize=12,
            xytext=(4, 4),
            textcoords="offset points",
        )

axD.set_xscale("log")
axD.set_yscale("log")
axD.tick_params(axis="both", which="major", labelsize=12)
axD.set_xlim(left=trade["time_med"].min() * 0.8, right=trade["time_med"].max() * 1.8)
# axD.set_title("Accuracy-speed trade-off")
axD.set_xlabel("Median total time (s)", fontsize=14)
axD.set_ylabel("Median MSE", fontsize=14)
axD.legend(
    title="Method",
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(0.7, 1),
    handlelength=4.0,
    fontsize=12,
)
axD.spines["top"].set_visible(False)
axD.spines["right"].set_visible(False)
fig.tight_layout()
out_trade = Path("../figs/panel_accuracy_speed_tradeoff.png")
out_trade.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_trade, dpi=800, bbox_inches="tight")
plt.show()
print("Saved:", out_trade)

# %%
