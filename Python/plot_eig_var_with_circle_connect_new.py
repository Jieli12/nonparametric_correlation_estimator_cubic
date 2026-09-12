"""
Author      : Jie Li
Date        : 2026-09-11 22:51:45
Last Edited : 2026-09-12 09:30:20
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/plot_eig_var_with_circle_connect_new.py
Description :

Plot eigenvalue variance and Yeo-7 network connectivity from separate band files
Generate individual PNG plots for each band






Copyright (c) 2026, Jie Li, lijie_12@outlook.com
All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from pycirclize import Circos

YEO7_NAMES = [
    "Visual",
    "Somatomotor",
    "Dorsal Attention",
    "Ventral Attention",
    "Limbic",
    "Frontoparietal",
    "Default",
]
YEO7_SHORT = ["Visual", "SomMot", "DAN", "VAN", "Limbic", "FPN", "DMN"]

# canonical Yeo-7 colors (visual/somatomotor/dorsattn/salventattn/limbic/cont/default)
YEO7_CMAP = {
    "Visual": "#781286",
    "Somatomotor": "#4682B4",
    "Dorsal Attention": "#008000",
    "Ventral Attention": "#C71585",
    "Limbic": "#D8BFD8",
    "Frontoparietal": "#E69F00",
    "Default": "#CD5C5C",
}
YEO7_CMAP_SHORT = {s: YEO7_CMAP[n] for s, n in zip(YEO7_SHORT, YEO7_NAMES)}

BANDS = ["delta", "theta", "alpha", "beta", "slow_gamma", "fast_gamma"]
BANDS_DISPLAY = ["Delta", "Theta", "Alpha", "Beta", "Slow gamma", "Fast gamma"]
TIME_STAMP = [100, 1100, 2100, 3100, 4100, 5100]
NUM_OBS = 5740


def data_to_fig(ax, fig, x, y):
    """Convert a point in axes-data coordinates to figure-fraction coordinates."""
    disp = ax.transData.transform((x, y))
    return fig.transFigure.inverted().transform(disp)


def draw_chord(ax, matrix, cmap, names):
    """Render one Yeo-7 chord diagram into an existing polar `ax`."""
    m = np.array(matrix, dtype=float)
    # suppress within-network (diagonal) links
    np.fill_diagonal(m, 0.0)
    # keep only connections above the median strength
    m[np.abs(m) < np.median(m)] = 0.0

    network_df = pd.DataFrame(m, index=names, columns=names)
    circos = Circos.chord_diagram(
        network_df,
        space=5,
        cmap=cmap,
        label_kws=dict(size=14, color="#222222"),
        link_kws=dict(ec="black", lw=0.4, direction=0),
    )
    circos.plotfig(ax=ax)  # draw into our inset axes (no new figure is created)


def plot_band(band_name, band_display_name, save_dir="../figs"):
    """Generate plot for a single band."""
    print(f"\n [Processing] {band_display_name} band...")

    # Load data
    data_file = f"../Results/eigvar_yeo7_network_{band_name}.npz"
    data = np.load(data_file, allow_pickle=True)
    eig_vars = data["eig_vars"]  # (5740,)
    yeo7_networks = data["yeo7_networks"]  # (7, 7, 5740)

    y = eig_vars
    x = np.arange(NUM_OBS)

    fig = plt.figure(figsize=(20, 9.5), dpi=800)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"{band_display_name} band - eigenvalue variance & Yeo-7 network connectivity",
        fontsize=20,
        y=0.985,
    )

    # ----------------------------------------------------------------
    #  1) main line axes (lower part of the figure)
    # ----------------------------------------------------------------
    # Keep center fixed (0.5) and shorten width to 3/4 of current (0.88 -> 0.66).
    main_ax = fig.add_axes([0.27, 0.27, 0.5, 0.26])
    main_ax.set_xlim(0, NUM_OBS - 1)
    y_pad = 0.05 * np.ptp(y)
    main_ax.set_ylim(y.min() - y_pad, y.max() + y_pad)

    main_ax.plot(x, y, color="#1f77b4", lw=4, zorder=3)
    main_ax.fill_between(x, y, y.min() - y_pad, color="#1f77b4", alpha=0.12, zorder=1)

    for t in TIME_STAMP:
        main_ax.axvline(t - 1, color="#b0b0b0", ls="--", lw=1.0, zorder=2)

    main_ax.scatter(
        [t - 1 for t in TIME_STAMP],
        [y[t - 1] for t in TIME_STAMP],
        s=48,
        color="#1f77b4",
        edgecolors="black",
        linewidths=1.1,
        zorder=6,
    )

    main_ax.set_xticks([t - 1 for t in TIME_STAMP])
    main_ax.set_xticklabels([str(t) for t in TIME_STAMP])
    main_ax.set_xlabel("Time (sample index)", fontsize=18)
    main_ax.set_ylabel("Variance of eigenvalues", fontsize=18)
    main_ax.tick_params(labelsize=18)
    main_ax.grid(alpha=0.3, ls="--", lw=0.6, zorder=0)

    # ----------------------------------------------------------------
    #  2) chord-diagram insets (one per time stamp, aligned to its x)
    # ----------------------------------------------------------------
    chord_cy_top, chord_cy_bottom, chord_h = 0.71, 0.03, 0.144
    # square bbox in inches (fig-fraction x/y scale differently)
    chord_w = chord_h * (fig.get_figheight() / fig.get_figwidth())
    col_centers = np.linspace(0.32, 0.68, 3)

    start_list, cx_list, insets, is_top_list = [], [], [], []
    for k, t in enumerate(TIME_STAMP):
        start = data_to_fig(main_ax, fig, t - 1, y[t - 1])
        cx = float(col_centers[k // 2])
        is_top = k % 2 == 0  # 1st/3rd/5th above, 2nd/4th/6th below
        chord_cy = chord_cy_top if is_top else chord_cy_bottom
        start_list.append(start)
        cx_list.append(cx)
        is_top_list.append(is_top)
        insets.append(
            fig.add_axes(
                [cx - chord_w, chord_cy - chord_h, 2 * chord_w, 2 * chord_h],
                projection="polar",
            )
        )
        if is_top:
            # time label above top-row circles
            fig.text(
                cx,
                chord_cy + chord_h + 0.018,
                f"t = {t}",
                ha="center",
                va="bottom",
                fontsize=16,
                color="#444444",
            )
        else:
            # time label below bottom-row circles
            fig.text(
                cx,
                chord_cy - chord_h - 0.016,
                f"t = {t}",
                ha="center",
                va="top",
                fontsize=16,
                color="#444444",
            )

    # Get time stamp indices for network matrices
    for k, (inset, t) in enumerate(zip(insets, TIME_STAMP)):
        # Find the index of t in time_stamp
        time_idx = TIME_STAMP.index(t)
        mat = yeo7_networks[:, :, t - 1]  # (7, 7) at time t
        draw_chord(inset, mat, YEO7_CMAP_SHORT, YEO7_SHORT)

    # ----------------------------------------------------------------
    #  3) arrows from each marked point to its chord diagram
    # ----------------------------------------------------------------
    for start, cx, is_top in zip(start_list, cx_list, is_top_list):
        chord_cy = chord_cy_top if is_top else chord_cy_bottom
        end = (
            (cx, chord_cy - 0.95 * chord_h)
            if is_top
            else (cx, chord_cy + 0.95 * chord_h)
        )
        fig.patches.append(
            FancyArrowPatch(
                posA=start,
                posB=end,
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=24,
                lw=1.6,
                color="#555555",
                shrinkA=2,
                shrinkB=4,
                zorder=4,
            )
        )

    # ----------------------------------------------------------------
    #  4) shared Yeo-7 network legend (full names + canonical colors)
    # ----------------------------------------------------------------
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c,
            markersize=14,
            label=name,
            markeredgecolor="none",
        )
        for name, c in YEO7_CMAP.items()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=7,
        frameon=False,
        fontsize=14,
        handletextpad=0.2,
        columnspacing=0.2,
        bbox_to_anchor=(0.5, 0.95),
    )

    # Save figure
    save_path = f"{save_dir}/eig_var_circle_connect_{band_name}.png"
    fig.savefig(save_path, dpi=800, bbox_inches="tight", facecolor="white")
    print(f"[Saved] {save_path}")
    plt.close(fig)


def main():
    """Process all bands and generate plots."""
    print("[Start] Generating plots for all bands...\n")

    for band_name, band_display_name in zip(BANDS, BANDS_DISPLAY):
        try:
            plot_band(band_name, band_display_name)
        except Exception as e:
            print(f"[Error] Failed to process {band_display_name} band.")

    print("\n[Done] All bands processed!")


if __name__ == "__main__":
    main()
