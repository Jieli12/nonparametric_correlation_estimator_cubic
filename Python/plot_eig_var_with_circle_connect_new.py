"""
Eigenvalue-variance and Yeo-7 chord figures for the v2 MEG analysis (R3.7).

Same layout as plot_eig_var_with_circle_connect_new.py (line plot with chord
diagrams at six time points), with the conventional counterparts added to the
line plot:
  - CL (main analysis): eigenvalue variance of the corrected matrix,
    (p - 1) * mean squared off-diagonal correlation (network_<band>_local.npz);
  - SW (tuned sliding window): the same summary (network_<band>_local_SW.npz);
  - static PEC: the same summary of the static correlation matrix of the
    log-envelopes (horizontal line).
Chord diagrams show the CL Yeo-7 network matrix at the six time points; as in
the original figures, links below the median are hidden for display only.

Usage:
    python plot_eig_var_circle_v2.py [--bands ...] [--out_dir ../Latex/MDPI_template_ACS/figs]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from plot_eig_var_with_circle_connect_new import (  # noqa: E402
    BANDS,
    BANDS_DISPLAY,
    TIME_STAMP,
    YEO7_CMAP,
    YEO7_CMAP_SHORT,
    YEO7_SHORT,
    data_to_fig,
    draw_chord,
)

FS_ENV = 10.0


def static_eig_var(env_file):
    y = np.load(env_file)["log_power"].astype(np.float64)
    R = np.corrcoef(y)
    p = R.shape[0]
    return (p - 1) * np.mean(R[np.triu_indices(p, 1)] ** 2)


def plot_band(band, display, runs, env_dir, out_dir):
    cl = np.load(runs / f"network_{band}_local.npz")
    y = cl["eigenvalue_variance_equivalent"]
    x = cl["times"]
    yeo = cl["yeo7"]
    f_sw = runs / f"network_{band}_local_SW.npz"
    y_sw = np.load(f_sw)["eigenvalue_variance_equivalent"] if f_sw.exists() else None
    y_static = static_eig_var(Path(env_dir) / f"power_envelope_v2_{band}.npz")

    fig = plt.figure(figsize=(20, 9.5), dpi=300)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"{display} band - eigenvalue variance & Yeo-7 network connectivity",
        fontsize=20,
        y=0.985,
    )
    main_ax = fig.add_axes([0.27, 0.27, 0.5, 0.26])
    allv = np.concatenate([y, [y_static]] + ([y_sw] if y_sw is not None else []))
    pad = 0.05 * np.ptp(allv) if np.ptp(allv) > 0 else 1.0
    main_ax.set_xlim(x[0], x[-1])
    main_ax.set_ylim(allv.min() - pad, allv.max() + pad)
    main_ax.plot(x, y, color="#2a78d6", lw=3, zorder=3, label="CL")
    if y_sw is not None:
        main_ax.plot(
            x, y_sw, color="#eb6834", lw=1.5, ls="--", zorder=2, label="SW (tuned)"
        )
    main_ax.axhline(
        y_static, color="#52514e", lw=1.5, ls=":", zorder=2, label="static PEC"
    )
    marks = [t - 1 for t in TIME_STAMP if t - 1 < len(y)]
    for t in marks:
        main_ax.axvline(x[t], color="#b0b0b0", ls="--", lw=1.0, zorder=1)
    main_ax.scatter(
        [x[t] for t in marks],
        [y[t] for t in marks],
        s=48,
        color="#2a78d6",
        edgecolors="black",
        linewidths=1.1,
        zorder=6,
    )
    main_ax.set_xticks([x[t] for t in marks], [f"{x[t] / FS_ENV:.0f} s" for t in marks])
    main_ax.set_xlabel("Time", fontsize=18)
    main_ax.set_ylabel("Variance of eigenvalues", fontsize=18)
    main_ax.tick_params(labelsize=16)
    main_ax.grid(alpha=0.3, ls="--", lw=0.6, zorder=0)
    main_ax.legend(
        fontsize=13,
        frameon=False,
        loc="upper right",
        ncol=3,
        bbox_to_anchor=(1.0, 1.18),
    )

    chord_cy_top, chord_cy_bottom, chord_h = 0.71, 0.03, 0.144
    chord_w = chord_h * (fig.get_figheight() / fig.get_figwidth())
    col_centers = np.linspace(0.32, 0.68, 3)
    for k, t in enumerate(marks):
        start = data_to_fig(main_ax, fig, x[t], y[t])
        cx = float(col_centers[k // 2])
        is_top = k % 2 == 0
        cy = chord_cy_top if is_top else chord_cy_bottom
        inset = fig.add_axes(
            [cx - chord_w, cy - chord_h, 2 * chord_w, 2 * chord_h], projection="polar"
        )
        draw_chord(inset, yeo[:, :, t], YEO7_CMAP_SHORT, YEO7_SHORT)
        fig.text(
            cx,
            cy + chord_h + 0.018 if is_top else cy - chord_h - 0.016,
            f"t = {x[t] / FS_ENV:.0f} s",
            ha="center",
            va="bottom" if is_top else "top",
            fontsize=16,
            color="#444444",
        )
        end = (cx, cy - 0.95 * chord_h) if is_top else (cx, cy + 0.95 * chord_h)
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
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c,
            markersize=14,
            label=n,
            markeredgecolor="none",
        )
        for n, c in YEO7_CMAP.items()
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
    out = Path(out_dir) / f"eig_var_circle_connect_{display.replace(' ', '_')}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    msg = f"{display}: CL eigenvalue variance median {np.median(y):.2f} [{y.min():.2f}, {y.max():.2f}]; static {y_static:.2f}"
    if y_sw is not None:
        msg += f"; SW median {np.median(y_sw):.2f}; corr(CL, SW) {np.corrcoef(y, y_sw)[0, 1]:.2f}"
    print(msg + f" -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bands", nargs="+", default=BANDS)
    ap.add_argument("--runs_dir", default="../Data/v2/runs")
    ap.add_argument("--env_dir", default="../Data/v2")
    ap.add_argument("--out_dir", default="../Latex/MDPI_template_ACS/figs")
    args = ap.parse_args()
    disp = dict(
        zip(BANDS, [d.capitalize() if " " not in d else d for d in BANDS_DISPLAY])
    )
    disp = {b: d[0].upper() + d[1:] for b, d in disp.items()}
    for band in args.bands:
        plot_band(band, disp[band], Path(args.runs_dir), args.env_dir, args.out_dir)


if __name__ == "__main__":
    main()
