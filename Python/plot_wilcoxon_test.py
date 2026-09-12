"""
Author      : Jie Li
Date        : 2026-09-09 10:32:16
Last Edited : 2026-09-11 15:30:47
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/plot_wilcoxon_test.py
Description :

Read ../results/combined_results_summary.csv once, then run paired Wilcoxon
signed-rank tests for CL vs LL at each
(corr_fun, kernel, sample_strategy, sample_size) cell using rep_idx pairing.

No argparse is needed. Just run:
    python plot_wilcoxon_test.py


Copyright (c) 2026, Jie Li, lijie_12@outlook.com
All Rights Reserved.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wilcoxon

ALPHA = 0.05
METHOD_CL = "CL"
METHOD_LL = "LL"

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = (BASE_DIR / "../Results/combined_results_summary.csv").resolve()
OUTPUT_DIR = (BASE_DIR / "../figs").resolve()


def load_and_group_pairs(csv_path: Path):
    """Load combined CSV once and build paired MSE containers."""
    grouped = defaultdict(lambda: {METHOD_CL: {}, METHOD_LL: {}})

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "corr_fun",
            "kernel",
            "sample_strategy",
            "method",
            "sample_size",
            "rep_idx",
            "mse",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

        for row in reader:
            method = row["method"].strip()
            if method not in (METHOD_CL, METHOD_LL):
                continue

            try:
                sample_size = int(row["sample_size"])
                rep_idx = int(row["rep_idx"])
                mse = float(row["mse"])
            except (TypeError, ValueError):
                continue

            key = (
                row["corr_fun"].strip(),
                row["kernel"].strip(),
                row["sample_strategy"].strip(),
                sample_size,
            )
            grouped[key][method][rep_idx] = mse

    return grouped


def run_wilcoxon_tests(grouped):
    """Run paired Wilcoxon tests per (corr, kernel, strategy, n)."""
    rows = []

    for key in sorted(grouped.keys()):
        corr_fun, kernel, sample_strategy, sample_size = key
        cl_map = grouped[key][METHOD_CL]
        ll_map = grouped[key][METHOD_LL]

        paired_rep_idx = sorted(set(cl_map.keys()) & set(ll_map.keys()))
        if not paired_rep_idx:
            continue

        cl = np.array([cl_map[i] for i in paired_rep_idx], dtype=float)
        ll = np.array([ll_map[i] for i in paired_rep_idx], dtype=float)

        valid = ~np.isnan(cl) & ~np.isnan(ll)
        cl = cl[valid]
        ll = ll[valid]
        n_pairs = int(cl.size)

        if n_pairs == 0:
            statistic = np.nan
            pvalue = np.nan
        else:
            # One-sided alternative: CL has smaller MSE than LL
            try:
                res = wilcoxon(cl, ll, alternative="less", zero_method="wilcox")
                statistic = float(res.statistic)
                pvalue = float(res.pvalue)
            except ValueError:
                statistic = np.nan
                pvalue = np.nan

        mean_diff = float(np.mean(cl - ll)) if n_pairs > 0 else np.nan
        median_diff = float(np.median(cl - ll)) if n_pairs > 0 else np.nan

        rows.append(
            {
                "corr_fun": corr_fun,
                "kernel": kernel,
                "sample_strategy": sample_strategy,
                "sample_size": sample_size,
                "n_pairs": n_pairs,
                "statistic": statistic,
                "pvalue": pvalue,
                "cl_mean_mse": float(np.mean(cl)) if n_pairs > 0 else np.nan,
                "ll_mean_mse": float(np.mean(ll)) if n_pairs > 0 else np.nan,
                "cl_std_mse": float(np.std(cl)) if n_pairs > 0 else np.nan,
                "ll_std_mse": float(np.std(ll)) if n_pairs > 0 else np.nan,
                "mean_diff_cl_minus_ll": mean_diff,
                "median_diff_cl_minus_ll": median_diff,
                "cl_better": (
                    bool((pvalue < ALPHA) and (mean_diff < 0))
                    if not np.isnan(pvalue)
                    else False
                ),
            }
        )

    return rows


def save_summary_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bh_fdr_decisions(pvalues: np.ndarray, q: float = 0.05):
    """Return BH-FDR rejection mask and threshold for a 1D p-value array."""
    p = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(p)
    rej = np.zeros(p.shape, dtype=bool)

    if not np.any(valid):
        return rej, np.nan

    pv = p[valid]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    crit = q * (np.arange(1, m + 1) / m)
    ok = np.where(ranked <= crit)[0]

    if ok.size == 0:
        return rej, np.nan

    k = int(ok.max()) + 1
    threshold = q * (k / m)
    local_rej = pv <= threshold
    rej[valid] = local_rej
    return rej, float(threshold)


def print_multiple_testing_robustness(rows, q: float = 0.05):
    """Print BH-FDR robustness summary globally and by sampling strategy."""
    if not rows:
        print("[BH-FDR] no rows to analyze")
        return

    pvals = np.array([r.get("pvalue", np.nan) for r in rows], dtype=float)
    mean_diff = np.array(
        [r.get("mean_diff_cl_minus_ll", np.nan) for r in rows], dtype=float
    )
    strategies = [r.get("sample_strategy", "") for r in rows]

    rej, thr = bh_fdr_decisions(pvals, q=q)
    total = int(np.isfinite(pvals).sum())
    n_rej = int(rej.sum())
    cl_disc = int(np.sum(rej & np.isfinite(mean_diff) & (mean_diff < 0)))
    ll_disc = int(np.sum(rej & np.isfinite(mean_diff) & (mean_diff > 0)))

    print("\n[Multiple testing robustness] Benjamini-Hochberg FDR")
    print(f"  q={q}")
    print(f"  discoveries: {n_rej}/{total}")
    print(f"  BH threshold: {thr if np.isfinite(thr) else 'none'}")
    print(f"  CL-favoring discoveries: {cl_disc}")
    print(f"  LL-favoring discoveries: {ll_disc}")

    unique_strategies = sorted(set(strategies))
    for s in unique_strategies:
        idx = np.array([x == s for x in strategies], dtype=bool)
        rej_s, thr_s = bh_fdr_decisions(pvals[idx], q=q)
        md_s = mean_diff[idx]
        total_s = int(np.isfinite(pvals[idx]).sum())
        n_rej_s = int(rej_s.sum())
        cl_disc_s = int(np.sum(rej_s & np.isfinite(md_s) & (md_s < 0)))
        ll_disc_s = int(np.sum(rej_s & np.isfinite(md_s) & (md_s > 0)))
        print(
            f"  {s}: discoveries={n_rej_s}/{total_s}, "
            f"BH threshold={thr_s if np.isfinite(thr_s) else 'none'}, "
            f"CL={cl_disc_s}, LL={ll_disc_s}"
        )


def plot_heatmaps(rows, output_dir: Path):
    """Create a single figure with one heatmap per sample_strategy panel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    strategies = sorted({r["sample_strategy"] for r in rows})
    if not strategies:
        return

    combos_global = sorted({(r["corr_fun"], r["kernel"]) for r in rows})
    sample_sizes_global = sorted({int(r["sample_size"]) for r in rows})
    combo_to_i = {c: i for i, c in enumerate(combos_global)}
    n_to_j = {n: j for j, n in enumerate(sample_sizes_global)}

    mats = {}
    all_finite = []
    for strategy in strategies:
        subset = [r for r in rows if r["sample_strategy"] == strategy]
        mat = np.full((len(sample_sizes_global), len(combos_global)), np.nan)

        for r in subset:
            i = n_to_j[int(r["sample_size"])]
            j = combo_to_i[(r["corr_fun"], r["kernel"])]
            p = r["pvalue"]
            md = r["mean_diff_cl_minus_ll"]
            if p is None or np.isnan(p):
                score = np.nan
            else:
                mag = -math.log10(max(float(p), 1e-300))
                score = mag if md < 0 else -mag
            mat[i, j] = score

        mats[strategy] = mat
        finite = mat[np.isfinite(mat)]
        if finite.size:
            all_finite.append(np.abs(finite))

    if all_finite:
        merged = np.concatenate(all_finite)
        vmax = float(np.nanpercentile(merged, 95))
    else:
        vmax = 1.0
    vmax = max(vmax, -math.log10(ALPHA))

    # Color design: vivid warm colors for CL wins, muted cool colors for LL wins.
    cmap = LinearSegmentedColormap.from_list(
        "cl_focus", ["#2f4b7c", "#f7f4ea", "#f6aa1c", "#b2182b"], N=256
    )

    n_panels = len(strategies)
    # Arrange panels vertically (n_panels x 1) for better x-axis readability.
    # Increase width and reduce height slightly (minimal change).
    fig_w = max(14, 6.5 * n_panels)
    fig_h = max(5.0, 0.24 * len(sample_sizes_global) + 2.5)
    fig, axes = plt.subplots(n_panels, 1, figsize=(fig_w, fig_h), sharex=True)
    if n_panels == 1:
        axes = [axes]
    else:
        axes = list(axes)

    # S1/S2 style panel labels
    strategy_alias = {s: f"S{i + 1}" for i, s in enumerate(strategies)}

    im = None
    for ax, strategy in zip(axes, strategies):
        mat = mats[strategy]
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

        ax.set_xticks(np.arange(len(combos_global)))
        # Only show x tick labels on the bottom axis to avoid clutter.
        if ax is axes[-1]:
            ax.set_xticklabels(
                [f"{c} | {k}" for c, k in combos_global], rotation=90, fontsize=14
            )
        else:
            ax.set_xticklabels([])
        ax.set_yticks(np.arange(len(sample_sizes_global)))
        ax.set_yticklabels(sample_sizes_global)

        # Label style follows plot_mse_sns.py style.
        # Show x-label only on the bottom axis (shared x-axis)
        if ax is axes[-1]:
            ax.set_xlabel(r"Correlation function $\times$ Kernel function", fontsize=14)
        else:
            ax.set_xlabel("")
        ax.set_ylabel("Sample Size", fontsize=14)
        ax.set_title(
            f"{strategy_alias[strategy]} | Wilcoxon CL vs LL, score=sign(LL-CL)*-log10(p)",
            fontsize=14,
        )

    if im is not None:
        # Put colorbar in a dedicated axis outside the panel area.
        fig.subplots_adjust(right=0.9)
        cax = fig.add_axes([0.915, 0.14, 0.015, 0.74])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("positive: CL better, negative: LL better")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    out_png = output_dir / "wilcoxon_heatmap_s1_s2.png"
    fig.savefig(out_png, dpi=600)


def plot_significance_rate(rows, output_dir: Path):
    """Plot fraction of significant CL wins across combinations by sample size."""
    if not rows:
        return

    strategies = sorted({r["sample_strategy"] for r in rows})
    sample_sizes = sorted({int(r["sample_size"]) for r in rows})

    fig, ax = plt.subplots(figsize=(9, 5))

    for strategy in strategies:
        ys = []
        for n in sample_sizes:
            sub = [
                r
                for r in rows
                if r["sample_strategy"] == strategy and int(r["sample_size"]) == n
            ]
            if not sub:
                ys.append(np.nan)
                continue
            wins = sum(bool(r["cl_better"]) for r in sub)
            ys.append(wins / len(sub))
        ax.plot(sample_sizes, ys, marker="o", label=strategy)

    ax.set_xscale("log")
    ax.set_xticks(sample_sizes)
    ax.tick_params(axis="both", labelsize=14)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Sample Size", fontsize=14)
    ax.set_ylabel("Fraction of cells with significant CL better", fontsize=14)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    ax.set_title(f"CL vs LL Wilcoxon win rate (alpha={ALPHA})")

    plt.tight_layout()
    out_png = output_dir / "wilcoxon_significance_rate.png"
    fig.savefig(out_png, dpi=600)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    print(f"Loading CSV once: {INPUT_CSV}")
    grouped = load_and_group_pairs(INPUT_CSV)
    print(f"Grouped cells: {len(grouped)}")

    rows = run_wilcoxon_tests(grouped)
    print(f"Computed Wilcoxon tests: {len(rows)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = OUTPUT_DIR / "wilcoxon_cl_vs_ll_summary.csv"
    save_summary_csv(rows, out_csv)
    print(f"Saved summary CSV: {out_csv}")

    plot_heatmaps(rows, OUTPUT_DIR)
    plot_significance_rate(rows, OUTPUT_DIR)
    print_multiple_testing_robustness(rows, q=0.05)
    plt.show()
    print(f"Saved figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
