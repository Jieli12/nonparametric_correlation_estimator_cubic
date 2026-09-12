"""
Author      : Jie Li
Date        : 2026-09-11 00:24:25
Last Edited : 2026-09-12 09:35:23
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/plot_rhythmicity.py
Description :








Copyright (c) 2026, Jie Li, lijie_12@outlook.com
All Rights Reserved.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BANDS = ["delta", "theta", "alpha", "beta", "slow_gamma", "fast_gamma"]
BANDS_plot = ["Delta", "Theta", "Alpha", "Beta", "Slow Gamma", "Fast Gamma"]


def main():
    results_path = Path("../Results")
    output_path = Path("../figs")
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = results_path / "rhythmicity_summary.csv"
    if not summary_path.exists():
        print(f"Error: {summary_path} not found")
        return

    df_summary = pd.read_csv(summary_path)

    sns.set_style("whitegrid")
    df = df_summary.copy()

    if "ar_global_periodicity_q_fdr" in df.columns:
        q_col = "ar_global_periodicity_q_fdr"
        period_col = (
            "ar_global_period_s"
            if "ar_global_period_s" in df.columns
            else "peak_period_s"
        )
        summary_title = "AR-global periodicity significance"
    else:
        q_col = "surrogate_q_fdr" if "surrogate_q_fdr" in df.columns else "surrogate_p"
        period_col = "peak_period_s"
        summary_title = "Surrogate periodicity significance"

    # select significant entries (use a conventional threshold)
    sig = df[df[q_col] < 0.005]
    # count of significant pairs per band (line plot)
    cnt = sig.groupby("band").size().reindex(BANDS, fill_value=0)

    # prepare violin/boxplot data: list of arrays by band
    period_data = [
        sig[sig["band"] == band][period_col].dropna().values for band in BANDS
    ]

    # choose an elegant colormap/palette
    try:
        palette = sns.color_palette("mako", n_colors=len(BANDS))
    except Exception:
        palette = sns.color_palette("viridis", n_colors=len(BANDS))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: counts as a line plot (AR-global periodicity significance overview)
    axes[0].plot(range(len(BANDS)), cnt.values, marker="o", lw=2)
    axes[0].set_xticks(range(len(BANDS)))
    axes[0].set_xticklabels(BANDS_plot, rotation=30, ha="right", fontsize=14)
    axes[0].set_ylabel("# significant network pairs", fontsize=16)
    axes[0].tick_params(axis="y", labelsize=14)
    axes[0].tick_params(axis="x", labelsize=14)
    axes[0].set_title(f"{summary_title} (q<0.005)", fontsize=16)

    # Right: violin + boxplot of significant dominant periods by band
    ax = axes[1]
    # if there is no data for any band, draw empty axes with labels
    if any(len(v) for v in period_data):
        # construct DataFrame for seaborn plotting to preserve band order
        plot_df = pd.concat(
            [
                pd.DataFrame({"band": band, period_col: period_data[i]})
                for i, band in enumerate(BANDS)
                if len(period_data[i]) > 0
            ],
            ignore_index=True,
        )
        if not plot_df.empty:
            sns.violinplot(
                x="band",
                y=period_col,
                data=plot_df,
                order=BANDS,
                palette=palette,
                cut=0,
                bw=0.2,
                inner=None,
                scale="width",
                ax=ax,
            )
            # overlay jittered points for clarity (helps distinguish close values)
            sns.stripplot(
                x="band",
                y=period_col,
                data=plot_df,
                order=BANDS,
                color="k",
                size=3,
                jitter=0.15,
                alpha=0.6,
                ax=ax,
            )
            # use log scale on y to spread close small-period values (e.g., gamma)
            try:
                ax.set_yscale("log")
            except Exception:
                pass
    else:
        # nothing significant: draw empty violin placeholders
        ax.text(0.5, 0.5, "No significant periods", ha="center", va="center")

    # remove automatic xlabel added by seaborn and set tick label size
    ax.set_xlabel("")
    ax.set_xticklabels(BANDS_plot, rotation=30, ha="right", fontsize=14)
    ax.set_ylabel("significant dominant periods (s)", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_title("Significant periods by band", fontsize=16)
    fig.tight_layout()
    plt.savefig(output_path / "multiband_significant_periods.png")
    plt.close(fig)

    print(f"Saved: {output_path / 'multiband_significant_periods.png'}")


if __name__ == "__main__":
    main()
