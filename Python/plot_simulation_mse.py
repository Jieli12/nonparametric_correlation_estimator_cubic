"""
Author      : Jie Li
Date        : 2026-09-06 23:30:01
Last Edited : 2026-09-11 15:08:45
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/plot_simulation_mse.py
Description : Generate deep_eff_rank_matrix.csv







Copyright (c) 2026, Jie Li, lijie_12@outlook.com
All Rights Reserved.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _to_ordered_level(series: pd.Series) -> pd.Series:
    """Convert labels like 'F1'/'K2'/'S1' or plain numeric strings to Int64 levels."""
    s = series.astype(str)
    numeric = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
    return numeric.astype("Int64")


def _derive_level_from_values(series: pd.Series) -> pd.Series:
    """Derive stable integer levels from observed values when no numeric token exists."""
    s = series.astype(str)
    order = list(dict.fromkeys(s.dropna().tolist()))
    mapping = {v: i + 1 for i, v in enumerate(order)}
    return s.map(mapping).astype("Int64")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize dataframe for ranking analysis."""
    work = df.copy()

    # Infer F/K/S level columns
    if "F_level" not in work.columns:
        if "corr_fun" in work.columns:
            work["F_level"] = _to_ordered_level(work["corr_fun"])
            if work["F_level"].isna().all():
                work["F_level"] = _derive_level_from_values(work["corr_fun"])
        else:
            work["F_level"] = pd.Series([pd.NA] * len(work), dtype="Int64")
    else:
        work["F_level"] = pd.to_numeric(work["F_level"], errors="coerce").astype(
            "Int64"
        )

    if "K_level" not in work.columns:
        if "kernel" in work.columns:
            work["K_level"] = _to_ordered_level(work["kernel"])
            if work["K_level"].isna().all():
                work["K_level"] = _derive_level_from_values(work["kernel"])
        else:
            work["K_level"] = pd.Series([pd.NA] * len(work), dtype="Int64")
    else:
        work["K_level"] = pd.to_numeric(work["K_level"], errors="coerce").astype(
            "Int64"
        )

    if "S_level" not in work.columns:
        if "sample_strategy" in work.columns:
            work["S_level"] = _to_ordered_level(work["sample_strategy"])
            if work["S_level"].isna().all():
                work["S_level"] = _derive_level_from_values(work["sample_strategy"])
        else:
            work["S_level"] = pd.Series([pd.NA] * len(work), dtype="Int64")
    else:
        work["S_level"] = pd.to_numeric(work["S_level"], errors="coerce").astype(
            "Int64"
        )

    # Numeric columns
    for col in [
        "sample_size",
        "mse",
        "h_opt",
        "time_bandwidth",
        "time_u_test",
        "rep_idx",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    # Compute total time
    if "time_bandwidth" in work.columns and "time_u_test" in work.columns:
        work["time_total"] = work["time_bandwidth"].fillna(0) + work[
            "time_u_test"
        ].fillna(0)
    elif "time_total" not in work.columns:
        work["time_total"] = np.nan

    return work


def generate_eff_rank_matrix(
    df: pd.DataFrame,
    out_dir: str = "../Results",
    mse_weight: float = 0.5,
    mse_weights: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
):
    """Generate deep_eff_rank_matrix.csv with method rankings across mse_weight values.

    Produces:
      - deep_eff_rank_matrix.csv : method x mse_weight rank matrix
        (rows = methods, columns = mse_weight, values = rank where 1 = best)

    The ranking is based on an efficiency score:
      eff_score = mse_weight * log10(MSE)_norm + (1 - mse_weight) * log10(time)_norm
    where normalization is min-max scaling to [0, 1] within each scenario.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns: method, sample_size, F_level, K_level, S_level, mse, time_total
    out_dir : str
        Output directory for CSV file. Default "../Results"
    mse_weight : float
        Primary MSE weight for default ranking (not used in matrix, kept for API compatibility)
    mse_weights : tuple
        Weights for which rankings are computed. Default (0.1, 0.2, ..., 0.9)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define scenario grouping dimensions
    scen = ["sample_size", "F_level", "K_level", "S_level"]

    # Aggregate to median MSE and time per scenario and method
    med = (
        df.groupby(scen + ["method"], observed=True)
        .agg(mse=("mse", "median"), time=("time_total", "median"))
        .reset_index()
    )

    # Compute per-scenario ranks (smaller is better for both MSE and time)
    med["rank_mse"] = med.groupby(scen, observed=True)["mse"].rank(ascending=True)
    med["rank_time"] = med.groupby(scen, observed=True)["time"].rank(ascending=True)

    # Log-scale representation
    med["log_mse"] = np.log10(med["mse"].clip(lower=1e-16))
    med["log_time"] = np.log10(med["time"].clip(lower=1e-16))

    # Min-max normalization to [0, 1] within each scenario
    def _minmax01(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi - lo > 1e-12:
            return (s - lo) / (hi - lo)
        return s * 0.0

    med["log_mse_n"] = med.groupby(scen, observed=True)["log_mse"].transform(_minmax01)
    med["log_time_n"] = med.groupby(scen, observed=True)["log_time"].transform(
        _minmax01
    )

    # ---- Compute rank_mse and rank_dual (average per-scenario ranks) ----
    rank_mse_by_method = (
        med.groupby("method", observed=True)["rank_mse"].mean().reset_index()
    )
    rank_mse_by_method.columns = ["method", "rank_mse"]

    # rank_dual = average of (rank_mse + rank_time) / 2
    med["rank_dual"] = (med["rank_mse"] + med["rank_time"]) / 2.0
    rank_dual_by_method = (
        med.groupby("method", observed=True)["rank_dual"].mean().reset_index()
    )
    rank_dual_by_method.columns = ["method", "rank_dual"]

    # Print summary of rank_mse and rank_dual
    summary_ranks = rank_mse_by_method.merge(
        rank_dual_by_method, on="method"
    ).sort_values("rank_dual")
    print("Method ranking summary (smaller is better):")
    print(summary_ranks.to_string(index=False))
    print()

    # Compute rankings across multiple mse_weight cases
    rows = []
    for w in mse_weights:
        w = float(w)
        wt = 1.0 - w
        # Efficiency score = weighted average of normalized log metrics
        score = w * med["log_mse_n"] + wt * med["log_time_n"]
        ag = (
            med[["method"]]
            .assign(score=score)
            .groupby("method", observed=True)["score"]
            .mean()
            .reset_index()
        )
        ag["rank"] = ag["score"].rank(ascending=True)
        ag.insert(0, "mse_weight", w)
        ag.insert(1, "time_weight", wt)
        rows.append(ag)

    rank_by_w = pd.concat(rows, ignore_index=True).sort_values(["mse_weight", "rank"])

    # Pivot to wide format: rows = methods, columns = mse_weight, values = rank
    wide = (
        rank_by_w.pivot(index="method", columns="mse_weight", values="rank")
        .rename_axis("mse_weight", axis=1)
        .sort_index()
    )

    # Merge rank_mse and rank_dual columns
    result = wide.reset_index()
    result = result.merge(rank_mse_by_method, on="method")
    result = result.merge(rank_dual_by_method, on="method")

    # Reorder columns: method, rank_mse, rank_dual, then mse_weight columns
    mse_weight_cols = [col for col in wide.columns if isinstance(col, float)]
    cols_order = ["method", "rank_mse", "rank_dual"] + sorted(mse_weight_cols)
    result = result[cols_order]

    result.to_csv(out_dir / "deep_eff_rank_matrix.csv", index=False)

    print("\nGenerated deep_eff_rank_matrix.csv (efficiency ranking by mse_weight):")
    print(result.to_string(index=False))
    print(f"\nFile saved to: {out_dir / 'deep_eff_rank_matrix.csv'}")

    return out_dir


def main():
    """Generate deep_eff_rank_matrix.csv only."""
    parser = argparse.ArgumentParser(
        description="Generate deep_eff_rank_matrix.csv (method rankings by mse_weight)"
    )
    parser.add_argument(
        "--mse-weight",
        type=float,
        default=0.5,
        help="(unused, kept for API compatibility). Default 0.5.",
    )
    args, _ = parser.parse_known_args()

    input_csv = Path("../Results/combined_results_summary.csv")
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    full_df = pd.read_csv(input_csv)
    full_df = prepare_dataframe(full_df)

    generate_eff_rank_matrix(full_df, out_dir="../Results", mse_weight=args.mse_weight)


if __name__ == "__main__":
    main()
