"""
Author      : Jie Li
Date        : 2026-08-16 22:02:20
Last Edited : 2026-09-11 13:59:19
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/load_combine_simulation_mse.py
Description :








Copyright (c) 2026, Jie Li, lijie_12@outlook.com
All Rights Reserved.
"""

# %% ========================================================================
# 				packages
# ===========================================================================
import pickle
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from data_to_dataframe_boundary import data_to_dataframe_boundary

# Convert loaded data to a tidy DataFrame for analysis/plotting.

# df = data_to_dataframe(data)
# # quick check
# print(df.head())


def load_and_combine_results(results_dir: str = "../Results") -> pd.DataFrame:
    """Load all `*_results.pkl` files in `results_dir`, convert and combine.

    Returns the concatenated DataFrame (may be empty if no files found).
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")

    pkl_files = sorted(results_path.glob("*_results.pkl"))
    dfs = []
    for p in pkl_files:
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
        except Exception as e:
            print(f"Warning: failed to load {p.name}: {e}")
            continue

        try:
            df_i = data_to_dataframe_boundary(d)
        except Exception as e:
            print(f"Warning: failed to convert {p.name} to DataFrame: {e}")
            continue

        if not df_i.empty:
            df_i["source_file"] = p.name
            dfs.append(df_i)

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
    else:
        df_all = pd.DataFrame(
            columns=[
                "corr_fun",
                "kernel",
                "sample_strategy",
                "method",
                "sample_size",
                "rep_idx",
                "mse",
                "mse_b",
                "h_opt",
                "time_bandwidth",
                "time_u_test",
                "source_file",
            ]
        )

    # # Save summary CSV next to results
    # out_csv = results_path / "combined_results_summary.csv"
    # try:
    #     df_all.to_csv(out_csv, index=False)
    #     print(f"✓ Combined results saved to: {out_csv}")
    # except Exception as e:
    #     print(f"Warning: failed to save combined CSV: {e}")

    return df_all


# Combine all result files in ../Results
df_all = load_and_combine_results("../Results")
print(df_all.shape)

# %%
# Mapping of original names to short codes
# Correlation functions -> F1..F9
#  F1: parabolic
#  F2: trigonometric_sine
#  F3: trigonometric_cosine
#  F4: cubic
#  F5: linear
#  F6: exponential
#  F7: semicircle_upper
#  F8: inverse_trig
#  F9: chebyshev_t3
#
# Kernel functions -> K1..K6
#  K1: gaussian
#  K2: epanechnikov
#  K3: triangular
#  K4: uniform
#  K5: quartic
#  K6: triweight
#
# Sampling strategies -> S1..S2
#  S1: uniform
#  S2: boundary
#
# Method name replacements
#  CUBIC -> CE
#  CUBIC+LL -> CL

# Apply the mappings to the DataFrame columns
corr_map = {
    "parabolic": "F1",
    "trigonometric_sine": "F2",
    "trigonometric_cosine": "F3",
    "cubic": "F4",
    "linear": "F5",
    "exponential": "F6",
    "semicircle_upper": "F7",
    "inverse_trig": "F8",
    "chebyshev_t3": "F9",
}

kernel_map = {
    "gaussian": "K1",
    "epanechnikov": "K2",
    "triangular": "K3",
    "uniform": "K4",
    "quartic": "K5",
    "triweight": "K6",
}

sample_map = {"uniform": "S1", "boundary": "S2"}

method_map = {"CUBIC": "CE", "CUBIC+LL": "CL"}

if "corr_fun" in df_all.columns:
    df_all["corr_fun"] = df_all["corr_fun"].map(corr_map).fillna(df_all["corr_fun"])

if "kernel" in df_all.columns:
    df_all["kernel"] = df_all["kernel"].map(kernel_map).fillna(df_all["kernel"])

if "sample_strategy" in df_all.columns:
    df_all["sample_strategy"] = (
        df_all["sample_strategy"].map(sample_map).fillna(df_all["sample_strategy"])
    )

if "method" in df_all.columns:
    df_all["method"] = df_all["method"].map(method_map).fillna(df_all["method"])

# Drop the last column 'source_file' if it exists
if "source_file" in df_all.columns:
    df_all.drop(columns=["source_file"], inplace=True)

print(df_all.head())
save_path = Path("../Results/combined_results_summary.csv")
pd.DataFrame(df_all).to_csv(save_path, index=False)
# %%
