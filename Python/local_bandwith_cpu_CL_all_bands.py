"""
Author      : Jie Li
Date        : 2026-09-12 09:05:18
Last Edited : 2026-09-12 09:05:42
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/local_bandwith_cpu_CL_all_bands.py
Description :








Copyright (c) 2026, Jie Li, lijie_12@outlook.com
All Rights Reserved.
"""

import os

import numpy as np
from local_bandwith_cpu_CL import select_pairwise_bandwidths_coarse_fine_cpu_CL

bands = ["delta", "theta", "alpha", "beta", "slow_gamma", "fast_gamma"]
# IMPORTANT: fs must equal the sampling rate of X. The power-envelope file
# power_envelope_delta_dfreq_none.npy is sampled at 10 Hz, so fs=10.0 here.
# If you instead feed the 1000 Hz broadband ROI time series, set fs=1000.0.
fs = 1000.0
coarse = np.geomspace(7.0, 500.0, 15)
for band in bands:
    print(f"Processing band: {band}")
    data = np.load(f"../Data/power_envelope_{band}_dfreq_none.npy")
    X = np.log(data + 1e-8)
    res = select_pairwise_bandwidths_coarse_fine_cpu_CL(
        X,
        coarse,
        fs=fs,
        fine_multipliers=(0.80, 0.90, 1.00, 1.10, 1.20),
        gap_ms=20.0,
        cv_stride=50,
        pair_batch=512,
        workers=-1,
        newton_iter=3,
        n_worker=min(os.cpu_count() or 1, 32),  # parallelize across cores
        compute_t1=False,  # T1 does not affect the CV-selected bandwidth
        verbose=True,
    )
    print("Saving bandwidths for band:", band)
    np.save(f"../Data/bandwidths_{band}_dfreq_none.npy", res)

print("All bands processed and bandwidths saved.")
