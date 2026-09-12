"""
Author      : Jie Li
Date        : 2026-09-11
Last Edited : 2026-09-12 08:53:57
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/rhythmicity_analysis_new.py
Description : Rhythmicity analysis from pre-computed Yeo-7 network connectivity
              AR-based periodicity test
              Reads results from get_eigvar_yeo7_network.py






Copyright (c) 2026, Jie Li, lijie_12@outlook.com
All Rights Reserved.
"""

import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, welch
from scipy.stats import f as f_dist
from statsmodels.tsa.ar_model import AutoReg

warnings.filterwarnings("ignore")

# =========================================================
# USER SETTINGS
# =========================================================
BANDS = ["delta", "theta", "alpha", "beta", "slow_gamma", "fast_gamma"]
NETWORKS = ["Visual", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

# File paths
DATA_DIR = "../Results"
OUT_DIR = "../Results"
ROI_FILE = "../Results/roi_names.npy"

# Signal processing parameters
FC_DT = 0.1  # 0.1 s between FC frames
FC_FS = 1.0 / FC_DT  # 10 Hz
NPERSEG = 1024

# Periodicity search range
MIN_PERIOD = 5.0  # seconds
MAX_PERIOD = 300.0  # seconds
MIN_FREQ = 1.0 / MAX_PERIOD
MAX_FREQ = 1.0 / MIN_PERIOD

# AR model parameters
AR_MIN_ORDER = 1
AR_MAX_ORDER = 20
AR_N_FREQ = 500

# Processing parameters
MASTER_SEED = 20240101

ALPHA = 0.05

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def psd(x, fs, nperseg):
    """Compute power spectral density using Welch's method."""
    x = np.nan_to_num(x - np.nanmean(x))
    return welch(
        x,
        fs=fs,
        window="hann",
        nperseg=min(nperseg, len(x)),
        noverlap=min(nperseg // 2, max(0, len(x) // 2 - 1)),
        detrend="constant",
        scaling="density",
    )


def acf(x):
    """Compute autocorrelation function."""
    x = np.nan_to_num(x - np.nanmean(x))
    n = len(x)
    m = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, n=m)
    a = np.fft.irfft(X * np.conj(X), n=m)[:n]
    a /= np.arange(n, 0, -1)
    return a / a[0] if a[0] else a


def prepare_signal_for_ar(x):
    y = np.nan_to_num(np.asarray(x, float) - np.nanmean(x))
    if len(y) < 20:
        return None
    sd = float(np.std(y))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return y / sd


def harmonic_design(n, freq_hz, fs):
    """Create harmonic regressor matrix."""
    t = np.arange(n) / fs
    w = 2 * np.pi * freq_hz
    return np.column_stack([np.ones(n), np.cos(w * t), np.sin(w * t)])


def select_ar_order(y, ar_min=AR_MIN_ORDER, ar_max=AR_MAX_ORDER):
    """Select AR order using AIC."""
    best_order = np.nan
    best_aic = np.inf
    for p in range(ar_min, ar_max + 1):
        try:
            model = AutoReg(y, lags=p, trend="c", old_names=False).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = p
        except Exception:
            continue
    return best_order


def ar_whiten(X, y, phi):
    """Apply AR whitening to regressor and response."""
    p = len(phi)
    yw = y[p:].copy()
    Xw = X[p:].copy()
    for lag in range(1, p + 1):
        yw -= phi[lag - 1] * y[p - lag : len(y) - lag]
        Xw -= phi[lag - 1] * X[p - lag : len(y) - lag]
    return Xw, yw


def ar_global_periodicity_test(
    x, fs, lo, hi, n_freq=AR_N_FREQ, ar_min=AR_MIN_ORDER, ar_max=AR_MAX_ORDER
):
    """AR-adjusted global periodicity test via harmonic-regression grid search.

    This uses a pointwise approach: fit AR model, then search over frequency grid to find the minimum p-value.
    """
    y = prepare_signal_for_ar(x)
    if y is None or not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return {
            "p": np.nan,
            "F": np.nan,
            "frequency": np.nan,
            "period": np.nan,
            "ar_order": np.nan,
        }
    if lo <= 0 or hi >= fs / 2:
        return {
            "p": np.nan,
            "F": np.nan,
            "frequency": np.nan,
            "period": np.nan,
            "ar_order": np.nan,
        }

    ar_order = select_ar_order(y, ar_min=ar_min, ar_max=ar_max)
    if not np.isfinite(ar_order):
        return {
            "p": np.nan,
            "F": np.nan,
            "frequency": np.nan,
            "period": np.nan,
            "ar_order": np.nan,
        }

    try:
        model = AutoReg(y, lags=int(ar_order), trend="c", old_names=False).fit()
    except Exception:
        return {
            "p": np.nan,
            "F": np.nan,
            "frequency": np.nan,
            "period": np.nan,
            "ar_order": ar_order,
        }
    phi = np.asarray(model.params)[1:]

    best = None
    n = len(y)
    for freq in np.linspace(lo, hi, int(n_freq)):
        X1 = harmonic_design(n, freq, fs)
        X0 = X1[:, [0]]
        X1w, yw = ar_whiten(X1, y, phi)
        X0w, _ = ar_whiten(X0, y, phi)

        if len(yw) <= 3:
            continue
        b1, *_ = np.linalg.lstsq(X1w, yw, rcond=None)
        b0, *_ = np.linalg.lstsq(X0w, yw, rcond=None)
        rss1 = float(np.dot(yw - X1w @ b1, yw - X1w @ b1))
        rss0 = float(np.dot(yw - X0w @ b0, yw - X0w @ b0))

        df1, df2 = 2, len(yw) - 3
        if df2 <= 0 or rss1 <= 0:
            continue
        num = max((rss0 - rss1) / df1, 0.0)
        den = rss1 / df2
        if den <= 0:
            continue
        F = num / den
        p = float(f_dist.sf(F, df1, df2))
        if (best is None) or (p < best["p"]):
            best = {
                "p": p,
                "F": float(F),
                "frequency": float(freq),
                "period": float(1.0 / freq),
                "ar_order": int(ar_order),
            }

    if best is None:
        return {
            "p": np.nan,
            "F": np.nan,
            "frequency": np.nan,
            "period": np.nan,
            "ar_order": int(ar_order),
        }
    return best


# =========================================================
# WORKER FUNCTION
# =========================================================
def _process_pair_task(task):
    """Process one Yeo-7 network pair."""
    try:
        band = task["band"]
        a = task["a"]
        b = task["b"]
        x = task["x"]
        fs = task["fs"]
        nperseg = task["nperseg"]
        min_period = task["min_period"]
        max_period = task["max_period"]
        ar_n_freq = task["ar_n_freq"]
        ar_min_order = task["ar_min_order"]
        ar_max_order = task["ar_max_order"]

        # PSD and AC
        f, p = psd(x, fs, nperseg)
        ac = acf(x)

        lo = 1.0 / max_period
        hi = 1.0 / min_period
        valid = (f >= lo) & (f <= hi) & (f > 0)
        peaks, _ = find_peaks(p)
        peaks = peaks[valid[peaks]] if len(peaks) else np.array([], dtype=int)

        if len(peaks):
            k = peaks[np.argmax(p[peaks])]
        else:
            valid_idx = np.flatnonzero(valid)
            if len(valid_idx):
                k = valid_idx[np.argmax(p[valid_idx])]
            else:
                k = 0

        pf = float(f[k]) if f is not None else np.nan
        pp = float(1.0 / pf) if pf and pf > 0 else np.nan

        baseline = np.median(p[valid]) if np.any(valid) else np.nan
        psr = (
            float(p[k] / baseline)
            if (baseline is not None and baseline > 0)
            else np.nan
        )

        # AR-global periodicity test (non-bootstrap)
        ar_res = ar_global_periodicity_test(
            x, fs, lo, hi, n_freq=ar_n_freq, ar_min=ar_min_order, ar_max=ar_max_order
        )
        ar_p = float(ar_res.get("p", np.nan))

        row = dict(
            band=band,
            network_a=NETWORKS[a],
            network_b=NETWORKS[b],
            peak_frequency_hz=pf,
            peak_period_s=pp,
            periodicity_strength=psr,
            observed_peak_power=float(p[k]) if f is not None else np.nan,
            ar_global_periodicity_p=ar_p,
            ar_global_periodicity_F=ar_res.get("F", np.nan),
            ar_global_frequency_hz=ar_res.get("frequency", np.nan),
            ar_global_period_s=ar_res.get("period", np.nan),
            ar_selected_order=ar_res.get("ar_order", np.nan),
        )

        metrics = dict(freqs=f, P=p, AC=ac, PF=pf, PP=pp, PSR=psr, ARP=ar_p)
        return (a, b, row, metrics)
    except Exception as e:
        traceback.print_exc()
        return (task.get("a"), task.get("b"), {"error": str(e)}, None)


# =========================================================
# MAIN
# =========================================================
def main():
    t_start = time.time()
    print("=" * 70)
    print("Rhythmicity Analysis - AR-based Periodicity Test")
    print(f"FC interval          : {FC_DT} s")
    print(f"FC sampling frequency: {FC_FS} Hz")
    print(f"Periodicity range    : {MIN_PERIOD}-{MAX_PERIOD} s")
    print(f"Frequency grid size  : {AR_N_FREQ}")
    print(f"AR order range       : {AR_MIN_ORDER}-{AR_MAX_ORDER} (AIC)")
    print(f"Processing mode      : Sequential (no parallelization)")
    print("=" * 70)

    # Load ROI names
    try:
        roi_names = np.load(ROI_FILE, allow_pickle=True)
        roi_names = [
            x.decode() if isinstance(x, bytes) else str(x) for x in roi_names.tolist()
        ]
    except Exception as e:
        print(f"Warning: Could not load ROI names from {ROI_FILE}: {e}")
        roi_names = [f"ROI_{i}" for i in range(200)]

    rows = []
    rng = np.random.default_rng(MASTER_SEED)

    for band in BANDS:
        print(f"\n[{band}] Loading precomputed Yeo-7 networks...")

        # Load precomputed Yeo-7 networks from get_eigvar_yeo7_network.py output
        data_file = Path(DATA_DIR) / f"eigvar_yeo7_network_{band}.npz"

        if not data_file.exists():
            print(f"  Warning: {data_file} not found, skipping {band}")
            continue

        data = np.load(data_file, allow_pickle=True)
        C = data["yeo7_networks"]  # (7, 7, num_observations)

        if C.ndim != 3 or C.shape[0] != 7 or C.shape[1] != 7:
            print(f"  Error: Expected shape (7, 7, T), got {C.shape}")
            continue

        print(f"  Loaded shape {C.shape}")

        freqs = None
        PSD = None
        AC = None
        PF = np.full((7, 7), np.nan)
        PP = np.full((7, 7), np.nan)
        PSR = np.full((7, 7), np.nan)
        ARP = np.full((7, 7), np.nan)

        # Process network pairs sequentially
        for a in range(7):
            for b in range(a, 7):
                x = C[a, b]
                task = {
                    "band": band,
                    "a": a,
                    "b": b,
                    "x": x,
                    "fs": FC_FS,
                    "nperseg": NPERSEG,
                    "min_period": MIN_PERIOD,
                    "max_period": MAX_PERIOD,
                    "ar_n_freq": AR_N_FREQ,
                    "ar_min_order": AR_MIN_ORDER,
                    "ar_max_order": AR_MAX_ORDER,
                    "seed": int(rng.integers(0, 2**31 - 1)),
                }

                a, b, row, metrics = _process_pair_task(task)
                if metrics is None:
                    print(f"  Pair ({a},{b}) failed: {row.get('error')}")
                    continue

                if freqs is None:
                    freqs = metrics.get("freqs")
                    PSD = np.full((7, 7, len(freqs)), np.nan)
                    AC = np.full((7, 7, metrics.get("AC").shape[0]), np.nan)

                PSD[a, b] = PSD[b, a] = metrics.get("P")
                AC[a, b] = AC[b, a] = metrics.get("AC")
                PF[a, b] = PF[b, a] = metrics.get("PF")
                PP[a, b] = PP[b, a] = metrics.get("PP")
                PSR[a, b] = PSR[b, a] = metrics.get("PSR")
                ARP[a, b] = ARP[b, a] = metrics.get("ARP")
                rows.append(row)

        # Save npz
        out_path = Path(OUT_DIR) / f"rhythmicity_{band}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            np.savez_compressed(
                out_path,
                frequencies=freqs,
                psd=PSD,
                acf=AC,
                peak_frequency=PF,
                peak_period=PP,
                periodicity_strength=PSR,
                ar_global_periodicity_p=ARP,
            )
            print(f"  Saved {out_path}")
        except Exception as e:
            print(f"  Error saving {out_path}: {e}")

    # Save CSV summary
    if len(rows):
        df = pd.DataFrame(rows)

        def bh_fdr(col):
            p = df[col].to_numpy(float)
            q = np.full(len(p), np.nan)
            ok = np.isfinite(p)
            pv = p[ok]
            if len(pv) == 0:
                return q
            order = np.argsort(pv)
            qr = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
            qr = np.minimum.accumulate(qr[::-1])[::-1]
            tmp = np.empty_like(qr)
            tmp[order] = np.minimum(qr, 1)
            q[ok] = tmp
            return q

        df["ar_global_periodicity_q_fdr"] = bh_fdr("ar_global_periodicity_p")

        csv_path = Path(OUT_DIR) / "rhythmicity_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[Results] Saved to {csv_path}")

        # Print summary statistics
        print("\n" + "=" * 70)
        print("Summary by band")
        summary = (
            df.groupby("band")
            .agg(
                n_pairs=("network_a", "size"),
                median_peak_freq=("peak_frequency_hz", "median"),
                median_ar_p=("ar_global_periodicity_p", "median"),
                median_ar_q=("ar_global_periodicity_q_fdr", "median"),
            )
            .reset_index()
        )
        print(summary.to_string(index=False))

        sig = df[df.ar_global_periodicity_q_fdr < ALPHA].sort_values(
            "ar_global_periodicity_q_fdr"
        )
        print(f"\nGlobal FDR-significant pairs (q < {ALPHA}): {len(sig)}")
        if len(sig) > 0:
            cols = [
                "band",
                "network_a",
                "network_b",
                "peak_frequency_hz",
                "ar_global_periodicity_p",
                "ar_global_periodicity_q_fdr",
            ]
            print(sig[cols].to_string(index=False))

    print(f"\nTotal elapsed: {time.time() - t_start:.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
