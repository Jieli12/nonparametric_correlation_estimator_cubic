"""
Author      : Jie Li
Date        : 2026-09-12 09:07:14
Last Edited : 2026-09-12 09:09:02
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/calculate_correlation_matrix.py
Description :

Build the per-band dynamic correlation matrix from saved bandwidth-selection
results.

For each band it loads
    Data/power_envelope_{band}_dfreq_none.npy     (power envelope, log-transformed)
    Data/bandwidths_{band}_dfreq_none.npy         (saved result dict from
                                                   local_bandwidth_cpu_CL_all_bands.py)
takes the optimal per-pair bandwidth matrix H_ms (n_roi x n_roi, in ms) out of
the saved results, and evaluates the local Gaussian correlation rho(u0) at
every time stamp with the same boundary-corrected local-linear (CL) estimator
used by local_bandwith_cpu_CL.py:

    A(u0) = NW[ X1^2 + X2^2 ]                                  (local constant)
    B(u0) = ( S2 T0 - S1 T1 ) / ( S2 S0 - S1^2 )               (local linear)
    rho(u0) = real root of   rho^3 - B rho^2 + (A - 1) rho - B = 0

where X1, X2 are the (log-transformed) envelopes of a ROI pair and the kernel
weights come from the Gaussian hollow kernel with the pair's optimal h_ij. The
moments are evaluated on the full time grid, so the estimates are boundary
corrected: S0 normalises the local average, and the S1/T1 terms correct the
local-linear fit where the kernel is truncated at the series edges.

The output is saved as a single (n_pairs, n_time_stamp) array with
    n_pairs     = n_roi * (n_roi - 1) / 2 = 19900 for the Schaefer 200 atlas
    n_time_stamp = n_time / out_stride
i.e. one row per ordered pair (i < j) and one column per output time stamp, in
the same pair order as roi_pairs(n_roi) (row-major upper triangle).



Copyright (c) 2026, Jie Li, lijie_12@outlook.com
All Rights Reserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from local_bandwith_cpu_CL import (
    gaussian_hollow_kernel_fft_moments,
    local_linear_B,
    roi_pairs,
    solve_rho_newton,
)
from scipy.fft import irfft, next_fast_len, rfft

BANDS = ["delta", "theta", "alpha", "beta", "slow_gamma", "fast_gamma"]
BAND_SYMBOLS = {
    "delta": "delta",
    "theta": "theta",
    "alpha": "alpha",
    "beta": "beta",
    "slow_gamma": "slow_gamma",
    "fast_gamma": "fast_gamma",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def detect_fs(n_time: int) -> float:
    """Infer the sampling rate of a power-envelope file.

    The recording is ~574 s long: the broadband/downsample-free envelopes have
    574 000 samples (1000 Hz) and the downsampled envelopes 5 740 samples
    (10 Hz, the target rate stored in the envelope metadata).
    """
    if n_time > 100_000:
        return 1000.0
    if n_time > 5_000:
        return 10.0
    raise ValueError(f"cannot infer fs from n_time={n_time}; pass --fs explicitly")


def load_band_results(path: str | Path) -> dict:
    """Load a saved bandwidth-selection result dict.

    The all-bands script stores the full result dict with np.save(...), i.e. a
    pickled 0-d object array; np.load(..., allow_pickle=True) recovers it.
    """
    obj = np.load(str(path), allow_pickle=True)
    if obj.ndim == 0:
        obj = obj.item()
    if not isinstance(obj, dict):
        raise TypeError(
            f"{path} does not contain a result dict " f"(got {type(obj).__name__})"
        )
    if "H_ms" not in obj:
        raise KeyError(f"{path} has no 'H_ms'; keys = {sorted(obj)}")
    return obj


def dynamic_correlation_matrix(
    X: np.ndarray,
    H_ms: np.ndarray,
    *,
    fs: float,
    gap_ms: float = 20.0,
    truncate: float = 4.0,
    workers: int = -1,
    out_stride: int = 1,
    pair_batch: int = 256,
    newton_iter: int = 3,
    verbose: bool = True,
):
    """Evaluate the CL local Gaussian correlation rho(u0) on the full grid.

    Parameters
    ----------
    X : (n_roi, n_time) float32/float64
        The (log-transformed) power envelopes used during bandwidth selection.
    H_ms : (n_roi, n_roi) float64
        Symmetric optimal per-pair bandwidth matrix, in milliseconds.
    fs : float
        Sampling rate of X (Hz). Must match the data the bandwidths were
        selected on (the bandwidths themselves are in ms and fs-independent).
    gap_ms, truncate : float
        Kernel parameters; must match the bandwidth-selection run.
    out_stride : int
        Keep every out_stride-th time stamp in the output. Default 1 = full
        resolution. Use e.g. 100 on the 1000 Hz envelopes to get the same
        number of columns (5 740) as the 10 Hz downsampled envelopes.
    pair_batch : int
        Pairs sharing one bandwidth are processed in batches of this size so
        peak memory stays bounded (relevant for the 1000 Hz envelopes).

    Returns
    -------
    rho : (n_pairs, n_out) float32 array
        Local correlation per ordered pair (i < j), one column per output time
        stamp. Values lie in (-1, 1).
    """
    wall0 = perf_counter()
    X = np.ascontiguousarray(X, dtype=np.float32)
    n_roi, n_time = X.shape
    pairs = roi_pairs(n_roi)
    n_pairs = len(pairs)
    if H_ms.shape != (n_roi, n_roi):
        raise ValueError(f"H_ms has shape {H_ms.shape}, expected ({n_roi}, {n_roi})")

    # Per-pair optimal bandwidth (ms), rounded to the fine-search precision.
    h = np.round(np.asarray(H_ms[pairs[:, 0], pairs[:, 1]], dtype=np.float64), 6)
    if np.any(h <= 0):
        raise ValueError("H_ms contains non-positive bandwidths")

    h_max = float(h.max())
    max_radius = int(np.ceil(truncate * h_max * fs / 1000.0))
    if n_time <= 2 * max_radius:
        raise ValueError(
            f"time series (n_time={n_time}) is too short for the maximum "
            f"bandwidth ({h_max:g} ms, radius {max_radius} samples)"
        )
    nfft = next_fast_len(n_time + 2 * max_radius)
    n_out = (n_time - 1) // out_stride + 1

    X2 = X * X
    FX2 = rfft(X2, n=nfft, axis=1, workers=workers)
    FONES = rfft(np.ones(n_time, dtype=np.float64), n=nfft)

    rho_out = np.empty((n_pairs, n_out), dtype=np.float32)

    # Group pairs by identical bandwidth so each kernel is built once.
    uniq_h, inv = np.unique(h, return_inverse=True)
    t_kernel = t_fft = t_score = 0.0
    for ui, hh in enumerate(uniq_h):
        t0 = perf_counter()
        K, K1, K2, radius = gaussian_hollow_kernel_fft_moments(
            float(hh), fs, gap_ms, truncate, nfft
        )
        # Kernel moments on the full grid (boundary aware via zero padding).
        S0 = irfft(FONES * K, n=nfft)[radius : radius + n_time].astype(
            np.float32, copy=False
        )
        S1 = irfft(FONES * K1, n=nfft)[radius : radius + n_time].astype(
            np.float32, copy=False
        )
        S2 = irfft(FONES * K2, n=nfft)[radius : radius + n_time].astype(
            np.float32, copy=False
        )
        t_kernel += perf_counter() - t0

        ids = np.flatnonzero(inv == ui)
        for bs in range(0, len(ids), pair_batch):
            g = ids[bs : bs + pair_batch]
            ii, jj = pairs[g, 0], pairs[g, 1]

            t0 = perf_counter()
            P = X[ii] * X[jj]
            Q = X2[ii] + X2[jj]
            FP = rfft(P, n=nfft, axis=1, workers=workers)
            FQ = rfft(Q, n=nfft, axis=1, workers=workers)
            # Odd first-moment kernel: the FFT convolution returns -T1.
            T0 = irfft(FP * K[None, :], n=nfft, axis=1, workers=workers)[
                :, radius : radius + n_time
            ]
            T1 = -irfft(FP * K1[None, :], n=nfft, axis=1, workers=workers)[
                :, radius : radius + n_time
            ]
            T0Q = irfft(FQ * K[None, :], n=nfft, axis=1, workers=workers)[
                :, radius : radius + n_time
            ]
            t_fft += perf_counter() - t0

            t0 = perf_counter()
            A = T0Q / np.maximum(S0[None, :], 1e-12)  # NW of X1^2 + X2^2
            B = local_linear_B(
                T0, T1, S0[None, :], S1[None, :], S2[None, :]
            )  # CL local-linear estimate of X1 X2
            rho = solve_rho_newton(A, B, n_iter=newton_iter)
            rho_out[g, :] = rho[:, ::out_stride].astype(np.float32, copy=False)
            t_score += perf_counter() - t0

            del P, Q, FP, FQ, T0, T1, T0Q, A, B, rho

    if verbose:
        print(
            f"pairs={n_pairs}, n_time={n_time}, n_out={n_out}, "
            f"n_uniq_h={len(uniq_h)}, nfft={nfft}, fs={fs:g} Hz"
        )
        print(
            f"  unique h: {uniq_h.min():.4g} .. {uniq_h.max():.4g} ms "
            f"(radius {int(np.ceil(truncate*uniq_h.min()*fs/1000.0))}..{max_radius} samples)"
        )
        print(
            f"  rho: min={rho_out.min():.4f}, max={rho_out.max():.4f}, "
            f"mean={rho_out.mean():.4f}, n_nan={np.isnan(rho_out).sum()}"
        )
        print(
            f"  time: kernel={t_kernel:.2f}s, fft={t_fft:.2f}s, "
            f"score={t_score:.2f}s, wall={perf_counter()-wall0:.2f}s"
        )
    return rho_out


def _bytes_str(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} PB"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "Data"),
        help="directory with envelope and saved bandwidth files",
    )
    ap.add_argument(
        "--bands", default=",".join(BANDS), help="comma-separated bands to process"
    )
    ap.add_argument(
        "--envelope-pattern",
        default="power_envelope_{band}_dfreq_none.npy",
        help="envelope file name template ({band} is substituted)",
    )
    ap.add_argument(
        "--result-pattern",
        default="bandwidths_{band}_dfreq_none.npy",
        help="saved bandwidth-selection result template ({band})",
    )
    ap.add_argument(
        "--out-pattern",
        default="dynamic_corr_{band}.npy",
        help="output file name template ({band})",
    )
    ap.add_argument(
        "--fs",
        type=float,
        default=0.0,
        help="sampling rate of the envelope (0 = auto-detect from n_time)",
    )
    ap.add_argument("--gap-ms", type=float, default=20.0)
    ap.add_argument("--truncate", type=float, default=4.0)
    ap.add_argument("--out-stride", type=int, default=1)
    ap.add_argument("--pair-batch", type=int, default=256)
    ap.add_argument("--newton-iter", type=int, default=3)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument(
        "--transform",
        choices=["log", "none"],
        default="log",
        help="transform applied to the envelope before computing correlation; "
        "'log' matches the bandwidth-selection runs (X = log(E + 1e-8))",
    )
    ap.add_argument(
        "--demo",
        action="store_true",
        help="run a small synthetic self-check instead of real files",
    )
    args = ap.parse_args()

    if args.demo:
        from local_bandwith_cpu_CL import synthetic_meg

        rng = np.random.default_rng(0)
        # A positive "power envelope": exponentiate the zero-mean synthetic so
        # the log transform used by the real pipeline is well defined.
        X = np.exp(0.5 * synthetic_meg(40, n_time=12_000, fs=10.0))
        X = np.log(X + 1e-8)
        # Fake smooth bandwidth field: slowly varying across ROI index.
        t = np.arange(40, dtype=np.float64)
        H = 120.0 + 60.0 * np.sin(2 * np.pi * t / 40.0)[:, None]
        H = np.maximum(H + H.T, 20.0)
        np.fill_diagonal(H, 0.0)
        rho = dynamic_correlation_matrix(
            X, H, fs=10.0, out_stride=4, pair_batch=64, verbose=True
        )
        finite = rho[~np.isnan(rho)]
        print(
            "demo rho: n_nonnan =",
            finite.size,
            "|rho|>0:",
            bool(np.any(np.abs(finite) > 1e-6)),
            "|rho| max:",
            float(np.abs(finite).max()),
        )
        return

    data_dir = Path(args.data_dir)
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    failed = []
    for band in bands:
        env_path = data_dir / args.envelope_pattern.format(band=band)
        res_path = data_dir / args.result_pattern.format(band=band)
        out_path = data_dir / args.out_pattern.format(band=band)

        if not env_path.exists():
            print(f"[skip] {band}: envelope not found: {env_path}")
            failed.append(band)
            continue
        if not res_path.exists():
            print(
                f"[skip] {band}: bandwidth results not found: {res_path}\n"
                f"       run Python/local_bandwidth_cpu_CL_all_bands.py first."
            )
            failed.append(band)
            continue

        print(f"\n=== {band} ===")
        data = np.load(env_path)
        X = np.log(data + 1e-8) if args.transform == "log" else data
        fs = args.fs if args.fs > 0 else detect_fs(X.shape[1])
        res = load_band_results(res_path)
        H_ms = np.asarray(res["H_ms"], dtype=np.float64)
        print(f"  envelope: {X.shape} {X.dtype}, fs={fs:g} Hz")
        print(
            f"  H_ms: {H_ms.shape}, h range {H_ms[H_ms>0].min():.4g}..{H_ms.max():.4g} ms"
        )

        rho = dynamic_correlation_matrix(
            X,
            H_ms,
            fs=fs,
            gap_ms=args.gap_ms,
            truncate=args.truncate,
            workers=args.workers,
            out_stride=args.out_stride,
            pair_batch=args.pair_batch,
            newton_iter=args.newton_iter,
        )
        np.save(out_path, rho)
        print(
            f"  saved: {out_path}  ({rho.shape[0]} x {rho.shape[1]}) "
            f"= {_bytes_str(rho.nbytes)}"
        )

    if failed:
        print(
            f"\nDone with {len(bands)-len(failed)}/{len(bands)} bands; "
            f"skipped: {failed}"
        )
    else:
        print(f"\nAll {len(bands)} bands processed.")


if __name__ == "__main__":
    main()
