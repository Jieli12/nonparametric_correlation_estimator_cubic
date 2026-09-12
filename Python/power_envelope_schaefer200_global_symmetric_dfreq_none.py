"""
Author      : Jie Li
Date        : 2026-09-11 17:49:46
Last Edited : 2026-09-12 09:49:25
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/power_envelope_schaefer200_global_symmetric_dfreq_none.py
Description :

MEG Schaefer-200 rank-aware global symmetric-orthogonalized power-envelope pipeline.

Purpose
-------
Generate rank-aware globally symmetric-orthogonalized power-envelope time series for all 200 Schaefer ROIs.

This script intentionally DOES NOT compute connectivity. It performs rank-aware global symmetric orthogonalization only.

Final pipeline
--------------
MEG / empty-room preprocessing
-> source space + BEM + forward model
-> empty-room noise covariance (once)
-> minimum-norm inverse operator (once)
-> broadband source reconstruction (once)
-> Schaefer-200 ROI time courses with mean_flip
-> diagnostic rank / singular-value checks
-> for each band:
        band-pass ROI signals
        Hilbert analytic signal
        power envelope = |analytic|^2
        low-pass envelope at 1 Hz
        resample envelope to 10 Hz


Rank-aware global symmetric orthogonalization
-----------------------------------------------
For the current dataset, the sensor-space SSS/info rank is approximately 80 and
the empty-room rank is approximately 73. The latest Schaefer-200 ROI diagnostic
shows a pronounced singular-value gap around rank 70 (e.g. delta: s70 ~ 3.73e-2
and s71 ~ 2.01e-7 after row scaling). Therefore this script does NOT use the
ordinary full-rank inverse (X X^T)^(-1/2), which would be singular.

Instead, for each frequency band it performs a rank-aware/truncated Löwdin
symmetric orthogonalization. If X = U_r S_r V_r^T is the truncated SVD, the
rank-aware symmetric-orthogonalized representation is U_r V_r^T, followed by
row-wise rescaling to preserve the original ROI RMS amplitudes. This keeps all
200 ROI rows while respecting the empirically supported effective rank r.

The rank is selected automatically from the largest adjacent singular-value
gap in the row-standardized band-limited ROI matrix. The full singular spectrum,
selected rank, explained variance and gap are saved for diagnostics.

This is a rank-deficient modification of multivariate symmetric orthogonalization;
it should be described as such in methods, rather than as ordinary full-rank
200-dimensional orthogonalization. Connectivity is intentionally NOT calculated.

Notes
-----
1. The experimental raw filename contains "_proc-sss", so it is assumed to
    have already undergone SSS/MaxFilter. It is NOT Maxwell-filtered again.
2. Empty-room Maxwell filtering is retained as a configurable option.
    Ideally the experimental and empty-room data should use compatible SSS
    settings / calibration / cross-talk / bad-channel handling.
3. A 0.5-Hz high-pass is applied to both recordings before source analysis.
    This removes very slow drift while preserving the lowest 1-4 Hz band.
4. Noise-covariance / inverse whitening rank is taken from the empty-room SSS
    information and passed explicitly to the covariance and inverse routines.

Copyright (c) 2026, Jie Li, lijie_12@outlook.com
All Rights Reserved.
"""

# %% ========================================================================
#                               Packages
# ===========================================================================
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import mne
import numpy as np
from mne.minimum_norm import (
    apply_inverse_raw,
    make_inverse_operator,
    read_inverse_operator,
    write_inverse_operator,
)
from scipy.signal import hilbert, resample_poly

# %% ========================================================================
#                           Parameter settings
# ===========================================================================
subject_id = "sub-CC110033"

data_dir = "/path/to/MEG-PEC"
subjects_dir = "/path/to/MEG-PEC/freesurfer"
out_dir = "/path/to/Projects/nonparametric_correlation_estimator_cubic/Results"

raw_fname = os.path.join(
    data_dir, "meg", f"{subject_id}_ses-rest_task-rest_proc-sss.fif"
)
empty_fname = os.path.join(data_dir, "emptyroom", f"emptyroom_{subject_id[4:]}.fif")
trans_fname = os.path.join(data_dir, "meg", f"{subject_id}-trans.fif")

Path(out_dir).mkdir(parents=True, exist_ok=True)

# Frequency bands retained exactly from the original script.
bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 40),
    "slow_gamma": (40, 80),
    "fast_gamma": (80, 120),
}

# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
BROADBAND_HIGHPASS = 0.5  # Hz
NOTCH_FREQS = (50.0, 100.0, 150.0)

# Experimental raw is already *_proc-sss.fif.
APPLY_MAXWELL_TO_RAW = False

# Retained from the original pipeline. Set False if the empty-room file has
# already been processed using the compatible SSS/MaxFilter pipeline.
APPLY_MAXWELL_TO_EMPTY = False
EMPTY_ST_DURATION = 10.0

# --------------------------------------------------------------------------
# Source reconstruction
# --------------------------------------------------------------------------
SOURCE_SPACING = "oct6"
SNR = 3.0
INVERSE_METHOD = "dSPM"
LOOSE = 0.2
DEPTH = 0.8

# --------------------------------------------------------------------------
# Envelope extraction
# --------------------------------------------------------------------------
ENVELOPE_LPF = 1.0  # keep slow envelope fluctuations below ~1 Hz
ENVELOPE_SFREQ = None  # saved envelope sampling frequency
USE_POWER = True  # True => |analytic|^2; False => |analytic|

# Save intermediate broadband ROI time series for later reuse.
SAVE_BROADBAND_ROI_TS = True

# Save the band-limited ROI signal as well as the envelope.
# Keeping this False saves disk space.
SAVE_BAND_ROI_TS = False

# --------------------------------------------------------------------------
# Rank-aware global symmetric orthogonalization
# --------------------------------------------------------------------------
ORTH_RANK_MODE = "spectral_gap"  # "spectral_gap", "explained_variance", or "fixed"
ORTH_FIXED_RANK = 70
ORTH_EXPLAINED_VARIANCE = 0.999
ORTH_MIN_RANK = 2
ORTH_MAX_RANK = 200
ORTH_MIN_GAP_RATIO = 10.0
ORTH_PRESERVE_ROW_RMS = True


# %% ========================================================================
#                           Utility functions
# ===========================================================================
def safe_tag(value):
    """Convert a numeric value to a filename-safe short tag."""
    return str(value).replace(".", "p")


def download_schaefer_200(subjects_dir):
    """Download Schaefer-2018 200-parcel annotation files to fsaverage."""
    label_dir = Path(subjects_dir) / "fsaverage" / "label"
    label_dir.mkdir(parents=True, exist_ok=True)

    base_url = (
        "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
        "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
        "Parcellations/FreeSurfer5.3/fsaverage/label/"
    )

    def _is_valid_annot(path: Path) -> bool:
        """Quick sanity check for a FreeSurfer .annot binary file."""
        try:
            with open(path, "rb") as f:
                header = f.read(4)

            if len(header) < 4:
                return False

            n_verts = int.from_bytes(header, byteorder="big", signed=True)
            if not (1000 <= n_verts <= 2_000_000):
                return False

            # Minimal bytes:
            # int32 n_vertices + [vertex, label] pairs + ctab flag.
            min_size = 4 + n_verts * 8 + 4
            return path.stat().st_size >= min_size

        except OSError:
            return False

    for hemi in ("lh", "rh"):
        fname = f"{hemi}.Schaefer2018_200Parcels_7Networks_order.annot"
        dest = label_dir / fname
        url = base_url + fname

        if _is_valid_annot(dest):
            print(f"Using existing valid annotation: {dest}")
            continue

        if dest.exists():
            print(f"Existing annotation is invalid; deleting: {dest}")
            dest.unlink(missing_ok=True)

        print(f"Downloading: {url}")
        urllib.request.urlretrieve(url, dest)

        if not _is_valid_annot(dest):
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to obtain a valid Schaefer annotation: {fname}\n"
                f"Tried URL: {url}\n"
                f"Please manually place it under: {label_dir}"
            )

        print(f"Downloaded: {dest}")


def print_rank_diagnostics(raw, raw_empty):
    """Print empirical and SSS/info-based sensor ranks."""
    print("\n========== SENSOR RANK DIAGNOSTICS ==========")

    print("\nRAW estimated rank (rank=None)")
    raw_rank_est = mne.compute_rank(raw, rank=None)

    print("\nRAW SSS/info rank (rank='info')")
    raw_rank_info = mne.compute_rank(raw, rank="info")

    print("\nEMPTY estimated rank (rank=None)")
    empty_rank_est = mne.compute_rank(raw_empty, rank=None, on_rank_mismatch="warn")

    print("\nEMPTY SSS/info rank (rank='info')")
    empty_rank_info = mne.compute_rank(raw_empty, rank="info")

    print("\nRank summary")
    print(f"raw estimated : {raw_rank_est}")
    print(f"raw info      : {raw_rank_info}")
    print(f"empty estimated: {empty_rank_est}")
    print(f"empty info     : {empty_rank_info}")

    return raw_rank_est, raw_rank_info, empty_rank_est, empty_rank_info


def standardize_rows(X):
    """
    Row-standardize an ROI x time matrix for numerical diagnostics only.

    This is NOT used to generate the saved envelopes.
    """
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=-1, keepdims=True)

    row_std = X.std(axis=-1, keepdims=True)
    eps = np.finfo(float).eps

    if np.any(row_std <= eps):
        bad = np.where(row_std[:, 0] <= eps)[0]
        raise RuntimeError(f"Found {len(bad)} constant ROI time series: {bad.tolist()}")

    return X / row_std


def singular_value_diagnostics(X, label, n_print=20):
    """
    Compute standardized singular-value diagnostics.

    Returns
    -------
    rank_numpy : int
        NumPy numerical matrix rank after row standardization.
    s_rel : ndarray
        Singular values divided by the largest singular value.
    """
    X_scaled = standardize_rows(X)

    rank_numpy = int(np.linalg.matrix_rank(X_scaled))
    s = np.linalg.svd(X_scaled, compute_uv=False)
    s_rel = s / s[0]

    print(f"\n========== {label} ==========")
    print(f"shape: {X.shape}")
    print(f"NumPy rank after row scaling: {rank_numpy}")
    print(f"first {min(n_print, len(s_rel))} relative singular values:")
    print(s_rel[:n_print])

    # A large discontinuity can be informative even when the formal numerical
    # rank is larger because of tiny floating-point tails.
    if len(s_rel) > 1:
        ratios = np.divide(
            s_rel[1:], s_rel[:-1], out=np.ones_like(s_rel[1:]), where=s_rel[:-1] > 0
        )
        drop_idx = int(np.argmin(ratios)) + 1
        print(
            "largest adjacent singular-value drop: "
            f"s[{drop_idx}] -> s[{drop_idx + 1}] "
            f"({s_rel[drop_idx - 1]:.3e} -> {s_rel[drop_idx]:.3e})"
        )

    return rank_numpy, s_rel


# %% ========================================================================
#                    Atlas: Schaefer 2018, 200 parcels
# ===========================================================================
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
download_schaefer_200(subjects_dir)

labels_fsaverage = mne.read_labels_from_annot(
    "fsaverage",
    parc="Schaefer2018_200Parcels_7Networks_order",
    subjects_dir=subjects_dir,
    hemi="both",
)

labels_fsaverage = [
    label
    for label in labels_fsaverage
    if "unknown" not in label.name and "Medial_Wall" not in label.name
]

labels_subject = mne.morph_labels(
    labels=labels_fsaverage,
    subject_from="fsaverage",
    subject_to=subject_id,
    subjects_dir=subjects_dir,
)

print(f"Morphed {len(labels_subject)} labels to subject {subject_id}")

roi_names = np.asarray([label.name for label in labels_subject], dtype=str)

if len(roi_names) != 200:
    print(
        "WARNING: expected 200 Schaefer parcels after filtering, "
        f"but obtained {len(roi_names)}."
    )

np.save(Path(out_dir) / "roi_names.npy", roi_names)
# %% ========================================================================
#                              MEG preprocessing
# ===========================================================================
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw_empty = mne.io.read_raw_fif(empty_fname, preload=True)

# --------------------------------------------------------------------------
# Bad channels
# --------------------------------------------------------------------------
# Ideally identify bad channels before Maxwell filtering. The experimental
# file is already SSS-processed, so preserve its existing bad-channel list.
manual_bads_raw = []
manual_bads_empty = []

if manual_bads_raw:
    raw.info["bads"] = sorted(set(raw.info["bads"]) | set(manual_bads_raw))

if manual_bads_empty:
    raw_empty.info["bads"] = sorted(
        set(raw_empty.info["bads"]) | set(manual_bads_empty)
    )

print(f"Raw bad channels before picking: {raw.info['bads']}")
print(f"Empty-room bad channels before picking: {raw_empty.info['bads']}")


# --------------------------------------------------------------------------
# Maxwell / SSS handling
# --------------------------------------------------------------------------
if APPLY_MAXWELL_TO_RAW:
    raw = mne.preprocessing.maxwell_filter(
        raw, st_duration=10.0, origin="auto", verbose=False
    )
else:
    print(
        "Experimental raw is assumed to be already SSS/MaxFilter processed; "
        "not running Maxwell filtering again."
    )

if APPLY_MAXWELL_TO_EMPTY:
    raw_empty = mne.preprocessing.maxwell_filter(
        raw_empty,
        st_duration=EMPTY_ST_DURATION,
        origin="auto",
        coord_frame="meg",
        verbose=False,
    )
else:
    print(
        "Skipping empty-room Maxwell filtering because " "APPLY_MAXWELL_TO_EMPTY=False."
    )


# --------------------------------------------------------------------------
# Keep only usable MEG channels.
# Do not interpolate bad MEG channels here; exclude them consistently instead.
# --------------------------------------------------------------------------
raw.pick(picks="meg", exclude="bads")
raw_empty.pick(picks="meg", exclude="bads")

common_meg = [ch for ch in raw.ch_names if ch in raw_empty.ch_names]

if not common_meg:
    raise RuntimeError(
        "Experimental raw and empty-room data have no common MEG channels."
    )

if len(common_meg) != len(raw.ch_names) or len(common_meg) != len(raw_empty.ch_names):
    print(
        "WARNING: raw and empty-room channel sets differ. "
        f"Using {len(common_meg)} common MEG channels."
    )

raw.pick(common_meg)
raw_empty.pick(common_meg)

if raw_empty.ch_names != raw.ch_names:
    raw_empty.reorder_channels(raw.ch_names)


# --------------------------------------------------------------------------
# Broadband high-pass to remove very slow drift.
# The lowest target band begins at 1 Hz, so 0.5 Hz preserves it.
# --------------------------------------------------------------------------
if BROADBAND_HIGHPASS is not None:
    raw.filter(
        l_freq=BROADBAND_HIGHPASS,
        h_freq=None,
        fir_design="firwin",
        phase="zero",
        verbose=False,
    )

    raw_empty.filter(
        l_freq=BROADBAND_HIGHPASS,
        h_freq=None,
        fir_design="firwin",
        phase="zero",
        verbose=False,
    )


# --------------------------------------------------------------------------
# Mains-noise removal.
# --------------------------------------------------------------------------
notch_freqs_valid = [f for f in NOTCH_FREQS if f < raw.info["sfreq"] / 2]

if notch_freqs_valid:
    raw.notch_filter(freqs=notch_freqs_valid, fir_design="firwin", verbose=False)
    raw_empty.notch_filter(freqs=notch_freqs_valid, fir_design="firwin", verbose=False)


# Optional ICA / SSP for ECG/EOG should be performed here when appropriate.
# It is intentionally not automated because component selection requires
# dataset-specific QC.


# --------------------------------------------------------------------------
# Sensor-rank diagnostics
# --------------------------------------------------------------------------
raw_rank_est, raw_rank_info, empty_rank_est, empty_rank_info = print_rank_diagnostics(
    raw, raw_empty
)

# We use the empty-room info rank explicitly for noise covariance / whitening.
# For the user's current dataset this is expected to be approximately {'meg': 73}.
noise_rank = empty_rank_info

print(f"\nRank used for noise covariance / inverse whitening: {noise_rank}")


# %% ========================================================================
#                       Source space / BEM / forward
# ===========================================================================
src_fname = Path(out_dir) / f"{subject_id}_{SOURCE_SPACING}-src.fif"

if src_fname.exists():
    print(f"Loading cached source space: {src_fname}")
    src = mne.read_source_spaces(src_fname)
else:
    print(f"Creating source space ({SOURCE_SPACING})...")
    src = mne.setup_source_space(
        subject_id, spacing=SOURCE_SPACING, subjects_dir=subjects_dir, add_dist="patch"
    )
    mne.write_source_spaces(src_fname, src, overwrite=True)


# --------------------------------------------------------------------------
# BEM solution
# --------------------------------------------------------------------------
bem_dir = Path(subjects_dir) / subject_id / "bem"
watershed_dir = bem_dir / "watershed"
bem_sol_fname = bem_dir / f"{subject_id}-bem-sol.fif"

try:
    if bem_sol_fname.exists():
        print(f"Loading existing BEM solution: {bem_sol_fname}")
        bem = mne.read_bem_solution(bem_sol_fname)

    else:
        if not watershed_dir.exists():
            print(f"Creating watershed BEM surfaces for {subject_id}...")
            mne.bem.make_watershed_bem(
                subject_id, subjects_dir=subjects_dir, overwrite=False
            )
        else:
            print(f"Watershed BEM surfaces already exist: {watershed_dir}")

        print("Creating single-layer MEG BEM model...")
        bem_model = mne.make_bem_model(
            subject=subject_id, ico=4, conductivity=(0.3,), subjects_dir=subjects_dir
        )

        bem = mne.make_bem_solution(bem_model)

        print(f"Saving BEM solution: {bem_sol_fname}")
        mne.write_bem_solution(bem_sol_fname, bem, overwrite=True)

except Exception as exc:
    print(
        "BEM creation/loading failed. Falling back to an automatically "
        f"fitted sphere model.\nReason: {exc}"
    )

    bem = mne.make_sphere_model(r0="auto", head_radius="auto", info=raw.info)


# --------------------------------------------------------------------------
# Coregistration
# --------------------------------------------------------------------------
if not Path(trans_fname).exists():
    raise RuntimeError(
        "Cannot find trans file. Run mne.gui.coregistration first:\n" f"{trans_fname}"
    )

trans = mne.read_trans(trans_fname)


# --------------------------------------------------------------------------
# Forward solution -- compute once and cache.
# --------------------------------------------------------------------------
fwd_fname = Path(out_dir) / f"{subject_id}_{SOURCE_SPACING}-fwd.fif"

if fwd_fname.exists():
    print(f"Loading cached forward solution: {fwd_fname}")
    fwd = mne.read_forward_solution(fwd_fname)
else:
    print("Computing forward solution...")

    fwd = mne.make_forward_solution(
        raw.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=4,
    )

    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)

    print(f"Forward solution saved: {fwd_fname}")


# %% ========================================================================
#                 Noise covariance / inverse / source estimate
# ===========================================================================
# Include key preprocessing choices in cache filenames so old incompatible
# covariance/inverse files are not accidentally reused.
cache_tag = (
    f"hp{safe_tag(BROADBAND_HIGHPASS)}" f"_noiseRank{noise_rank.get('meg', 'auto')}"
)

cov_fname = Path(out_dir) / f"{subject_id}_{cache_tag}_emptyroom-cov.fif"

if cov_fname.exists():
    print(f"Loading cached noise covariance: {cov_fname}")
    noise_cov = mne.read_cov(cov_fname)
else:
    print("Computing empty-room noise covariance...")

    noise_cov = mne.compute_raw_covariance(
        raw_empty, tmin=0, tmax=None, method="empirical", rank=noise_rank, verbose=False
    )

    mne.write_cov(cov_fname, noise_cov, overwrite=True)

    print(f"Noise covariance saved: {cov_fname}")


# --------------------------------------------------------------------------
# Inverse operator -- compute once and cache.
# Explicitly use the empty-room noise rank for whitening.
# --------------------------------------------------------------------------
inv_fname = Path(out_dir) / f"{subject_id}_{SOURCE_SPACING}_{cache_tag}-inv.fif"

if inv_fname.exists():
    print(f"Loading cached inverse operator: {inv_fname}")
    inv_op = read_inverse_operator(inv_fname)
else:
    print("Creating inverse operator...")

    inv_op = make_inverse_operator(
        raw.info,
        fwd,
        noise_cov,
        loose=LOOSE,
        depth=DEPTH,
        rank=noise_rank,
        verbose=False,
    )

    write_inverse_operator(inv_fname, inv_op, overwrite=True)

    print(f"Inverse operator saved: {inv_fname}")


# --------------------------------------------------------------------------
# Broadband continuous source reconstruction -- ONCE.
# --------------------------------------------------------------------------
lambda2 = 1.0 / SNR**2

print("Applying broadband inverse solution...")

stc = apply_inverse_raw(
    raw,
    inv_op,
    lambda2=lambda2,
    method=INVERSE_METHOD,
    pick_ori="normal",
    verbose=False,
)

print(f"STC data shape: {stc.data.shape}")


# %% ========================================================================
#                           ROI time courses
# ===========================================================================
# mean_flip reduces cancellation caused by differently oriented cortical
# surface normals within the same parcel.
#
# allow_empty=False prevents empty parcels from silently entering the pipeline.
print("Extracting Schaefer-200 ROI time courses...")

roi_ts = stc.extract_label_time_course(
    labels_subject, inv_op["src"], mode="mean_flip", allow_empty=False
)

roi_ts = np.asarray(roi_ts)

n_rois, n_times = roi_ts.shape

print(f"Broadband ROI time-series shape: {roi_ts.shape}")

if n_rois != len(roi_names):
    raise RuntimeError(
        f"ROI time-series count ({n_rois}) does not match "
        f"ROI-name count ({len(roi_names)})."
    )

if SAVE_BROADBAND_ROI_TS:
    np.save(Path(out_dir) / f"{subject_id}_schaefer200_roi_ts_broadband.npy", roi_ts)


# --------------------------------------------------------------------------
# Broadband ROI diagnostic only.
# IMPORTANT: this does NOT block envelope extraction.
# --------------------------------------------------------------------------
broadband_rank, broadband_srel = singular_value_diagnostics(
    roi_ts, label="BROADBAND ROI SINGULAR-VALUE DIAGNOSTICS", n_print=20
)


def compute_rank_from_spectrum(
    s_rel,
    mode="spectral_gap",
    fixed_rank=70,
    explained_variance=0.999,
    min_rank=2,
    max_rank=200,
    min_gap_ratio=10.0,
):
    """Select an effective rank from a relative singular-value spectrum."""
    s_rel = np.asarray(s_rel, dtype=float)
    s_rel = s_rel[np.isfinite(s_rel) & (s_rel >= 0)]
    if s_rel.size == 0 or s_rel[0] <= 0:
        raise RuntimeError("Invalid singular-value spectrum.")

    max_rank = min(int(max_rank), s_rel.size)
    min_rank = max(1, min(int(min_rank), max_rank))

    if mode == "fixed":
        r = int(fixed_rank)
    elif mode == "explained_variance":
        ev = np.cumsum(s_rel**2) / np.sum(s_rel**2)
        r = int(np.searchsorted(ev, explained_variance) + 1)
    elif mode == "spectral_gap":
        if s_rel.size == 1:
            r = 1
        else:
            ratios = s_rel[:-1] / np.maximum(s_rel[1:], np.finfo(float).tiny)
            # Search only within the permitted rank interval.
            lo = min_rank - 1
            hi = max_rank - 1
            idx = lo + int(np.argmax(ratios[lo:hi]))
            best_ratio = float(ratios[idx])
            r = idx + 1 if best_ratio >= min_gap_ratio else None
            if r is None:
                ev = np.cumsum(s_rel**2) / np.sum(s_rel**2)
                r = int(np.searchsorted(ev, explained_variance) + 1)
    else:
        raise ValueError(f"Unknown ORTH_RANK_MODE={mode!r}")

    r = max(min_rank, min(int(r), max_rank))
    return r


def row_standardize_for_rank(X):
    """Center and RMS-standardize rows for rank/eigenvalue diagnostics."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(Xc**2, axis=1))
    if np.any(rms <= np.finfo(float).eps):
        bad = np.where(rms <= np.finfo(float).eps)[0]
        raise RuntimeError(f"Constant/near-constant ROI time series: {bad.tolist()}")
    return Xc / rms[:, None], rms


def global_rank_aware_symmetric_orthogonalize(
    X,
    rank_mode=ORTH_RANK_MODE,
    fixed_rank=ORTH_FIXED_RANK,
    explained_variance=ORTH_EXPLAINED_VARIANCE,
    min_rank=ORTH_MIN_RANK,
    max_rank=ORTH_MAX_RANK,
    min_gap_ratio=ORTH_MIN_GAP_RATIO,
    preserve_row_rms=ORTH_PRESERVE_ROW_RMS,
):
    """Rank-aware global symmetric (Löwdin) orthogonalization.

    Parameters
    ----------
    X : ndarray, shape (n_rois, n_times)
        Band-limited ROI signals.

    Returns
    -------
    X_orth : ndarray, shape (n_rois, n_times)
        Rank-aware globally symmetric-orthogonalized ROI signals. All ROI rows
        are retained; their effective rank is r.
    info : dict
        Rank, singular spectrum, explained variance and row scales.

    Notes
    -----
    For centered X = U_r S_r V_r^T, the truncated Löwdin transform is

        X_orth = U_r V_r^T.

    This equals the Moore-Penrose/pseudoinverse version of
    (X X^T)^(-1/2) X on the retained subspace. Because r < n_rois here, the
    resulting 200 x 200 row Gram matrix is a rank-r projector rather than I.
    Optional row-RMS rescaling restores the original ROI amplitudes.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D ROI x time matrix.")

    Xs, row_rms = row_standardize_for_rank(X)
    n_rois, n_times = Xs.shape

    # Row covariance is only n_rois x n_rois. Its eigendecomposition gives the
    # squared singular values and left singular vectors without forming a full
    # 200 x 574000 SVD.
    C = (Xs @ Xs.T) / max(n_times - 1, 1)
    C = 0.5 * (C + C.T)
    eigvals, U = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    U = U[:, order]
    eigvals = np.maximum(eigvals, 0.0)
    s = np.sqrt(eigvals)
    s_rel = s / max(s[0], np.finfo(float).eps)

    r = compute_rank_from_spectrum(
        s_rel,
        mode=rank_mode,
        fixed_rank=fixed_rank,
        explained_variance=explained_variance,
        min_rank=min_rank,
        max_rank=max_rank,
        min_gap_ratio=min_gap_ratio,
    )

    # U_r^T Xs = S_r V_r^T, hence V_r^T = S_r^{-1} U_r^T Xs.
    Ur = U[:, :r]
    sr = s[:r]
    if np.any(sr <= np.finfo(float).eps):
        raise RuntimeError("Selected rank contains a zero singular value.")

    VrT = (Ur.T @ Xs) / sr[:, None]
    X_orth = Ur @ VrT

    # The rank-aware Löwdin solution has unit-scale rows in aggregate, but its
    # row RMS is not generally identical to the input. Restore original row RMS
    # if requested, following the usual power-preservation motivation.
    if preserve_row_rms:
        orth_rms = np.sqrt(np.mean(X_orth**2, axis=1))
        if np.any(orth_rms <= np.finfo(float).eps):
            bad = np.where(orth_rms <= np.finfo(float).eps)[0]
            raise RuntimeError(f"Zero-RMS orthogonalized ROI rows: {bad.tolist()}")
        X_orth = X_orth * (row_rms / orth_rms)[:, None]

    # Center again after the numerical operations.
    X_orth -= X_orth.mean(axis=1, keepdims=True)

    ev = np.cumsum(s_rel**2) / np.sum(s_rel**2)
    if r < len(s_rel):
        gap_ratio = s_rel[r - 1] / max(s_rel[r], np.finfo(float).tiny)
    else:
        gap_ratio = np.inf

    info = {
        "rank": int(r),
        "singular_values": s,
        "singular_values_relative": s_rel,
        "explained_variance": ev,
        "gap_ratio_at_rank": float(gap_ratio),
        "row_rms_original": row_rms,
        "n_rois": int(n_rois),
        "n_times": int(n_times),
    }
    return X_orth, info


def compute_power_envelope_from_band_signal(
    band_ts, sfreq, envelope_lpf=1.0, envelope_sfreq=10.0, use_power=True
):
    """Hilbert -> amplitude/power envelope -> LPF -> downsample."""
    sfreq = float(sfreq)
    analytic = hilbert(band_ts, axis=-1)
    envelope = np.abs(analytic)
    if use_power:
        envelope = envelope**2

    if envelope_lpf is not None:
        envelope = mne.filter.filter_data(
            envelope,
            sfreq=sfreq,
            l_freq=None,
            h_freq=float(envelope_lpf),
            method="fir",
            fir_design="firwin",
            phase="zero",
            verbose=False,
        )

    final_sfreq = sfreq
    if envelope_sfreq is not None:
        target = float(envelope_sfreq)
        ratio = Fraction(target / sfreq).limit_denominator(10000)
        envelope = resample_poly(
            envelope, up=ratio.numerator, down=ratio.denominator, axis=-1
        )
        final_sfreq = sfreq * ratio.numerator / ratio.denominator

    # The power/amplitude envelope is non-negative by construction (|a| or
    # |a|^2). The zero-phase 1-Hz low-pass and the resampling filter are linear
    # and can ring slightly below zero where the envelope is small (Gibbs
    # undershoot), which is unphysical for a power envelope and would produce
    # NaNs in the downstream log transform. Clip at zero so the returned
    # envelope is a valid non-negative power/amplitude envelope.
    envelope = np.maximum(envelope, 0.0)

    return envelope, final_sfreq


# %% ========================================================================
#                 Rank-aware global symmetric orthogonalization
# ===========================================================================
sfreq = float(raw.info["sfreq"])
n_rois, n_times = roi_ts.shape

if n_rois != 200:
    raise RuntimeError(f"Expected 200 Schaefer ROIs, found {n_rois}.")

print("\n============================================================")
print("RANK-AWARE GLOBAL SYMMETRIC ORTHOGONALIZATION")
print("============================================================")
print(f"ROIs: {n_rois}")
print(f"Input sampling frequency: {sfreq:.6f} Hz")
print(f"Rank selection mode: {ORTH_RANK_MODE}")
print("Connectivity: NOT calculated")

# Broadband diagnostic only. It is not used to force one rank across bands.
broadband_rank, broadband_srel = singular_value_diagnostics(
    roi_ts, label="BROADBAND ROI DIAGNOSTICS", n_print=20
)

diagnostic_records = {}

for band_name, (l_freq, h_freq) in bands.items():
    print(
        f"\n============================================================\n"
        f"Processing {band_name}: {l_freq}-{h_freq} Hz\n"
        f"============================================================"
    )

    # 1. Band-pass ROI source signals.
    band_ts = mne.filter.filter_data(
        roi_ts,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",
        phase="zero",
        verbose=False,
    )

    # 2. Diagnose rank and spectral gap BEFORE orthogonalization.
    band_rank_numpy, band_srel = singular_value_diagnostics(
        band_ts, label=f"{band_name.upper()} ROI DIAGNOSTICS", n_print=20
    )

    # 3. Global rank-aware symmetric/Löwdin orthogonalization.
    band_ts_orth, orth_info = global_rank_aware_symmetric_orthogonalize(band_ts)
    selected_rank = orth_info["rank"]

    print(f"{band_name}: NumPy numerical rank      = {band_rank_numpy}")
    print(f"{band_name}: selected effective rank   = {selected_rank}")
    print(
        f"{band_name}: gap ratio at selected rank = {orth_info['gap_ratio_at_rank']:.3e}"
    )
    print(f"{band_name}: band ROI shape             = {band_ts.shape}")
    print(f"{band_name}: orth ROI shape             = {band_ts_orth.shape}")

    # QC: row-wise RMS preservation.
    rms_before = np.sqrt(np.mean(band_ts**2, axis=1))
    rms_after = np.sqrt(np.mean(band_ts_orth**2, axis=1))
    rms_rel_error = np.max(
        np.abs(rms_after - rms_before) / np.maximum(rms_before, np.finfo(float).eps)
    )
    print(f"{band_name}: max relative row-RMS error = {rms_rel_error:.3e}")

    # 4. Hilbert power envelope AFTER leakage correction.
    envelope, env_sfreq = compute_power_envelope_from_band_signal(
        band_ts_orth,
        sfreq=sfreq,
        envelope_lpf=ENVELOPE_LPF,
        envelope_sfreq=ENVELOPE_SFREQ,
        use_power=USE_POWER,
    )

    print(f"{band_name}: power-envelope shape      = {envelope.shape}")
    print(f"{band_name}: envelope sfreq             = {env_sfreq:.6f} Hz")

    if SAVE_BAND_ROI_TS:
        np.save(Path(out_dir) / f"{subject_id}_{band_name}_band_roi_ts.npy", band_ts)
        np.save(
            Path(out_dir) / f"{subject_id}_{band_name}_band_roi_ts_global_orth.npy",
            band_ts_orth,
        )

    # Simple downstream .npy file: 200 x T_envelope.
    np.save(Path(out_dir) / f"power_envelope_{band_name}_dfreq_none.npy", envelope)

    # Self-describing output.
    np.savez_compressed(
        Path(out_dir) / f"power_envelope_{band_name}_dfreq_none.npz",
        power_envelope=envelope,
        roi_names=roi_names,
        envelope_sfreq=np.asarray(env_sfreq),
        original_sfreq=np.asarray(sfreq),
        l_freq=np.asarray(float(l_freq)),
        h_freq=np.asarray(float(h_freq)),
        envelope_lpf=np.asarray(float(ENVELOPE_LPF)),
        use_power=np.asarray(USE_POWER),
        subject_id=np.asarray(subject_id),
        atlas=np.asarray("Schaefer2018_200Parcels_7Networks_order"),
        orthogonalization=np.asarray("rank_aware_global_symmetric_lowdin"),
        selected_rank=np.asarray(selected_rank),
        rank_selection_mode=np.asarray(ORTH_RANK_MODE),
        gap_ratio_at_rank=np.asarray(orth_info["gap_ratio_at_rank"]),
        connectivity_calculated=np.asarray(False),
    )

    diagnostic_records[band_name] = {
        "numpy_rank": int(band_rank_numpy),
        "selected_rank": int(selected_rank),
        "singular_values_relative": band_srel,
        "explained_variance": orth_info["explained_variance"],
        "gap_ratio_at_rank": float(orth_info["gap_ratio_at_rank"]),
    }

    del band_ts, band_ts_orth, envelope


# %% ========================================================================
#                         Save diagnostics / metadata
# ===========================================================================
diag_kwargs = {
    "subject_id": np.asarray(subject_id),
    "roi_names": roi_names,
    "raw_rank_estimated_meg": np.asarray(raw_rank_est.get("meg", np.nan), dtype=float),
    "raw_rank_info_meg": np.asarray(raw_rank_info.get("meg", np.nan), dtype=float),
    "empty_rank_estimated_meg": np.asarray(
        empty_rank_est.get("meg", np.nan), dtype=float
    ),
    "empty_rank_info_meg": np.asarray(empty_rank_info.get("meg", np.nan), dtype=float),
    "noise_rank_used_meg": np.asarray(noise_rank.get("meg", np.nan), dtype=float),
    "broadband_roi_numpy_rank": np.asarray(broadband_rank, dtype=int),
    "broadband_roi_singular_values_relative": broadband_srel,
}

for band_name, record in diagnostic_records.items():
    diag_kwargs[f"{band_name}_roi_numpy_rank"] = np.asarray(
        record["numpy_rank"], dtype=int
    )
    diag_kwargs[f"{band_name}_selected_effective_rank"] = np.asarray(
        record["selected_rank"], dtype=int
    )
    diag_kwargs[f"{band_name}_roi_singular_values_relative"] = record[
        "singular_values_relative"
    ]
    diag_kwargs[f"{band_name}_explained_variance"] = record["explained_variance"]
    diag_kwargs[f"{band_name}_gap_ratio_at_rank"] = np.asarray(
        record["gap_ratio_at_rank"]
    )

np.savez_compressed(
    Path(out_dir) / f"{subject_id}_power_envelope_rank_diagnostics.npz", **diag_kwargs
)

np.savez_compressed(
    Path(out_dir) / f"{subject_id}_power_envelope_metadata.npz",
    subject_id=np.asarray(subject_id),
    atlas=np.asarray("Schaefer2018_200Parcels_7Networks_order"),
    roi_names=roi_names,
    source_spacing=np.asarray(SOURCE_SPACING),
    inverse_method=np.asarray(INVERSE_METHOD),
    snr=np.asarray(SNR),
    loose=np.asarray(LOOSE),
    depth=np.asarray(DEPTH),
    broadband_highpass_hz=np.asarray(BROADBAND_HIGHPASS),
    envelope_type=np.asarray("power" if USE_POWER else "amplitude"),
    envelope_lowpass_hz=np.asarray(ENVELOPE_LPF),
    envelope_target_sfreq_hz=np.asarray(ENVELOPE_SFREQ),
    global_symmetric_orthogonalization=np.asarray(True),
    orthogonalization_method=np.asarray("rank_aware_global_symmetric_lowdin"),
    rank_selection_mode=np.asarray(ORTH_RANK_MODE),
    fixed_rank=np.asarray(ORTH_FIXED_RANK),
    explained_variance_target=np.asarray(ORTH_EXPLAINED_VARIANCE),
    min_rank=np.asarray(ORTH_MIN_RANK),
    max_rank=np.asarray(ORTH_MAX_RANK),
    min_gap_ratio=np.asarray(ORTH_MIN_GAP_RATIO),
    preserve_row_rms=np.asarray(ORTH_PRESERVE_ROW_RMS),
    connectivity_calculated=np.asarray(False),
)

print("\n============================================================")
print("All frequency-band power envelopes have been saved.")
print("Atlas: Schaefer-200")
print("Leakage correction: RANK-AWARE GLOBAL SYMMETRIC (Löwdin)")
print("Connectivity: NOT calculated")
print("============================================================")
