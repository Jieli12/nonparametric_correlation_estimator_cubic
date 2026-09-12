"""
Author      : Jie Li
Date        : 2026-08-16 22:16:03
Last Edited : 2026-09-11 13:53:04
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/data_to_dataframe_boundary.py
Description :

Boundary-MSE aware conversion of experiment result pickles to a tidy
pandas DataFrame.

This module mirrors `result_utils.data_to_dataframe` and adds an extra
column `mse_b` (boundary MSE) computed only over the *boundary* test
points:

    MSE_B = (1 / m_B) * sum_{u_j^test in (-1, -t] U [t, 1)}
                        (rho_hat(u_j^test) - rho(u_j^test))^2

where t = `boundary` (default 0.9) and m_B is the number of test points
that fall inside the boundary regions.

Because the raw pickles store the estimated curve (`rho_hat`) and the
test grid (`u_test` in config) but NOT the true curve, the true values
`rho(u_test)` are recomputed on the fly from the correlation function
(see `utils.get_correlation_function`). If that is not possible the
`mse_b` column is filled with NaN.

Columns returned:
    corr_fun, kernel, sample_strategy, method, sample_size, rep_idx,
    mse, mse_b, h_opt, time_bandwidth, time_u_test

Copyright (c) 2026, Jie Li, lijie_12@outlook.com
All Rights Reserved.
"""

from typing import Any

import numpy as np
import pandas as pd

BASE_COLUMNS = [
    "corr_fun",
    "kernel",
    "sample_strategy",
    "method",
    "sample_size",
    "rep_idx",
    "mse",
    "h_opt",
    "time_bandwidth",
    "time_u_test",
]

ALL_COLUMNS = BASE_COLUMNS[:7] + ["mse_b"] + BASE_COLUMNS[7:]


# =============================================================================
#  boundary helpers
# =============================================================================
def boundary_mask(u, boundary: float = 0.9) -> np.ndarray:
    """Boolean mask of boundary test points in (-1, -t] U [t, 1).

    Parameters
    ----------
    u : array_like
        Test points (should lie in [-1, 1]).
    boundary : float
        Threshold t (default 0.9).

    Returns
    -------
    np.ndarray[bool]
        True where ``u`` is inside either boundary region.
    """
    u = np.asarray(u, dtype=float)
    return ((u > -1.0) & (u <= -boundary)) | ((u >= boundary) & (u < 1.0))


def compute_mse_b(
    rho_hat: Any, u_test: Any, rho_true: Any, boundary: float = 0.9
) -> float:
    """Boundary MSE over test points in (-1, -t] U [t, 1).

    ``mse_b = mean_{mask} (rho_hat - rho_true)^2``; returns NaN when there
    is no boundary point or when the inputs are unusable.

    Parameters
    ----------
    rho_hat : array_like
        Estimated curve at every test point (same length as ``u_test``).
    u_test : array_like
        Test points.
    rho_true : array_like
        True curve at every test point (same length as ``u_test``).
    boundary : float
        Boundary threshold t (default 0.9).

    Returns
    -------
    float
        Boundary MSE, or NaN.
    """
    rho_hat = np.asarray(rho_hat, dtype=float)
    u_test = np.asarray(u_test, dtype=float)
    rho_true = np.asarray(rho_true, dtype=float)

    if rho_hat.ndim == 0 or rho_hat.size == 0 or u_test.size == 0:
        return np.nan
    if rho_hat.size != u_test.size or rho_true.size != u_test.size:
        return np.nan
    if not np.all(np.isfinite(rho_hat)) or not np.all(np.isfinite(rho_true)):
        return np.nan

    mask = boundary_mask(u_test, boundary=boundary)
    if not np.any(mask):
        return np.nan
    m_b = int(np.count_nonzero(mask))
    return float(np.mean((rho_hat[mask] - rho_true[mask]) ** 2))


# =============================================================================
#  true-curve helper
# =============================================================================
def _rho_true_from_config(u_test: Any, cfg: dict) -> Any:
    """Recompute rho(u_test) from the correlation function named in ``cfg``.

    Returns a float ndarray on success, else None (callers then use NaN).
    """
    if u_test is None:
        return None
    if not cfg:
        return None
    corr_name = cfg.get("correlation_function") or cfg.get("corr_fun")
    if corr_name is None:
        return None
    try:
        # Lazy import: keeps this module lightweight until actually needed.
        from utils import get_correlation_function

        rho_func = get_correlation_function(corr_name)
        return np.asarray([rho_func.rho(u) for u in u_test], dtype=float)
    except Exception:
        return None


def _row(
    method,
    res,
    sample_size,
    rep_idx,
    corr,
    kernel,
    sample_strategy,
    u_test,
    rho_true,
    boundary,
):
    """Build one tidy row (includes the boundary MSE)."""
    rho_hat = res.get("rho_hat") if isinstance(res, dict) else None
    mse_b = (
        compute_mse_b(rho_hat, u_test, rho_true, boundary=boundary)
        if (rho_hat is not None and rho_true is not None)
        else np.nan
    )
    return {
        "corr_fun": corr,
        "kernel": kernel,
        "sample_strategy": sample_strategy,
        "method": method,
        "sample_size": sample_size,
        "rep_idx": rep_idx,
        "mse": res.get("mse", np.nan) if isinstance(res, dict) else np.nan,
        "mse_b": mse_b,
        "h_opt": res.get("h_opt", np.nan) if isinstance(res, dict) else np.nan,
        "time_bandwidth": (
            res.get("time_bandwidth", np.nan) if isinstance(res, dict) else np.nan
        ),
        "time_u_test": (
            res.get("time_u_test", np.nan) if isinstance(res, dict) else np.nan
        ),
    }


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ALL_COLUMNS)


def _get_cfg(cfg):
    if not cfg:
        return (None, None, None)
    corr = cfg.get("correlation_function") or cfg.get("corr_fun")
    kernel = cfg.get("kernel_function") or cfg.get("kernel")
    sample_strategy = cfg.get("sampling_strategy") or cfg.get("sample_strategy")
    return (corr, kernel, sample_strategy)


# =============================================================================
#  main converter
# =============================================================================
def data_to_dataframe_boundary(data: Any, boundary: float = 0.9) -> pd.DataFrame:
    """Convert experiment result `data` into a tidy DataFrame with `mse_b`.

    Accepts the same formats as `result_utils.data_to_dataframe`:
    - the `all_results` dict produced by `run_comparison`
    - a single-run `load_results` output
    - a list of such runs

    In addition to the columns produced by `data_to_dataframe`, the result
    contains the `mse_b` column (boundary MSE over
    (-1, -t] U [t, 1), with t = `boundary`).

    Parameters
    ----------
    data : Any
        Loaded pickle content.
    boundary : float
        Boundary threshold t (default 0.9).

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns ``ALL_COLUMNS``.
    """
    rows = []

    # List of runs -> concat
    if isinstance(data, list):
        df_list = [data_to_dataframe_boundary(d, boundary=boundary) for d in data]
        if df_list:
            return pd.concat(df_list, ignore_index=True)
        return _empty_frame()

    # all_results format produced by `run_comparison`
    if isinstance(data, dict) and "sample_sizes" in data and "config" in data:
        cfg = data.get("config", {})
        corr, kernel, sample_strategy = _get_cfg(cfg)
        u_test = cfg.get("u_test")
        rho_true = _rho_true_from_config(u_test, cfg)

        for n, reps in data["sample_sizes"].items():
            for rep in reps:
                estimators = rep.get("estimators") or rep.get("results") or {}
                for method, res in estimators.items():
                    rows.append(
                        _row(
                            method,
                            res,
                            int(rep.get("n") or n),
                            rep.get("rep_idx", None),
                            corr,
                            kernel,
                            sample_strategy,
                            u_test,
                            rho_true,
                            boundary,
                        )
                    )
        return pd.DataFrame(rows, columns=ALL_COLUMNS)

    # Single-run format: data['results'][estimator]
    if isinstance(data, dict) and "results" in data:
        cfg = data.get("config", {})
        corr, kernel, sample_strategy = _get_cfg(cfg)
        u_test = cfg.get("u_test")
        rho_true = _rho_true_from_config(u_test, cfg)
        sample_size = cfg.get("n") or cfg.get("sample_size") or None
        for method, res in data["results"].items():
            rows.append(
                _row(
                    method,
                    res,
                    int(sample_size) if sample_size is not None else None,
                    None,
                    corr,
                    kernel,
                    sample_strategy,
                    u_test,
                    rho_true,
                    boundary,
                )
            )
        return pd.DataFrame(rows, columns=ALL_COLUMNS)

    # Fallback: heuristically find estimator dicts inside the structure
    if isinstance(data, dict):
        cfg = data.get("config", {})
        corr, kernel, sample_strategy = _get_cfg(cfg)
        u_test = cfg.get("u_test")
        rho_true = _rho_true_from_config(u_test, cfg)

        def _collect(d, prefix=None):
            if not isinstance(d, dict):
                return
            if any(k in d for k in ("mse", "h_opt", "time_bandwidth", "time_u_test")):
                rows.append(
                    _row(
                        prefix,
                        d,
                        None,
                        None,
                        corr,
                        kernel,
                        sample_strategy,
                        u_test,
                        rho_true,
                        boundary,
                    )
                )
                return
            for k, v in d.items():
                _collect(v, prefix=k if prefix is None else f"{prefix}/{k}")

        _collect(data)
        return pd.DataFrame(rows, columns=ALL_COLUMNS)

    # Nothing matched -> empty DataFrame with expected columns
    return _empty_frame()
