"""
Author      : Jie Li
Date        : 2026-09-11 11:53:56
Last Edited : 2026-09-11 11:58:38
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/compare_estimation_method.py
Description :

Monte Carlo Comparison of Estimation Methods
=============================================

CLI tool for comparing 4 estimation methods (NW, LL, CE, CL) across
different sample sizes using Monte Carlo simulations.

Features:
    - Parallel Monte Carlo repetitions (outer loop parallelization)
    - Sequential grid search within each repetition (avoids nested parallelization)
    - Flexible configuration via command-line arguments
    - Results saved in pickle format for downstream analysis

Usage:
    python compare_estimation_method.py --n_jobs 4 --n_repeats 200 --verbose 1

Copyright (c) 2026, Jie Li, lijie_12@outlook.com
All Rights Reserved.
"""

import argparse
import logging
import os
import pickle
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from utils import (
    BandwidthSelector,
    CubicEquationEstimator,
    CubicWithLLCorrection,
    LocalLinearEstimator,
    NadarayaWatsonEstimator,
    _get_parallel_prefer,
    boundary_dense_sampling,
    get_correlation_function,
    get_kernel,
    uniform_sampling,
)

warnings.filterwarnings("ignore")


# =============================================================================
# Logger Setup
# =============================================================================


def setup_logger(log_file=None, verbose=0):
    """
    Setup logger for both console and file output.

    Parameters
    ----------
    log_file : str or Path, optional
        Path to log file
    verbose : int, default=0
        Verbosity level (0=INFO, 1=DEBUG)

    Returns
    -------
    logger : logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger("compare_estimation")
    logger.setLevel(logging.DEBUG if verbose > 0 else logging.INFO)
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose > 0 else logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# Single Repetition Worker
# =============================================================================


def run_single_repetition(
    rep_idx,
    n_samples,
    rho_func,
    kernel_name,
    u_test,
    h_grid,
    estimator_names,
    seed,
    verbose=0,
):
    """
    Run a single Monte Carlo repetition for all 4 estimation methods.

    This function is designed to be called in parallel across repetitions.
    Within each repetition, grid_search and estimate_batch run sequentially
    (n_jobs=1) to avoid nested parallelization overhead.

    Parameters
    ----------
    rep_idx : int
        Repetition index (0-based)
    n_samples : int
        Sample size for this repetition
    rho_func : CorrelationFunction
        True correlation function
    kernel_name : str
        Kernel function name
    u_test : ndarray
        Test points for evaluation
    h_grid : ndarray
        Grid of bandwidth values for search
    estimators_config : dict
        Dictionary mapping estimator names to estimator instances
    seed : int
        Base random seed (will be offset by rep_idx)
    verbose : int
        Verbosity level

    Returns
    -------
    results : dict
        Results for this repetition:
        {
            'rep_idx': int,
            'n': int,
            'seed_used': int,
            'estimators': {
                'NW': {
                    'h_opt': float,
                    'mse': float,
                    'rho_hat': ndarray,
                    'time_bandwidth': float,
                    'time_u_test': float,
                    'time_total': float,
                },
                'LL': {...},
                'CUBIC': {...},
                'CUBIC+LL': {...}
            }
        }
    """
    # Unique seed for this repetition (ensures reproducibility and independence)
    rep_seed = seed + rep_idx

    # Generate data for this repetition
    U_train, X1_train, X2_train = rho_func.generate_data(
        n=n_samples, random_state=rep_seed
    )

    # True values at test points
    rho_true = np.array([rho_func.rho(u) for u in u_test])

    # Storage for this repetition
    rep_results = {
        "rep_idx": rep_idx,
        "n": n_samples,
        "seed_used": rep_seed,
        "estimators": {},
    }

    # Loop over all specified estimator names and construct instances locally.
    # This avoids sending estimator instances to worker processes (pickling issues).
    for est_name in estimator_names:
        # Map estimator name to constructor
        if est_name == "NW":
            estimator_cls = NadarayaWatsonEstimator
        elif est_name == "LL":
            estimator_cls = LocalLinearEstimator
        elif est_name == "CUBIC":
            estimator_cls = CubicEquationEstimator
        elif est_name == "CUBIC+LL":
            estimator_cls = CubicWithLLCorrection
        else:
            raise ValueError(f"Unknown estimator name: {est_name}")

        # instantiate estimator inside worker to keep it local to the process/thread
        estimator = estimator_cls(kernel=kernel_name)
        try:
            # Bandwidth selection (sequential within this repetition)
            bandwidth_selector = BandwidthSelector(
                U_train=U_train,
                X1_train=X1_train,
                X2_train=X2_train,
                estimator=estimator,
                rho_true_func=rho_func.rho,
            )

            # Grid search with n_jobs=1 (sequential) to avoid nested parallelization
            t_bw_start = time.perf_counter()
            result_bw = bandwidth_selector.grid_search(
                h_grid=h_grid,
                u_test=u_test,
                method="mse",
                n_jobs=1,  # Sequential within each repetition
                verbose=0,
            )
            time_bandwidth = time.perf_counter() - t_bw_start
            h_opt = result_bw["optimal_h"]

            # Estimate at all test points with optimal bandwidth (sequential)
            t_est_start = time.perf_counter()
            rho_hat = estimator.estimate_batch(
                u_test,
                U_train,
                X1_train,
                X2_train,
                h_opt,
                n_jobs=1,  # Sequential within each repetition
                verbose=0,
            )
            time_u_test = time.perf_counter() - t_est_start
            time_total = time_bandwidth + time_u_test

            # Compute MSE
            mse = np.mean((rho_hat - rho_true) ** 2)

            # Store results for this estimator
            rep_results["estimators"][est_name] = {
                "h_opt": float(h_opt),
                "mse": float(mse),
                "rho_hat": rho_hat,  # Keep full array for further analysis
                "time_bandwidth": float(time_bandwidth),
                "time_u_test": float(time_u_test),
                "time_total": float(time_total),
            }

            if verbose > 0:
                print(
                    f"  Timing n={n_samples}, rep={rep_idx}, {est_name}: "
                    f"bw={time_bandwidth:.4f}s, u_test={time_u_test:.4f}s, "
                    f"total={time_total:.4f}s"
                )

        except Exception as e:
            # Log error but don't stop the entire experiment
            if verbose > 0:
                print(f"  Warning: {est_name} failed at rep {rep_idx}: {str(e)}")
            rep_results["estimators"][est_name] = {
                "h_opt": np.nan,
                "mse": np.nan,
                "rho_hat": np.full_like(u_test, np.nan),
                "time_bandwidth": np.nan,
                "time_u_test": np.nan,
                "time_total": np.nan,
                "error": str(e),
            }

    return rep_results


# =============================================================================
# Main Comparison Function
# =============================================================================


def run_comparison(
    n_list,
    n_repeats,
    correlation_function,
    kernel_function,
    sampling_strategy,
    number_test,
    h_grid_start,
    h_grid_end,
    h_grid_num,
    n_jobs,
    seed,
    verbose,
    logger,
):
    """
    Run Monte Carlo comparison across all sample sizes.

    Parallelization strategy:
    - Outer loop: Parallelize over Monte Carlo repetitions (most efficient)
    - Inner loop: Sequential grid_search and estimate_batch (avoids nesting)

    Parameters
    ----------
    n_list : list of int
        Sample sizes to test
    n_repeats : int
        Number of Monte Carlo repetitions per sample size
    correlation_function : str
        Name of correlation function
    kernel_function : str
        Name of kernel function
    sampling_strategy : str
        Sampling strategy for test points
    number_test : int
        Number of test points
    h_grid_start : float
        Start of logspace for bandwidth grid
    h_grid_end : float
        End of logspace for bandwidth grid
    h_grid_num : int
        Number of points in bandwidth grid
    n_jobs : int
        Number of parallel jobs for Monte Carlo repetitions
    seed : int
        Base random seed
    verbose : int
        Verbosity level
    logger : logging.Logger
        Logger instance

    Returns
    -------
    all_results : dict
        Nested dictionary with all results:
        {
            'config': {...},
            'sample_sizes': {
                100: [rep_0_results, rep_1_results, ...],
                200: [...],
                ...
            },
            'timing_summary': {
                100: {
                    'NW': {'sum_bandwidth': ..., 'sum_u_test': ..., 'sum_total': ...},
                    ...
                },
                ...
            }
        }
    """
    logger.info("=" * 78)
    logger.info("MONTE CARLO COMPARISON OF ESTIMATION METHODS")
    logger.info("=" * 78)
    logger.info("")

    # Configuration summary
    logger.info("Configuration:")
    logger.info(f"  Sample sizes: {n_list}")
    logger.info(f"  Monte Carlo repetitions: {n_repeats}")
    logger.info(f"  Correlation function: {correlation_function}")
    logger.info(f"  Kernel function: {kernel_function}")
    logger.info(f"  Sampling strategy: {sampling_strategy}")
    logger.info(f"  Number of test points: {number_test}")
    logger.info(
        f"  Bandwidth grid: logspace({h_grid_start}, {h_grid_end}, {h_grid_num})"
    )
    logger.info(f"  Parallel jobs: {n_jobs}")

    # Detect host OS and pick a sensible default parallel backend
    platform_system = platform.system().lower()
    if platform_system == "darwin":
        detected_os = "macOS"
        default_backend = "threads"
    elif platform_system == "linux":
        detected_os = "Linux"
        default_backend = "processes"
    elif platform_system.startswith("win"):
        detected_os = "Windows"
        default_backend = "processes"
    else:
        detected_os = platform.system()
        default_backend = "threads"

    # Allow environment override via NCE_PARALLEL_BACKEND
    env_pref = os.environ.get("NCE_PARALLEL_BACKEND", None)
    if env_pref:
        if env_pref.lower() == "auto":
            prefer_backend = default_backend
            env_note = "(env: auto)"
        elif env_pref.lower() in ("threads", "processes"):
            prefer_backend = env_pref.lower()
            env_note = f"(env: {env_pref.lower()})"
        else:
            prefer_backend = default_backend
            env_note = f"(env invalid: {env_pref})"
    else:
        prefer_backend = default_backend
        env_note = ""

    logger.info(f"  Host OS detected: {detected_os}")
    logger.info(f"  Parallel backend chosen: {prefer_backend} {env_note}")
    logger.info(f"  Random seed: {seed}")
    logger.info("")

    # Initialize correlation function and kernel
    try:
        rho_func = get_correlation_function(correlation_function)
        logger.info(f"✓ Correlation function loaded: {rho_func.name}")
        logger.info(f"  Formula: {rho_func.formula}")
    except Exception as e:
        logger.error(f"✗ Failed to load correlation function: {e}")
        raise

    try:
        kernel = get_kernel(kernel_function)
        logger.info(f"✓ Kernel function loaded: {kernel.name}")
    except Exception as e:
        logger.error(f"✗ Failed to load kernel function: {e}")
        raise

    logger.info("")

    # Create test points (same for all repetitions and sample sizes)
    if sampling_strategy == "uniform":
        u_test = uniform_sampling(n_points=number_test)
    elif sampling_strategy == "boundary":
        # Boundary-dense: 25% left, 50% center, 25% right
        n_left = int(number_test * 0.25)
        n_center = int(number_test * 0.50)
        n_right = number_test - n_left - n_center
        u_test = boundary_dense_sampling(
            n_boundary_left=n_left,
            n_interior=n_center,
            n_boundary_right=n_right,
            boundary_threshold=0.9,
        )
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    logger.info(f"✓ Created {len(u_test)} test points ({sampling_strategy} strategy)")
    logger.info(f"  Range: [{u_test.min():.4f}, {u_test.max():.4f}]")
    logger.info("")

    # Create bandwidth grid (same for all repetitions and sample sizes)
    h_grid = np.logspace(h_grid_start, h_grid_end, h_grid_num)
    logger.info(f"✓ Created bandwidth grid with {len(h_grid)} points")
    logger.info(f"  Range: [{h_grid.min():.6f}, {h_grid.max():.6f}]")
    logger.info("")

    # Estimator names - actual instances are created inside worker to avoid pickling
    estimator_names = ["NW", "LL", "CUBIC", "CUBIC+LL"]
    logger.info("✓ Prepared estimator names: NW, LL, CUBIC, CUBIC+LL")
    logger.info("")

    # Storage for all results
    all_results = {
        "config": {
            "n_list": n_list,
            "n_repeats": n_repeats,
            "correlation_function": correlation_function,
            "kernel_function": kernel_function,
            "sampling_strategy": sampling_strategy,
            "number_test": number_test,
            "h_grid_start": h_grid_start,
            "h_grid_end": h_grid_end,
            "h_grid_num": h_grid_num,
            "n_jobs": n_jobs,
            "seed": seed,
            "u_test": u_test,
            "h_grid": h_grid,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "sample_sizes": {},
        "timing_summary": {},
    }

    # Loop over sample sizes
    total_start_time = time.time()

    for n_idx, n_samples in enumerate(n_list):
        logger.info("=" * 78)
        logger.info(f"Sample size {n_idx + 1}/{len(n_list)}: n = {n_samples}")
        logger.info("=" * 78)

        size_start_time = time.time()

        # Parallel execution over Monte Carlo repetitions
        logger.info(
            f"Running {n_repeats} Monte Carlo repetitions in parallel (n_jobs={n_jobs})..."
        )

        if n_jobs == 1:
            # Sequential execution
            rep_results_list = []
            iterator = (
                tqdm(range(n_repeats), desc=f"n={n_samples}")
                if TQDM_AVAILABLE
                else range(n_repeats)
            )
            for rep_idx in iterator:
                rep_result = run_single_repetition(
                    rep_idx=rep_idx,
                    n_samples=n_samples,
                    rho_func=rho_func,
                    kernel_name=kernel_function,
                    u_test=u_test,
                    h_grid=h_grid,
                    estimator_names=estimator_names,
                    seed=seed,
                    verbose=verbose,
                )
                rep_results_list.append(rep_result)
        else:
            # Parallel execution over repetitions
            try:
                rep_results_list = Parallel(
                    n_jobs=n_jobs, prefer=prefer_backend, verbose=0
                )(
                    delayed(run_single_repetition)(
                        rep_idx=rep_idx,
                        n_samples=n_samples,
                        rho_func=rho_func,
                        kernel_name=kernel_function,
                        u_test=u_test,
                        h_grid=h_grid,
                        estimator_names=estimator_names,
                        seed=seed,
                        verbose=verbose,
                    )
                    for rep_idx in (
                        tqdm(range(n_repeats), desc=f"n={n_samples}")
                        if TQDM_AVAILABLE
                        else range(n_repeats)
                    )
                )
            except Exception:
                # Fallback to threads if processes fail
                if prefer_backend == "processes":
                    logger.warning("  Process backend failed, retrying with threads...")
                    rep_results_list = Parallel(
                        n_jobs=n_jobs, prefer="threads", verbose=0
                    )(
                        delayed(run_single_repetition)(
                            rep_idx=rep_idx,
                            n_samples=n_samples,
                            rho_func=rho_func,
                            kernel_name=kernel_function,
                            u_test=u_test,
                            h_grid=h_grid,
                            estimator_names=estimator_names,
                            seed=seed,
                            verbose=verbose,
                        )
                        for rep_idx in range(n_repeats)
                    )
                else:
                    raise

        size_elapsed = time.time() - size_start_time

        # Store results for this sample size
        all_results["sample_sizes"][n_samples] = rep_results_list

        # Aggregate timing for each (sample size, estimator) combination
        timing_summary_n = {}
        for est_name in ["NW", "LL", "CUBIC", "CUBIC+LL"]:
            bw_values = [
                rep["estimators"][est_name]["time_bandwidth"]
                for rep in rep_results_list
                if not np.isnan(rep["estimators"][est_name]["time_bandwidth"])
            ]
            u_test_values = [
                rep["estimators"][est_name]["time_u_test"]
                for rep in rep_results_list
                if not np.isnan(rep["estimators"][est_name]["time_u_test"])
            ]
            total_values = [
                rep["estimators"][est_name]["time_total"]
                for rep in rep_results_list
                if not np.isnan(rep["estimators"][est_name]["time_total"])
            ]

            timing_summary_n[est_name] = {
                "sum_bandwidth": float(np.sum(bw_values)) if bw_values else np.nan,
                "sum_u_test": float(np.sum(u_test_values)) if u_test_values else np.nan,
                "sum_total": float(np.sum(total_values)) if total_values else np.nan,
                "mean_bandwidth": float(np.mean(bw_values)) if bw_values else np.nan,
                "mean_u_test": (
                    float(np.mean(u_test_values)) if u_test_values else np.nan
                ),
                "mean_total": float(np.mean(total_values)) if total_values else np.nan,
            }

        all_results["timing_summary"][n_samples] = timing_summary_n

        # Compute summary statistics
        logger.info("")
        logger.info(f"✓ Completed {n_repeats} repetitions in {size_elapsed:.2f}s")
        logger.info("")
        logger.info("Summary statistics (MSE):")
        logger.info(
            f"  {'Estimator':<12} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}"
        )
        logger.info("  " + "-" * 60)

        for est_name in ["NW", "LL", "CUBIC", "CUBIC+LL"]:
            mse_values = [
                rep["estimators"][est_name]["mse"]
                for rep in rep_results_list
                if not np.isnan(rep["estimators"][est_name]["mse"])
            ]
            if len(mse_values) > 0:
                logger.info(
                    f"  {est_name:<12} {np.mean(mse_values):>12.6f} {np.std(mse_values):>12.6f} "
                    f"{np.min(mse_values):>12.6f} {np.max(mse_values):>12.6f}"
                )
            else:
                logger.info(
                    f"  {est_name:<12} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}"
                )

        logger.info("")
        logger.info("Timing summary by (n, method):")
        logger.info(
            f"  {'Estimator':<12} {'Sum BW(s)':>12} {'Sum U(s)':>12} {'Sum Total(s)':>14} "
            f"{'Mean BW(s)':>12} {'Mean U(s)':>12} {'Mean Total(s)':>14}"
        )
        logger.info("  " + "-" * 96)

        for est_name in ["NW", "LL", "CUBIC", "CUBIC+LL"]:
            stats = timing_summary_n[est_name]
            if np.isnan(stats["sum_total"]):
                logger.info(
                    f"  {est_name:<12} {'N/A':>12} {'N/A':>12} {'N/A':>14} "
                    f"{'N/A':>12} {'N/A':>12} {'N/A':>14}"
                )
            else:
                logger.info(
                    f"  {est_name:<12} {stats['sum_bandwidth']:>12.4f} {stats['sum_u_test']:>12.4f} "
                    f"{stats['sum_total']:>14.4f} {stats['mean_bandwidth']:>12.4f} "
                    f"{stats['mean_u_test']:>12.4f} {stats['mean_total']:>14.4f}"
                )

        logger.info("")

    total_elapsed = time.time() - total_start_time
    logger.info("=" * 78)
    logger.info("MONTE CARLO COMPARISON COMPLETED")
    logger.info("=" * 78)
    logger.info(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)")
    logger.info(f"Total repetitions: {len(n_list) * n_repeats}")
    logger.info("")

    return all_results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo Comparison of Estimation Methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Parallel execution
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=6,
        help="Number of parallel jobs for Monte Carlo repetitions (use -1 for all cores)",
    )

    # Bandwidth grid
    parser.add_argument(
        "--h_grid_start",
        type=float,
        default=-3,
        help="Start of logspace for bandwidth grid (10^h_grid_start)",
    )
    parser.add_argument(
        "--h_grid_end",
        type=float,
        default=0,
        help="End of logspace for bandwidth grid (10^h_grid_end)",
    )
    parser.add_argument(
        "--h_grid_num", type=int, default=200, help="Number of points in bandwidth grid"
    )

    # Verbosity
    parser.add_argument(
        "--verbose", type=int, default=0, help="Verbosity level (0=INFO, 1=DEBUG)"
    )

    # Sample sizes
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[100, 200, 400, 778, 1600, 3200],
        help="List of sample sizes to test",
    )

    # Monte Carlo repetitions
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=500,
        help="Number of Monte Carlo repetitions per sample size",
    )

    # Correlation function
    parser.add_argument(
        "--correlation_function",
        type=str,
        default="parabolic",
        choices=[
            "parabolic",
            "trigonometric_sine",
            "trigonometric_cosine",
            "cubic",
            "linear",
            "exponential",
            "semicircle_upper",
            "inverse_trig",
            "chebyshev_t3",
        ],
        help="Correlation function to use",
    )

    # Kernel function
    parser.add_argument(
        "--kernel_function",
        type=str,
        default="gaussian",
        choices=[
            "gaussian",
            "epanechnikov",
            "triangular",
            "uniform",
            "quartic",
            "triweight",
        ],
        help="Kernel function to use",
    )

    # Sampling strategy
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="uniform",
        choices=["uniform", "boundary"],
        help="Sampling strategy for test points",
    )

    # Bandwidth selection method (currently only 'mse' is used internally)
    parser.add_argument(
        "--bandwidth_selection_method",
        type=str,
        default="mse",
        choices=["mse", "cv"],
        help="Bandwidth selection method (currently only 'mse' is implemented)",
    )

    # Number of test points
    parser.add_argument(
        "--number_test",
        type=int,
        default=200,
        help="Number of test points for evaluation",
    )

    # Output file
    parser.add_argument(
        "--output_file",
        type=str,
        default="results.pkl",
        help="Path to save results (pickle format)",
    )

    # Random seed
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Setup output directory and logger
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = output_path.with_suffix(".log")
    logger = setup_logger(log_file=log_file, verbose=args.verbose)

    try:
        # Run comparison
        all_results = run_comparison(
            n_list=args.n,
            n_repeats=args.n_repeats,
            correlation_function=args.correlation_function,
            kernel_function=args.kernel_function,
            sampling_strategy=args.sampling_strategy,
            number_test=args.number_test,
            h_grid_start=args.h_grid_start,
            h_grid_end=args.h_grid_end,
            h_grid_num=args.h_grid_num,
            n_jobs=args.n_jobs,
            seed=args.seed,
            verbose=args.verbose,
            logger=logger,
        )

        # Save results
        logger.info("Saving results...")
        with open(output_path, "wb") as f:
            pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"✓ Results saved to: {output_path}")
        logger.info(f"✓ Log saved to: {log_file}")
        logger.info("")
        logger.info("=" * 78)
        logger.info("SUCCESS")
        logger.info("=" * 78)

        return 0

    except Exception as e:
        logger.error("")
        logger.error("=" * 78)
        logger.error("FATAL ERROR")
        logger.error("=" * 78)
        logger.error(f"Error: {str(e)}")
        logger.error("")
        import traceback

        logger.error(traceback.format_exc())
        logger.error("=" * 78)
        return 1


if __name__ == "__main__":
    sys.exit(main())
