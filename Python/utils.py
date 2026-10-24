"""
Author      : Jie Li
Date        : 2026-09-11 11:43:33
Last Edited : 2026-10-24 14:51:47
Last Author : Jie Li
File Path   : /nonparametric_correlation_estimator_cubic/Python/utils.py
Description :


Nonparametric Correlation Estimation - Utility Library
=======================================================

A modular and extensible toolkit for nonparametric correlation estimation.

Components:
-----------
1. Kernel Functions (extensible)
    - Gaussian, Epanechnikov, Triangular, Uniform, etc.

2. Correlation Functions (extensible)
    - Parabolic, Trigonometric, Cubic, Linear, Custom, etc.

3. Sampling Strategies
    - Uniform sampling
    - Boundary-dense sampling
    - Custom sampling patterns

4. Estimators
    - Nadaraya-Watson (NW)
    - Local Linear (LL)
    - Cubic Equation (CE)
    - Cubic with LL correction (CL)

5. Bandwidth Selection Methods
    - Grid Search
    - Rule of Thumb (Silverman, Scott)
    - Cross-Validation (Leave-One-Out, K-Fold)
    - Plugin methods

6. Evaluation Framework
    - MSE calculation
    - Bandwidth optimization
    - Multi-method comparison

Copyright (c) 2026, Jie Li, lijie_12@outlook.com
All Rights Reserved.
"""

import os
import sys
import warnings
from abc import ABC, abstractmethod

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import lstsq
from scipy.optimize import brentq, minimize, minimize_scalar

warnings.filterwarnings("ignore")


def _get_parallel_prefer():
    """
    Get joblib backend preference with cross-platform defaults.

    Environment variable override:
        NCE_PARALLEL_BACKEND in {"auto", "threads", "processes"}

    Defaults:
        - macOS: threads (avoid resource_tracker cleanup noise on some setups)
        - Linux/others: processes (often faster for CPU-heavy pure Python tasks)
    """
    backend = os.getenv("NCE_PARALLEL_BACKEND", "auto").strip().lower()
    if backend in {"threads", "processes"}:
        return backend
    if sys.platform == "darwin":
        return "threads"
    return "processes"


# Set random seed for reproducibility
# np.random.seed(42)


# =============================================================================
# 1. KERNEL FUNCTIONS (Extensible)
# =============================================================================


class KernelFunction(ABC):
    """Abstract base class for kernel functions."""

    @abstractmethod
    def evaluate(self, t):
        """Evaluate kernel at point t."""
        pass

    @property
    @abstractmethod
    def name(self):
        """Kernel name."""
        pass

    def weights(self, u_eval, u_train, h):
        """
        Compute kernel weights for estimation.

        Parameters:
        -----------
        u_eval : float
            Evaluation point
        u_train : array
            Training points
        h : float
            Bandwidth

        Returns:
        --------
        weights : array
            Kernel weights (unnormalized)
        """
        t = (u_train - u_eval) / h
        return self.evaluate(t)

    def normalized_weights(self, u_eval, u_train, h):
        """Compute normalized kernel weights (sum to 1)."""
        w = self.weights(u_eval, u_train, h)
        sum_w = np.sum(w)
        if sum_w < 1e-10:
            return np.ones_like(w) / len(w)
        return w / sum_w


class GaussianKernel(KernelFunction):
    """Gaussian (Normal) kernel: K(t) = (2*pi)^(-1/2) exp(-t^2/2)"""

    @property
    def name(self):
        return "Gaussian"

    def evaluate(self, t):
        return np.exp(-0.5 * t**2) / np.sqrt(2 * np.pi)


class EpanechnikovKernel(KernelFunction):
    """Epanechnikov kernel: K(t) = 3/4(1-t^2) for |t|<=1, 0 otherwise"""

    @property
    def name(self):
        return "Epanechnikov"

    def evaluate(self, t):
        return np.where(np.abs(t) <= 1, 0.75 * (1 - t**2), 0)


class TriangularKernel(KernelFunction):
    """Triangular kernel: K(t) = (1-|t|) for |t|<=1, 0 otherwise"""

    @property
    def name(self):
        return "Triangular"

    def evaluate(self, t):
        return np.where(np.abs(t) <= 1, 1 - np.abs(t), 0)


class UniformKernel(KernelFunction):
    """Uniform kernel: K(t) = 1/2 for |t|<=1, 0 otherwise"""

    @property
    def name(self):
        return "Uniform"

    def evaluate(self, t):
        return np.where(np.abs(t) <= 1, 0.5, 0)


class QuarticKernel(KernelFunction):
    """Quartic (Biweight) kernel: K(t) = 15/16(1-t^2)^2 for |t|<=1"""

    @property
    def name(self):
        return "Quartic"

    def evaluate(self, t):
        return np.where(np.abs(t) <= 1, (15 / 16) * (1 - t**2) ** 2, 0)


class TriweightKernel(KernelFunction):
    """Triweight kernel: K(t) = 35/32(1-t^2)^3 for |t|<=1"""

    @property
    def name(self):
        return "Triweight"

    def evaluate(self, t):
        return np.where(np.abs(t) <= 1, (35 / 32) * (1 - t**2) ** 3, 0)


# Kernel registry for easy access
KERNEL_REGISTRY = {
    "gaussian": GaussianKernel,
    "epanechnikov": EpanechnikovKernel,
    "triangular": TriangularKernel,
    "uniform": UniformKernel,
    "quartic": QuarticKernel,
    "triweight": TriweightKernel,
}


def get_kernel(kernel_name):
    """
    Factory function to get kernel by name.

    Parameters:
    -----------
    kernel_name : str
        Name of kernel ('gaussian', 'epanechnikov', etc.)

    Returns:
    --------
    kernel : KernelFunction
        Kernel function instance
    """
    if kernel_name.lower() not in KERNEL_REGISTRY:
        raise ValueError(
            f"Unknown kernel: {kernel_name}. "
            f"Available: {list(KERNEL_REGISTRY.keys())}"
        )
    return KERNEL_REGISTRY[kernel_name.lower()]()


# =============================================================================
# 2. CORRELATION FUNCTIONS (Extensible)
# =============================================================================


class CorrelationFunction(ABC):
    """Abstract base class for correlation functions rho(u)."""

    @abstractmethod
    def rho(self, u):
        """Evaluate rho(u)."""
        pass

    @property
    @abstractmethod
    def name(self):
        """Function name."""
        pass

    @property
    @abstractmethod
    def formula(self):
        """Mathematical formula as string."""
        pass

    def generate_data(self, n, random_state=None):
        r"""
        Generate n observations (U, X1, X2) following the correlation model.

        Model:
            $U \sim \text{Uniform}(-1, 1)$

            $(X_1, X_2) | U=u \sim \mathcal{N}(0, \Sigma(\rho(u)))$

            where $\Sigma = \begin{bmatrix} 1 & \rho(u) \\ \rho(u) & 1 \end{bmatrix}$

        Parameters
        ----------
        n : int
            Number of observations
        random_state : int, optional
            Random seed

        Returns
        -------
        U : array, shape (n,)
            Covariate values
        X1 : array, shape (n,)
            First response variable
        X2 : array, shape (n,)
            Second response variable
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Generate U ~ Uniform(-1, 1)
        U = np.random.uniform(-1, 1, n)

        # Generate (X1, X2) conditionally
        X1 = np.zeros(n)
        X2 = np.zeros(n)

        for i in range(n):
            rho_u = self.rho(U[i])
            # Covariance matrix
            mean = [0, 0]
            cov = [[1, rho_u], [rho_u, 1]]
            X1[i], X2[i] = np.random.multivariate_normal(mean, cov)

        return U, X1, X2


class ParabolicCorrelation(CorrelationFunction):
    """rho(u) = 1 - 2u^2 (Parabolic, inverted)"""

    @property
    def name(self):
        return "Parabolic"

    @property
    def formula(self):
        return "rho(u) = 1 - 2u^2"

    def rho(self, u):
        return 1 - 2 * u**2


class TrigonometricSineCorrelation(CorrelationFunction):
    """rho(u) = sin(pi*u/2) (Smooth, S-shaped)"""

    @property
    def name(self):
        return "Trigonometric Sine"

    @property
    def formula(self):
        return "rho(u) = sin(pi*u/2)"

    def rho(self, u):
        return np.sin(np.pi * u / 2)


class TrigonometricCosineCorrelation(CorrelationFunction):
    """rho(u) = cos(pi*u/2) (Smooth, S-shaped)"""

    @property
    def name(self):
        return "Trigonometric Cosine"

    @property
    def formula(self):
        return "rho(u) = cos(pi*u/2)"

    def rho(self, u):
        return np.cos(np.pi * u / 2)


class CubicCorrelation(CorrelationFunction):
    """rho(u) = u^3 (Monotone, odd)"""

    @property
    def name(self):
        return "Cubic"

    @property
    def formula(self):
        return "rho(u) = u^3"

    def rho(self, u):
        return u**3


class LinearCorrelation(CorrelationFunction):
    """rho(u) = u (Linear)"""

    @property
    def name(self):
        return "Linear"

    @property
    def formula(self):
        return "rho(u) = u"

    def rho(self, u):
        return u


class ExponentialCorrelation(CorrelationFunction):
    """rho(u) = (exp(u) - exp(-1)) / (exp(1) - exp(-1)) (Exponential, normalized)"""

    @property
    def name(self):
        return "Exponential"

    @property
    def formula(self):
        return "rho(u) = (e^u - e^(-1)) / (e - e^(-1))"

    def rho(self, u):
        return (np.exp(u) - np.exp(-1)) / (np.exp(1) - np.exp(-1))


class SemiCircleUpperCorrelation(CorrelationFunction):
    """rho(u) = sqrt(1 - u^2) (Semi-circle upper half, convex)"""

    @property
    def name(self):
        return "Semi-Circle Upper"

    @property
    def formula(self):
        return "rho(u) = sqrt(1 - u^2)"

    def rho(self, u):
        return np.sqrt(1 - u**2)


class InverseTrigonometricCorrelation(CorrelationFunction):
    """rho(u) = (2/pi) * arcsin(u) (Inverse trigonometric, normalized)"""

    @property
    def name(self):
        return "Inverse Trigonometric"

    @property
    def formula(self):
        return "rho(u) = (2/pi)arcsin(u)"

    def rho(self, u):
        return (2 / np.pi) * np.arcsin(u)


class ChebyshevT3Correlation(CorrelationFunction):
    """rho(u) = 4u^3 - 3u (Chebyshev polynomial T3)"""

    @property
    def name(self):
        return "Chebyshev T3"

    @property
    def formula(self):
        return "rho(u) = 4u^3 - 3u"

    def rho(self, u):
        return 4 * u**3 - 3 * u


class CustomCorrelation(CorrelationFunction):
    """Custom correlation function from user-provided callable."""

    def __init__(self, rho_func, name="Custom", formula="User-defined"):
        """
        Parameters:
        -----------
        rho_func : callable
            Function rho(u) that takes scalar/array and returns correlation
        name : str
            Name of the function
        formula : str
            Mathematical formula
        """
        self._rho_func = rho_func
        self._name = name
        self._formula = formula

    @property
    def name(self):
        return self._name

    @property
    def formula(self):
        return self._formula

    def rho(self, u):
        return self._rho_func(u)


# Correlation function registry
CORRELATION_REGISTRY = {
    "parabolic": ParabolicCorrelation,
    "trigonometric_sine": TrigonometricSineCorrelation,
    "trigonometric_cosine": TrigonometricCosineCorrelation,
    "cubic": CubicCorrelation,
    "linear": LinearCorrelation,
    "exponential": ExponentialCorrelation,
    "semicircle_upper": SemiCircleUpperCorrelation,
    "inverse_trig": InverseTrigonometricCorrelation,
    "chebyshev_t3": ChebyshevT3Correlation,
}


def get_correlation_function(name):
    """Factory function to get correlation function by name."""
    if name.lower() not in CORRELATION_REGISTRY:
        raise ValueError(
            f"Unknown correlation function: {name}. "
            f"Available: {list(CORRELATION_REGISTRY.keys())}"
        )
    return CORRELATION_REGISTRY[name.lower()]()


# =============================================================================
# 3. SAMPLING STRATEGIES
# =============================================================================


def uniform_sampling(n_points=50, u_min=-0.95, u_max=0.95):
    """
    Uniform sampling strategy.

    Parameters:
    -----------
    n_points : int
        Number of evaluation points
    u_min, u_max : float
        Range of evaluation

    Returns:
    --------
    u_test : array
        Evaluation points
    """
    return np.linspace(u_min, u_max, n_points)


def boundary_dense_sampling(
    n_boundary_left=20, n_interior=15, n_boundary_right=20, boundary_threshold=0.9
):
    """
    Boundary-dense sampling strategy.

    Concentrates more points near boundaries (|u| > threshold).

    Parameters:
    -----------
    n_boundary_left : int
        Number of points in left boundary region (u < -threshold)
    n_interior : int
        Number of points in interior region (-threshold <= u <= threshold)
    n_boundary_right : int
        Number of points in right boundary region (u > threshold)
    boundary_threshold : float
        Threshold for boundary region

    Returns:
    --------
    u_test : array
        Evaluation points (sorted)
    """
    u_left = np.linspace(-1, -boundary_threshold, n_boundary_left)
    u_middle = np.linspace(-boundary_threshold, boundary_threshold, n_interior)
    u_right = np.linspace(boundary_threshold, 1, n_boundary_right)

    u_test = np.sort(np.concatenate([u_left, u_middle, u_right]))
    return u_test


def adaptive_sampling(U_train, n_points=50, method="density"):
    """
    Adaptive sampling based on training data density.

    Parameters:
    -----------
    U_train : array
        Training covariate values
    n_points : int
        Number of evaluation points
    method : str
        'density': more points where data is dense
        'sparse': more points where data is sparse

    Returns:
    --------
    u_test : array
        Evaluation points
    """
    if method == "density":
        # Sample from empirical distribution (more points where data is dense)
        u_test = np.random.choice(U_train, size=n_points, replace=True)
        u_test = np.sort(u_test)
    elif method == "sparse":
        # Inverse density (more points in sparse regions)
        # Use quantiles
        percentiles = np.linspace(0, 100, n_points)
        u_test = np.percentile(U_train, percentiles)
    else:
        raise ValueError(f"Unknown adaptive method: {method}")

    return u_test


def custom_sampling(u_points):
    """
    Custom sampling from user-provided points.

    Parameters:
    -----------
    u_points : array-like
        Custom evaluation points

    Returns:
    --------
    u_test : array
        Sorted evaluation points
    """
    return np.sort(np.asarray(u_points))


# =============================================================================
# 4. ESTIMATORS
# =============================================================================


class CorrelationEstimator(ABC):
    """Abstract base class for correlation estimators."""

    def __init__(self, kernel="gaussian"):
        """
        Parameters:
        -----------
        kernel : str or KernelFunction
            Kernel function to use
        """
        if isinstance(kernel, str):
            self.kernel = get_kernel(kernel)
        elif isinstance(kernel, KernelFunction):
            self.kernel = kernel
        else:
            raise TypeError("kernel must be str or KernelFunction instance")

    @abstractmethod
    def estimate(self, u_eval, U_train, X1_train, X2_train, h):
        """
        Estimate rho(u_eval) using training data and bandwidth h.

        Parameters:
        -----------
        u_eval : float
            Evaluation point
        U_train : array
            Training covariate values
        X1_train : array
            Training X1 values
        X2_train : array
            Training X2 values
        h : float
            Bandwidth

        Returns:
        --------
        rho_hat : float
            Estimated correlation at u_eval
        """
        pass

    @property
    @abstractmethod
    def name(self):
        """Estimator name."""
        pass

    @property
    @abstractmethod
    def abbreviation(self):
        """Short abbreviation (for plotting)."""
        pass

    def estimate_batch(
        self, u_eval_array, U_train, X1_train, X2_train, h, n_jobs=1, verbose=0
    ):
        """
        Estimate rho at multiple evaluation points (parallelizable).

        Parameters
        ----------
        u_eval_array : array-like
            Array of evaluation points
        U_train : array
            Training covariate values
        X1_train : array
            Training X1 values
        X2_train : array
            Training X2 values
        h : float
            Bandwidth
        n_jobs : int, default=1
            Number of parallel jobs
            - 1: Sequential execution (default, backward compatible)
            - -1: Use all available CPU cores
            - n > 1: Use n CPU cores
        verbose : int, default=0
            Verbosity level (0=silent, 1=progress bar)

        Returns
        -------
        rho_hat_array : ndarray
            Estimated correlations at each evaluation point

        Notes
        -----
        This method is particularly beneficial for:
        - Large number of evaluation points (> 20)
        - Computationally expensive estimators (LL, CUBIC+LL)
        - Bandwidth grid search (called internally)

        Examples
        --------
        >>> estimator = LocalLinearEstimator(kernel='gaussian')
        >>> u_test = np.linspace(-1, 1, 100)
        >>> rho_hat = estimator.estimate_batch(
        ...     u_test, U_train, X1_train, X2_train, h=0.1, n_jobs=8
        ... )
        """
        u_eval_array = np.asarray(u_eval_array)

        if n_jobs == 1:
            # Sequential execution (default, backward compatible)
            rho_hat_array = np.array(
                [self.estimate(u, U_train, X1_train, X2_train, h) for u in u_eval_array]
            )
        else:
            # Parallel execution
            if verbose > 0:
                try:
                    from tqdm import tqdm

                    iterator = tqdm(
                        u_eval_array, desc=f"{self.abbreviation} estimation"
                    )
                except ImportError:
                    iterator = u_eval_array
                    if verbose > 0:
                        print(f"Note: Install tqdm for progress bar (pip install tqdm)")
            else:
                iterator = u_eval_array

            prefer_backend = _get_parallel_prefer()
            try:
                rho_hat_array = np.array(
                    Parallel(n_jobs=n_jobs, prefer=prefer_backend)(
                        delayed(self.estimate)(u, U_train, X1_train, X2_train, h)
                        for u in iterator
                    )
                )
            except Exception:
                # Graceful fallback: if process backend fails, retry with threads.
                if prefer_backend == "processes":
                    rho_hat_array = np.array(
                        Parallel(n_jobs=n_jobs, prefer="threads")(
                            delayed(self.estimate)(u, U_train, X1_train, X2_train, h)
                            for u in iterator
                        )
                    )
                else:
                    raise

        return rho_hat_array


class NadarayaWatsonEstimator(CorrelationEstimator):
    r"""
    Nadaraya-Watson (NW) estimator.

    Standard kernel-weighted average.

    \hat{\rho}(u) = \frac{2\sum_{i=1}^{n} X_{1i}X_{2i}K_h(U_i - u)}{\sum_{i=1}^{n} (X_{1i}^2 + X_{2i}^2)K_h(U_i - u)}
    """

    @property
    def name(self):
        return "Nadaraya-Watson"

    @property
    def abbreviation(self):
        return "NW"

    def estimate(self, u_eval, U_train, X1_train, X2_train, h):
        weights = self.kernel.normalized_weights(u_eval, U_train, h)
        rho_hat = (
            2
            * np.sum(weights * X1_train * X2_train)
            / np.sum(weights * (X1_train**2 + X2_train**2))
        )
        return rho_hat


class LocalLinearEstimator(CorrelationEstimator):
    r"""
    Local Linear (LL) estimator using kernel-weighted log-likelihood.

    Uses Fisher z-transformation to ensure $\rho \in [-1, 1]$.

    Method:

    1. Define $\gamma_i = \alpha_0 + \alpha_1(U_i - u_0)$
    2. Transform: $\rho_i = \tanh(\gamma_i)$
    3. Maximize kernel-weighted log-likelihood
    4. Return: $\hat{\rho}(u_0) = \tanh(\hat{\alpha}_0)$

    Objective function:

    $\ell^*(\alpha_0, \alpha_1) = \sum_{i=1}^{n} \left[ -\ln(\text{sech}(\gamma_i)) - \frac{\cosh^2(\gamma_i)(X_{1i}^2 + X_{2i}^2) - \sinh(2\gamma_i)X_{1i}X_{2i}}{2} \right] K_h(U_i - u_0)$

    where $\gamma_i = \alpha_0 + \alpha_1(U_i - u_0)$
    """

    @property
    def name(self):
        return "Local Linear"

    @property
    def abbreviation(self):
        return "LL"

    def estimate(self, u_eval, U_train, X1_train, X2_train, h):
        """
        Estimate rho(u_eval) using kernel-weighted maximum likelihood.

        Parameters:
        -----------
        u_eval : float
            Evaluation point
        U_train : array
            Training covariate values
        X1_train : array
            Training X1 values
        X2_train : array
            Training X2 values
        h : float
            Bandwidth

        Returns:
        --------
        rho_hat : float
            Estimated correlation at u_eval, guaranteed in [-1, 1]
        """
        # Compute kernel weights
        weights = self.kernel.weights(u_eval, U_train, h)

        # Filter out zero-weight observations for efficiency
        mask = weights > 1e-10
        if np.sum(mask) < 2:
            # Fallback to simple weighted average if too few points
            return self._fallback_estimate(u_eval, U_train, X1_train, X2_train, h)

        U_local = U_train[mask]
        X1_local = X1_train[mask]
        X2_local = X2_train[mask]
        weights_local = weights[mask]

        # Center U for numerical stability
        U_centered = U_local - u_eval

        # Define negative log-likelihood (we minimize the negative)
        def neg_log_likelihood(alpha):
            alpha_0, alpha_1 = alpha
            gamma = alpha_0 + alpha_1 * U_centered

            # sech(gamma) = 1/cosh(gamma)
            cosh_gamma = np.cosh(gamma)
            sech_gamma = 1.0 / cosh_gamma

            # Avoid log(0) by clipping
            log_sech = np.log(np.clip(sech_gamma, 1e-15, None))

            # sinh(2*gamma)
            sinh_2gamma = np.sinh(2 * gamma)

            # Quadratic form term
            quad_term = (
                cosh_gamma**2 * (X1_local**2 + X2_local**2)
                - sinh_2gamma * X1_local * X2_local
            ) / 2.0

            # Weighted log-likelihood
            ll = np.sum(weights_local * (-log_sech - quad_term))

            return -ll  # Return negative for minimization

        # Gradient of negative log-likelihood
        def neg_log_likelihood_grad(alpha):
            alpha_0, alpha_1 = alpha
            gamma = alpha_0 + alpha_1 * U_centered

            # Derivatives of gamma w.r.t. alpha
            dgamma_dalpha0 = 1.0
            dgamma_dalpha1 = U_centered

            # Compute hyperbolic functions
            cosh_gamma = np.cosh(gamma)
            sinh_gamma = np.sinh(gamma)
            tanh_gamma = np.tanh(gamma)

            # d(log(sech(gamma)))/dgamma = -tanh(gamma)
            dlog_sech_dgamma = -tanh_gamma

            # d(quad_term)/dgamma
            # quad_term = \frac{\cosh^2(\gamma)(X_1^2 + X_2^2) - \sinh(2\gamma)X_1X_2}{2}
            # \frac{d(\text{quad\_term})}{d\gamma} = \cosh(\gamma)\sinh(\gamma)(X_1^2 + X_2^2) - \cosh(2\gamma)X_1X_2
            cosh_2gamma = np.cosh(2 * gamma)
            dquad_dgamma = (
                cosh_gamma * sinh_gamma * (X1_local**2 + X2_local**2)
                - cosh_2gamma * X1_local * X2_local
            )

            # Total derivative of log-likelihood w.r.t. gamma
            dll_dgamma = -dlog_sech_dgamma - dquad_dgamma

            # Chain rule for alpha_0 and alpha_1
            grad_alpha0 = -np.sum(weights_local * dll_dgamma * dgamma_dalpha0)
            grad_alpha1 = -np.sum(weights_local * dll_dgamma * dgamma_dalpha1)

            return np.array([grad_alpha0, grad_alpha1])

        # Initial guess: use simple weighted least squares
        try:
            X_design = np.column_stack([np.ones_like(U_centered), U_centered])
            y = X1_local * X2_local
            # Weighted normal equations without explicitly forming diagonal matrix W.
            XtW = X_design.T * weights_local
            XtWX = XtW @ X_design
            XtWy = XtW @ y
            beta_init = np.linalg.solve(XtWX, XtWy)

            # Transform to alpha space using arctanh (Fisher z)
            rho_init = np.clip(beta_init[0], -0.99, 0.99)
            alpha_0_init = np.arctanh(rho_init)
            alpha_1_init = beta_init[1] / (1 - rho_init**2)  # Approximate scaling
        except (np.linalg.LinAlgError, RuntimeWarning):
            alpha_0_init = 0.0
            alpha_1_init = 0.0

        # Optimize using L-BFGS-B
        result = minimize(
            neg_log_likelihood,
            x0=[alpha_0_init, alpha_1_init],
            method="L-BFGS-B",
            jac=neg_log_likelihood_grad,
            options={"maxiter": 100, "ftol": 1e-6},
        )

        if result.success:
            alpha_0_hat = result.x[0]
        else:
            # If optimization fails, use initial guess
            alpha_0_hat = alpha_0_init

        # Transform back to correlation using tanh
        rho_hat = np.tanh(alpha_0_hat)

        return rho_hat

    def _fallback_estimate(self, u_eval, U_train, X1_train, X2_train, h):
        """Fallback to simple weighted average when too few points."""
        weights = self.kernel.normalized_weights(u_eval, U_train, h)
        rho_hat = np.sum(weights * X1_train * X2_train)
        return np.clip(rho_hat, -1, 1)


class CubicEquationEstimator(CorrelationEstimator):
    r"""
    Cubic Equation (CUBIC) estimator.

    Solves cubic equation to guarantee $\rho \in [-1, 1]$.

    Method:
    1. Estimate $A(u) = \mathbb{E}[X_1^2 + X_2^2 \mid U=u]$
    2. Estimate $B(u) = \mathbb{E}[X_1 X_2 \mid U=u]$
    3. Solve: $\rho^3 - B\rho^2 + (A-1)\rho - B = 0$
    4. Return root in $[-1, 1]$

    Reference: Paper Proposition 1
    """

    @property
    def name(self):
        return "Cubic Equation"

    @property
    def abbreviation(self):
        return "CUBIC"

    def estimate(self, u_eval, U_train, X1_train, X2_train, h):
        weights = self.kernel.normalized_weights(u_eval, U_train, h)

        # Estimate A and B
        A = np.sum(weights * (X1_train**2 + X2_train**2))
        B = np.sum(weights * X1_train * X2_train)

        # Solve cubic equation
        return self._solve_cubic(A, B)

    def _cubic_equation(self, rho, A, B):
        r"""Cubic equation: $g(\rho) = \rho^3 - B\rho^2 + (A-1)\rho - B$"""
        return rho**3 - B * rho**2 + (A - 1) * rho - B

    def _objective_from_AB(self, rho, A, B):
        """Profile objective implied by weighted Gaussian likelihood."""
        rho_safe = np.clip(rho, -1 + 1e-10, 1 - 1e-10)
        denom = 1.0 - rho_safe**2
        return (A - 2.0 * rho_safe * B) / denom + np.log(denom)

    def _solve_cubic(self, A, B):
        """Find root in [-1, 1], preferring the minimizer of local criterion."""
        g_minus1 = self._cubic_equation(-1, A, B)
        g_plus1 = self._cubic_equation(1, A, B)

        try:
            roots = self._find_all_real_roots(A, B)
            valid_roots = [
                np.clip(r, -1.0, 1.0) for r in roots if -1.0 - 1e-10 <= r <= 1.0 + 1e-10
            ]

            if len(valid_roots) > 0:
                scores = [self._objective_from_AB(r, A, B) for r in valid_roots]
                root = valid_roots[int(np.argmin(scores))]
            elif g_minus1 * g_plus1 <= 0:
                root = brentq(self._cubic_equation, -1, 1, args=(A, B))
            else:
                # Numerical fallback: clip B to feasible range to recover a bracketed root.
                B_clipped = np.clip(B, -0.5 * A, 0.5 * A)
                g_minus1_clip = self._cubic_equation(-1, A, B_clipped)
                g_plus1_clip = self._cubic_equation(1, A, B_clipped)
                if g_minus1_clip * g_plus1_clip <= 0:
                    root = brentq(self._cubic_equation, -1, 1, args=(A, B_clipped))
                else:
                    root = np.clip(B, -1.0, 1.0)
        except (ValueError, IndexError):
            root = np.clip(B, -1.0, 1.0)

        return float(root)

    def _find_all_real_roots(self, A, B):
        """Find all real roots using np.roots."""
        coeffs = [1, -B, A - 1, -B]
        roots = np.roots(coeffs)
        real_roots = roots[np.abs(roots.imag) < 1e-8].real
        return sorted(real_roots)


class CubicWithLLCorrection(CorrelationEstimator):
    """
    Cubic with Local Linear Correction (CL) estimator.

    Combines:
    - CUBIC's constraint guarantee (rho in [-1, 1])
    - LL's improved boundary behavior for B(u)

    Method:
    1. Estimate A(u) using kernel weights (same as CUBIC)
    2. Estimate B(u) using Local Linear regression
    3. Solve cubic equation with (A_kernel, B_LL)

    This is expected to perform better at large sample sizes due to
    LL's O(h^2) boundary bias vs. local-constant weighting's O(h) boundary bias.
    """

    @property
    def name(self):
        return "Cubic with LL Correction"

    @property
    def abbreviation(self):
        return "CL"

    def estimate(self, u_eval, U_train, X1_train, X2_train, h):
        weights = self.kernel.normalized_weights(u_eval, U_train, h)

        # A: Standard kernel weighting
        A = np.sum(weights * (X1_train**2 + X2_train**2))

        # B: Local Linear correction
        B_ll = self._estimate_B_local_linear(u_eval, U_train, X1_train, X2_train, h)

        # Solve cubic
        return self._solve_cubic(A, B_ll)

    def _estimate_B_local_linear(self, u_eval, U_train, X1_train, X2_train, h):
        """Estimate B(u) = E[X1 X2 | U=u] using local linear regression."""
        weights = self.kernel.normalized_weights(u_eval, U_train, h)
        y = X1_train * X2_train

        X = np.column_stack([np.ones_like(U_train), U_train - u_eval])
        # Weighted normal equations without explicitly forming diagonal matrix W.
        XtW = X.T * weights
        XtWX = XtW @ X
        XtWy = XtW @ y

        try:
            beta = np.linalg.solve(XtWX, XtWy)
            B_ll = beta[0]
        except np.linalg.LinAlgError:
            B_ll = np.sum(weights * y)

        return B_ll

    def _cubic_equation(self, rho, A, B):
        """Cubic equation."""
        return rho**3 - B * rho**2 + (A - 1) * rho - B

    def _objective_from_AB(self, rho, A, B):
        """Profile objective implied by weighted Gaussian likelihood."""
        rho_safe = np.clip(rho, -1 + 1e-10, 1 - 1e-10)
        denom = 1.0 - rho_safe**2
        return (A - 2.0 * rho_safe * B) / denom + np.log(denom)

    def _solve_cubic(self, A, B):
        """Solve cubic equation by selecting the best valid real root."""
        g_minus1 = self._cubic_equation(-1, A, B)
        g_plus1 = self._cubic_equation(1, A, B)

        try:
            roots = self._find_all_real_roots(A, B)
            valid_roots = [
                np.clip(r, -1.0, 1.0) for r in roots if -1.0 - 1e-10 <= r <= 1.0 + 1e-10
            ]

            if len(valid_roots) > 0:
                scores = [self._objective_from_AB(r, A, B) for r in valid_roots]
                root = valid_roots[int(np.argmin(scores))]
            elif g_minus1 * g_plus1 <= 0:
                root = brentq(self._cubic_equation, -1, 1, args=(A, B))
            else:
                # Numerical fallback: clip B to feasible range to recover a bracketed root.
                B_clipped = np.clip(B, -0.5 * A, 0.5 * A)
                g_minus1_clip = self._cubic_equation(-1, A, B_clipped)
                g_plus1_clip = self._cubic_equation(1, A, B_clipped)
                if g_minus1_clip * g_plus1_clip <= 0:
                    root = brentq(self._cubic_equation, -1, 1, args=(A, B_clipped))
                else:
                    root = np.clip(B, -1.0, 1.0)
        except (ValueError, IndexError):
            root = np.clip(B, -1.0, 1.0)

        return float(root)

    def _find_all_real_roots(self, A, B):
        """Find all real roots."""
        coeffs = [1, -B, A - 1, -B]
        roots = np.roots(coeffs)
        real_roots = roots[np.abs(roots.imag) < 1e-8].real
        return sorted(real_roots)


# Estimator registry
ESTIMATOR_REGISTRY = {
    "nw": NadarayaWatsonEstimator,
    "ll": LocalLinearEstimator,
    "cubic": CubicEquationEstimator,
    "cubic_ll": CubicWithLLCorrection,
}


def get_estimator(name, kernel="gaussian"):
    """Factory function to get estimator by name."""
    if name.lower() not in ESTIMATOR_REGISTRY:
        raise ValueError(
            f"Unknown estimator: {name}. "
            f"Available: {list(ESTIMATOR_REGISTRY.keys())}"
        )
    return ESTIMATOR_REGISTRY[name.lower()](kernel=kernel)


# =============================================================================
# 5. BANDWIDTH SELECTION METHODS
# =============================================================================


class BandwidthSelector:
    """Bandwidth selection methods for kernel smoothing."""

    def __init__(self, U_train, X1_train, X2_train, estimator, rho_true_func=None):
        """
        Parameters:
        -----------
        U_train : array
            Training covariate values
        X1_train : array
            Training X1 values
        X2_train : array
            Training X2 values
        estimator : CorrelationEstimator
            Estimator to use
        rho_true_func : callable, optional
            True correlation function (for validation)
        """
        self.U_train = U_train
        self.X1_train = X1_train
        self.X2_train = X2_train
        self.estimator = estimator
        self.rho_true_func = rho_true_func
        self.n = len(U_train)

    def grid_search(self, h_grid, u_test=None, method="mse", n_jobs=1, verbose=0):
        """
        Grid search over bandwidth values with optional parallelization.

        Parameters
        ----------
        h_grid : array-like
            Grid of bandwidth values to try
        u_test : array, optional
            Test points for evaluation. If None, uses U_train
        method : str, default='mse'
            'mse': minimize MSE (requires rho_true_func)
            'cv': minimize cross-validation score
        n_jobs : int, default=1
            Number of parallel jobs for bandwidth search
            - 1: Sequential execution (default, backward compatible)
            - -1: Use all available CPU cores
            - n > 1: Use n CPU cores
            Note: Parallelizes over different bandwidth values (outer loop)
        verbose : int, default=0
            Verbosity level (0=silent, 1=progress info)

        Returns
        -------
        result : dict
            {
                'h_grid': bandwidth grid,
                'scores': scores for each h,
                'optimal_h': best bandwidth,
                'min_score': minimum score
            }

        Notes
        -----
        Parallelization Strategy:
        - Parallelizes over bandwidth candidates (h_grid)
        - Each task evaluates one bandwidth on all test points
        - Most effective when len(h_grid) > 20
        - Expected speedup: 6-8x on 8-core CPU

        Examples
        --------
        >>> selector = BandwidthSelector(U, X1, X2, estimator, rho_func.rho)
        >>> result = selector.grid_search(
        ...     h_grid=np.logspace(-1.5, 0, 100),
        ...     u_test=u_test,
        ...     n_jobs=8,
        ...     verbose=1
        ... )
        """
        if u_test is None:
            u_test = self.U_train

        if n_jobs == 1:
            # Sequential execution (default, backward compatible)
            scores = []
            for h in h_grid:
                if method == "mse":
                    if self.rho_true_func is None:
                        raise ValueError("MSE method requires rho_true_func")
                    score = self._compute_mse(h, u_test)
                elif method == "cv":
                    score = self._cross_validation_score(h)
                else:
                    raise ValueError(f"Unknown method: {method}")
                scores.append(score)
        else:
            # Parallel execution over bandwidth grid
            if verbose > 0:
                print(
                    f"Grid search with {len(h_grid)} bandwidths using {n_jobs} jobs..."
                )

            def evaluate_bandwidth(h):
                if method == "mse":
                    if self.rho_true_func is None:
                        raise ValueError("MSE method requires rho_true_func")
                    return self._compute_mse(h, u_test)
                elif method == "cv":
                    return self._cross_validation_score(h)
                else:
                    raise ValueError(f"Unknown method: {method}")

            prefer_backend = _get_parallel_prefer()
            try:
                scores = Parallel(n_jobs=n_jobs, prefer=prefer_backend)(
                    delayed(evaluate_bandwidth)(h) for h in h_grid
                )
            except Exception:
                # Graceful fallback: if process backend fails, retry with threads.
                if prefer_backend == "processes":
                    scores = Parallel(n_jobs=n_jobs, prefer="threads")(
                        delayed(evaluate_bandwidth)(h) for h in h_grid
                    )
                else:
                    raise

        scores = np.array(scores)
        opt_idx = np.argmin(scores)

        if verbose > 0:
            print(
                f"Optimal bandwidth: h = {h_grid[opt_idx]:.6f}, score = {scores[opt_idx]:.6f}"
            )

        return {
            "h_grid": np.array(h_grid),
            "scores": scores,
            "optimal_h": h_grid[opt_idx],
            "min_score": scores[opt_idx],
        }

    def _compute_mse(self, h, u_test):
        """
        Compute MSE for given bandwidth.

        Note: Uses sequential execution within each bandwidth evaluation.
        Parallelization happens at the bandwidth grid level (in grid_search).
        """
        # Use sequential batch estimation (n_jobs=1) to avoid nested parallelization
        rho_hat = self.estimator.estimate_batch(
            u_test, self.U_train, self.X1_train, self.X2_train, h, n_jobs=1
        )
        rho_true = np.array([self.rho_true_func(u) for u in u_test])
        return np.mean((rho_hat - rho_true) ** 2)

    def _cross_validation_score(self, h, k=None):
        """
        Leave-one-out or K-fold cross-validation score.

        The CV objective is the held-out negative Gaussian log-likelihood
        contribution under the estimated correlation:

            ell(x; rho) = (x1^2 + x2^2 - 2 rho x1 x2) / (1 - rho^2)
                            + log(1 - rho^2)

        This is consistent with the paper's local likelihood criterion.

        Parameters:
        -----------
        h : float
            Bandwidth
        k : int, optional
            Number of folds. If None, use leave-one-out (k=n)

        Returns:
        --------
        cv_score : float
            Cross-validation score (lower is better)
        """
        if k is None or k >= self.n:
            # Leave-one-out
            return self._leave_one_out_cv(h)
        else:
            # K-fold
            return self._k_fold_cv(h, k)

    @staticmethod
    def _gaussian_negloglik_contribution(x1, x2, rho):
        """Compute held-out negative Gaussian log-likelihood contribution."""
        rho_safe = np.clip(rho, -1 + 1e-10, 1 - 1e-10)
        denom = 1.0 - rho_safe**2
        return (x1**2 + x2**2 - 2.0 * rho_safe * x1 * x2) / denom + np.log(denom)

    def _leave_one_out_cv(self, h):
        """Leave-one-out cross-validation with held-out log-likelihood."""
        cv_errors = []

        for i in range(self.n):
            # Leave out i-th observation
            U_train_cv = np.delete(self.U_train, i)
            X1_train_cv = np.delete(self.X1_train, i)
            X2_train_cv = np.delete(self.X2_train, i)

            # Predict at left-out point
            u_test = self.U_train[i]
            rho_hat = self.estimator.estimate(
                u_test, U_train_cv, X1_train_cv, X2_train_cv, h
            )

            # Held-out likelihood score at (U_i, X1_i, X2_i)
            ll_i = self._gaussian_negloglik_contribution(
                self.X1_train[i], self.X2_train[i], rho_hat
            )
            cv_errors.append(ll_i)

        return np.mean(cv_errors)

    def _k_fold_cv(self, h, k):
        """K-fold cross-validation with held-out log-likelihood."""
        # Use a fixed RNG for deterministic fold splits and reproducible CV scores.
        rng = np.random.default_rng(42)
        indices = rng.permutation(self.n)
        fold_size = self.n // k
        cv_errors = []

        for fold in range(k):
            # Define test fold
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < k - 1 else self.n
            test_idx = indices[test_start:test_end]
            train_idx = np.setdiff1d(indices, test_idx)

            # Split data
            U_train_cv = self.U_train[train_idx]
            X1_train_cv = self.X1_train[train_idx]
            X2_train_cv = self.X2_train[train_idx]

            # Predict on test fold (batch for efficiency)
            U_test_fold = self.U_train[test_idx]
            X1_test_fold = self.X1_train[test_idx]
            X2_test_fold = self.X2_train[test_idx]

            rho_hat_fold = self.estimator.estimate_batch(
                U_test_fold, U_train_cv, X1_train_cv, X2_train_cv, h, n_jobs=1
            )

            ll_fold = self._gaussian_negloglik_contribution(
                X1_test_fold, X2_test_fold, rho_hat_fold
            )
            cv_errors.extend(ll_fold.tolist())

        return np.mean(cv_errors)

    def rule_of_thumb_silverman(self):
        """
        Silverman's rule of thumb bandwidth.

        h = 1.06 * sigma * n^(-1/5)

        where sigma is the standard deviation of U.

        Returns:
        --------
        h : float
            Bandwidth
        """
        sigma = np.std(self.U_train)
        h = 1.06 * sigma * self.n ** (-1 / 5)
        return h

    def rule_of_thumb_scott(self):
        """
        Scott's rule of thumb bandwidth.

        h = sigma * n^(-1/5)

        Returns:
        --------
        h : float
            Bandwidth
        """
        sigma = np.std(self.U_train)
        h = sigma * self.n ** (-1 / 5)
        return h

    def plugin_bandwidth(self):
        """
        Plugin bandwidth selector (placeholder).

        This is a more sophisticated method that estimates the optimal
        bandwidth based on second derivative estimation.


        Returns:
        --------
        h : float
            Bandwidth
        """
        # For now, use Scott's rule as default
        return self.rule_of_thumb_scott()


# =============================================================================
# 6. EVALUATION FRAMEWORK
# =============================================================================


def evaluate_estimator(
    estimator, U_train, X1_train, X2_train, u_test, rho_true_func, h_values
):
    """
    Evaluate a single estimator across multiple bandwidths.

    Parameters:
    -----------
    estimator : CorrelationEstimator
        Estimator to evaluate
    U_train, X1_train, X2_train : arrays
        Training data
    u_test : array
        Test points
    rho_true_func : callable
        True correlation function
    h_values : array-like
        Bandwidth values to try

    Returns:
    --------
    results : dict
        {
            'name': estimator name,
            'h_grid': bandwidth grid,
            'mse_grid': MSE for each bandwidth,
            'optimal_h': best bandwidth,
            'min_mse': minimum MSE,
            'optimal_estimates': estimates at optimal h,
            'rho_true': true values at u_test
        }
    """
    mse_grid = []
    estimates_grid = {}
    rho_true = np.array([rho_true_func(u) for u in u_test])

    for h in h_values:
        rho_hat = np.array(
            [estimator.estimate(u, U_train, X1_train, X2_train, h) for u in u_test]
        )
        mse = np.mean((rho_hat - rho_true) ** 2)
        mse_grid.append(mse)
        estimates_grid[h] = rho_hat

    mse_grid = np.array(mse_grid)
    opt_idx = np.argmin(mse_grid)

    return {
        "name": estimator.name,
        "abbreviation": estimator.abbreviation,
        "h_grid": np.array(h_values),
        "mse_grid": mse_grid,
        "optimal_h": h_values[opt_idx],
        "min_mse": mse_grid[opt_idx],
        "optimal_estimates": estimates_grid[h_values[opt_idx]],
        "rho_true": rho_true,
    }


def compare_estimators(
    estimators, U_train, X1_train, X2_train, u_test, rho_true_func, h_values
):
    """
    Compare multiple estimators.

    Parameters:
    -----------
    estimators : list of CorrelationEstimator
        Estimators to compare
    U_train, X1_train, X2_train : arrays
        Training data
    u_test : array
        Test points
    rho_true_func : callable
        True correlation function
    h_values : array-like
        Bandwidth values to try

    Returns:
    --------
    results : dict
        Dictionary keyed by estimator abbreviation, containing results
        from evaluate_estimator for each method
    """
    results = {}

    for estimator in estimators:
        result = evaluate_estimator(
            estimator, U_train, X1_train, X2_train, u_test, rho_true_func, h_values
        )
        results[estimator.abbreviation.lower()] = result

    return results


# =============================================================================
# 7. UTILITY FUNCTIONS
# =============================================================================


def compute_constraint_violations(rho_estimates):
    """
    Count how many estimates violate the [-1, 1] constraint.

    Parameters:
    -----------
    rho_estimates : array
        Correlation estimates

    Returns:
    --------
    n_violations : int
        Number of violations
    violation_indices : array
        Indices where violations occur
    """
    violations = (rho_estimates < -1) | (rho_estimates > 1)
    n_violations = np.sum(violations)
    violation_indices = np.where(violations)[0]

    return n_violations, violation_indices


def summary_statistics(results):
    """
    Compute summary statistics for evaluation results.

    Parameters:
    -----------
    results : dict
        Results from evaluate_estimator or compare_estimators

    Returns:
    --------
    summary : dict
        Summary statistics for each method
    """
    summary = {}

    for method_key, method_results in results.items():
        rho_hat = method_results["optimal_estimates"]
        rho_true = method_results["rho_true"]
        errors = rho_hat - rho_true

        n_violations, _ = compute_constraint_violations(rho_hat)

        summary[method_key] = {
            "name": method_results["name"],
            "optimal_h": method_results["optimal_h"],
            "min_mse": method_results["min_mse"],
            "mean_error": np.mean(errors),
            "median_error": np.median(errors),
            "max_abs_error": np.max(np.abs(errors)),
            "n_violations": n_violations,
            "violation_rate": n_violations / len(rho_hat),
        }

    return summary


def print_comparison_table(summary):
    """
    Print a formatted comparison table.

    Parameters:
    -----------
    summary : dict
        Summary statistics from summary_statistics()
    """
    print("\n" + "=" * 90)
    print("ESTIMATOR COMPARISON")
    print("=" * 90)
    print(
        f"{'Method':<20} {'Optimal h':<12} {'MSE':<12} {'Max Error':<12} {'Violations':<12}"
    )
    print("-" * 90)

    for method_key, stats in summary.items():
        print(
            f"{stats['name']:<20} {stats['optimal_h']:<12.4f} "
            f"{stats['min_mse']:<12.6f} {stats['max_abs_error']:<12.4f} "
            f"{stats['n_violations']:<12d}"
        )

    print("=" * 90)


# =============================================================================
# 8. CONVENIENCE FUNCTIONS
# =============================================================================


def create_default_setup(
    correlation_func="parabolic",
    n_samples=200,
    kernel="gaussian",
    sampling_strategy="uniform",
    n_test_points=50,
):
    """
    Create a default experimental setup.

    Parameters:
    -----------
    correlation_func : str or CorrelationFunction
        Correlation function to use
    n_samples : int
        Number of training samples
    kernel : str or KernelFunction
        Kernel function
    sampling_strategy : str
        'uniform' or 'boundary'
    n_test_points : int
        Number of test points

    Returns:
    --------
    setup : dict
        {
            'rho_func': correlation function,
            'U_train': training U,
            'X1_train': training X1,
            'X2_train': training X2,
            'u_test': test points,
            'kernel': kernel function,
            'estimators': list of estimators
        }
    """
    # Get correlation function
    if isinstance(correlation_func, str):
        rho_func = get_correlation_function(correlation_func)
    else:
        rho_func = correlation_func

    # Generate data
    U_train, X1_train, X2_train = rho_func.generate_data(n_samples, random_state=42)

    # Create test points
    if sampling_strategy == "uniform":
        u_test = uniform_sampling(n_test_points)
    elif sampling_strategy == "boundary":
        u_test = boundary_dense_sampling()
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    # Get kernel
    if isinstance(kernel, str):
        kernel_obj = get_kernel(kernel)
    else:
        kernel_obj = kernel

    # Create estimators
    estimators = [
        NadarayaWatsonEstimator(kernel=kernel_obj),
        LocalLinearEstimator(kernel=kernel_obj),
        CubicEquationEstimator(kernel=kernel_obj),
        CubicWithLLCorrection(kernel=kernel_obj),
    ]

    return {
        "rho_func": rho_func,
        "U_train": U_train,
        "X1_train": X1_train,
        "X2_train": X2_train,
        "u_test": u_test,
        "kernel": kernel_obj,
        "estimators": estimators,
    }
