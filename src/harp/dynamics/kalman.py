"""DMD-based Kalman filter with spherical observation noise regularisation.

This implements the filter described in Appendix A of the paper:

  "Global Persistence, Local Residual Structure:
   Forecasting Heterogeneous Investment Panels"

Key design choices:

  1. **Spherical R** (Eq. 5): Observation noise is set to the mean squared
     projection residual onto the K-dimensional subspace, times I_N.
     This avoids estimating an N x N covariance when T << N.

  2. **Spectral radius clipping**: The transition matrix F is uniformly
     scaled so max|lambda_k| <= 0.99, preventing explosive dynamics.

  3. **Adaptive Q** (Eq. 6): Process noise is updated via exponential
     smoothing of state-correction outer products, with quarterly reset.

  4. **Joseph form**: The covariance update uses the numerically stable
     (I - KU)P(I - KU)' + KRK' form.

Usage::

    from harp.dynamics.kalman import SphericalKalmanFilter

    kf = SphericalKalmanFilter(K=8, lambda_q=0.3, q_init=0.5, sr_clip=0.99)
    kf.initialise(U, F, train_residuals)

    for t in test_quarters:
        pred = kf.predict(ewm_mean)
        kf.update(observed_residual, ewm_mean)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def spherical_R(residuals: NDArray, U: NDArray) -> NDArray:
    """Compute spherical observation noise covariance (Eq. 5).

    R = sigma^2_perp * I_N, where sigma^2_perp is the mean squared
    projection residual of the training data onto the K-dim subspace.

    Args:
        residuals: Training residuals, shape (T, N) or (N, T).
        U: Orthonormal basis, shape (N, K).

    Returns:
        R: Diagonal matrix (N, N).
    """
    N = U.shape[0]
    # Ensure residuals are (T, N)
    if residuals.shape[-1] != N and residuals.shape[0] == N:
        residuals = residuals.T
    # Project and compute residual
    projected = residuals @ U @ U.T
    perp = residuals - projected
    sigma2 = max(float(np.mean(perp ** 2)), 1e-8)
    return np.eye(N) * sigma2


def clip_spectral_radius(F: NDArray, max_sr: float = 0.99) -> NDArray:
    """Clip transition matrix so max|eigenvalue| <= max_sr.

    Applies uniform scalar multiplication: F <- F * min(1, max_sr / max|lambda|).
    This preserves the eigenvector structure exactly.

    Args:
        F: Transition matrix, shape (K, K).
        max_sr: Maximum allowed spectral radius.

    Returns:
        Clipped F, shape (K, K).
    """
    eigvals = np.linalg.eigvals(F)
    mx = float(np.max(np.abs(eigvals)))
    if mx > max_sr:
        return F * (max_sr / mx)
    return F


class SphericalKalmanFilter:
    """Kalman filter for DMD-based panel forecasting with spherical R.

    Operates in K-dimensional modal coordinates. The observation model
    maps modal state alpha to N-dimensional residuals via the basis U.

    Args:
        K: Number of modes (state dimension).
        lambda_q: Exponential smoothing weight for adaptive Q (default 0.3).
        q_init: Initial process noise scale Q_0 = q_init * I_K (default 0.5).
        sr_clip: Spectral radius clipping threshold (default 0.99).
        q_floor: Minimum diagonal value for Q regularisation (default 1e-6).
    """

    def __init__(
        self,
        K: int,
        lambda_q: float = 0.3,
        q_init: float = 0.5,
        sr_clip: float = 0.99,
        q_floor: float = 1e-6,
    ):
        self.K = K
        self.lambda_q = lambda_q
        self.q_init = q_init
        self.sr_clip = sr_clip
        self.q_floor = q_floor

        # State
        self.alpha: NDArray = np.zeros(K)
        self.P: NDArray = np.eye(K)
        self.Q: NDArray = np.eye(K) * q_init

        # Model
        self.F: NDArray = np.eye(K) * 0.99
        self.U: NDArray = np.zeros((0, K))
        self.R: NDArray = np.eye(0)

        # Prediction cache
        self._alpha_pred: NDArray = np.zeros(K)
        self._P_pred: NDArray = np.eye(K)

    def initialise(
        self,
        U: NDArray,
        F: NDArray,
        train_residuals: NDArray,
    ) -> None:
        """Set up the filter with a new basis and transition matrix.

        Args:
            U: Modal basis, shape (N, K).
            F: Transition matrix, shape (K, K). Will be spectral-radius clipped.
            train_residuals: Training residuals for R estimation, shape (T, N).
        """
        N = U.shape[0]
        K = U.shape[1]
        self.K = K
        self.U = U
        self.F = clip_spectral_radius(F, self.sr_clip)
        self.R = spherical_R(train_residuals, U)

        # Reset state
        self.alpha = np.zeros(K)
        self.P = np.eye(K)
        self.Q = np.eye(K) * self.q_init

    def reset_state(self) -> None:
        """Reset filter state (alpha, P, Q) to initial values.

        Call this at each quarterly basis re-estimation to prevent
        the adaptive Q from collapsing to zero.
        """
        self.alpha = np.zeros(self.K)
        self.P = np.eye(self.K)
        self.Q = np.eye(self.K) * self.q_init

    def predict(self, ewm_mean: NDArray | None = None) -> NDArray:
        """One-step-ahead prediction in observation space.

        Args:
            ewm_mean: EWM mean of training residuals, shape (N,).
                Added to the modal prediction. Pass None or zeros if
                residuals are already zero-mean.

        Returns:
            Predicted residual vector, shape (N,).
        """
        # Prediction step
        self._alpha_pred = self.F @ self.alpha
        self._P_pred = self.F @ self.P @ self.F.T + self.Q

        # Map to observation space
        pred = self.U @ self._alpha_pred
        if ewm_mean is not None:
            pred = pred + ewm_mean

        if not np.all(np.isfinite(pred)):
            return np.zeros(self.U.shape[0]) if ewm_mean is None else ewm_mean.copy()
        return pred

    def update(self, observed_residual: NDArray, ewm_mean: NDArray | None = None) -> None:
        """Incorporate new observation and update state.

        Args:
            observed_residual: Actual residual vector, shape (N,).
            ewm_mean: Same as in predict(). Subtracted before modal update.
        """
        N = self.U.shape[0]
        obs = observed_residual
        if ewm_mean is not None:
            obs = obs - ewm_mean

        # Innovation covariance S = U P_pred U' + R
        S = self.U @ self._P_pred @ self.U.T + self.R

        # Kalman gain K = P_pred U' S^{-1}
        try:
            Kg = self._P_pred @ self.U.T @ np.linalg.solve(S, np.eye(N))
        except np.linalg.LinAlgError:
            Kg = np.zeros((self.K, N))

        # State update
        innovation = obs - self.U @ self._alpha_pred
        self.alpha = self._alpha_pred + Kg @ innovation

        # Covariance update (Joseph form for numerical stability)
        IKU = np.eye(self.K) - Kg @ self.U
        self.P = IKU @ self._P_pred @ IKU.T + Kg @ self.R @ Kg.T

        # Adaptive Q update (Eq. 6)
        state_correction = self.alpha - self._alpha_pred
        self.Q = ((1 - self.lambda_q) * self.Q
                  + self.lambda_q * np.outer(state_correction, state_correction))
        self.Q = (self.Q + self.Q.T) / 2 + np.eye(self.K) * self.q_floor

    def run_filter(
        self,
        residuals: NDArray,
        U: NDArray,
        F: NDArray,
        ewm_mean: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Run the full filter on a sequence of residuals.

        Convenience method that initialises and runs predict/update in a loop.

        Args:
            residuals: Observed residuals, shape (T, N).
            U: Modal basis, shape (N, K).
            F: Transition matrix, shape (K, K).
            ewm_mean: Optional EWM mean, shape (N,).

        Returns:
            predictions: One-step-ahead predictions, shape (T, N).
            filtered_states: Filtered modal states, shape (T, K).
        """
        T, N = residuals.shape
        self.initialise(U, F, residuals)

        predictions = np.zeros((T, N))
        filtered_states = np.zeros((T, self.K))

        for t in range(T):
            predictions[t] = self.predict(ewm_mean)
            self.update(residuals[t], ewm_mean)
            filtered_states[t] = self.alpha.copy()

        return predictions, filtered_states
