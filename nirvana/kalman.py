"""
Causal Online Kalman Filter Module
===================================
State-space dynamic filtering with strictly causal updates (no lookahead bias).
Eliminates high-frequency microstructure noise and non-stationary baseline drift.
"""

from typing import Optional, Union
import numpy as np


class CausalOnlineKalman:
    """
    1D Causal Online Kalman Filter.
    x_k = x_{k-1} + w_k,  w_k ~ N(0, q)
    z_k = x_k + v_k,      v_k ~ N(0, r)
    """

    def __init__(self, q: float = 1e-3, r: float = 1e-2):
        self.q = float(q)
        self.r = float(r)
        self.k_est: Optional[float] = None
        self.p_est: float = 1.0

    def reset(self):
        self.k_est = None
        self.p_est = 1.0

    def update(self, measurement: float) -> float:
        if measurement is None or np.isnan(measurement):
            if self.k_est is not None:
                self.p_est += self.q
                return self.k_est
            return 0.0

        if self.k_est is None:
            self.k_est = float(measurement)
            return self.k_est

        p_pred = self.p_est + self.q
        k_gain = p_pred / (p_pred + self.r)
        self.k_est += k_gain * (float(measurement) - self.k_est)
        self.p_est = (1.0 - k_gain) * p_pred
        return self.k_est

    def filter_series(self, series: Union[list, np.ndarray]) -> np.ndarray:
        """Filter entire 1D array sequentially maintaining strict causality."""
        self.reset()
        res = np.empty(len(series), dtype=float)
        for i, val in enumerate(series):
            res[i] = self.update(val)
        return res


class MultivariateCausalKalman:
    """
    Multivariate Online Causal Kalman Filter for tracking N interconnected banking nodes.
    Features:
    - Mean-reverting AR(1) state transition (phi parameter) to prevent post-crisis drift
    - Symmetrized error covariance P = 0.5*(P + P.T) for numerical stability across 2,000+ sessions
    - Dynamic partial observation update for unbalanced IPO panels
    """

    def __init__(
        self,
        n_dim: int,
        q_diag: float = 1e-3,
        r_diag: Union[float, np.ndarray, list] = 1e-2,
        phi: float = 0.98,
        outlier_threshold: Optional[float] = None,
    ):
        self.n = n_dim
        self.phi = float(phi)
        self.outlier_threshold = float(outlier_threshold) if outlier_threshold is not None else None
        self.Q = np.eye(self.n) * float(q_diag)
        if isinstance(r_diag, (np.ndarray, list)):
            self.R = np.diag(np.asarray(r_diag, dtype=float))
        else:
            self.R = np.eye(self.n) * float(r_diag)
        self.state: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.P = np.eye(self.n)

    def reset(self):
        self.state = None
        self.mu = None
        self.P = np.eye(self.n)

    def update(self, measurement_vector: np.ndarray) -> np.ndarray:
        z = np.asarray(measurement_vector, dtype=float)
        if self.state is None:
            valid_mask = ~np.isnan(z)
            if not np.any(valid_mask):
                # Defer initialization until first session with at least one valid measurement
                return np.zeros(self.n, dtype=float)
            fill_val = float(np.nanmean(z))
            self.state = np.where(valid_mask, z, fill_val)
            self.mu = self.state.copy()
            return self.state

        # Prediction step with Mean-Reversion: x_{t|t-1} = mu + phi * (x_{t-1} - mu)
        self.state = self.mu + self.phi * (self.state - self.mu)
        P_pred = (self.phi ** 2) * self.P + self.Q

        observed_mask = ~np.isnan(z)
        if not np.any(observed_mask):
            self.P = 0.5 * (P_pred + P_pred.T)
            return self.state

        if np.all(observed_mask):
            # All variables observed
            S = P_pred + self.R
            K = P_pred @ np.linalg.pinv(S)
            innovation = z - self.state
            if self.outlier_threshold is not None:
                # Huber-robust bounding on normalized innovation to protect from fat-tail jumps
                std_innov = np.sqrt(np.maximum(np.diag(S), 1e-8))
                z_score = innovation / std_innov
                clipped_z = np.clip(z_score, -self.outlier_threshold, self.outlier_threshold)
                innovation = clipped_z * std_innov

            self.state = self.state + K @ innovation
            # Joseph stabilized covariance update: P = (I - K) P_pred (I - K)^T + K R K^T
            I_K = np.eye(self.n) - K
            P_new = I_K @ P_pred @ I_K.T + K @ self.R @ K.T
            self.P = 0.5 * (P_new + P_new.T)
            np.fill_diagonal(self.P, np.maximum(np.diag(self.P), 1e-8))
            # Slowly update baseline mean (alpha = 0.005)
            self.mu += 0.005 * (self.state - self.mu)
            return self.state

        # Partial observation update (Dynamic node entry / Unbalanced panel)
        obs_idx = np.where(observed_mask)[0]
        H_obs = np.eye(self.n)[obs_idx]  # shape (k, n)
        R_obs = self.R[np.ix_(obs_idx, obs_idx)]  # shape (k, k)
        z_obs = z[obs_idx]  # shape (k,)

        S_obs = H_obs @ P_pred @ H_obs.T + R_obs
        K_obs = P_pred @ H_obs.T @ np.linalg.pinv(S_obs)
        innovation_obs = z_obs - H_obs @ self.state
        if self.outlier_threshold is not None:
            std_innov_obs = np.sqrt(np.maximum(np.diag(S_obs), 1e-8))
            z_score_obs = innovation_obs / std_innov_obs
            clipped_z_obs = np.clip(z_score_obs, -self.outlier_threshold, self.outlier_threshold)
            innovation_obs = clipped_z_obs * std_innov_obs

        self.state = self.state + K_obs @ innovation_obs
        # Joseph stabilized form: P = (I - K_obs H_obs) P_pred (I - K_obs H_obs)^T + K_obs R_obs K_obs^T
        I_KH = np.eye(self.n) - K_obs @ H_obs
        P_new = I_KH @ P_pred @ I_KH.T + K_obs @ R_obs @ K_obs.T
        self.P = 0.5 * (P_new + P_new.T)
        np.fill_diagonal(self.P, np.maximum(np.diag(self.P), 1e-8))
        self.mu += 0.005 * (self.state - self.mu)

        return self.state

    def filter_dataframe(self, df_values: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
        """Processes (T, N) matrix or DataFrame sequentially."""
        self.reset()
        mat = getattr(df_values, "values", df_values)
        mat = np.asarray(mat, dtype=float)
        t, n = mat.shape
        out = np.empty((t, n), dtype=float)
        for i in range(t):
            out[i] = self.update(mat[i])
        return out
