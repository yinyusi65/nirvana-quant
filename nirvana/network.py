"""
Nirvana Systemic Risk Connectedness Engine
==========================================
Replaces / enhances traditional Diebold-Yılmaz (2012, 2014) TVP-VAR GFEVD with:
- Causal State-Space Kalman Filtering (zero lookahead)
- Base-12 Manifold Phase Resonance Tracking
- Apoptosis-pruned Causal Transfer Estimation

Produces academic-grade Systemic Risk Tables:
- Pairwise Directional Connectedness (S_ij)
- Directional TO Others (Transmitter of Risk)
- Directional FROM Others (Receiver of Risk)
- NET Systemic Spillover (NET = TO - FROM)
- Total Connectedness Index (TCI)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd

from .kalman import MultivariateCausalKalman
from .manifold import Base12Manifold


@dataclass
class SystemicRiskReport:
    spillover_table: pd.DataFrame
    total_connectedness_index: float
    normalized_tci: float
    net_spillovers: Dict[str, float]
    top_transmitters: List[str]
    top_receivers: List[str]
    asset_weighted: bool = False
    p_values_table: Optional[pd.DataFrame] = None
    eigenvector_hub: Optional[Dict[str, float]] = None
    eigenvector_authority: Optional[Dict[str, float]] = None
    betweenness_centrality: Optional[Dict[str, float]] = None
    flight_to_safety_index: Optional[Dict[str, float]] = None
    market_beta: Optional[Dict[str, float]] = None
    market_variance_share: Optional[float] = None

    def summary(self) -> str:
        lines = []
        lines.append("=" * 78)
        lines.append("NIRVANA SYSTEMIC RISK CONNECTEDNESS REPORT (BANKING NETWORK)")
        lines.append("=" * 78)
        lines.append(f"Raw Total Connectedness Index (TCI):        {self.total_connectedness_index:.2f}%")
        lines.append(f"Dimension-Normalized TCI (Adjusted for N):  {self.normalized_tci:.2f}%")
        lines.append(f"Asset-Weighted Footprint Active:            {'YES' if self.asset_weighted else 'NO'}")
        if self.market_variance_share is not None:
            lines.append(f"Market Orthogonalization Active:            YES (Market Factor: {self.market_variance_share:.2f}% var)")
        if self.p_values_table is not None:
            lines.append("Bootstrap Edge Significance Active:         YES (p < 0.05 Apoptosis)")
        lines.append("\nSpillover Matrix (%):")
        lines.append(self.spillover_table.to_string())
        lines.append("\nTop Systemic Risk Transmitters (NET > 0):")
        for b in self.top_transmitters:
            lines.append(f"  - {b}: +{self.net_spillovers[b]:.2f}%")
        lines.append("\nTop Systemic Risk Receivers / Absorbers (NET < 0):")
        for b in self.top_receivers:
            lines.append(f"  - {b}: {self.net_spillovers[b]:.2f}%")
        if self.eigenvector_hub:
            lines.append("\nTop Hub Centrality (Systemic Shock Transmitters):")
            sorted_hub = sorted(self.eigenvector_hub.items(), key=lambda x: x[1], reverse=True)[:5]
            for b, score in sorted_hub:
                lines.append(f"  - {b}: {score:.2f}%")
        if self.eigenvector_authority:
            lines.append("\nTop Authority Centrality (Systemic Shock Vulnerability / Sinks):")
            sorted_auth = sorted(self.eigenvector_authority.items(), key=lambda x: x[1], reverse=True)[:5]
            for b, score in sorted_auth:
                lines.append(f"  - {b}: {score:.2f}%")
        if self.betweenness_centrality:
            lines.append("\nTop Betweenness Centrality (Bridge / Transmission Bottlenecks):")
            sorted_bc = sorted(self.betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            for b, score in sorted_bc:
                lines.append(f"  - {b}: {score:.2f}%")
        if self.flight_to_safety_index:
            lines.append("\nTop Flight-to-Safety Absorbers (Safe-Haven Assets):")
            sorted_fts = sorted(self.flight_to_safety_index.items(), key=lambda x: x[1], reverse=True)[:5]
            for b, score in sorted_fts:
                lines.append(f"  - {b}: {score:.2f}%")
        lines.append("=" * 78)
        return "\n".join(lines)


@dataclass
class DynamicSystemicRiskReport:
    tci_history: pd.Series
    normalized_tci_history: pd.Series
    rolling_net_spillovers: pd.DataFrame
    phase_transition_dates: List[Union[pd.Timestamp, str]]
    mean_tci: float
    max_tci: float
    max_tci_date: Optional[Union[pd.Timestamp, str]] = None

    def summary(self) -> str:
        lines = []
        lines.append("=" * 78)
        lines.append("NIRVANA DYNAMIC TIME-VARYING RISK REPORT (ROLLING CONNECTEDNESS)")
        lines.append("=" * 78)
        lines.append(f"Mean Rolling TCI:             {self.mean_tci:.2f}%")
        lines.append(f"Max TCI Peak Stress:          {self.max_tci:.2f}% (Observed at: {self.max_tci_date})")
        lines.append(f"Detected Phase Transitions:   {len(self.phase_transition_dates)} events")
        for d in self.phase_transition_dates[:5]:
            lines.append(f"  - Surge Transition: {d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else d}")
        lines.append("=" * 78)
        return "\n".join(lines)


# Decoupled domain-specific parameters imported from banking domain module
from .domains.banking import DEFAULT_EXCHANGE_BANDS, DEFAULT_ASSET_WEIGHTS


class NirvanaConnectedness:
    """
    Connectedness estimator for banking networks.
    Features:
    - Base-12 Dual-Clock Manifold harmonic transformation
    - Causal Online Multivariate Kalman Filter
    - Exchange Price-Band Normalization (HOSE 7% vs HNX 10% vs UPCoM 15%)
    - Multi-lag asymmetric causal transmission (T+1, T+2, T+3 settlement cycle)
    - Sub-linear Asset Elasticity Footprint (sqrt assets)
    - Bootstrap edge significance & Empirical Fisher Apoptosis
    - Dimension-adjusted Normalized TCI
    """

    def __init__(
        self,
        bank_names: Optional[List[str]] = None,
        lags: Optional[Union[int, List[int]]] = None,
        lag_weights: Optional[List[float]] = None,
        multi_lag: bool = True,
        kalman_q: float = 1e-3,
        kalman_r: float = 1e-2,
        kalman_phi: float = 0.98,
        kalman_outlier_threshold: Optional[float] = None,
        apoptosis_threshold: float = 0.015,
        p_value_threshold: float = 0.05,
        n_boot: int = 0,
        orthogonalize_market: bool = False,
        n_market_factors: int = 1,
        causality_mode: str = "linear",
        threshold_quantile: float = 0.70,
        normalize_bands: bool = True,
        exchange_bands: Optional[Dict[str, float]] = None,
        asset_weights: Optional[Union[Dict[str, float], bool, str]] = None,
        asset_elasticity: float = 0.5,
        volume_quantum: float = 1000.0,
        macro_periods: Optional[Tuple[float, float]] = None,
    ):
        self.bank_names = bank_names
        self.lags = lags
        self.lag_weights = lag_weights
        self.multi_lag = multi_lag
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r
        self.kalman_phi = kalman_phi
        self.kalman_outlier_threshold = kalman_outlier_threshold
        self.apoptosis_threshold = apoptosis_threshold
        self.p_value_threshold = p_value_threshold
        self.n_boot = n_boot
        self.orthogonalize_market = orthogonalize_market
        self.n_market_factors = n_market_factors
        self.causality_mode = causality_mode
        self.threshold_quantile = threshold_quantile
        self.normalize_bands = normalize_bands
        self.exchange_bands = exchange_bands
        self.asset_weights = asset_weights
        self.asset_elasticity = asset_elasticity
        self.volume_quantum = volume_quantum
        self.macro_periods = macro_periods
        self.manifold = Base12Manifold(volume_quantum=volume_quantum, macro_periods=macro_periods)

    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        asset_weights: Optional[Union[Dict[str, float], bool, str]] = None,
        asset_elasticity: Optional[float] = None,
        normalize_bands: Optional[bool] = None,
        multi_lag: Optional[bool] = None,
        lags: Optional[Union[int, List[int]]] = None,
        lag_weights: Optional[List[float]] = None,
        n_boot: Optional[int] = None,
        orthogonalize_market: Optional[bool] = None,
        exchange_bands: Optional[Dict[str, float]] = None,
        kalman_q: Optional[float] = None,
        kalman_r: Optional[float] = None,
        kalman_phi: Optional[float] = None,
        apoptosis_threshold: Optional[float] = None,
        p_value_threshold: Optional[float] = None,
        n_market_factors: Optional[int] = None,
        causality_mode: Optional[str] = None,
        threshold_quantile: Optional[float] = None,
        kalman_outlier_threshold: Optional[float] = None,
        macro_periods: Optional[Tuple[float, float]] = None,
    ) -> SystemicRiskReport:
        # Resolve hyperparameters (fit-level overrides self-level)
        active_lags = self.lags if lags is None else lags
        active_lag_weights = self.lag_weights if lag_weights is None else lag_weights
        active_multi_lag = self.multi_lag if multi_lag is None else multi_lag
        active_kalman_q = self.kalman_q if kalman_q is None else kalman_q
        active_kalman_r = self.kalman_r if kalman_r is None else kalman_r
        active_kalman_phi = self.kalman_phi if kalman_phi is None else kalman_phi
        active_kalman_outlier_thresh = (
            self.kalman_outlier_threshold
            if kalman_outlier_threshold is None
            else kalman_outlier_threshold
        )
        active_macro_periods = self.macro_periods if macro_periods is None else macro_periods
        active_apoptosis_threshold = (
            self.apoptosis_threshold if apoptosis_threshold is None else apoptosis_threshold
        )
        active_p_val_thresh = (
            self.p_value_threshold if p_value_threshold is None else p_value_threshold
        )
        active_n_boot = self.n_boot if n_boot is None else n_boot
        active_ortho = (
            self.orthogonalize_market if orthogonalize_market is None else orthogonalize_market
        )
        active_n_market_factors = (
            self.n_market_factors if n_market_factors is None else n_market_factors
        )
        active_causality_mode = (
            self.causality_mode if causality_mode is None else causality_mode
        )
        active_threshold_quantile = (
            self.threshold_quantile if threshold_quantile is None else threshold_quantile
        )
        active_norm_bands = self.normalize_bands if normalize_bands is None else normalize_bands
        active_exchange_bands = self.exchange_bands if exchange_bands is None else exchange_bands
        active_asset_weights = self.asset_weights if asset_weights is None else asset_weights
        active_asset_elasticity = (
            self.asset_elasticity if asset_elasticity is None else asset_elasticity
        )

        if active_asset_weights is True or active_asset_weights == "default":
            active_asset_weights = DEFAULT_ASSET_WEIGHTS

        if isinstance(data, pd.DataFrame):
            names = list(data.columns)
            raw_mat = data.values.copy()
        else:
            names = self.bank_names or [f"B_{i+1}" for i in range(data.shape[1])]
            raw_mat = data.copy()

        t_steps, n_nodes = raw_mat.shape

        if n_nodes < 2:
            single_table = pd.DataFrame(
                [[100.0, 0.0]], index=names, columns=names + ["FROM_others"]
            )
            row_to = pd.Series([0.0, 0.0], index=names + ["FROM_others"], name="TO_others")
            row_net = pd.Series([0.0, 0.0], index=names + ["FROM_others"], name="NET_spillover")
            single_table = pd.concat([single_table, row_to.to_frame().T, row_net.to_frame().T])
            return SystemicRiskReport(
                spillover_table=single_table,
                total_connectedness_index=0.0,
                normalized_tci=0.0,
                net_spillovers={names[0]: 0.0},
                top_transmitters=[],
                top_receivers=[],
                asset_weighted=False,
                eigenvector_hub={names[0]: 100.0},
                eigenvector_authority={names[0]: 100.0},
                betweenness_centrality={names[0]: 0.0},
                flight_to_safety_index={names[0]: 0.0},
            )

        # 0. Exchange Price-Band Heteroskedasticity Mapping
        # In state-space filtering, wider price bands (UPCoM 15%, HNX 10%) imply wider microstructure noise.
        # We calibrate this into per-node observation noise covariance R_i rather than distorting log-price scales.
        r_vec = np.full(n_nodes, active_kalman_r)
        if active_norm_bands and isinstance(data, pd.DataFrame):
            band_map = active_exchange_bands or DEFAULT_EXCHANGE_BANDS
            for col_idx, b in enumerate(names):
                band = band_map.get(b, 0.07)
                r_vec[col_idx] = active_kalman_r * ((band / 0.07) ** 2)

        # 1. Causal Kalman State Estimation with Mean Reversion & Exchange Noise Calibration
        mv_kalman = MultivariateCausalKalman(
            n_dim=n_nodes,
            q_diag=active_kalman_q,
            r_diag=r_vec,
            phi=active_kalman_phi,
            outlier_threshold=active_kalman_outlier_thresh,
        )
        filtered_states = mv_kalman.filter_dataframe(raw_mat)

        # 2. Dynamic Innovation Residuals
        innovations = raw_mat - filtered_states

        # 2b. Multi-Factor SVD Market Orthogonalization
        market_beta_dict = None
        market_var_share = None
        if active_ortho:
            k_factors = min(max(1, active_n_market_factors), n_nodes - 1)
            u, s, _ = np.linalg.svd(innovations - np.mean(innovations, axis=0), full_matrices=False)
            factors = u[:, :k_factors] * s[:k_factors]
            sys_mean = np.mean(innovations, axis=1)
            for f_idx in range(k_factors):
                if np.corrcoef(factors[:, f_idx], sys_mean)[0, 1] < 0:
                    factors[:, f_idx] = -factors[:, f_idx]
            market_var_share = float(np.sum(s[:k_factors]**2) / np.sum(s**2) * 100.0)

            idio_innovations = np.empty_like(innovations)
            betas = {}
            X_reg = np.column_stack([np.ones(len(factors)), factors])
            for col_idx in range(n_nodes):
                y = innovations[:, col_idx]
                b = np.linalg.lstsq(X_reg, y, rcond=None)[0]
                idio_innovations[:, col_idx] = y - X_reg @ b
                betas[names[col_idx]] = float(np.round(b[1], 3)) if k_factors == 1 else [float(np.round(val, 3)) for val in b[1:]]

            innovations = idio_innovations
            market_beta_dict = betas

        # 3. Base-12 Manifold Harmonic Phase Transform (Aligned with financial cycles)
        calendar_idx = np.arange(t_steps)
        dates_arg = data.index if isinstance(data, pd.DataFrame) else None
        manifold_engine = (
            Base12Manifold(volume_quantum=self.volume_quantum, macro_periods=active_macro_periods)
            if active_macro_periods is not None
            else self.manifold
        )
        manifold_coords = manifold_engine.batch_transform(calendar_idx, dates=dates_arg)
        gamma = manifold_coords[:, 8]  # Phase resonance index

        # 4. Causal Transfer & Dynamic Covariance Matrix
        weighted_innovations = innovations * (1.0 + 0.5 * gamma[:, None])
        min_p = min(10, max(2, t_steps - 1))
        cov_df = pd.DataFrame(weighted_innovations).cov(min_periods=min_p).fillna(0.0)
        cov_mat = cov_df.values

        # Standardize into correlation / impact matrix
        std_devs = np.sqrt(np.maximum(np.diag(cov_mat), 1e-8))
        corr_mat = cov_mat / (np.outer(std_devs, std_devs) + 1e-8)
        np.nan_to_num(corr_mat, copy=False, nan=0.0)

        # 5. Multi-Lag Directional Asymmetry (T+2.5 Settlement Flow: Lags 1, 2, 3)
        # Canonical Diebold-Yilmaz (2012, 2014) convention:
        # K_ij is impact ON bank i (Row i = Target, Future) FROM bank j (Col j = Source, Past)
        if active_lags is None:
            lags_to_eval = [1, 2, 3] if active_multi_lag else [1]
        elif isinstance(active_lags, int):
            lags_to_eval = list(range(1, active_lags + 1))
        else:
            lags_to_eval = list(active_lags)

        if active_lag_weights is None:
            decay_weights = np.array([0.5, 0.3, 0.2] if active_multi_lag else [1.0])
            if len(lags_to_eval) != len(decay_weights):
                decay_weights = np.ones(len(lags_to_eval)) / float(len(lags_to_eval))
        else:
            raw_lw = np.asarray(active_lag_weights, dtype=float)
            decay_weights = raw_lw / (np.sum(raw_lw) + 1e-8)

        def _calc_lead_lag(mat: np.ndarray) -> np.ndarray:
            asym = np.zeros((n_nodes, n_nodes))
            for l_idx, lag in enumerate(lags_to_eval):
                p = mat[:-lag]
                f = mat[lag:]
                if active_causality_mode == "threshold":
                    # Upper quantile threshold activation for tail/panic shocks
                    q_val = np.quantile(np.abs(p), active_threshold_quantile, axis=0, keepdims=True)
                    p_act = np.where(np.abs(p) >= q_val, p, 0.0)
                    p_norm = (p_act - np.mean(p_act, axis=0)) / (np.std(p_act, axis=0) + 1e-8)
                    f_norm = (f - np.mean(f, axis=0)) / (np.std(f, axis=0) + 1e-8)
                else:
                    p_norm = (p - np.mean(p, axis=0)) / (np.std(p, axis=0) + 1e-8)
                    f_norm = (f - np.mean(f, axis=0)) / (np.std(f, axis=0) + 1e-8)
                c_mat = (f_norm.T @ p_norm) / float(len(p) - 1)
                np.nan_to_num(c_mat, copy=False, nan=0.0)
                # Anti-symmetric net directional lead: Delta_ij = c_ij - c_ji
                # Delta_ij > 0 means source j leads target i
                asym += decay_weights[l_idx] * (c_mat - c_mat.T)
            np.fill_diagonal(asym, 0.0)
            return np.clip(asym, -1.0, 1.0)

        lead_lag_asymmetry = _calc_lead_lag(weighted_innovations)

        # 6. Raw Spillover Kernel: K_ij = Corr_ij * Asym_ij (Row i = Target, Col j = Source)
        kernel = np.abs(corr_mat) * (0.5 + 0.5 * lead_lag_asymmetry)

        # 7. Bootstrap Significance Testing (if active_n_boot > 0)
        p_val_mat = None
        if active_n_boot > 0:
            # ponytail: circular roll per column preserves autocorrelation while destroying cross-correlation
            boot_exceed_counts = np.zeros((n_nodes, n_nodes))
            rng = np.random.default_rng(seed=42)

            for _ in range(active_n_boot):
                shuffled_innov = np.empty_like(weighted_innovations)
                for col in range(n_nodes):
                    shift = int(rng.integers(20, t_steps - 20))
                    shuffled_innov[:, col] = np.roll(weighted_innovations[:, col], shift)

                boot_corr = np.abs(np.corrcoef(shuffled_innov, rowvar=False))
                np.nan_to_num(boot_corr, copy=False, nan=0.0)
                boot_asym = _calc_lead_lag(shuffled_innov)
                boot_stat = boot_corr * (0.5 + 0.5 * boot_asym)
                boot_exceed_counts += (boot_stat >= kernel)

            p_val_mat = boot_exceed_counts / float(active_n_boot)
            # Excise non-significant edges (p >= active_p_val_thresh) via Apoptosis
            insignificant_mask = (p_val_mat >= active_p_val_thresh) & ~np.eye(n_nodes, dtype=bool)
            kernel[insignificant_mask] = 0.0

        # Baseline Apoptosis threshold pruning
        kernel[kernel < active_apoptosis_threshold] = 0.0
        # Standardized own-variance diagonal is 1.0 (consistent with correlation off-diagonals)
        np.fill_diagonal(kernel, 1.0)

        # 8. Row normalization (Each bank's total variance explained = 100%)
        row_sums = kernel.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        norm_kernel = (kernel / row_sums) * 100.0

        # 9. Compute Academic Metrics
        from_others = np.sum(norm_kernel, axis=1) - np.diag(norm_kernel)
        raw_to_others = np.sum(norm_kernel, axis=0) - np.diag(norm_kernel)

        # Asset-weighting modulation
        is_weighted = False
        if active_asset_weights is not None:
            # Sub-linear asset elasticity (default 0.5 = square root of assets)
            valid_vals = [float(v) for v in active_asset_weights.values() if float(v) > 0]
            median_w = float(np.median(valid_vals)) if valid_vals else 100.0
            raw_w = np.array([float(active_asset_weights.get(b, median_w)) ** active_asset_elasticity for b in names])
            mean_w = np.mean(raw_w) if np.mean(raw_w) > 0 else 1.0
            norm_w = raw_w / mean_w
            to_others = raw_to_others * norm_w
            is_weighted = True
        else:
            to_others = raw_to_others

        net_spillover = to_others - from_others

        # Total Connectedness Index (Raw & Dimension-Normalized)
        tci = float(np.sum(from_others) / n_nodes)
        max_possible_tci = (n_nodes - 1.0) / float(n_nodes) * 100.0
        normalized_tci = float((tci / max_possible_tci) * 100.0)

        # Build Presentation DataFrame
        table = pd.DataFrame(
            np.round(norm_kernel, 2),
            index=names,
            columns=names,
        )
        table["FROM_others"] = np.round(from_others, 2)

        row_to = pd.Series(np.round(to_others, 2), index=names, name="TO_others")
        row_to["FROM_others"] = np.round(np.sum(from_others), 2)

        row_net = pd.Series(np.round(net_spillover, 2), index=names, name="NET_spillover")
        row_net["FROM_others"] = np.round(tci, 2)

        table = pd.concat([table, row_to.to_frame().T, row_net.to_frame().T])

        net_dict = {names[i]: float(net_spillover[i]) for i in range(n_nodes)}
        sorted_transmitters = sorted(
            [k for k, v in net_dict.items() if v > 0],
            key=lambda k: net_dict[k],
            reverse=True,
        )
        sorted_receivers = sorted(
            [k for k, v in net_dict.items() if v <= 0],
            key=lambda k: net_dict[k],
        )

        p_val_df = pd.DataFrame(np.round(p_val_mat, 3), index=names, columns=names) if p_val_mat is not None else None

        # 10. Complex Graph Topology Centrality Metrics
        # Adjacency A = norm_kernel with zero diagonal (A_ij is impact on i from j)
        A = norm_kernel.copy()
        np.fill_diagonal(A, 0.0)

        # Hub Centrality: Transmitting power of source j into the network (Perron-Frobenius damped power iteration on A^T)
        v_hub = np.ones(n_nodes) / np.sqrt(n_nodes)
        v_teleport = np.ones(n_nodes) / np.sqrt(n_nodes)
        for _ in range(50):
            v_next = 0.85 * (A.T @ v_hub) + 0.15 * v_teleport
            norm = np.linalg.norm(v_next)
            if norm > 1e-12:
                v_hub = v_next / norm
            else:
                break
        hub_centrality = {names[i]: float(np.round(v_hub[i] / (np.sum(v_hub) + 1e-12) * 100.0, 2)) for i in range(n_nodes)}

        # Authority / Vulnerability Centrality: Vulnerability of receiver i absorbing from system (Perron-Frobenius damped power iteration on A)
        v_auth = np.ones(n_nodes) / np.sqrt(n_nodes)
        for _ in range(50):
            v_next = 0.85 * (A @ v_auth) + 0.15 * v_teleport
            norm = np.linalg.norm(v_next)
            if norm > 1e-12:
                v_auth = v_next / norm
            else:
                break
        authority_centrality = {names[i]: float(np.round(v_auth[i] / (np.sum(v_auth) + 1e-12) * 100.0, 2)) for i in range(n_nodes)}

        # Betweenness Centrality (Bridge / Bottleneck Banks via Floyd-Warshall on transmission friction)
        # Directed transmission: edge u -> v transmits shock with strength norm_kernel[v, u] (norm_kernel.T[u, v]).
        A_trans = norm_kernel.T.copy()
        np.fill_diagonal(A_trans, 0.0)
        D = np.where(A_trans > 0, 1.0 / (A_trans + 1e-4), 1e6)
        np.fill_diagonal(D, 0.0)

        for k in range(n_nodes):
            D = np.minimum(D, D[:, k:k+1] + D[k:k+1, :])

        bc = np.zeros(n_nodes)
        for k in range(n_nodes):
            diff = (D[:, k:k+1] + D[k:k+1, :]) - D
            is_on_path = (np.abs(diff) < 1e-5) & (D < 1e5)
            is_on_path[k, :] = False
            is_on_path[:, k] = False
            np.fill_diagonal(is_on_path, False)
            bc[k] = np.sum(is_on_path)

        denom = (n_nodes - 1) * (n_nodes - 2) if n_nodes > 2 else 1.0
        bc_norm = (bc / denom) * 100.0
        betweenness_centrality = {names[i]: float(np.round(bc_norm[i], 2)) for i in range(n_nodes)}

        # Flight-to-Safety Index (Safe-haven absorption under systemic lead-lag stress)
        # Vectorized calculation: safe havens exhibit negative correlation with systemic distress
        fts = np.zeros(n_nodes)
        for l_idx, lag in enumerate(lags_to_eval):
            p = innovations[:-lag]
            f = innovations[lag:]
            p_norm = (p - np.mean(p, axis=0)) / (np.std(p, axis=0) + 1e-8)
            f_norm = (f - np.mean(f, axis=0)) / (np.std(f, axis=0) + 1e-8)
            c_mat = (f_norm.T @ p_norm) / float(len(p) - 1)
            np.nan_to_num(c_mat, copy=False, nan=0.0)
            neg_c = np.maximum(-c_mat, 0.0)
            np.fill_diagonal(neg_c, 0.0)
            fts += decay_weights[l_idx] * np.sum(neg_c, axis=1)

        fts_norm = (fts / (np.sum(fts) + 1e-12)) * 100.0 if np.sum(fts) > 0 else fts
        flight_to_safety = {names[i]: float(np.round(fts_norm[i], 2)) for i in range(n_nodes)}

        return SystemicRiskReport(
            spillover_table=table,
            total_connectedness_index=tci,
            normalized_tci=normalized_tci,
            net_spillovers=net_dict,
            top_transmitters=sorted_transmitters,
            top_receivers=sorted_receivers,
            asset_weighted=is_weighted,
            p_values_table=p_val_df,
            eigenvector_hub=hub_centrality,
            eigenvector_authority=authority_centrality,
            betweenness_centrality=betweenness_centrality,
            flight_to_safety_index=flight_to_safety,
            market_beta=market_beta_dict,
            market_variance_share=market_var_share,
        )

    def fit_regimes(
        self,
        data: pd.DataFrame,
        regimes: Optional[Dict[str, tuple]] = None,
        asset_weights: Optional[Dict[str, float]] = None,
        **fit_kwargs,
    ) -> Dict[str, SystemicRiskReport]:
        """
        Fits Nirvana connectedness across distinct macro/financial regimes.
        Default regimes:
        - Pre-Covid (2017-2019)
        - Covid-19 Crisis (2020-2021)
        - Bond & SCB Shock (2022-2023)
        - Post-Crisis Recovery (2024-2025)
        """
        if regimes is None:
            regimes = {
                "Pre-Covid (2017-2019)": ("2017-01-01", "2019-12-31"),
                "Covid-19 Crisis (2020-2021)": ("2020-01-01", "2021-12-31"),
                "Bond & SCB Shock (2022-2023)": ("2022-01-01", "2023-12-31"),
                "Post-Crisis Recovery (2024-2025)": ("2024-01-01", "2025-12-31"),
            }

        reports = {}
        is_dt = isinstance(data.index, pd.DatetimeIndex)
        for regime_name, (start_d, end_d) in regimes.items():
            if is_dt:
                start_ts = pd.to_datetime(start_d)
                end_ts = pd.to_datetime(end_d) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
                mask = (data.index >= start_ts) & (data.index <= end_ts)
            else:
                mask = (data.index >= str(start_d)) & (data.index <= str(end_d))
            sub_df = data.loc[mask]
            if len(sub_df) < 15:
                continue
            rep = self.fit(sub_df, asset_weights=asset_weights, **fit_kwargs)
            reports[regime_name] = rep

        return reports

    def rolling_tci(
        self,
        data: pd.DataFrame,
        window: int = 200,
        step: int = 20,
        asset_weights: Optional[Dict[str, float]] = None,
        **fit_kwargs,
    ) -> pd.Series:
        """
        Calculates time-varying dynamic Total Connectedness Index (TCI) across rolling windows.
        """
        dates = data.index
        n = len(data)
        tci_series = {}

        for end_idx in range(window, n + 1, step):
            sub_df = data.iloc[end_idx - window : end_idx].ffill().bfill().fillna(0.0)
            rep = self.fit(sub_df, asset_weights=asset_weights, **fit_kwargs)
            tci_series[dates[end_idx - 1]] = rep.total_connectedness_index

        return pd.Series(tci_series, name="Nirvana_Rolling_TCI")

    def rolling_fit(
        self,
        data: pd.DataFrame,
        window: int = 120,
        step: int = 10,
        asset_weights: Optional[Dict[str, float]] = None,
        surge_std_threshold: float = 2.0,
        **fit_kwargs,
    ) -> DynamicSystemicRiskReport:
        """
        Executes rolling time-varying network estimation across windows.
        Tracks dynamic TCI, normalized TCI, per-node Net spillovers,
        and flags structural phase transitions (spikes > surge_std_threshold * rolling std).
        """
        dates = data.index
        n = len(data)
        tci_dict = {}
        norm_tci_dict = {}
        net_records = []
        date_records = []

        for end_idx in range(window, n + 1, step):
            sub_df = data.iloc[end_idx - window : end_idx].ffill().bfill().fillna(0.0)
            rep = self.fit(sub_df, asset_weights=asset_weights, **fit_kwargs)
            d = dates[end_idx - 1]
            tci_dict[d] = rep.total_connectedness_index
            norm_tci_dict[d] = rep.normalized_tci
            net_records.append(rep.net_spillovers)
            date_records.append(d)

        tci_series = pd.Series(tci_dict, name="Rolling_TCI")
        norm_tci_series = pd.Series(norm_tci_dict, name="Rolling_Normalized_TCI")
        rolling_net_df = pd.DataFrame(net_records, index=date_records)

        # Detect Phase Transitions (sudden surges exceeding moving baseline)
        phase_transitions = []
        if len(tci_series) >= 5:
            rolling_mean = tci_series.rolling(min(10, len(tci_series)), min_periods=3).mean()
            rolling_std = tci_series.rolling(min(10, len(tci_series)), min_periods=3).std().fillna(1.0)
            z_surges = (tci_series - rolling_mean) / (rolling_std + 1e-6)
            surge_mask = z_surges >= surge_std_threshold
            phase_transitions = list(tci_series.index[surge_mask])

        max_tci_val = float(tci_series.max()) if len(tci_series) > 0 else 0.0
        max_tci_dt = tci_series.idxmax() if len(tci_series) > 0 else None
        mean_tci_val = float(tci_series.mean()) if len(tci_series) > 0 else 0.0

        return DynamicSystemicRiskReport(
            tci_history=tci_series,
            normalized_tci_history=norm_tci_series,
            rolling_net_spillovers=rolling_net_df,
            phase_transition_dates=phase_transitions,
            mean_tci=mean_tci_val,
            max_tci=max_tci_val,
            max_tci_date=max_tci_dt,
        )
