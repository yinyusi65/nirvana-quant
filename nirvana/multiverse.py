"""
Multiverse Counterfactual Reconstruction Engine
===============================================
Synthesizes unobserved latent market volatility for pre-IPO periods (2017 - 2020)
anchored on:
1. Structural Peer-Group Dynamics (Big4, Tier-1 Retail, Tier-2 Commercial, Tier-3 Regional)
2. Macroeconomic Market Factors (VN-Index volatility & co-movement)
3. Base-12 Dual-Clock Manifold Harmonics (Calendar & Quarterly BCTC cycles)
4. Multiverse Ensemble Simulation with Phase 2 Apoptosis Pruning

Eliminates the 'IPO jump shock' (nén lò xo) and enables 100% continuous,
academically rigorous systemic risk modeling across all 27 banks (2017 - 2025).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .manifold import Base12Manifold


from .domains.banking import DEFAULT_BANKING_TIERS


class MultiverseReconstructor:
    """
    Multiverse Counterfactual Synthesis Engine.
    """

    def __init__(
        self,
        n_universes: int = 3,
        boundary_window: int = 30,
        apoptosis_threshold: float = 2.5,
        reg_lambda: float = 0.5,
        tier_mapping: Optional[Dict[str, List[str]]] = None,
        volume_quantum: float = 1000.0,
    ):
        self.n_universes = n_universes
        self.boundary_window = boundary_window
        self.apoptosis_threshold = apoptosis_threshold
        self.reg_lambda = reg_lambda
        self.tier_mapping = tier_mapping or DEFAULT_BANKING_TIERS
        self.volume_quantum = volume_quantum
        self.manifold = Base12Manifold(volume_quantum=volume_quantum)

    def _get_tier_for_bank(self, bank: str, active_tiers: Dict[str, List[str]]) -> str:
        for tier_name, banks in active_tiers.items():
            if bank in banks:
                return tier_name
        return list(active_tiers.keys())[0] if active_tiers else "TIER2_COMMERCIAL"

    @staticmethod
    def _deterministic_seed(text: str) -> int:
        return sum((i + 1) * ord(c) * (31 ** (i % 7)) for i, c in enumerate(text)) % (2**31)

    def reconstruct(
        self,
        bank_panel: pd.DataFrame,
        market_index: Optional[pd.Series] = None,
        reg_lambda: Optional[float] = None,
        boundary_window: Optional[int] = None,
        apoptosis_threshold: Optional[float] = None,
        tier_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Synthesizes latent volatility for unlisted historical dates across all banks.

        Parameters:
        - bank_panel: (T, N) DataFrame of daily Parkinson log-volatility (2017 - 2025).
        - market_index: Optional Series of daily market index (e.g. VNINDEX).
        """
        active_lambda = self.reg_lambda if reg_lambda is None else reg_lambda
        active_window = self.boundary_window if boundary_window is None else boundary_window
        active_thresh = self.apoptosis_threshold if apoptosis_threshold is None else apoptosis_threshold
        active_tiers = self.tier_mapping if tier_mapping is None else tier_mapping

        df = bank_panel.copy()
        dates = df.index
        n_dates, n_banks = df.shape
        calendar_idx = np.arange(n_dates)

        # 1. Base-12 Manifold Harmonic Features (Quarterly & Annual cycles)
        manifold_features = self.manifold.batch_transform(calendar_idx, dates=dates)
        m_cos_1 = manifold_features[:, 1]
        m_sin_1 = manifold_features[:, 0]
        m_cos_2 = manifold_features[:, 3]
        m_sin_2 = manifold_features[:, 2]

        # 2. Market Baseline Factor (harmonized to annualized log-volatility scale)
        if market_index is not None:
            aligned_mkt = market_index.reindex(dates).ffill().bfill()
            if float(np.nanmean(aligned_mkt)) > 10.0:
                # Price index (e.g. VN-Index closing levels): compute annualized log-volatility percentage
                mkt_ret = np.log(np.maximum(aligned_mkt / aligned_mkt.shift(1).bfill(), 1e-8)).values
                mkt_vol = np.log(100.0 * np.sqrt(252.0 * np.maximum(mkt_ret ** 2, 1e-4)))
            else:
                mkt_vol = aligned_mkt.values
        else:
            # Fallback: cross-sectional average of core9 always-active banks
            core_banks = [c for c in ["VCB", "CTG", "BID", "ACB", "MBB"] if c in df.columns]
            mkt_vol = df[core_banks].mean(axis=1).values

        # 3. Process each bank individually
        reconstructed = df.copy()

        for bank in df.columns:
            series = df[bank].values
            first_idx = df[bank].first_valid_index()
            tier_name = self._get_tier_for_bank(bank, active_tiers)
            peer_banks = [
                b for b in active_tiers.get(tier_name, [])
                if b in df.columns and b != bank
            ]

            # Compute historical peer average (using available peers at each t)
            if len(peer_banks) > 0:
                peer_vol = df[peer_banks].mean(axis=1).bfill().ffill().fillna(0.0).values
            else:
                peer_vol = df.mean(axis=1).bfill().ffill().fillna(0.0).values

            system_vol = df.mean(axis=1).bfill().ffill().fillna(0.0).values

            # If bank has zero observed data across the entire panel, synthesize from peer tier
            if first_idx is None:
                rng = np.random.default_rng(seed=self._deterministic_seed(bank))
                y_synth_all = peer_vol + rng.normal(0, 0.2, n_dates)
                reconstructed[bank] = y_synth_all
                continue

            first_pos = int(df.index.get_loc(first_idx))

            # If bank was already listed from day 1, no synthesis needed
            if first_pos == 0:
                continue

            # Target bank was unlisted for dates: [0 : first_pos]
            # Training period: [first_pos : n_dates]

            # Feature matrix: [mkt_vol, peer_vol, system_vol, m_cos_1, m_sin_1]
            X = np.column_stack([
                mkt_vol,
                peer_vol,
                system_vol,
                m_cos_1,
                m_sin_1,
                np.ones(n_dates),
            ])
            np.nan_to_num(X, copy=False, nan=0.0)

            # Observed training set
            X_train = X[first_pos:]
            y_train = series[first_pos:]
            np.nan_to_num(y_train, copy=False, nan=0.0)

            # Regularized least squares (Ridge regression) to learn fundamental sensitivity
            reg_lambda_val = active_lambda
            eye = np.eye(X.shape[1])
            eye[-1, -1] = 0.0  # Do not regularize intercept
            beta = np.linalg.pinv(X_train.T @ X_train + reg_lambda_val * eye) @ (X_train.T @ y_train)

            # Calculate empirical residual standard deviation from observed period
            residuals_train = y_train - (X_train @ beta)
            sigma_noise = max(float(np.std(residuals_train)), 0.15)

            # 4. Generate Multiverse Predictions with calibrated stochastic noise
            # ponytail: deterministic invariant seed per bank for 100% cross-platform reproducibility
            rng = np.random.default_rng(seed=self._deterministic_seed(bank))
            X_unlisted = X[:first_pos]
            y_pred_base = X_unlisted @ beta

            # Universe 1: Core Fundamental Benchmark with real residual variance
            u1 = y_pred_base + rng.normal(0, sigma_noise, first_pos)

            # Universe 2: Liquidity Stress Regime (Correlated with active peers)
            u2 = 0.6 * peer_vol[:first_pos] + 0.4 * y_pred_base + rng.normal(0, sigma_noise * 1.1, first_pos)

            # Universe 3: Macroeconomic Market Anchor Regime
            u3 = y_pred_base + 0.25 * (mkt_vol[:first_pos] - np.mean(mkt_vol)) + rng.normal(0, sigma_noise * 0.9, first_pos)

            # Stack multiverse branches
            multiverse_matrix = np.column_stack([u1, u2, u3])

            # Meta-Ensemble with Apoptosis Pruning:
            # Excise branches that diverge beyond apoptosis_threshold std devs from median
            median_pred = np.median(multiverse_matrix, axis=1)
            residuals = np.abs(multiverse_matrix - median_pred[:, None])
            std_proxy = np.std(residuals) + 1e-6

            weights = np.where(residuals / std_proxy > active_thresh, 0.0, 1.0)
            weights_sum = weights.sum(axis=1, keepdims=True)
            weights_sum[weights_sum == 0] = 1.0
            norm_weights = weights / weights_sum

            y_synthetic = np.sum(multiverse_matrix * norm_weights, axis=1)

            # 5. Smooth Boundary Bridging (Hermite polynomial transition)
            # Use post-IPO equilibrium median (first 5 sessions) to avoid bridging into ±40% first-day IPO spikes
            post_ipo_eq = float(np.nanmedian(series[first_pos : first_pos + min(5, len(series) - first_pos)]))
            if np.isnan(post_ipo_eq):
                post_ipo_eq = float(series[first_pos])
            window = min(active_window, first_pos)

            if window > 0:
                bridge_synthetic = y_synthetic[-window:]
                # Smooth S-curve (cubic Hermite) transition weights from 0 to 1
                t_norm = np.linspace(0.0, 1.0, window)
                smooth_ramp = t_norm * t_norm * (3.0 - 2.0 * t_norm)
                delta = post_ipo_eq - bridge_synthetic[-1]
                y_synthetic[-window:] = bridge_synthetic + smooth_ramp * delta

            # Inject synthesized values into panel
            reconstructed.iloc[:first_pos, reconstructed.columns.get_loc(bank)] = y_synthetic

        # Intermittent post-IPO trading halts / holiday suspensions fallback
        reconstructed = reconstructed.ffill().bfill().fillna(0.0)
        return reconstructed
