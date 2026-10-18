"""
Multiplex Network Fusion Engine
================================
Couples Slow Fundamental Layer (Quarterly Financial Reports: CAMEL/NPL/CAR)
with Fast High-Frequency Layer (Daily Stock Return / Volatility).

Solves the multi-timescale mismatch in banking systemic risk modeling.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from .network import DEFAULT_ASSET_WEIGHTS


class MultiplexNetworkEngine:
    """
    Two-layer multiplex network for financial systems:
    - Layer 1 (Fundamental Macro/Balance-sheet): Low-frequency (Quarterly)
    - Layer 2 (Market Microstructure/Price): High-frequency (Daily)
    """

    def __init__(self, coupling_strength: float = 0.35):
        self.coupling_strength = float(np.clip(coupling_strength, 0.0, 1.0))

    def fuse_layers(
        self,
        market_connectivity: pd.DataFrame,
        balance_sheet_weights: Optional[Union[Dict[str, float], bool, str]] = None,
        interbank_exposures: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Fuses the market risk connectedness matrix with fundamental balance-sheet constraints.

        Parameters:
        - market_connectivity: (N, N) DataFrame of market spillovers.
        - balance_sheet_weights: Dict of {Bank: systemic_weight} based on Total Assets, NPL, or CAR.
        - interbank_exposures: (N, N) matrix of bilateral interbank deposits/lendings if available.
        """
        fused = market_connectivity.copy()
        excluded = {"FROM_others", "TO_others", "NET_spillover"}
        banks = [c for c in fused.columns if c in fused.index and c not in excluded]
        fused_core = fused.loc[banks, banks].values.astype(float)

        if balance_sheet_weights is True or balance_sheet_weights == "default":
            balance_sheet_weights = DEFAULT_ASSET_WEIGHTS

        # 1. Fundamental asset size / fragility modulation
        if balance_sheet_weights:
            weights = np.array([balance_sheet_weights.get(b, 1.0) for b in banks], dtype=float)
            mean_w = np.mean(weights) if np.mean(weights) > 0 else 1.0
            weights = weights / mean_w
            # Modulate outgoing risk by sender bank's fundamental vulnerability (Column j = Transmitter)
            fused_core = fused_core * weights[None, :]

        # 2. Interbank bilateral direct exposure network injection
        if interbank_exposures is not None:
            ib_core = interbank_exposures.reindex(index=banks, columns=banks, fill_value=0.0).values.astype(float)
            ib_row_sums = np.sum(ib_core, axis=1, keepdims=True)
            has_ib = (ib_row_sums > 0)
            ib_norm = np.where(has_ib, (ib_core / (ib_row_sums + 1e-8)) * 100.0, fused_core)
            fused_core = np.where(
                has_ib,
                (1.0 - self.coupling_strength) * fused_core + self.coupling_strength * ib_norm,
                fused_core,
            )

        # Re-normalize rows to 100%
        row_sums = fused_core.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        fused_core = (fused_core / row_sums) * 100.0

        res_df = pd.DataFrame(np.round(fused_core, 2), index=banks, columns=banks)
        return res_df
