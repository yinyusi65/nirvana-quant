"""
Nirvana Domain Module: Banking & Financial Systems
===================================================
Domain-specific parameters, regulatory bands, asset weights, and structural tiers
for Vietnamese commercial banks and Basel IV systemic risk analysis.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


# Structural Banking Tiers in the Vietnamese Banking System
DEFAULT_BANKING_TIERS: Dict[str, List[str]] = {
    "BIG4_STATE": ["VCB", "CTG", "BID"],
    "TIER1_RETAIL_CORP": ["TCB", "VPB", "MBB", "ACB", "HDB", "VIB", "TPB", "SHB", "STB"],
    "TIER2_COMMERCIAL": ["MSB", "OCB", "LPB", "EIB", "SSB", "NAB"],
    "TIER3_REGIONAL_RESTRUCTURING": ["BAB", "BVB", "KLB", "NVB", "PGB", "SGB", "VAB", "VBB"],
}

# Statutory Exchange Price Limits (heteroskedasticity bounds)
DEFAULT_EXCHANGE_BANDS: Dict[str, float] = {
    # HNX (10% limit)
    "NVB": 0.10, "BAB": 0.10,
    # UPCoM (15% limit)
    "BVB": 0.15, "SGB": 0.15, "PGB": 0.15, "VBB": 0.15,
    "KLB": 0.15, "ABB": 0.15, "NAB": 0.15, "VAB": 0.15,
    # All others default to HOSE (7% limit)
}

# Empirical 2024 Total Assets (Trillion VND) for all 27 Vietnamese commercial banks
DEFAULT_ASSET_WEIGHTS: Dict[str, float] = {
    # Big4 State Banks (~1,800 - 2,300 Trillion VND)
    "BID": 2300.0, "CTG": 2000.0, "VCB": 1800.0,
    # Tier-1 Joint-Stock Retail & Corporate (~600 - 900 Trillion VND)
    "MBB": 900.0, "TCB": 850.0, "VPB": 820.0, "ACB": 720.0,
    "STB": 670.0, "SHB": 630.0, "HDB": 600.0,
    # Tier-2 Commercial (~200 - 450 Trillion VND)
    "VIB": 410.0, "LPB": 380.0, "TPB": 360.0, "MSB": 270.0,
    "SSB": 270.0, "OCB": 240.0, "EIB": 200.0, "NAB": 200.0,
    # Tier-3 Regional & Restructuring (~35 - 160 Trillion VND)
    "ABB": 160.0, "VAB": 110.0, "BAB": 100.0, "NVB": 90.0,
    "KLB": 85.0, "BVB": 85.0, "PGB": 55.0, "SGB": 35.0, "VBB": 35.0,
}


def classify_sifi_status(
    net_spillovers: Dict[str, float],
    hub_scores: Optional[Dict[str, float]] = None,
    asset_weights: Optional[Dict[str, float]] = None,
    sifi_net_cutoff: float = 2.0,
) -> pd.DataFrame:
    """
    Classifies banks into Systemically Important Financial Institutions (SIFIs)
    based on Net Transmitted Risk (NET > 0) and Hub Centrality.
    """
    banks = list(net_spillovers.keys())
    assets = asset_weights or DEFAULT_ASSET_WEIGHTS
    
    records = []
    for b in banks:
        net = net_spillovers.get(b, 0.0)
        hub = (hub_scores or {}).get(b, 0.0)
        asset = assets.get(b, 100.0)
        
        is_sifi = (net >= sifi_net_cutoff) or (hub >= 5.0 and asset >= 500.0)
        records.append({
            "bank": b,
            "net_spillover": round(net, 2),
            "hub_score": round(hub, 2),
            "asset_trillion_vnd": asset,
            "sifi_status": "SIFI_CRITICAL" if is_sifi else "NON_SIFI",
            "recommended_capital_buffer": "1.50%" if is_sifi and asset > 1500 else ("1.00%" if is_sifi else "0.00%"),
        })
        
    df = pd.DataFrame(records)
    return df.sort_values(by="net_spillover", ascending=False).reset_index(drop=True)
