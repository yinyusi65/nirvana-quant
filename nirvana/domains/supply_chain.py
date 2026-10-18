"""
Nirvana Domain Module: Supply Chain & Industrial Contagion
===========================================================
Applies Nirvana causal graph connectedness to:
- Bullwhip effect propagation across multi-tier supplier networks
- Bottleneck supplier identification (Component shortage risk)
- Lead-time volatility cascading across industrial sectors
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def analyze_supply_chain_risk(
    spillover_report,
    tier_levels: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Translates Nirvana connectedness metrics into supply chain risk indicators:
    - Shock Originator (Upstream Bullwhip Emitter): High net transmitter
    - Vulnerable Assembler (Downstream Receiver): High authority receiver
    - Strategic Bottleneck: High betweenness bridging critical tiers
    """
    nodes = list(spillover_report.net_spillovers.keys())
    tiers = tier_levels or {}
    
    records = []
    for node in nodes:
        net = spillover_report.net_spillovers.get(node, 0.0)
        hub = (spillover_report.eigenvector_hub or {}).get(node, 0.0)
        auth = (spillover_report.eigenvector_authority or {}).get(node, 0.0)
        bc = (spillover_report.betweenness_centrality or {}).get(node, 0.0)
        
        tier = tiers.get(node, 1)
        role = "REGULAR_SUPPLIER"
        if hub >= 5.0 and net > 0:
            role = "BULLWHIP_ORIGINATOR (UPSTREAM RISK)"
        elif auth >= 5.0 and net < 0:
            role = "HIGH_VULNERABILITY_ASSEMBLER"
        elif bc >= 10.0:
            role = "SOLE_SOURCE_BOTTLENECK"
            
        records.append({
            "supplier_or_tier": node,
            "tier_level": tier,
            "net_spillover": round(net, 2),
            "hub_emitter_score": round(hub, 2),
            "vulnerability_sink_score": round(auth, 2),
            "bottleneck_centrality": round(bc, 2),
            "strategic_role": role,
            "buffer_stock_recommendation": "INCREASE_SAFETY_STOCK" if "BULLWHIP" in role or "BOTTLENECK" in role else "STANDARD_JIT",
        })
        
    df = pd.DataFrame(records)
    return df.sort_values(by="bottleneck_centrality", ascending=False).reset_index(drop=True)
