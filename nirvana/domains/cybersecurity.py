"""
Nirvana Domain Module: Cybersecurity & Network Infrastructure Contagion
========================================================================
Applies Nirvana causal graph connectedness to:
- Autonomous System (AS) / BGP route flapping and traffic spillover
- DDoS amplification hub detection & botnet synchronization tracking
- Subnet bottleneck / bridge identification (Betweenness centrality)
- Vulnerability sink / target server identification (Authority centrality)
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def analyze_cyber_contagion(
    spillover_report,
    ip_or_node_metadata: Optional[Dict[str, Dict]] = None,
    ddos_hub_cutoff: float = 4.0,
) -> pd.DataFrame:
    """
    Translates a Nirvana systemic risk report into cybersecurity topology metrics:
    - Attack / Amplification Hub: High out-degree shock transmitter
    - Critical Target / Vulnerable Sink: High in-degree authority receiver
    - Chokepoint / Bridge Node: High betweenness centrality router
    """
    nodes = list(spillover_report.net_spillovers.keys())
    meta = ip_or_node_metadata or {}
    
    records = []
    for node in nodes:
        net = spillover_report.net_spillovers.get(node, 0.0)
        hub = (spillover_report.eigenvector_hub or {}).get(node, 0.0)
        auth = (spillover_report.eigenvector_authority or {}).get(node, 0.0)
        bc = (spillover_report.betweenness_centrality or {}).get(node, 0.0)
        
        node_type = "UNKNOWN"
        if hub >= ddos_hub_cutoff and net > 0:
            node_type = "AMPLIFICATION_SOURCE / BOTNET_HUB"
        elif auth >= ddos_hub_cutoff and net < 0:
            node_type = "CRITICAL_TARGET / AT_RISK_SINK"
        elif bc >= 10.0:
            node_type = "INFRASTRUCTURE_CHOKEPOINT"
        else:
            node_type = "NORMAL_TRANSIT"
            
        records.append({
            "node": node,
            "net_flow_shock": round(net, 2),
            "hub_score": round(hub, 2),
            "sink_authority_score": round(auth, 2),
            "betweenness_bottleneck": round(bc, 2),
            "cyber_classification": node_type,
            "action_priority": "URGENT_ISOLATE" if "AMPLIFICATION" in node_type or "CRITICAL_TARGET" in node_type else "MONITOR",
        })
        
    df = pd.DataFrame(records)
    return df.sort_values(by="hub_score", ascending=False).reset_index(drop=True)
