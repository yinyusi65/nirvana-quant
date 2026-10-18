"""
NIRVANA DOMAIN MODULE: CONSUMER & RETAIL ECONOMETRICS
======================================================
Translates Nirvana directed lead-lag networks and Base-12 harmonic manifolds
into consumer economics intelligence:
- Essential Consumer Staples vs. Discretionary / Luxury Goods
- Base-12 Lunar Calendar & Seasonal Consumption Spikes (Tết, Back-to-school)
- Cost-push Pass-through vs. Demand-pull Shock Propagation
- Consumer Sentiment & Purchasing Power Elasticity
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

DEFAULT_CONSUMER_CATEGORIES: Dict[str, str] = {
    # Consumer Staples (Hàng Tiêu dùng Thiết yếu - Co giãn cầu thấp)
    "VNM": "Staples_Dairy",
    "MSN": "Staples_Integrated_FMCG",
    "SAB": "Staples_Beverages",
    "KDC": "Staples_Food_Oils",
    "QNS": "Staples_Sugar_Soymilk",
    "DBC": "Staples_Protein_Agri",
    "BAF": "Staples_Protein_3F",
    
    # Consumer Discretionary & Retail (Hàng Tiêu dùng Không thiết yếu & Bán lẻ - Co giãn cầu cao)
    "MWG": "Discretionary_Retail_Electronics",
    "PNJ": "Discretionary_Luxury_Jewelry",
    "FRT": "Discretionary_Healthcare_Tech",
    "DGW": "Discretionary_Distribution_ICT",
    "VRE": "Discretionary_Retail_RealEstate",
    "PET": "Discretionary_Distribution_Petro",
}

DEFAULT_EXPENDITURE_WEIGHTS: Dict[str, float] = {
    "VNM": 85.0,  # Market Cap / Revenue scale proxy (Thousand Billion VND)
    "MSN": 110.0,
    "MWG": 70.0,
    "PNJ": 35.0,
    "SAB": 75.0,
    "VRE": 45.0,
    "FRT": 25.0,
    "DGW": 12.0,
    "KDC": 15.0,
    "QNS": 18.0,
    "DBC": 14.0,
    "BAF": 8.0,
}


def analyze_consumer_dynamics(
    report: Any,
    categories: Optional[Dict[str, str]] = None,
    expenditure_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Translates systemic risk report into consumer & retail econometric insights.

    Parameters:
    - report: SystemicRiskReport from NirvanaConnectedness.fit()
    - categories: Mapping of {ticker: category_name}
    - expenditure_weights: Scale proxy for market footprint

    Returns:
    - Structured dictionary with category-level spillovers, demand leaders, and resilience rankings.
    """
    cats = categories or DEFAULT_CONSUMER_CATEGORIES
    weights = expenditure_weights or DEFAULT_EXPENDITURE_WEIGHTS

    net_spillovers = report.net_spillovers
    hubs = report.eigenvector_hub
    authorities = report.eigenvector_authority

    # Aggregate by category: Staples vs Discretionary
    staples_net: List[float] = []
    discretionary_net: List[float] = []
    category_summary: Dict[str, List[float]] = {}

    for node, net_val in net_spillovers.items():
        cat = cats.get(node, "Other")
        category_summary.setdefault(cat, []).append(net_val)
        if "Staples" in cat:
            staples_net.append(net_val)
        elif "Discretionary" in cat:
            discretionary_net.append(net_val)

    mean_staples = float(np.mean(staples_net)) if staples_net else 0.0
    mean_discretionary = float(np.mean(discretionary_net)) if discretionary_net else 0.0

    # Categorize roles:
    # 1. Cost-Push / Pricing Power Transmitters (NET > 0 and high Hub)
    pricing_leaders = [
        node for node in report.top_transmitters
        if "Staples" in cats.get(node, "")
    ]

    # 2. Purchasing Power Bellwethers (high Hub in Discretionary)
    discretionary_bellwethers = [
        node for node in report.top_transmitters
        if "Discretionary" in cats.get(node, "")
    ]

    # 3. Demand Absorbers (NET < 0)
    demand_absorbers = report.top_receivers

    # 4. Resilience Ranking (Flight-to-Safety if present, else low authority)
    resilience_ranking: List[str] = []
    if hasattr(report, "flight_to_safety_index") and report.flight_to_safety_index:
        resilience_ranking = sorted(
            report.flight_to_safety_index.keys(),
            key=lambda k: report.flight_to_safety_index[k],
            reverse=True
        )
    else:
        resilience_ranking = sorted(
            authorities.keys(),
            key=lambda k: authorities[k]
        )

    return {
        "staples_mean_net_spillover": round(mean_staples, 2),
        "discretionary_mean_net_spillover": round(mean_discretionary, 2),
        "dominant_regime": (
            "Cost-Push Inflation (Staples Leading)"
            if mean_staples > mean_discretionary
            else "Discretionary Demand-Pull (Discretionary Leading)"
        ),
        "pricing_power_leaders": pricing_leaders,
        "discretionary_bellwethers": discretionary_bellwethers,
        "most_vulnerable_demand_absorbers": demand_absorbers[:5],
        "top_resilient_staples": [n for n in resilience_ranking if "Staples" in cats.get(n, "")][:5],
    }
