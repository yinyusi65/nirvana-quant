"""
NIRVANA DOMAIN-SPECIFIC SPECIALIZED EXTENSION MODULES
======================================================
Domain modules for translating universal Nirvana causal graph dynamics into
sector-specific actionable intelligence:
- banking: Basel IV, SIFI, bank capital buffers, HOSE/HNX/UPCoM limits
- cybersecurity: DDoS amplification, botnet hubs, BGP chokepoints
- supply_chain: Bullwhip effect, sole-source bottleneck suppliers
"""

from .banking import (
    DEFAULT_BANKING_TIERS,
    DEFAULT_EXCHANGE_BANDS,
    DEFAULT_ASSET_WEIGHTS,
    classify_sifi_status,
)
from .cybersecurity import analyze_cyber_contagion
from .supply_chain import analyze_supply_chain_risk
from .consumer import (
    DEFAULT_CONSUMER_CATEGORIES,
    DEFAULT_EXPENDITURE_WEIGHTS,
    analyze_consumer_dynamics,
)
from .substitute import (
    DEFAULT_PROTEIN_SUBSTITUTE_ROLES,
    analyze_substitution_thresholds,
)

__all__ = [
    "DEFAULT_BANKING_TIERS",
    "DEFAULT_EXCHANGE_BANDS",
    "DEFAULT_ASSET_WEIGHTS",
    "classify_sifi_status",
    "analyze_cyber_contagion",
    "analyze_supply_chain_risk",
    "DEFAULT_CONSUMER_CATEGORIES",
    "DEFAULT_EXPENDITURE_WEIGHTS",
    "analyze_consumer_dynamics",
    "DEFAULT_PROTEIN_SUBSTITUTE_ROLES",
    "analyze_substitution_thresholds",
]
