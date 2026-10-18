"""
NIRVANA: Chrono-Cancer Meta-Architecture for High-Dimensional Financial Risk & Network Dynamics
================================================================================================
A causal, manifold-based quantitative engine for financial time-series, volatility spillover,
and systemic risk analysis across interconnected banking systems.

Author: yinyusi65 & Nirvana Quant Team
License: MIT / Academic Research
"""

__version__ = "1.0.0"

from .manifold import Base12Manifold, DualClockCoordinates
from .kalman import CausalOnlineKalman, MultivariateCausalKalman
from .tree import NirvanaTreeEngine, TrainingSample
from .network import (
    NirvanaConnectedness,
    SystemicRiskReport,
    DynamicSystemicRiskReport,
    DEFAULT_ASSET_WEIGHTS,
    DEFAULT_EXCHANGE_BANDS,
)
from .multiplex import MultiplexNetworkEngine
from .multiverse import MultiverseReconstructor
from .volatility import compute_parkinson, compute_garman_klass, compute_yang_zhang
from .static import (
    NirvanaStaticBooster,
    StaticEconometricReport,
    NirvanaTreeRegressor,
    NirvanaTwoStageRegressor,
    parse_formula,
)
from .stata import stata_regress, StataRegressionResult, stata_summarize, stata_pwcorr
from .proof import (
    prove_phase2_variable,
    Phase2MultiverseProofReport,
    rank_phase2_discovered_variables,
    Phase2RankedVariablesReport,
    DiscoveredVariableItem,
)
from . import domains

__all__ = [
    "Base12Manifold",
    "DualClockCoordinates",
    "CausalOnlineKalman",
    "MultivariateCausalKalman",
    "NirvanaTreeEngine",
    "TrainingSample",
    "NirvanaStaticBooster",
    "StaticEconometricReport",
    "NirvanaTreeRegressor",
    "NirvanaTwoStageRegressor",
    "parse_formula",
    "stata_regress",
    "StataRegressionResult",
    "stata_summarize",
    "stata_pwcorr",
    "prove_phase2_variable",
    "Phase2MultiverseProofReport",
    "rank_phase2_discovered_variables",
    "Phase2RankedVariablesReport",
    "DiscoveredVariableItem",
    "NirvanaConnectedness",
    "SystemicRiskReport",
    "DynamicSystemicRiskReport",
    "DEFAULT_ASSET_WEIGHTS",
    "DEFAULT_EXCHANGE_BANDS",
    "MultiplexNetworkEngine",
    "MultiverseReconstructor",
    "compute_parkinson",
    "compute_garman_klass",
    "compute_yang_zhang",
    "domains",
]
