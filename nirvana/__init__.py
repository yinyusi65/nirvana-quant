"""
NIRVANA: Chrono-Cancer Meta-Architecture for High-Dimensional Financial Risk & Network Dynamics
================================================================================================
A causal, manifold-based quantitative engine for financial time-series, volatility spillover,
and systemic risk analysis across interconnected banking systems.

Author: Luu Van Toan (Đại Ca) & Antigravity Quant Team
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
from . import domains

__all__ = [
    "Base12Manifold",
    "DualClockCoordinates",
    "CausalOnlineKalman",
    "MultivariateCausalKalman",
    "NirvanaTreeEngine",
    "TrainingSample",
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
