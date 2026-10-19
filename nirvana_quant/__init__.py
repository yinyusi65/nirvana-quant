"""
NIRVANA QUANT: Chrono-Cancer Meta-Architecture
==============================================
Alias package enabling direct `import nirvana_quant` as well as `import nirvana`.
"""
import nirvana as _nirvana
from nirvana import *
from nirvana import __version__

# Map modules so `import nirvana_quant.manifold` works seamlessly
import nirvana.manifold as manifold
import nirvana.kalman as kalman
import nirvana.tree as tree
import nirvana.network as network
import nirvana.multiplex as multiplex
import nirvana.multiverse as multiverse
import nirvana.volatility as volatility
import nirvana.static as static
import nirvana.stata as stata
import nirvana.proof as proof

__all__ = _nirvana.__all__ + [
    'manifold', 'kalman', 'tree', 'network', 'multiplex', 
    'multiverse', 'volatility', 'static', 'stata', 'proof'
]
