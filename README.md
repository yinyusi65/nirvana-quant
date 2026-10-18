# NIRVANA: Chrono-Cancer Meta-Architecture for Cyclical Time-Series Analysis

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20%20Header--Only-purple.svg)](nirvana_cpp/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yinyusi65/nirvana-algorithm/blob/main/nirvana_bank_colab_demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NIRVANA** is a high-performance framework designed for **high-dimensional cyclical time-series analysis**, directional lead-lag network inference, causal state-space filtering, and topological shock transmission across interconnected complex systems.

---

## 🌟 Core Architecture & Features

The framework is built upon five foundational mathematical engines:

1. **Dual-Clock Base-12 Manifold (`nirvana.manifold`)**:
   - Maps asynchronous temporal cycles and discrete activity ticks onto a 2-Torus ($\mathbb{T}^2 = S^1 \times S^1$).
   - Calculates phase resonance coordinates $\gamma(t) = \cos(\theta_{\text{macro}} - \theta_{\text{micro}})$.
   - Includes automatic dominant cycle detection via Discrete Fourier Transform (DFT) power spectrum analysis.

2. **Multivariate Causal Kalman State-Space Filter (`nirvana.kalman`)**:
   - Strictly causal online state estimation with zero lookahead bias.
   - Joseph-stabilized covariance formulation $P = (I - KH)P_{\text{pred}}(I - KH)^T + KRK^T$.
   - Huber-robust innovation bounding to resist fat-tailed shocks, sensor glitches, and abnormal outlier spikes.
   - Dynamic partial observation updates for unbalanced time-series panels.

3. **12-Layer Tree Engine with Apoptosis Pruning (`nirvana.tree`)**:
   - Deep piecewise regression engine with max depth 12.
   - Empirical Fisher Information Matrix ($F_N$) and margin-density ($\rho_{\text{margin}}$) scoring.
   - Selective apoptosis pruning in deep layers ($d \ge 7$) to excise noise overfitting.

4. **Directional Connectedness & Lead-Lag Network (`nirvana.network`)**:
   - Multi-lag convolution kernels capturing delayed spillover dynamics across nodes.
   - Truncated SVD decomposition for stripping common pervasive macro/systemic factors.
   - Kleinberg HITS graph centrality: separates primary shock transmitters (Hub Centrality) from systemic shock absorbers (Authority Centrality).
   - Dynamic rolling window estimation with statistical surge/phase transition detection.

5. **Multiverse Counterfactual Synthesis (`nirvana.multiverse`)**:
   - Reconstructs unobserved, delayed, or missing intervals across $M$ parallel stochastic universe paths.
   - Apoptosis pruning of divergent simulation branches via median-distance bounds.
   - Cubic Hermite S-curve interpolation ($3t^2 - 2t^3$) for seamless boundary transitions without artificial jump shocks.

---

## 📦 Domain Modules (`nirvana.domains`)

Nirvana isolates domain-specific semantics into modular extensions while keeping the core mathematical algorithm universal:

* **`nirvana.domains.banking`**: Systemic importance classification (SIFI), price-band heteroskedasticity mapping, and sub-linear balance-sheet asset scaling.
* **`nirvana.domains.cybersecurity`**: Network attack propagation, botnet/DDoS command-and-control hub detection, sink authority identification, and traffic burst filtering.
* **`nirvana.domains.supply_chain`**: Multi-tier bullwhip effect propagation, critical bottleneck supplier discovery, and cascading disruption modeling.

---

## ⚡ C++20 Header-Only Engine (`nirvana_cpp/`)

For ultra-low latency, mission-critical systems, and embedded hardware, Nirvana provides an institutional-grade, zero-dependency C++20 implementation:

* **Header-only**: Simply `#include <nirvana/nirvana.hpp>`.
* **Zero external dependencies**: Built-in contiguous matrix algebra, Gauss-Jordan inversion, Ridge pseudoinverse, and deflated truncated SVD (no BLAS, LAPACK, or Eigen required).
* **Fully tested**: 204/204 automated unit test assertions passed.

```bash
# Compile and run test suite with GCC 12+
g++ -std=c++20 -O3 -I nirvana_cpp/include nirvana_cpp/tests/test_core.cpp -o test_core
./test_core
```

---

## 🚀 Installation

### Option 1: Direct from GitHub (Google Colab / Local)
```bash
!pip install git+https://github.com/yinyusi65/nirvana-algorithm.git
```

### Option 2: Local Editable Mode
```bash
git clone https://github.com/yinyusi65/nirvana-algorithm.git
cd nirvana-algorithm
pip install -e .
```

---

## 💻 Quickstart: Analyzing Cyclical Multivariate Signals

```python
import numpy as np
import pandas as pd
from nirvana import NirvanaConnectedness, Base12Manifold

# 1. Generate multi-channel cyclical time series (T samples, N nodes)
T, N = 252, 5
t = np.linspace(0, 10, T)
data = pd.DataFrame({
    f"Node_{i}": np.sin(t + i * 0.5) + np.random.normal(0, 0.1, T)
    for i in range(N)
})

# 2. Configure Nirvana Connectedness Engine
engine = NirvanaConnectedness(
    lags=[1, 2],
    lag_weights=[0.7, 0.3],
    kalman_q=1e-3,
    kalman_r=1e-2,
    apoptosis_threshold=0.015,
    orthogonalize_market=True,
    n_market_factors=1
)

# 3. Fit and generate directed lead-lag network report
report = engine.fit(data)
print(report.summary())

# 4. Access directional spillover metrics
print(f"Total Connectedness Index: {report.total_connectedness_index:.2f}%")
print("Top Transmitters:", report.top_transmitters)
print("Top Receivers:", report.top_receivers)
```

---

## 📜 Authors & Citation
- **yinyusi65**
