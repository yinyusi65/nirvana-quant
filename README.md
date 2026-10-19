# NIRVANA: Chrono-Cancer Meta-Architecture for Cyclical Time-Series Analysis

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20%20Header--Only-purple.svg)](nirvana_cpp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NIRVANA** is an institutional-grade, mathematically rigorous framework designed for **high-dimensional cyclical time-series analysis**, directional lead-lag network inference, causal state-space filtering, and topological shock transmission across interconnected complex systems.

It combines state-space econometrics, non-linear machine learning with biological apoptosis regularization, and topological counterfactual synthesis to uncover unobserved systemic risk and latent financial flows.

---

## 🌟 Core Architecture & Engines

The framework is structured into five foundational mathematical engines:

1. **Dual-Clock Base-12 Manifold (`nirvana.manifold`)**:
   - Maps asynchronous temporal cycles and discrete transaction ticks onto a 2-Torus ($\mathbb{T}^2 = S^1 \times S^1$).
   - Calculates phase resonance coordinates $\gamma(t) = \cos(\theta_{\text{macro}} - \theta_{\text{micro}})$.
   - Dominant cycle detection via Discrete Fourier Transform (DFT) power spectrum analysis.

2. **Causal Kalman State-Space TVP Regressor (`nirvana.kalman`)**:
   - Strictly causal online state estimation with zero lookahead bias.
   - Joseph-form stabilized covariance updating: $P_t = (I - K_t H)P_{t|t-1}(I - K_t H)^T + K_t R K_t^T$.
   - Huber-robust innovation bounding resisting heavy-tailed anomalies and liquidity flash shocks.
   - Fully integrated as both a time-series filter and a time-varying parameter regressor (`CausalKalmanTVPRegressor`).

3. **12-Layer Tree Engine with Fisher Apoptosis (`nirvana.tree`)**:
   - Deep piecewise regression engine with maximum depth 12.
   - Empirical Fisher Information Matrix ($F_N$) and margin-density ($\rho_{\text{margin}}$) scoring.
   - Selective apoptosis pruning in deep layers ($d \ge 7$) to excise noise overfitting.

4. **Directional Connectedness & Lead-Lag Network (`nirvana.network`)**:
   - Multi-lag convolution kernels capturing delayed spillover dynamics across nodes.
   - Truncated SVD decomposition for stripping common pervasive systemic market factors.
   - Kleinberg HITS graph centrality: separates primary shock transmitters (Hub Centrality) from systemic shock absorbers (Authority Centrality).
   - Generalized Forecast Error Variance Decomposition (GFEVD Koop-Pesaran-Potter).

5. **Multiverse Counterfactual Synthesis & Inversion (`nirvana.multiverse`)**:
   - Reconstructs unobserved, pre-listing, or missing intervals across $M$ parallel stochastic universe paths.
   - Apoptosis pruning of divergent simulation branches via median-distance bounds.
   - Smooth cubic Hermite S-curve ($3t^2 - 2t^3$) boundary bridging to prevent artificial listing jump shocks.
   - Counterfactual topological inversion: maps market network spillovers to unobserved bilateral institutional balances.

6. **Phase 2 Booster & Split Conformal Prediction (`nirvana.static`, `nirvana.proof`)**:
   - Two-stage gradient-residual boosting with non-linear tree residuals.
   - Distribution-free Split Conformal Prediction Intervals guaranteeing exact finite-sample coverage:
     $$P\Big(Y_{\text{true}} \in [\hat{Y} - q, \hat{Y} + q]\Big) \ge 1 - \alpha$$
   - Multiverse invariance testing across 100 stochastic resampled universes.

---

## 📦 Domain Modules (`nirvana.domains`)

Nirvana isolates domain-specific semantics into clean, modular extensions:

* **`nirvana.domains.banking`**: Systemic importance classification (SIFI), exchange price-band bounds, asset scale anchors, and interbank placement/borrowing structural inversion.
* **`nirvana.domains.cybersecurity`**: Network attack propagation, botnet/DDoS command-and-control hub detection, sink authority identification, and traffic burst filtering.
* **`nirvana.domains.supply_chain`**: Multi-tier bullwhip effect propagation, critical bottleneck supplier discovery, and cascading disruption modeling.

---

## ⚡ C++20 Header-Only Engine (`nirvana_cpp/`)

For ultra-low latency, mission-critical systems, and embedded hardware, Nirvana provides an institutional-grade, zero-dependency C++20 implementation with full algorithmic parity:

* **Header-only**: Simply `#include <nirvana/nirvana.hpp>`.
* **Zero external dependencies**: Built-in contiguous matrix algebra, Gauss-Jordan inversion, Ridge pseudoinverse, and deflated truncated SVD (no external linear algebra libraries required).
* **Cross-platform**: Compatible with GCC 12+, Clang 15+, and MSVC 19.30+.

---

## 🚀 Installation

### Option 1: Direct from GitHub (Colab / Kaggle / Local)
```bash
pip install git+https://github.com/yinyusi65/nirvana-algorithm.git
```

### Option 2: Local Development Mode
```bash
git clone https://github.com/yinyusi65/nirvana-algorithm.git
cd nirvana-algorithm
pip install -e .
```

---

## 💻 Quickstart Examples

### 1. Directional Network Spillover Analysis
```python
import numpy as np
import pandas as pd
from nirvana import NirvanaConnectedness

# Multivariate cyclical series (T samples, N nodes)
T, N = 500, 5
data = pd.DataFrame(
    np.random.randn(T, N),
    columns=[f"Node_{i}" for i in range(N)]
)

engine = NirvanaConnectedness(
    lags=[1, 2],
    kalman_q=1e-3,
    kalman_r=1e-2,
    orthogonalize_market=True
)

report = engine.fit(data)
print(f"Total Connectedness Index (TCI): {report.total_connectedness_index:.2f}%")
print("Top Transmitters:", report.top_transmitters)
print("Top Receivers:", report.top_receivers)
```

### 2. Two-Stage Booster with Conformal Prediction Intervals
```python
import numpy as np
from nirvana import NirvanaTwoStageRegressor

X = np.random.randn(100, 4)
y = 2.0 * X[:, 0] - 1.5 * (X[:, 1] ** 2) + np.random.randn(100) * 0.2

model = NirvanaTwoStageRegressor(base_estimator="ols", max_depth=5)
model.fit(X, y)

# Predict point forecasts and 95% conformal confidence bounds
y_pred, y_lower, y_upper = model.predict_interval(X, alpha=0.05)
print(f"Predicted: {y_pred[0]:.2f}, 95% Bound: [{y_lower[0]:.2f}, {y_upper[0]:.2f}]")
```

---

## 📜 Authors & Citation
* **Lưu Văn Toàn** (`yinyusi65`)
