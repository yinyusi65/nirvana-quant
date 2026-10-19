# nirvana-quant: A Statistical and Econometric Framework for Multivariate Cyclical Time Series

<p align="center">
  <a href="https://pypi.org/project/nirvana-quant/"><img src="https://img.shields.io/pypi/v/nirvana-quant.svg?color=007ec6&logo=pypi&logoColor=white" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/nirvana-quant/"><img src="https://img.shields.io/pypi/dm/nirvana-quant?color=blue&logo=python&logoColor=white" alt="Downloads"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg?logo=python&logoColor=white" alt="Python Versions"></a>
  <a href="nirvana_cpp/"><img src="https://img.shields.io/badge/C%2B%2B-20%20Header--Only-purple.svg?logo=cplusplus&logoColor=white" alt="C++20"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-green.svg" alt="License: GPL v3"></a>
  <a href="https://colab.research.google.com/github/yinyusi65/nirvana-algorithm/blob/main/test_nirvana_netbank_full_pipeline.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

---

`nirvana-quant` is an open-source Python and C++ research framework for **multivariate time-series econometrics**, state-space filtering, directional network spillover modeling, and counterfactual panel reconstruction.

The library bridges classical econometric foundations (Vector Autoregressions, Diebold–Yılmaz connectedness, Kalman filters) with modern non-parametric methods (piecewise regression trees, Fisher Information regularization, and finite-sample Split Conformal Prediction).

---

## 🚀 Interactive Notebook Demo (1-Click Run)

Test the complete end-to-end systemic risk pipeline across 27 commercial banks with interactive network graphs, SVD market decoupling, domino default contagion, and Basel IV macroprudential capital surcharge calibration:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yinyusi65/nirvana-algorithm/blob/main/test_nirvana_netbank_full_pipeline.ipynb)

---

## 📊 Visualizations & Topological Risk Analytics

The engine extracts systemic interbank connections from multivariate asset price series:

| Pairwise Spillover Intensity Heatmap (27x27) | Directed Systemic Risk Network Topology |
| :---: | :---: |
| <img src="banking_econometrics/nirvana_net_v2/output/banking_pairwise_spillover_heatmap.png" width="450"/> | <img src="banking_econometrics/nirvana_net_v2/output/banking_network_directed_topology.png" width="450"/> |

| Net Transmitter vs. Receiver Ranking | Hub vs. Authority Scatterplot |
| :---: | :---: |
| <img src="banking_econometrics/nirvana_net_v2/output/banking_net_spillover_ranking.png" width="450"/> | <img src="banking_econometrics/nirvana_net_v2/output/banking_hub_vs_authority_scatter.png" width="450"/> |

---

## ⚡ Quickstart in 3 Lines

```bash
pip install nirvana-quant
```

```python
import pandas as pd
from nirvana import NirvanaConnectedness

# Fit a TVP-VAR causal Kalman network model
model = NirvanaConnectedness(kalman_q=1e-3, kalman_r=1e-2, orthogonalize_market=True)
results = model.fit(df_asset_returns)

print(f"Total Connectedness Index (TCI): {results.total_connectedness_index:.2f}%")
print("Top Shock Transmitters:", results.top_transmitters[:5])
print("Top Shock Receivers:", results.top_receivers[:5])
```

---

## 🔬 Core Methodological Modules

### 1. Harmonic Feature Mapping (`nirvana.manifold`)
* Maps multi-frequency cyclical signals (such as quarterly financial reporting and annual cycles) onto periodic manifold coordinates ($\mathbb{T}^2 = S^1 \times S^1$).
* Incorporates Discrete Fourier Transform (DFT) power spectral estimation to identify dominant frequencies empirically.

### 2. Causal State-Space Filtering & TVP (`nirvana.kalman`)
* Implements sequential Kalman state-space estimation with strictly causal updates (zero lookahead bias).
* Utilizes the Joseph-stabilized covariance formulation:
  $$P_t = (I - K_t H) P_{t|t-1} (I - K_t H)^T + K_t R K_t^T$$
  to ensure positive semi-definiteness under ill-conditioned measurement environments.
* Employs Huber-type innovation bounds to limit sensitivity to fat-tailed shocks.

### 3. Regularized Piecewise Regression (`nirvana.tree`)
* Captures non-linear residual structures via bounded piecewise regression trees.
* Utilizes Fisher Information Matrix metrics and sample-margin density criteria to guide terminal-node pruning and mitigate finite-sample overfitting.

### 4. Directional Spillover & Network Topology (`nirvana.network`)
* Computes directional connectedness matrices via Vector Autoregressions (VAR) and Generalized Forecast Error Variance Decomposition (GFEVD, Koop–Pesaran–Potter).
* Employs Singular Value Decomposition (SVD) to isolate pervasive common market factors from idiosyncratic innovations.
* Evaluates Kleinberg graph centrality measures (Hub and Authority scores) to distinguish net transmitters from net receivers of shocks.

### 5. Counterfactual Panel Reconstruction (`nirvana.multiverse`)
* Implements counterfactual imputation for unbalanced historical panels (such as pre-listing asset periods).
* Evaluates sensitivity across structured simulation paths anchored on peer-group dynamics and macroeconomic benchmarks.
* Uses cubic Hermite transition polynomials ($3t^2 - 2t^3$) to ensure boundary continuity between reconstructed and observed sample periods.

### 6. Two-Stage Estimation & Conformal Prediction (`nirvana.static`, `nirvana.proof`)
* Combines parametric base estimators (OLS, Ridge, Lasso) with non-linear residual trees in a two-stage additive framework.
* Computes distribution-free prediction intervals via Split Conformal Prediction, satisfying finite-sample coverage guarantees:
  $$P\Big(Y_{n+1} \in [\hat{Y}_{n+1} - q, \hat{Y}_{n+1} + q]\Big) \ge 1 - \alpha$$

---

## 🏛️ Domain Extensions (`nirvana.domains`)

Domain-specific structural parameters are maintained in modular extensions:

* **`banking`**: Empirical asset-scaling adjustments, exchange price-band limits, and interbank liquidity position mappings for systemic risk studies.
* **`cybersecurity`**: Directed network graph modeling for anomalous traffic flows and node centrality analysis.
* **`supply_chain`**: Multi-tier propagation models for inventory volatility and supplier dependency shocks.

---

## 💻 C++ Implementation (`nirvana_cpp/`)

For low-latency execution and integration into compiled pipelines, the repository includes a self-contained C++20 header library:

* Header-only: `#include <nirvana/nirvana.hpp>`.
* Self-contained linear algebra routines (matrix decomposition, SVD, pseudoinverse) without mandatory external dependencies.
* Compatible with standard C++20 compilers (GCC 12+, Clang 15+, MSVC 19.30+).

---

## 📦 Installation Options

### From PyPI (Recommended)
```bash
pip install nirvana-quant
```

### From GitHub
```bash
pip install git+https://github.com/yinyusi65/nirvana-algorithm.git
```

### From Local Source
```bash
git clone https://github.com/yinyusi65/nirvana-algorithm.git
cd nirvana-algorithm
pip install -e .
```

---

## 📖 Usage Examples

### 1. Two-Stage Regression with Conformal Prediction Intervals
```python
import numpy as np
from nirvana import NirvanaTwoStageRegressor

# Cross-sectional sample
np.random.seed(42)
X = np.random.randn(120, 3)
y = 1.8 * X[:, 0] - 0.7 * (X[:, 1] ** 2) + np.random.randn(120) * 0.15

# Fit two-stage additive model
regressor = NirvanaTwoStageRegressor(base_estimator="ols", max_depth=4)
regressor.fit(X, y)

# Compute 95% conformal prediction intervals
y_pred, y_lower, y_upper = regressor.predict_interval(X, alpha=0.05)
print(f"Point Forecast: {y_pred[0]:.3f}, 95% Interval: [{y_lower[0]:.3f}, {y_upper[0]:.3f}]")
```

---

## 👤 Authors & Maintainers
* **yinyusi65**

## 📄 License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
