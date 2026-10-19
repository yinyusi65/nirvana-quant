# Nirvana: A Statistical and Econometric Framework for Multivariate Cyclical Time Series

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20%20Header--Only-purple.svg)](nirvana_cpp/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Nirvana** is an open-source research toolkit for **multivariate time-series econometrics**, state-space filtering, directional network spillover modeling, and counterfactual panel reconstruction. 

The library bridges classical econometric methodologies (Diebold–Yılmaz variance decompositions, Kalman state-space filtering, regularized regression) with modern statistical learning tools (piecewise regression trees, information-theoretic regularization, and split conformal inference).

---

## Methodological Overview

The toolkit provides modular components for processing, modeling, and diagnosing high-dimensional time series:

### 1. Harmonic Feature Mapping (`nirvana.manifold`)
* Maps multi-frequency cyclical indicators (such as reporting periods and calendar cycles) to periodic coordinates using toroidal projections ($\mathbb{T}^2 = S^1 \times S^1$).
* Employs Discrete Fourier Transform (DFT) spectral estimation to identify dominant periodicity without requiring pre-specified frequencies.

### 2. Causal State-Space Filtering & Time-Varying Parameters (`nirvana.kalman`)
* Implements sequential Kalman filtering and online parameter estimation with strictly causal updates (no lookahead bias).
* Utilizes the Joseph stabilized covariance form:
  $$P_t = (I - K_t H) P_{t|t-1} (I - K_t H)^T + K_t R K_t^T$$
  to maintain positive semi-definiteness under ill-conditioned measurement environments.
* Incorporates Huber-type innovation bounds to limit sensitivity to outlier observations.

### 3. Regularized Piecewise Regression (`nirvana.tree`)
* Evaluates non-linear residual structures via bounded piecewise regression trees.
* Utilizes Fisher Information and sample-margin density metrics to guide terminal-node pruning and mitigate overfitting on finite samples.

### 4. Directional Spillover & Network Topology (`nirvana.network`)
* Estimates directional connectedness based on Vector Autoregressions (VAR) and Generalized Forecast Error Variance Decomposition (GFEVD, Koop–Pesaran–Potter).
* Employs Singular Value Decomposition (SVD) to isolate common systematic market variance from idiosyncratic node innovations.
* Evaluates directed graph centrality measures, including Kleinberg's Hub and Authority metrics, to distinguish net transmitters from net receivers of shocks.

### 5. Counterfactual Panel Reconstruction (`nirvana.multiverse`)
* Implements counterfactual imputation for unbalanced historical panels (e.g., pre-listing periods in asset cross-sections).
* Evaluates sensitivity across structured simulation paths anchored by peer-group dynamics and macroeconomic benchmarks.
* Uses cubic Hermite transition polynomials to ensure continuous boundaries between reconstructed and observed sample periods.

### 6. Two-Stage Estimation & Conformal Prediction (`nirvana.static`, `nirvana.proof`)
* Integrates parametric base estimators (OLS, Ridge, Lasso) with non-linear residual trees in a two-stage additive framework.
* Generates distribution-free prediction intervals via Split Conformal Prediction, satisfying finite-sample coverage guarantees:
  $$P\Big(Y_{n+1} \in [\hat{Y}_{n+1} - q, \hat{Y}_{n+1} + q]\Big) \ge 1 - \alpha$$

---

## Domain Implementations (`nirvana.domains`)

Nirvana separates computational algorithms from domain-specific data structures:

* **`banking`**: Empirical asset-scaling adjustments, trading band volatility constraints, and interbank liquidity position mappings for Basel systemic risk studies.
* **`cybersecurity`**: Directed network graph modeling for anomalous traffic flows, command-and-control propagation, and denial-of-service node centrality.
* **`supply_chain`**: Multi-tier propagation models for inventory volatility and supplier dependency shocks.

---

## C++ Implementation (`nirvana_cpp/`)

For environments requiring minimal runtime overhead, Nirvana includes a self-contained C++20 header library:

* Header-only structure: `#include <nirvana/nirvana.hpp>`.
* Self-contained linear algebra routines (matrix decomposition, SVD, pseudoinverse) without mandatory external dependencies.
* Compatible with standard C++20 compilers (GCC 12+, Clang 15+, MSVC 19.30+).

---

## Installation

### From PyPI
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

## Usage Examples

### Example 1: Directional Spillover Analysis
```python
import numpy as np
import pandas as pd
from nirvana import NirvanaConnectedness

# Example: T daily observations across N entities
np.random.seed(42)
T, N = 500, 5
returns_df = pd.DataFrame(
    np.random.randn(T, N),
    columns=[f"Asset_{i}" for i in range(N)]
)

# Initialize and estimate connectedness model
model = NirvanaConnectedness(
    lags=[1, 2],
    kalman_q=1e-3,
    kalman_r=1e-2,
    orthogonalize_market=True
)
results = model.fit(returns_df)

print(f"Total Connectedness Index: {results.total_connectedness_index:.2f}%")
print("Top Transmitters:", results.top_transmitters)
print("Top Receivers:", results.top_receivers)
```

### Example 2: Two-Stage Regression with Conformal Prediction Bounds
```python
import numpy as np
from nirvana import NirvanaTwoStageRegressor

# Generate non-linear synthetic cross-section
np.random.seed(42)
X = np.random.randn(120, 3)
y = 1.8 * X[:, 0] - 0.7 * (X[:, 1] ** 2) + np.random.randn(120) * 0.15

# Fit model with OLS baseline and tree-based residual capture
regressor = NirvanaTwoStageRegressor(base_estimator="ols", max_depth=4)
regressor.fit(X, y)

# Compute 95% conformal prediction intervals
y_pred, y_lower, y_upper = regressor.predict_interval(X, alpha=0.05)
print(f"Point Forecast: {y_pred[0]:.3f}, 95% Interval: [{y_lower[0]:.3f}, {y_upper[0]:.3f}]")
```

---

## Authors & Maintainers
* **yinyusi65**

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
