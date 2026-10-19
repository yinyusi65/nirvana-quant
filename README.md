# nirvana-quant: A Statistical and Econometric Framework for Multivariate Cyclical Time Series

[![PyPI Version](https://img.shields.io/pypi/v/nirvana-quant.svg?color=blue)](https://pypi.org/project/nirvana-quant/)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20%20Header--Only-purple.svg)](nirvana_cpp/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

`nirvana-quant` is an open-source Python and C++ research library for **multivariate time-series econometrics**, state-space filtering, directional network spillover modeling, and counterfactual panel reconstruction.

The toolkit provides statistical implementations combining classical time-series econometrics (Vector Autoregressions, Diebold–Yılmaz spillover decompositions, Kalman filtering) with modern non-parametric learning (piecewise regression trees, Fisher Information regularization, and finite-sample conformal prediction).

---

## Methodological Modules

The package provides six core mathematical modules:

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

## Domain Extensions (`nirvana.domains`)

Domain-specific conventions and structural parameters are maintained in modular extensions:

* **`banking`**: Empirical asset-scaling adjustments, exchange price-band limits, and interbank liquidity position mappings for systemic risk studies.
* **`cybersecurity`**: Directed network graph modeling for anomalous traffic flows and node centrality analysis.
* **`supply_chain`**: Multi-tier propagation models for inventory volatility and supplier dependency shocks.

---

## C++ Implementation (`nirvana_cpp/`)

For low-latency execution and integration into compiled pipelines, the repository includes a self-contained C++20 header library:

* Header-only: `#include <nirvana/nirvana.hpp>`.
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

### 1. Directional Spillover Analysis
```python
import numpy as np
import pandas as pd
from nirvana import NirvanaConnectedness

# Simulated daily returns across N entities
np.random.seed(42)
T, N = 500, 5
returns_df = pd.DataFrame(
    np.random.randn(T, N),
    columns=[f"Asset_{i}" for i in range(N)]
)

# Estimate directional connectedness network
model = NirvanaConnectedness(
    lags=[1, 2],
    kalman_q=1e-3,
    kalman_r=1e-2,
    orthogonalize_market=True
)
results = model.fit(returns_df)

print(f"Total Connectedness Index (TCI): {results.total_connectedness_index:.2f}%")
print("Top Shock Transmitters:", results.top_transmitters)
print("Top Shock Receivers:", results.top_receivers)
```

### 2. Two-Stage Regression with Conformal Prediction Intervals
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

## Authors & Maintainers
* **yinyusi65**

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
