# Nirvana C++20 Header-Only Library

[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20%20Header--Only-purple.svg)](include/nirvana/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](../LICENSE)

`nirvana_cpp` is a self-contained, header-only C++20 implementation of the **Nirvana** econometric and statistical time-series framework. It provides identical mathematical routines to the Python engine while optimizing for execution speed, zero external dependencies, and minimal memory overhead.

---

## Architecture

The C++ library contains self-contained header modules organized under `include/nirvana/`:

```
nirvana_cpp/
├── include/nirvana/
│   ├── nirvana.hpp      # Primary aggregate header
│   ├── types.hpp        # Contiguous matrix algebra, SVD, and pseudoinverse routines
│   ├── manifold.hpp     # Toroidal feature projections and spectral frequency estimation
│   ├── kalman.hpp       # Causal state-space Kalman filter and TVP regression
│   ├── tree.hpp         # Piecewise regression trees with Fisher Information pruning
│   ├── network.hpp      # Directional connectedness, GFEVD, and Kleinberg graph centrality
│   ├── multiverse.hpp   # Counterfactual path synthesis and Hermite boundary interpolation
│   └── static.hpp       # Two-stage additive regression and conformal prediction intervals
└── CMakeLists.txt       # CMake interface library target
```

---

## Mathematical Components

### 1. Matrix Algebra & Decompositions (`types.hpp`)
* Implements cache-friendly contiguous dynamic arrays (`Matrix`, `Vector`).
* Zero external linear algebra dependencies: includes Gauss–Jordan inversion, Ridge regularized pseudoinverse, and deflated truncated Singular Value Decomposition (SVD).

### 2. Harmonic Manifold Mapping (`manifold.hpp`)
* Projects temporal and discrete activity indicators onto a 2-Torus $\mathbb{T}^2 = S^1 \times S^1$:
  $$\theta_{\text{macro}} = \frac{2\pi \cdot (t \pmod P)}{P}, \quad \theta_{\text{micro}} = \frac{2\pi \cdot (\lfloor V / V_q \rfloor \pmod{12})}{12}$$
  $$\gamma(t) = \cos(\theta_{\text{macro}} - \theta_{\text{micro}})$$
* Discrete Fourier Transform (DFT) spectral estimator identifies periodicities without prior frequency assumptions.

### 3. Causal Kalman Filtering & TVP (`kalman.hpp`)
* Formulates time-varying parameter updates with strictly causal state estimation:
  $$P_t = (I - K_t H) P_{t|t-1} (I - K_t H)^T + K_t R K_t^T$$
* Huber-clipped innovation updates prevent covariance divergence under fat-tailed or high-volatility shocks:
  $$\tilde{y}_{k, \text{clipped}} = \text{clamp}\left(\frac{\tilde{y}_k}{\sigma_S}, -\tau, \tau\right) \cdot \sigma_S$$

### 4. Regularized Piecewise Regression Trees (`tree.hpp`)
* Evaluates non-linear residual structures via bounded piecewise regression trees.
* Utilizes Fisher Information Matrix metrics and sample-margin density criteria to guide terminal-node pruning and mitigate finite-sample overfitting.

### 5. Directional Network Spillover (`network.hpp`)
* Estimates directional connectedness based on Generalized Forecast Error Variance Decomposition (GFEVD).
* Truncated SVD orthogonalization separates common systematic market drivers from idiosyncratic innovations.
* Evaluates Kleinberg graph centrality (Hub and Authority scores) to identify primary transmitters and absorbers of shocks.

### 6. Counterfactual Panel Reconstruction (`multiverse.hpp`)
* Implements counterfactual imputation for unbalanced historical panels using multi-path stochastic simulations.
* Cubic Hermite interpolation ($3t^2 - 2t^3$) ensures continuous boundaries between imputed and observed data.

### 7. Two-Stage Estimation & Conformal Inference (`static.hpp`)
* Additive two-stage framework: combines parametric base estimators with non-linear residual trees.
* Computes finite-sample distribution-free Split Conformal Prediction Intervals:
  $$P\Big(Y_{n+1} \in [\hat{Y}_{n+1} - q, \hat{Y}_{n+1} + q]\Big) \ge 1 - \alpha$$

---

## Integration

### Single-Command Compilation
Include the `include/` directory when compiling with any C++20-compliant compiler:

```bash
# GCC / Clang
g++ -std=c++20 -O3 -I include your_program.cpp -o your_program

# MSVC
cl /std:c++20 /O2 /I include your_program.cpp
```

### Modern CMake Target
Add the library as a subdirectory in your project's `CMakeLists.txt`:

```cmake
add_subdirectory(nirvana_cpp)
target_link_libraries(your_target PRIVATE nirvana)
```

---

## Usage Example

```cpp
#include <nirvana/nirvana.hpp>
#include <iostream>

int main() {
    using namespace nirvana;

    // 1. Initialize Directional Connectedness Engine
    NirvanaConnectedness engine;
    engine.lags_ = {1, 2};
    engine.orthogonalize_market_ = true;
    engine.n_market_factors_ = 1;
    engine.kalman_outlier_threshold_ = 3.0; // Huber bounds

    // 2. Prepare returns matrix (T time periods, N entities)
    Matrix returns(252, 4);
    // ... populate matrix ...

    // 3. Compute directional spillover metrics
    SystemicRiskReport report = engine.fit(returns, {"Asset_A", "Asset_B", "Asset_C", "Asset_D"});
    report.print_summary();

    return 0;
}
```

---

## Authors & Maintainers
* **yinyusi65**

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](../LICENSE) file for details.
