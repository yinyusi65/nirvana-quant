# ☸ NIRVANA QUANT C++20 ENGINE
> **Chrono-Cancer Meta-Architecture for High-Dimensional Lead-Lag Dynamical Systems**  
> *Zero-dependency, Header-Only, Institutional-Grade C++20 Quantitative Engine.*

---

## 🏛️ 1. Architecture Overview

`nirvana_cpp` is a high-performance C++20 reimplementation of the Nirvana Quant framework. Designed for ultra-low latency, mission-critical algorithmic stability, and cross-platform portability without external dependencies (no BLAS, LAPACK, Eigen, or Boost required).

```
nirvana_cpp/
├── include/nirvana/
│   ├── nirvana.hpp      # Umbrella master header
│   ├── types.hpp        # Contiguous Matrix, Ridge Pinv, Truncated SVD, Linear Algebra
│   ├── manifold.hpp     # Dual-Clock Base-12 Torus Manifold (S¹ x S¹) & DFT Period Estimator
│   ├── kalman.hpp       # Multivariate Causal Kalman Filter with Huber Innovation Clipping
│   ├── tree.hpp         # 12-layer Tree with Empirical Fisher Information & Apoptosis Pruning
│   ├── network.hpp      # Directional Connectedness, SVD Multi-Factor Beta, HITS Hub/Authority
│   └── multiverse.hpp   # Multiverse Ensemble Reconstructor & Cubic Hermite Boundary Ramp
├── tests/
│   └── test_core.cpp    # 204/204 automated functional tests
└── CMakeLists.txt       # Modern CMake build configuration
```

---

## ⚡ 2. Core Mathematical Pillars

### A. Base-12 Manifold Harmonics (`manifold.hpp`)
Maps asynchronous calendar ticks and discrete volume/information quantities onto a 2-Torus $\mathbb{T}^2 = S^1 \times S^1$:
$$\theta_{\text{macro}} = \frac{2\pi \cdot (t \pmod P)}{P}, \quad \theta_{\text{micro}} = \frac{2\pi \cdot (\lfloor V / V_q \rfloor \pmod{12})}{12}$$
$$\gamma(t) = \cos(\theta_{\text{macro}} - \theta_{\text{micro}})$$

### B. Multivariate Causal Kalman Filter with Huber Bounding (`kalman.hpp`)
Prevents explosive covariance degradation during fat-tailed market shocks via Joseph stabilized covariance updates and Huber-clipped innovations:
$$\tilde{y}_k = z_k - H \hat{x}_{k|k-1}, \quad \tilde{y}_{k, \text{clipped}} = \text{clamp}\left(\frac{\tilde{y}_k}{\sigma_S}, -\tau, \tau\right) \cdot \sigma_S$$
$$P_k = (I - K H) P_{k|k-1} (I - K H)^T + K R K^T$$

### C. Phase 2 Empirical Fisher Information Apoptosis (`tree.hpp`)
Excises overparameterized leaf nodes in deep layers ($d \ge 7$) when the local Fisher noise ratio spikes:
$$s_{\text{cancer}} = \frac{F_N}{\rho_{\text{margin}} \sqrt{N + 1}} \cdot (1 - \text{OOF\_Consistency})$$

### D. Multi-Factor SVD Market Orthogonalization & Directed Network (`network.hpp`)
Strips $K$ principal market components via deflated truncated SVD before computing directed lead-lag spillovers:
$$Y_{\text{idiosyncratic}} = Y - \sum_{k=1}^K s_k u_k v_k^T$$
Calculates total spillover index ($TCI$), dimension-normalized $TCI_N = TCI \times \frac{N}{N-1}$, HITS Hubs (shock transmitters), and Authorities (systemic absorbers).

### E. Multiverse Counterfactual Synthesis (`multiverse.hpp`)
Reconstructs unobserved pre-listing or missing historical series across $M$ parallel universe trajectories, pruning outliers via median-distance apoptosis and smoothing boundary jumps via cubic Hermite interpolation:
$$H(t) = 3t^2 - 2t^3$$

---

## 🚀 3. Quick Start

### Direct Single-Command Compilation (GCC / Clang)

```bash
# Compile and run test suite (204 assertions)
g++ -std=c++20 -O3 -I include tests/test_core.cpp -o test_core
./test_core
```

### Modern CMake Integration

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
ctest --output-on-failure
```

---

## 💻 4. Minimal Code Snippet

```cpp
#include <nirvana/nirvana.hpp>
#include <iostream>

int main() {
    using namespace nirvana;

    // 1. Initialize Engine with SVD Multi-Factor Orthogonalization
    NirvanaConnectedness engine;
    engine.lags_ = {1, 2};
    engine.orthogonalize_market_ = true;
    engine.n_market_factors_ = 1;
    engine.kalman_outlier_threshold_ = 3.0; // Huber robust fat-tail bounds

    // 2. Load or generate returns matrix (T samples, N nodes)
    Matrix returns(252, 4); // [T=252, N=4]
    // ... fill returns ...

    // 3. Compute Directed Lead-Lag Network Report
    SystemicRiskReport report = engine.fit(returns, {"Node_A", "Node_B", "Node_C", "Node_D"});
    report.print_summary();

    return 0;
}
```
