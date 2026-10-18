# NIRVANA QUANT: Chrono-Cancer Meta-Architecture for Financial Systemic Risk

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20%20Header--Only-purple.svg)](nirvana_cpp/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yinyusi65/nirvana-algorithm/blob/main/nirvana_bank_colab_demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NIRVANA** is a high-dimensional financial econometrics and systemic risk package designed to analyze interconnected commercial banking networks, replacing and outperforming classical **Diebold-Yılmaz (2012, 2014) TVP-VAR GFEVD** models.

---

## 🌟 Core Theoretical Pillars

1. **Dual-Clock Base-12 Manifold ($\mathbb{T}^2 = S^1 \times S^1$)**:
   - Maps macro financial reporting frequencies (12 months / 4 quarters BCTC) and micro-information transaction flow into harmonic resonance coordinates $\gamma(t) = \cos(\theta_{\text{macro}} - \theta_{\text{micro}})$.
2. **Causal Online Kalman State-Space Filtering**:
   - Strips idiosyncratic microstructure noise with strictly causal updates (zero lookahead bias).
3. **Phase 2 Empirical Fisher Information & Margin-Density Apoptosis**:
   - Excises spurious noise-correlated edges without arbitrary lag length selection or parameter over-fitting.
4. **Multiplex Banking Network Fusion**:
   - Seamlessly integrates slow quarterly balance-sheet fundamentals (CAMEL, NPL, CAR) with fast daily stock volatility spillover.

---

## 🚀 Installation

### Option 1: Direct from GitHub on Google Colab / Local
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

## ⚡ Quickstart: Banking Systemic Risk Analysis (9 Core Banks)

```python
import pandas as pd
from nirvana import NirvanaConnectedness, MultiplexNetworkEngine

# 1. Load Bank Volatility / Floor Return Panel Data
url = "https://raw.githubusercontent.com/<user>/netbank/main/02_data_processed/core9_panel_data_floor_2017_2025.csv"
df = pd.read_csv(url, index_col=0, parse_dates=True)

# 2. Fit Nirvana Causal Connectedness Engine
engine = NirvanaConnectedness()
report = engine.fit(df)

# 3. Print Academic Spillover Matrix
print(report.summary())

# 4. Access Connectedness Table (Pandas DataFrame)
print("\nTotal Connectedness Index (TCI):", report.total_connectedness_index, "%")
display(report.spillover_table)
```

### Deep-Tuning Hyperparameters & Custom Scenarios

Nirvana estimators follow the standard `scikit-learn` / `statsmodels` conventions. All hyperparameters can be set at initialization or runtime:

```python
engine = NirvanaConnectedness(
    lags=[1, 2, 3],                  # Settlement lag structure (int or list of ints)
    lag_weights=[0.5, 0.3, 0.2],     # Convolution decay weights across lags
    kalman_q=1e-3,                   # Process noise covariance
    kalman_r=1e-2,                   # Measurement noise covariance
    kalman_phi=0.98,                 # Mean-reversion persistence AR(1)
    apoptosis_threshold=0.015,       # Baseline noise pruning cutoff (1.5%)
    n_boot=200,                      # Circular-roll bootstrap iterations
    p_value_threshold=0.05,          # Statistical significance threshold (p < 0.05)
    orthogonalize_market=True,       # Strip VN-Index / market common factor via SVD
    normalize_bands=True,            # Price-band heteroskedasticity mapping (HOSE/HNX/UPCoM)
    asset_weights="default",         # Scale outgoing risk by bank total assets
    asset_elasticity=0.5             # Sub-linear asset scaling exponent
)
```

See [skills/nirvana-quant/SKILL.md](skills/nirvana-quant/SKILL.md) for the complete parameter reference, mathematical rationales, and copy-paste recipes for banking, international equities, and custom sectors.

---

## 📊 Academic Comparison: Nirvana vs. Diebold-Yılmaz (DY)

| Metric | Classical Diebold-Yılmaz (2012) | Nirvana Chrono-Architecture |
| :--- | :--- | :--- |
| **Model Class** | Vector Autoregression (VAR) | State-Space Manifold + Causal Kernel |
| **Lag & Horizon Sensitivity** | Highly sensitive to $p$ and $H$ | Invariant (Continuous harmonic tracking) |
| **Microstructure Noise** | Distorts variance decomposition | Filtered via Online Causal Kalman |
| **Multi-timescale Fusion** | Requires temporal aggregation | Dual-Clock Multiplex Network ($\mathbb{T}^2$) |
| **Small-sample Overfitting** | Degrees of freedom crisis ($N^2 p$) | Phase 2 Apoptosis Pruning |

---

## 📜 Authors & Citation
- **yinyusi65**
