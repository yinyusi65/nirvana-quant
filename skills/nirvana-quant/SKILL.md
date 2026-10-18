---
name: nirvana-quant
description: Deep tuning and execution guide for Nirvana Quant (Chrono-Cancer Meta-Architecture). Configures causal Kalman filtering, Base-12 manifold harmonics, multi-lag directional connectedness, Fisher apoptosis pruning, SVD market orthogonalization, multiplex banking fusion, and multiverse counterfactual synthesis across custom financial datasets.
metadata:
  icon: monitoring
---

# Nirvana Quant: Deep Tuning & Execution Skill Specification

`nirvana` is an academic-grade financial econometrics and systemic risk library. It replaces classical TVP-VAR / Diebold-Yılmaz GFEVD models with **causal state-space filtering**, **Base-12 manifold harmonics**, **Fisher margin-density apoptosis**, and **counterfactual multiverse simulation**.

This skill documents all configurable hyperparameters and provides ready-to-run recipes across diverse market scenarios.

---

## 1. Architecture & Design Principles

All Nirvana estimators follow the **scikit-learn / statsmodels** API convention:
* Hyperparameters can be set **at initialization (`__init__`)** or overridden **at runtime (`fit()` / `reconstruct()`)**.
* Intelligent defaults are provided so the model runs out of the box with `engine.fit(df)`.
* Every parameter has clear mathematical meaning rooted in financial physics and market microstructure.

```text
[ Raw Volatility Panel (T x N) ]
               │
               ▼
[ 0. Price-Band Heteroskedasticity Mapping ] ──> normalize_bands, exchange_bands
               │
               ▼
[ 1. Causal Online Kalman State-Space ] ───────> kalman_q, kalman_r, kalman_phi
               │
               ▼
[ 2. SVD Market Orthogonalization ] ───────────> orthogonalize_market
               │
               ▼
[ 3. Base-12 Torus Manifold Harmonics ] ───────> volume_quantum, dates
               │
               ▼
[ 4. Multi-Lag Directional Asymmetry ] ────────> lags, lag_weights, multi_lag
               │
               ▼
[ 5. Bootstrap Edge Significance ] ────────────> n_boot, p_value_threshold
               │
               ▼
[ 6. Empirical Fisher Apoptosis Pruning ] ─────> apoptosis_threshold
               │
               ▼
[ 7. Sub-linear Asset-Weighting Footprint ] ───> asset_weights, asset_elasticity
               │
               ▼
[ Systemic Risk Spillover Matrix & Network Topology ]
```

---

## 2. Complete Hyperparameter Reference

### 2.1 `NirvanaConnectedness`

Estimates directional systemic risk spillovers, pairwise linkages, Total Connectedness Index (TCI), and complex network centralities.

| Parameter | Type | Default | Description & Financial Rationale |
| :--- | :--- | :--- | :--- |
| `lags` | `int` or `List[int]` | `[1, 2, 3]` | Lag orders to evaluate. If an integer $p$ is provided (e.g. `lags=2`), it expands automatically to `[1, 2]`. Set to `[1]` for pure next-day impact, or `[1, 2, 3]` for T+2.5 settlement flow. |
| `lag_weights` | `List[float]` | `[0.5, 0.3, 0.2]` | Relative weights for each lag in the lead-lag convolution. Normalized to sum to 1.0. Decays exponentially or linearly across settlement days. |
| `multi_lag` | `bool` | `True` | Quick toggle. When `True` and `lags=None`, uses `[1, 2, 3]`. When `False`, restricts to `[1]`. |
| `kalman_q` | `float` | `1e-3` | Process noise variance $Q = q \cdot I$. Controls state evolution flexibility. Smaller values enforce smoother structural trajectories; larger values react faster to high-frequency shocks. |
| `kalman_r` | `float` | `1e-2` | Baseline measurement noise variance $R = r \cdot I$. Calibrates observation noise. In markets with wide price bands, per-node $R_i$ scales automatically with $(band / 0.07)^2$. |
| `kalman_phi` | `float` | `0.98` | State persistence (AR(1) mean-reversion coefficient). $\phi \in (0, 1)$. Default 0.98 maintains high persistence typical of financial volatility without unit-root drift. |
| `apoptosis_threshold` | `float` | `0.015` | Baseline edge pruning cutoff ($1.5\%$). Links weaker than this threshold are considered noise/cancerous and pruned to zero. Set to `0.0` to preserve dense graph. |
| `n_boot` | `int` | `0` | Number of circular-roll bootstrap iterations for edge statistical significance testing. When `n_boot > 0` (e.g. `200` or `500`), computes empirical p-values. |
| `kalman_outlier_threshold` | `float` | `None` | Huber-robust bounding cutoff for innovations (e.g. `3.5` or `4.0` standard deviations). Protects covariance tracking against extreme fat-tail jumps, sensor glitches, or flash crashes. |
| `orthogonalize_market` | `bool` | `False` | When `True`, extracts latent common factors via SVD and regresses them out of innovations, revealing idiosyncratic node-to-node contagion. |
| `n_market_factors` | `int` | `1` | Number of orthogonal common factors $k$ to strip via SVD when `orthogonalize_market=True`. Set $k > 1$ for multi-cluster systems. |
| `causality_mode` | `str` | `'linear'` | Directional causality calculation mode: `'linear'` (standard lagged cross-correlation) or `'threshold'` (tail/panic-activated transmission). |
| `threshold_quantile` | `float` | `0.70` | Upper quantile cutoff for threshold causality mode. Shocks are only transmitted when source activity exceeds this quantile. |
| `macro_periods` | `Tuple[float, float]` | `(63.0, 252.0)` | Primary and secondary macro cycles for Base-12 manifold. Custom values (e.g. `(24.0, 168.0)` for hourly/weekly network traffic) adapt the manifold to non-financial domains. |
| `normalize_bands` | `bool` | `True` | Adjusts observation noise $R_i$ according to legal exchange price limits. Eliminates artificial 4.6x variance distortion between UPCoM (15%), HNX (10%), and HOSE (7%). |
| `exchange_bands` | `Dict[str, float]` | `DEFAULT_EXCHANGE_BANDS` | Custom dictionary mapping ticker -> price limit (e.g. `{"VCB": 0.07, "NVB": 0.10, "BVB": 0.15}`). Set to `{}` or uniform values for unconstrained international markets. |
| `asset_weights` | `Dict[str, float]` or `bool` or `str` | `None` | Total asset size (e.g. in Trillion VND or Billion USD). Outgoing risk from bank $j$ is scaled by its asset footprint. Pass `True` or `"default"` for built-in 27 Vietnamese bank assets. |
| `asset_elasticity` | `float` | `0.5` | Sub-linear scaling exponent $w_i = \text{Assets}_i^\alpha$. Default $\alpha = 0.5$ (square root) balances systemic footprint without allowing trillion-VND state giants to completely obliterate regional bank signals. |
| `volume_quantum` | `float` | `1000.0` | Quantum threshold for Base-12 information time mapping. |

---

### 2.2 `MultiverseReconstructor`

Synthesizes latent unobserved market volatility for pre-IPO periods (e.g. 2017–2020) using counterfactual multi-universe simulation loops.

| Parameter | Type | Default | Description & Financial Rationale |
| :--- | :--- | :--- | :--- |
| `n_universes` | `int` | `3` | Number of parallel stochastic universe simulation loops (Universe 1: Fundamental Benchmark, Universe 2: Liquidity Stress Regime, Universe 3: Macroeconomic Market Anchor). |
| `boundary_window` | `int` | `30` | Number of trading sessions before IPO date over which a cubic Hermite smooth S-curve ramp transitions synthetic volatility into post-IPO equilibrium, eliminating IPO jump shock. |
| `apoptosis_threshold` | `float` | `2.5` | Cutoff in units of standard deviations from median across multiverse branches. Outlier simulation paths exceeding this threshold are pruned before meta-ensemble blending. |
| `reg_lambda` | `float` | `0.5` | Ridge regression $L_2$ shrinkage parameter for fundamental factor sensitivity estimation. |
| `tier_mapping` | `Dict[str, List[str]]` | `DEFAULT_BANKING_TIERS` | Structural peer groupings. Default groups Vietnamese banks into `BIG4_STATE`, `TIER1_RETAIL_CORP`, `TIER2_COMMERCIAL`, and `TIER3_REGIONAL_RESTRUCTURING`. Easily customizable for other sectors. |
| `volume_quantum` | `float` | `1000.0` | Base-12 manifold quantum. |

---

### 2.3 `MultiplexNetworkEngine`

Couples fast high-frequency market volatility with slow low-frequency balance-sheet fundamentals or interbank bilateral exposures.

| Parameter | Type | Default | Description & Financial Rationale |
| :--- | :--- | :--- | :--- |
| `coupling_strength` | `float` | `0.35` | Multiplex inter-layer coupling weight $\omega \in [0, 1]$. When bilateral interbank exposure data is available, blends market risk with credit exposure: $(1-\omega) \cdot S_{\text{mkt}} + \omega \cdot S_{\text{interbank}}$. |
| `balance_sheet_weights` | `Dict[str, float]` | `"default"` | Weight vector (Assets, NPL ratio, or 1/CAR) modulating outgoing transmitter strength. |
| `interbank_exposures` | `pd.DataFrame` | `None` | Bilateral matrix of interbank lending / deposit exposures. |

---

## 3. Practical Scenarios & Code Recipes

### Scenario A: Standard Vietnamese Banking Systemic Risk (Academic Default)
* Setup: 27 Banks, T+2.5 settlement (lags 1, 2, 3), HOSE/HNX/UPCoM price band normalization, asset-weighted, with bootstrap significance testing ($p < 0.05$).

```python
import pandas as pd
from nirvana import NirvanaConnectedness

# Load panel of daily log-volatilities
df = pd.read_csv("data/all27_panel_data_floor_2021_2025.csv", index_col=0, parse_dates=True)

# Initialize with academic parameters
engine = NirvanaConnectedness(
    lags=[1, 2, 3],
    lag_weights=[0.5, 0.3, 0.2],
    kalman_q=1e-3,
    kalman_r=1e-2,
    normalize_bands=True,
    asset_weights="default",
    asset_elasticity=0.5,
    n_boot=200,
    p_value_threshold=0.05,
    apoptosis_threshold=0.015,
)

report = engine.fit(df)
print(report.summary())
print(f"Total Connectedness Index: {report.total_connectedness_index:.2f}%")
print(f"Normalized TCI (adjusted for N=27): {report.normalized_tci:.2f}%")
```

---

### Scenario B: Idiosyncratic Bank Contagion (Purging Market Sockets via SVD)
* Setup: Bóc tách tác động của VN-Index để đo lường lan truyền rủi ro thực chất giữa các ngân hàng.

```python
engine = NirvanaConnectedness(
    lags=2,                          # Auto-expands to [1, 2]
    orthogonalize_market=True,       # Strip market common factor
    apoptosis_threshold=0.02,        # Prune weak links < 2%
)

report = engine.fit(df)
print(f"Market Variance Explained: {report.market_variance_share:.2f}%")
print("Top Idiosyncratic Transmitters:", report.top_transmitters[:5])
print("Top Shock Absorbers:", report.top_receivers[:5])
```

---

### Scenario C: International Equities / Unconstrained Markets (e.g. US S&P 500 Banks)
* Setup: US financial institutions (JPM, BAC, C, WFC, GS, MS) with T+1 settlement and no statutory price bands.

```python
us_banks_df = pd.read_csv("us_banks_volatility.csv", index_col=0, parse_dates=True)

us_assets = {
    "JPM": 3875.0, "BAC": 3050.0, "C": 2375.0, 
    "WFC": 1880.0, "GS": 1590.0, "MS": 1190.0
}

engine = NirvanaConnectedness(
    lags=[1],                        # T+1 US settlement
    normalize_bands=False,           # No exchange price bands
    asset_weights=us_assets,
    asset_elasticity=0.5,
    kalman_phi=0.95,
)

report = engine.fit(us_banks_df)
print(report.summary())
```

---

### Scenario D: Custom Sector Multiverse Synthesis (e.g. Insurance or Real Estate)
* Setup: Synthesizing unlisted history for non-bank sectors using custom peer groups.

```python
from nirvana import MultiverseReconstructor

# Define custom sector tiers
energy_tiers = {
    "UPSTREAM": ["GAS", "PVD", "PVS"],
    "MIDSTREAM_REFINERY": ["BSR", "PLX", "OIL"],
    "DOWNSTREAM_DISTRIB": ["CNG", "PVG", "PGS"]
}

reconstructor = MultiverseReconstructor(
    n_universes=5,                   # 5 simulation loops
    boundary_window=20,              # 20-session Hermite ramp
    apoptosis_threshold=2.0,         # Stricter pruning of simulation divergence
    reg_lambda=0.8,                  # Higher regularization for thin data
    tier_mapping=energy_tiers
)

reconstructed_panel = reconstructor.reconstruct(energy_panel_df)
```

---

### Scenario E: Multiplex Stress-Testing (Credit Exposures + Market Spillover)
* Setup: Fusion of market volatility spillovers with bilateral interbank credit exposure matrix.

```python
from nirvana import MultiplexNetworkEngine

# report.spillover_table is (N x N) market risk matrix
# interbank_df is (N x N) bilateral credit matrix from annual reports
multiplex = MultiplexNetworkEngine(coupling_strength=0.40)

fused_matrix = multiplex.fuse_layers(
    market_connectivity=report.spillover_table,
    balance_sheet_weights="default",
    interbank_exposures=interbank_df
)
print("Multiplex Fused Risk Topology:")
print(fused_matrix)
```

---

## 4. Troubleshooting & Best Practices

1. **If TCI is unexpectedly high (> 70%):**
   - Check if `orthogonalize_market=True`. During systemic panics, market-wide co-movement inflates TCI. Stripping the market factor reveals true idiosyncratic contagion.
2. **If graph is too dense / noisy:**
   - Increase `apoptosis_threshold` from `0.015` to `0.025` or `0.030`.
   - Enable bootstrap pruning: set `n_boot=200` and `p_value_threshold=0.05`.
3. **If UPCoM/HNX small banks dominate transmitters artificially:**
   - Ensure `normalize_bands=True` is enabled so wider price-band noise is properly calibrated into the Kalman filter.
4. **Short historical sample ($T < 100$):**
   - Set `lags=[1]` to avoid parameter overfitting.
   - Set `kalman_q=1e-4` to stabilize state estimation.
