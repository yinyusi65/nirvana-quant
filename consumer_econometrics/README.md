# 🛒 NIRVANA CONSUMER ECONOMETRICS
> **High-Dimensional Cyclical Demand Dynamics, Seasonality Harmonics & Inflation Pass-Through**  
> *Module: `nirvana.domains.consumer` | Author: `yinyusi65`*

---

## 📌 1. Research Overview

This module extends the **Nirvana Chrono-Cancer Meta-Architecture** to the **Consumer Goods & Retail Sector**. Rather than modeling financial solvency, it investigates:

1. **Cost-Push vs. Demand-Pull Propagation**:
   - Do essential staples (`VNM`, `MSN`, `SAB`, `DBC`) lead and transmit input cost inflation downstream?
   - Or do discretionary retailers (`MWG`, `PNJ`, `FRT`, `DGW`, `VRE`) lead as bellwethers of consumer confidence and disposable income?
2. **Base-12 Seasonal Harmonics ($\mathbb{T}^2 = S^1 \times S^1$)**:
   - Captures sharp annual consumption spikes around the Lunar New Year (Tết) and quarterly cyclical surges.
3. **Idiosyncratic Brand Market Share Spillovers**:
   - Strips pervasive macro market shocks (VN-Index) via truncated SVD to reveal true inter-brand competitive dynamics.

---

## 🏛️ 2. Structure & Dataset

```
consumer_econometrics/
├── data/
│   └── consumer_panel_2019_2025.csv  # 1,826 trading sessions x 12 consumer champions
├── run_consumer_analysis.py           # End-to-end econometric pipeline
├── test_consumer.py                   # Automated unit test suite (100% pass)
└── README.md                          # Methodology and findings documentation
```

### Universe of 12 Vietnamese Consumer Champions:

| Ticker | Category | Economic Role & Sensitivity |
| :--- | :--- | :--- |
| **VNM** | Consumer Staples (Dairy) | Low price elasticity, defensive pricing power anchor |
| **MSN** | Consumer Staples (FMCG & WinMart) | Integrated consumer goods & modern retail network |
| **SAB** | Consumer Staples (Beverages) | Alcohol & beverage market, seasonal holiday sensitivity |
| **KDC** | Consumer Staples (Food & Oils) | Edible oils, ice cream, seasonal mooncake demand |
| **QNS** | Consumer Staples (Sugar & Soymilk)| Raw agricultural input costs & mass consumer dairy |
| **DBC** | Consumer Staples (Protein Agri) | 3F pork supply chain, feed costs & swine price cycle |
| **BAF** | Consumer Staples (Protein 3F) | Modern clean pork chain, volatile commodity input costs |
| **MWG** | Discretionary Retail (Electronics & Grocery) | Primary bellwether of mass consumer purchasing power |
| **PNJ** | Discretionary Retail (Jewelry & Luxury) | Wealth hedge, wedding/Valentine & Tết seasonal surge |
| **FRT** | Discretionary Retail (Pharmacy Long Châu) | Rapid retail pharmacy expansion & consumer tech |
| **DGW** | Discretionary Retail (ICT Distribution) | Technology distribution & supply chain importer |
| **VRE** | Discretionary Retail (Commercial Malls) | Retail floor leasing & foot traffic bellwether |

---

## ⚡ 3. Quickstart & Execution

```bash
# 1. Run full econometric analysis
python consumer_econometrics/run_consumer_analysis.py

# 2. Run unit tests
python consumer_econometrics/test_consumer.py
```

---

## 📊 4. Core Empirical Findings (2019 – 2025)

1. **Market Orthogonalization**:
   - The common macroeconomic market factor explains **24.51%** of total consumer volatility.
   - Idiosyncratic Connectedness ($TCI$) across the 12 brands is **40.12%** ($TCI_N = 43.77\%$).
2. **Pricing Power Transmitters (NET > 0)**:
   - `MSN` (+31.70%), `SAB` (+22.45%), `VNM` (+20.46%), and `MWG` (+18.75%) act as primary systemic transmitters. Essential staples lead the price discovery process due to cost-push pass-through.
3. **Cyclical Demand Shock Absorbers (NET < 0)**:
   - `BAF` (-21.79%), `DGW` (-17.56%), `KDC` (-14.56%), and `DBC` (-14.46%) act as systemic receivers, absorbing downstream demand contractions and raw material price shocks.
4. **Dominant Macro Regime**:
   - **Cost-Push Inflation (Staples Leading)**: Staples mean net spillover is $+1.66\%$ compared to $-1.65\%$ for discretionary retail, reflecting that input costs and food inflation drive the rhythm of consumer spending in Vietnam.

The above analysis focuses on dynamic cyclical time-series connectedness.
For static cross-sectional micro-econometrics (e.g. substitute goods and Phase 2 R^2 boosting), see `static_econometrics/`.


