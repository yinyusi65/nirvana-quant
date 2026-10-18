"""
NIRVANA CONSUMER ECONOMETRICS TEST RUNNER
=========================================
Empirical study on cyclical dynamics, inflation pass-through, and demand contagion
across 12 major Vietnamese consumer & retail enterprises (2019 - 2025).

Entities:
- Essential Staples: VNM, MSN, SAB, KDC, QNS, DBC, BAF
- Discretionary & Retail: MWG, PNJ, FRT, DGW, VRE
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nirvana import (
    NirvanaConnectedness,
    Base12Manifold,
    MultiverseReconstructor,
)
from nirvana.domains.consumer import (
    DEFAULT_CONSUMER_CATEGORIES,
    DEFAULT_EXPENDITURE_WEIGHTS,
    analyze_consumer_dynamics,
)


def main():
    print("==============================================================================")
    print("NIRVANA QUANT: CONSUMER & RETAIL CYCLICAL ECONOMETRICS (2019 - 2025)")
    print("Chrono-Cancer Meta-Architecture for High-Dimensional Demand & Price Contagion")
    print("==============================================================================\n")

    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'consumer_panel_2019_2025.csv')
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"[+] Loaded Consumer Panel: {df.shape[0]} trading days x {df.shape[1]} champions.")
    print(f"    - Staples:       VNM, MSN, SAB, KDC, QNS, DBC, BAF")
    print(f"    - Discretionary: MWG, PNJ, FRT, DGW, VRE\n")

    # 2. Cycle Detection via Discrete Fourier Transform on Base-12 Manifold
    print("[+] Step 1: Base-12 Torus Manifold Dominant Cycle Estimation...")
    sample_signal = df['MWG'].values
    p1, p2 = Base12Manifold.estimate_dominant_periods(sample_signal)
    print(f"    -> Detected Primary Seasonal Cycle:    ~{p1:.1f} trading sessions (Quarterly / Lunar Season)")
    print(f"    -> Detected Secondary Annual Cycle:    ~{p2:.1f} trading sessions (Annual Tết / Fiscal Year)\n")

    # 3. Fit Nirvana Causal Connectedness Engine
    print("[+] Step 2: Estimating Directional Lead-Lag Network (Market-Orthogonalized)...")
    engine = NirvanaConnectedness(
        lags=[1, 2, 3],
        lag_weights=[0.5, 0.3, 0.2],
        kalman_q=1e-3,
        kalman_r=1e-2,
        kalman_phi=0.98,
        kalman_outlier_threshold=3.0,
        apoptosis_threshold=0.015,
        orthogonalize_market=True,
        n_market_factors=1,
        asset_weights=DEFAULT_EXPENDITURE_WEIGHTS,
        asset_elasticity=0.5
    )

    report = engine.fit(df)
    print(report.summary())

    # 4. Domain Specific Consumer & Retail Insights
    print("\n[+] Step 3: Domain Semantic Synthesis (Staples vs. Discretionary)...")
    insights = analyze_consumer_dynamics(report)

    print(f"    - Dominant Macro Regime:             {insights['dominant_regime']}")
    print(f"    - Staples Mean Net Spillover:        {insights['staples_mean_net_spillover']:+.2f}%")
    print(f"    - Discretionary Mean Net Spillover:  {insights['discretionary_mean_net_spillover']:+.2f}%")
    print(f"    - Cost-Push / Pricing Power Leaders: {', '.join(insights['pricing_power_leaders'])}")
    print(f"    - Discretionary Demand Bellwethers:  {', '.join(insights['discretionary_bellwethers'])}")
    print(f"    - Top Demand Shock Absorbers:        {', '.join(insights['most_vulnerable_demand_absorbers'])}")
    print(f"    - Top Resilient Staples:             {', '.join(insights['top_resilient_staples'])}")

    # 5. Dynamic Rolling Analysis & Phase Transitions
    print("\n[+] Step 4: Dynamic Rolling Connectedness (Window=126, Step=10)...")
    rolling_rep = engine.rolling_fit(df, window=126, step=10, surge_std_threshold=2.0)
    print(rolling_rep.summary())

    print("\n==============================================================================")
    print("[✓] Consumer Econometrics Analysis Finished Successfully!")
    print("==============================================================================")


if __name__ == '__main__':
    main()
