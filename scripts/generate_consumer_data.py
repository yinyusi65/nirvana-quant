import numpy as np
import pandas as pd
import os

# Create directories
os.makedirs('consumer_econometrics/data', exist_ok=True)

# Generate trading calendar from 2019-01-02 to 2025-12-31 (~1750 trading days)
dates = pd.bdate_range(start='2019-01-02', end='2025-12-31')
T = len(dates)

np.random.seed(42)

# Macro Drivers
# 1. Macro Inflation / CPI pressure cycle
t_norm = np.linspace(0, 7 * 2 * np.pi, T) # 7 years
# Annual Lunar / Tết seasonality (surge around Dec/Jan, index ~ month 12 or 1)
annual_cycle = np.sin(t_norm - np.pi / 2) # Peaks around year end
cpi_trend = 0.02 + 0.015 * np.sin(t_norm * 0.5) # Cyclical inflation

# 12 Consumer Champions
nodes = [
    'VNM', 'MSN', 'SAB', 'KDC', 'QNS', 'DBC', 'BAF', # Staples
    'MWG', 'PNJ', 'FRT', 'DGW', 'VRE'                # Discretionary & Retail
]

# Baseline volatility profiles and sensitivities
# Staples: lower beta to discretionary cycle, higher cost-push lead
# Discretionary: high beta to macro demand, seasonal Tết surges
data = {}

# Common market factor
mkt_shock = np.random.normal(0, 0.015, T)
# Inflation shock (cost push)
cost_shock = np.random.normal(0, 0.012, T)

for node in nodes:
    base_drift = 0.0002
    noise = np.random.normal(0, 0.018, T)
    
    if node in ['VNM', 'MSN', 'QNS', 'KDC']:
        # Stable staples: lower volatility, leads cost-push
        vol = (
            base_drift
            + 0.4 * mkt_shock
            + 0.5 * cost_shock
            + 0.15 * annual_cycle * 0.01
            + noise * 0.8
        )
    elif node in ['DBC', 'BAF']:
        # Agricultural protein: highly sensitive to pork price cycle & feed costs
        pork_cycle = np.sin(t_norm * 1.5)
        vol = (
            base_drift
            + 0.5 * mkt_shock
            + 0.7 * cost_shock
            + 0.3 * pork_cycle * 0.015
            + noise * 1.2
        )
    elif node == 'PNJ':
        # Gold / Luxury / Wealth hedge: strong Tết & Valentine/Wedding season (Q1/Q4)
        gold_cycle = np.sin(t_norm - np.pi / 3)
        vol = (
            base_drift
            + 0.6 * mkt_shock
            - 0.2 * cost_shock
            + 0.4 * gold_cycle * 0.015
            + noise * 1.1
        )
    elif node == 'FRT':
        # Pharmacy Long Châu boom post-2021
        growth_trend = np.linspace(-0.005, 0.008, T)
        vol = (
            base_drift + growth_trend
            + 0.5 * mkt_shock
            + noise * 1.3
        )
    else: # MWG, DGW, VRE
        # Discretionary retail: highly cyclical, consumer confidence sensitive
        vol = (
            base_drift
            + 0.8 * mkt_shock
            - 0.3 * cost_shock
            + 0.3 * annual_cycle * 0.02
            + noise * 1.2
        )
        
    # Covid shock simulation in March-April 2020 (days ~ 300 to 350)
    # Retail plunged, Staples surged relative
    if node in ['MWG', 'VRE', 'PNJ']:
        vol[300:340] -= np.random.uniform(0.01, 0.03, 40)
    elif node in ['VNM', 'MSN', 'DBC']:
        vol[300:340] += np.random.uniform(0.005, 0.015, 40)
        
    data[node] = np.round(vol, 6)

df = pd.DataFrame(data, index=dates)
df.index.name = 'Date'

output_path = 'consumer_econometrics/data/consumer_panel_2019_2025.csv'
df.to_csv(output_path)
print(f'Generated {output_path}: {df.shape[0]} sessions x {df.shape[1]} consumer champions.')
