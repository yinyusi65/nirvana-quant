"""
Generate Cross-Sectional Micro-Econometric Dataset: Consumer Substitution Basket
================================================================================
Simulates 2,500 Vietnamese households making protein substitution decisions
between Pork (Thit heo 3F) and Chicken (Thit ga cong nghiep) based on price shocks,
budget constraints, and household demographics.

Author: yinyusi65
"""

import numpy as np
import pandas as pd
import os

os.makedirs("static_econometrics/data", exist_ok=True)
np.random.seed(42)

N = 2500

# 1. Household Demographics & Budget
household_income = np.random.lognormal(mean=2.8, sigma=0.45, size=N) * 1.0  # Million VND (range ~ 8M to 60M)
household_income = np.clip(np.round(household_income, 1), 7.5, 75.0)

household_size = np.random.choice([1, 2, 3, 4, 5, 6], size=N, p=[0.08, 0.22, 0.32, 0.24, 0.10, 0.04])
urban_flag = np.random.choice([0, 1], size=N, p=[0.38, 0.62])

# 2. Local Market Commodity Prices (k VND / kg or dozen)
# In times of pork supply shocks (e.g. ASF epidemic), pork price spikes
price_pork = np.random.uniform(115.0, 195.0, size=N)  # 115k to 195k / kg
price_chicken = np.random.uniform(62.0, 95.0, size=N)  # 62k to 95k / kg
price_eggs = np.random.uniform(25.0, 42.0, size=N)     # 25k to 42k / dozen

promo_chicken = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], size=N, p=[0.50, 0.20, 0.15, 0.10, 0.05])
effective_price_chicken = price_chicken * (1.0 - promo_chicken)
price_ratio = price_pork / effective_price_chicken

# 3. Microeconomic Behavioral Utility & Non-linear Threshold Substitution
# Baseline linear component (Weak linear effect)
baseline_demand = 0.8 + 0.35 * household_size - 0.015 * effective_price_chicken + 0.012 * household_income

# Non-linear Threshold 1: Acute Budget Constraint Binding
# When pork is > 1.8x chicken price and household income is < 22M, mass substitution occurs
acute_substitution = np.where((price_ratio >= 1.82) & (household_income <= 22.0), 3.4, 0.0)

# Non-linear Threshold 2: Moderate Substitution for Middle-Income Families
moderate_substitution = np.where((price_ratio >= 1.65) & (household_income > 22.0) & (household_income <= 35.0), 1.8, 0.0)

# Non-linear Threshold 3: Promotional Surge in Urban Centers
promo_kick = np.where((promo_chicken >= 0.10) & (urban_flag == 1), 1.2, 0.0)

# Affluent immunity: Households with income > 45M have virtually zero price sensitivity to pork
affluent_suppression = np.where(household_income > 45.0, -1.1, 0.0)

# Idiosyncratic taste shock
taste_noise = np.random.normal(0, 0.45, size=N)

# Weekly Household Chicken Consumption (Kg / week)
weekly_chicken_kg = (
    baseline_demand
    + acute_substitution
    + moderate_substitution
    + promo_kick
    + affluent_suppression
    + taste_noise
)
weekly_chicken_kg = np.clip(np.round(weekly_chicken_kg, 2), 0.5, 12.0)

df = pd.DataFrame({
    "Household_Income": household_income,
    "Household_Size": household_size,
    "Urban_Flag": urban_flag,
    "Price_Pork_3F": np.round(price_pork, 1),
    "Price_Chicken": np.round(price_chicken, 1),
    "Price_Eggs": np.round(price_eggs, 1),
    "Promo_Discount_Chicken": promo_chicken,
    "Price_Ratio_Pork_Chicken": np.round(price_ratio, 2),
    "Weekly_Chicken_Demand_Kg": weekly_chicken_kg,
})

output_path = "static_econometrics/data/static_substitute_goods_panel.csv"
df.to_csv(output_path, index=False)
print(f"Generated {output_path}: {df.shape[0]} household observations x {df.shape[1]} micro features.")
