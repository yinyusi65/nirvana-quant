"""
NIRVANA DISCOVERED VARIABLES RANKING & REAL-WORLD TRANSLATION RUNNER
=====================================================================
Ranks top 5 variables discovered by Nirvana Phase 2, proves their econometric
significance and multiverse robustness, and translates each into a concrete
real-world concept with specific survey questions and data collection protocols.

Dataset: 2,500 Vietnamese Household Panel (Pork <-> Chicken Substitution)
Author: yinyusi65
"""

import sys
import os
import pandas as pd

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nirvana import (
    NirvanaTwoStageRegressor,
    parse_formula,
)


def run_ranking():
    # 1. Load micro panel dataset
    data_path = os.path.join(os.path.dirname(__file__), "data", "static_substitute_goods_panel.csv")
    df = pd.read_csv(data_path)

    # 2. Formula specification
    formula = (
        "Weekly_Chicken_Demand_Kg ~ Household_Income + Household_Size + "
        "Price_Pork_3F + Price_Chicken + Price_Ratio_Pork_Chicken"
    )
    X, y, feature_names = parse_formula(formula, df)

    print("\n[+] Step 1: Fitting Phase 1 (Baseline OLS) + Phase 2 (Apoptosis Tree)...")
    booster = NirvanaTwoStageRegressor(base_estimator="ols", max_depth=8, apoptosis_start_depth=5)
    booster.fit(X, y)

    print("\n[+] Step 2: Ranking Top 5 Discovered Variables & Executing 100-Universe Proof...")
    ranked_report = booster.rank_discovered_variables(X, y, top_k=5, n_universes=100, seed=42)

    print("\n" + ranked_report.summary())


if __name__ == "__main__":
    run_ranking()
