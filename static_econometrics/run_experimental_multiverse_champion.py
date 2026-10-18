"""
NIRVANA EXPERIMENTAL LAB: MULTIVERSE CHAMPION RUNNER
=====================================================
Demonstrating the isolated experimental module on household data.
Strictly isolated from production core.

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

from nirvana.experimental import MultiverseChampionExtractor
from nirvana import parse_formula


def run_experimental():
    data_path = os.path.join(os.path.dirname(__file__), "data", "static_substitute_goods_panel.csv")
    df = pd.read_csv(data_path)

    formula = (
        "Weekly_Chicken_Demand_Kg ~ Household_Income + Household_Size + "
        "Price_Pork_3F + Price_Chicken + Price_Ratio_Pork_Chicken"
    )
    X, y, feature_names = parse_formula(formula, df)

    print("[Experimental Lab] Initializing MultiverseChampionExtractor...")
    extractor = MultiverseChampionExtractor(base_estimator="ols", max_depth=8)

    print("[Experimental Lab] Screening 100 Universes with Safety Bounds...")
    report = extractor.fit_and_extract(X, y, top_k=5, n_universes=100, seed=42)

    print("\n" + report.summary())


if __name__ == "__main__":
    run_experimental()
