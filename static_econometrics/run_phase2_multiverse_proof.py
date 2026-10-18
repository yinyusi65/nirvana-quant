"""
NIRVANA PHASE 2 & MULTIVERSE EMPIRICAL PROOF RUNNER
====================================================
Formally verifies and mathematically proves the validity of the newly
discovered non-linear latent variable and structural threshold rules
extracted from Phase 2 across 100 parallel counterfactual universes.

Dataset: 2,500 Vietnamese Household Micro-Panel (Pork <-> Chicken Substitution)
Target:  Weekly_Chicken_Demand_Kg

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
    prove_phase2_variable,
    parse_formula,
)


def run_proof():
    # 1. Load static dataset
    data_path = os.path.join(os.path.dirname(__file__), "data", "static_substitute_goods_panel.csv")
    df = pd.read_csv(data_path)

    # 2. Formula specification
    formula = (
        "Weekly_Chicken_Demand_Kg ~ Household_Income + Household_Size + "
        "Price_Pork_3F + Price_Chicken + Price_Ratio_Pork_Chicken"
    )
    X, y, feature_names = parse_formula(formula, df)

    print("\n[+] Step 1: Fitting Nirvana Two-Stage Booster on Weak Baseline Model...")
    booster = NirvanaTwoStageRegressor(base_estimator="ols", max_depth=8, apoptosis_start_depth=5)
    booster.fit(X, y)

    print("\n[+] Step 2: Executing Three-Pillar Econometric & Multiverse Proof Engine (100 Universes)...")
    proof_report = booster.prove_phase2_multiverse(X, y, n_universes=100, seed=42)

    print("\n" + proof_report.summary())


if __name__ == "__main__":
    run_proof()
