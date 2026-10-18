"""
NIRVANA STATIC DATA ECONOMETRICS: SUBSTITUTE GOODS PIPELINE
============================================================
Two-Stage Econometric Booster for cross-sectional micro datasets.
Demonstrates how Nirvana Phase 2 rescues weak Phase 1 linear models (OLS)
via Empirical Fisher Margin-Density Apoptosis.

Author: yinyusi65
"""

import sys
import os
import numpy as np
import pandas as pd

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nirvana import NirvanaStaticBooster
from nirvana.domains.substitute import (
    DEFAULT_PROTEIN_SUBSTITUTE_ROLES,
    analyze_substitution_thresholds,
)


def run_pipeline():
    print("=" * 78)
    print("NIRVANA STATIC ECONOMETRICS MODULE: PRODUCT SUBSTITUTION & PHASE 2 BOOSTER")
    print("Empirical Fisher Margin-Density Apoptosis on Cross-Sectional Household Data")
    print("=" * 78)

    # 1. Load static micro panel
    data_path = os.path.join(os.path.dirname(__file__), "data", "static_substitute_goods_panel.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_substitute_data.py first.")

    df = pd.read_csv(data_path)
    print(f"\n[+] Loaded Static Micro Dataset: {df.shape[0]} observations x {df.shape[1]} features.")
    print("    - Target Variable:  Weekly_Chicken_Demand_Kg (Household weekly consumption)")
    print("    - Substitute Pair:  Pork (Anchor Good) <-> Chicken (Close Substitute)")

    feature_cols = [
        "Household_Income",
        "Household_Size",
        "Urban_Flag",
        "Price_Pork_3F",
        "Price_Chicken",
        "Price_Eggs",
        "Promo_Discount_Chicken",
        "Price_Ratio_Pork_Chicken",
    ]
    target_col = "Weekly_Chicken_Demand_Kg"

    X = df[feature_cols]
    y = df[target_col]

    # 2. Fit Nirvana Static Booster
    print("\n[+] Step 1: Fitting Two-Stage Econometric Pipeline (Phase 1 OLS + Phase 2 Tree)...")
    booster = NirvanaStaticBooster(
        cancer_sensitivity=1.0,
        max_depth=10,
        apoptosis_start_depth=6,
        learning_rate=1.0,
        min_samples_split=6,
        split_quantiles=[10.0, 20.0, 35.0, 50.0, 65.0, 80.0, 90.0],
    )
    report = booster.fit(X, y)

    # 3. Print Academic Econometric Report
    print("\n" + report.summary())

    # 4. Extract Domain-Specific Substitution Insights
    print("\n[+] Step 2: Extracting Specialized Domain Substitution Metrics...")
    sub_metrics = analyze_substitution_thresholds(report)
    print(f"    - Dominant Driver Feature:       {sub_metrics['primary_non_linear_driver']}")
    print(f"    - Critical Price Ratio Trigger:  {sub_metrics['critical_price_ratio_trigger']}")
    print(f"    - Binding Income Cutoff:         {sub_metrics['binding_income_threshold_million_vnd']} Million VND")
    print(f"    - Classified Economic Regime:    {sub_metrics['economic_regime']}")
    print(f"    - Variance Resolved by Phase 2:  {sub_metrics['unexplained_variance_resolved_pct']:.2f}%")

    # 5. Out-of-Sample 5-Fold Cross Validation
    print("\n[+] Step 3: Running 5-Fold Out-of-Sample (OOS) Cross-Validation Audit...")
    cv_metrics = booster.evaluate_cv(X, y, k_folds=5)
    print("=" * 78)
    print("5-FOLD CROSS-VALIDATION OUT-OF-SAMPLE (OOS) GENERALIZATION METRICS:")
    print(f"   - Phase 1 Baseline OOS Mean R^2: {cv_metrics['r2_phase1_cv_mean']:.4f} (+/- {cv_metrics['r2_phase1_cv_std']:.4f})")
    print(f"   - Phase 2 Nirvana OOS Mean R^2:  {cv_metrics['r2_phase2_cv_mean']:.4f} (+/- {cv_metrics['r2_phase2_cv_std']:.4f})")
    print(f"   - Net Out-of-Sample R^2 Gain:    +{cv_metrics['delta_r2_cv_mean']:.4f} (+{cv_metrics['delta_r2_cv_mean'] * 100:.2f} percentage points)")
    print("=" * 78)

    print("\n[✓] Static Econometrics Pipeline Finished Successfully!\n")


if __name__ == "__main__":
    run_pipeline()
