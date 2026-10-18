"""
NIRVANA GENERAL-PURPOSE STATIC ESTIMATOR & PIPELINE WORKFLOW
=============================================================
Demonstrates Nirvana running smoothly as a general-purpose estimator:
- Formula interface ('y ~ x1 + x2 + ...')
- Scikit-learn Pipeline with StandardScaler and NirvanaTwoStageRegressor
- K-fold Cross Validation evaluation

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

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

from nirvana import (
    NirvanaTreeRegressor,
    NirvanaTwoStageRegressor,
    parse_formula,
)


def run_pipeline():
    print("=" * 78)
    print("NIRVANA GENERAL-PURPOSE STATIC PIPELINE")
    print("Scikit-Learn Pipeline Integration, Formula Syntax & Cross-Validation")
    print("=" * 78)

    # 1. Load Dataset
    data_path = os.path.join(os.path.dirname(__file__), "data", "static_substitute_goods_panel.csv")
    df = pd.read_csv(data_path)

    # 2. Formula Syntax Interface (R / Stata style)
    formula = (
        "Weekly_Chicken_Demand_Kg ~ Household_Income + Household_Size + "
        "Price_Pork_3F + Price_Chicken + Price_Ratio_Pork_Chicken"
    )
    print(f"\n[+] Step 1: Parsing Formula Specification:\n    '{formula}'")
    X, y, feature_names = parse_formula(formula, df)
    print(f"    -> Extracted {len(feature_names)} features across {len(y)} observations.")

    # 3. Scikit-learn Pipeline Integration
    print("\n[+] Step 2: Testing Scikit-Learn Pipeline Compatibility...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("booster", NirvanaTwoStageRegressor(base_estimator=Ridge(alpha=1.0), max_depth=8, apoptosis_start_depth=5)),
    ])

    print("    -> Fitting Pipeline on full dataset...")
    pipe.fit(X, y)
    r2_score = pipe.score(X, y)
    print(f"    -> Full Sample Fit R^2: {r2_score * 100:.2f}%")

    # 4. Out-of-Sample 5-Fold Cross Validation
    print("\n[+] Step 3: Running 5-Fold Cross Validation...")
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
    print(f"    -> 5-Fold CV R^2 Mean: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # 5. Standalone Tree Regressor
    print("\n[+] Step 4: Testing Standalone NirvanaTreeRegressor...")
    tree = NirvanaTreeRegressor(max_depth=8, apoptosis_start_depth=5)
    tree.fit(X, y)
    print(f"    -> Standalone Tree R^2: {tree.score(X, y) * 100:.2f}%")
    print("    -> Top Feature Importances:")
    for feat, imp in sorted(zip(feature_names, tree.feature_importances_), key=lambda x: -x[1])[:3]:
        print(f"       * {feat:25s}: {imp * 100:6.2f}%")

    print("\n[✓] General-Purpose Static Pipeline Executed Stably and Successfully!\n")


if __name__ == "__main__":
    run_pipeline()
