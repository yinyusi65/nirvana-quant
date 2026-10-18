"""
UNIT TESTS: NIRVANA STATIC DATA ECONOMETRICS MODULE
====================================================
Validates static cross-sectional dataset loading, two-stage econometric boosting,
Phase 2 R^2 elevation, and specialized domain substitution threshold analysis.

Author: yinyusi65
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nirvana import NirvanaStaticBooster, StaticEconometricReport
from nirvana.domains.substitute import (
    DEFAULT_PROTEIN_SUBSTITUTE_ROLES,
    analyze_substitution_thresholds,
)


def test_static_panel_loading():
    data_path = os.path.join(os.path.dirname(__file__), "data", "static_substitute_goods_panel.csv")
    assert os.path.exists(data_path), "Dataset file exists"
    df = pd.read_csv(data_path)
    assert df.shape[0] == 2500, "2500 observations in panel"
    assert df.shape[1] == 9, "9 columns in panel"
    assert not df.isna().any().any(), "Zero missing values"


def test_static_booster_and_r2_elevation():
    data_path = os.path.join(os.path.dirname(__file__), "data", "static_substitute_goods_panel.csv")
    df = pd.read_csv(data_path)

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

    booster = NirvanaStaticBooster(
        cancer_sensitivity=1.0,
        max_depth=8,
        apoptosis_start_depth=5,
        min_samples_split=6,
    )
    report = booster.fit(df[feature_cols], df[target_col])

    assert isinstance(report, StaticEconometricReport)
    assert report.r2_phase2 > report.r2_phase1 + 0.20, "Phase 2 elevates R^2 by > 20 pp"
    assert report.r2_phase2 > 0.85, "Phase 2 R^2 exceeds 85%"
    assert report.f_p_value < 1e-5, "Incremental F-test highly significant"

    preds = booster.predict(df[feature_cols])
    assert len(preds) == 2500
    assert np.isclose(booster.score(df[feature_cols], df[target_col]), report.r2_phase2)


def test_domain_substitution_threshold_extraction():
    data_path = os.path.join(os.path.dirname(__file__), "data", "static_substitute_goods_panel.csv")
    df = pd.read_csv(data_path)

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

    booster = NirvanaStaticBooster(max_depth=6, apoptosis_start_depth=4)
    report = booster.fit(df[feature_cols], df[target_col])

    sub_metrics = analyze_substitution_thresholds(report)
    assert "primary_non_linear_driver" in sub_metrics
    assert "economic_regime" in sub_metrics
    assert sub_metrics["economic_regime"] == "Threshold-Driven Budget Substitution"


def test_stata_diagnostics_on_static_panel():
    data_path = os.path.join(os.path.dirname(__file__), "data", "static_substitute_goods_panel.csv")
    df = pd.read_csv(data_path)

    from nirvana import stata_regress, stata_summarize, stata_pwcorr
    res = stata_regress(
        df,
        y="Weekly_Chicken_Demand_Kg",
        x=["Household_Income", "Price_Pork_3F", "Price_Chicken", "Price_Ratio_Pork_Chicken"],
        robust=True,
    )
    assert res.r_squared > 0.45
    assert res.is_robust
    assert "F(" in res.summary()

    # Post-estimation diagnostics
    vif = res.estat_vif(df)
    assert vif["mean_vif"] > 1.0
    het = res.estat_hettest()
    assert het["df"] == 1
    ov = res.estat_ovtest(df)
    assert ov["f_stat"] > 0.0
    dw = res.estat_dwatson()
    assert 1.0 < dw["d_stat"] < 3.0
    sk = res.sktest()
    assert sk["jb_stat"] >= 0.0

    sum_str = stata_summarize(df)
    assert "Household_Income" in sum_str
    corr_str = stata_pwcorr(df)
    assert "Weekly_Chicken_Demand_Kg" in corr_str


def test_general_purpose_sklearn_pipeline_and_stability():
    data_path = os.path.join(os.path.dirname(__file__), "data", "static_substitute_goods_panel.csv")
    df = pd.read_csv(data_path)

    from nirvana import (
        NirvanaTreeRegressor,
        NirvanaTwoStageRegressor,
        parse_formula,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    # 1. Formula parser
    formula = "Weekly_Chicken_Demand_Kg ~ Household_Income + Price_Pork_3F + Price_Chicken + Price_Ratio_Pork_Chicken"
    X, y, feats = parse_formula(formula, df)
    assert len(feats) == 4
    assert len(y) == 2500

    # 2. Standalone tree regressor
    tree = NirvanaTreeRegressor(max_depth=6)
    tree.fit(X, y)
    assert tree.score(X, y) > 0.60
    assert len(tree.feature_importances_) == 4

    # 3. TwoStageRegressor in Pipeline & cross_val_score
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("booster", NirvanaTwoStageRegressor(base_estimator=Ridge(alpha=1.0), max_depth=6)),
    ])
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert len(pred) == 2500
    assert pipe.score(X, y) > 0.70

    cv_scores = cross_val_score(pipe, X.iloc[:400], y.iloc[:400], cv=3, scoring="r2")
    assert len(cv_scores) == 3
    assert np.all(np.isfinite(cv_scores))

    # 4. Numerical Stability: Constant target / edge case
    X_edge = np.random.randn(20, 2)
    y_const = np.ones(20) * 5.0
    safe_reg = NirvanaTwoStageRegressor(max_depth=4)
    safe_reg.fit(X_edge, y_const)
    pred_const = safe_reg.predict(X_edge)
    assert np.all(np.isfinite(pred_const))


def test_phase2_multiverse_variable_proof():
    """Verify that Phase 2 discovered variables are proven econometrically and across universes."""
    from nirvana import NirvanaTwoStageRegressor, prove_phase2_variable

    np.random.seed(42)
    N = 300
    x1 = np.random.uniform(10, 50, size=N)
    x2 = np.random.uniform(1, 10, size=N)
    # Non-linear threshold interaction
    y = 2.0 + 0.5 * x1 + 1.2 * x2 + np.where((x1 > 30) & (x2 > 5), 15.0, 0.0) + np.random.normal(0, 1.0, size=N)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    model = NirvanaTwoStageRegressor(base_estimator="ols", max_depth=6)
    model.fit(df[["x1", "x2"]], df["y"])

    proof = model.prove_phase2_multiverse(df[["x1", "x2"]], df["y"], n_universes=25, seed=42)

    assert proof.r2_augmented > proof.r2_base
    assert proof.partial_r2 > 0.40
    assert proof.gamma_p_value < 0.001
    assert proof.multiverse_survival_rate >= 0.90
    assert proof.multiverse_sign_consistency == 1.0
    assert proof.is_reset_cured or proof.reset_f_augmented < proof.reset_f_base


def test_structural_terminal_protections():
    """Verify James-Stein shrinkage, soft splits, sign constraints, and SVD factor compression."""
    from nirvana import NirvanaTreeEngine, TrainingSample, NirvanaTwoStageRegressor

    # 1. James-Stein Hierarchical Bayesian Shrinkage on small leaf
    samples = [TrainingSample(features=[float(i)], target=5.0) for i in range(20)]
    # Extreme outlier spike at leaf with only 2 samples
    samples.append(TrainingSample(features=[25.0], target=25.0))
    samples.append(TrainingSample(features=[26.0], target=25.0))

    tree_unshrunk = NirvanaTreeEngine(hierarchical_shrinkage=0.0, min_samples_leaf=1)
    tree_unshrunk.train(samples)

    tree_shrunk = NirvanaTreeEngine(hierarchical_shrinkage=5.0, min_samples_leaf=1)
    tree_shrunk.train(samples)

    pred_unshrunk = tree_unshrunk.predict_one([25.5])
    pred_shrunk = tree_shrunk.predict_one([25.5])

    assert pred_shrunk < pred_unshrunk, "Hierarchical shrinkage pulls outlier leaf towards parent mean"
    assert 5.0 < pred_shrunk < 20.0, f"Shrunk prediction {pred_shrunk} is between parent (5.0) and sample mean (25.0)"

    rules = tree_shrunk.export_all_symbolic_rules()
    assert any("shrinkage_lambda" in r and r["shrinkage_lambda"] < 1.0 for r in rules), "Rules export shrinkage lambda"

    # 2. Smooth Transition Regimes (Soft splits)
    tree_soft = NirvanaTreeEngine(soft_splits=True, transition_gamma=3.0, min_samples_leaf=1)
    tree_soft.train(samples)
    p_left = tree_soft.predict_one([19.0])
    p_mid = tree_soft.predict_one([22.0])
    p_right = tree_soft.predict_one([25.0])
    assert p_left < p_mid < p_right, "Soft splits provide continuous intermediate transitions across boundary"

    # 3. Economic Sign Constraints
    N = 60
    np.random.seed(42)
    x0 = np.linspace(1, 10, N)
    x1 = np.linspace(1, 10, N) + np.random.normal(0, 0.5, N)
    y = 5.0 + 3.0 * x0 - 2.0 * x1 + np.random.normal(0, 0.2, N)
    df_toy = pd.DataFrame({"PosVar": x0, "NegVar": x1, "y": y})

    model_sign = NirvanaTwoStageRegressor(
        base_estimator="ols",
        sign_constraints={"PosVar": 1, "NegVar": -1},
        max_depth=4,
    )
    model_sign.fit(df_toy[["PosVar", "NegVar"]], df_toy["y"])
    assert model_sign.beta_[1] >= 0.0, "Feature 0 conforms to positive sign constraint"
    assert model_sign.beta_[2] <= 0.0, "Feature 1 conforms to negative sign constraint"

    # 4. SVD Factor Compression for Small N
    model_svd = NirvanaTwoStageRegressor(
        base_estimator="ols",
        factor_compression=True,
        n_components=1,
        max_depth=4,
    )
    model_svd.fit(df_toy[["PosVar", "NegVar"]], df_toy["y"])
    assert model_svd.pca_components_ is not None
    assert model_svd.pca_components_.shape == (1, 2)
    preds_svd = model_svd.predict(df_toy[["PosVar", "NegVar"]])
    assert len(preds_svd) == N
    assert np.all(np.isfinite(preds_svd))


if __name__ == "__main__":
    test_static_panel_loading()
    test_static_booster_and_r2_elevation()
    test_domain_substitution_threshold_extraction()
    test_stata_diagnostics_on_static_panel()
    test_general_purpose_sklearn_pipeline_and_stability()
    test_phase2_multiverse_variable_proof()
    test_structural_terminal_protections()
    print("ALL STATIC ECONOMETRICS TESTS PASSED 100%!")

