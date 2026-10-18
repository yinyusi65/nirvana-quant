"""
NIRVANA DIVERSE STATIC ECONOMETRICS SHOWCASE
=============================================
Demonstrating Nirvana running on classical cross-sectional econometric models:
1. Mincer Human Capital Earnings Equation (Labor Economics)
2. Cobb-Douglas / Translog Production Function (Firm Production)
3. Hedonic Real Estate Pricing (Asset Valuation)

Zero network dependencies - 100% pure static tabular econometrics.
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nirvana import (
    NirvanaTwoStageRegressor,
    NirvanaTreeRegressor,
    stata_regress,
    stata_summarize,
    parse_formula,
)


def run_mincer_wage_model():
    print("\n" + "=" * 78)
    print("1. MODEL: PHƯƠNG TRÌNH TIỀN LƯƠNG MINCER (LABOR ECONOMICS)")
    print("   ln(Wage) = f(Education, Experience, Training, Gender, Urban)")
    print("=" * 78)
    np.random.seed(42)
    N = 2000

    # Features
    educ = np.random.choice([9, 12, 14, 16, 18, 20], size=N, p=[0.1, 0.3, 0.2, 0.25, 0.1, 0.05])
    exper = np.random.uniform(1, 35, size=N)
    training_cert = np.random.binomial(1, 0.35, size=N)
    urban = np.random.binomial(1, 0.60, size=N)

    # True non-linear data generating process:
    # Threshold at Master/PhD level (educ >= 16) + concave experience + training multiplier
    base_log_wage = 1.5 + 0.06 * educ + 0.04 * exper - 0.0006 * (exper ** 2) + 0.15 * training_cert + 0.20 * urban
    # Non-linear threshold interaction: High education + Certificate gives bonus return
    bonus = np.where((educ >= 16) & (training_cert == 1), 0.35, 0.0)
    noise = np.random.normal(0, 0.25, size=N)
    log_wage = base_log_wage + bonus + noise

    df_mincer = pd.DataFrame({
        "log_wage": log_wage,
        "education": educ,
        "experience": exper,
        "experience_sq": exper ** 2,
        "training_cert": training_cert,
        "urban": urban,
    })

    # Stata Regression
    stata_res = stata_regress(df_mincer, "log_wage", ["education", "experience", "experience_sq", "training_cert", "urban"])
    print(stata_res.summary())

    # Ramsey RESET test
    reset = stata_res.estat_ovtest()
    print(f"\n[Stata estat ovtest] Ramsey RESET F-stat: {reset['f_stat']:.2f} (p-value: {reset['p_value']:.4e})")

    # Nirvana Two-Stage Booster
    booster = NirvanaTwoStageRegressor(base_estimator="ols", max_depth=6)
    X, y, feats = parse_formula("log_wage ~ education + experience + experience_sq + training_cert + urban", df_mincer)
    report = booster.fit(X, y)
    print(f"\n[Nirvana Phase 2 Booster] Base R^2: {report.r2_phase1 * 100:.2f}% -> Nirvana R^2: {report.r2_phase2 * 100:.2f}% (Gain: +{report.r2_gain * 100:.2f} pp)")


def run_cobb_douglas_model():
    print("\n" + "=" * 78)
    print("2. MODEL: HÀM SẢN XUẤT COBB-DOUGLAS / TRANSLOG (FIRM PRODUCTION)")
    print("   ln(Output) = f(ln_Capital, ln_Labor, R&D_Intensity, Foreign_Ownership)")
    print("=" * 78)
    np.random.seed(101)
    N = 1500

    capital = np.exp(np.random.normal(6.0, 1.2, size=N))
    labor = np.exp(np.random.normal(4.5, 0.9, size=N))
    rd_intensity = np.random.uniform(0.01, 0.15, size=N)
    fdi = np.random.binomial(1, 0.25, size=N)

    ln_k = np.log(capital)
    ln_l = np.log(labor)

    # Output with scale economies threshold
    ln_output = 2.0 + 0.45 * ln_k + 0.40 * ln_l + 1.8 * rd_intensity + 0.25 * fdi
    # Synergistic interaction: FDI + High R&D creates technological leap
    synergy = np.where((fdi == 1) & (rd_intensity > 0.08), 0.50, 0.0)
    ln_output = ln_output + synergy + np.random.normal(0, 0.20, size=N)

    df_firm = pd.DataFrame({
        "ln_output": ln_output,
        "ln_k": ln_k,
        "ln_l": ln_l,
        "rd_intensity": rd_intensity,
        "fdi": fdi,
    })

    stata_res = stata_regress(df_firm, "ln_output", ["ln_k", "ln_l", "rd_intensity", "fdi"])
    print(stata_res.summary())

    booster = NirvanaTwoStageRegressor(base_estimator="ols", max_depth=6)
    report = booster.fit(df_firm[["ln_k", "ln_l", "rd_intensity", "fdi"]], df_firm["ln_output"])
    print(f"\n[Nirvana Phase 2 Booster] Base R^2: {report.r2_phase1 * 100:.2f}% -> Nirvana R^2: {report.r2_phase2 * 100:.2f}% (Gain: +{report.r2_gain * 100:.2f} pp)")


def run_hedonic_pricing_model():
    print("\n" + "=" * 78)
    print("3. MODEL: ĐỊNH GIÁ BẤT ĐỘNG SẢN HEDONIC (REAL ESTATE ASSET VALUATION)")
    print("   House_Price = f(Area_m2, Bedrooms, Distance_CBD_Km, Metro_Near, High_End)")
    print("=" * 78)
    np.random.seed(88)
    N = 3000

    area = np.random.uniform(40, 250, size=N)
    bedrooms = np.clip(np.round(area / 45) + np.random.choice([-1, 0, 1], size=N), 1, 6)
    dist_cbd = np.random.uniform(1.0, 25.0, size=N)
    metro = np.random.binomial(1, 0.30, size=N)

    # Hedonic price (Million VND / m2)
    base_price = 30.0 + 0.15 * area + 3.0 * bedrooms - 1.8 * dist_cbd + 15.0 * metro
    # Luxury jump: Area > 150m2 near metro has luxury premium
    luxury_premium = np.where((area > 150) & (dist_cbd < 5.0), 35.0, 0.0)
    noise = np.random.normal(0, 8.0, size=N)
    total_price = (base_price + luxury_premium + noise)

    df_house = pd.DataFrame({
        "Price_Million_VND": total_price,
        "Area_m2": area,
        "Bedrooms": bedrooms,
        "Distance_CBD_Km": dist_cbd,
        "Near_Metro": metro,
    })

    stata_res = stata_regress(df_house, "Price_Million_VND", ["Area_m2", "Bedrooms", "Distance_CBD_Km", "Near_Metro"], robust=True)
    print(stata_res.summary())

    booster = NirvanaTwoStageRegressor(base_estimator="ols", max_depth=6)
    report = booster.fit(df_house[["Area_m2", "Bedrooms", "Distance_CBD_Km", "Near_Metro"]], df_house["Price_Million_VND"])
    print(f"\n[Nirvana Phase 2 Booster] Base R^2: {report.r2_phase1 * 100:.2f}% -> Nirvana R^2: {report.r2_phase2 * 100:.2f}% (Gain: +{report.r2_gain * 100:.2f} pp)")


if __name__ == "__main__":
    run_mincer_wage_model()
    run_cobb_douglas_model()
    run_hedonic_pricing_model()
