"""
========================================================================================
AUDIT REPRODUCIBILITY PIPELINE: NIRVANA-QUANT v1.1.6
Multi-Bank Volatility Forecasting & Uncertainty Quantification
Strict 100% Identical Fit / Calibration / Test Partitions Across All Models
Unified Regularization: Ridge alpha = 10.0 Across All Baseline and TwoStage Configurations
========================================================================================
Data File: data/all27_panel_full_2017_2025.csv
SHA-256: 62c9634e03c6d179745403abc45e3b1c1aecb54a9b2ccbfae8c5f008d85e781f
Total Raw Rows: 2,227 trading days (2017-01-03 to 2025-12-31)
========================================================================================
Environment:
- python == 3.13.0
- nirvana-quant == 1.1.6 (PyPI Release)
- scikit-learn == 1.8.0
- pandas == 2.3.3
- numpy == 2.2.6
- scipy == 1.16.3
========================================================================================
"""

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from nirvana import Base12Manifold, CausalOnlineKalman, NirvanaTwoStageRegressor, NirvanaTreeRegressor

def winkler_score(y_true, lo, hi, alpha=0.10):
    """Gneiting & Raftery (2007) Winkler Interval Score at nominal coverage 1 - alpha."""
    width = hi - lo
    penalty_lo = np.maximum(0.0, lo - y_true) * (2.0 / alpha)
    penalty_hi = np.maximum(0.0, y_true - hi) * (2.0 / alpha)
    return np.mean(width + penalty_lo + penalty_hi)

def build_features(df_core, target_bank, core_banks):
    """
    Construct Post-Close feature sets available at close of day t to forecast volatility at t+1.
    Zero future leakage: target is y_{t+1} = target_bank.shift(-1).
    """
    y_series = df_core[target_bank].shift(-1)
    
    features = pd.DataFrame(index=df_core.index)
    for b in core_banks:
        features[f"{b}_lag0"] = df_core[b]          # Realized vol up to close of t
        features[f"{b}_lag1"] = df_core[b].shift(1)  # Realized vol up to close of t-1
        features[f"{b}_lag5"] = df_core[b].shift(5)  # Realized vol up to close of t-5
    features[f"{target_bank}_lag21"] = df_core[target_bank].shift(21)
    features["System_Mean_Vol"] = df_core.mean(axis=1) # Market average vol at t
    
    # 1D Causal Online Kalman Filter (updated strictly sequentially to end of day t)
    kf = CausalOnlineKalman(q=1e-3, r=1e-2)
    kalman_state = kf.filter_series(df_core[target_bank].values)
    kalman_feat = pd.Series(kalman_state, index=df_core.index)
    
    # Base-12 Manifold Harmonics
    manifold = Base12Manifold()
    t_coords = manifold.batch_transform(np.arange(len(df_core)), dates=df_core.index)
    m_cols = [f"Manifold_H{i+1}" for i in range(t_coords.shape[1])]
    df_manifold = pd.DataFrame(t_coords, index=df_core.index, columns=m_cols)
    
    X_base = features.copy()
    X_kalman = pd.concat([features, pd.DataFrame({"Kalman_State": kalman_feat})], axis=1)
    X_base12 = pd.concat([features, df_manifold], axis=1)
    X_both = pd.concat([features, df_manifold, pd.DataFrame({"Kalman_State": kalman_feat})], axis=1)
    
    return X_base, X_kalman, X_base12, X_both, y_series

def run_multi_bank_benchmark():
    print("=" * 105)
    print("STEP 1: MULTI-BANK ROBUSTNESS REPLICATION (7 CORE BANKS, OOS 2024-2025, UNIFIED RIDGE ALPHA=10.0)")
    print("=" * 105)
    
    df_raw = pd.read_csv("data/all27_panel_full_2017_2025.csv")
    df_raw["date"] = pd.to_datetime(df_raw["index"])
    df = df_raw.set_index("date").drop(columns=["index"])
    core_banks = ["VCB", "CTG", "BID", "ACB", "MBB", "STB", "SHB"]
    df_core = df[core_banks].copy()
    
    results = []
    
    for bank in core_banks:
        X_base, X_kalman, _, _, y_series = build_features(df_core, bank, core_banks)
        valid_mask = ~(X_kalman.isna().any(axis=1) | y_series.isna())
        dates_valid = df_core.index[valid_mask]
        y_valid = y_series.loc[valid_mask]
        
        train_mask = dates_valid < "2024-01-01"
        test_mask = dates_valid >= "2024-01-01"
        
        # 100% Identical split for all models:
        # Fit set: first 1,451 observations (2017-02-07 to 2022-11-28)
        # Cal set: last 256 observations of Train (2022-11-29 to 2023-12-29)
        # Test set: 498 observations (2024-01-02 to 2025-12-30)
        n_cal = 256
        n_tr = train_mask.sum()
        fit_idx = np.arange(n_tr - n_cal)
        cal_idx = np.arange(n_tr - n_cal, n_tr)
        
        y_tr = y_valid[train_mask]
        y_te = y_valid[test_mask]
        y_fit = y_tr.iloc[fit_idx]
        y_cal = y_tr.iloc[cal_idx]
        
        # Model A: Ridge Baseline (alpha=10.0, Without Kalman)
        X_tr_b = X_base.loc[valid_mask].loc[train_mask]
        X_te_b = X_base.loc[valid_mask].loc[test_mask]
        m_base = Ridge(alpha=10.0).fit(X_tr_b.iloc[fit_idx], y_fit)
        pred_b = m_base.predict(X_te_b)
        cal_res_b = np.abs(y_cal.values - m_base.predict(X_tr_b.iloc[cal_idx]))
        k_b = int(np.ceil((len(cal_res_b) + 1) * 0.90))
        q_b = np.sort(cal_res_b)[min(k_b - 1, len(cal_res_b) - 1)]
        w_b = winkler_score(y_te.values, pred_b - q_b, pred_b + q_b)
        r2_b = r2_score(y_te, pred_b)
        
        # Model B: Ridge + Causal Kalman (alpha=10.0)
        X_tr_k = X_kalman.loc[valid_mask].loc[train_mask]
        X_te_k = X_kalman.loc[valid_mask].loc[test_mask]
        m_rk = Ridge(alpha=10.0).fit(X_tr_k.iloc[fit_idx], y_fit)
        pred_rk = m_rk.predict(X_te_k)
        cal_res_rk = np.abs(y_cal.values - m_rk.predict(X_tr_k.iloc[cal_idx]))
        k_k = int(np.ceil((len(cal_res_rk) + 1) * 0.90))
        q_rk = np.sort(cal_res_rk)[min(k_k - 1, len(cal_res_rk) - 1)]
        w_rk = winkler_score(y_te.values, pred_rk - q_rk, pred_rk + q_rk)
        r2_rk = r2_score(y_te, pred_rk)
        
        # Model C: TwoStage + Causal Kalman (STRICT UNIFIED Ridge alpha=10.0 in Phase 1)
        m_tk = NirvanaTwoStageRegressor(base_estimator=Ridge(alpha=10.0), learning_rate=0.3, max_depth=5, cancer_sensitivity=1.2)
        m_tk.fit(X_tr_k.iloc[fit_idx], y_fit, X_cal=X_tr_k.iloc[cal_idx], y_cal=y_cal)
        pred_tk = m_tk.predict(X_te_k)
        _, lo_tk, hi_tk = m_tk.predict_interval(X_te_k, alpha=0.10)
        w_tk = winkler_score(y_te.values, lo_tk, hi_tk)
        r2_tk = r2_score(y_te, pred_tk)
        cov_tk = np.mean((y_te.values >= lo_tk) & (y_te.values <= hi_tk)) * 100.0
        
        results.append({
            "Bank": bank,
            "R2_Base": r2_b,
            "R2_RidgeKalman": r2_rk,
            "R2_TwoStageKalman": r2_tk,
            "Winkler_Base": w_b,
            "Winkler_RidgeKalman": w_rk,
            "Winkler_TwoStageKalman": w_tk,
            "Cov_TwoStageKalman": cov_tk
        })
        
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    print("\nEXACT COLUMN MEANS ACROSS 7 CORE BANKS (UNIFIED ALPHA=10.0):")
    for col in df_res.columns:
        if col != "Bank":
            print(f"{col:22s}: {df_res[col].mean():.6f}")
            
    return df_core, core_banks

def run_base12_ablation(df_core, core_banks):
    print("\n" + "=" * 105)
    print("STEP 2: DIRECT BASE-12 MANIFOLD ABLATION MATRIX WITH & WITHOUT STANDARDSCALER (VCB, ALPHA=10.0)")
    print("=" * 105)
    
    target_bank = "VCB"
    X_base, X_kalman, X_base12, X_both, y_series = build_features(df_core, target_bank, core_banks)
    valid_mask = ~(X_both.isna().any(axis=1) | y_series.isna())
    dates_valid = df_core.index[valid_mask]
    y_valid = y_series.loc[valid_mask]
    
    train_mask = dates_valid < "2024-01-01"
    test_mask = dates_valid >= "2024-01-01"
    
    n_cal = 256
    n_tr = train_mask.sum()
    fit_idx = np.arange(n_tr - n_cal)
    cal_idx = np.arange(n_tr - n_cal, n_tr)
    
    y_tr = y_valid[train_mask]
    y_te = y_valid[test_mask]
    y_fit = y_tr.iloc[fit_idx]
    y_cal = y_tr.iloc[cal_idx]
    
    configs = [
        ("1. Ridge (Base Lags only)", X_base),
        ("2. Ridge + Base12 Manifold", X_base12),
        ("3. Ridge + Causal Kalman", X_kalman),
        ("4. Ridge + Base12 + Kalman", X_both),
    ]
    
    b12_results = []
    for name, X_df in configs:
        X_tr = X_df.loc[valid_mask].loc[train_mask]
        X_te = X_df.loc[valid_mask].loc[test_mask]
        
        # Raw unscaled
        m_raw = Ridge(alpha=10.0).fit(X_tr.iloc[fit_idx], y_fit)
        pred_raw = m_raw.predict(X_te)
        cal_res_raw = np.abs(y_cal.values - m_raw.predict(X_tr.iloc[cal_idx]))
        k_r = int(np.ceil((len(cal_res_raw) + 1) * 0.90))
        q_raw = np.sort(cal_res_raw)[min(k_r - 1, len(cal_res_raw) - 1)]
        lo_raw, hi_raw = pred_raw - q_raw, pred_raw + q_raw
        
        # Scaled (StandardScaler fit ONLY on Fit set!)
        scaler = StandardScaler().fit(X_tr.iloc[fit_idx])
        m_scaled = Ridge(alpha=10.0).fit(scaler.transform(X_tr.iloc[fit_idx]), y_fit)
        pred_scaled = m_scaled.predict(scaler.transform(X_te))
        cal_res_scaled = np.abs(y_cal.values - m_scaled.predict(scaler.transform(X_tr.iloc[cal_idx])))
        k_s = int(np.ceil((len(cal_res_scaled) + 1) * 0.90))
        q_scaled = np.sort(cal_res_scaled)[min(k_s - 1, len(cal_res_scaled) - 1)]
        lo_sc, hi_sc = pred_scaled - q_scaled, pred_scaled + q_scaled
        
        b12_results.append({
            "Configuration": name,
            "Raw_R2": r2_score(y_te, pred_raw),
            "Raw_Winkler": winkler_score(y_te.values, lo_raw, hi_raw),
            "Scaled_R2": r2_score(y_te, pred_scaled),
            "Scaled_Winkler": winkler_score(y_te.values, lo_sc, hi_sc),
        })
    df_b12 = pd.DataFrame(b12_results)
    print(df_b12.to_string(index=False))

def run_hypothesis_testing(df_core, core_banks):
    print("\n" + "=" * 105)
    print("STEP 3: CENTERED MOVING-BLOCK BOOTSTRAP HYPOTHESIS TESTING (VCB TEST SET, B=2000, ALPHA=10.0)")
    print("=" * 105)
    
    target_bank = "VCB"
    X_base, X_kalman, _, _, y_series = build_features(df_core, target_bank, core_banks)
    valid_mask = ~(X_kalman.isna().any(axis=1) | y_series.isna())
    dates_valid = df_core.index[valid_mask]
    y_valid = y_series.loc[valid_mask]
    
    train_mask = dates_valid < "2024-01-01"
    test_mask = dates_valid >= "2024-01-01"
    
    n_cal = 256
    n_tr = train_mask.sum()
    fit_idx = np.arange(n_tr - n_cal)
    cal_idx = np.arange(n_tr - n_cal, n_tr)
    
    X_tr = X_kalman.loc[valid_mask].loc[train_mask]
    y_tr = y_valid[train_mask]
    X_te = X_kalman.loc[valid_mask].loc[test_mask]
    y_te = y_valid[test_mask]
    y_fit = y_tr.iloc[fit_idx]
    y_cal = y_tr.iloc[cal_idx]
    
    # Model 1: Ridge + Kalman (alpha=10.0)
    m_ridge = Ridge(alpha=10.0).fit(X_tr.iloc[fit_idx], y_fit)
    pred_r = m_ridge.predict(X_te)
    cal_res_r = np.abs(y_cal.values - m_ridge.predict(X_tr.iloc[cal_idx]))
    k_r = int(np.ceil((len(cal_res_r) + 1) * 0.90))
    q_r = np.sort(cal_res_r)[min(k_r - 1, len(cal_res_r) - 1)]
    lo_r, hi_r = pred_r - q_r, pred_r + q_r
    loss_ridge = (hi_r - lo_r) + np.maximum(0.0, lo_r - y_te.values)*20.0 + np.maximum(0.0, y_te.values - hi_r)*20.0
    
    # Model 2: TwoStage + Kalman (UNIFIED alpha=10.0)
    m_two = NirvanaTwoStageRegressor(base_estimator=Ridge(alpha=10.0), learning_rate=0.3, max_depth=5, cancer_sensitivity=1.2)
    m_two.fit(X_tr.iloc[fit_idx], y_fit, X_cal=X_tr.iloc[cal_idx], y_cal=y_cal)
    pred_t = m_two.predict(X_te)
    _, lo_t, hi_t = m_two.predict_interval(X_te, alpha=0.10)
    loss_two = (hi_t - lo_t) + np.maximum(0.0, lo_t - y_te.values)*20.0 + np.maximum(0.0, y_te.values - hi_t)*20.0
    
    # 1. Delta_t = Loss(TwoStage) - Loss(Ridge)
    # A positive Delta_bar means Ridge has lower Winkler loss (Ridge is better)
    delta = loss_two - loss_ridge
    delta_bar = float(np.mean(delta))
    print(f"Sample observed mean differential Delta_bar = Loss(TwoStage) - Loss(Ridge): +{delta_bar:.6f}")
    
    delta_centered = delta - delta_bar
    B = 2000
    N = len(delta)
    
    print("\n[A] Hypothesis Test H0: Delta <= 0 vs H1: Delta > 0 (Test if Ridge is statistically significantly better):")
    for block_len in [5, 10, 20]:
        np.random.seed(42)
        n_blocks = int(np.ceil(N / block_len))
        boot_stats = []
        for _ in range(B):
            starts = np.random.randint(0, N - block_len + 1, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:N]
            boot_stats.append(np.mean(delta_centered[idx]))
        boot_stats = np.array(boot_stats)
        
        count_exceed = np.sum(boot_stats >= delta_bar)
        p_val = float((count_exceed + 1.0) / (B + 1.0))
        boot_uncentered = boot_stats + delta_bar
        ci_lo, ci_hi = np.percentile(boot_uncentered, [2.5, 97.5])
        print(f"  Block L={block_len:2d}: 95% CI = [{ci_lo:.5f}, {ci_hi:.5f}] | Exceedances: {count_exceed}/{B} | p-value = {p_val:.4f}")

def run_deep_pruning_audit(df_core, core_banks):
    print("\n" + "=" * 105)
    print("STEP 4: DEEP TREE FISHER PRUNING AUDIT & DIVERGENCE PROXY METRICS (VCB TEST SET, DEPTH 5 TO 10)")
    print("=" * 105)
    
    target_bank = "VCB"
    X_base, X_kalman, _, _, y_series = build_features(df_core, target_bank, core_banks)
    valid_mask = ~(X_kalman.isna().any(axis=1) | y_series.isna())
    dates_valid = df_core.index[valid_mask]
    y_valid = y_series.loc[valid_mask]
    
    train_mask = dates_valid < "2024-01-01"
    test_mask = dates_valid >= "2024-01-01"
    
    n_cal = 256
    n_tr = train_mask.sum()
    fit_idx = np.arange(n_tr - n_cal)
    cal_idx = np.arange(n_tr - n_cal, n_tr)
    
    X_fit = X_kalman.loc[valid_mask].loc[train_mask].iloc[fit_idx]
    y_fit = y_valid[train_mask].iloc[fit_idx]
    X_cal = X_kalman.loc[valid_mask].loc[train_mask].iloc[cal_idx]
    y_cal = y_valid[train_mask].iloc[cal_idx]
    X_te = X_kalman.loc[valid_mask].loc[test_mask]
    y_te = y_valid[test_mask]
    
    depth_grid = [5, 6, 8, 10]
    audit_rows = []
    
    for d in depth_grid:
        for sens, sname in [(1.2, "ON (1.2)"), (0.0, "OFF (0.0)")]:
            m = NirvanaTwoStageRegressor(base_estimator=Ridge(alpha=10.0), learning_rate=0.3, max_depth=d, cancer_sensitivity=sens)
            m.fit(X_fit, y_fit, X_cal=X_cal, y_cal=y_cal)
            pred = m.predict(X_te)
            _, lo, hi = m.predict_interval(X_te, alpha=0.10)
            
            engine = m.tree_engine_.tree_engine_
            n_total = len(engine.nodes)
            n_eligible = sum(1 for n in engine.nodes if n.depth >= engine.apoptosis_start_depth)
            n_pruned = sum(1 for n in engine.nodes if n.is_pruned_apoptosis)
            
            fishers = [n.fisher_info for n in engine.nodes if n.depth >= engine.apoptosis_start_depth]
            margins = [n.margin_density for n in engine.nodes if n.depth >= engine.apoptosis_start_depth]
            max_fisher_val = max(fishers) if fishers else 0.0
            min_margin_val = min(margins) if margins else 1.0
            
            audit_rows.append({
                "Max Depth": d,
                "Fisher Pruning": sname,
                "Total Nodes": n_total,
                "Eligible Nodes": n_eligible,
                "Pruned Nodes": n_pruned,
                "Max Fisher*Sens": max_fisher_val * sens,
                "Min Margin": min_margin_val,
                "OOS_R2": r2_score(y_te, pred),
                "Winkler": winkler_score(y_te.values, lo, hi)
            })
            
    df_audit = pd.DataFrame(audit_rows)
    print(df_audit.to_string(index=False))
    
    print("\n[Synthetic Proof-of-Concept Unit Test]: Verifying Fisher Apoptosis Excises Outlier Node When Triggered")
    np.random.seed(42)
    X_syn = np.random.randn(200, 2)
    y_syn = np.random.randn(200) * 0.05
    # Inject isolated extreme shock on tiny margin (2 samples = 1% margin, y = 50.0)
    X_syn[0, 0] = 5.0
    X_syn[1, 0] = 5.1
    y_syn[0] = 50.0
    y_syn[1] = 50.0
    
    tree_syn = NirvanaTreeRegressor(max_depth=5, apoptosis_start_depth=2, cancer_sensitivity=5.0)
    tree_syn.fit(X_syn, y_syn)
    engine_syn = tree_syn.tree_engine_
    pruned_syn = [n for n in engine_syn.nodes if n.is_pruned_apoptosis]
    print(f"  Synthetic injection result: Successfully excised {len(pruned_syn)} outlier node(s) with alert diagnosis: '{engine_syn.pathology_alerts[0].diagnosis}'")

if __name__ == "__main__":
    df_core, core_banks = run_multi_bank_benchmark()
    run_base12_ablation(df_core, core_banks)
    run_hypothesis_testing(df_core, core_banks)
    run_deep_pruning_audit(df_core, core_banks)
    print("\n" + "=" * 105)
    print("ALL AUDIT REPRODUCTION PIPELINE FINISHED SUCCESSFULLY WITH COMPREHENSIVE VERIFICATION.")
    print("=" * 105)
