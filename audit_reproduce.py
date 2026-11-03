"""
========================================================================================
AUDIT REPRODUCIBILITY PIPELINE: NIRVANA-QUANT v1.1.6
Multi-Bank Volatility Forecasting & Uncertainty Quantification
Strict 100% Identical Fit / Calibration / Test Partitions Across All Models
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

from nirvana import Base12Manifold, CausalOnlineKalman, NirvanaTwoStageRegressor, NirvanaTreeRegressor

def winkler_score(y_true, lo, hi, alpha=0.10):
    """Gneiting & Raftery (2007) Winkler Interval Score at nominal coverage 1 - alpha."""
    width = hi - lo
    penalty_lo = np.maximum(0.0, lo - y_true) * (2.0 / alpha)
    penalty_hi = np.maximum(0.0, y_true - hi) * (2.0 / alpha)
    return np.mean(width + penalty_lo + penalty_hi)

def build_features(df_core, target_bank, core_banks):
    """
    Construct Post-Close feature set available at close of day t to forecast volatility at t+1.
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
    X_full = pd.concat([features, df_manifold, pd.DataFrame({"Kalman_State": kalman_feat})], axis=1)
    
    return X_base, X_kalman, X_full, y_series

def run_multi_bank_benchmark():
    print("=" * 100)
    print("STEP 1: MULTI-BANK ROBUSTNESS REPLICATION (7 CORE BANKS, OOS 2024-2025, N=498)")
    print("=" * 100)
    
    df_raw = pd.read_csv("data/all27_panel_full_2017_2025.csv")
    df_raw["date"] = pd.to_datetime(df_raw["index"])
    df = df_raw.set_index("date").drop(columns=["index"])
    core_banks = ["VCB", "CTG", "BID", "ACB", "MBB", "STB", "SHB"]
    df_core = df[core_banks].copy()
    
    results = []
    
    for bank in core_banks:
        X_base, X_kalman, _, y_series = build_features(df_core, bank, core_banks)
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
        
        # Model A: Ridge Baseline (Without Kalman)
        X_tr_b = X_base.loc[valid_mask].loc[train_mask]
        X_te_b = X_base.loc[valid_mask].loc[test_mask]
        m_base = Ridge(alpha=10.0).fit(X_tr_b.iloc[fit_idx], y_fit)
        pred_b = m_base.predict(X_te_b)
        cal_res_b = np.abs(y_cal.values - m_base.predict(X_tr_b.iloc[cal_idx]))
        k = int(np.ceil((len(cal_res_b) + 1) * 0.90))
        q_b = np.sort(cal_res_b)[min(k - 1, len(cal_res_b) - 1)]
        w_b = winkler_score(y_te.values, pred_b - q_b, pred_b + q_b)
        r2_b = r2_score(y_te, pred_b)
        
        # Model B: Ridge + Causal Kalman
        X_tr_k = X_kalman.loc[valid_mask].loc[train_mask]
        X_te_k = X_kalman.loc[valid_mask].loc[test_mask]
        m_rk = Ridge(alpha=10.0).fit(X_tr_k.iloc[fit_idx], y_fit)
        pred_rk = m_rk.predict(X_te_k)
        cal_res_rk = np.abs(y_cal.values - m_rk.predict(X_tr_k.iloc[cal_idx]))
        k = int(np.ceil((len(cal_res_rk) + 1) * 0.90))
        q_rk = np.sort(cal_res_rk)[min(k - 1, len(cal_res_rk) - 1)]
        w_rk = winkler_score(y_te.values, pred_rk - q_rk, pred_rk + q_rk)
        r2_rk = r2_score(y_te, pred_rk)
        
        # Model C: TwoStage + Causal Kalman (v1.1.6)
        m_tk = NirvanaTwoStageRegressor(base_estimator="ridge", learning_rate=0.3, max_depth=5, cancer_sensitivity=1.2)
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
    print("\nEXACT COLUMN MEANS ACROSS 7 CORE BANKS:")
    for col in df_res.columns:
        if col != "Bank":
            print(f"{col:22s}: {df_res[col].mean():.6f}")
            
    return df_core, core_banks

def run_hypothesis_testing(df_core, core_banks):
    print("\n" + "=" * 100)
    print("STEP 2: CENTERED MOVING-BLOCK BOOTSTRAP HYPOTHESIS TESTING (VCB TEST SET, B=2000)")
    print("=" * 100)
    
    target_bank = "VCB"
    X_base, X_kalman, _, y_series = build_features(df_core, target_bank, core_banks)
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
    
    # FIXED: Explicit definition of y_cal in this function scope
    y_fit = y_tr.iloc[fit_idx]
    y_cal = y_tr.iloc[cal_idx]
    
    m_ridge = Ridge(alpha=10.0).fit(X_tr.iloc[fit_idx], y_fit)
    pred_r = m_ridge.predict(X_te)
    cal_res_r = np.abs(y_cal.values - m_ridge.predict(X_tr.iloc[cal_idx]))
    k = int(np.ceil((len(cal_res_r) + 1) * 0.90))
    q_r = np.sort(cal_res_r)[min(k - 1, len(cal_res_r) - 1)]
    lo_r, hi_r = pred_r - q_r, pred_r + q_r
    loss_ridge = (hi_r - lo_r) + np.maximum(0.0, lo_r - y_te.values)*20.0 + np.maximum(0.0, y_te.values - hi_r)*20.0
    
    m_two = NirvanaTwoStageRegressor(base_estimator="ridge", learning_rate=0.3, max_depth=5, cancer_sensitivity=1.2)
    m_two.fit(X_tr.iloc[fit_idx], y_fit, X_cal=X_tr.iloc[cal_idx], y_cal=y_cal)
    pred_t = m_two.predict(X_te)
    _, lo_t, hi_t = m_two.predict_interval(X_te, alpha=0.10)
    loss_two = (hi_t - lo_t) + np.maximum(0.0, lo_t - y_te.values)*20.0 + np.maximum(0.0, y_te.values - hi_t)*20.0
    
    diff = loss_ridge - loss_two
    d_bar = float(np.mean(diff))
    print(f"Sample observed mean differential d_bar: {d_bar:.6f}")
    
    # Exact Centered Moving-Block Bootstrap Under H0 (mean d <= 0)
    diff_centered = diff - d_bar
    B = 2000
    N = len(diff)
    
    for block_len in [5, 10, 20]:
        np.random.seed(42)
        n_blocks = int(np.ceil(N / block_len))
        boot_stats = []
        for _ in range(B):
            starts = np.random.randint(0, N - block_len + 1, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:N]
            boot_stats.append(np.mean(diff_centered[idx]))
        boot_stats = np.array(boot_stats)
        
        # Monte Carlo p-value with finite-sample continuity correction: (count + 1) / (B + 1)
        count_exceed = np.sum(boot_stats >= d_bar)
        p_val_corrected = float((count_exceed + 1.0) / (B + 1.0))
        
        boot_uncentered = boot_stats + d_bar
        ci_lo, ci_hi = np.percentile(boot_uncentered, [2.5, 97.5])
        
        print(f"Block Length L={block_len:2d}: 95% CI = [{ci_lo:.5f}, {ci_hi:.5f}] | Exceedances: {count_exceed}/{B} | Centered p-value = {p_val_corrected:.4f}")

def run_residual_ablation(df_core, core_banks):
    print("\n" + "=" * 100)
    print("STEP 3: 2x2 RESIDUAL LEARNING & FISHER PRUNING ABLATION (VCB TEST SET 2024-2025)")
    print("=" * 100)
    
    target_bank = "VCB"
    X_base, X_kalman, _, y_series = build_features(df_core, target_bank, core_banks)
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
    
    # Pre-generate 5-Fold Walk-Forward OOF residuals
    tscv = TimeSeriesSplit(n_splits=5)
    oof_residuals = np.zeros(len(y_fit))
    oof_mask = np.zeros(len(y_fit), dtype=bool)
    for tr_split, val_split in tscv.split(X_fit):
        m_split = Ridge(alpha=10.0).fit(X_fit.iloc[tr_split], y_fit.iloc[tr_split])
        oof_residuals[val_split] = y_fit.iloc[val_split].values - m_split.predict(X_fit.iloc[val_split])
        oof_mask[val_split] = True
        
    def evaluate_variant(res_type, sens):
        if res_type == "In-Sample":
            m = NirvanaTwoStageRegressor(base_estimator="ridge", learning_rate=0.3, max_depth=5, cancer_sensitivity=sens)
            m.fit(X_fit, y_fit, X_cal=X_cal, y_cal=y_cal)
            pred = m.predict(X_te)
            _, lo, hi = m.predict_interval(X_te, alpha=0.10)
        else: # Walk-Forward OOF
            base_full = Ridge(alpha=10.0).fit(X_fit, y_fit)
            tree = NirvanaTreeRegressor(max_depth=5, cancer_sensitivity=sens)
            tree.fit(X_fit.iloc[oof_mask].values, oof_residuals[oof_mask])
            pred = base_full.predict(X_te) + 0.3 * tree.predict(X_te.values)
            pred_cal = base_full.predict(X_cal) + 0.3 * tree.predict(X_cal.values)
            cal_res = np.abs(y_cal.values - pred_cal)
            k = int(np.ceil((len(cal_res) + 1) * 0.90))
            q = np.sort(cal_res)[min(k - 1, len(cal_res) - 1)]
            lo, hi = pred - q, pred + q
            
        mae = mean_absolute_error(y_te, pred)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        r2 = r2_score(y_te, pred)
        cov = np.mean((y_te.values >= lo) & (y_te.values <= hi)) * 100.0
        w = winkler_score(y_te.values, lo, hi)
        return mae, rmse, r2, cov, w
        
    grid = [
        ("Config A (In-Sample, Pruning ON: sens=1.2)", "In-Sample", 1.2),
        ("Config B (In-Sample, Pruning OFF: sens=0.0)", "In-Sample", 0.0),
        ("Config C (Walk-Forward OOF, Pruning ON: sens=1.2)", "OOF", 1.2),
        ("Config D (Walk-Forward OOF, Pruning OFF: sens=0.0)", "OOF", 0.0),
    ]
    
    ablation_res = []
    for name, rtype, sens in grid:
        mae, rmse, r2, cov, w = evaluate_variant(rtype, sens)
        ablation_res.append({
            "Ablation Configuration": name,
            "MAE": mae,
            "RMSE": rmse,
            "OOS_R2": r2,
            "Coverage_90%": cov,
            "Winkler_Score": w
        })
    df_ab = pd.DataFrame(ablation_res)
    print(df_ab.to_string(index=False))

if __name__ == "__main__":
    df_core, core_banks = run_multi_bank_benchmark()
    run_hypothesis_testing(df_core, core_banks)
    run_residual_ablation(df_core, core_banks)
    print("\n" + "=" * 100)
    print("ALL AUDIT REPRODUCTION PIPELINE FINISHED SUCCESSFULLY WITH ZERO ERRORS.")
    print("=" * 100)
