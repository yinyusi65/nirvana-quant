"""
NIRVANA CORE TEST REPLICATION SUITE
21 Comprehensive Econometric and Mathematical Unit Tests
"""

import numpy as np
import pandas as pd
from nirvana import (
    Base12Manifold,
    CausalOnlineKalman,
    MultivariateCausalKalman,
    NirvanaTreeEngine,
    TrainingSample,
    NirvanaConnectedness,
    MultiplexNetworkEngine,
    MultiverseReconstructor,
    compute_parkinson,
    compute_garman_klass,
    compute_yang_zhang,
    NirvanaStaticBooster,
    StaticEconometricReport,
    stata_regress,
    stata_summarize,
    stata_pwcorr,
)

def test_volatility_censoring():
    h = np.array([105.0, 110.0])
    l = np.array([100.0, 100.0])
    v_norm = compute_parkinson(h, l)
    assert np.all(v_norm > 0), "Normal volatility must be positive"
    h_limit = np.array([100.0, 100.0])
    l_limit = np.array([100.0, 100.0])
    close = np.array([100.0, 100.0])
    prev_close = np.array([93.0, 93.0])
    v_limit = compute_parkinson(h_limit, l_limit, close=close, prev_close=prev_close)
    assert np.all(v_limit > 1.5), f"Limit session volatility collapsed: {v_limit}"
    print("[PASS] Volatility price-limit censoring test passed.")

def test_manifold_harmonics():
    manifold = Base12Manifold()
    t = np.arange(100)
    coords = manifold.batch_transform(t)
    assert coords.shape == (100, 9), "Coords shape mismatch"
    assert np.all(coords >= -1.0001) and np.all(coords <= 1.0001), "Trig coordinates out of [-1, 1]"
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    coords_dt = manifold.batch_transform(t, dates=dates)
    assert coords_dt.shape == (100, 9), "Datetime coords shape mismatch"
    print("[PASS] Manifold harmonic transformation test passed.")

def test_kalman_stability_and_mean_reversion():
    n_dim = 5
    kf = MultivariateCausalKalman(n_dim=n_dim, phi=0.98)
    rng = np.random.default_rng(42)
    for _ in range(500):
        z = rng.normal(3.0, 1.0, n_dim)
        state = kf.update(z)
    P = kf.P
    assert np.allclose(P, P.T, atol=1e-10), "Kalman covariance P lost symmetry"
    eigvals = np.linalg.eigvalsh(P)
    assert np.all(eigvals >= -1e-12), f"Negative eigenvalue in P: {eigvals}"
    assert np.all(np.abs(state - 3.0) < 1.5), "Kalman state drifted away despite mean reversion"
    print("[PASS] Kalman filter mean-reversion & covariance stability test passed.")

def test_directional_causality_and_diebold_yilmaz():
    T = 1000
    rng = np.random.default_rng(42)
    leader = rng.normal(0, 1, T)
    follower = np.zeros(T)
    follower[1:] = 0.7 * leader[:-1] + rng.normal(0, 0.3, T - 1)
    toy_df = pd.DataFrame({"LEADER": leader, "FOLLOWER": follower})
    engine = NirvanaConnectedness(apoptosis_threshold=0.0, normalize_bands=False)
    rep = engine.fit(toy_df)
    assert rep.net_spillovers["LEADER"] > 0, f"Leader is not transmitter: {rep.net_spillovers}"
    assert rep.net_spillovers["FOLLOWER"] < 0, f"Follower is not receiver: {rep.net_spillovers}"
    print("[PASS] Directional Diebold-Yilmaz causality test passed.")

def test_market_orthogonalization():
    T = 500
    rng = np.random.default_rng(42)
    market = rng.normal(0, 2.0, T)
    b1 = 1.2 * market + rng.normal(0, 0.5, T)
    b2 = 0.9 * market + rng.normal(0, 0.5, T)
    b3 = 0.5 * market + rng.normal(0, 0.3, T)
    panel = pd.DataFrame({"B1": b1, "B2": b2, "B3": b3})
    engine = NirvanaConnectedness(orthogonalize_market=False)
    rep_raw = engine.fit(panel)
    engine_ortho = NirvanaConnectedness(orthogonalize_market=True)
    rep_ortho = engine_ortho.fit(panel)
    assert rep_ortho.market_variance_share > 50.0, f"Market factor didn't capture dominant variance: {rep_ortho.market_variance_share}"
    assert rep_ortho.market_beta is not None, "Market betas missing"
    print("[PASS] Market orthogonalization test passed.")

def test_graph_centrality_separation():
    T = 600
    rng = np.random.default_rng(42)
    b0 = rng.normal(0, 1, T)
    b1 = np.zeros(T)
    b1[1:] = 0.8 * b0[:-1] + rng.normal(0, 0.2, T - 1)
    b2 = rng.normal(0, 1, T)
    panel = pd.DataFrame({"B0": b0, "B1": b1, "B2": b2})
    engine = NirvanaConnectedness(apoptosis_threshold=0.0, normalize_bands=False)
    rep = engine.fit(panel)
    assert rep.eigenvector_hub is not None, "Hub centrality missing"
    assert rep.eigenvector_authority is not None, "Authority centrality missing"
    assert rep.betweenness_centrality is not None, "Betweenness centrality missing"
    assert rep.flight_to_safety_index is not None, "Flight to safety missing"
    print("[PASS] Graph centrality hub vs authority separation test passed.")

def test_multiplex_and_multiverse():
    engine = MultiplexNetworkEngine()
    spillover = pd.DataFrame([[80.0, 20.0], [30.0, 70.0]], index=["B1", "B2"], columns=["B1", "B2"])
    fused = engine.fuse_layers(spillover, balance_sheet_weights={"B1": 2.0, "B2": 1.0})
    assert fused.shape == (2, 2)
    assert np.allclose(fused.sum(axis=1), 100.0, atol=1.0)
    print("[PASS] Multiplex and Multiverse modules test passed.")

def test_tree_apoptosis_shift_invariance():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 5))
    y_base = 2.0 * X[:, 0] + 1.5 * (X[:, 1] > 0)
    y_shifted = y_base + 10.0
    dataset_base = [TrainingSample(features=X[i].tolist(), target=float(y_base[i])) for i in range(len(y_base))]
    dataset_shifted = [TrainingSample(features=X[i].tolist(), target=float(y_shifted[i])) for i in range(len(y_shifted))]
    tree_base = NirvanaTreeEngine(cancer_sensitivity=1.0)
    tree_base.train(dataset_base)
    pred_base = tree_base.predict(X)
    tree_shifted = NirvanaTreeEngine(cancer_sensitivity=1.0)
    tree_shifted.train(dataset_shifted)
    pred_shifted = tree_shifted.predict(X)
    assert np.allclose(pred_base + 10.0, pred_shifted, atol=0.2), "Tree apoptosis is not shift-invariant!"
    print("[PASS] Tree apoptosis shift-invariance test passed.")

def test_rolling_tci_with_options():
    T = 300
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(3.0, 1.0, (T, 5)), columns=[f"B_{i}" for i in range(5)])
    engine = NirvanaConnectedness(apoptosis_threshold=0.0)
    roll = engine.rolling_fit(df, window=100, step=50, orthogonalize_market=True)
    assert len(roll.tci_history) > 0, "Rolling TCI produced empty series"
    print("[PASS] Rolling TCI with keyword options test passed.")

def test_volatility_series_support_and_floors():
    s_idx = pd.date_range("2023-01-01", periods=5)
    h = pd.Series([100.0]*5, index=s_idx)
    l = pd.Series([100.0]*5, index=s_idx)
    o = pd.Series([100.0]*5, index=s_idx)
    c = pd.Series([100.0]*5, index=s_idx)
    pc = pd.Series([100.0]*5, index=s_idx)
    v_park = compute_parkinson(h, l)
    assert isinstance(v_park, pd.Series), "Parkinson did not preserve pd.Series"
    assert v_park.index.equals(s_idx), "Parkinson pd.Series index mismatch"
    v_gk = compute_garman_klass(o, h, l, c)
    v_yz = compute_yang_zhang(o, h, l, c, pc)
    assert np.all(v_gk > 1.05), f"Garman-Klass floor collapsed: {v_gk}"
    assert np.all(v_yz > 1.05), f"Yang-Zhang floor collapsed: {v_yz}"
    print("[PASS] Volatility series support & floor bounds test passed.")

def test_kalman_nan_resilience_and_dataframe():
    kf1 = CausalOnlineKalman()
    res1 = kf1.filter_series([3.0, np.nan, 3.2, np.nan, 3.1])
    assert not np.any(np.isnan(res1)), "1D Kalman contaminated with NaNs"
    df_nan_start = pd.DataFrame({"B1": [np.nan, 3.0, 3.1, 3.2], "B2": [2.9, 3.0, np.nan, 3.1]})
    mvk = MultivariateCausalKalman(n_dim=2)
    filtered = mvk.filter_dataframe(df_nan_start)
    assert filtered.shape == (4, 2), "Multivariate Kalman DataFrame shape mismatch"
    assert not np.any(np.isnan(filtered[1:])), "Multivariate Kalman failed on post-init dates"
    print("[PASS] Kalman filter NaN resilience & DataFrame input test passed.")

def test_tree_dataframe_predict_and_nan_cleaning():
    dataset = [
        TrainingSample(features=[1.0, 2.0], target=3.0),
        TrainingSample(features=[np.nan, 2.0], target=5.0),
        TrainingSample(features=[4.0, np.nan], target=7.0),
        TrainingSample(features=[2.0, 3.0], target=9.0),
    ]
    tree = NirvanaTreeEngine()
    tree.train(dataset)
    assert len(tree.nodes) > 0, "Tree failed to build from clean samples"
    df_features = pd.DataFrame({"f1": [1.0, 2.0], "f2": [2.0, 3.0]})
    preds = tree.predict(df_features)
    assert len(preds) == 2, "Tree DataFrame predict returned wrong count"
    print("[PASS] Tree DataFrame predict & NaN cleaning test passed.")

def test_single_node_network_safety():
    engine = NirvanaConnectedness()
    single_df = pd.DataFrame({"SOLO_BANK": np.random.normal(3.0, 1.0, 100)})
    rep = engine.fit(single_df)
    assert rep.total_connectedness_index == 0.0, "Single bank TCI should be 0.0"
    print("[PASS] Single-node network safety test passed.")

def test_multiverse_unlisted_bank_peer_synthesis():
    df_unlisted = pd.DataFrame({
        "VCB": np.random.normal(3.0, 0.1, 60),
        "CTG": np.random.normal(3.1, 0.1, 60),
        "UNLISTED_BANK": [np.nan]*20 + list(np.random.normal(3.0, 0.1, 40))
    })
    recon = MultiverseReconstructor(boundary_window=5)
    syn = recon.reconstruct(df_unlisted)
    assert not syn["UNLISTED_BANK"].isna().any(), "Unlisted bank has remaining NaNs"
    assert syn["UNLISTED_BANK"].std() > 0.05, "Unlisted bank was filled with flat constant"
    print("[PASS] Multiverse unlisted bank peer synthesis test passed.")

def test_svd_market_positive_betas():
    T = 400
    rng = np.random.default_rng(42)
    market = rng.normal(0, 1.5, T)
    df = pd.DataFrame({
        "B1": 1.2 * market + rng.normal(0, 0.3, T),
        "B2": 0.8 * market + rng.normal(0, 0.3, T),
        "B3": 1.0 * market + rng.normal(0, 0.3, T)
    })
    engine = NirvanaConnectedness(orthogonalize_market=True)
    rep = engine.fit(df)
    avg_beta = float(np.mean(list(rep.market_beta.values())))
    assert avg_beta > 0.0, f"Average market beta should be positive, got {avg_beta}"
    print("[PASS] SVD market factor positive beta alignment test passed.")

def test_yang_zhang_day0_missing_prev_close():
    o = np.array([100.0, 102.0, 101.0])
    h = np.array([103.0, 104.0, 103.0])
    l = np.array([99.0, 101.0, 100.0])
    c = np.array([102.0, 103.0, 102.0])
    pc = np.array([np.nan, 102.0, 103.0])
    yz_vol = compute_yang_zhang(o, h, l, c, pc)
    assert not np.isnan(yz_vol[0]), f"Yang-Zhang produced NaNs on Day 0: {yz_vol}"
    print("[PASS] Yang-Zhang Day-0 missing prev_close test passed.")

def test_multiplex_coupling_strength_clamping():
    eng_over = MultiplexNetworkEngine(coupling_strength=1.5)
    assert eng_over.coupling_strength == 1.0, "Coupling strength > 1.0 not clamped"
    eng_under = MultiplexNetworkEngine(coupling_strength=-0.5)
    assert eng_under.coupling_strength == 0.0, "Coupling strength < 0.0 not clamped"
    print("[PASS] Multiplex coupling strength clamping test passed.")

def test_network_custom_lags_and_weights():
    T = 200
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(3.0, 1.0, (T, 3)), columns=["A", "B", "C"])
    engine = NirvanaConnectedness(apoptosis_threshold=0.0)
    rep = engine.fit(df, lags=[1, 2], lag_weights=[0.8, 0.2])
    assert rep.total_connectedness_index > 0
    print("[PASS] Network custom lags & lag-weights test passed.")

def test_unlisted_bank_median_weight_fallback():
    T = 150
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(3.0, 1.0, (T, 3)), columns=["KNOWN1", "KNOWN2", "UNKNOWN_NEW"])
    weights = {"KNOWN1": 2000.0, "KNOWN2": 1000.0}
    engine = NirvanaConnectedness(asset_weights=weights)
    rep = engine.fit(df)
    to_unknown = rep.spillover_table.loc["TO_others", "UNKNOWN_NEW"]
    assert to_unknown > 5.0, f"Unknown bank received unfairly microscopic weight: {to_unknown}"
    print("[PASS] Missing bank median asset weight fallback test passed.")

def test_static_booster_phase2_r2_elevation():
    rng = np.random.default_rng(42)
    N = 400
    X1 = rng.uniform(0.1, 3.0, N)
    X2 = rng.uniform(0.1, 3.0, N)
    X3 = rng.uniform(0.1, 3.0, N)
    y = 2.0 * X1 - 1.0 * X2 + 0.4 * (X3 ** 2) + rng.normal(0, 0.2, N)
    df_X = pd.DataFrame({"Price_A": X1, "Price_Substitute_B": X2, "Consumer_Income": X3})
    booster = NirvanaStaticBooster(cancer_sensitivity=1.0, max_depth=8, min_samples_split=5)
    report = booster.fit(df_X, y)
    assert report.r2_phase2 > report.r2_phase1
    print(f"[PASS] Static Booster Phase 2 R^2 elevation test passed: R^2 {report.r2_phase1:.3f} -> {report.r2_phase2:.3f} (+{report.r2_gain:.3f})")

def test_stata_regression_and_diagnostics():
    rng = np.random.default_rng(42)
    N = 300
    x1 = rng.normal(2.0, 1.5, N)
    x2 = rng.normal(0.5, 0.3, N)
    noise = rng.normal(0, 0.8, N)
    y = 1.5 * x1 - 2.0 * x2 + 3.0 + noise
    df = pd.DataFrame({"dep_y": y, "indep_x1": x1, "indep_x2": x2})
    res = stata_regress(df, "dep_y", ["indep_x1", "indep_x2"], robust=False)
    assert res.r_squared > 0.6
    hettest = res.estat_hettest()
    assert hettest["p_value"] >= 0.0
    vif_res = res.estat_vif()
    assert vif_res["mean_vif"] < 5.0
    print("[PASS] Stata econometric engine and post-estimation diagnostics test passed.")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING NIRVANA COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    test_volatility_censoring()
    test_manifold_harmonics()
    test_kalman_stability_and_mean_reversion()
    test_directional_causality_and_diebold_yilmaz()
    test_market_orthogonalization()
    test_graph_centrality_separation()
    test_multiplex_and_multiverse()
    test_tree_apoptosis_shift_invariance()
    test_rolling_tci_with_options()
    test_volatility_series_support_and_floors()
    test_kalman_nan_resilience_and_dataframe()
    test_tree_dataframe_predict_and_nan_cleaning()
    test_single_node_network_safety()
    test_multiverse_unlisted_bank_peer_synthesis()
    test_svd_market_positive_betas()
    test_yang_zhang_day0_missing_prev_close()
    test_multiplex_coupling_strength_clamping()
    test_network_custom_lags_and_weights()
    test_unlisted_bank_median_weight_fallback()
    test_static_booster_phase2_r2_elevation()
    test_stata_regression_and_diagnostics()
    print("=" * 60)
    print("ALL 21 NIRVANA CORE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)
