"""
Nirvana Core Quantitative Test Suite
====================================
Ponytail runnable self-check with pure assert statements.
Validates all 7 academic fixes and mathematical invariants.
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
    # 1. Normal session
    h = np.array([105.0, 110.0])
    l = np.array([100.0, 100.0])
    v_norm = compute_parkinson(h, l)
    assert np.all(v_norm > 0), "Normal volatility must be positive"

    # 2. Limit-hit session (H == L but price jumped 7% limit down)
    h_limit = np.array([100.0])
    l_limit = np.array([100.0])
    close = np.array([93.0])
    prev_close = np.array([100.0])
    v_limit = compute_parkinson(h_limit, l_limit, close=close, prev_close=prev_close)
    # Volatility must not collapse into negative log values
    assert v_limit[0] > 1.5, f"Limit session volatility collapsed: {v_limit[0]}"
    print("[PASS] Volatility price-limit censoring test passed.")


def test_manifold_harmonics():
    manifold = Base12Manifold()
    t = np.arange(100)
    coords = manifold.batch_transform(t)
    assert coords.shape == (100, 9), "Coords shape mismatch"
    assert np.all(coords >= -1.0001) and np.all(coords <= 1.0001), "Trig coordinates out of [-1, 1]"

    # Test with DatetimeIndex
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    coords_dt = manifold.batch_transform(t, dates=dates)
    assert coords_dt.shape == (100, 9), "Datetime coords shape mismatch"
    print("[PASS] Manifold harmonic transformation test passed.")


def test_kalman_stability_and_mean_reversion():
    n_dim = 5
    kf = MultivariateCausalKalman(n_dim=n_dim, phi=0.98)
    np.random.seed(42)

    # Simulate 500 steps
    for _ in range(500):
        z = np.random.randn(n_dim) + 3.0
        state = kf.update(z)
        # Check symmetry
        assert np.allclose(kf.P, kf.P.T, atol=1e-10), "Kalman covariance P lost symmetry"
        # Check positive semi-definiteness
        eigvals = np.linalg.eigvalsh(kf.P)
        assert np.all(eigvals >= -1e-12), f"Negative eigenvalue in P: {np.min(eigvals)}"

    # Check mean reversion kept state near 3.0
    assert np.all(np.abs(state - 3.0) < 1.5), "Kalman state drifted away despite mean reversion"
    print("[PASS] Kalman filter mean-reversion & covariance stability test passed.")


def test_directional_causality_and_diebold_yilmaz():
    # Synthetic network: Bank 0 (Leader) drives Bank 1 (Follower) with lag 1
    T = 1000
    np.random.seed(42)
    leader = np.random.randn(T)
    follower = 0.7 * np.roll(leader, 1) + 0.3 * np.random.randn(T)
    toy_df = pd.DataFrame({"LEADER": leader, "FOLLOWER": follower})

    engine = NirvanaConnectedness(apoptosis_threshold=0.0)
    rep = engine.fit(toy_df, normalize_bands=False)

    # Leader must be NET Transmitter (> 0), Follower must be NET Receiver (< 0)
    assert rep.net_spillovers["LEADER"] > 0, f"Leader is not transmitter: {rep.net_spillovers['LEADER']}"
    assert rep.net_spillovers["FOLLOWER"] < 0, f"Follower is not receiver: {rep.net_spillovers['FOLLOWER']}"
    assert rep.top_transmitters[0] == "LEADER"
    assert rep.top_receivers[0] == "FOLLOWER"

    # Row sums of spillover matrix must equal 100%
    sub_table = rep.spillover_table.loc[["LEADER", "FOLLOWER"], ["LEADER", "FOLLOWER"]]
    row_sums = sub_table.sum(axis=1).values
    assert np.allclose(row_sums, [100.0, 100.0], atol=1e-1), f"Row sums not 100%: {row_sums}"
    print("[PASS] Directional Diebold-Yilmaz causality test passed.")


def test_market_orthogonalization():
    T = 500
    np.random.seed(42)
    market = np.random.randn(T) * 2.0
    b1 = 1.2 * market + np.random.randn(T) * 0.5
    b2 = 0.9 * market + np.random.randn(T) * 0.5
    b3 = 0.3 * market + np.random.randn(T) * 0.5
    panel = pd.DataFrame({"B1": b1, "B2": b2, "B3": b3})

    engine = NirvanaConnectedness()
    rep_raw = engine.fit(panel, orthogonalize_market=False)
    rep_ortho = engine.fit(panel, orthogonalize_market=True)

    assert rep_ortho.market_variance_share > 50.0, f"Market factor didn't capture dominant variance: {rep_ortho.market_variance_share}"
    assert rep_ortho.market_beta is not None, "Market betas missing"
    assert rep_ortho.total_connectedness_index < rep_raw.total_connectedness_index, "Orthogonalized TCI should be lower than raw TCI"
    print("[PASS] Market orthogonalization test passed.")


def test_graph_centrality_separation():
    T = 600
    np.random.seed(42)
    b0 = np.random.randn(T)
    b1 = 0.8 * np.roll(b0, 1) + 0.2 * np.random.randn(T)
    b2 = 0.8 * np.roll(b1, 1) + 0.2 * np.random.randn(T)
    panel = pd.DataFrame({"B0": b0, "B1": b1, "B2": b2})

    engine = NirvanaConnectedness(apoptosis_threshold=0.0)
    rep = engine.fit(panel, normalize_bands=False)

    assert rep.eigenvector_hub is not None, "Hub centrality missing"
    assert rep.eigenvector_authority is not None, "Authority centrality missing"
    assert rep.betweenness_centrality is not None, "Betweenness centrality missing"
    assert rep.flight_to_safety_index is not None, "Flight to safety missing"

    # B0 transmits, so B0 should have high hub centrality
    assert rep.eigenvector_hub["B0"] > rep.eigenvector_hub["B2"], "B0 should have higher hub score than B2"
    # B2 receives, so B2 should have high authority centrality
    assert rep.eigenvector_authority["B2"] > rep.eigenvector_authority["B0"], "B2 should have higher authority score than B0"
    print("[PASS] Graph centrality hub vs authority separation test passed.")


def test_multiplex_and_multiverse():
    # Multiplex
    engine = MultiplexNetworkEngine()
    spillover = pd.DataFrame(
        [[80.0, 20.0], [30.0, 70.0]],
        index=["B1", "B2"],
        columns=["B1", "B2"]
    )
    fused = engine.fuse_layers(spillover, balance_sheet_weights={"B1": 2.0, "B2": 1.0})
    assert fused.shape == (2, 2)
    assert np.allclose(fused.sum(axis=1).values, [100.0, 100.0], atol=1e-1)

    # Multiverse
    raw = pd.DataFrame({
        "B1": [np.nan, np.nan, 3.2, 3.5, 3.4] * 20,
        "B2": [3.1, 3.3, 3.2, 3.4, 3.3] * 20,
        "VCB": [3.0, 3.1, 3.2, 3.0, 3.1] * 20,
        "CTG": [3.1, 3.0, 3.2, 3.1, 3.2] * 20,
        "BID": [3.2, 3.1, 3.1, 3.2, 3.3] * 20,
    })
    recon = MultiverseReconstructor(boundary_window=5)
    # Interbank defensive reindex
    ib_sparse = pd.DataFrame([[100.0]], index=["B1"], columns=["B1"])
    fused_sparse = engine.fuse_layers(spillover, interbank_exposures=ib_sparse)
    assert fused_sparse.shape == (2, 2), "Failed defensive interbank reindexing"

    # Multiverse with price-level market index
    mkt_price = pd.Series(1200.0 + np.cumsum(np.random.randn(len(raw))), index=raw.index)
    syn_mkt = recon.reconstruct(raw, market_index=mkt_price)
    assert syn_mkt.isna().sum().sum() == 0, "Synthesized panel with price index has NaNs"
    print("[PASS] Multiplex and Multiverse modules test passed.")


def test_tree_apoptosis_shift_invariance():
    # Verify that target shift y -> y + 10.0 preserves tree behavior without spurious apoptosis pruning
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y_base = 2.0 * X[:, 0] - 1.5 * X[:, 1]
    y_shifted = y_base + 10.0

    dataset_base = [TrainingSample(features=X[i].tolist(), target=float(y_base[i])) for i in range(len(X))]
    dataset_shifted = [TrainingSample(features=X[i].tolist(), target=float(y_shifted[i])) for i in range(len(X))]

    tree_base = NirvanaTreeEngine(cancer_sensitivity=1.0)
    tree_base.train(dataset_base)
    pred_base = tree_base.predict(X)

    tree_shifted = NirvanaTreeEngine(cancer_sensitivity=1.0)
    tree_shifted.train(dataset_shifted)
    pred_shifted = tree_shifted.predict(X)

    # Predictions must be invariant to shift (pred_shifted == pred_base + 10.0)
    assert np.allclose(pred_shifted, pred_base + 10.0, atol=0.2), "Tree apoptosis is not shift-invariant!"
    print("[PASS] Tree apoptosis shift-invariance test passed.")


def test_rolling_tci_with_options():
    T = 300
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(T, 5) + 3.0, columns=[f"B_{i}" for i in range(5)])
    engine = NirvanaConnectedness(apoptosis_threshold=0.0)

    # Test rolling TCI with market orthogonalization
    roll = engine.rolling_tci(df, window=100, step=50, orthogonalize_market=True)
    assert len(roll) > 0, "Rolling TCI produced empty series"
    assert not roll.isna().any(), "Rolling TCI contains NaNs"
    print("[PASS] Rolling TCI with keyword options test passed.")


def test_volatility_series_support_and_floors():
    s_idx = pd.date_range("2023-01-01", periods=5)
    h = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=s_idx)
    l = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=s_idx)
    o = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=s_idx)
    c = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=s_idx)
    pc = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=s_idx)

    # Series return type test
    v_park = compute_parkinson(h, l)
    assert isinstance(v_park, pd.Series), "Parkinson did not preserve pd.Series"
    assert v_park.index.equals(s_idx), "Parkinson pd.Series index mismatch"

    # Floor test: all estimators must not collapse below ~1.06 (log(100*sqrt(252*1e-4)))
    v_gk = compute_garman_klass(o, h, l, c)
    v_yz = compute_yang_zhang(o, h, l, c, pc)
    assert np.all(v_gk >= 1.05), f"Garman-Klass floor collapsed: {v_gk.values}"
    assert np.all(v_yz >= 1.05), f"Yang-Zhang floor collapsed: {v_yz.values}"
    print("[PASS] Volatility series support & floor bounds test passed.")


def test_kalman_nan_resilience_and_dataframe():
    # 1D Kalman NaN resilience
    kf1 = CausalOnlineKalman()
    res1 = kf1.filter_series([3.0, np.nan, 3.2, np.nan, 3.1])
    assert not np.isnan(res1).any(), "1D Kalman contaminated with NaNs"

    # Multivariate Kalman all-NaN initial day & DataFrame input
    df_nan_start = pd.DataFrame({
        "B1": [np.nan, 3.0, 3.1, 3.2],
        "B2": [np.nan, 2.9, 3.0, 3.1],
    })
    mvk = MultivariateCausalKalman(n_dim=2)
    filtered = mvk.filter_dataframe(df_nan_start)
    assert filtered.shape == (4, 2), "Multivariate Kalman DataFrame shape mismatch"
    assert not np.isnan(filtered[1:]).any(), "Multivariate Kalman failed on post-init dates"
    print("[PASS] Kalman filter NaN resilience & DataFrame input test passed.")


def test_tree_dataframe_predict_and_nan_cleaning():
    dataset = [
        TrainingSample([1.0, 2.0], 3.0),
        TrainingSample([np.nan, 2.0], 3.0),  # dirty sample
        TrainingSample([1.0, 2.0], np.nan),  # dirty sample
        TrainingSample([2.0, 3.0], 5.0),
        TrainingSample([3.0, 4.0], 7.0),
        TrainingSample([4.0, 5.0], 9.0),
    ]
    tree = NirvanaTreeEngine()
    tree.train(dataset)
    assert len(tree.nodes) > 0, "Tree failed to build from clean samples"

    # Test DataFrame predict
    df_features = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["f1", "f2"])
    preds = tree.predict(df_features)
    assert len(preds) == 2, "Tree DataFrame predict returned wrong count"
    assert not np.isnan(preds).any(), "Tree predict returned NaNs"
    print("[PASS] Tree DataFrame predict & NaN cleaning test passed.")


def test_single_node_network_safety():
    engine = NirvanaConnectedness()
    single_df = pd.DataFrame({"SOLO_BANK": np.random.randn(100) + 3.0})
    rep = engine.fit(single_df)
    assert rep.total_connectedness_index == 0.0, "Single bank TCI should be 0.0"
    assert rep.normalized_tci == 0.0, "Single bank normalized TCI should be 0.0"
    print("[PASS] Single-node network safety test passed.")


def test_multiverse_unlisted_bank_peer_synthesis():
    df_unlisted = pd.DataFrame({
        "VCB": [3.0, 3.1, 3.2] * 20,
        "CTG": [3.1, 3.0, 3.2] * 20,
        "UNLISTED_BANK": [np.nan] * 60,  # Never listed
    })
    recon = MultiverseReconstructor(boundary_window=5)
    syn = recon.reconstruct(df_unlisted)
    assert syn["UNLISTED_BANK"].isna().sum() == 0, "Unlisted bank has remaining NaNs"
    assert np.std(syn["UNLISTED_BANK"]) > 0.05, "Unlisted bank was filled with flat constant"
    print("[PASS] Multiverse unlisted bank peer synthesis test passed.")


def test_svd_market_positive_betas():
    T = 400
    np.random.seed(42)
    market = np.random.randn(T) * 1.5
    df = pd.DataFrame({
        "B1": market + np.random.randn(T) * 0.3,
        "B2": market * 1.2 + np.random.randn(T) * 0.3,
        "B3": market * 0.8 + np.random.randn(T) * 0.3,
    })
    engine = NirvanaConnectedness()
    rep = engine.fit(df, orthogonalize_market=True)
    avg_beta = float(np.mean(list(rep.market_beta.values())))
    assert avg_beta > 0.0, f"Average market beta should be positive, got {avg_beta}"
    print("[PASS] SVD market factor positive beta alignment test passed.")


def test_yang_zhang_day0_missing_prev_close():
    o = np.array([100.0, 102.0, 101.0])
    h = np.array([103.0, 104.0, 103.0])
    l = np.array([99.0, 101.0, 100.0])
    c = np.array([102.0, 103.0, 102.0])
    pc = np.array([np.nan, 102.0, 103.0])  # Day 0 is NaN
    yz_vol = compute_yang_zhang(o, h, l, c, pc)
    assert not np.isnan(yz_vol).any(), f"Yang-Zhang produced NaNs on Day 0: {yz_vol}"
    print("[PASS] Yang-Zhang Day-0 missing prev_close test passed.")


def test_multiplex_coupling_strength_clamping():
    eng_over = MultiplexNetworkEngine(coupling_strength=1.5)
    assert eng_over.coupling_strength == 1.0, "Coupling strength > 1.0 not clamped"
    eng_under = MultiplexNetworkEngine(coupling_strength=-0.5)
    assert eng_under.coupling_strength == 0.0, "Coupling strength < 0.0 not clamped"
    print("[PASS] Multiplex coupling strength clamping test passed.")


def test_network_custom_lags_and_weights():
    T = 200
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(T, 3) + 3.0, columns=["A", "B", "C"])
    engine = NirvanaConnectedness(apoptosis_threshold=0.0)
    rep = engine.fit(df, lags=[1, 2], lag_weights=[0.8, 0.2])
    assert rep.total_connectedness_index > 0, "Custom lags failed to compute TCI"
    print("[PASS] Network custom lags & lag-weights test passed.")


def test_unlisted_bank_median_weight_fallback():
    T = 150
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(T, 3) + 3.0, columns=["KNOWN1", "KNOWN2", "UNKNOWN_NEW"])
    engine = NirvanaConnectedness()
    weights = {"KNOWN1": 2000.0, "KNOWN2": 1000.0}
    rep = engine.fit(df, asset_weights=weights)
    # UNKNOWN_NEW should not be crushed to 1/2000th of KNOWN1
    table = rep.spillover_table
    to_unknown = table.loc["TO_others", "UNKNOWN_NEW"]
    assert to_unknown > 5.0, f"Unknown bank received unfairly microscopic weight: {to_unknown}"
    print("[PASS] Missing bank median asset weight fallback test passed.")


def test_static_booster_phase2_r2_elevation():
    # Verify that NirvanaStaticBooster rescues weak linear models on static tabular data
    np.random.seed(42)
    N = 400
    X1 = np.random.randn(N)
    X2 = np.random.randn(N)
    X3 = np.random.randn(N)
    # Target has strong non-linear threshold interaction (substitute switching):
    y = 0.1 * X1 + np.where(X2 > 0.5, 3.0, 0.0) * np.where(X3 < -0.2, 2.0, -1.0) + np.random.randn(N) * 0.4

    df_X = pd.DataFrame({"Price_A": X1, "Price_Substitute_B": X2, "Consumer_Income": X3})

    booster = NirvanaStaticBooster(cancer_sensitivity=1.0, max_depth=8, apoptosis_start_depth=5)
    report = booster.fit(df_X, y)

    # 1. Phase 1 must be a weak model (R^2 < 0.25)
    assert report.r2_phase1 < 0.25, f"Baseline model was not weak: R^2 = {report.r2_phase1}"

    # 2. Phase 2 must significantly elevate R^2 (R^2 > 0.70)
    assert report.r2_phase2 > 0.70, f"Phase 2 failed to elevate R^2: R^2 = {report.r2_phase2}"
    assert report.r2_gain > 0.45, f"R^2 gain insufficient: +Delta R^2 = {report.r2_gain}"
    assert report.f_p_value < 0.01, f"F-test not statistically significant: p = {report.f_p_value}"

    # 3. Predict works as expected
    preds = booster.predict(df_X)
    assert len(preds) == N
    assert np.isclose(booster.score(df_X, y), report.r2_phase2)

    # 4. Out-of-sample CV confirms genuine generalizability without noise overfitting
    cv = booster.evaluate_cv(df_X, y, k_folds=4)
    assert cv["delta_r2_cv_mean"] > 0.35, f"CV Delta R^2 weak: {cv['delta_r2_cv_mean']}"
    print(f"[PASS] Static Booster Phase 2 R^2 elevation test passed: R^2 {report.r2_phase1:.3f} -> {report.r2_phase2:.3f} (+{report.r2_gain:.3f})")


def test_stata_regression_and_diagnostics():
    np.random.seed(42)
    N = 300
    x1 = np.random.randn(N) * 2.0
    x2 = np.random.randn(N) * 1.5
    # True DGP: y = 2.0 + 1.5 * x1 - 0.8 * x2 + heteroskedastic noise
    noise = np.random.randn(N) * (0.5 + 0.3 * np.abs(x1))
    y = 2.0 + 1.5 * x1 - 0.8 * x2 + noise

    df = pd.DataFrame({"dep_y": y, "indep_x1": x1, "indep_x2": x2})

    # 1. Stata regress
    res = stata_regress(df, y="dep_y", x=["indep_x1", "indep_x2"], robust=False)
    assert res.n_obs == N
    assert res.df_model == 2
    assert res.df_resid == N - 3
    assert res.r_squared > 0.70
    assert np.isclose(res.coefficients["indep_x1"], 1.5, atol=0.2)
    assert np.isclose(res.coefficients["indep_x2"], -0.8, atol=0.2)

    # 2. Summary text output
    summary_txt = res.summary()
    assert "Number of obs   =" in summary_txt
    assert "Adj R-squared" in summary_txt
    assert "Root MSE" in summary_txt

    # 3. Robust OLS
    res_rob = stata_regress(df, y="dep_y", x=["indep_x1", "indep_x2"], robust=True)
    assert res_rob.is_robust
    assert res_rob.std_errors["indep_x1"] > 0.0

    # 4. Breusch-Pagan hettest
    hettest = res.estat_hettest()
    assert hettest["chi2"] >= 0.0
    assert "Breusch-Pagan" in hettest["display"]

    # 5. VIF test
    vif_res = res.estat_vif(df)
    assert vif_res["mean_vif"] >= 1.0
    assert len(vif_res["vif_table"]) == 2

    # 6. Ramsey RESET ovtest
    ovtest = res.estat_ovtest(df)
    assert ovtest["df_num"] == 3
    assert "Ramsey RESET" in ovtest["display"]

    # 7. Durbin-Watson
    dw = res.estat_dwatson()
    assert 0.0 < dw["d_stat"] < 4.0

    # 8. Normality sktest
    norm_res = res.sktest()
    assert norm_res["jb_stat"] >= 0.0

    # 9. Summarize and pwcorr
    sum_txt = stata_summarize(df)
    assert "indep_x1" in sum_txt
    detail_txt = stata_summarize(df, detail=True)
    assert "Percentiles" in detail_txt
    corr_txt = stata_pwcorr(df)
    assert "dep_y" in corr_txt
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

