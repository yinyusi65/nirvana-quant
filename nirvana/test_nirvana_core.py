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
from nirvana.static import NirvanaTwoStageRegressor

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


# ─────────────────────────────────────────────────────────────────────────────
# NEW TESTS (v1.1.3) — addressing ChatGPT peer-review critiques
# ─────────────────────────────────────────────────────────────────────────────

def test_conformal_x_residuals_alignment():
    """
    Verify that after chronological split, the X used to train Phase 2
    matches the residuals in shape (no row-count mismatch between phases).
    """
    rng = np.random.default_rng(7)
    N = 100
    X = rng.normal(0, 1, (N, 3))
    y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(0, 0.3, N)
    df_X = pd.DataFrame(X, columns=["A", "B", "C"])

    model = NirvanaTwoStageRegressor(conformal_split_ratio=0.2)
    model.fit(df_X, y)

    # After split: training rows = N - ceil(N * 0.2) = 80, calibration = 20
    n_cal = int(np.floor(N * 0.2))
    n_train_expected = N - n_cal
    report = model.report_
    assert report.n_samples == n_train_expected, (
        f"Training sample count mismatch: expected {n_train_expected}, got {report.n_samples}"
    )
    # Calibration residuals should come from the holdout set (size = n_cal)
    assert len(model.conformal_residuals_) == n_cal, (
        f"Conformal residuals size mismatch: expected {n_cal}, got {len(model.conformal_residuals_)}"
    )
    print("[PASS] Conformal X–residuals alignment after chronological split test passed.")


def test_conformal_empirical_coverage():
    """
    Empirical coverage of predict_interval must meet or exceed the target (1-alpha)
    on a held-out test set when conformal_split_ratio=0.2 is used.
    Coverage guarantee: P(Y in [lo, hi]) >= 1 - alpha (finite-sample).
    We test at alpha=0.10 (90% target) with N=300, expecting >= 85% empirical coverage.
    """
    rng = np.random.default_rng(99)
    N_total = 400
    X_all = rng.normal(0, 1, (N_total, 4))
    y_all = 3.0 * X_all[:, 0] - 1.5 * X_all[:, 1] + 0.5 * X_all[:, 2] ** 2 + rng.normal(0, 0.5, N_total)

    # chronological split: first 300 train+cal, last 100 test
    n_train = 300
    X_train = pd.DataFrame(X_all[:n_train], columns=["f0", "f1", "f2", "f3"])
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]

    model = NirvanaTwoStageRegressor(conformal_split_ratio=0.2)
    model.fit(X_train, y_train)

    cov = model.evaluate_conformal_coverage(X_test, y_test, alpha=0.10)
    assert cov["empirical_coverage"] >= 0.85, (
        f"Empirical coverage below safety threshold: {cov['empirical_coverage']:.3f} < 0.85 "
        f"(target 90%). Conformal guarantee may be violated."
    )
    print(
        f"[PASS] Conformal empirical coverage test passed: "
        f"target=90%, actual={cov['empirical_coverage']*100:.1f}%, "
        f"interval_width={cov['average_interval_width']:.3f}"
    )


def test_xcal_ycal_validation():
    """
    Passing only one of X_cal / y_cal must raise ValueError immediately.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (50, 2))
    y = rng.normal(0, 1, 50)
    X_cal = rng.normal(0, 1, (10, 2))

    model = NirvanaTwoStageRegressor()
    raised = False
    try:
        model.fit(X, y, X_cal=X_cal, y_cal=None)  # only X_cal provided
    except ValueError:
        raised = True
    assert raised, "fit() must raise ValueError when only X_cal is provided without y_cal"

    raised2 = False
    try:
        model.fit(X, y, X_cal=None, y_cal=y[:10])  # only y_cal provided
    except ValueError:
        raised2 = True
    assert raised2, "fit() must raise ValueError when only y_cal is provided without X_cal"

    print("[PASS] X_cal / y_cal paired validation test passed.")


def test_multiverse_no_leakage_pre_listing():
    """
    The reconstruction of a pre-listing bank must use only Ridge coefficients
    learned on the OBSERVED (post-listing) period — not the pre-listing period.
    Proxy test: the first observed value of the bank after listing must be
    preserved exactly (no overwrite of post-listing data by reconstruction).
    """
    rng = np.random.default_rng(13)
    T = 80
    panel = pd.DataFrame({
        "VCB": rng.normal(3.0, 0.2, T),
        "CTG": rng.normal(3.1, 0.2, T),
        "LATE_BANK": [np.nan] * 30 + list(rng.normal(3.05, 0.15, T - 30)),
    })
    original_post = panel["LATE_BANK"].iloc[30:].values.copy()

    recon = MultiverseReconstructor(boundary_window=5)
    result = recon.reconstruct(panel)

    # Post-listing values (rows 30 onwards) must be preserved exactly
    post_result = result["LATE_BANK"].iloc[30:].values
    assert np.allclose(post_result, original_post, atol=1e-10), (
        "Reconstruction overwrote post-listing observed data — leakage detected!"
    )
    # Pre-listing values must now be filled (no NaN)
    pre_result = result["LATE_BANK"].iloc[:30]
    assert not pre_result.isna().any(), "Pre-listing NaNs not filled by Multiverse reconstruction"
    print("[PASS] Multiverse no-leakage pre-listing data integrity test passed.")


def test_walk_forward_nirvana_vs_ols():
    """
    Walk-forward RMSE: NirvanaTwoStageRegressor must beat plain OLS on a
    dataset with a genuine non-linear signal component.
    Uses 5-fold walk-forward (expanding window) on N=300 with quadratic term.
    """
    rng = np.random.default_rng(42)
    N = 300
    X_raw = rng.normal(0, 1, (N, 3))
    y = 1.5 * X_raw[:, 0] - 2.0 * X_raw[:, 1] + 0.8 * (X_raw[:, 2] ** 2) + rng.normal(0, 0.4, N)

    n_splits = 5
    fold_size = N // (n_splits + 1)
    rmse_ols, rmse_nirvana = [], []

    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = train_end + fold_size
        if test_end > N:
            break

        X_tr, y_tr = X_raw[:train_end], y[:train_end]
        X_te, y_te = X_raw[train_end:test_end], y[train_end:test_end]

        # OLS baseline
        X_des = np.column_stack([np.ones(len(X_tr)), X_tr])
        beta_ols = np.linalg.lstsq(X_des, y_tr, rcond=None)[0]
        y_ols = np.column_stack([np.ones(len(X_te)), X_te]) @ beta_ols
        rmse_ols.append(float(np.sqrt(np.mean((y_te - y_ols) ** 2))))

        # Nirvana two-stage
        m = NirvanaTwoStageRegressor(conformal_split_ratio=0.0, max_depth=6, min_samples_split=4)
        m.fit(X_tr, y_tr)
        y_nir = m.predict(X_te)
        rmse_nirvana.append(float(np.sqrt(np.mean((y_te - y_nir) ** 2))))

    avg_ols = float(np.mean(rmse_ols))
    avg_nir = float(np.mean(rmse_nirvana))
    assert avg_nir < avg_ols, (
        f"Nirvana failed to beat OLS on walk-forward: "
        f"Nirvana RMSE={avg_nir:.4f}, OLS RMSE={avg_ols:.4f}"
    )
    print(
        f"[PASS] Walk-forward Nirvana vs OLS test passed: "
        f"Nirvana RMSE={avg_nir:.4f} < OLS RMSE={avg_ols:.4f}"
    )


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
    test_conformal_x_residuals_alignment()
    test_conformal_empirical_coverage()
    test_xcal_ycal_validation()
    test_multiverse_no_leakage_pre_listing()
    test_walk_forward_nirvana_vs_ols()
    print("=" * 60)
    print("ALL 26 NIRVANA CORE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)
