#include <nirvana/nirvana.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>

using namespace nirvana;

// Lightweight test assertion helper
static int g_tests_passed = 0;
static int g_tests_total = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        g_tests_total++; \
        if (!(cond)) { \
            std::cerr << "[-] FAILED: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            std::abort(); \
        } else { \
            g_tests_passed++; \
        } \
    } while (0)

// ----------------------------------------------------------------------------
// 1. Matrix and Linear Algebra Tests
// ----------------------------------------------------------------------------
void test_matrix_basics() {
    std::cout << "[+] Running test_matrix_basics...\n";

    Matrix A(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix B(2, 2, {2.0, 0.0, 1.0, 2.0});

    // Matrix multiplication
    Matrix C = A * B;
    // [1*2 + 2*1, 1*0 + 2*2] = [4, 4]
    // [3*2 + 4*1, 3*0 + 4*2] = [10, 8]
    TEST_ASSERT(std::abs(C(0, 0) - 4.0) < 1e-9, "MatMul (0,0)");
    TEST_ASSERT(std::abs(C(0, 1) - 4.0) < 1e-9, "MatMul (0,1)");
    TEST_ASSERT(std::abs(C(1, 0) - 10.0) < 1e-9, "MatMul (1,0)");
    TEST_ASSERT(std::abs(C(1, 1) - 8.0) < 1e-9, "MatMul (1,1)");

    // Transpose
    Matrix At = A.transpose();
    TEST_ASSERT(At(0, 1) == 3.0 && At(1, 0) == 2.0, "Transpose");

    // Inversion
    Matrix A_inv = A.inverse(1e-8);
    Matrix Identity = A * A_inv;
    TEST_ASSERT(std::abs(Identity(0, 0) - 1.0) < 1e-5, "Inv (0,0)");
    TEST_ASSERT(std::abs(Identity(0, 1) - 0.0) < 1e-5, "Inv (0,1)");
    TEST_ASSERT(std::abs(Identity(1, 0) - 0.0) < 1e-5, "Inv (1,0)");
    TEST_ASSERT(std::abs(Identity(1, 1) - 1.0) < 1e-5, "Inv (1,1)");

    // Truncated SVD on Diagonal Matrix
    Matrix D(3, 3, 0.0);
    D(0, 0) = 5.0;
    D(1, 1) = 2.0;
    D(2, 2) = 0.5;
    Matrix U, V;
    std::vector<double> S;
    D.svd_truncated(2, U, S, V);
    TEST_ASSERT(S.size() == 2, "Truncated SVD top 2 components");
    TEST_ASSERT(std::abs(S[0] - 5.0) < 1e-2, "SVD singular value 1");
    TEST_ASSERT(std::abs(S[1] - 2.0) < 1e-2, "SVD singular value 2");
}

// ----------------------------------------------------------------------------
// 2. Base-12 Manifold & Dual Clock Coordinates Tests
// ----------------------------------------------------------------------------
void test_manifold() {
    std::cout << "[+] Running test_manifold...\n";

    Base12Manifold manifold(1000.0, {12.0, 144.0});
    
    // Transform single step at index 12 (1 full cycle of macro period 1)
    DualClockCoordinates coords = manifold.transform(12, 1000.0);
    // theta_m1 = 2 * pi * (12 % 12) / 12 = 0 -> sin ~ 0, cos ~ 1
    TEST_ASSERT(std::abs(coords.macro_sin_1) < 1e-6, "Manifold macro sin 1 at t=12");
    TEST_ASSERT(std::abs(coords.macro_cos_1 - 1.0) < 1e-6, "Manifold macro cos 1 at t=12");
    
    // Feature dimension vector
    std::vector<double> feat = coords.to_vector();
    TEST_ASSERT(feat.size() == 9, "Dual clock coordinate vector dimension (9)");

    // Batch transformation
    std::vector<int> cal_indices(100);
    for (int i = 0; i < 100; ++i) cal_indices[i] = i;
    Matrix feat_matrix = manifold.batch_transform(cal_indices);
    TEST_ASSERT(feat_matrix.rows() == 100 && feat_matrix.cols() == 9, "Batch manifold matrix shape");

    // DFT Dominant period detection
    std::vector<double> sine_signal(120);
    for (size_t t = 0; t < sine_signal.size(); ++t) {
        // Frequency of 1 cycle every 20 steps
        sine_signal[t] = std::sin(2.0 * M_PI * static_cast<double>(t) / 20.0);
    }
    auto [p1, p2] = Base12Manifold::estimate_dominant_periods(sine_signal);
    TEST_ASSERT(std::abs(p1 - 20.0) <= 2.0, "Dominant period estimation accuracy (~20.0)");
}

// ----------------------------------------------------------------------------
// 3. Multivariate Causal Kalman Filter with Huber Clipping
// ----------------------------------------------------------------------------
void test_kalman_filter() {
    std::cout << "[+] Running test_kalman_filter...\n";

    size_t dim = 3;
    MultivariateCausalKalman kf(dim, 1e-3, 1e-2, 0.98, 2.5);

    // Initial state check
    TEST_ASSERT(kf.state_.size() == dim, "Kalman state size");

    // Feed steady observations
    std::vector<double> obs = {1.0, 2.0, -1.0};
    for (int t = 0; t < 20; ++t) {
        kf.update(obs);
    }
    for (size_t i = 0; i < dim; ++i) {
        TEST_ASSERT(std::abs(kf.state_[i] - obs[i]) < 0.2, "Kalman convergence to observation");
    }

    // Fat-tail outlier test (Huber innovation clipping)
    std::vector<double> outlier = {1000.0, 2.0, -1.0};
    kf.update(outlier);
    // Outlier in dimension 0 should be bounded, not jumping to 1000
    TEST_ASSERT(kf.state_[0] < 50.0, "Huber outlier innovation clipping resistance");

    // Reset check
    kf.reset();
    TEST_ASSERT(std::abs(kf.state_[0]) < 1e-9, "Kalman reset state to zero");
}

// ----------------------------------------------------------------------------
// 4. Nirvana 12-Layer Tree Engine & Fisher Apoptosis Pruning
// ----------------------------------------------------------------------------
void test_tree_engine() {
    std::cout << "[+] Running test_tree_engine...\n";

    NirvanaTreeEngine tree(1.0, 0.5);

    // Synthetic linear relationship with 3 features: y = 2*x0 - 1.5*x1 + x2
    size_t n_samples = 150;
    std::vector<TrainingSample> dataset(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        double val = static_cast<double>(i) * 0.05;
        double x0 = std::sin(val);
        double x1 = std::cos(val);
        double x2 = val * 0.1;
        dataset[i].features = {x0, x1, x2};
        dataset[i].target = 2.0 * x0 - 1.5 * x1 + x2;
        dataset[i].weight = 1.0;
    }

    tree.train(dataset);
    TEST_ASSERT(!tree.nodes_.empty(), "Tree nodes populated");

    // Predict on sample
    double pred = tree.predict(dataset[10].features);
    double target = dataset[10].target;
    TEST_ASSERT(std::abs(pred - target) < 1.0, "Tree prediction close to actual target");

    // Check depth constraint
    for (const auto& node : tree.nodes_) {
        TEST_ASSERT(node.depth <= NirvanaTreeEngine::MAX_DEPTH, "Max depth within 12 layers");
    }
}

// ----------------------------------------------------------------------------
// 5. Nirvana Directional Connectedness, Lead-Lag & SVD Multi-Factor
// ----------------------------------------------------------------------------
void test_directional_connectedness() {
    std::cout << "[+] Running test_directional_connectedness...\n";

    std::vector<std::string> bank_names = {"VCB", "BID", "CTG", "TCB"};
    NirvanaConnectedness conn;
    conn.lags_ = {1};
    conn.lag_weights_ = {1.0};
    conn.orthogonalize_market_ = true;
    conn.n_market_factors_ = 1;

    // Generate synthetic time series with known lead-lag: VCB leads BID and CTG
    size_t T = 150;
    size_t N = bank_names.size();
    Matrix returns(T, N);
    
    for (size_t t = 1; t < T; ++t) {
        double shock = std::sin(static_cast<double>(t) * 0.1);
        returns(t, 0) = shock;                                   // VCB (leader)
        returns(t, 1) = 0.7 * returns(t - 1, 0) + 0.1 * shock;  // BID (lags VCB by 1)
        returns(t, 2) = 0.5 * returns(t - 1, 0) + 0.1 * shock;  // CTG (lags VCB by 1)
        returns(t, 3) = 0.2 * shock;                             // TCB
    }

    SystemicRiskReport report = conn.fit(returns, bank_names);
    TEST_ASSERT(report.node_names.size() == N, "Report node count");
    TEST_ASSERT(report.total_connectedness_index >= 0.0 && report.total_connectedness_index <= 100.0, "Total spillover index in [0, 100]");

    // Check net spillovers: VCB should be a net transmitter
    double net_vcb = report.net_spillovers["VCB"];
    double net_bid = report.net_spillovers["BID"];
    TEST_ASSERT(net_vcb > net_bid, "Directional lead-lag: Leader VCB net transmitter > receiver BID");

    // Hub and Authority centrality
    TEST_ASSERT(report.eigenvector_hub.count("VCB") > 0, "VCB hub score exists");
    TEST_ASSERT(report.eigenvector_authority.count("BID") > 0, "BID authority score exists");

    // Dynamic Rolling fit
    DynamicSystemicRiskReport dyn_report = conn.rolling_fit(returns, 80, 10, 2.0, bank_names);
    TEST_ASSERT(!dyn_report.tci_history.empty(), "Rolling fit produces TCI time series");
    TEST_ASSERT(dyn_report.mean_tci >= 0.0, "Mean rolling TCI is non-negative");
}

// ----------------------------------------------------------------------------
// 6. Multiverse Reconstructor & Hermite Blending
// ----------------------------------------------------------------------------
void test_multiverse_reconstructor() {
    std::cout << "[+] Running test_multiverse_reconstructor...\n";

    MultiverseReconstructor recon(3, 10, 2.5, 0.5, 1000.0);

    size_t T = 60;
    size_t N = 3;
    Matrix panel(T, N, 0.0);

    // Node 0 & 1 are observed for all T
    for (size_t t = 0; t < T; ++t) {
        panel(t, 0) = 1.0 + 0.05 * t;
        panel(t, 1) = 0.8 + 0.04 * t;
    }

    // Node 2 has missing pre-listing data for first 20 days (NaN)
    for (size_t t = 0; t < 20; ++t) {
        panel(t, 2) = std::numeric_limits<double>::quiet_NaN();
    }
    for (size_t t = 20; t < T; ++t) {
        panel(t, 2) = 1.2 + 0.06 * t;
    }

    Matrix reconstructed = recon.reconstruct(panel);
    TEST_ASSERT(reconstructed.rows() == T && reconstructed.cols() == N, "Reconstructed panel dimensions");

    // Check that pre-listing NaNs are synthesized
    for (size_t t = 0; t < 20; ++t) {
        TEST_ASSERT(!std::isnan(reconstructed(t, 2)), "Pre-listing NaN replaced by synthesized values");
    }

    // Check that post-listing observed values are preserved
    for (size_t t = 20; t < T; ++t) {
        TEST_ASSERT(std::abs(reconstructed(t, 2) - panel(t, 2)) < 1e-9, "Post-listing observed values preserved");
    }
}

// ----------------------------------------------------------------------------
// 7. Symbolic Rule Representations (Boolean, LSTR, Hinge Splines)
// ----------------------------------------------------------------------------
void test_symbolic_rules() {
    std::cout << "[+] Running test_symbolic_rules...\n";

    SymbolicRule rule;
    rule.conditions.push_back({0, true, 22.84});  // X0 <= 22.84
    rule.conditions.push_back({1, false, 1.81});  // X1 > 1.81

    std::vector<std::string> feat_names = {"Income", "PriceRatio"};

    // 1. Boolean Expression
    std::string bool_str = rule.to_boolean_expr(feat_names);
    TEST_ASSERT(bool_str == "I(Income <= 22.84 & PriceRatio > 1.81)", "Boolean expression generation");

    // 2. Continuous LSTR Expression
    std::string lstr_str = rule.to_lstr_expr(feat_names, 2.5);
    TEST_ASSERT(lstr_str.find("exp( 2.5 * (Income - 22.84)") != std::string::npos, "LSTR expression structure condition 1");
    TEST_ASSERT(lstr_str.find("exp( -2.5 * (PriceRatio - 1.81)") != std::string::npos, "LSTR expression structure condition 2");

    // 3. Continuous Hinge Spline Expression
    std::string hinge_str = rule.to_hinge_expr(feat_names);
    TEST_ASSERT(hinge_str.find("max(0, 22.84 - Income)") != std::string::npos, "Hinge spline expression condition 1");
    TEST_ASSERT(hinge_str.find("max(0, PriceRatio - 1.81)") != std::string::npos, "Hinge spline expression condition 2");

    // 4. Exact Evaluations
    std::vector<double> sample_match = {20.0, 2.5};     // 20 <= 22.84 AND 2.5 > 1.81 -> True (1.0)
    std::vector<double> sample_nomatch = {25.0, 2.5};   // 25 > 22.84 -> False (0.0)

    TEST_ASSERT(rule.evaluate(sample_match) == 1.0, "Rule evaluate matches");
    TEST_ASSERT(rule.evaluate(sample_nomatch) == 0.0, "Rule evaluate non-matches");

    double lstr_val = rule.evaluate_lstr(sample_match, 2.5);
    TEST_ASSERT(lstr_val > 0.5 && lstr_val <= 1.0, "LSTR smooth continuous evaluation in (0.5, 1.0]");

    double hinge_val = rule.evaluate_hinge(sample_match);
    TEST_ASSERT(std::abs(hinge_val - (2.84 * 0.69)) < 1e-4, "Hinge spline product evaluation");
}

// ----------------------------------------------------------------------------
// 8. Nirvana Two-Stage Econometric Regressor & Multiverse Invariance
// ----------------------------------------------------------------------------
void test_two_stage_regressor() {
    std::cout << "[+] Running test_two_stage_regressor...\n";

    size_t N = 80;
    size_t K = 2;
    Matrix X(N, K);
    std::vector<double> y(N);

    // Synthetic nonlinear process with explicit tipping threshold:
    // y = 2.0 + 1.5*X0 - 1.0*X1 + 4.0 * I(X0 > 0.5) + noise
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> u_dist(-1.0, 1.0);
    std::normal_distribution<double> eps_dist(0.0, 0.15);

    for (size_t i = 0; i < N; ++i) {
        X(i, 0) = u_dist(rng);
        X(i, 1) = u_dist(rng);
        double step = (X(i, 0) > 0.5) ? 4.0 : 0.0;
        y[i] = 2.0 + 1.5 * X(i, 0) - 1.0 * X(i, 1) + step + eps_dist(rng);
    }

    std::vector<std::string> feat_names = {"Feature_0", "Feature_1"};
    NirvanaTwoStageRegressor reg(6, 3, 1.0);
    reg.fit(X, y, feat_names);

    // Verify Phase 1 OLS statistics
    TEST_ASSERT(reg.base_report_.n_obs == N, "Base OLS N observations");
    TEST_ASSERT(reg.base_report_.r_squared > 0.50, "Base OLS R^2 exceeds 50%");
    TEST_ASSERT(reg.base_report_.f_stat > 10.0, "Base OLS F-statistic is significant");
    TEST_ASSERT(reg.base_report_.coefficients.size() == K + 1, "Base OLS includes intercept + features");

    // Verify Multiverse Invariance Screening across 30 Universes
    MultiverseInvarianceReport rep = reg.screen_multiverse_champions(X, y, 2, 30, 42, feat_names);

    TEST_ASSERT(rep.r2_augmented >= rep.r2_base, "Augmented R^2 >= Base R^2");
    TEST_ASSERT(!rep.champions.empty(), "Discovered Multiverse Champions extracted");
    TEST_ASSERT(rep.champions[0].multiverse_survival_rate > 0.50, "Top champion demonstrates high multiverse survival rate");
    TEST_ASSERT(!rep.champions[0].boolean_expr.empty(), "Champion exports Boolean equation");
    TEST_ASSERT(!rep.champions[0].lstr_expr.empty(), "Champion exports smooth LSTR equation");
    TEST_ASSERT(!rep.champions[0].hinge_expr.empty(), "Champion exports continuous Hinge Spline equation");
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "NIRVANA QUANT C++20 ENGINE - CORE TEST SUITE\n";
    std::cout << "============================================================\n";

    test_matrix_basics();
    test_manifold();
    test_kalman_filter();
    test_tree_engine();
    test_directional_connectedness();
    test_multiverse_reconstructor();
    test_symbolic_rules();
    test_two_stage_regressor();

    std::cout << "============================================================\n";
    std::cout << "[SUCCESS] ALL " << g_tests_passed << "/" << g_tests_total << " ASSERTIONS PASSED!\n";
    std::cout << "============================================================\n";

    return 0;
}
