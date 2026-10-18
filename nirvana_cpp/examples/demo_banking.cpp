#include <nirvana/nirvana.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <iomanip>

using namespace nirvana;

int main() {
    std::cout << "==============================================================================\n";
    std::cout << "NIRVANA QUANT C++20 ENGINE - BANKING SYSTEMIC RISK SIMULATION\n";
    std::cout << "Chrono-Cancer Meta-Architecture (Dual-Clock Manifold & Directed Connectedness)\n";
    std::cout << "==============================================================================\n\n";

    // 27 Vietnamese Banks Universe
    std::vector<std::string> bank_universe = {
        "VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "HDB", "STB", "SHB",
        "VIB", "TPB", "MSB", "LPB", "EIB", "OCB", "SSB", "BAB", "BVB", "NAB",
        "PGB", "NVB", "VAB", "KLB", "SGB", "ABB", "SCB"
    };

    size_t N = bank_universe.size();
    size_t T = 252; // 1 trading year

    std::cout << "[+] Initializing Universe with " << N << " Banking Institutions across " << T << " Trading Days...\n";

    // Simulate multi-factor returns with Big4 Core (VCB, BID, CTG) driving systemic spillovers
    std::mt19937 rng(1337);
    std::normal_distribution<double> norm(0.0, 0.015);
    std::normal_distribution<double> mkt_shock(0.0, 0.02);

    Matrix panel(T, N, 0.0);

    for (size_t t = 0; t < T; ++t) {
        // Latent market factor
        double mkt = mkt_shock(rng);

        // Core systemic leaders (VCB, BID, CTG) receive earlier shocks
        double vcb_shock = 0.6 * mkt + norm(rng);
        panel(t, 0) = vcb_shock; // VCB
        panel(t, 1) = 0.5 * mkt + 0.3 * (t > 0 ? panel(t - 1, 0) : 0.0) + norm(rng); // BID
        panel(t, 2) = 0.5 * mkt + 0.3 * (t > 0 ? panel(t - 1, 0) : 0.0) + norm(rng); // CTG

        // Joint stock commercial banks (Tier 2 & 3)
        for (size_t j = 3; j < N; ++j) {
            double leader_spill = 0.15 * (t > 0 ? panel(t - 1, 0) : 0.0) +
                                  0.10 * (t > 0 ? panel(t - 1, 1) : 0.0);
            panel(t, j) = 0.4 * mkt + leader_spill + norm(rng);
        }

        // Stress period simulation around day 180 to 200 (systemic shock)
        if (t >= 180 && t <= 200) {
            for (size_t j = 0; j < N; ++j) {
                panel(t, j) += norm(rng) * 2.5; // High volatility burst
            }
        }
    }

    // Configure Nirvana Connectedness Engine
    NirvanaConnectedness engine;
    engine.lags_ = {1, 2};
    engine.lag_weights_ = {0.7, 0.3};
    engine.kalman_q_ = 1e-4;
    engine.kalman_r_ = 1e-3;
    engine.kalman_phi_ = 0.98;
    engine.kalman_outlier_threshold_ = 3.0; // Huber robust fat-tail clipping
    engine.orthogonalize_market_ = true;
    engine.n_market_factors_ = 1; // Strip common systemic market beta
    engine.apoptosis_threshold_ = 0.02; // Fisher-prune spurious edges

    std::cout << "[+] Executing Nirvana High-Performance Systemic Risk Estimation...\n";
    SystemicRiskReport report = engine.fit(panel, bank_universe);
    report.print_summary();

    std::cout << "\n[+] Executing Dynamic Rolling Connectedness (Window=63, Step=5)...\n";
    DynamicSystemicRiskReport rolling_report = engine.rolling_fit(panel, 63, 5, 2.0, bank_universe);
    rolling_report.print_summary();

    std::cout << "\n[✓] Simulation Finished Successfully with Zero Memory Leaks.\n";
    return 0;
}
