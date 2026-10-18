#ifndef NIRVANA_MULTIVERSE_HPP
#define NIRVANA_MULTIVERSE_HPP

#include "types.hpp"
#include "manifold.hpp"
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>

namespace nirvana {

/**
 * @brief Multiverse Counterfactual Synthesis Engine.
 * Synthesizes unobserved latent histories (e.g. pre-IPO periods) anchored on:
 * 1. Structural Peer-Group Dynamics
 * 2. Macroeconomic Market Factors
 * 3. Base-12 Dual-Clock Manifold Harmonics
 * 4. Multiverse Ensemble Simulation with Apoptosis Pruning & Hermite Boundary Ramp
 */
class MultiverseReconstructor {
public:
    size_t n_universes_{3};
    size_t boundary_window_{30};
    double apoptosis_threshold_{2.5};
    double reg_lambda_{0.5};
    double volume_quantum_{1000.0};
    std::map<std::string, std::vector<std::string>> tier_mapping_{};

    MultiverseReconstructor(
        size_t n_universes = 3,
        size_t boundary_window = 30,
        double apoptosis_threshold = 2.5,
        double reg_lambda = 0.5,
        double volume_quantum = 1000.0
    ) : n_universes_(n_universes),
        boundary_window_(boundary_window),
        apoptosis_threshold_(apoptosis_threshold),
        reg_lambda_(reg_lambda),
        volume_quantum_(volume_quantum) {}

    static unsigned int deterministic_seed(const std::string& text) {
        unsigned int h = 0;
        for (size_t i = 0; i < text.size(); ++i) {
            h = h * 31 + static_cast<unsigned int>(text[i]);
        }
        return h;
    }

    /**
     * @brief Reconstructs missing/unlisted history for each node.
     * Takes raw (T, N) panel where missing values are represented by NaN.
     */
    Matrix reconstruct(const Matrix& raw_panel, const std::vector<double>& market_index = {}) const {
        size_t t_steps = raw_panel.rows();
        size_t n_nodes = raw_panel.cols();
        Matrix reconstructed = raw_panel;

        // Base-12 Manifold harmonics
        Base12Manifold manifold(volume_quantum_);
        std::vector<int> cal_indices(t_steps);
        std::iota(cal_indices.begin(), cal_indices.end(), 0);
        Matrix coords = manifold.batch_transform(cal_indices);

        // Market baseline vector
        std::vector<double> mkt_vol(t_steps, 0.0);
        if (market_index.size() == t_steps) {
            mkt_vol = market_index;
        } else {
            // Cross-sectional mean across valid entries
            for (size_t t = 0; t < t_steps; ++t) {
                double sum = 0.0;
                size_t cnt = 0;
                for (size_t j = 0; j < n_nodes; ++j) {
                    double val = raw_panel(t, j);
                    if (!std::isnan(val)) {
                        sum += val;
                        cnt++;
                    }
                }
                mkt_vol[t] = (cnt > 0) ? (sum / cnt) : 0.0;
            }
        }

        // Process each column
        for (size_t col = 0; col < n_nodes; ++col) {
            std::vector<double> series = raw_panel.col(col);

            // Find first valid index
            size_t first_pos = 0;
            bool found_valid = false;
            for (size_t t = 0; t < t_steps; ++t) {
                if (!std::isnan(series[t])) {
                    first_pos = t;
                    found_valid = true;
                    break;
                }
            }

            if (!found_valid) {
                // Entire series is missing: synthesize from market baseline
                std::mt19937 rng(deterministic_seed("Node_" + std::to_string(col)));
                std::normal_distribution<double> dist(0.0, 0.2);
                for (size_t t = 0; t < t_steps; ++t) {
                    reconstructed(t, col) = mkt_vol[t] + dist(rng);
                }
                continue;
            }

            if (first_pos == 0) {
                // Already listed from day 1
                continue;
            }

            // Target node was unobserved for [0 : first_pos]
            // Observed period: [first_pos : t_steps]
            size_t n_train = t_steps - first_pos;
            if (n_train < 10) continue;

            // Design matrix X: [mkt_vol, m_cos_1, m_sin_1, 1.0]
            Matrix X_train(n_train, 4);
            Matrix y_train(n_train, 1);
            for (size_t i = 0; i < n_train; ++i) {
                size_t t = first_pos + i;
                X_train(i, 0) = mkt_vol[t];
                X_train(i, 1) = coords(t, 1); // m_cos_1
                X_train(i, 2) = coords(t, 0); // m_sin_1
                X_train(i, 3) = 1.0;
                y_train(i, 0) = series[t];
            }

            // Ridge regression: beta = (X^T * X + lambda * I)^(-1) * X^T * y
            Matrix pinv_X = Matrix::ridge_pinv(X_train, reg_lambda_);
            Matrix beta = pinv_X * y_train;

            // Calculate residual variance
            Matrix y_hat = X_train * beta;
            double res_var = 0.0;
            for (size_t i = 0; i < n_train; ++i) {
                double diff = y_train(i, 0) - y_hat(i, 0);
                res_var += diff * diff;
            }
            double sigma_noise = std::max(0.15, std::sqrt(res_var / n_train));

            // Generate Multiverse branches for unobserved [0 : first_pos]
            std::mt19937 rng(deterministic_seed("Node_" + std::to_string(col)));
            std::normal_distribution<double> dist_u1(0.0, sigma_noise);
            std::normal_distribution<double> dist_u2(0.0, sigma_noise * 1.15);
            std::normal_distribution<double> dist_u3(0.0, sigma_noise * 0.85);

            std::vector<double> y_synthetic(first_pos);
            for (size_t t = 0; t < first_pos; ++t) {
                double base_pred = mkt_vol[t] * beta(0, 0) +
                                   coords(t, 1) * beta(1, 0) +
                                   coords(t, 0) * beta(2, 0) +
                                   1.0 * beta(3, 0);

                double u1 = base_pred + dist_u1(rng);
                double u2 = 0.6 * mkt_vol[t] + 0.4 * base_pred + dist_u2(rng);
                double u3 = base_pred + 0.25 * (mkt_vol[t] - mkt_vol[0]) + dist_u3(rng);

                // Apoptosis branch filtering
                std::vector<double> branch_vals = {u1, u2, u3};
                std::sort(branch_vals.begin(), branch_vals.end());
                double median_val = branch_vals[1];

                double ensemble = 0.0;
                double w_sum = 0.0;
                for (double b : {u1, u2, u3}) {
                    double diff = std::abs(b - median_val);
                    if (diff <= apoptosis_threshold_ * sigma_noise) {
                        ensemble += b;
                        w_sum += 1.0;
                    }
                }
                y_synthetic[t] = (w_sum > 0.0) ? (ensemble / w_sum) : median_val;
            }

            // Cubic Hermite smooth boundary bridging to prevent first-day jump shock
            double post_ipo_val = series[first_pos];
            size_t window = std::min(boundary_window_, first_pos);
            if (window > 0) {
                double delta = post_ipo_val - y_synthetic[first_pos - 1];
                for (size_t w = 0; w < window; ++w) {
                    size_t t_idx = (first_pos - window) + w;
                    double t_norm = static_cast<double>(w + 1) / window;
                    double smooth_ramp = t_norm * t_norm * (3.0 - 2.0 * t_norm); // Hermite S-curve
                    y_synthetic[t_idx] += smooth_ramp * delta;
                }
            }

            // Inject into reconstructed matrix
            for (size_t t = 0; t < first_pos; ++t) {
                reconstructed(t, col) = y_synthetic[t];
            }
        }

        return reconstructed;
    }
};

} // namespace nirvana

#endif // NIRVANA_MULTIVERSE_HPP
