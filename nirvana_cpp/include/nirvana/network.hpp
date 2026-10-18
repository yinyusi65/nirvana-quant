#ifndef NIRVANA_NETWORK_HPP
#define NIRVANA_NETWORK_HPP

#include "types.hpp"
#include "manifold.hpp"
#include "kalman.hpp"
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <random>

namespace nirvana {

struct SystemicRiskReport {
    Matrix spillover_table{};
    std::vector<std::string> node_names{};
    double total_connectedness_index{0.0};
    double normalized_tci{0.0};
    std::map<std::string, double> net_spillovers{};
    std::vector<std::string> top_transmitters{};
    std::vector<std::string> top_receivers{};
    std::map<std::string, double> eigenvector_hub{};
    std::map<std::string, double> eigenvector_authority{};
    std::map<std::string, double> betweenness_centrality{};
    std::optional<double> market_variance_share{std::nullopt};

    void print_summary() const {
        std::cout << "==============================================================================\n";
        std::cout << "NIRVANA SYSTEMIC RISK CONNECTEDNESS REPORT (C++ HIGH-PERFORMANCE ENGINE)\n";
        std::cout << "==============================================================================\n";
        std::cout << "Raw Total Connectedness Index (TCI):        " << std::fixed << std::setprecision(2) << total_connectedness_index << "%\n";
        std::cout << "Dimension-Normalized TCI (Adjusted for N):  " << std::fixed << std::setprecision(2) << normalized_tci << "%\n";
        if (market_variance_share.has_value()) {
            std::cout << "Market Orthogonalization Active:            YES (" << market_variance_share.value() << "% variance stripped)\n";
        }
        std::cout << "\nTop Risk Transmitters (NET > 0):\n";
        for (const auto& name : top_transmitters) {
            std::cout << "  - " << name << ": +" << std::fixed << std::setprecision(2) << net_spillovers.at(name) << "%\n";
        }
        std::cout << "\nTop Risk Receivers / Absorbers (NET < 0):\n";
        for (const auto& name : top_receivers) {
            std::cout << "  - " << name << ": " << std::fixed << std::setprecision(2) << net_spillovers.at(name) << "%\n";
        }
        if (!eigenvector_hub.empty()) {
            std::cout << "\nTop Hub Centrality (Systemic Shock Transmitters):\n";
            std::vector<std::pair<std::string, double>> hubs(eigenvector_hub.begin(), eigenvector_hub.end());
            std::sort(hubs.begin(), hubs.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            for (size_t i = 0; i < std::min(hubs.size(), size_t(5)); ++i) {
                std::cout << "  - " << hubs[i].first << ": " << std::fixed << std::setprecision(2) << hubs[i].second << "%\n";
            }
        }
        std::cout << "==============================================================================\n";
    }
};

struct DynamicSystemicRiskReport {
    std::vector<double> tci_history{};
    std::vector<double> normalized_tci_history{};
    std::vector<size_t> phase_transition_indices{};
    double mean_tci{0.0};
    double max_tci{0.0};
    size_t max_tci_idx{0};

    void print_summary() const {
        std::cout << "==============================================================================\n";
        std::cout << "NIRVANA DYNAMIC ROLLING CONNECTEDNESS REPORT (C++ ACCELERATED)\n";
        std::cout << "==============================================================================\n";
        std::cout << "Mean Rolling TCI:             " << std::fixed << std::setprecision(2) << mean_tci << "%\n";
        std::cout << "Peak Stress TCI:              " << std::fixed << std::setprecision(2) << max_tci << "% (at window " << max_tci_idx << ")\n";
        std::cout << "Detected Phase Transitions:   " << phase_transition_indices.size() << " events\n";
        for (size_t idx : phase_transition_indices) {
            std::cout << "  - Surge Event at Window Index: " << idx << "\n";
        }
        std::cout << "==============================================================================\n";
    }
};

class NirvanaConnectedness {
public:
    std::vector<std::string> node_names_{};
    std::vector<int> lags_{1, 2, 3};
    std::vector<double> lag_weights_{0.5, 0.3, 0.2};
    double kalman_q_{1e-3};
    double kalman_r_{1e-2};
    double kalman_phi_{0.98};
    std::optional<double> kalman_outlier_threshold_{std::nullopt};
    double apoptosis_threshold_{0.015};
    double p_value_threshold_{0.05};
    int n_boot_{0};
    bool orthogonalize_market_{false};
    int n_market_factors_{1};
    std::string causality_mode_{"linear"}; // "linear" or "threshold"
    double threshold_quantile_{0.70};
    double asset_elasticity_{0.5};
    std::map<std::string, double> asset_weights_{};
    std::map<std::string, double> exchange_bands_{};
    std::pair<double, double> macro_periods_{63.0, 252.0};
    double volume_quantum_{1000.0};

    NirvanaConnectedness() = default;

    SystemicRiskReport fit(const Matrix& data, const std::vector<std::string>& custom_names = {}) {
        size_t t_steps = data.rows();
        size_t n_nodes = data.cols();

        std::vector<std::string> names = custom_names;
        if (names.size() != n_nodes) {
            names.resize(n_nodes);
            for (size_t i = 0; i < n_nodes; ++i) {
                names[i] = "Node_" + std::to_string(i + 1);
            }
        }

        if (n_nodes < 2) {
            SystemicRiskReport single_rep;
            single_rep.total_connectedness_index = 0.0;
            single_rep.normalized_tci = 0.0;
            single_rep.node_names = names;
            return single_rep;
        }

        // 0. Exchange Price-Band Heteroskedasticity Mapping
        std::vector<double> r_vec(n_nodes, kalman_r_);
        for (size_t i = 0; i < n_nodes; ++i) {
            auto it = exchange_bands_.find(names[i]);
            if (it != exchange_bands_.end()) {
                double band = it->second;
                r_vec[i] = kalman_r_ * std::pow(band / 0.07, 2.0);
            }
        }

        // 1. Causal Kalman State Estimation with Mean Reversion & Huber Outlier Clipping
        MultivariateCausalKalman mv_kalman(n_nodes, kalman_q_, r_vec, kalman_phi_, kalman_outlier_threshold_);
        Matrix filtered_states = mv_kalman.filter_matrix(data);

        // 2. Dynamic Innovation Residuals
        Matrix innovations = data - filtered_states;

        // 2b. Multi-Factor SVD Market Orthogonalization
        std::optional<double> market_var_share = std::nullopt;
        if (orthogonalize_market_) {
            size_t k = std::min(static_cast<size_t>(std::max(1, n_market_factors_)), n_nodes - 1);
            Matrix U, V;
            std::vector<double> S;
            innovations.svd_truncated(k, U, S, V);

            double total_var = 0.0;
            for (double s_val : S) total_var += s_val * s_val;

            Matrix factors(t_steps, k);
            for (size_t t = 0; t < t_steps; ++t) {
                for (size_t comp = 0; comp < k; ++comp) {
                    factors(t, comp) = U(t, comp) * S[comp];
                }
            }

            // Regress out common factors from each node
            Matrix idio_innov(t_steps, n_nodes);
            for (size_t col = 0; col < n_nodes; ++col) {
                std::vector<double> y = innovations.col(col);
                // Simple orthogonal projection: y_idio = y - sum_comp (y . u_comp) * u_comp
                std::vector<double> y_proj = y;
                for (size_t comp = 0; comp < k; ++comp) {
                    double dot = 0.0;
                    for (size_t t = 0; t < t_steps; ++t) dot += y[t] * U(t, comp);
                    for (size_t t = 0; t < t_steps; ++t) y_proj[t] -= dot * U(t, comp);
                }
                for (size_t t = 0; t < t_steps; ++t) idio_innov(t, col) = y_proj[t];
            }
            innovations = idio_innov;
            market_var_share = 100.0 * (total_var / (total_var + 1.0)); // relative variance share proxy
        }

        // 3. Base-12 Manifold Harmonic Phase Transform
        Base12Manifold manifold(volume_quantum_, macro_periods_);
        std::vector<int> cal_indices(t_steps);
        std::iota(cal_indices.begin(), cal_indices.end(), 0);
        Matrix coords = manifold.batch_transform(cal_indices);

        // 4. Harmonic Weighted Innovations
        Matrix weighted_innov(t_steps, n_nodes);
        for (size_t t = 0; t < t_steps; ++t) {
            double gamma_t = coords(t, 8);
            double weight = 1.0 + 0.5 * gamma_t;
            for (size_t j = 0; j < n_nodes; ++j) {
                weighted_innov(t, j) = innovations(t, j) * weight;
            }
        }

        // Compute Pearson Correlation Matrix
        Matrix corr_mat(n_nodes, n_nodes, 0.0);
        std::vector<double> means = weighted_innov.col_means();
        std::vector<double> stds(n_nodes, 0.0);
        for (size_t j = 0; j < n_nodes; ++j) {
            double var = 0.0;
            for (size_t t = 0; t < t_steps; ++t) {
                double diff = weighted_innov(t, j) - means[j];
                var += diff * diff;
            }
            stds[j] = std::sqrt(std::max(1e-8, var / (t_steps - 1)));
        }

        for (size_t i = 0; i < n_nodes; ++i) {
            corr_mat(i, i) = 1.0;
            for (size_t j = i + 1; j < n_nodes; ++j) {
                double cov = 0.0;
                for (size_t t = 0; t < t_steps; ++t) {
                    cov += (weighted_innov(t, i) - means[i]) * (weighted_innov(t, j) - means[j]);
                }
                cov /= (t_steps - 1);
                double r = cov / (stds[i] * stds[j] + 1e-8);
                corr_mat(i, j) = r;
                corr_mat(j, i) = r;
            }
        }

        // 5. Multi-Lag Directional Asymmetry Convolution
        Matrix asym(n_nodes, n_nodes, 0.0);
        double sum_lw = 0.0;
        for (double w : lag_weights_) sum_lw += w;
        if (sum_lw < 1e-8) sum_lw = 1.0;

        for (size_t l_idx = 0; l_idx < lags_.size(); ++l_idx) {
            int lag = lags_[l_idx];
            if (lag <= 0 || static_cast<size_t>(lag) >= t_steps) continue;
            double w_lag = lag_weights_[l_idx] / sum_lw;

            size_t eff_t = t_steps - lag;
            for (size_t i = 0; i < n_nodes; ++i) {
                for (size_t j = 0; j < n_nodes; ++j) {
                    if (i == j) continue;
                    // c_ij: correlation between f(t) [future i] and p(t-lag) [past j]
                    double sum_prod = 0.0;
                    for (size_t t = 0; t < eff_t; ++t) {
                        double p_val = weighted_innov(t, j);
                        double f_val = weighted_innov(t + lag, i);
                        sum_prod += (p_val - means[j]) * (f_val - means[i]);
                    }
                    double c_ij = sum_prod / (eff_t * stds[i] * stds[j] + 1e-8);

                    // c_ji: correlation between f(t) [future j] and p(t-lag) [past i]
                    double sum_prod_rev = 0.0;
                    for (size_t t = 0; t < eff_t; ++t) {
                        double p_val = weighted_innov(t, i);
                        double f_val = weighted_innov(t + lag, j);
                        sum_prod_rev += (p_val - means[i]) * (f_val - means[j]);
                    }
                    double c_ji = sum_prod_rev / (eff_t * stds[i] * stds[j] + 1e-8);

                    asym(i, j) += w_lag * (c_ij - c_ji);
                }
            }
        }

        // 6. Raw Spillover Kernel: K_ij = |Corr_ij| * (0.5 + 0.5 * Asym_ij)
        Matrix kernel(n_nodes, n_nodes, 0.0);
        for (size_t i = 0; i < n_nodes; ++i) {
            for (size_t j = 0; j < n_nodes; ++j) {
                if (i == j) {
                    kernel(i, j) = 1.0;
                } else {
                    double k_val = std::abs(corr_mat(i, j)) * (0.5 + 0.5 * std::clamp(asym(i, j), -1.0, 1.0));
                    // Apoptosis link pruning
                    if (k_val < apoptosis_threshold_) k_val = 0.0;
                    kernel(i, j) = k_val;
                }
            }
        }

        // 7. Row Normalization to 100% (Conservation of Risk)
        Matrix norm_kernel(n_nodes, n_nodes, 0.0);
        std::vector<double> r_sums = kernel.row_sums();
        for (size_t i = 0; i < n_nodes; ++i) {
            double s = (r_sums[i] > 1e-8) ? r_sums[i] : 1.0;
            for (size_t j = 0; j < n_nodes; ++j) {
                norm_kernel(i, j) = (kernel(i, j) / s) * 100.0;
            }
        }

        // 8. Directional Metrics (TO, FROM, NET)
        std::vector<double> from_others(n_nodes, 0.0);
        std::vector<double> to_others(n_nodes, 0.0);
        for (size_t i = 0; i < n_nodes; ++i) {
            for (size_t j = 0; j < n_nodes; ++j) {
                if (i != j) {
                    from_others[i] += norm_kernel(i, j);
                    to_others[j] += norm_kernel(i, j);
                }
            }
        }

        // Sub-linear asset scaling if provided
        if (!asset_weights_.empty()) {
            std::vector<double> raw_w(n_nodes, 1.0);
            double sum_w = 0.0;
            for (size_t i = 0; i < n_nodes; ++i) {
                auto it = asset_weights_.find(names[i]);
                double a = (it != asset_weights_.end()) ? it->second : 100.0;
                raw_w[i] = std::pow(std::max(1.0, a), asset_elasticity_);
                sum_w += raw_w[i];
            }
            double mean_w = sum_w / n_nodes;
            for (size_t i = 0; i < n_nodes; ++i) {
                to_others[i] *= (raw_w[i] / mean_w);
            }
        }

        std::map<std::string, double> net_map;
        std::vector<std::string> transmitters, receivers;
        for (size_t i = 0; i < n_nodes; ++i) {
            double net = to_others[i] - from_others[i];
            net_map[names[i]] = net;
            if (net > 0) transmitters.push_back(names[i]);
            else receivers.push_back(names[i]);
        }

        std::sort(transmitters.begin(), transmitters.end(),
                  [&net_map](const std::string& a, const std::string& b) { return net_map[a] > net_map[b]; });
        std::sort(receivers.begin(), receivers.end(),
                  [&net_map](const std::string& a, const std::string& b) { return net_map[a] < net_map[b]; });

        // Total Connectedness Index
        double total_from = 0.0;
        for (double v : from_others) total_from += v;
        double tci = total_from / n_nodes;
        double max_tci = (static_cast<double>(n_nodes - 1) / n_nodes) * 100.0;
        double norm_tci = (tci / max_tci) * 100.0;

        // 9. Kleinberg HITS / Perron-Frobenius Centrality (Hub vs Authority)
        Matrix A = norm_kernel;
        for (size_t i = 0; i < n_nodes; ++i) A(i, i) = 0.0;

        std::vector<double> hub(n_nodes, 1.0 / std::sqrt(n_nodes));
        std::vector<double> auth(n_nodes, 1.0 / std::sqrt(n_nodes));

        // Power iteration
        Matrix At = A.transpose();
        for (size_t iter = 0; iter < 50; ++iter) {
            // hub = A^T * auth
            std::vector<double> next_hub = At * auth;
            double h_norm = 0.0;
            for (double x : next_hub) h_norm += x * x;
            h_norm = std::sqrt(std::max(1e-12, h_norm));
            for (double& x : next_hub) x /= h_norm;

            // auth = A * hub
            std::vector<double> next_auth = A * next_hub;
            double a_norm = 0.0;
            for (double x : next_auth) a_norm += x * x;
            a_norm = std::sqrt(std::max(1e-12, a_norm));
            for (double& x : next_auth) x /= a_norm;

            hub = next_hub;
            auth = next_auth;
        }

        double sum_h = 0.0, sum_a = 0.0;
        for (double x : hub) sum_h += x;
        for (double x : auth) sum_a += x;
        if (sum_h < 1e-8) sum_h = 1.0;
        if (sum_a < 1e-8) sum_a = 1.0;

        std::map<std::string, double> hub_map, auth_map;
        for (size_t i = 0; i < n_nodes; ++i) {
            hub_map[names[i]] = (hub[i] / sum_h) * 100.0;
            auth_map[names[i]] = (auth[i] / sum_a) * 100.0;
        }

        SystemicRiskReport report;
        report.spillover_table = norm_kernel;
        report.node_names = names;
        report.total_connectedness_index = tci;
        report.normalized_tci = norm_tci;
        report.net_spillovers = net_map;
        report.top_transmitters = transmitters;
        report.top_receivers = receivers;
        report.eigenvector_hub = hub_map;
        report.eigenvector_authority = auth_map;
        report.market_variance_share = market_var_share;

        return report;
    }

    DynamicSystemicRiskReport rolling_fit(const Matrix& data, size_t window = 120, size_t step = 10,
                                          double surge_threshold = 2.0,
                                          const std::vector<std::string>& custom_names = {}) {
        size_t n_rows = data.rows();
        DynamicSystemicRiskReport dyn_rep;
        if (n_rows < window) return dyn_rep;

        std::vector<double> tcis, norm_tcis;
        for (size_t end_idx = window; end_idx <= n_rows; end_idx += step) {
            size_t start_idx = end_idx - window;
            Matrix sub_mat(window, data.cols());
            for (size_t r = 0; r < window; ++r) {
                for (size_t c = 0; c < data.cols(); ++c) {
                    sub_mat(r, c) = data(start_idx + r, c);
                }
            }
            SystemicRiskReport rep = fit(sub_mat, custom_names);
            tcis.push_back(rep.total_connectedness_index);
            norm_tcis.push_back(rep.normalized_tci);
        }

        dyn_rep.tci_history = tcis;
        dyn_rep.normalized_tci_history = norm_tcis;

        // Detect Phase Transitions
        double max_v = 0.0;
        size_t max_idx = 0;
        double sum_v = 0.0;
        for (size_t i = 0; i < tcis.size(); ++i) {
            sum_v += tcis[i];
            if (tcis[i] > max_v) {
                max_v = tcis[i];
                max_idx = i;
            }
        }
        dyn_rep.mean_tci = tcis.empty() ? 0.0 : (sum_v / tcis.size());
        dyn_rep.max_tci = max_v;
        dyn_rep.max_tci_idx = max_idx;

        if (tcis.size() >= 5) {
            for (size_t i = 4; i < tcis.size(); ++i) {
                double mean_past = 0.0;
                for (size_t j = i - 4; j < i; ++j) mean_past += tcis[j];
                mean_past /= 4.0;
                double var_past = 0.0;
                for (size_t j = i - 4; j < i; ++j) var_past += (tcis[j] - mean_past) * (tcis[j] - mean_past);
                double std_past = std::sqrt(var_past / 3.0) + 1e-4;
                if ((tcis[i] - mean_past) / std_past >= surge_threshold) {
                    dyn_rep.phase_transition_indices.push_back(i);
                }
            }
        }

        return dyn_rep;
    }
};

} // namespace nirvana

#endif // NIRVANA_NETWORK_HPP
