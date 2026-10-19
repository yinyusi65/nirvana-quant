#ifndef NIRVANA_KALMAN_HPP
#define NIRVANA_KALMAN_HPP

#include "types.hpp"
#include <vector>
#include <cmath>
#include <limits>
#include <optional>

namespace nirvana {

/**
 * @brief Multivariate Causal Online Kalman Filter.
 * Features:
 * - Strictly causal state-space filtering (zero lookahead)
 * - Mean-reverting AR(1) persistence (phi)
 * - Joseph stabilized covariance update P = (I - K) P_pred (I - K)^T + K R K^T
 * - Huber-robust innovation bounding to handle fat-tailed jumps & sensor spikes
 * - Dynamic partial observation update for unbalanced/missing node panels
 */
class MultivariateCausalKalman {
public:
    size_t n_{0};
    double phi_{0.98};
    std::optional<double> outlier_threshold_{std::nullopt};
    Matrix Q_{};
    Matrix R_{};
    std::vector<double> state_{};
    std::vector<double> mu_{};
    Matrix P_{};
    bool initialized_{false};

    MultivariateCausalKalman(
        size_t n_dim,
        double q_diag = 1e-3,
        double r_diag = 1e-2,
        double phi = 0.98,
        std::optional<double> outlier_threshold = std::nullopt
    ) : n_(n_dim),
        phi_(phi),
        outlier_threshold_(outlier_threshold),
        Q_(Matrix::eye(n_dim, q_diag)),
        R_(Matrix::eye(n_dim, r_diag)),
        state_(n_dim, 0.0),
        mu_(n_dim, 0.0),
        P_(Matrix::eye(n_dim, 1.0)),
        initialized_(false) {}

    MultivariateCausalKalman(
        size_t n_dim,
        double q_diag,
        const std::vector<double>& r_diag_vec,
        double phi = 0.98,
        std::optional<double> outlier_threshold = std::nullopt
    ) : n_(n_dim),
        phi_(phi),
        outlier_threshold_(outlier_threshold),
        Q_(Matrix::eye(n_dim, q_diag)),
        R_(Matrix::diag(r_diag_vec)),
        state_(n_dim, 0.0),
        mu_(n_dim, 0.0),
        P_(Matrix::eye(n_dim, 1.0)),
        initialized_(false) {}

    void reset() {
        state_.assign(n_, 0.0);
        mu_.assign(n_, 0.0);
        P_ = Matrix::eye(n_, 1.0);
        initialized_ = false;
    }

    std::vector<double> update(const std::vector<double>& z) {
        if (!initialized_) {
            // Check if at least one observation is valid
            bool any_valid = false;
            double sum_valid = 0.0;
            size_t count_valid = 0;
            for (double val : z) {
                if (!std::isnan(val)) {
                    any_valid = true;
                    sum_valid += val;
                    count_valid++;
                }
            }
            if (!any_valid) {
                return std::vector<double>(n_, 0.0);
            }
            double fill_val = sum_valid / count_valid;
            for (size_t i = 0; i < n_; ++i) {
                state_[i] = std::isnan(z[i]) ? fill_val : z[i];
                mu_[i] = state_[i];
            }
            initialized_ = true;
            return state_;
        }

        // 1. Prediction step with Mean-Reversion: x_{t|t-1} = mu + phi * (x_{t-1} - mu)
        for (size_t i = 0; i < n_; ++i) {
            state_[i] = mu_[i] + phi_ * (state_[i] - mu_[i]);
        }
        Matrix P_pred = P_ * (phi_ * phi_) + Q_;

        // 2. Identify observed nodes
        std::vector<size_t> obs_idx;
        for (size_t i = 0; i < n_; ++i) {
            if (!std::isnan(z[i])) {
                obs_idx.push_back(i);
            }
        }

        if (obs_idx.empty()) {
            P_ = P_pred * 0.5 + P_pred.transpose() * 0.5;
            return state_;
        }

        if (obs_idx.size() == n_) {
            // Full observation update
            Matrix S = P_pred + R_;
            Matrix S_inv = S.inverse(1e-8);
            Matrix K = P_pred * S_inv;

            std::vector<double> innovation(n_);
            for (size_t i = 0; i < n_; ++i) {
                innovation[i] = z[i] - state_[i];
                if (outlier_threshold_.has_value()) {
                    double thresh = outlier_threshold_.value();
                    double std_s = std::sqrt(std::max(S(i, i), 1e-8));
                    double z_sc = innovation[i] / std_s;
                    double clipped_z = std::clamp(z_sc, -thresh, thresh);
                    innovation[i] = clipped_z * std_s;
                }
            }

            std::vector<double> k_innov = K * innovation;
            for (size_t i = 0; i < n_; ++i) {
                state_[i] += k_innov[i];
            }

            // Joseph form: P = (I - K) P_pred (I - K)^T + K R K^T
            Matrix I_K = Matrix::eye(n_) - K;
            Matrix P_new = I_K * P_pred * I_K.transpose() + K * R_ * K.transpose();
            P_ = P_new * 0.5 + P_new.transpose() * 0.5;
            for (size_t i = 0; i < n_; ++i) {
                P_(i, i) = std::max(P_(i, i), 1e-8);
            }

            // Adaptive slow mean update
            for (size_t i = 0; i < n_; ++i) {
                mu_[i] += 0.005 * (state_[i] - mu_[i]);
            }
            return state_;
        }

        // Partial observation update (Dynamic node entry / missing rows)
        size_t k_obs = obs_idx.size();
        Matrix H_obs = Matrix::zeros(k_obs, n_);
        Matrix R_sub = Matrix::zeros(k_obs, k_obs);
        std::vector<double> z_sub(k_obs);

        for (size_t i = 0; i < k_obs; ++i) {
            size_t orig_idx = obs_idx[i];
            H_obs(i, orig_idx) = 1.0;
            R_sub(i, i) = R_(orig_idx, orig_idx);
            z_sub[i] = z[orig_idx];
        }

        Matrix S_obs = H_obs * P_pred * H_obs.transpose() + R_sub;
        Matrix S_inv = S_obs.inverse(1e-8);
        Matrix K_obs = P_pred * H_obs.transpose() * S_inv;

        std::vector<double> H_state = H_obs * state_;
        std::vector<double> innovation(k_obs);
        for (size_t i = 0; i < k_obs; ++i) {
            innovation[i] = z_sub[i] - H_state[i];
            if (outlier_threshold_.has_value()) {
                double thresh = outlier_threshold_.value();
                double std_s = std::sqrt(std::max(S_obs(i, i), 1e-8));
                double z_sc = innovation[i] / std_s;
                double clipped_z = std::clamp(z_sc, -thresh, thresh);
                innovation[i] = clipped_z * std_s;
            }
        }

        std::vector<double> k_innov = K_obs * innovation;
        for (size_t i = 0; i < n_; ++i) {
            state_[i] += k_innov[i];
        }

        Matrix I_KH = Matrix::eye(n_) - K_obs * H_obs;
        Matrix P_new = I_KH * P_pred * I_KH.transpose() + K_obs * R_sub * K_obs.transpose();
        P_ = P_new * 0.5 + P_new.transpose() * 0.5;
        for (size_t i = 0; i < n_; ++i) {
            P_(i, i) = std::max(P_(i, i), 1e-8);
            mu_[i] += 0.005 * (state_[i] - mu_[i]);
        }

        return state_;
    }

    Matrix filter_matrix(const Matrix& raw_data) {
        reset();
        size_t t = raw_data.rows();
        size_t n = raw_data.cols();
        Matrix filtered(t, n);

        for (size_t i = 0; i < t; ++i) {
            std::vector<double> row_z = raw_data.row(i);
            std::vector<double> state_t = update(row_z);
            for (size_t j = 0; j < n; ++j) {
                filtered(i, j) = state_t[j];
            }
        }
        return filtered;
    }
};

/**
 * @brief Causal Kalman Time-Varying Parameter (TVP) Regressor.
 * ============================================================
 * State space:
 *   Observation: y_t = x_t^T beta_t + v_t,        v_t ~ N(0, R)
 *   Transition:  beta_t = phi * beta_{t-1} + w_t,   w_t ~ N(0, Q)
 */
class CausalKalmanTVPRegressor {
public:
    double q_diag_{1e-4};
    double r_scalar_{1e-2};
    double phi_{0.999};
    bool fit_intercept_{true};

    std::vector<double> beta_{};
    Matrix P_{};
    size_t n_features_{0};
    bool fitted_{false};

    CausalKalmanTVPRegressor(
        double q_diag = 1e-4,
        double r_scalar = 1e-2,
        double phi = 0.999,
        bool fit_intercept = true
    ) : q_diag_(q_diag), r_scalar_(r_scalar), phi_(phi), fit_intercept_(fit_intercept) {}

    void fit(const Matrix& X, const std::vector<double>& y) {
        size_t N = X.rows();
        size_t K = X.cols();
        size_t d = fit_intercept_ ? (K + 1) : K;
        n_features_ = d;

        beta_.assign(d, 0.0);
        P_ = Matrix::eye(d, 1.0);
        Matrix Q = Matrix::eye(d, q_diag_);
        double R = r_scalar_;

        // Sequential online Kalman updates
        for (size_t t = 0; t < N; ++t) {
            std::vector<double> xt(d);
            if (fit_intercept_) {
                xt[0] = 1.0;
                for (size_t j = 0; j < K; ++j) xt[j + 1] = X(t, j);
            } else {
                for (size_t j = 0; j < K; ++j) xt[j] = X(t, j);
            }
            double yt = y[t];

            // 1. Prediction
            std::vector<double> beta_pred(d);
            for (size_t j = 0; j < d; ++j) beta_pred[j] = phi_ * beta_[j];
            Matrix P_pred = P_ * (phi_ * phi_) + Q;

            // 2. Innovation
            double y_hat = 0.0;
            for (size_t j = 0; j < d; ++j) y_hat += xt[j] * beta_pred[j];
            double innov = yt - y_hat;

            // xt^T * P_pred * xt
            double xt_P_xt = 0.0;
            std::vector<double> P_xt(d, 0.0);
            for (size_t r = 0; r < d; ++r) {
                for (size_t c = 0; c < d; ++c) {
                    P_xt[r] += P_pred(r, c) * xt[c];
                }
                xt_P_xt += xt[r] * P_xt[r];
            }
            double S = std::max(1e-8, xt_P_xt + R);

            // 3. Kalman Gain K = P_pred * xt / S
            std::vector<double> K_gain(d);
            for (size_t j = 0; j < d; ++j) K_gain[j] = P_xt[j] / S;

            // 4. State Update
            for (size_t j = 0; j < d; ++j) beta_[j] = beta_pred[j] + K_gain[j] * innov;

            // Joseph covariance update: P = (I - K xt^T) P_pred (I - K xt^T)^T + K R K^T
            Matrix I_Kx = Matrix::eye(d);
            for (size_t r = 0; r < d; ++r) {
                for (size_t c = 0; c < d; ++c) {
                    I_Kx(r, c) -= K_gain[r] * xt[c];
                }
            }
            Matrix P_new = I_Kx * P_pred * I_Kx.transpose();
            for (size_t r = 0; r < d; ++r) {
                for (size_t c = 0; c < d; ++c) {
                    P_new(r, c) += K_gain[r] * K_gain[c] * R;
                }
            }
            P_ = P_new * 0.5 + P_new.transpose() * 0.5;
        }
        fitted_ = true;
    }

    std::vector<double> predict(const Matrix& X) const {
        if (!fitted_) throw std::runtime_error("CausalKalmanTVPRegressor is not fitted.");
        size_t N = X.rows();
        size_t K = X.cols();
        std::vector<double> preds(N, 0.0);

        for (size_t i = 0; i < N; ++i) {
            double p = 0.0;
            if (fit_intercept_) {
                p += beta_[0];
                for (size_t j = 0; j < K; ++j) p += X(i, j) * beta_[j + 1];
            } else {
                for (size_t j = 0; j < K; ++j) p += X(i, j) * beta_[j];
            }
            preds[i] = p;
        }
        return preds;
    }
};

} // namespace nirvana

#endif // NIRVANA_KALMAN_HPP
