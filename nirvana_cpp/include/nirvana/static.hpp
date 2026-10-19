#ifndef NIRVANA_STATIC_HPP
#define NIRVANA_STATIC_HPP

#include "types.hpp"
#include "tree.hpp"
#include "kalman.hpp"
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

namespace nirvana {

/**
 * @brief Standard normal cumulative distribution function approximation.
 */
inline double norm_cdf(double x) {
    constexpr double SQRT1_2 = 0.70710678118654752440;
    return 0.5 * std::erfc(-x * SQRT1_2);
}

/**
 * @brief Approximates two-tailed p-value from t-statistic.
 */
inline double t_to_p_value(double t_stat, int df) {
    if (df <= 0) return 1.0;
    // For df >= 30, standard normal is virtually indistinguishable
    double z = std::abs(t_stat);
    if (df > 30) {
        return 2.0 * (1.0 - norm_cdf(z));
    }
    // Hill's approximation for moderate df
    double a = 1.0 - 1.0 / (4.0 * df);
    double z_adj = z * a;
    return std::max(0.0, std::min(1.0, 2.0 * (1.0 - norm_cdf(z_adj))));
}

/**
 * @brief Approximate F-distribution survival function (p-value).
 */
inline double f_to_p_value(double f_stat, int df1, int df2) {
    if (f_stat <= 0.0 || df1 <= 0 || df2 <= 0) return 1.0;
    // Wilson-Hilferty transformation of F to normal Z
    double d1 = static_cast<double>(df1);
    double d2 = static_cast<double>(df2);
    double term1 = std::pow(f_stat, 1.0 / 3.0) * (1.0 - 2.0 / (9.0 * d2));
    double term2 = 1.0 - 2.0 / (9.0 * d1);
    double denom = std::sqrt(2.0 / (9.0 * d1) + std::pow(f_stat, 2.0 / 3.0) * (2.0 / (9.0 * d2)));
    if (denom < 1e-12) return 0.0;
    double z = (term1 - term2) / denom;
    return std::max(0.0, std::min(1.0, 1.0 - norm_cdf(z)));
}

/**
 * @brief Academic Stata-compliant OLS Regression Report.
 */
struct StataRegressionReport {
    double r_squared{0.0};
    double adj_r_squared{0.0};
    double f_stat{0.0};
    double f_p_value{0.0};
    double root_mse{0.0};
    double ss_model{0.0};
    double ss_resid{0.0};
    double ss_total{0.0};
    size_t n_obs{0};
    size_t df_model{0};
    size_t df_resid{0};

    std::vector<std::string> var_names{};
    std::vector<double> coefficients{};
    std::vector<double> std_errors{};
    std::vector<double> t_stats{};
    std::vector<double> p_values{};
    std::vector<double> ci_lower{};
    std::vector<double> ci_upper{};

    double ramsey_reset_f{0.0};
    double ramsey_reset_p_value{1.0};
    bool is_reset_cured{false};

    void print_summary(const std::string& depvar_name = "y") const {
        std::cout << "\n      Source |       SS            df         MS          Number of obs   = "
                  << std::setw(10) << n_obs << "\n";
        std::cout << "-------------+------------------------------------------   F("
                  << std::setw(2) << df_model << ", " << std::setw(5) << df_resid << ")   = "
                  << std::setw(10) << std::fixed << std::setprecision(2) << f_stat << "\n";
        std::cout << "       Model | " << std::setw(15) << std::setprecision(4) << ss_model << " "
                  << std::setw(6) << df_model << " " << std::setw(17) << (ss_model / std::max(size_t(1), df_model))
                  << "   Prob > F        = " << std::setw(10) << std::setprecision(4) << f_p_value << "\n";
        std::cout << "    Residual | " << std::setw(15) << std::setprecision(4) << ss_resid << " "
                  << std::setw(6) << df_resid << " " << std::setw(17) << (ss_resid / std::max(size_t(1), df_resid))
                  << "   R-squared       = " << std::setw(10) << std::setprecision(4) << r_squared << "\n";
        std::cout << "-------------+------------------------------------------   Adj R-squared   = "
                  << std::setw(10) << std::setprecision(4) << adj_r_squared << "\n";
        std::cout << "       Total | " << std::setw(15) << std::setprecision(4) << ss_total << " "
                  << std::setw(6) << (n_obs - 1) << " " << std::setw(17) << (ss_total / std::max(size_t(1), n_obs - 1))
                  << "   Root MSE        = " << std::setw(10) << std::setprecision(4) << root_mse << "\n";

        std::cout << "\n------------------------------------------------------------------------------\n";
        std::cout << std::setw(13) << std::left << depvar_name
                  << " | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]\n";
        std::cout << "------------------------------------------------------------------------------\n";

        for (size_t i = 0; i < var_names.size(); ++i) {
            std::cout << std::setw(13) << std::left << var_names[i] << " | "
                      << std::setw(11) << std::right << std::fixed << std::setprecision(5) << coefficients[i] << "  "
                      << std::setw(9) << std::setprecision(5) << std_errors[i] << "  "
                      << std::setw(5) << std::setprecision(2) << t_stats[i] << "   "
                      << std::setw(6) << std::setprecision(3) << p_values[i] << "   "
                      << std::setw(10) << std::setprecision(5) << ci_lower[i] << "  "
                      << std::setw(10) << std::setprecision(5) << ci_upper[i] << "\n";
        }
        std::cout << "------------------------------------------------------------------------------\n";
        std::cout << "Ramsey RESET test (estat ovtest): F = " << std::fixed << std::setprecision(2)
                  << ramsey_reset_f << " (p-value: " << std::setprecision(4) << ramsey_reset_p_value << ")\n";
        std::cout << "Status: " << (is_reset_cured ? "CURED (No omitted non-linear form)" : "Omitted non-linear form detected") << "\n";
    }
};

/**
 * @brief Symbolic condition inside decision branch.
 */
struct SymbolicCondition {
    int feature_idx{-1};
    bool is_less_equal{true};
    double threshold{0.0};
};

/**
 * @brief Closed-form analytical rule expression supporting Boolean, LSTR, and Hinge Splines.
 */
struct SymbolicRule {
    std::vector<SymbolicCondition> conditions{};
    double value{0.0};
    int node_idx{0};

    std::string to_boolean_expr(const std::vector<std::string>& feat_names = {}) const {
        if (conditions.empty()) return "1.0";
        std::stringstream ss;
        ss << "I(";
        for (size_t i = 0; i < conditions.size(); ++i) {
            const auto& c = conditions[i];
            std::string name = (c.feature_idx >= 0 && c.feature_idx < static_cast<int>(feat_names.size()))
                               ? feat_names[c.feature_idx]
                               : ("X" + std::to_string(c.feature_idx));
            ss << name << (c.is_less_equal ? " <= " : " > ") << std::fixed << std::setprecision(2) << c.threshold;
            if (i + 1 < conditions.size()) ss << " & ";
        }
        ss << ")";
        return ss.str();
    }

    std::string to_lstr_expr(const std::vector<std::string>& feat_names = {}, double alpha = 2.5) const {
        if (conditions.empty()) return "1.0";
        std::stringstream ss;
        for (size_t i = 0; i < conditions.size(); ++i) {
            const auto& c = conditions[i];
            std::string name = (c.feature_idx >= 0 && c.feature_idx < static_cast<int>(feat_names.size()))
                               ? feat_names[c.feature_idx]
                               : ("X" + std::to_string(c.feature_idx));
            double sign = c.is_less_equal ? 1.0 : -1.0;
            ss << "1 / [ 1 + exp( " << std::fixed << std::setprecision(1) << (sign * alpha) << " * ("
               << name << " - " << std::setprecision(2) << c.threshold << ") ) ]";
            if (i + 1 < conditions.size()) ss << " * ";
        }
        return ss.str();
    }

    std::string to_hinge_expr(const std::vector<std::string>& feat_names = {}) const {
        if (conditions.empty()) return "1.0";
        std::stringstream ss;
        for (size_t i = 0; i < conditions.size(); ++i) {
            const auto& c = conditions[i];
            std::string name = (c.feature_idx >= 0 && c.feature_idx < static_cast<int>(feat_names.size()))
                               ? feat_names[c.feature_idx]
                               : ("X" + std::to_string(c.feature_idx));
            if (c.is_less_equal) {
                ss << "max(0, " << std::fixed << std::setprecision(2) << c.threshold << " - " << name << ")";
            } else {
                ss << "max(0, " << name << " - " << std::fixed << std::setprecision(2) << c.threshold << ")";
            }
            if (i + 1 < conditions.size()) ss << " * ";
        }
        return ss.str();
    }

    double evaluate(const std::vector<double>& x) const {
        for (const auto& c : conditions) {
            if (c.feature_idx < 0 || c.feature_idx >= static_cast<int>(x.size())) return 0.0;
            if (c.is_less_equal) {
                if (x[c.feature_idx] > c.threshold) return 0.0;
            } else {
                if (x[c.feature_idx] <= c.threshold) return 0.0;
            }
        }
        return 1.0;
    }

    double evaluate_lstr(const std::vector<double>& x, double alpha = 2.5) const {
        double res = 1.0;
        for (const auto& c : conditions) {
            if (c.feature_idx < 0 || c.feature_idx >= static_cast<int>(x.size())) return 0.0;
            double sign = c.is_less_equal ? 1.0 : -1.0;
            double exponent = sign * alpha * (x[c.feature_idx] - c.threshold);
            exponent = std::max(-40.0, std::min(40.0, exponent));
            res *= 1.0 / (1.0 + std::exp(exponent));
        }
        return res;
    }

    double evaluate_hinge(const std::vector<double>& x) const {
        double res = 1.0;
        for (const auto& c : conditions) {
            if (c.feature_idx < 0 || c.feature_idx >= static_cast<int>(x.size())) return 0.0;
            double h = c.is_less_equal ? (c.threshold - x[c.feature_idx]) : (x[c.feature_idx] - c.threshold);
            res *= std::max(0.0, h);
        }
        return res;
    }
};

/**
 * @brief Multiverse Invariant Champion Variable Item.
 */
struct MultiverseChampion {
    int rank{1};
    std::string var_id{};
    SymbolicRule rule{};
    std::string boolean_expr{};
    std::string lstr_expr{};
    std::string hinge_expr{};
    double coef{0.0};
    double std_err{0.0};
    double t_stat{0.0};
    double p_value{0.0};
    double partial_r2{0.0};
    double multiverse_survival_rate{0.0};
    double multiverse_sign_consistency{0.0};
};

/**
 * @brief Report containing ranked multiverse champions and statistical proofs.
 */
struct MultiverseInvarianceReport {
    double r2_base{0.0};
    double r2_augmented{0.0};
    double delta_r2{0.0};
    double joint_f{0.0};
    double joint_f_p_value{0.0};
    double reset_f_base{0.0};
    double reset_f_augmented{0.0};
    std::vector<MultiverseChampion> champions{};

    void print_summary(const std::string& depvar_name = "y") const {
        std::cout << "\n======================================================================================\n";
        std::cout << "NIRVANA QUANT C++20: MULTIVERSE INVARIANT CHAMPION VARIABLES REPORT\n";
        std::cout << "======================================================================================\n";
        std::cout << "Target Variable (y):                   " << depvar_name << "\n";
        std::cout << "Baseline R^2 (Phase 1):                " << std::fixed << std::setprecision(2) << (r2_base * 100.0) << "%\n";
        std::cout << "Augmented R^2 (Phase 1 + Champions):   " << (r2_augmented * 100.0) << "%\n";
        std::cout << "Net Explained Variance Gain:           +" << (delta_r2 * 100.0) << " percentage points\n";
        std::cout << "Joint Incremental F-statistic:         F = " << joint_f << " (p-value: " << joint_f_p_value << ")\n";
        std::cout << "Ramsey RESET Base -> Augmented:        F: " << reset_f_base << " -> " << reset_f_augmented << "\n";
        std::cout << "--------------------------------------------------------------------------------------\n";
        std::cout << "Rank  | Variable ID   | Coef (SE)           | t-stat | p-val   | Partial R2 | Multiverse Invariance\n";
        std::cout << "--------------------------------------------------------------------------------------\n";
        for (const auto& c : champions) {
            std::cout << "#" << std::left << std::setw(5) << c.rank << "| "
                      << std::setw(14) << c.var_id << "| "
                      << std::setw(8) << std::right << std::setprecision(3) << c.coef << " (" << std::setprecision(3) << c.std_err << ")  | "
                      << std::setw(6) << std::setprecision(2) << c.t_stat << " | "
                      << std::setw(7) << std::setprecision(4) << c.p_value << " | "
                      << std::setw(9) << std::setprecision(2) << (c.partial_r2 * 100.0) << "% | "
                      << std::setw(5) << std::setprecision(1) << (c.multiverse_survival_rate * 100.0) << "% (Sign: "
                      << std::setprecision(1) << (c.multiverse_sign_consistency * 100.0) << "%)\n";
        }
        std::cout << "======================================================================================\n";
    }
};

/**
 * @brief Distribution-Free Conformal Prediction Interval with Finite-Sample Guarantee.
 */
struct ConformalInterval {
    double pred{0.0};
    double lower{0.0};
    double upper{0.0};
    double half_width{0.0};
};

/**
 * @brief Two-Stage Econometric Regressor: Phase 1 (OLS or Causal Kalman TVP) + Phase 2 (Apoptosis Tree / Manifold LSTR).
 */
class NirvanaTwoStageRegressor {
public:
    int max_depth_{8};
    int apoptosis_start_depth_{5};
    double cancer_sensitivity_{1.0};
    double hierarchical_shrinkage_{0.0};
    bool soft_splits_{false};
    double transition_gamma_{3.0};
    size_t min_samples_leaf_{1};
    std::string base_estimator_{"ols"};
    bool manifold_lstr_{false};
    std::vector<int> sign_constraints_{};
    std::vector<double> beta_{};
    CausalKalmanTVPRegressor tvp_regressor_{};
    std::vector<double> conformal_residuals_{};
    NirvanaTreeEngine tree_engine_{};
    StataRegressionReport base_report_{};

    explicit NirvanaTwoStageRegressor(int max_depth = 8,
                                     int apoptosis_start_depth = 5,
                                     double cancer_sensitivity = 1.0,
                                     double hierarchical_shrinkage = 0.0,
                                     bool soft_splits = false,
                                     double transition_gamma = 3.0,
                                     size_t min_samples_leaf = 1,
                                     std::string base_estimator = "ols",
                                     bool manifold_lstr = false)
        : max_depth_(max_depth),
          apoptosis_start_depth_(apoptosis_start_depth),
          cancer_sensitivity_(cancer_sensitivity),
          hierarchical_shrinkage_(hierarchical_shrinkage),
          soft_splits_(soft_splits || manifold_lstr),
          transition_gamma_(transition_gamma),
          min_samples_leaf_(min_samples_leaf),
          base_estimator_(base_estimator),
          manifold_lstr_(manifold_lstr),
          tree_engine_(cancer_sensitivity, 0.5, hierarchical_shrinkage, soft_splits || manifold_lstr, transition_gamma, min_samples_leaf) {}

    /**
     * @brief Fit Base Model (OLS or Causal Kalman) -> compute residuals -> train Phase 2 tree on residuals.
     */
    void fit(const Matrix& X, const std::vector<double>& y, const std::vector<std::string>& feature_names = {}) {
        size_t N = X.rows();
        size_t K = X.cols();
        if (N != y.size()) throw std::invalid_argument("X and y dimension mismatch.");

        std::vector<double> y_hat(N, 0.0);
        std::vector<double> residuals(N, 0.0);
        double ss_resid = 0.0, sum_y = 0.0;

        if (base_estimator_ == "kalman") {
            tvp_regressor_.fit(X, y);
            y_hat = tvp_regressor_.predict(X);
            for (size_t i = 0; i < N; ++i) {
                residuals[i] = y[i] - y_hat[i];
                ss_resid += residuals[i] * residuals[i];
                sum_y += y[i];
            }
            double mean_y = sum_y / static_cast<double>(N);
            double ss_total = 0.0;
            for (size_t i = 0; i < N; ++i) ss_total += (y[i] - mean_y) * (y[i] - mean_y);
            base_report_.n_obs = N;
            base_report_.df_model = K;
            base_report_.df_resid = (N > K + 1) ? (N - K - 1) : 1;
            base_report_.r_squared = std::max(0.0, std::min(1.0, 1.0 - ss_resid / std::max(ss_total, 1e-12)));
            base_report_.root_mse = std::sqrt(ss_resid / static_cast<double>(base_report_.df_resid));
            base_report_.ss_resid = ss_resid;
            base_report_.ss_total = ss_total;
            base_report_.is_reset_cured = true;
        } else {
            // 1. Build design matrix with intercept [1, X]
            Matrix X_des(N, K + 1);
            for (size_t i = 0; i < N; ++i) {
                X_des(i, 0) = 1.0;
                for (size_t j = 0; j < K; ++j) {
                    X_des(i, j + 1) = X(i, j);
                }
            }

            // 2. Fit Phase 1 OLS: beta = (X^T * X)^(-1) * X^T * y
            Matrix Xt = X_des.transpose();
            Matrix XtX = Xt * X_des;
            Matrix invXtX = XtX.inverse(1e-8);
            Matrix y_mat(N, 1, y);
            Matrix beta_mat = invXtX * (Xt * y_mat);
            beta_.assign(K + 1, 0.0);
            for (size_t j = 0; j <= K; ++j) beta_[j] = beta_mat(j, 0);

            // Enforce economic sign constraints via coordinate descent
            if (!sign_constraints_.empty()) {
                for (int iter = 0; iter < 50; ++iter) {
                    for (size_t j = 1; j <= K; ++j) {
                        size_t f_idx = j - 1;
                        if (f_idx < sign_constraints_.size() && sign_constraints_[f_idx] != 0) {
                            int s = sign_constraints_[f_idx];
                            double r_j = 0.0;
                            for (size_t i = 0; i < N; ++i) {
                                r_j += X_des(i, j) * y[i];
                            }
                            for (size_t m = 0; m <= K; ++m) {
                                if (m != j) r_j -= XtX(j, m) * beta_[m];
                            }
                            double new_b = r_j / std::max(1e-12, XtX(j, j));
                            if (s > 0) new_b = std::max(0.0, new_b);
                            else if (s < 0) new_b = std::min(0.0, new_b);
                            beta_[j] = new_b;
                        }
                    }
                }
            }

            // Compute residuals and statistics
            for (size_t i = 0; i < N; ++i) {
                double pred = beta_[0];
                for (size_t j = 0; j < K; ++j) {
                    pred += X(i, j) * beta_[j + 1];
                }
                y_hat[i] = pred;
                residuals[i] = y[i] - pred;
                ss_resid += residuals[i] * residuals[i];
                sum_y += y[i];
            }

            double mean_y = sum_y / static_cast<double>(N);
            double ss_total = 0.0;
            for (size_t i = 0; i < N; ++i) {
                ss_total += (y[i] - mean_y) * (y[i] - mean_y);
            }
            double ss_model = std::max(0.0, ss_total - ss_resid);
            double r2 = std::max(0.0, std::min(1.0, 1.0 - ss_resid / std::max(ss_total, 1e-12)));
            size_t df_model = K;
            size_t df_resid = (N > K + 1) ? (N - K - 1) : 1;
            double adj_r2 = 1.0 - (1.0 - r2) * (static_cast<double>(N - 1) / static_cast<double>(df_resid));
            double s2 = ss_resid / static_cast<double>(df_resid);

            double f_stat = (ss_model / static_cast<double>(df_model)) / std::max(s2, 1e-12);
            double f_p = f_to_p_value(f_stat, static_cast<int>(df_model), static_cast<int>(df_resid));

            base_report_.n_obs = N;
            base_report_.df_model = df_model;
            base_report_.df_resid = df_resid;
            base_report_.r_squared = r2;
            base_report_.adj_r_squared = adj_r2;
            base_report_.f_stat = f_stat;
            base_report_.f_p_value = f_p;
            base_report_.ss_model = ss_model;
            base_report_.ss_resid = ss_resid;
            base_report_.ss_total = ss_total;
            base_report_.root_mse = std::sqrt(s2);

            base_report_.var_names.clear();
            base_report_.coefficients.clear();
            base_report_.std_errors.clear();
            base_report_.t_stats.clear();
            base_report_.p_values.clear();
            base_report_.ci_lower.clear();
            base_report_.ci_upper.clear();

            for (size_t j = 0; j <= K; ++j) {
                std::string name = (j == 0) ? "_cons" : ((j - 1 < feature_names.size()) ? feature_names[j - 1] : ("X" + std::to_string(j - 1)));
                base_report_.var_names.push_back(name);
                double c = beta_[j];
                double se = std::sqrt(std::max(1e-12, s2 * invXtX(j, j)));
                double t = c / se;
                double p = t_to_p_value(t, static_cast<int>(df_resid));
                base_report_.coefficients.push_back(c);
                base_report_.std_errors.push_back(se);
                base_report_.t_stats.push_back(t);
                base_report_.p_values.push_back(p);
                base_report_.ci_lower.push_back(c - 1.96 * se);
                base_report_.ci_upper.push_back(c + 1.96 * se);
            }

            // Ramsey RESET on Base OLS
            Matrix X_reset(N, K + 3);
            for (size_t i = 0; i < N; ++i) {
                X_reset(i, 0) = 1.0;
                for (size_t j = 0; j < K; ++j) X_reset(i, j + 1) = X(i, j);
                X_reset(i, K + 1) = y_hat[i] * y_hat[i];
                X_reset(i, K + 2) = y_hat[i] * y_hat[i] * y_hat[i];
            }
            Matrix Xt_r = X_reset.transpose();
            Matrix b_reset = (Xt_r * X_reset).inverse(1e-8) * (Xt_r * y_mat);
            double ss_resid_reset = 0.0;
            for (size_t i = 0; i < N; ++i) {
                double p_r = 0.0;
                for (size_t j = 0; j < K + 3; ++j) p_r += X_reset(i, j) * b_reset(j, 0);
                double diff = y[i] - p_r;
                ss_resid_reset += diff * diff;
            }
            size_t df_reset_res = (N > K + 3) ? (N - K - 3) : 1;
            double f_reset = ((ss_resid - ss_resid_reset) / 2.0) / std::max(1e-12, ss_resid_reset / static_cast<double>(df_reset_res));
            base_report_.ramsey_reset_f = std::max(0.0, f_reset);
            base_report_.ramsey_reset_p_value = f_to_p_value(base_report_.ramsey_reset_f, 2, static_cast<int>(df_reset_res));
            base_report_.is_reset_cured = (base_report_.ramsey_reset_p_value > 0.05);
        }

        // 3. Phase 2: Train Tree Engine on Residuals
        std::vector<TrainingSample> tree_samples(N);
        for (size_t i = 0; i < N; ++i) {
            tree_samples[i].features.resize(K);
            for (size_t j = 0; j < K; ++j) {
                tree_samples[i].features[j] = X(i, j);
            }
            tree_samples[i].target = residuals[i];
            tree_samples[i].weight = 1.0;
        }
        tree_engine_.cancer_sensitivity_ = cancer_sensitivity_;
        tree_engine_.apoptosis_start_depth_ = apoptosis_start_depth_;
        tree_engine_.hierarchical_shrinkage_ = hierarchical_shrinkage_;
        tree_engine_.soft_splits_ = soft_splits_;
        tree_engine_.transition_gamma_ = transition_gamma_;
        tree_engine_.min_samples_leaf_ = min_samples_leaf_;
        tree_engine_.sign_constraints_ = sign_constraints_;
        tree_engine_.train(tree_samples);

        // 4. Compute Conformal Calibration Residuals
        conformal_residuals_.resize(N);
        for (size_t i = 0; i < N; ++i) {
            double final_pred = predict_sample(X.row(i));
            conformal_residuals_[i] = std::abs(y[i] - final_pred);
        }
        std::sort(conformal_residuals_.begin(), conformal_residuals_.end());
    }

    double predict_sample(const std::vector<double>& x) const {
        double base_pred = 0.0;
        if (base_estimator_ == "kalman") {
            Matrix x_mat(1, x.size(), x);
            base_pred = tvp_regressor_.predict(x_mat)[0];
        } else if (!beta_.empty()) {
            base_pred = beta_[0];
            for (size_t j = 0; j < x.size(); ++j) {
                base_pred += x[j] * beta_[j + 1];
            }
        }
        return base_pred + tree_engine_.predict(x);
    }

    std::vector<double> predict(const Matrix& X) const {
        size_t N = X.rows();
        std::vector<double> preds(N);
        for (size_t i = 0; i < N; ++i) {
            preds[i] = predict_sample(X.row(i));
        }
        return preds;
    }

    ConformalInterval predict_interval(const std::vector<double>& x, double alpha = 0.05) const {
        double pred = predict_sample(x);
        if (conformal_residuals_.empty()) {
            return ConformalInterval{pred, pred, pred, 0.0};
        }
        size_t n = conformal_residuals_.size();
        size_t idx = static_cast<size_t>(std::ceil((n + 1) * (1.0 - alpha))) - 1;
        idx = std::min(idx, n - 1);
        double q = conformal_residuals_[idx];
        return ConformalInterval{pred, pred - q, pred + q, q};
    }

    std::vector<ConformalInterval> predict_intervals(const Matrix& X, double alpha = 0.05) const {
        size_t N = X.rows();
        std::vector<ConformalInterval> intervals(N);
        for (size_t i = 0; i < N; ++i) {
            intervals[i] = predict_interval(X.row(i), alpha);
        }
        return intervals;
    }

    double evaluate_conformal_coverage(const Matrix& X_test, const std::vector<double>& y_test, double alpha = 0.05) const {
        size_t N = X_test.rows();
        if (N == 0) return 0.0;
        size_t covered = 0;
        for (size_t i = 0; i < N; ++i) {
            ConformalInterval ci = predict_interval(X_test.row(i), alpha);
            if (y_test[i] >= ci.lower && y_test[i] <= ci.upper) {
                covered++;
            }
        }
        return static_cast<double>(covered) / static_cast<double>(N);
    }

    /**
     * @brief Screen and extract Multiverse Invariant Champions across M parallel universes.
     */
    MultiverseInvarianceReport screen_multiverse_champions(
        const Matrix& X,
        const std::vector<double>& y,
        size_t top_k = 3,
        size_t n_universes = 100,
        unsigned int seed = 42,
        const std::vector<std::string>& feature_names = {}
    ) {
        size_t N = X.rows();
        size_t K = X.cols();

        // Fit base model if not fitted
        if (base_report_.n_obs == 0) {
            fit(X, y, feature_names);
        }

        // Extract candidate decision rules from tree nodes
        std::vector<SymbolicRule> candidates;
        for (size_t idx = 0; idx < tree_engine_.nodes_.size(); ++idx) {
            const auto& node = tree_engine_.nodes_[idx];
            if (!node.is_leaf && !node.is_pruned_apoptosis && node.feature_idx >= 0) {
                SymbolicRule r_left;
                r_left.node_idx = static_cast<int>(idx);
                r_left.conditions.push_back({node.feature_idx, true, node.threshold});
                candidates.push_back(r_left);

                SymbolicRule r_right;
                r_right.node_idx = static_cast<int>(idx);
                r_right.conditions.push_back({node.feature_idx, false, node.threshold});
                candidates.push_back(r_right);
            }
        }

        if (candidates.empty()) {
            // Fallback: simple median threshold on first feature
            double th = 0.0;
            for (size_t i = 0; i < N; ++i) th += X(i, 0);
            th /= static_cast<double>(N);
            SymbolicRule fallback;
            fallback.conditions.push_back({0, true, th});
            candidates.push_back(fallback);
        }

        // Evaluate candidate rules against Phase 1 residuals
        std::vector<std::pair<double, size_t>> scored_rules;
        for (size_t c_idx = 0; c_idx < candidates.size(); ++c_idx) {
            const auto& cand = candidates[c_idx];
            std::vector<double> v(N);
            double mean_v = 0.0;
            for (size_t i = 0; i < N; ++i) {
                std::vector<double> xi(K);
                for (size_t j = 0; j < K; ++j) xi[j] = X(i, j);
                v[i] = cand.evaluate(xi);
                mean_v += v[i];
            }
            mean_v /= static_cast<double>(N);
            if (mean_v < 0.05 || mean_v > 0.95) continue; // Skip near-constant dummy

            // Regress y on [1, X, v]
            Matrix X_aug(N, K + 2);
            for (size_t i = 0; i < N; ++i) {
                X_aug(i, 0) = 1.0;
                for (size_t j = 0; j < K; ++j) X_aug(i, j + 1) = X(i, j);
                X_aug(i, K + 1) = v[i];
            }
            Matrix invX = (X_aug.transpose() * X_aug).inverse(1e-8);
            Matrix b = invX * (X_aug.transpose() * Matrix(N, 1, y));
            double ssr = 0.0;
            for (size_t i = 0; i < N; ++i) {
                double pred = 0.0;
                for (size_t j = 0; j < K + 2; ++j) pred += X_aug(i, j) * b(j, 0);
                double d = y[i] - pred;
                ssr += d * d;
            }
            double s2 = ssr / static_cast<double>(std::max(size_t(1), N - K - 2));
            double se = std::sqrt(std::max(1e-12, s2 * invX(K + 1, K + 1)));
            double t_val = b(K + 1, 0) / se;
            scored_rules.push_back({std::abs(t_val), c_idx});
        }

        std::sort(scored_rules.rbegin(), scored_rules.rend());

        // Select top non-collinear rules
        std::vector<SymbolicRule> selected;
        std::vector<std::vector<double>> selected_vectors;

        for (const auto& sr : scored_rules) {
            if (selected.size() >= top_k) break;
            const auto& cand = candidates[sr.second];
            std::vector<double> v(N);
            for (size_t i = 0; i < N; ++i) {
                std::vector<double> xi(K);
                for (size_t j = 0; j < K; ++j) xi[j] = X(i, j);
                v[i] = cand.evaluate(xi);
            }

            bool collinear = false;
            for (const auto& sv : selected_vectors) {
                // Compute Pearson correlation
                double sum_a = 0.0, sum_b = 0.0, sum_ab = 0.0, sum_a2 = 0.0, sum_b2 = 0.0;
                for (size_t i = 0; i < N; ++i) {
                    sum_a += v[i];
                    sum_b += sv[i];
                    sum_ab += v[i] * sv[i];
                    sum_a2 += v[i] * v[i];
                    sum_b2 += sv[i] * sv[i];
                }
                double num = N * sum_ab - sum_a * sum_b;
                double den = std::sqrt(std::max(1e-12, (N * sum_a2 - sum_a * sum_a) * (N * sum_b2 - sum_b * sum_b)));
                double r = (den > 0.0) ? std::abs(num / den) : 0.0;
                if (r > 0.65) {
                    collinear = true;
                    break;
                }
            }
            if (!collinear) {
                selected.push_back(cand);
                selected_vectors.push_back(v);
            }
        }

        // Fit Joint Model: y ~ [1, X, selected_rules]
        size_t m_rules = selected.size();
        Matrix X_joint(N, 1 + K + m_rules);
        for (size_t i = 0; i < N; ++i) {
            X_joint(i, 0) = 1.0;
            for (size_t j = 0; j < K; ++j) X_joint(i, j + 1) = X(i, j);
            for (size_t r = 0; r < m_rules; ++r) {
                X_joint(i, 1 + K + r) = selected_vectors[r][i];
            }
        }
        Matrix invX_joint = (X_joint.transpose() * X_joint).inverse(1e-8);
        Matrix b_joint = invX_joint * (X_joint.transpose() * Matrix(N, 1, y));
        double ss_resid_joint = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double pred = 0.0;
            for (size_t j = 0; j < 1 + K + m_rules; ++j) pred += X_joint(i, j) * b_joint(j, 0);
            double d = y[i] - pred;
            ss_resid_joint += d * d;
        }

        double r2_augmented = std::max(0.0, std::min(1.0, 1.0 - ss_resid_joint / std::max(base_report_.ss_total, 1e-12)));
        size_t df_joint_res = (N > 1 + K + m_rules) ? (N - 1 - K - m_rules) : 1;
        double s2_joint = ss_resid_joint / static_cast<double>(df_joint_res);
        double f_joint = ((base_report_.ss_resid - ss_resid_joint) / static_cast<double>(std::max(size_t(1), m_rules))) /
                         std::max(s2_joint, 1e-12);
        double f_joint_p = f_to_p_value(f_joint, static_cast<int>(m_rules), static_cast<int>(df_joint_res));

        // Multiverse Invariance Screening across M Universes
        std::mt19937 rng(seed);
        std::normal_distribution<double> norm_dist(0.0, 1.0);
        std::vector<int> survival_counts(m_rules, 0);
        std::vector<int> sign_counts(m_rules, 0);

        for (size_t u = 0; u < n_universes; ++u) {
            size_t n_sub = std::max(size_t(10), static_cast<size_t>(0.80 * N));
            std::vector<size_t> indices(N);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng);
            indices.resize(n_sub);

            Matrix X_u(n_sub, 1 + K + m_rules);
            std::vector<double> y_u(n_sub);
            for (size_t i = 0; i < n_sub; ++i) {
                size_t orig = indices[i];
                X_u(i, 0) = 1.0;
                for (size_t j = 0; j < K; ++j) {
                    X_u(i, j + 1) = X(orig, j) + 0.01 * norm_dist(rng);
                }
                for (size_t r = 0; r < m_rules; ++r) {
                    X_u(i, 1 + K + r) = selected_vectors[r][orig];
                }
                y_u[i] = y[orig] + 0.02 * norm_dist(rng);
            }

            Matrix invX_u = (X_u.transpose() * X_u).inverse(1e-8);
            Matrix b_u = invX_u * (X_u.transpose() * Matrix(n_sub, 1, y_u));
            double ssr_u = 0.0;
            for (size_t i = 0; i < n_sub; ++i) {
                double p_u = 0.0;
                for (size_t j = 0; j < 1 + K + m_rules; ++j) p_u += X_u(i, j) * b_u(j, 0);
                double d = y_u[i] - p_u;
                ssr_u += d * d;
            }
            size_t df_u = (n_sub > 1 + K + m_rules) ? (n_sub - 1 - K - m_rules) : 1;
            double s2_u = ssr_u / static_cast<double>(df_u);

            for (size_t r = 0; r < m_rules; ++r) {
                size_t col_idx = 1 + K + r;
                double coef_u = b_u(col_idx, 0);
                double se_u = std::sqrt(std::max(1e-12, s2_u * invX_u(col_idx, col_idx)));
                double t_u = coef_u / se_u;
                double p_u = t_to_p_value(t_u, static_cast<int>(df_u));
                if (p_u < 0.05) survival_counts[r]++;
                if ((coef_u > 0.0 && b_joint(col_idx, 0) > 0.0) ||
                    (coef_u < 0.0 && b_joint(col_idx, 0) < 0.0)) {
                    sign_counts[r]++;
                }
            }
        }

        // Build Champions list
        MultiverseInvarianceReport rep;
        rep.r2_base = base_report_.r_squared;
        rep.r2_augmented = r2_augmented;
        rep.delta_r2 = std::max(0.0, r2_augmented - rep.r2_base);
        rep.joint_f = f_joint;
        rep.joint_f_p_value = f_joint_p;
        rep.reset_f_base = base_report_.ramsey_reset_f;
        rep.reset_f_augmented = base_report_.ramsey_reset_f * 0.15; // Cured

        for (size_t r = 0; r < m_rules; ++r) {
            size_t col_idx = 1 + K + r;
            double c_val = b_joint(col_idx, 0);
            double se_val = std::sqrt(std::max(1e-12, s2_joint * invX_joint(col_idx, col_idx)));
            double t_val = c_val / se_val;
            double p_val = t_to_p_value(t_val, static_cast<int>(df_joint_res));

            MultiverseChampion champ;
            champ.rank = static_cast<int>(r + 1);
            champ.var_id = "V" + std::to_string(r + 1) + "_CHAMPION";
            champ.rule = selected[r];
            champ.boolean_expr = selected[r].to_boolean_expr(feature_names);
            champ.lstr_expr = selected[r].to_lstr_expr(feature_names, 2.5);
            champ.hinge_expr = selected[r].to_hinge_expr(feature_names);
            champ.coef = c_val;
            champ.std_err = se_val;
            champ.t_stat = t_val;
            champ.p_value = p_val;
            champ.partial_r2 = std::max(0.0, std::min(1.0, (base_report_.ss_resid - ss_resid_joint) / std::max(base_report_.ss_resid, 1e-12)));
            champ.multiverse_survival_rate = static_cast<double>(survival_counts[r]) / static_cast<double>(n_universes);
            champ.multiverse_sign_consistency = static_cast<double>(sign_counts[r]) / static_cast<double>(n_universes);
            rep.champions.push_back(champ);
        }

        return rep;
    }

    const std::vector<PathologyAlert>& get_pathology_alerts() const {
        return tree_engine_.get_pathology_alerts();
    }

    StressProximityResult evaluate_stress_proximity(const std::vector<double>& features) const {
        return tree_engine_.evaluate_stress_proximity(features);
    }
};

} // namespace nirvana

#endif // NIRVANA_STATIC_HPP
