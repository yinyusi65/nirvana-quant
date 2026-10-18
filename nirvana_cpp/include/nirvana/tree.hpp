#ifndef NIRVANA_TREE_HPP
#define NIRVANA_TREE_HPP

#include "types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

namespace nirvana {

struct TrainingSample {
    std::vector<double> features;
    double target{0.0};
    double weight{1.0};
};

struct PathologyAlert {
    int node_idx{0};
    int depth{0};
    double divergence_sigma{0.0};
    double margin_density{0.0};
    size_t sample_count{0};
    std::vector<double> feature_fingerprint{};
    std::string trigger_rule{""};
    std::string diagnosis{"Idiosyncratic Extreme Divergence / Latent Counterfactual Fragility Excised by Apoptosis"};
};

struct StressProximityResult {
    double proximity_pct{0.0};
    int nearest_case_idx{-1};
    int nearest_depth{-1};
    std::string risk_level{"NORMAL"};
    double distance{999.0};
    std::vector<double> nearest_fingerprint{};
    std::string diagnosis{""};
};

struct FlatTreeNode {
    int depth{0};
    int feature_idx{-1};
    double threshold{0.0};
    double value{0.0};
    double raw_value{0.0};
    int parent_idx{-1};
    bool is_leaf{true};
    bool is_pruned_apoptosis{false};
    size_t sample_count{0};
    double fisher_info{0.0};
    double margin_density{0.0};
    double feature_std{1.0};
    double shrinkage_lambda{1.0};
    std::vector<double> feature_fingerprint{};
    int left_child{-1};
    int right_child{-1};
};

/**
 * @brief 12-layer Tree Engine featuring Phase 2 Empirical Fisher Information & Margin-Density Apoptosis,
 * Hierarchical Bayesian James-Stein Shrinkage, Soft LSTR Regimes, and Economic Sign Constraints.
 */
class NirvanaTreeEngine {
public:
    static constexpr int MAX_DEPTH = 12;
    static constexpr int APOPTOSIS_START_DEPTH = 7;

    double cancer_sensitivity_{1.0};
    double oof_consistency_{0.5};
    int apoptosis_start_depth_{APOPTOSIS_START_DEPTH};
    double baseline_var_{1e-8};
    double hierarchical_shrinkage_{0.0};
    bool soft_splits_{false};
    double transition_gamma_{3.0};
    size_t min_samples_leaf_{1};
    std::vector<int> sign_constraints_{};
    std::vector<FlatTreeNode> nodes_{};
    std::vector<PathologyAlert> pathology_alerts_{};

    explicit NirvanaTreeEngine(double cancer_sensitivity = 1.0,
                               double oof_consistency = 0.5,
                               double hierarchical_shrinkage = 0.0,
                               bool soft_splits = false,
                               double transition_gamma = 3.0,
                               size_t min_samples_leaf = 1,
                               int apoptosis_start_depth = APOPTOSIS_START_DEPTH)
        : cancer_sensitivity_(cancer_sensitivity),
          oof_consistency_(oof_consistency),
          apoptosis_start_depth_(apoptosis_start_depth),
          hierarchical_shrinkage_(hierarchical_shrinkage),
          soft_splits_(soft_splits),
          transition_gamma_(transition_gamma),
          min_samples_leaf_(min_samples_leaf) {}

    void clear() {
        nodes_.clear();
        pathology_alerts_.clear();
    }

    const std::vector<PathologyAlert>& get_pathology_alerts() const {
        return pathology_alerts_;
    }

    StressProximityResult evaluate_stress_proximity(const std::vector<double>& features) const {
        if (pathology_alerts_.empty()) {
            return StressProximityResult{0.0, -1, -1, "NORMAL", 999.0, {}, ""};
        }

        double min_dist = std::numeric_limits<double>::infinity();
        size_t nearest_idx = 0;

        for (size_t i = 0; i < pathology_alerts_.size(); ++i) {
            const auto& alert = pathology_alerts_[i];
            if (alert.feature_fingerprint.size() == features.size()) {
                double d2 = 0.0;
                for (size_t j = 0; j < features.size(); ++j) {
                    double diff = features[j] - alert.feature_fingerprint[j];
                    d2 += diff * diff;
                }
                double dist = std::sqrt(d2);
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_idx = i;
                }
            }
        }

        if (std::isinf(min_dist)) {
            return StressProximityResult{0.0, -1, -1, "NORMAL", 999.0, {}, ""};
        }

        double scale = std::max(std::sqrt(static_cast<double>(features.size())), 1.0);
        double proximity = 100.0 * std::exp(-min_dist / scale);
        std::string risk = (proximity >= 80.0) ? "CRITICAL" : ((proximity >= 50.0) ? "ELEVATED" : "NORMAL");

        const auto& nearest = pathology_alerts_[nearest_idx];
        return StressProximityResult{
            proximity,
            nearest.node_idx,
            nearest.depth,
            risk,
            min_dist,
            nearest.feature_fingerprint,
            nearest.diagnosis
        };
    }

    void train(const std::vector<TrainingSample>& dataset) {
        clear();
        if (dataset.empty()) return;

        // Filter valid samples
        std::vector<const TrainingSample*> clean_samples;
        for (const auto& s : dataset) {
            if (!std::isnan(s.target)) {
                bool valid_f = true;
                for (double f : s.features) {
                    if (std::isnan(f)) { valid_f = false; break; }
                }
                if (valid_f) clean_samples.push_back(&s);
            }
        }
        if (clean_samples.empty()) return;

        // Compute baseline variance
        double sum_y = 0.0, sum_y2 = 0.0;
        for (const auto* s : clean_samples) {
            sum_y += s->target;
            sum_y2 += s->target * s->target;
        }
        double mean_y = sum_y / clean_samples.size();
        baseline_var_ = std::max(1e-8, (sum_y2 / clean_samples.size()) - (mean_y * mean_y));

        nodes_.emplace_back();
        build_node(0, clean_samples, 0);
        run_apoptosis_audit();
    }

    void run_apoptosis_audit() {
        if (nodes_.empty()) return;
        double root_val = nodes_[0].value;
        double root_count = std::max(1.0, static_cast<double>(nodes_[0].sample_count));
        pathology_alerts_.clear();

        for (size_t i = 0; i < nodes_.size(); ++i) {
            auto& node = nodes_[i];
            if (node.depth >= apoptosis_start_depth_) {
                double diff = node.value - root_val;
                double fisher = (diff * diff) / (baseline_var_ + 1e-9);
                double margin = static_cast<double>(node.sample_count) / root_count;
                node.fisher_info = fisher;
                node.margin_density = margin;

                if (fisher * cancer_sensitivity_ > 15.0 && margin < 0.05) {
                    node.is_pruned_apoptosis = true;

                    PathologyAlert alert;
                    alert.node_idx = static_cast<int>(i);
                    alert.depth = node.depth;
                    alert.divergence_sigma = std::sqrt(std::max(0.0, fisher));
                    alert.margin_density = margin;
                    alert.sample_count = node.sample_count;
                    alert.feature_fingerprint = node.feature_fingerprint;
                    if (node.feature_idx >= 0) {
                        alert.trigger_rule = "Feature " + std::to_string(node.feature_idx) + " <= " + std::to_string(node.threshold);
                    }
                    pathology_alerts_.push_back(alert);
                }
            }
        }
    }

    double predict_soft(const std::vector<double>& features) const {
        if (nodes_.empty()) return 0.0;
        std::vector<std::pair<int, double>> stack;
        stack.emplace_back(0, 1.0);
        double total_val = 0.0;
        double total_w = 0.0;
        while (!stack.empty()) {
            auto [curr, prob] = stack.back();
            stack.pop_back();
            if (prob < 1e-6) continue;
            const auto& node = nodes_[curr];
            if (node.is_leaf || node.is_pruned_apoptosis || node.feature_idx < 0) {
                total_val += prob * node.value;
                total_w += prob;
                continue;
            }
            double f_val = (node.feature_idx < static_cast<int>(features.size())) ? features[node.feature_idx] : 0.0;
            double scale = std::max(node.feature_std, 1e-4);
            double z = transition_gamma_ * (f_val - node.threshold) / scale;
            z = std::max(-25.0, std::min(25.0, z));
            double p_right = 1.0 / (1.0 + std::exp(-z));
            double p_left = 1.0 - p_right;
            if (node.left_child >= 0) stack.emplace_back(node.left_child, prob * p_left);
            if (node.right_child >= 0) stack.emplace_back(node.right_child, prob * p_right);
        }
        return (total_w > 0.0) ? (total_val / total_w) : nodes_[0].value;
    }

    double predict(const std::vector<double>& features) const {
        if (nodes_.empty()) return 0.0;
        if (soft_splits_) return predict_soft(features);

        int curr = 0;
        while (curr >= 0 && curr < static_cast<int>(nodes_.size())) {
            const auto& node = nodes_[curr];
            if (node.is_leaf || node.is_pruned_apoptosis) {
                return node.value;
            }
            if (node.feature_idx < 0 || node.feature_idx >= static_cast<int>(features.size())) {
                return node.value;
            }
            if (features[node.feature_idx] <= node.threshold) {
                curr = node.left_child;
            } else {
                curr = node.right_child;
            }
        }
        return nodes_[0].value;
    }

private:
    void build_node(int node_idx, const std::vector<const TrainingSample*>& samples, int depth) {
        FlatTreeNode& node = nodes_[node_idx];
        node.depth = depth;
        node.sample_count = samples.size();

        if (!samples.empty()) {
            size_t k = samples[0]->features.size();
            node.feature_fingerprint.assign(k, 0.0);
            for (const auto* s : samples) {
                for (size_t j = 0; j < k; ++j) {
                    node.feature_fingerprint[j] += s->features[j];
                }
            }
            for (size_t j = 0; j < k; ++j) {
                node.feature_fingerprint[j] /= static_cast<double>(samples.size());
            }
        }

        double sum_y = 0.0, sum_w = 0.0;
        for (const auto* s : samples) {
            sum_y += s->target * s->weight;
            sum_w += s->weight;
        }
        double raw_val = (sum_w > 0.0) ? (sum_y / sum_w) : 0.0;
        node.raw_value = raw_val;

        // Hierarchical Bayesian James-Stein Shrinkage towards parent node
        if (node.parent_idx >= 0 && hierarchical_shrinkage_ > 0.0) {
            double n_s = static_cast<double>(samples.size());
            double lambda_s = n_s / (n_s + hierarchical_shrinkage_);
            double parent_val = nodes_[node.parent_idx].value;
            node.value = lambda_s * raw_val + (1.0 - lambda_s) * parent_val;
            node.shrinkage_lambda = lambda_s;
        } else {
            node.value = raw_val;
            node.shrinkage_lambda = 1.0;
        }

        if (depth >= MAX_DEPTH || samples.size() < 4) {
            node.is_leaf = true;
            return;
        }

        // Find best split
        size_t n_feats = samples[0]->features.size();
        int best_feat = -1;
        double best_thresh = 0.0;
        double best_loss_reduction = 0.0;

        for (size_t f = 0; f < n_feats; ++f) {
            std::vector<double> vals;
            vals.reserve(samples.size());
            for (const auto* s : samples) vals.push_back(s->features[f]);
            std::sort(vals.begin(), vals.end());

            int req_s = (f < sign_constraints_.size()) ? sign_constraints_[f] : 0;

            // Test candidate quantiles
            for (double q : {0.25, 0.50, 0.75}) {
                size_t q_idx = static_cast<size_t>(q * (vals.size() - 1));
                double thresh = vals[q_idx];

                double left_sum = 0.0, left_cnt = 0.0;
                double right_sum = 0.0, right_cnt = 0.0;
                for (const auto* s : samples) {
                    if (s->features[f] <= thresh) {
                        left_sum += s->target;
                        left_cnt += 1.0;
                    } else {
                        right_sum += s->target;
                        right_cnt += 1.0;
                    }
                }
                if (left_cnt < static_cast<double>(min_samples_leaf_) ||
                    right_cnt < static_cast<double>(min_samples_leaf_)) continue;

                double left_mean = left_sum / left_cnt;
                double right_mean = right_sum / right_cnt;
                double reduction = left_cnt * (left_mean - node.value) * (left_mean - node.value) +
                                   right_cnt * (right_mean - node.value) * (right_mean - node.value);

                // Enforce economic sign constraints
                if (req_s != 0) {
                    if (req_s > 0 && right_mean < left_mean) reduction *= 0.05;
                    else if (req_s < 0 && right_mean > left_mean) reduction *= 0.05;
                }

                if (reduction > best_loss_reduction) {
                    best_loss_reduction = reduction;
                    best_feat = static_cast<int>(f);
                    best_thresh = thresh;
                }
            }
        }

        if (best_feat < 0 || best_loss_reduction < 1e-6) {
            node.is_leaf = true;
            return;
        }

        node.feature_idx = best_feat;
        node.threshold = best_thresh;
        node.is_leaf = false;

        // Compute feature std for smooth transition
        double f_sum = 0.0, f_sum2 = 0.0;
        for (const auto* s : samples) {
            double fv = s->features[best_feat];
            f_sum += fv;
            f_sum2 += fv * fv;
        }
        double f_mean = f_sum / samples.size();
        double f_var = std::max(1e-8, (f_sum2 / samples.size()) - (f_mean * f_mean));
        node.feature_std = std::sqrt(f_var);

        // Partition
        std::vector<const TrainingSample*> left_samples, right_samples;
        double min_margin = std::numeric_limits<double>::infinity();
        for (const auto* s : samples) {
            double margin = std::abs(s->features[best_feat] - best_thresh);
            if (margin < min_margin) min_margin = margin;
            if (s->features[best_feat] <= best_thresh) {
                left_samples.push_back(s);
            } else {
                right_samples.push_back(s);
            }
        }
        node.margin_density = std::max(min_margin, 1e-4);

        // Empirical Fisher Information F_N = 1/N * sum(g_i^2)
        double sum_g2 = 0.0;
        for (const auto* s : samples) {
            double g = s->target - node.value;
            sum_g2 += g * g;
        }
        node.fisher_info = sum_g2 / std::max(1.0, static_cast<double>(samples.size()));

        int left_idx = static_cast<int>(nodes_.size());
        nodes_.emplace_back();
        nodes_.back().parent_idx = node_idx;

        int right_idx = static_cast<int>(nodes_.size());
        nodes_.emplace_back();
        nodes_.back().parent_idx = node_idx;

        nodes_[node_idx].left_child = left_idx;
        nodes_[node_idx].right_child = right_idx;

        build_node(left_idx, left_samples, depth + 1);
        build_node(right_idx, right_samples, depth + 1);
    }
};

} // namespace nirvana

#endif // NIRVANA_TREE_HPP
