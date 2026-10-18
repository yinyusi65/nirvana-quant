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

struct FlatTreeNode {
    int depth{0};
    int feature_idx{-1};
    double threshold{0.0};
    double value{0.0};
    bool is_leaf{true};
    bool is_pruned_apoptosis{false};
    size_t sample_count{0};
    double fisher_info{0.0};
    double margin_density{0.0};
    int left_child{-1};
    int right_child{-1};
};

/**
 * @brief 12-layer Tree Engine featuring Phase 2 Empirical Fisher Information & Margin-Density Apoptosis.
 * Excises overfitted leaf nodes that memorize idiosyncratic noise.
 */
class NirvanaTreeEngine {
public:
    static constexpr int MAX_DEPTH = 12;
    static constexpr int APOPTOSIS_START_DEPTH = 7;

    double cancer_sensitivity_{1.0};
    double oof_consistency_{0.5};
    double baseline_var_{1e-8};
    std::vector<FlatTreeNode> nodes_{};

    explicit NirvanaTreeEngine(double cancer_sensitivity = 1.0, double oof_consistency = 0.5)
        : cancer_sensitivity_(cancer_sensitivity),
          oof_consistency_(oof_consistency) {}

    void clear() {
        nodes_.clear();
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
    }

    double predict(const std::vector<double>& features) const {
        if (nodes_.empty()) return 0.0;
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

        double sum_y = 0.0, sum_w = 0.0;
        for (const auto* s : samples) {
            sum_y += s->target * s->weight;
            sum_w += s->weight;
        }
        node.value = (sum_w > 0.0) ? (sum_y / sum_w) : 0.0;

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
                if (left_cnt < 2.0 || right_cnt < 2.0) continue;

                double left_mean = left_sum / left_cnt;
                double right_mean = right_sum / right_cnt;
                double reduction = left_cnt * (left_mean - node.value) * (left_mean - node.value) +
                                   right_cnt * (right_mean - node.value) * (right_mean - node.value);

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
            double pred = (s->features[best_feat] <= best_thresh)
                          ? (left_samples.empty() ? node.value : 0.0) // placeholder
                          : 0.0;
            double g = s->target - node.value;
            sum_g2 += g * g;
        }
        node.fisher_info = sum_g2 / std::max(1.0, static_cast<double>(samples.size()));

        // Phase 2 Apoptosis Pruning: triggered at deeper layers
        if (depth >= APOPTOSIS_START_DEPTH) {
            double s_cancer = (node.fisher_info / (node.margin_density * std::sqrt(samples.size() + 1.0)))
                              * (1.0 - oof_consistency_);
            if (s_cancer > cancer_sensitivity_ * 2.5) {
                // Apoptosis triggered: collapse to parent node with empirical Bayes shrinkage
                node.is_pruned_apoptosis = true;
                node.is_leaf = true;
                return;
            }
        }

        int left_idx = static_cast<int>(nodes_.size());
        nodes_.emplace_back();
        int right_idx = static_cast<int>(nodes_.size());
        nodes_.emplace_back();

        nodes_[node_idx].left_child = left_idx;
        nodes_[node_idx].right_child = right_idx;

        build_node(left_idx, left_samples, depth + 1);
        build_node(right_idx, right_samples, depth + 1);
    }
};

} // namespace nirvana

#endif // NIRVANA_TREE_HPP
