"""
Real-Tree Topology & Fisher Information Apoptosis Module
=========================================================
12-layer decision tree engine featuring Phase 2 Empirical Fisher Information & Margin-Density Apoptosis.
Excises cancerous/overfitting nodes that memorize noise rather than learning causal invariance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class TrainingSample:
    features: List[float]
    target: float
    weight: float = 1.0


@dataclass
class FlatTreeNode:
    depth: int = 0
    feature_idx: int = -1
    threshold: float = 0.0
    value: float = 0.0
    raw_leaf_value: float = 0.0
    parent_idx: int = -1
    is_leaf: bool = True
    is_pruned_apoptosis: bool = False
    sample_count: int = 0
    fisher_info: float = 0.0
    margin_density: float = 0.0
    feature_std: float = 1.0
    shrinkage_lambda: float = 1.0
    left_child: int = -1
    right_child: int = -1


class NirvanaTreeEngine:
    """
    12-layer Real-Data Tree Engine with Empirical Fisher Apoptosis,
    Hierarchical Bayesian James-Stein Shrinkage, Soft LSTR Regimes, and Economic Sign Constraints.
    """

    MAX_DEPTH = 12
    APOPTOSIS_START_DEPTH = 7

    def __init__(
        self,
        cancer_sensitivity: float = 1.0,
        oof_consistency: float = 0.5,
        max_depth: int = 12,
        apoptosis_start_depth: int = 7,
        split_quantiles: Optional[List[float]] = None,
        min_samples_split: int = 4,
        min_samples_leaf: int = 1,
        hierarchical_shrinkage: float = 0.0,
        soft_splits: bool = False,
        transition_gamma: float = 3.0,
        sign_constraints: Optional[Dict[Union[int, str], int]] = None,
    ):
        self.cancer_sensitivity = float(cancer_sensitivity)
        self.oof_consistency = float(oof_consistency)
        self.max_depth = int(max_depth)
        self.apoptosis_start_depth = int(apoptosis_start_depth)
        self.split_quantiles = split_quantiles if split_quantiles is not None else [10.0, 25.0, 50.0, 75.0, 90.0]
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.hierarchical_shrinkage = float(hierarchical_shrinkage)
        self.soft_splits = bool(soft_splits)
        self.transition_gamma = float(transition_gamma)
        self.sign_constraints = sign_constraints or {}
        self.nodes: List[FlatTreeNode] = []
        self.baseline_var: float = 1e-8
        self.feature_names: List[str] = []
        self.n_features_: int = 0

    def clear(self):
        self.nodes.clear()

    def train(self, dataset: List[TrainingSample]):
        self.clear()
        if not dataset:
            return

        # Clean NaN samples
        clean_dataset = [
            s for s in dataset
            if not np.isnan(s.target) and not any(np.isnan(f) for f in s.features)
        ]
        if not clean_dataset:
            return

        targets = np.array([s.target for s in clean_dataset], dtype=float)
        self.baseline_var = max(float(np.var(targets)), 1e-8)

        root = FlatTreeNode(depth=0, sample_count=len(clean_dataset))
        self.nodes.append(root)

        root_indices = list(range(len(clean_dataset)))
        self._build_sub_tree(0, root_indices, clean_dataset)

        # Run Phase 2: Empirical Fisher & Margin Density Apoptosis
        self._run_apoptosis_audit()

    def _build_sub_tree(
        self, node_idx: int, sample_indices: List[int], dataset: List[TrainingSample]
    ):
        depth = self.nodes[node_idx].depth
        n_samples = len(sample_indices)

        # Compute mean target & James-Stein Hierarchical Bayesian Shrinkage
        node_vals = [dataset[i].target for i in sample_indices]
        raw_val = float(np.mean(node_vals))
        self.nodes[node_idx].raw_leaf_value = raw_val
        self.nodes[node_idx].sample_count = n_samples

        parent_idx = self.nodes[node_idx].parent_idx
        if parent_idx >= 0 and self.hierarchical_shrinkage > 0.0:
            lambda_s = float(n_samples) / (float(n_samples) + self.hierarchical_shrinkage)
            parent_val = self.nodes[parent_idx].value
            self.nodes[node_idx].value = float(lambda_s * raw_val + (1.0 - lambda_s) * parent_val)
            self.nodes[node_idx].shrinkage_lambda = float(lambda_s)
        else:
            self.nodes[node_idx].value = raw_val
            self.nodes[node_idx].shrinkage_lambda = 1.0

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            self.nodes[node_idx].is_leaf = True
            return

        # Find best split across features
        n_features = len(dataset[0].features)
        best_gain = -1.0
        best_feat = -1
        best_thresh = 0.0
        best_left_idx: List[int] = []
        best_right_idx: List[int] = []

        curr_var = float(np.var(node_vals)) * n_samples

        # Sample feature subset or evaluate all features
        for f in range(n_features):
            vals = [dataset[i].features[f] for i in sample_indices]
            unique_vals = np.percentile(vals, self.split_quantiles)

            # Check domain economic sign constraints for feature f
            req_sign = 0
            if self.sign_constraints:
                if f in self.sign_constraints:
                    req_sign = self.sign_constraints[f]
                elif self.feature_names and f < len(self.feature_names) and self.feature_names[f] in self.sign_constraints:
                    req_sign = self.sign_constraints[self.feature_names[f]]

            for thresh in unique_vals:
                left = [i for i in sample_indices if dataset[i].features[f] <= thresh]
                right = [i for i in sample_indices if dataset[i].features[f] > thresh]

                # Enforce min_samples_leaf constraint
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue

                var_left = float(np.var([dataset[i].target for i in left])) * len(left)
                var_right = float(np.var([dataset[i].target for i in right])) * len(right)
                gain = curr_var - (var_left + var_right)

                # Penalize or discard splits violating economic monotonicity
                if req_sign != 0:
                    mean_l = float(np.mean([dataset[i].target for i in left]))
                    mean_r = float(np.mean([dataset[i].target for i in right]))
                    if req_sign > 0 and mean_r < mean_l:
                        gain *= 0.05
                    elif req_sign < 0 and mean_r > mean_l:
                        gain *= 0.05

                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = float(thresh)
                    best_left_idx = left
                    best_right_idx = right

        if best_gain <= 0.0 or not best_left_idx or not best_right_idx:
            self.nodes[node_idx].is_leaf = True
            return

        # Split verified
        self.nodes[node_idx].is_leaf = False
        self.nodes[node_idx].feature_idx = best_feat
        self.nodes[node_idx].threshold = best_thresh
        best_feat_vals = [dataset[i].features[best_feat] for i in sample_indices]
        self.nodes[node_idx].feature_std = max(float(np.std(best_feat_vals)), 1e-4)

        # Create left child (inherits parent_idx)
        left_child_idx = len(self.nodes)
        self.nodes.append(FlatTreeNode(depth=depth + 1, parent_idx=node_idx))
        self.nodes[node_idx].left_child = left_child_idx

        # Create right child (inherits parent_idx)
        right_child_idx = len(self.nodes)
        self.nodes.append(FlatTreeNode(depth=depth + 1, parent_idx=node_idx))
        self.nodes[node_idx].right_child = right_child_idx

        # Recurse
        self._build_sub_tree(left_child_idx, best_left_idx, dataset)
        self._build_sub_tree(right_child_idx, best_right_idx, dataset)

    def _run_apoptosis_audit(self):
        """
        Phase 2 Apoptosis:
        Audit each node at depth >= apoptosis_start_depth.
        Excise leaves with pathological Fisher Information / fragile margin density.
        """
        root_val = self.nodes[0].value if self.nodes else 0.0
        for i in range(len(self.nodes)):
            if self.nodes[i].depth >= self.apoptosis_start_depth and not self.nodes[i].is_leaf:
                # Calculate proxy empirical Fisher info (shift-invariant divergence from baseline root mean)
                fisher = ((self.nodes[i].value - root_val) ** 2) / (self.baseline_var + 1e-9)
                margin = float(self.nodes[i].sample_count) / max(float(self.nodes[0].sample_count), 1.0)
                self.nodes[i].fisher_info = fisher
                self.nodes[i].margin_density = margin

                # Apoptosis condition: high variance shock with razor-thin sample support
                if fisher * self.cancer_sensitivity > 15.0 and margin < 0.03:
                    self.nodes[i].is_pruned_apoptosis = True

    def fit(self, X: Union[np.ndarray, "pd.DataFrame"], y: Union[np.ndarray, "pd.Series"]):
        """Fit tree on static matrix X and target vector y."""
        if hasattr(X, "columns"):
            self.feature_names = [str(col) for col in X.columns]
        elif hasattr(X, "shape") and len(X.shape) > 1:
            if not self.feature_names or len(self.feature_names) != X.shape[1]:
                self.feature_names = [f"X{i}" for i in range(X.shape[1])]
        elif not self.feature_names:
            self.feature_names = []

        X_mat = getattr(X, "values", X)
        y_vec = getattr(y, "values", y)
        X_mat = np.asarray(X_mat, dtype=float)
        y_vec = np.asarray(y_vec, dtype=float).ravel()

        self.n_features_ = X_mat.shape[1] if len(X_mat.shape) > 1 else 1

        samples = [
            TrainingSample(features=X_mat[i].tolist(), target=float(y_vec[i]))
            for i in range(len(y_vec))
        ]
        self.train(samples)
        return self

    def score(self, X: Union[np.ndarray, "pd.DataFrame"], y: Union[np.ndarray, "pd.Series"]) -> float:
        """Compute R^2 determination coefficient on static data."""
        y_vec = np.asarray(getattr(y, "values", y), dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y_vec - y_pred) ** 2)
        ss_tot = np.sum((y_vec - np.mean(y_vec)) ** 2)
        if ss_tot < 1e-12:
            return 1.0 if ss_res < 1e-12 else 0.0
        return float(1.0 - ss_res / ss_tot)

    def get_pruning_stats(self) -> dict:
        """Return audit metrics of Phase 2 Empirical Fisher Apoptosis."""
        total = len(self.nodes)
        pruned = sum(1 for n in self.nodes if n.is_pruned_apoptosis)
        leaves = sum(1 for n in self.nodes if n.is_leaf and not n.is_pruned_apoptosis)
        return {
            "total_nodes": total,
            "pruned_apoptosis_nodes": pruned,
            "active_leaves": leaves,
            "prune_rate": float(pruned / max(total, 1)),
        }

    def get_feature_importances(self) -> dict:
        """Calculate normalized variance reduction attributable to each feature."""
        gains = {}
        for n in self.nodes:
            if not n.is_leaf and not n.is_pruned_apoptosis and n.feature_idx >= 0:
                feat_name = (
                    self.feature_names[n.feature_idx]
                    if n.feature_idx < len(self.feature_names)
                    else f"Feature_{n.feature_idx}"
                )
                gains[feat_name] = gains.get(feat_name, 0.0) + (1.0 / (n.depth + 1))

        total_gain = sum(gains.values())
        if total_gain > 0:
            return {k: float(v / total_gain) for k, v in sorted(gains.items(), key=lambda x: -x[1])}
        return {k: 0.0 for k in self.feature_names}

    def predict_soft(self, features: List[float]) -> float:
        """
        Smooth Transition Regression (LSTR / Soft Decision Tree Traversal).
        Evaluates observations with continuous logistic regime weights instead of hard truncations.
        """
        if not self.nodes:
            return 0.0

        stack = [(0, 1.0)]
        total_val = 0.0
        total_weight = 0.0

        while stack:
            curr_idx, prob = stack.pop()
            if prob < 1e-6:
                continue
            node = self.nodes[curr_idx]
            if node.is_leaf or node.is_pruned_apoptosis or node.feature_idx < 0:
                total_val += prob * node.value
                total_weight += prob
                continue

            f_val = features[node.feature_idx]
            scale = max(node.feature_std, 1e-4)
            z = self.transition_gamma * (f_val - node.threshold) / scale
            z = max(min(z, 25.0), -25.0)
            p_right = 1.0 / (1.0 + np.exp(-z))
            p_left = 1.0 - p_right

            if node.left_child >= 0:
                stack.append((node.left_child, prob * p_left))
            if node.right_child >= 0:
                stack.append((node.right_child, prob * p_right))

        return float(total_val / total_weight) if total_weight > 0 else self.nodes[0].value

    def predict_one(self, features: List[float]) -> float:
        if not self.nodes:
            return 0.0

        if self.soft_splits:
            return self.predict_soft(features)

        curr_idx = 0
        while curr_idx >= 0 and curr_idx < len(self.nodes):
            node = self.nodes[curr_idx]
            if node.is_leaf or node.is_pruned_apoptosis:
                return node.value

            f_val = features[node.feature_idx]
            if f_val <= node.threshold:
                curr_idx = node.left_child
            else:
                curr_idx = node.right_child

        return 0.0

    def predict(self, feature_matrix: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
        mat = getattr(feature_matrix, "values", feature_matrix)
        mat = np.asarray(mat, dtype=float)
        return np.array([self.predict_one(row.tolist()) for row in mat])

    def export_all_symbolic_rules(self) -> List[dict]:
        """
        Traverse the tree and export every single branch/leaf path
        as an exact closed-form boolean formula interacting with base features.
        Includes James-Stein shrinkage metrics and leaf sample support.
        """
        if not self.nodes:
            return []

        rules = []

        def _traverse(node_idx: int, conditions: List[str]):
            if node_idx < 0 or node_idx >= len(self.nodes):
                return
            node = self.nodes[node_idx]

            if node.is_leaf or node.is_pruned_apoptosis:
                formula_str = " AND ".join(conditions) if conditions else "TRUE (Root)"
                rules.append({
                    "node_idx": node_idx,
                    "depth": node.depth,
                    "formula": formula_str,
                    "value": float(node.value),
                    "raw_value": float(node.raw_leaf_value),
                    "shrinkage_lambda": float(node.shrinkage_lambda),
                    "sample_count": node.sample_count,
                    "is_pruned": node.is_pruned_apoptosis,
                })
                return

            f_name = (
                self.feature_names[node.feature_idx]
                if node.feature_idx < len(self.feature_names)
                else f"X{node.feature_idx}"
            )

            # Left branch: <= threshold
            _traverse(node.left_child, conditions + [f"({f_name} <= {node.threshold:.3f})"])
            # Right branch: > threshold
            _traverse(node.right_child, conditions + [f"({f_name} > {node.threshold:.3f})"])

        _traverse(0, [])
        return rules
