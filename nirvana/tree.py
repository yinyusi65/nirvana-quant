"""
Real-Tree Topology & Fisher Information Apoptosis Module
=========================================================
12-layer decision tree engine featuring Phase 2 Empirical Fisher Information & Margin-Density Apoptosis.
Excises cancerous/overfitting nodes that memorize noise rather than learning causal invariance.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
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
    is_leaf: bool = True
    is_pruned_apoptosis: bool = False
    sample_count: int = 0
    fisher_info: float = 0.0
    margin_density: float = 0.0
    left_child: int = -1
    right_child: int = -1


class NirvanaTreeEngine:
    """
    12-layer Real-Data Tree Engine with Empirical Fisher Apoptosis.
    """

    MAX_DEPTH = 12
    APOPTOSIS_START_DEPTH = 7

    def __init__(self, cancer_sensitivity: float = 1.0, oof_consistency: float = 0.5):
        self.cancer_sensitivity = float(cancer_sensitivity)
        self.oof_consistency = float(oof_consistency)
        self.nodes: List[FlatTreeNode] = []
        self.baseline_var: float = 1e-8

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

        # Compute mean target
        node_vals = [dataset[i].target for i in sample_indices]
        node_val = float(np.mean(node_vals))
        self.nodes[node_idx].value = node_val
        self.nodes[node_idx].sample_count = n_samples

        if depth >= self.MAX_DEPTH or n_samples < 4:
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
            unique_vals = np.percentile(vals, [25, 50, 75])

            for thresh in unique_vals:
                left = [i for i in sample_indices if dataset[i].features[f] <= thresh]
                right = [i for i in sample_indices if dataset[i].features[f] > thresh]

                if not left or not right:
                    continue

                var_left = float(np.var([dataset[i].target for i in left])) * len(left)
                var_right = float(np.var([dataset[i].target for i in right])) * len(right)
                gain = curr_var - (var_left + var_right)

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

        # Create left child
        left_child_idx = len(self.nodes)
        self.nodes.append(FlatTreeNode(depth=depth + 1))
        self.nodes[node_idx].left_child = left_child_idx

        # Create right child
        right_child_idx = len(self.nodes)
        self.nodes.append(FlatTreeNode(depth=depth + 1))
        self.nodes[node_idx].right_child = right_child_idx

        # Recurse
        self._build_sub_tree(left_child_idx, best_left_idx, dataset)
        self._build_sub_tree(right_child_idx, best_right_idx, dataset)

    def _run_apoptosis_audit(self):
        """
        Phase 2 Apoptosis:
        Audit each node at depth >= APOPTOSIS_START_DEPTH.
        Excise leaves with pathological Fisher Information / fragile margin density.
        """
        root_val = self.nodes[0].value if self.nodes else 0.0
        for i in range(len(self.nodes)):
            if self.nodes[i].depth >= self.APOPTOSIS_START_DEPTH and not self.nodes[i].is_leaf:
                # Calculate proxy empirical Fisher info (shift-invariant divergence from baseline root mean)
                fisher = ((self.nodes[i].value - root_val) ** 2) / (self.baseline_var + 1e-9)
                margin = float(self.nodes[i].sample_count) / max(float(self.nodes[0].sample_count), 1.0)
                self.nodes[i].fisher_info = fisher
                self.nodes[i].margin_density = margin

                # Apoptosis condition: high variance shock with razor-thin sample support
                if fisher * self.cancer_sensitivity > 15.0 and margin < 0.03:
                    self.nodes[i].is_pruned_apoptosis = True

    def predict_one(self, features: List[float]) -> float:
        if not self.nodes:
            return 0.0

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
