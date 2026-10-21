"""
Nirvana Ensemble Module: Multi-Scale Pyramid and Advanced Ensembling Architecture.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from nirvana.static import NirvanaTreeRegressor


class NirvanaPyramidEnsemble(BaseEstimator, RegressorMixin):
    """
    KIM TỰ THÁP ĐA TẦNG NIRVANA (MULTI-SCALE PYRAMID ENSEMBLE)
    
    Quét dải quang phổ độ sâu từ vĩ mô (Tầng 8) đến vi mô cực hạn (Tầng 18)
    nhằm triệt tiêu nhiễu và chiết xuất các đặc trưng kích hoạt van phi tuyến
    cho các mô hình sau huấn luyện (Downstream Models).
    """

    def __init__(
        self,
        depth_schedule: Optional[Dict[int, int]] = None,
        subsample_ratio: float = 0.85,
        apoptosis_start_depth: int = 6,
        cancer_sensitivity: float = 1.0,
        random_state: int = 42,
    ):
        if depth_schedule is None:
            self.depth_schedule = {
                8: 25, 9: 25, 10: 25,    # Vĩ mô cốt lõi: Neo giữ xu hướng chính
                11: 15, 12: 15, 13: 15,  # Vùng bùng nổ phương sai Nirvana
                14: 10, 15: 10, 16: 10,  # Vùng biên thử nghiệm
                17: 5, 18: 5             # Vi mô cực hạn: Bắt tín hiệu đuôi rủi ro
            }
        else:
            self.depth_schedule = depth_schedule

        self.subsample_ratio = subsample_ratio
        self.apoptosis_start_depth = apoptosis_start_depth
        self.cancer_sensitivity = cancer_sensitivity
        self.random_state = random_state

        self.trees_: List[NirvanaTreeRegressor] = []
        self.tree_depths_: List[int] = []

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """Huấn luyện quần thể cây đa tầng trên các mẫu con Bootstrap."""
        X_mat = getattr(X, "values", X)
        y_vec = getattr(y, "values", y)
        X_mat = np.asarray(X_mat, dtype=float)
        y_vec = np.asarray(y_vec, dtype=float).ravel()

        np.random.seed(self.random_state)
        n_samples = len(X_mat)
        sub_size = max(4, int(self.subsample_ratio * n_samples))

        self.trees_ = []
        self.tree_depths_ = []

        for depth, n_loops in sorted(self.depth_schedule.items()):
            for _ in range(n_loops):
                boot_idx = np.random.choice(n_samples, size=sub_size, replace=True)
                tree = NirvanaTreeRegressor(
                    max_depth=depth,
                    apoptosis_start_depth=self.apoptosis_start_depth,
                    cancer_sensitivity=self.cancer_sensitivity,
                )
                tree.fit(X_mat[boot_idx], y_vec[boot_idx])
                self.trees_.append(tree)
                self.tree_depths_.append(depth)

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Dự báo trung bình cộng từ toàn bộ quần thể cây kim tự tháp."""
        X_mat = getattr(X, "values", X)
        X_mat = np.asarray(X_mat, dtype=float)
        all_preds = np.array([tree.predict(X_mat) for tree in self.trees_])
        return np.mean(all_preds, axis=0)

    def extract_valve_features(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Chiết xuất ma trận đặc trưng van đa tầng cho các mô hình giai đoạn 2.
        Xuất ra: (Kỳ vọng tín hiệu Mean, Độ bất định tín hiệu Uncertainty) ứng với từng tầng độ sâu.
        """
        X_mat = getattr(X, "values", X)
        X_mat = np.asarray(X_mat, dtype=float)
        all_preds = np.array([tree.predict(X_mat) for tree in self.trees_])

        valve_features = []
        feature_names = []
        unique_depths = sorted(list(set(self.tree_depths_)))

        for d in unique_depths:
            d_indices = [i for i, depth in enumerate(self.tree_depths_) if depth == d]
            depth_mean = np.mean(all_preds[d_indices], axis=0)
            depth_var = np.var(all_preds[d_indices], axis=0)

            valve_features.append(depth_mean)
            valve_features.append(depth_var)
            feature_names.append(f"Valve_Depth_{d}_Mean")
            feature_names.append(f"Valve_Depth_{d}_Uncertainty")

        return pd.DataFrame(np.array(valve_features).T, columns=feature_names)
