"""
NIRVANA STATIC DATA ECONOMETRICS & GENERAL-PURPOSE ESTIMATOR MODULE
====================================================================
General-purpose econometric and machine learning tools for static cross-sectional
and tabular datasets. Provides scikit-learn compliant estimators, universal two-stage
residual boosters, and formula interfaces.

Components:
1. NirvanaTreeRegressor:
   Standalone 12-layer non-linear decision tree regressor with Empirical Fisher
   Information and Margin-Density Apoptosis. Fully scikit-learn compliant (BaseEstimator,
   RegressorMixin, Pipeline, GridSearchCV compatible).

2. NirvanaTwoStageRegressor (Universal Meta-Estimator):
   Accepts ANY arbitrary base model (OLS, Ridge, Lasso, ElasticNet, SVR, DecisionTree, etc.)
   as Phase 1, and fits Nirvana Phase 2 Apoptosis Tree on its residuals to rescue weak R^2 models.

3. parse_formula:
   R / Stata-style formula parser ("y ~ x1 + x2 + x3") for DataFrame workflows.

Author: yinyusi65 & Nirvana Quant Team
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    class BaseEstimator:
        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class RegressorMixin:
        pass

    SKLEARN_AVAILABLE = False

from .tree import NirvanaTreeEngine, TrainingSample


def parse_formula(formula: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Parse an R / Stata-style regression formula (e.g. 'y ~ x1 + x2 + x3').

    Parameters
    ----------
    formula : str
        Formula string formatted as 'depvar ~ indepvar1 + indepvar2 + ...'
    df : pd.DataFrame
        Dataset containing the variables.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, List[str]]
        (X dataframe, y series, list of feature names)
    """
    if "~" not in formula:
        raise ValueError(f"Invalid formula '{formula}'. Must contain '~' separating dependent and independent variables.")

    lhs, rhs = formula.split("~", 1)
    depvar = lhs.strip()
    indepvars = [v.strip() for v in rhs.split("+") if v.strip()]

    missing_cols = [c for c in [depvar] + indepvars if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

    clean_df = df[[depvar] + indepvars].dropna()
    return clean_df[indepvars], clean_df[depvar], indepvars


@dataclass
class StaticEconometricReport:
    """
    Academic econometric report comparing Phase 1 (Baseline Weak Model)
    against Phase 2 (Nirvana Empirical Fisher Apoptosis Enhancement).
    """

    n_samples: int
    n_features: int
    r2_phase1: float
    r2_phase2: float
    r2_gain: float
    r2_relative_improvement_pct: float
    residual_variance_reduction_pct: float
    f_statistic: float
    f_p_value: float
    phase1_coefficients: Dict[str, float]
    feature_importances: Dict[str, float]
    pruning_stats: Dict[str, Union[int, float]]
    top_substitution_thresholds: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 78,
            "NIRVANA STATIC ECONOMETRICS REPORT: PHASE 2 MODEL ENHANCEMENT",
            "=" * 78,
            f"Sample Size (N):                     {self.n_samples}",
            f"Feature Count (K):                   {self.n_features}",
            "-" * 78,
            "1. MODEL EXPLANATORY POWER & R-SQUARED ELEVATION:",
            f"   - Phase 1 Baseline R^2 (Weak Model): {self.r2_phase1:.4f} ({self.r2_phase1 * 100:.2f}%)",
            f"   - Phase 2 Nirvana R^2 (Enhanced):    {self.r2_phase2:.4f} ({self.r2_phase2 * 100:.2f}%)",
            f"   - Absolute R^2 Gain (+Delta R^2):    +{self.r2_gain:.4f} (+{self.r2_gain * 100:.2f} percentage points)",
            f"   - Relative Explanatory Gain:         +{self.r2_relative_improvement_pct:.2f}% of unexplained variance resolved",
            f"   - Residual Variance Reduction:       {self.residual_variance_reduction_pct:.2f}%",
            f"   - Incremental F-Statistic:           {self.f_statistic:.2f} (p-value: {self.f_p_value:.4e})",
            "-" * 78,
            "2. PHASE 2 EMPIRICAL FISHER MARGIN-DENSITY APOPTOSIS AUDIT:",
            f"   - Total Tree Nodes Generated:        {self.pruning_stats.get('total_nodes', 0)}",
            f"   - Active Structural Leaves Retained: {self.pruning_stats.get('active_leaves', 0)}",
            f"   - Cancerous Overfitting Nodes Excised: {self.pruning_stats.get('pruned_apoptosis_nodes', 0)}",
            f"   - Cancer Pruning Ratio:              {self.pruning_stats.get('prune_rate', 0.0) * 100:.2f}%",
            "-" * 78,
            "3. TOP NON-LINEAR FEATURE DRIVERS (PHASE 2 VARIANCE ABSORPTION):",
        ]
        for feat, imp in list(self.feature_importances.items())[:6]:
            lines.append(f"   * {feat:25s}: {imp * 100:6.2f}%")

        if self.top_substitution_thresholds:
            lines.append("-" * 78)
            lines.append("4. DISCOVERED STRUCTURAL & SUBSTITUTION THRESHOLDS:")
            for rule in self.top_substitution_thresholds[:6]:
                lines.append(f"   -> {rule}")

        lines.append("=" * 78)
        return "\n".join(lines)


class NirvanaTreeRegressor(BaseEstimator, RegressorMixin):
    """
    General-Purpose Non-Linear Tree Regressor with Empirical Fisher Apoptosis.
    Fully compliant with the Scikit-Learn Estimator API.
    """

    def __init__(
        self,
        max_depth: int = 10,
        cancer_sensitivity: float = 1.0,
        oof_consistency: float = 0.5,
        apoptosis_start_depth: int = 6,
        split_quantiles: Optional[List[float]] = None,
        min_samples_split: int = 4,
        min_samples_leaf: int = 1,
        hierarchical_shrinkage: float = 0.0,
        soft_splits: bool = False,
        transition_gamma: float = 3.0,
        sign_constraints: Optional[Dict[Union[int, str], int]] = None,
    ):
        self.max_depth = max_depth
        self.cancer_sensitivity = cancer_sensitivity
        self.oof_consistency = oof_consistency
        self.apoptosis_start_depth = apoptosis_start_depth
        self.split_quantiles = split_quantiles
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.hierarchical_shrinkage = hierarchical_shrinkage
        self.soft_splits = soft_splits
        self.transition_gamma = transition_gamma
        self.sign_constraints = sign_constraints

        self.tree_engine_: Optional[NirvanaTreeEngine] = None
        self.feature_names_: List[str] = []
        self.n_features_in_: int = 0

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit decision tree with Fisher margin apoptosis on static tabular data.
        """
        if hasattr(X, "columns"):
            self.feature_names_ = [str(c) for c in X.columns]
        elif hasattr(X, "shape") and len(X.shape) > 1:
            self.feature_names_ = [f"X{i}" for i in range(X.shape[1])]
        else:
            self.feature_names_ = ["X0"]

        X_mat = getattr(X, "values", X)
        y_vec = getattr(y, "values", y)
        X_mat = np.asarray(X_mat, dtype=float)
        y_vec = np.asarray(y_vec, dtype=float).ravel()

        self.n_features_in_ = X_mat.shape[1] if len(X_mat.shape) > 1 else 1

        self.tree_engine_ = NirvanaTreeEngine(
            cancer_sensitivity=self.cancer_sensitivity,
            oof_consistency=self.oof_consistency,
            max_depth=self.max_depth,
            apoptosis_start_depth=self.apoptosis_start_depth,
            split_quantiles=self.split_quantiles or [10.0, 20.0, 35.0, 50.0, 65.0, 80.0, 90.0],
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            hierarchical_shrinkage=self.hierarchical_shrinkage,
            soft_splits=self.soft_splits,
            transition_gamma=self.transition_gamma,
            sign_constraints=self.sign_constraints,
        )
        self.tree_engine_.feature_names = self.feature_names_
        self.tree_engine_.fit(X_mat, y_vec)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.tree_engine_ is None:
            raise ValueError("NirvanaTreeRegressor is not fitted yet.")
        return self.tree_engine_.predict(X)

    def score(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> float:
        y_vec = np.asarray(getattr(y, "values", y), dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y_vec - y_pred) ** 2)
        ss_tot = np.sum((y_vec - np.mean(y_vec)) ** 2)
        if ss_tot < 1e-12:
            return 1.0 if ss_res < 1e-12 else 0.0
        return float(1.0 - ss_res / ss_tot)

    @property
    def feature_importances_(self) -> np.ndarray:
        if self.tree_engine_ is None:
            raise ValueError("Model not fitted yet.")
        imp_dict = self.tree_engine_.get_feature_importances()
        return np.array([imp_dict.get(feat, 0.0) for feat in self.feature_names_])

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "cancer_sensitivity": self.cancer_sensitivity,
            "oof_consistency": self.oof_consistency,
            "apoptosis_start_depth": self.apoptosis_start_depth,
            "split_quantiles": self.split_quantiles,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "hierarchical_shrinkage": self.hierarchical_shrinkage,
            "soft_splits": self.soft_splits,
            "transition_gamma": self.transition_gamma,
            "sign_constraints": self.sign_constraints,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def export_all_symbolic_rules(self) -> List[dict]:
        """Export every single active branch/leaf path as an exact symbolic formula."""
        if self.tree_engine_ is None:
            raise ValueError("Model is not fitted yet.")
        return self.tree_engine_.export_all_symbolic_rules()


class NirvanaTwoStageRegressor(BaseEstimator, RegressorMixin):
    """
    Universal Two-Stage Residual Booster Meta-Estimator.

    Wraps ANY base regression model (Phase 1: OLS, Ridge, Lasso, ElasticNet, SVR, etc.)
    and elevates its R^2 by training Nirvana Phase 2 (Apoptosis Tree) on its residuals:
    y_final = y_hat_base + learning_rate * e_hat_phase2
    """

    def __init__(
        self,
        base_estimator: Any = "ols",
        cancer_sensitivity: float = 1.0,
        oof_consistency: float = 0.5,
        max_depth: int = 10,
        apoptosis_start_depth: int = 6,
        learning_rate: float = 1.0,
        ridge_lambda: float = 1e-4,
        split_quantiles: Optional[List[float]] = None,
        min_samples_split: int = 4,
        min_samples_leaf: int = 1,
        hierarchical_shrinkage: float = 0.0,
        soft_splits: bool = False,
        transition_gamma: float = 3.0,
        sign_constraints: Optional[Dict[Union[int, str], int]] = None,
        factor_compression: bool = False,
        n_components: Optional[int] = None,
    ):
        self.base_estimator = base_estimator
        self.cancer_sensitivity = cancer_sensitivity
        self.oof_consistency = oof_consistency
        self.max_depth = max_depth
        self.apoptosis_start_depth = apoptosis_start_depth
        self.learning_rate = learning_rate
        self.ridge_lambda = ridge_lambda
        self.split_quantiles = split_quantiles
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.hierarchical_shrinkage = hierarchical_shrinkage
        self.soft_splits = soft_splits
        self.transition_gamma = transition_gamma
        self.sign_constraints = sign_constraints
        self.factor_compression = factor_compression
        self.n_components = n_components

        self.fitted_base_estimator_: Any = None
        self.tree_engine_: Optional[NirvanaTreeRegressor] = None
        self.beta_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []
        self.pca_mean_: Optional[np.ndarray] = None
        self.pca_components_: Optional[np.ndarray] = None
        self.report_: Optional[StaticEconometricReport] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit Phase 1 base model + Phase 2 Nirvana Apoptosis Tree on residuals.
        """
        if hasattr(X, "columns"):
            self.feature_names_ = [str(c) for c in X.columns]
        elif hasattr(X, "shape") and len(X.shape) > 1:
            self.feature_names_ = [f"X{i}" for i in range(X.shape[1])]
        else:
            self.feature_names_ = ["X0"]

        X_mat = getattr(X, "values", X)
        y_vec = getattr(y, "values", y)
        X_mat = np.asarray(X_mat, dtype=float)
        y_vec = np.asarray(y_vec, dtype=float).ravel()
        N, K = X_mat.shape

        # --- Optional Factor Compression (SVD Orthogonalization) to protect degrees of freedom ---
        self.pca_mean_ = None
        self.pca_components_ = None
        X_reg = X_mat
        if self.factor_compression and K > 1:
            k_comp = self.n_components or max(1, min(K - 1, max(1, N // 3)))
            self.pca_mean_ = np.mean(X_mat, axis=0)
            X_center = X_mat - self.pca_mean_
            _, _, Vt = np.linalg.svd(X_center, full_matrices=False)
            self.pca_components_ = Vt[:k_comp]
            X_reg = X_center @ self.pca_components_.T

        # --- Phase 1: Fit Base Model ---
        K_reg = X_reg.shape[1]
        if self.base_estimator == "ols" or not SKLEARN_AVAILABLE:
            X_design = np.column_stack([np.ones(N), X_reg])
            reg_eye = self.ridge_lambda * np.eye(K_reg + 1)
            reg_eye[0, 0] = 0.0
            self.beta_ = np.linalg.solve(X_design.T @ X_design + reg_eye, X_design.T @ y_vec)

            # Enforce Economic Sign Constraints via Gauss-Seidel Coordinate Descent
            if self.sign_constraints and not self.factor_compression:
                beta_c = self.beta_.copy()
                XTX = X_design.T @ X_design + reg_eye
                XTy = X_design.T @ y_vec
                diag_XTX = np.diag(XTX)
                for _ in range(50):
                    for j in range(1, K + 1):
                        col_name = self.feature_names_[j - 1]
                        s = self.sign_constraints.get(col_name, self.sign_constraints.get(j - 1, 0))
                        if s != 0:
                            r_j = XTy[j] - (XTX[j] @ beta_c - XTX[j, j] * beta_c[j])
                            new_b = r_j / max(diag_XTX[j], 1e-12)
                            if s > 0:
                                new_b = max(0.0, new_b)
                            elif s < 0:
                                new_b = min(0.0, new_b)
                            beta_c[j] = new_b
                self.beta_ = beta_c

            y_pred_phase1 = X_design @ self.beta_
            self.fitted_base_estimator_ = "ols"
        elif isinstance(self.base_estimator, str):
            est_name = self.base_estimator.lower()
            if est_name == "ridge":
                base = Ridge(alpha=self.ridge_lambda * 100.0)
            elif est_name == "lasso":
                base = Lasso(alpha=0.01)
            elif est_name == "elasticnet":
                base = ElasticNet(alpha=0.01, l1_ratio=0.5)
            else:
                base = LinearRegression()
            base.fit(X_reg, y_vec)
            self.fitted_base_estimator_ = base
            y_pred_phase1 = base.predict(X_reg)
        else:
            # Custom user-supplied scikit-learn regressor
            base = self.base_estimator
            base.fit(X_reg, y_vec)
            self.fitted_base_estimator_ = base
            y_pred_phase1 = base.predict(X_reg)

        residuals_phase1 = y_vec - y_pred_phase1
        ss_tot = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
        ss_res_1 = float(np.sum(residuals_phase1 ** 2))
        r2_phase1 = max(0.0, float(1.0 - ss_res_1 / max(ss_tot, 1e-12)))

        # --- Phase 2: Fit Nirvana Apoptosis Tree on Residuals ---
        self.tree_engine_ = NirvanaTreeRegressor(
            max_depth=self.max_depth,
            cancer_sensitivity=self.cancer_sensitivity,
            oof_consistency=self.oof_consistency,
            apoptosis_start_depth=self.apoptosis_start_depth,
            split_quantiles=self.split_quantiles,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            hierarchical_shrinkage=self.hierarchical_shrinkage,
            soft_splits=self.soft_splits,
            transition_gamma=self.transition_gamma,
            sign_constraints=self.sign_constraints,
        )
        self.tree_engine_.fit(X, residuals_phase1)

        residuals_pred_phase2 = self.tree_engine_.predict(X_mat)
        y_pred_final = y_pred_phase1 + self.learning_rate * residuals_pred_phase2
        residuals_final = y_vec - y_pred_final

        ss_res_final = float(np.sum(residuals_final ** 2))
        r2_phase2 = max(0.0, float(1.0 - ss_res_final / max(ss_tot, 1e-12)))
        r2_gain = max(0.0, r2_phase2 - r2_phase1)

        unexplained_phase1 = max(1.0 - r2_phase1, 1e-6)
        r2_relative_improvement = float((r2_gain / unexplained_phase1) * 100.0)
        res_var_reduction = float(((ss_res_1 - ss_res_final) / max(ss_res_1, 1e-12)) * 100.0)

        # Incremental F-statistic
        df_1 = K + 1
        active_leaves = max(
            1, sum(1 for n in self.tree_engine_.tree_engine_.nodes if n.is_leaf and not n.is_pruned_apoptosis)
        )
        df_2 = active_leaves
        df_denom = max(1, N - df_1 - df_2)
        ms_num = max(0.0, ss_res_1 - ss_res_final) / df_2
        ms_denom = max(ss_res_final, 1e-12) / df_denom
        f_stat = float(ms_num / ms_denom)
        f_pval = float(1.0 - stats.f.cdf(f_stat, df_2, df_denom))

        coef_dict = {}
        if self.beta_ is not None:
            coef_dict["Intercept"] = float(self.beta_[0])
            if self.pca_components_ is not None:
                orig_beta = self.pca_components_.T @ self.beta_[1:]
                for idx, col in enumerate(self.feature_names_):
                    coef_dict[col] = float(orig_beta[idx])
            else:
                for idx, col in enumerate(self.feature_names_):
                    coef_dict[col] = float(self.beta_[idx + 1])
        elif hasattr(self.fitted_base_estimator_, "coef_"):
            for idx, col in enumerate(self.feature_names_):
                coef_dict[col] = float(self.fitted_base_estimator_.coef_[idx])

        pruning_stats = self.tree_engine_.tree_engine_.get_pruning_stats()
        feat_importances = self.tree_engine_.tree_engine_.get_feature_importances()

        threshold_rules = []
        for node in self.tree_engine_.tree_engine_.nodes[:15]:
            if not node.is_leaf and not node.is_pruned_apoptosis and node.feature_idx >= 0:
                feat = (
                    self.feature_names_[node.feature_idx]
                    if node.feature_idx < len(self.feature_names_)
                    else f"X{node.feature_idx}"
                )
                threshold_rules.append(
                    f"Layer {node.depth} Split: {feat} <= {node.threshold:.3f} "
                    f"(Samples: {node.sample_count}, Value: {node.value:.3f})"
                )

        self.report_ = StaticEconometricReport(
            n_samples=N,
            n_features=K,
            r2_phase1=r2_phase1,
            r2_phase2=r2_phase2,
            r2_gain=r2_gain,
            r2_relative_improvement_pct=r2_relative_improvement,
            residual_variance_reduction_pct=res_var_reduction,
            f_statistic=f_stat,
            f_p_value=f_pval,
            phase1_coefficients=coef_dict,
            feature_importances=feat_importances,
            pruning_stats=pruning_stats,
            top_substitution_thresholds=threshold_rules,
        )
        return self

    def __getattr__(self, name: str) -> Any:
        report = self.__dict__.get("report_", None)
        if report is not None and hasattr(report, name):
            return getattr(report, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.fitted_base_estimator_ is None or self.tree_engine_ is None:
            raise ValueError("NirvanaTwoStageRegressor is not fitted yet.")

        X_mat = getattr(X, "values", X)
        X_mat = np.asarray(X_mat, dtype=float)

        if self.factor_compression and self.pca_components_ is not None:
            X_center = X_mat - self.pca_mean_
            X_reg = X_center @ self.pca_components_.T
        else:
            X_reg = X_mat

        if self.fitted_base_estimator_ == "ols":
            X_design = np.column_stack([np.ones(X_reg.shape[0]), X_reg])
            y_pred_base = X_design @ self.beta_
        else:
            y_pred_base = self.fitted_base_estimator_.predict(X_reg)

        residuals_pred_phase2 = self.tree_engine_.predict(X_mat)
        return y_pred_base + self.learning_rate * residuals_pred_phase2

    def score(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> float:
        y_vec = np.asarray(getattr(y, "values", y), dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y_vec - y_pred) ** 2)
        ss_tot = np.sum((y_vec - np.mean(y_vec)) ** 2)
        if ss_tot < 1e-12:
            return 1.0 if ss_res < 1e-12 else 0.0
        return float(1.0 - ss_res / ss_tot)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "base_estimator": self.base_estimator,
            "cancer_sensitivity": self.cancer_sensitivity,
            "oof_consistency": self.oof_consistency,
            "max_depth": self.max_depth,
            "apoptosis_start_depth": self.apoptosis_start_depth,
            "learning_rate": self.learning_rate,
            "ridge_lambda": self.ridge_lambda,
            "split_quantiles": self.split_quantiles,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "hierarchical_shrinkage": self.hierarchical_shrinkage,
            "soft_splits": self.soft_splits,
            "transition_gamma": self.transition_gamma,
            "sign_constraints": self.sign_constraints,
            "factor_compression": self.factor_compression,
            "n_components": self.n_components,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def evaluate_cv(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        k_folds: int = 5,
        random_seed: int = 42,
    ) -> Dict[str, float]:
        """
        Run K-fold Out-Of-Sample Cross Validation.
        """
        X_mat = getattr(X, "values", X)
        y_vec = getattr(y, "values", y)
        X_mat = np.asarray(X_mat, dtype=float)
        y_vec = np.asarray(y_vec, dtype=float).ravel()

        N = len(y_vec)
        rng = np.random.default_rng(random_seed)
        indices = np.arange(N)
        rng.shuffle(indices)

        fold_size = N // k_folds
        r2_phase1_list = []
        r2_phase2_list = []

        for f in range(k_folds):
            val_idx = indices[f * fold_size : (f + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)

            X_tr, y_tr = X_mat[train_idx], y_vec[train_idx]
            X_val, y_val = X_mat[val_idx], y_vec[val_idx]

            reg = NirvanaTwoStageRegressor(
                base_estimator=self.base_estimator,
                cancer_sensitivity=self.cancer_sensitivity,
                oof_consistency=self.oof_consistency,
                max_depth=self.max_depth,
                apoptosis_start_depth=self.apoptosis_start_depth,
                learning_rate=self.learning_rate,
                ridge_lambda=self.ridge_lambda,
                split_quantiles=self.split_quantiles,
                min_samples_split=self.min_samples_split,
            )
            reg.fit(X_tr, y_tr)

            # Base model score on val
            if reg.fitted_base_estimator_ == "ols":
                X_val_des = np.column_stack([np.ones(len(val_idx)), X_val])
                y_val_pred_1 = X_val_des @ reg.beta_
            else:
                y_val_pred_1 = reg.fitted_base_estimator_.predict(X_val)

            ss_tot_val = float(np.sum((y_val - np.mean(y_val)) ** 2))
            ss_res_1_val = float(np.sum((y_val - y_val_pred_1) ** 2))
            r2_1_val = max(0.0, 1.0 - ss_res_1_val / max(ss_tot_val, 1e-12))
            r2_phase1_list.append(r2_1_val)

            r2_2_val = reg.score(X_val, y_val)
            r2_phase2_list.append(r2_2_val)

        return {
            "r2_phase1_cv_mean": float(np.mean(r2_phase1_list)),
            "r2_phase1_cv_std": float(np.std(r2_phase1_list)),
            "r2_phase2_cv_mean": float(np.mean(r2_phase2_list)),
            "r2_phase2_cv_std": float(np.std(r2_phase2_list)),
            "delta_r2_cv_mean": float(np.mean(r2_phase2_list) - np.mean(r2_phase1_list)),
        }

    def prove_phase2_multiverse(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        n_universes: int = 100,
        seed: int = 42,
    ):
        """
        Rigorously prove the statistical validity, Ramsey RESET cure,
        and multiverse invariance of the discovered variable.
        """
        from .proof import prove_phase2_variable
        return prove_phase2_variable(self, X, y, n_universes=n_universes, seed=seed)

    def rank_discovered_variables(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        top_k: int = 5,
        n_universes: int = 100,
        seed: int = 42,
    ):
        """
        Rank top-K discovered variables from Phase 2, prove their econometric
        significance and multiverse invariance, and map to real-world collection protocols.
        """
        from .proof import rank_phase2_discovered_variables
        return rank_phase2_discovered_variables(
            self, X, y, top_k=top_k, n_universes=n_universes, seed=seed
        )

    def export_all_symbolic_rules(self) -> List[dict]:
        """Export every single active branch/leaf path in Phase 2 as an exact symbolic formula."""
        if self.tree_engine_ is None:
            raise ValueError("Model is not fitted yet.")
        return self.tree_engine_.export_all_symbolic_rules()


class NirvanaStaticBooster(NirvanaTwoStageRegressor):
    """
    Two-Stage Econometric Booster for Static Cross-Sectional Data.
    Returns StaticEconometricReport on .fit(X, y) for econometric workflows.
    """

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> StaticEconometricReport:
        super().fit(X, y)
        return self.report_
