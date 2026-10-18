"""
NIRVANA STATA ECONOMETRIC ENGINE & STATISTICAL DIAGNOSTICS MODULE
==================================================================
Provides publication-grade econometric estimation, classical Stata-formatted
regression tables, and rigorous post-estimation hypothesis testing:

1. Stata-style OLS Regression Table (regress y x1 x2 ...):
   - ANOVA Source table (SS, df, MS, F-stat, Prob > F, R^2, Adj R^2, Root MSE)
   - Coefficient table: Coef, Std. Err., t, P>|t|, [95% Conf. Interval]
   - Robust Huber-White standard errors (vce(robust))

2. Stata Post-Estimation Diagnostics:
   - estat hettest: Breusch-Pagan / Cook-Weisberg test for heteroskedasticity
   - estat imtest:  White's general heteroskedasticity test
   - estat vif:     Variance Inflation Factor & Tolerance for multicollinearity
   - estat ovtest:  Ramsey RESET test for functional misspecification / omitted variables
   - estat dwatson: Durbin-Watson autocorrelation test
   - sktest / jb:   Jarque-Bera & Skewness-Kurtosis normality test on residuals

3. Stata Summary & Correlation Tables:
   - summarize (sum, detail): Obs, Mean, Std. Dev., Min, Max, p25, p50, p75, Skew, Kurt
   - pwcorr: Pairwise correlation matrix with significance stars (* p<0.05, ** p<0.01, *** p<0.001)

Author: yinyusi65 & Nirvana Quant Team
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StataRegressionResult:
    """
    Stores full econometric output matching Stata's 'regress' command.
    """

    depvar: str
    indepvars: List[str]
    n_obs: int
    df_model: int
    df_resid: int
    df_total: int
    ss_model: float
    ss_resid: float
    ss_total: float
    ms_model: float
    ms_resid: float
    f_stat: float
    prob_f: float
    r_squared: float
    adj_r_squared: float
    root_mse: float
    coefficients: Dict[str, float]
    std_errors: Dict[str, float]
    t_stats: Dict[str, float]
    p_values: Dict[str, float]
    ci_lower: Dict[str, float]
    ci_upper: Dict[str, float]
    fitted_values: np.ndarray
    residuals: np.ndarray
    is_robust: bool = False
    X_df: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        """
        Render text output matching Stata's iconic terminal layout.
        """
        lines = []
        # Header ANOVA table (Top)
        col1_w = 12
        col_num_w = 15
        
        lines.append(f"{'Source':>{col1_w}} | {'SS':^{col_num_w}} {'df':>6}   {'MS':^{col_num_w}}   Number of obs   = {self.n_obs:>10,d}")
        lines.append(f"{'-' * col1_w}-+-{'-' * (col_num_w * 2 + 11)}   F({self.df_model:>2d}, {self.df_resid:>5d})   = {self.f_stat:>10.2f}")
        lines.append(f"{'Model':>{col1_w}} | {self.ss_model:>{col_num_w}.4f} {self.df_model:>6d}   {self.ms_model:>{col_num_w}.4f}   Prob > F        = {self.prob_f:>10.4f}")
        lines.append(f"{'Residual':>{col1_w}} | {self.ss_resid:>{col_num_w}.4f} {self.df_resid:>6d}   {self.ms_resid:>{col_num_w}.4f}   R-squared       = {self.r_squared:>10.4f}")
        lines.append(f"{'-' * col1_w}-+-{'-' * (col_num_w * 2 + 11)}   Adj R-squared   = {self.adj_r_squared:>10.4f}")
        lines.append(f"{'Total':>{col1_w}} | {self.ss_total:>{col_num_w}.4f} {self.df_total:>6d}   {self.ms_resid:>{col_num_w}.4f}   Root MSE        = {self.root_mse:>10.4f}")
        lines.append("")

        # Coefficients table (Bottom)
        cov_type = " (Robust Std. Err.)" if self.is_robust else ""
        lines.append("-" * 78)
        lines.append(f"{self.depvar:>13s} | Coefficient  Std. err.{cov_type:10s}      t    P>|t|     [95% conf. interval]")
        lines.append("-" * 78)

        # Regressors
        all_vars = self.indepvars + ["_cons"]
        for var in all_vars:
            coef = self.coefficients[var]
            se = self.std_errors[var]
            t = self.t_stats[var]
            p = self.p_values[var]
            cil = self.ci_lower[var]
            ciu = self.ci_upper[var]
            
            p_str = f"{p:.3f}" if p >= 0.0005 else "0.000"
            display_var = var if len(var) <= 13 else var[:12] + "."
            lines.append(
                f"{display_var:>13s} | {coef:>12.5f} {se:>11.5f} {t:>8.2f} {p_str:>8s}     {cil:>9.5f}  {ciu:>9.5f}"
            )

        lines.append("-" * 78)
        return "\n".join(lines)

    # --- Stata Post-Estimation Diagnostics Methods ---

    def estat_hettest(self) -> Dict[str, Any]:
        """
        Breusch-Pagan / Cook-Weisberg test for heteroskedasticity.
        H0: Constant variance (homoskedasticity).
        Regresses normalized squared residuals on fitted values.
        """
        n = self.n_obs
        e2 = self.residuals ** 2
        s2 = np.mean(e2)
        u = e2 / max(s2, 1e-12) - 1.0  # centered normalized residuals

        # Regress u on fitted values y_hat
        y_hat = self.fitted_values - np.mean(self.fitted_values)
        denom = float(np.sum(y_hat ** 2))
        if denom < 1e-12:
            chi2_stat = 0.0
        else:
            slope = float(np.sum(u * y_hat) / denom)
            explained_ss = (slope ** 2) * denom
            chi2_stat = float(0.5 * explained_ss)

        p_val = float(1.0 - stats.chi2.cdf(chi2_stat, df=1))

        output_str = (
            "Breusch-Pagan / Cook-Weisberg test for heteroskedasticity\n"
            "         Ho: Constant variance\n"
            f"         Variables: fitted values of {self.depvar}\n\n"
            f"         chi2(1)      = {chi2_stat:>8.2f}\n"
            f"         Prob > chi2  = {p_val:>8.4f}"
        )
        return {
            "test": "Breusch-Pagan / Cook-Weisberg",
            "chi2": chi2_stat,
            "df": 1,
            "p_value": p_val,
            "display": output_str,
        }

    def estat_vif(self, X_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Variance Inflation Factor (VIF) and Tolerance (1/VIF).
        Detects multicollinearity among independent variables.
        """
        target_df = X_df if X_df is not None else self.X_df
        if target_df is None:
            raise ValueError("X_df was not provided during regression or estat_vif.")
        vif_data = []
        clean_X = target_df[self.indepvars].dropna()
        N, K = clean_X.shape

        for i, col in enumerate(self.indepvars):
            y_i = clean_X[col].values
            other_cols = [c for c in self.indepvars if c != col]
            if not other_cols:
                vif = 1.0
            else:
                X_other = clean_X[other_cols].values
                X_des = np.column_stack([np.ones(len(X_other)), X_other])
                beta = np.linalg.pinv(X_des.T @ X_des) @ X_des.T @ y_i
                y_pred = X_des @ beta
                ss_tot = float(np.sum((y_i - np.mean(y_i)) ** 2))
                ss_res = float(np.sum((y_i - y_pred) ** 2))
                r2 = max(0.0, min(1.0 - ss_res / max(ss_tot, 1e-12), 0.999999))
                vif = float(1.0 / (1.0 - r2))

            vif_data.append({
                "Variable": col,
                "VIF": vif,
                "1/VIF": float(1.0 / vif),
            })

        mean_vif = float(np.mean([v["VIF"] for v in vif_data]))

        lines = [
            f"{'Variable':>15s} | {'VIF':>10s}   {'1/VIF':>10s}",
            f"{'-' * 15}-+-{'-' * 25}",
        ]
        for row in sorted(vif_data, key=lambda x: -x["VIF"]):
            lines.append(f"{row['Variable']:>15s} | {row['VIF']:>10.2f}   {row['1/VIF']:>10.6f}")
        lines.append(f"{'-' * 15}-+-{'-' * 25}")
        lines.append(f"{'Mean VIF':>15s} | {mean_vif:>10.2f}")

        return {
            "vif_table": vif_data,
            "mean_vif": mean_vif,
            "display": "\n".join(lines),
        }

    def estat_ovtest(self, X_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Ramsey Regression Equation Specification Error Test (RESET).
        Ho: model has no omitted variables (linear functional form is correct).
        Regresses y on X, y_hat^2, y_hat^3, y_hat^4.
        """
        target_df = X_df if X_df is not None else self.X_df
        if target_df is None:
            raise ValueError("X_df was not provided during regression or estat_ovtest.")
        clean_X = target_df[self.indepvars].dropna()
        X_mat = clean_X.values
        N, K = X_mat.shape

        y_hat = self.fitted_values
        y_hat2 = y_hat ** 2
        y_hat3 = y_hat ** 3
        y_hat4 = y_hat ** 4

        # Unrestricted regression with powers of fitted values
        X_unres = np.column_stack([np.ones(N), X_mat, y_hat2, y_hat3, y_hat4])
        y = self.fitted_values + self.residuals

        beta_unres = np.linalg.pinv(X_unres.T @ X_unres) @ X_unres.T @ y
        e_unres = y - X_unres @ beta_unres
        ss_unres = float(np.sum(e_unres ** 2))

        df_num = 3
        df_denom = N - K - 1 - df_num
        f_stat = float(((self.ss_resid - ss_unres) / df_num) / (ss_unres / max(df_denom, 1)))
        f_stat = max(0.0, f_stat)
        p_val = float(1.0 - stats.f.cdf(f_stat, df_num, df_denom))

        output_str = (
            f"Ramsey RESET test using powers of the fitted values of {self.depvar}\n"
            "         Ho: model has no omitted variables\n\n"
            f"         F({df_num}, {df_denom})   = {f_stat:>8.2f}\n"
            f"         Prob > F     = {p_val:>8.4f}"
        )
        return {
            "test": "Ramsey RESET",
            "f_stat": f_stat,
            "df_num": df_num,
            "df_denom": df_denom,
            "p_value": p_val,
            "display": output_str,
        }

    def estat_dwatson(self) -> Dict[str, Any]:
        """
        Durbin-Watson d-statistic for first-order autocorrelation of residuals.
        """
        diff = np.diff(self.residuals)
        dw = float(np.sum(diff ** 2) / max(np.sum(self.residuals ** 2), 1e-12))
        output_str = f"Durbin-Watson d-statistic({self.df_model + 1:2d}, {self.n_obs:5d}) = {dw:.4f}"
        return {"test": "Durbin-Watson", "d_stat": dw, "display": output_str}

    def sktest(self) -> Dict[str, Any]:
        """
        Skewness/Kurtosis & Jarque-Bera tests for normality of residuals.
        """
        n = self.n_obs
        e = self.residuals - np.mean(self.residuals)
        s2 = np.mean(e ** 2)
        skew = float(np.mean(e ** 3) / max(s2 ** 1.5, 1e-12))
        kurt = float(np.mean(e ** 4) / max(s2 ** 2, 1e-12))  # Pearson kurtosis (normal = 3)

        # Jarque-Bera statistic
        jb_stat = float((n / 6.0) * (skew ** 2 + ((kurt - 3.0) ** 2) / 4.0))
        p_val = float(1.0 - stats.chi2.cdf(jb_stat, df=2))

        lines = [
            "Skewness/Kurtosis & Jarque-Bera tests for Normality of Residuals",
            f"{'-' * 70}",
            f"{'Variable':>12s} | {'Obs':>8s}   {'Pr(Skew)':>10s}   {'Pr(Kurt)':>10s}   {'chi2(2)':>10s}   {'Prob>chi2':>10s}",
            f"{'-' * 12}-+-{'-' * 55}",
            f"{'Residuals':>12s} | {n:>8,d}   {skew:>10.4f}   {kurt:>10.4f}   {jb_stat:>10.2f}   {p_val:>10.4f}",
            f"{'-' * 70}",
        ]
        return {
            "skewness": skew,
            "kurtosis": kurt,
            "jb_stat": jb_stat,
            "p_value": p_val,
            "display": "\n".join(lines),
        }


def stata_regress(
    df: pd.DataFrame,
    y: str,
    x: List[str],
    robust: bool = False,
) -> StataRegressionResult:
    """
    Fit an Ordinary Least Squares (OLS) regression matching Stata's 'regress' command.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing dependent and independent variables.
    y : str
        Name of dependent variable.
    x : List[str]
        List of independent regressor column names.
    robust : bool, default=False
        If True, estimates Huber-White robust standard errors (vce(robust)).

    Returns
    -------
    StataRegressionResult
        Rich econometric result dataclass with Stata formatting and post-estimation tests.
    """
    clean_df = df[[y] + x].dropna()
    N = len(clean_df)
    K = len(x)

    y_vec = clean_df[y].values.astype(float)
    X_mat = clean_df[x].values.astype(float)

    # Design matrix with constant term
    X_des = np.column_stack([X_mat, np.ones(N)])
    var_names = list(x) + ["_cons"]

    # OLS Estimator: beta = (X'X)^-1 X'y
    XtX = X_des.T @ X_des
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X_des.T @ y_vec)

    fitted = X_des @ beta
    resid = y_vec - fitted

    # Degrees of freedom
    df_model = K
    df_resid = N - K - 1
    df_total = N - 1

    # Sum of squares
    ss_tot = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
    ss_res = float(np.sum(resid ** 2))
    ss_mod = max(0.0, ss_tot - ss_res)

    ms_mod = ss_mod / max(df_model, 1)
    ms_res = ss_res / max(df_resid, 1)

    f_stat = ms_mod / max(ms_res, 1e-12)
    prob_f = float(1.0 - stats.f.cdf(f_stat, df_model, df_resid))

    r_sq = max(0.0, min(1.0, 1.0 - ss_res / max(ss_tot, 1e-12)))
    adj_r_sq = 1.0 - (1.0 - r_sq) * (df_total / max(df_resid, 1))
    root_mse = float(np.sqrt(ms_res))

    # Covariance Matrix of Parameters
    if not robust:
        cov_matrix = ms_res * XtX_inv
    else:
        # Huber-White Sandwich Estimator HC1: (N / (N - K - 1)) * (X'X)^-1 * (sum e_i^2 x_i x_i') * (X'X)^-1
        u2 = resid ** 2
        meat = (X_des.T * u2) @ X_des
        hc1_scale = float(N / max(df_resid, 1))
        cov_matrix = hc1_scale * (XtX_inv @ meat @ XtX_inv)

    std_errs = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-14))
    t_vals = beta / std_errs
    p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_vals), df=df_resid))

    t_crit = float(stats.t.ppf(0.975, df=df_resid))
    ci_low = beta - t_crit * std_errs
    ci_high = beta + t_crit * std_errs

    coef_d = {var: float(beta[i]) for i, var in enumerate(var_names)}
    se_d = {var: float(std_errs[i]) for i, var in enumerate(var_names)}
    t_d = {var: float(t_vals[i]) for i, var in enumerate(var_names)}
    p_d = {var: float(p_vals[i]) for i, var in enumerate(var_names)}
    cil_d = {var: float(ci_low[i]) for i, var in enumerate(var_names)}
    ciu_d = {var: float(ci_high[i]) for i, var in enumerate(var_names)}

    return StataRegressionResult(
        depvar=y,
        indepvars=list(x),
        n_obs=N,
        df_model=df_model,
        df_resid=df_resid,
        df_total=df_total,
        ss_model=ss_mod,
        ss_resid=ss_res,
        ss_total=ss_tot,
        ms_model=ms_mod,
        ms_resid=ms_res,
        f_stat=float(f_stat),
        prob_f=prob_f,
        r_squared=float(r_sq),
        adj_r_squared=float(adj_r_sq),
        root_mse=root_mse,
        coefficients=coef_d,
        std_errors=se_d,
        t_stats=t_d,
        p_values=p_d,
        ci_lower=cil_d,
        ci_upper=ciu_d,
        fitted_values=fitted,
        residuals=resid,
        is_robust=robust,
        X_df=clean_df[list(x)],
    )


def stata_summarize(
    df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    detail: bool = False,
) -> str:
    """
    Summary statistics table matching Stata's 'summarize' / 'sum, detail'.
    """
    cols = variables if variables is not None else list(df.select_dtypes(include=[np.number]).columns)
    lines = []

    if not detail:
        lines.append(f"{'Variable':>15s} | {'Obs':>8s}   {'Mean':>10s}   {'Std. dev.':>10s}   {'Min':>10s}   {'Max':>10s}")
        lines.append(f"{'-' * 15}-+-{'-' * 52}")
        for c in cols:
            s = df[c].dropna()
            n = len(s)
            mean = float(np.mean(s)) if n > 0 else 0.0
            std = float(np.std(s, ddof=1)) if n > 1 else 0.0
            vmin = float(np.min(s)) if n > 0 else 0.0
            vmax = float(np.max(s)) if n > 0 else 0.0
            lines.append(f"{c:>15s} | {n:>8,d}   {mean:>10.4f}   {std:>10.4f}   {vmin:>10.4f}   {vmax:>10.4f}")
    else:
        for c in cols:
            s = df[c].dropna()
            n = len(s)
            p = np.percentile(s, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            mean = float(np.mean(s))
            std = float(np.std(s, ddof=1))
            skew = float(stats.skew(s))
            kurt = float(stats.kurtosis(s, fisher=False))
            lines.append(f"\n                            {c}")
            lines.append("-" * 65)
            lines.append(f"      Percentiles      Smallest")
            lines.append(f" 1%   {p[0]:>10.4f}    {np.min(s):>10.4f}")
            lines.append(f" 5%   {p[1]:>10.4f}    {p[1]:>10.4f}         Obs             {n:>10,d}")
            lines.append(f"10%   {p[2]:>10.4f}    {p[2]:>10.4f}         Sum of wgt.     {n:>10,d}")
            lines.append(f"25%   {p[3]:>10.4f}    {p[3]:>10.4f}")
            lines.append(f"                                        Mean            {mean:>10.4f}")
            lines.append(f"50%   {p[4]:>10.4f}                         Std. dev.       {std:>10.4f}")
            lines.append(f"                       Largest")
            lines.append(f"75%   {p[5]:>10.4f}    {p[5]:>10.4f}         Variance        {std**2:>10.4f}")
            lines.append(f"90%   {p[6]:>10.4f}    {p[6]:>10.4f}         Skewness        {skew:>10.4f}")
            lines.append(f"95%   {p[7]:>10.4f}    {p[7]:>10.4f}         Kurtosis        {kurt:>10.4f}")
            lines.append(f"99%   {p[8]:>10.4f}    {np.max(s):>10.4f}")
            lines.append("-" * 65)

    return "\n".join(lines)


def stata_pwcorr(
    df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    sig: bool = True,
    star_level: float = 0.05,
) -> str:
    """
    Pairwise correlation matrix matching Stata's 'pwcorr varlist, star(0.05) sig'.
    """
    cols = variables if variables is not None else list(df.select_dtypes(include=[np.number]).columns)
    K = len(cols)
    clean_df = df[cols].dropna()
    N = len(clean_df)

    corr_matrix = np.corrcoef(clean_df.values, rowvar=False)

    lines = ["             | " + "  ".join([f"{c:>10s}" for c in cols])]
    lines.append("-------------+-" + "--".join(["-" * 10 for _ in cols]))

    for i in range(K):
        row_str = f"{cols[i]:>12s} | "
        sig_str = f"{'':>12s} | "
        for j in range(K):
            if j > i:
                continue
            r = corr_matrix[i, j]
            # p-value of correlation
            if i == j:
                p_val = 1.0
                star = " "
            else:
                t_stat = r * np.sqrt((N - 2) / max(1.0 - r**2, 1e-12))
                p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=N - 2))
                star = "*" if p_val < star_level else " "

            row_str += f"{r:>9.4f}{star} "
            sig_str += f"({p_val:>7.4f})  "

        lines.append(row_str)
        if sig and i > 0:
            lines.append(sig_str)

    return "\n".join(lines)
