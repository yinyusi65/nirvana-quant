"""
High-Frequency Volatility Estimators Module
===========================================
Implements academic range-based volatility estimators:
1. Parkinson (1980): High-Low range (diffusion only)
2. Garman-Klass (1980): Open-High-Low-Close (captures intraday trend)
3. Yang-Zhang (2000): Overnight Jump + Drift Independent (continuous + discrete jump)

Solves the 'overnight gap blindness' of standard Parkinson volatility.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd


def compute_parkinson(
    high: Union[np.ndarray, pd.Series, list],
    low: Union[np.ndarray, pd.Series, list],
    close: Optional[Union[np.ndarray, pd.Series, list]] = None,
    prev_close: Optional[Union[np.ndarray, pd.Series, list]] = None,
    min_floor_var: float = 1e-4,
) -> Union[np.ndarray, pd.Series]:
    """
    Parkinson (1980) Volatility with Price-Limit Censoring Correction.
    When H == L (limit-up or limit-down frozen trading session):
    If close and prev_close are provided, recovers true volatility from overnight jump return.
    Otherwise applies calibrated min_floor_var (default 1e-4 ~ 15.8% annualized).
    """
    is_series = isinstance(high, pd.Series)
    idx = high.index if is_series else None

    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float) if close is not None else None
    pc = np.asarray(prev_close, dtype=float) if prev_close is not None else None

    h_l = np.maximum(h / np.maximum(l, 1e-8), 1.0)
    var = (np.log(h_l) ** 2) / (4.0 * np.log(2.0))

    # Censoring correction: if H == L and price jumped significantly
    if c is not None and pc is not None:
        jump_ret = np.abs(np.log(np.maximum(c / np.maximum(pc, 1e-8), 1e-8)))
        is_limit = (h_l <= 1.0001) & (jump_ret > 0.01)
        jump_var = (jump_ret ** 2) / (2.0 * np.log(2.0))
        var = np.where(is_limit, np.maximum(var, jump_var), var)

    var = np.maximum(var, min_floor_var)
    res = np.log(100.0 * np.sqrt(252.0 * var))
    return pd.Series(res, index=idx, name="Parkinson_Vol") if is_series else res


def compute_garman_klass(
    open_p: Union[np.ndarray, pd.Series, list],
    high: Union[np.ndarray, pd.Series, list],
    low: Union[np.ndarray, pd.Series, list],
    close: Union[np.ndarray, pd.Series, list],
    min_floor_var: float = 1e-4,
) -> Union[np.ndarray, pd.Series]:
    """Garman-Klass (1980) Volatility (Intraday drift-robust)."""
    is_series = isinstance(high, pd.Series)
    idx = high.index if is_series else None

    o = np.asarray(open_p, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)

    h_l = np.maximum(h / np.maximum(l, 1e-8), 1.0)
    c_o = np.maximum(c / np.maximum(o, 1e-8), 1e-8)

    term1 = 0.5 * (np.log(h_l) ** 2)
    term2 = (2.0 * np.log(2.0) - 1.0) * (np.log(c_o) ** 2)
    var = np.maximum(term1 - term2, min_floor_var)
    res = np.log(100.0 * np.sqrt(252.0 * var))
    return pd.Series(res, index=idx, name="Garman_Klass_Vol") if is_series else res


def compute_yang_zhang(
    open_p: Union[np.ndarray, pd.Series, list],
    high: Union[np.ndarray, pd.Series, list],
    low: Union[np.ndarray, pd.Series, list],
    close: Union[np.ndarray, pd.Series, list],
    prev_close: Union[np.ndarray, pd.Series, list],
    k: float = 0.34,
    min_floor_var: float = 1e-4,
) -> Union[np.ndarray, pd.Series]:
    """
    Yang-Zhang (2000) Volatility:
    Captures discrete overnight gaps (Close_{t-1} -> Open_t) + continuous intraday drift.
    """
    is_series = isinstance(high, pd.Series)
    idx = high.index if is_series else None

    o = np.asarray(open_p, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    pc = np.asarray(prev_close, dtype=float)

    # Open-to-close variance
    c_o = np.maximum(c / np.maximum(o, 1e-8), 1e-8)
    var_open_close = np.log(c_o) ** 2

    # Overnight gap variance (fall back to open-to-close variance if Day 0 prev_close is missing)
    valid_pc = ~np.isnan(pc)
    safe_pc = np.where(valid_pc, pc, o)
    o_c_prev = np.maximum(o / np.maximum(safe_pc, 1e-8), 1e-8)
    var_overnight = np.where(valid_pc, np.log(o_c_prev) ** 2, var_open_close)

    # Rogers-Satchell intraday variance (guaranteed non-negative by bounds h >= 1.0 and l <= 1.0)
    h_c = np.maximum(h / np.maximum(c, 1e-8), 1.0)
    h_o = np.maximum(h / np.maximum(o, 1e-8), 1.0)
    l_c = np.maximum(np.minimum(l / np.maximum(c, 1e-8), 1.0), 1e-8)
    l_o = np.maximum(np.minimum(l / np.maximum(o, 1e-8), 1.0), 1e-8)
    var_rs = np.log(h_c) * np.log(h_o) + np.log(l_c) * np.log(l_o)

    var_total = var_overnight + k * var_open_close + (1.0 - k) * var_rs
    var_total = np.maximum(var_total, min_floor_var)
    res = np.log(100.0 * np.sqrt(252.0 * var_total))
    return pd.Series(res, index=idx, name="Yang_Zhang_Vol") if is_series else res
