"""
Base-12 Dual-Clock Manifold Module
===================================
Maps linear discrete time onto a 2-Torus manifold (S1 x S1) using Base-12 harmonic octaves.
Synchronizes macro-economic calendar time (Quarterly BCTC cycle) with micro-information flow.
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class DualClockCoordinates:
    macro_sin_1: float
    macro_cos_1: float
    macro_sin_2: float
    macro_cos_2: float
    micro_sin_1: float
    micro_cos_1: float
    micro_sin_2: float
    micro_cos_2: float
    resonance_gamma: float

    def to_vector(self) -> List[float]:
        return [
            self.macro_sin_1,
            self.macro_cos_1,
            self.macro_sin_2,
            self.macro_cos_2,
            self.micro_sin_1,
            self.micro_cos_1,
            self.micro_sin_2,
            self.micro_cos_2,
            self.resonance_gamma,
        ]


class Base12Manifold:
    """
    Transforms arbitrary time series into Base-12 Dual-Clock Manifold.
    - Macro Octave 1: Primary structural cycle (default 63 sessions / quarter)
    - Macro Octave 2: Secondary macro cycle (default 252 sessions / year)
    - Micro Octaves: Information / Volume quanta
    Supports custom macro_periods for cross-domain adaptation (e.g. 24h & 168h for network traffic).
    """

    def __init__(
        self,
        volume_quantum: float = 1000.0,
        macro_periods: Optional[Tuple[float, float]] = None,
    ):
        self.volume_quantum = max(volume_quantum, 1e-4)
        self.macro_periods = (
            (float(macro_periods[0]), float(macro_periods[1]))
            if macro_periods is not None
            else (63.0, 252.0)
        )

    @staticmethod
    def estimate_dominant_periods(
        signal: Union[np.ndarray, pd.Series], top_k: int = 2
    ) -> List[float]:
        """
        Estimates dominant cyclical periods from a raw 1D time-series using FFT spectral power.
        Useful for uncalibrated non-financial domains (IoT sensors, cyber packet rates, climate).
        """
        s = np.asarray(signal, dtype=float)
        s = s[~np.isnan(s)]
        if len(s) < 16:
            return [63.0, 252.0]
        # Linear detrending to remove baseline drift
        s_detrend = s - np.linspace(s[0], s[-1], len(s))
        fft_vals = np.fft.rfft(s_detrend)
        freqs = np.fft.rfftfreq(len(s_detrend))
        power = np.abs(fft_vals) ** 2
        power[0] = 0.0  # Zero DC component

        idx_sorted = np.argsort(power)[::-1]
        periods = []
        for idx in idx_sorted:
            if freqs[idx] > 0:
                p = 1.0 / freqs[idx]
                if 2.0 <= p <= len(s):
                    periods.append(float(round(p, 1)))
                    if len(periods) == top_k:
                        break
        return periods if len(periods) == top_k else [63.0, 252.0]

    def transform(
        self,
        calendar_index: int,
        cumulative_volume: float = 0.0,
    ) -> DualClockCoordinates:
        p1, p2 = self.macro_periods
        theta_m1 = (2.0 * math.pi * (calendar_index % p1)) / p1
        m_sin_1 = math.sin(theta_m1)
        m_cos_1 = math.cos(theta_m1)

        theta_m2 = (2.0 * math.pi * (calendar_index % p2)) / p2
        m_sin_2 = math.sin(theta_m2)
        m_cos_2 = math.cos(theta_m2)

        # Volume/Information Time (Micro Base-12)
        volume_bars = int(cumulative_volume / self.volume_quantum)
        theta_v1 = (2.0 * math.pi * (volume_bars % 12)) / 12.0
        v_sin_1 = math.sin(theta_v1)
        v_cos_1 = math.cos(theta_v1)

        theta_v2 = (2.0 * math.pi * (volume_bars % 144)) / 144.0
        v_sin_2 = math.sin(theta_v2)
        v_cos_2 = math.cos(theta_v2)

        # Phase Resonance Index: gamma(t) = cos(theta_macro - theta_micro)
        resonance_gamma = math.cos(theta_m1 - theta_v1)

        return DualClockCoordinates(
            macro_sin_1=m_sin_1,
            macro_cos_1=m_cos_1,
            macro_sin_2=m_sin_2,
            macro_cos_2=m_cos_2,
            micro_sin_1=v_sin_1,
            micro_cos_1=v_cos_1,
            micro_sin_2=v_sin_2,
            micro_cos_2=v_cos_2,
            resonance_gamma=resonance_gamma,
        )

    def batch_transform(
        self,
        calendar_indices: Union[np.ndarray, pd.Series, list, range, int],
        cumulative_volumes: Optional[Union[np.ndarray, pd.Series, list]] = None,
        dates: Optional[Union[pd.DatetimeIndex, list, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Batch transformation for pandas Series / numpy arrays.
        Supports actual calendar dates or empirical trading sessions.
        Returns shape (N, 9).
        """
        cal_idx = np.atleast_1d(np.asarray(calendar_indices, dtype=int))
        n = len(cal_idx)

        if cumulative_volumes is None:
            cum_vol = np.zeros(n, dtype=float)
        else:
            cum_vol = np.nan_to_num(np.asarray(cumulative_volumes, dtype=float), nan=0.0)
            if len(cum_vol) != n:
                cum_vol = np.resize(cum_vol, n)

        p1, p2 = self.macro_periods
        if dates is not None and self.macro_periods == (63.0, 252.0):
            try:
                dt_idx = pd.to_datetime(dates)
                day_of_year = dt_idx.dayofyear.values
                quarter_len = 365.25 / 4.0
                theta_m1 = (2.0 * np.pi * (day_of_year % quarter_len)) / quarter_len
                theta_m2 = (2.0 * np.pi * (day_of_year / 365.25))
            except Exception:
                theta_m1 = (2.0 * np.pi * (cal_idx % p1)) / p1
                theta_m2 = (2.0 * np.pi * (cal_idx % p2)) / p2
        else:
            theta_m1 = (2.0 * np.pi * (cal_idx % p1)) / p1
            theta_m2 = (2.0 * np.pi * (cal_idx % p2)) / p2

        v_bars = np.maximum((cum_vol / self.volume_quantum).astype(int), 0)
        theta_v1 = (2.0 * np.pi * (v_bars % 12)) / 12.0
        theta_v2 = (2.0 * np.pi * (v_bars % 144)) / 144.0

        gamma = np.cos(theta_m1 - theta_v1)

        features = np.column_stack([
            np.sin(theta_m1),
            np.cos(theta_m1),
            np.sin(theta_m2),
            np.cos(theta_m2),
            np.sin(theta_v1),
            np.cos(theta_v1),
            np.sin(theta_v2),
            np.cos(theta_v2),
            gamma,
        ])
        return features
