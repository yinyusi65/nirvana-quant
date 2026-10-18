"""
UNIT TESTS: NIRVANA CONSUMER ECONOMETRICS MODULE
=================================================
Validates consumer domain data loading, cycle detection,
directed network estimation, and semantic category synthesis.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nirvana import NirvanaConnectedness, Base12Manifold
from nirvana.domains.consumer import (
    DEFAULT_CONSUMER_CATEGORIES,
    DEFAULT_EXPENDITURE_WEIGHTS,
    analyze_consumer_dynamics,
)


def test_consumer_panel_loading():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'consumer_panel_2019_2025.csv')
    assert os.path.exists(data_path), "Consumer dataset file exists"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    assert df.shape[0] > 1000, "Panel has over 1000 sessions"
    assert df.shape[1] == 12, "Panel contains 12 consumer champions"
    assert not df.isna().any().any(), "Zero missing values in panel"


def test_consumer_manifold_dominant_period():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'consumer_panel_2019_2025.csv')
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    p1, p2 = Base12Manifold.estimate_dominant_periods(df['VNM'].values)
    assert p1 > 0 and p2 > 0, "Periods must be positive"
    assert p1 <= p2, "p1 <= p2 ordering preserved"


def test_consumer_connectedness_and_domain_analysis():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'consumer_panel_2019_2025.csv')
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    engine = NirvanaConnectedness(
        lags=[1],
        kalman_q=1e-3,
        kalman_r=1e-2,
        apoptosis_threshold=0.015,
        orthogonalize_market=True,
        n_market_factors=1,
        asset_weights=DEFAULT_EXPENDITURE_WEIGHTS,
    )
    report = engine.fit(df)

    assert 0.0 <= report.total_connectedness_index <= 100.0, "TCI in [0, 100]"
    assert report.market_variance_share is not None, "Market variance share computed"

    insights = analyze_consumer_dynamics(report)
    assert "dominant_regime" in insights, "Regime classified"
    assert len(insights["pricing_power_leaders"]) > 0, "Pricing power leaders identified"
    assert len(insights["most_vulnerable_demand_absorbers"]) > 0, "Demand absorbers identified"


if __name__ == '__main__':
    test_consumer_panel_loading()
    test_consumer_manifold_dominant_period()
    test_consumer_connectedness_and_domain_analysis()
    print("ALL CONSUMER TESTS PASSED 100%!")
