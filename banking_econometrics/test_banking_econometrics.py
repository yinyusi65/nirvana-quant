"""
Test Suite for Banking Econometrics Module
=========================================
Verifies the complete execution pipeline for 27 Vietnamese commercial banks.

Author: yinyusi65
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from banking_econometrics.run_banking_27_analysis import run_banking_systemic_risk_pipeline


def test_banking_pipeline_execution(tmp_path):
    output_dir = str(tmp_path / "banking_test_output")
    results = run_banking_systemic_risk_pipeline(output_dir=output_dir)

    assert "rep_balanced" in results
    assert "rep_macro" in results
    assert "rep_ortho" in results
    assert "df_cross" in results
    assert "proof_rep" in results

    # Verify TCI metrics are within academic reasonable bounds
    assert 60.0 <= results["rep_balanced"].total_connectedness_index <= 90.0
    assert 60.0 <= results["rep_macro"].total_connectedness_index <= 90.0
    assert results["rep_ortho"].total_connectedness_index < results["rep_macro"].total_connectedness_index

    # Verify 27 banks are present in cross-sectional dataframe
    assert len(results["df_cross"]) == 27
    assert "Net_Spillover" in results["df_cross"].columns
    assert "Market_Beta" in results["df_cross"].columns

    # Verify output files exist
    assert os.path.exists(os.path.join(output_dir, "banking_systemic_risk_table_2017_2025.csv"))
    assert os.path.exists(os.path.join(output_dir, "banking_macro_regimes_summary.csv"))
    assert os.path.exists(os.path.join(output_dir, "banking_execution_report.txt"))
