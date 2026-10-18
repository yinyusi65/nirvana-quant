"""
NIRVANA DOMAIN EXTENSION: MICRO-ECONOMETRIC PRODUCT SUBSTITUTION
=================================================================
Specialized domain module for analyzing cross-price elasticity,
budget constraint binding thresholds, and consumer brand/product switching.

Author: yinyusi65 & Nirvana Quant Team
"""

from typing import Any, Dict, List, Optional
import numpy as np


DEFAULT_PROTEIN_SUBSTITUTE_ROLES = {
    "Pork_3F": "Anchor Good (Primary Household Staple)",
    "Chicken_CP": "Close Substitute (Budget-Conscious Protein Alternative)",
    "Eggs": "Secondary Macro-Buffer (Low-Cost Calorie Cushion)",
}


def analyze_substitution_thresholds(report: Any) -> Dict[str, Any]:
    """
    Extracts structural micro-econometric switching thresholds from
    a fitted Nirvana StaticEconometricReport.

    Parameters
    ----------
    report : StaticEconometricReport
        The report object returned by NirvanaStaticBooster.fit().

    Returns
    -------
    Dict[str, Any]
        Dictionary containing extracted price ratio triggers, binding income
        cutoffs, and economic interpretations.
    """
    threshold_rules = getattr(report, "top_substitution_thresholds", [])
    feat_importances = getattr(report, "feature_importances", {})

    # Detect dominant splitting features
    top_driver = list(feat_importances.keys())[0] if feat_importances else "Unknown"

    # Identify primary switching boundaries from rule descriptions
    critical_price_ratio = None
    binding_income_threshold = None

    for rule in threshold_rules:
        rule_lower = rule.lower()
        if "ratio" in rule_lower and critical_price_ratio is None:
            # Parse threshold number
            try:
                parts = rule.split("<=")
                if len(parts) > 1:
                    val_str = parts[1].split()[0]
                    critical_price_ratio = float(val_str)
            except Exception:
                pass
        elif "income" in rule_lower and binding_income_threshold is None:
            try:
                parts = rule.split("<=")
                if len(parts) > 1:
                    val_str = parts[1].split()[0]
                    binding_income_threshold = float(val_str)
            except Exception:
                pass

    return {
        "primary_non_linear_driver": top_driver,
        "critical_price_ratio_trigger": critical_price_ratio,
        "binding_income_threshold_million_vnd": binding_income_threshold,
        "r2_phase1_linear_baseline": getattr(report, "r2_phase1", 0.0),
        "r2_phase2_nirvana_enhanced": getattr(report, "r2_phase2", 0.0),
        "unexplained_variance_resolved_pct": getattr(report, "r2_relative_improvement_pct", 0.0),
        "economic_regime": (
            "Threshold-Driven Budget Substitution"
            if getattr(report, "r2_gain", 0.0) > 0.15
            else "Linear Proportional Substitution"
        ),
    }
