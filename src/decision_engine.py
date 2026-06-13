# src/decision_engine.py

from src.bayesian import probability_b_wins
from src.frequentist import (
    calculate_conversion_test,
    calculate_ctr_test,
    calculate_revenue_test
)
from src.early_stopping import (
    find_early_stop,
    users_saved
)


def risk_level(probability):

    if probability >= 0.99:
        return "Low"

    elif probability >= 0.95:
        return "Medium"

    else:
        return "High"


def recommend_action(df):

    prob_b_better = probability_b_wins(df)

    _, conversion_p = calculate_conversion_test(df)
    _, revenue_p = calculate_revenue_test(df)

    if (
        prob_b_better > 0.95
        and conversion_p < 0.05
        and revenue_p < 0.05
    ):

        return "Deploy Variant B"

    elif prob_b_better > 0.80:

        return "Continue Experiment"

    else:

        return "Insufficient Evidence"


def executive_summary(df):

    prob_b_better = probability_b_wins(df)

    _, conversion_p = calculate_conversion_test(df)
    _, ctr_p = calculate_ctr_test(df)
    _, revenue_p = calculate_revenue_test(df)

    stop_point = find_early_stop(df)

    saved = users_saved(df)

    risk = risk_level(prob_b_better)

    recommendation = recommend_action(df)

    summary = f"""
    ===== EXECUTIVE SUMMARY =====

    Probability B Wins: {prob_b_better:.2%}

    Conversion P-Value: {conversion_p:.4e}

    CTR P-Value: {ctr_p:.4e}

    Revenue P-Value: {revenue_p:.4e}

    Early Stop Point: {stop_point}% traffic

    Users Saved: {saved:,}

    Risk Level: {risk}

    Recommendation: {recommendation}

    =============================
    """

    return summary