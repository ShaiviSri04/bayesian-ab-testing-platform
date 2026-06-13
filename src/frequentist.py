# src/frequentist.py

from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import ttest_ind


def calculate_conversion_test(df):

    variant_a = df[df["variant"] == "A"]
    variant_b = df[df["variant"] == "B"]

    success_a = variant_a["converted"].sum()
    success_b = variant_b["converted"].sum()

    users_a = len(variant_a)
    users_b = len(variant_b)

    z_stat, p_value = proportions_ztest(
        [success_a, success_b],
        [users_a, users_b]
    )

    return z_stat, p_value


def calculate_ctr_test(df):

    variant_a = df[df["variant"] == "A"]
    variant_b = df[df["variant"] == "B"]

    clicks_a = variant_a["clicked"].sum()
    clicks_b = variant_b["clicked"].sum()

    users_a = len(variant_a)
    users_b = len(variant_b)

    z_stat, p_value = proportions_ztest(
        [clicks_a, clicks_b],
        [users_a, users_b]
    )

    return z_stat, p_value


def calculate_revenue_test(df):

    revenue_a = df.loc[
        df["variant"] == "A",
        "experiment_revenue"
    ]

    revenue_b = df.loc[
        df["variant"] == "B",
        "experiment_revenue"
    ]

    t_stat, p_value = ttest_ind(
        revenue_a,
        revenue_b,
        equal_var=False,
        nan_policy="omit"
    )

    return t_stat, p_value