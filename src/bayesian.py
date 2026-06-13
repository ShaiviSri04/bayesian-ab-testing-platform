# src/bayesian.py

import numpy as np
from scipy.stats import beta


def calculate_posteriors(df):

    variant_a = df[df["variant"] == "A"]
    variant_b = df[df["variant"] == "B"]

    conversions_a = variant_a["converted"].sum()
    conversions_b = variant_b["converted"].sum()

    failures_a = len(variant_a) - conversions_a
    failures_b = len(variant_b) - conversions_b

    # Beta(1,1) prior

    alpha_a = conversions_a + 1
    beta_a = failures_a + 1

    alpha_b = conversions_b + 1
    beta_b = failures_b + 1

    return alpha_a, beta_a, alpha_b, beta_b


def probability_b_wins(df, n_samples=100000):

    alpha_a, beta_a, alpha_b, beta_b = calculate_posteriors(df)

    samples_a = beta.rvs(
        alpha_a,
        beta_a,
        size=n_samples
    )

    samples_b = beta.rvs(
        alpha_b,
        beta_b,
        size=n_samples
    )

    probability = np.mean(samples_b > samples_a)

    return probability


def credible_interval(df, n_samples=100000):

    alpha_a, beta_a, alpha_b, beta_b = calculate_posteriors(df)

    samples_a = beta.rvs(
        alpha_a,
        beta_a,
        size=n_samples
    )

    samples_b = beta.rvs(
        alpha_b,
        beta_b,
        size=n_samples
    )

    uplift_samples = samples_b - samples_a

    lower = np.percentile(uplift_samples, 2.5)
    upper = np.percentile(uplift_samples, 97.5)

    return lower, upper