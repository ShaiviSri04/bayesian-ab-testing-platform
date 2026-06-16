import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
import streamlit as st

from src.utils import load_data

from src.bayesian import (
    calculate_posteriors,
    probability_b_wins,
    credible_interval,
)

st.set_page_config(
    page_title="Bayesian Inference",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(html: str) -> None:
    """Render an HTML block, stripping per-line leading whitespace so
    markdown does not mistake indented HTML for a code block."""
    lines = [line.strip() for line in html.strip().splitlines()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Global styling (matches app.py / Overview / EDA / Frequentist)
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0F1117;
        color: #F5F3FF;
    }

    .main .block-container {
        max-width: 1150px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #111318;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* ── Page header ── */
    .page-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #F5F3FF;
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin-bottom: 0.35rem;
    }
    .page-subtitle {
        font-size: 1.0rem;
        color: #A1A1AA;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #A1A1AA;
        margin-bottom: 0.9rem;
        margin-top: 0.25rem;
    }

    /* ── Card kicker (small label above card titles) ── */
    .card-kicker {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #A78BFA;
        margin-bottom: 0.4rem;
    }

    /* ── Bordered containers (st.container(border=True)) ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #181C25;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 18px !important;
        padding: 0.6rem 0.5rem;
        transition: border-color 0.18s ease;
        position: relative;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(167,139,250,0.25);
    }

    .content-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F5F3FF;
        letter-spacing: -0.01em;
        margin-bottom: 0.5rem;
    }
    .content-card-sub {
        font-size: 0.85rem;
        color: #A1A1AA;
        margin-bottom: 1rem;
    }
    .content-card-body {
        font-size: 0.93rem;
        color: #A1A1AA;
        line-height: 1.85;
        margin: 0;
    }

    /* ── KPI metric cards (with accent line) ── */
    div[data-testid="stMetric"] {
        background-color: #181C25;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 1.35rem 1.4rem 1.25rem;
        transition: border-color 0.18s ease, transform 0.18s ease;
        position: relative;
        overflow: hidden;
        height: 100%;
    }
    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #A78BFA 0%, rgba(167,139,250,0.15) 100%);
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(167,139,250,0.25);
        transform: translateY(-1px);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #A1A1AA;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F5F3FF;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.35;
        margin-top: 0.35rem;
        word-break: keep-all;
    }
    div[data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }

    /* ── Pills / badges under metrics ── */
    .metric-foot {
        margin-top: 0.7rem;
    }
    .pill {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        white-space: nowrap;
    }
    .pill-active {
        background-color: #221D34;
        color: #C4B5FD;
        border: 1px solid rgba(167,139,250,0.25);
    }
    .pill-soft {
        background-color: #1B1830;
        color: #A78BFA;
        border: 1px solid rgba(167,139,250,0.12);
    }
    .pill-muted {
        background-color: transparent;
        color: #71717A;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Custom data tables ── */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .data-table thead th {
        text-align: left;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #71717A;
        padding: 0 0.75rem 0.7rem 0.75rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .data-table tbody td {
        padding: 0.85rem 0.75rem;
        color: #D4D4D8;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-weight: 500;
    }
    .data-table tbody tr:last-child td {
        border-bottom: none;
    }
    .data-table tbody tr:hover {
        background-color: rgba(167,139,250,0.04);
    }
    .data-table td.row-label {
        font-weight: 600;
        color: #F5F3FF;
    }
    .data-table td.mono {
        color: #A1A1AA;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    /* ── Interpretation / info cards (no green/red banners) ── */
    .info-card {
        background-color: #1B1830;
        border: 1px solid rgba(167,139,250,0.15);
        border-radius: 14px;
        padding: 1.15rem 1.4rem;
        font-size: 0.95rem;
        color: #D4D4D8;
        line-height: 1.85;
    }
    .info-card strong {
        color: #C4B5FD;
        font-weight: 600;
    }

    /* ── Footer ── */
    .footer-note {
        font-size: 0.85rem;
        color: #A1A1AA;
        text-align: center;
        padding-top: 1.5rem;
        margin-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# Data & Bayesian calculations
# ──────────────────────────────────────────────────────────────────────────
df = load_data()

prob_b = probability_b_wins(df)

lower, upper = credible_interval(df)

alpha_a, beta_a, alpha_b, beta_b = calculate_posteriors(df)

conversion_a = alpha_a / (alpha_a + beta_a)
conversion_b = alpha_b / (alpha_b + beta_b)

uplift = ((conversion_b - conversion_a) / conversion_a) * 100

risk = 1 - prob_b

# ──────────────────────────────────────────────────────────────────────────
# SECTION 0 — Header
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Bayesian Inference</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Estimate winning probabilities and quantify '
    'uncertainty using Bayesian statistics.</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# SECTION 1 — Hero Card
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Why Bayesian Analysis?</div>', unsafe_allow_html=True)
    render_html(
        """
        <div class="content-card-body">
            Unlike traditional hypothesis testing, Bayesian inference directly
            estimates the probability that one variant is better than another,
            making experiment results easier to interpret, communicate, and
            act upon.
        </div>
        """
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 2 — KPI Cards
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Bayesian Metrics</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(label="P(B Wins)", value=f"{prob_b:.2%}")
    if prob_b > 0.95:
        pill_class, pill_text = "pill-active", "Strong evidence"
    elif prob_b > 0.80:
        pill_class, pill_text = "pill-soft", "Promising"
    else:
        pill_class, pill_text = "pill-muted", "Inconclusive"
    render_html(f'<div class="metric-foot"><span class="pill {pill_class}">{pill_text}</span></div>')

with k2:
    st.metric(label="Credible Interval", value=f"[{lower:.2%}, {upper:.2%}]")
    render_html('<div class="metric-foot"><span class="pill pill-muted">Uplift range, B \u2212 A</span></div>')

with k3:
    st.metric(label="Expected Uplift", value=f"{uplift:.2f}%")
    uplift_class = "pill-active" if uplift > 0 else "pill-muted"
    uplift_text = "Positive" if uplift > 0 else "Negative"
    render_html(f'<div class="metric-foot"><span class="pill {uplift_class}">{uplift_text}</span></div>')

with k4:
    st.metric(label="Risk Level", value=f"{risk:.2%}")
    if risk < 0.05:
        risk_class, risk_text = "pill-active", "Low risk"
    elif risk < 0.20:
        risk_class, risk_text = "pill-soft", "Moderate risk"
    else:
        risk_class, risk_text = "pill-muted", "Elevated risk"
    render_html(f'<div class="metric-foot"><span class="pill {risk_class}">{risk_text}</span></div>')

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 3 — Posterior Parameters
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Posterior</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Posterior Parameters</div>', unsafe_allow_html=True)
    render_html(
        """
        <div class="content-card-sub">
            Posterior distributions combine prior beliefs with observed
            experiment data.
        </div>
        """
    )

    posterior_rows = (
        f'<tr><td class="row-label">A</td>'
        f'<td class="mono">{alpha_a:.2f}</td>'
        f'<td class="mono">{beta_a:.2f}</td></tr>'
        f'<tr><td class="row-label">B</td>'
        f'<td class="mono">{alpha_b:.2f}</td>'
        f'<td class="mono">{beta_b:.2f}</td></tr>'
    )

    render_html(
        f"""
        <table class="data-table">
            <thead>
                <tr>
                    <th>Variant</th>
                    <th>Alpha</th>
                    <th>Beta</th>
                </tr>
            </thead>
            <tbody>
                {posterior_rows}
            </tbody>
        </table>
        """
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 4 — Posterior Distribution Visualization
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Posterior Distributions</div>', unsafe_allow_html=True)
    render_html(
        """
        <div class="content-card-sub">
            Visualize the uncertainty around the conversion rates of both
            variants after observing experiment data.
        </div>
        """
    )

    x = np.linspace(0, 0.20, 1000)

    y_a = beta.pdf(x, alpha_a, beta_a)
    y_b = beta.pdf(x, alpha_b, beta_b)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#181C25")
    ax.set_facecolor("#181C25")

    ax.plot(x, y_a, label="Variant A", linewidth=2, color="#71717A")
    ax.plot(x, y_b, label="Variant B", linewidth=2, color="#A78BFA")

    ax.fill_between(x, y_a, alpha=0.18, color="#71717A")
    ax.fill_between(x, y_b, alpha=0.18, color="#A78BFA")

    ax.set_xlabel("Conversion Rate", color="#A1A1AA")
    ax.set_ylabel("Density", color="#A1A1AA")

    ax.tick_params(colors="#A1A1AA")

    for spine in ax.spines.values():
        spine.set_color("#2A2E3A")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="y", color="#2A2E3A", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    legend = ax.legend(facecolor="#1B1830", edgecolor="#2A2E3A")
    for text in legend.get_texts():
        text.set_color("#D4D4D8")

    st.pyplot(fig)

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 5 — Probability Interpretation
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Inference</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Probability Interpretation</div>', unsafe_allow_html=True)

    if prob_b > 0.95:
        interpretation_text = (
            "Variant B has overwhelming evidence of outperforming Variant A "
            "and is highly likely to be the better experience for users."
        )
    elif prob_b > 0.80:
        interpretation_text = (
            "Variant B currently appears stronger than Variant A, though "
            "additional data may further reduce uncertainty."
        )
    else:
        interpretation_text = (
            "Current evidence remains inconclusive, and additional "
            "experimentation may be beneficial before making a decision."
        )

    render_html(f'<div class="info-card">{interpretation_text}</div>')

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 6 — Credible Interval
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Uncertainty</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Credible Interval</div>', unsafe_allow_html=True)
    render_html(
        """
        <div class="content-card-sub">
            A credible interval represents the range within which the true
            conversion uplift is likely to fall with high probability.
        </div>
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(label="Lower Bound", value=f"{lower:.2%}")

    with c2:
        st.metric(label="Upper Bound", value=f"{upper:.2%}")

    st.caption("95% credible interval of uplift (Variant B \u2212 Variant A).")

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 7 — Business Takeaway
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Decision</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Business Takeaway</div>', unsafe_allow_html=True)

    if prob_b >= 0.95:
        takeaway_text = (
            f"Bayesian analysis estimates a <strong>{prob_b:.2%}</strong> "
            f"probability that Variant B outperforms Variant A, with an "
            f"expected uplift of <strong>{uplift:.2f}%</strong>. Deployment "
            f"risk remains low at <strong>{risk:.2%}</strong>, providing "
            f"strong evidence in favor of launching Variant B into "
            f"production."
        )
    elif prob_b >= 0.80:
        takeaway_text = (
            f"Variant B currently demonstrates promising results with a "
            f"<strong>{prob_b:.2%}</strong> probability of outperforming "
            f"Variant A and an expected uplift of "
            f"<strong>{uplift:.2f}%</strong>. With a deployment risk of "
            f"<strong>{risk:.2%}</strong>, additional data collection may "
            f"further reduce uncertainty before deployment."
        )
    else:
        takeaway_text = (
            f"Current evidence remains inconclusive, with a winning "
            f"probability of only <strong>{prob_b:.2%}</strong> and a "
            f"deployment risk of <strong>{risk:.2%}</strong>. Further "
            f"experimentation is recommended before making deployment "
            f"decisions."
        )

    render_html(f'<div class="info-card">{takeaway_text}</div>')

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 8 — Deployment Recommendation
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Recommendation</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Deployment Recommendation</div>', unsafe_allow_html=True)

    if prob_b >= 0.95:
        recommendation = "Deploy Variant B"
    elif prob_b >= 0.80:
        recommendation = "Continue Experiment"
    else:
        recommendation = "Insufficient Evidence"

    c1, c2 = st.columns(2)

    with c1:
        st.metric(label="Recommendation", value=recommendation)
        if prob_b >= 0.95:
            rec_class, rec_text = "pill-active", "High confidence"
        elif prob_b >= 0.80:
            rec_class, rec_text = "pill-soft", "Monitor further"
        else:
            rec_class, rec_text = "pill-muted", "Low confidence"
        render_html(f'<div class="metric-foot"><span class="pill {rec_class}">{rec_text}</span></div>')

    with c2:
        st.metric(label="Risk Level", value=f"{risk:.2%}")
        render_html(
            f'<div class="metric-foot"><span class="pill pill-muted">'
            f'P(A Wins) = {(1 - prob_b):.2%}</span></div>'
        )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 9 — Experiment Conclusion
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Conclusion</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Experiment Conclusion</div>', unsafe_allow_html=True)

    conclusion_text = (
        f"Bayesian inference estimates a <strong>{prob_b:.2%}</strong> "
        f"probability that Variant B is superior to Variant A. The "
        f"estimated conversion uplift is <strong>{uplift:.2f}%</strong>, "
        f"with a 95% credible interval ranging from "
        f"<strong>{lower:.2%}</strong> to <strong>{upper:.2%}</strong>. "
        f"These results provide a probabilistic foundation for experiment "
        f"decisions while explicitly quantifying uncertainty and "
        f"deployment risk."
    )

    render_html(f'<div class="info-card">{conclusion_text}</div>')

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 10 — Posterior Mean Conversion Rates
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Posterior Mean Conversion Rates</div>', unsafe_allow_html=True)
    render_html(
        """
        <div class="content-card-sub">
            Posterior means estimate the expected conversion rate after
            combining prior beliefs with observed data.
        </div>
        """
    )

    means_rows = (
        f'<tr><td class="row-label">A</td>'
        f'<td class="mono">{conversion_a:.2%}</td></tr>'
        f'<tr><td class="row-label">B</td>'
        f'<td class="mono">{conversion_b:.2%}</td></tr>'
    )

    render_html(
        f"""
        <table class="data-table">
            <thead>
                <tr>
                    <th>Variant</th>
                    <th>Posterior Mean Conversion</th>
                </tr>
            </thead>
            <tbody>
                {means_rows}
            </tbody>
        </table>
        """
    )

# ──────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-note">Proceed to Early Stopping to determine when '
    'experiments can safely be terminated.</div>',
    unsafe_allow_html=True,
)