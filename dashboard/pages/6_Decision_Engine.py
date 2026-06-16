import streamlit as st
import pandas as pd

from src.utils import load_data

from src.bayesian import probability_b_wins

from src.frequentist import (
    calculate_conversion_test,
    calculate_ctr_test,
    calculate_revenue_test,
)

from src.early_stopping import (
    find_early_stop,
    users_saved,
)

from src.decision_engine import (
    recommend_action,
    risk_level,
)

st.set_page_config(
    page_title="Decision Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(html: str) -> None:
    """Render an HTML block, stripping per-line leading whitespace so
    markdown does not mistake indented HTML for a code block."""
    lines = [line.strip() for line in html.strip().splitlines()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Global styling (matches app.py / Overview / EDA / Frequentist / Bayesian /
# Early Stopping). No inline <span> pills are used anywhere on this page —
# all badges are rendered through st.metric, bordered containers, and
# block-level divs only.
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

    .sidebar-brand-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F5F3FF;
        letter-spacing: -0.01em;
        margin-bottom: 0.15rem;
    }
    .sidebar-brand-sub {
        font-size: 0.8rem;
        color: #A1A1AA;
        margin-bottom: 1rem;
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

    /* ── Card kicker ── */
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

    /* ── Dataframe container ── */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        overflow: hidden;
    }

    /* ── Info cards (block-level, no inline spans) ── */
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

    /* ── Status badge cards (replaces inline pills) ── */
    .badge-row {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    .badge-card {
        background-color: #14121C;
        border: 1px solid rgba(255,255,255,0.06);
        border-left: 3px solid #A78BFA;
        border-radius: 10px;
        padding: 0.75rem 1.1rem;
        flex: 1;
        min-width: 220px;
    }
    .badge-card.risk-low {
        border-left-color: #A78BFA;
    }
    .badge-card.risk-medium {
        border-left-color: #71717A;
    }
    .badge-card.risk-high {
        border-left-color: #4B4458;
    }
    .badge-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #71717A;
        margin-bottom: 0.3rem;
    }
    .badge-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F5F3FF;
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
# Sidebar branding
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand-title">A/B Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-brand-sub">Experiment Analytics Engine</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# Data & computed metrics
# ──────────────────────────────────────────────────────────────────────────
df = load_data()

prob_b = probability_b_wins(df)

_, p_conv = calculate_conversion_test(df)
_, p_ctr = calculate_ctr_test(df)
_, p_rev = calculate_revenue_test(df)

stop_point = find_early_stop(df)
saved = users_saved(df)

recommendation = recommend_action(df)
risk = risk_level(prob_b)

stop_reached = stop_point is not None
stop_display = f"{stop_point:.0f}%" if stop_reached else "Not Reached"

risk_css_class = {
    "Low": "risk-low",
    "Medium": "risk-medium",
    "High": "risk-high",
}.get(risk, "risk-medium")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 0 — Header
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Decision Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Translate statistical evidence into '
    'actionable business decisions.</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# SECTION 1 — Hero Card
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Decision Intelligence</div>', unsafe_allow_html=True)
    render_html(
        """
        <div class="content-card-body">
            The Decision Engine consolidates Bayesian inference, frequentist
            testing, and early stopping analysis into a unified
            recommendation framework. Rather than interpreting multiple
            statistical outputs independently, teams can rely on a single
            evidence-based recommendation to support product decisions.
        </div>
        """
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 2 — KPI Cards
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Decision Metrics</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(label="Recommendation", value=recommendation)

with k2:
    st.metric(label="Risk Level", value=risk)

with k3:
    st.metric(label="P(B Wins)", value=f"{prob_b:.2%}")

with k4:
    st.metric(label="Users Saved", value=f"{saved:,}")

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 3 — Executive Recommendation
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Decision</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Executive Recommendation</div>', unsafe_allow_html=True)

    if recommendation == "Deploy Variant B":
        message = (
            f"Variant B demonstrates superior performance across multiple "
            f"statistical frameworks while maintaining "
            f"**{risk.lower()} deployment risk**. Bayesian inference "
            f"estimates a **{prob_b:.2%} probability** that Variant B "
            f"outperforms Variant A. The evidence strongly supports moving "
            f"forward with Variant B."
        )
    elif recommendation == "Continue Experiment":
        message = (
            f"Current evidence favors Variant B with a winning probability "
            f"of **{prob_b:.2%}**. However, additional observations may "
            f"further reduce uncertainty before a final decision is made."
        )
    else:
        message = (
            "Existing evidence remains inconclusive. Additional traffic "
            "and experimentation are recommended before selecting a "
            "winning variant."
        )

    st.markdown(message)

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 4 — Evidence Summary
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Evidence</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Evidence Summary</div>', unsafe_allow_html=True)

    evidence = pd.DataFrame(
        {
            "Metric": [
                "P(B Wins)",
                "Conversion Test",
                "CTR Test",
                "Revenue Test",
                "Early Stopping",
            ],
            "Value": [
                f"{prob_b:.2%}",
                f"{p_conv:.2e}",
                f"{p_ctr:.2e}",
                f"{p_rev:.2e}",
                stop_display,
            ],
            "Status": [
                "Pass" if prob_b > 0.95 else "Review",
                "Significant" if p_conv < 0.05 else "Not Significant",
                "Significant" if p_ctr < 0.05 else "Not Significant",
                "Significant" if p_rev < 0.05 else "Not Significant",
                "Triggered" if stop_reached else "Not Triggered",
            ],
        }
    )

    st.dataframe(
        evidence,
        width="stretch",
        hide_index=True,
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 5 — Risk Assessment
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Risk Assessment</div>', unsafe_allow_html=True)

    if risk == "Low":
        risk_text = (
            "Current evidence suggests minimal deployment risk. The "
            "probability of selecting the wrong variant is very small "
            "under the observed data."
        )
    elif risk == "Medium":
        risk_text = (
            "The experiment favors Variant B, though some uncertainty "
            "remains. Additional traffic may further strengthen "
            "confidence."
        )
    else:
        risk_text = (
            "The experiment currently exhibits elevated uncertainty. "
            "More observations should be collected before making a "
            "decision."
        )

    render_html(f'<div class="info-card">{risk_text}</div>')

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 6 — Executive Summary
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Executive Summary</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.metric(label="Probability B Wins", value=f"{prob_b:.2%}")
        st.metric(label="Users Saved", value=f"{saved:,}")
        st.metric(label="Early Stop Point", value=stop_display)

    with c2:
        st.metric(label="Conversion P-Value", value=f"{p_conv:.2e}")
        st.metric(label="CTR P-Value", value=f"{p_ctr:.2e}")
        st.metric(label="Revenue P-Value", value=f"{p_rev:.2e}")

    st.write("")

    # Status badges, rendered as block-level cards rather than inline
    # <span> pills, which Streamlit can fail to sanitize correctly.
    render_html(
        f"""
        <div class="badge-row">
            <div class="badge-card {risk_css_class}">
                <div class="badge-label">Recommendation</div>
                <div class="badge-value">{recommendation}</div>
            </div>
            <div class="badge-card {risk_css_class}">
                <div class="badge-label">Risk</div>
                <div class="badge-value">{risk}</div>
            </div>
        </div>
        """
    )

    st.write("")

    stop_sentence = (
        f"The experiment reached the stopping threshold after "
        f"**{stop_point:.0f}%** of traffic, saving approximately "
        f"**{saved:,}** user exposures."
        if stop_reached
        else (
            f"The experiment has not yet reached the stopping threshold, "
            f"though approximately **{saved:,}** user exposures have been "
            f"observed so far."
        )
    )

    st.markdown(
        f"""
        Variant B demonstrates a **{prob_b:.2%}** probability of
        outperforming Variant A.

        Statistical tests indicate {"significant" if (p_conv < 0.05 and p_ctr < 0.05 and p_rev < 0.05) else "mixed"} differences
        across conversion, CTR, and revenue metrics.

        {stop_sentence}
        """
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 7 — Final Decision
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Conclusion</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Final Decision</div>', unsafe_allow_html=True)

    final_text = (
        f"Across Bayesian inference, frequentist testing, and early "
        f"stopping analysis, the experiment consistently favors "
        f"<strong>Variant B</strong>.<br><br>"
        f"Recommendation: <strong>{recommendation}</strong><br>"
        f"Estimated risk level: <strong>{risk}</strong><br>"
        f"Probability that Variant B wins: <strong>{prob_b:.2%}</strong>"
    )

    render_html(f'<div class="info-card">{final_text}</div>')

# ──────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-note">This concludes the experiment lifecycle '
    'from exploration to deployment recommendation.</div>',
    unsafe_allow_html=True,
)