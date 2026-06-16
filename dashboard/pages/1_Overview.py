import streamlit as st

from src.utils import load_data
from src.bayesian import probability_b_wins
from src.eda import conversion_rates, get_revenue_metrics

st.set_page_config(
    page_title="Overview",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# Global styling (matches app.py)
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

    /* Page header */
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

    /* Section labels */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #A1A1AA;
        margin-bottom: 0.9rem;
        margin-top: 0.25rem;
    }

    /* KPI metric cards */
    div[data-testid="stMetric"] {
        background-color: #181C25;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        transition: border-color 0.18s ease, transform 0.18s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(167,139,250,0.25);
        transform: translateY(-1px);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #A1A1AA;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F5F3FF;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.85rem;
        color: #C4B5FD !important;
    }

    /* Bordered containers (st.container(border=True)) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #181C25;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 18px !important;
        padding: 0.5rem 0.4rem;
        transition: border-color 0.18s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(167,139,250,0.25);
    }

    .content-card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #F5F3FF;
        margin-bottom: 0.5rem;
    }
    .content-card-sub {
        font-size: 0.85rem;
        color: #A1A1AA;
        margin-bottom: 0.75rem;
    }

    /* Info cards (replace success/warning/error) */
    .info-card {
        background-color: #1B1830;
        border: 1px solid rgba(167,139,250,0.15);
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        font-size: 0.95rem;
        color: #D4D4D8;
        line-height: 1.7;
    }

    /* Pill / chip */
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.25rem;
    }
    .pill {
        display: inline-block;
        background-color: #221D34;
        color: #C4B5FD;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        border: 1px solid rgba(167,139,250,0.15);
        white-space: nowrap;
    }

    /* Streamlit native chart tweaks */
    [data-testid="stArrowVegaLiteChart"] {
        background-color: transparent;
    }

    /* Footer */
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
# Data & calculations
# ──────────────────────────────────────────────────────────────────────────
df = load_data()

users = len(df)

prob_b = probability_b_wins(df)

winner = "Variant B" if prob_b > 0.5 else "Variant A"

revenue = get_revenue_metrics(df)

revenue_a = revenue.loc[revenue["variant"] == "A", "experiment_revenue"].iloc[0]
revenue_b = revenue.loc[revenue["variant"] == "B", "experiment_revenue"].iloc[0]

revenue_lift = revenue_b - revenue_a
revenue_pct = (
    revenue_lift / revenue_a
) * 100

conversion = conversion_rates(df)

# ──────────────────────────────────────────────────────────────────────────
# SECTION 1 — Header
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Overview</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">High-level summary of experiment performance and business impact.</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# SECTION 2 — KPI Cards
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(label="Users", value=f"{users:,}")

with kpi2:
    st.metric(label="P(B Wins)", value=f"{prob_b:.1%}")

with kpi3:
    st.metric(label="Winner", value=winner)

with kpi4:
    st.metric(
        label="Revenue/User Lift",
        value=f"+{revenue_pct:.2f}%"
    )
    

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 3 — Charts Row
# ──────────────────────────────────────────────────────────────────────────
chart_left, chart_right = st.columns(2, gap="medium")

with chart_left:
    with st.container(border=True):
        st.markdown('<div class="content-card-title">Conversion Rate by Variant</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="content-card-sub">Share of users converting, per variant.</div>',
            unsafe_allow_html=True,
        )
        conv_chart = conversion.set_index("variant")
        st.bar_chart(
            conv_chart["converted"],
            color="#A78BFA",
            use_container_width=True,
        )

with chart_right:
    with st.container(border=True):
        st.markdown('<div class="content-card-title">Revenue per User</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="content-card-sub">Average revenue generated per user, per variant.</div>',
            unsafe_allow_html=True,
        )
        rev_chart = revenue.set_index("variant")
        st.bar_chart(
            rev_chart["experiment_revenue"],
            color="#A78BFA",
            use_container_width=True,
        )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 4 — Experiment Health
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="content-card-title">Experiment Health</div>', unsafe_allow_html=True)

    if prob_b > 0.95:
        health_message = "Variant B shows strong evidence of outperforming Variant A."
    elif prob_b > 0.80:
        health_message = "The experiment is promising but should continue collecting more data."
    else:
        health_message = "Current evidence is insufficient for deployment decisions."

    st.markdown(f'<div class="info-card">{health_message}</div>', unsafe_allow_html=True)

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 5 — Key Insights
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Key Insights</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="pill-row">
        <span class="pill">High Confidence</span>
        <span class="pill">Revenue Positive</span>
        <span class="pill">Experiment Stable</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-note">Explore detailed analysis using the sidebar modules.</div>',
    unsafe_allow_html=True,
)