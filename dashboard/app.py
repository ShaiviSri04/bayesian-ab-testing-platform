import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Bayesian A/B Testing Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# Global styling
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* App background */
    .stApp {
        background-color: #0F1117;
        color: #F5F3FF;
    }

    .main .block-container {
        max-width: 1150px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
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

    /* Hero */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #F5F3FF;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #A1A1AA;
        font-weight: 400;
        margin-bottom: 1.75rem;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(167,139,250,0.10), rgba(167,139,250,0.02));
        border: 1px solid rgba(167,139,250,0.18);
        border-radius: 18px;
        padding: 1.5rem 1.75rem;
        backdrop-filter: blur(6px);
        margin-bottom: 2.25rem;
    }
    .hero-card p {
        font-size: 0.98rem;
        color: #D4D4D8;
        line-height: 1.75;
        margin: 0;
    }

    /* Section headings */
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

    /* Generic content card */
    .content-card {
        background-color: #181C25;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 18px;
        padding: 1.6rem 1.75rem;
        height: 100%;
        transition: border-color 0.18s ease;
    }
    .content-card:hover {
        border-color: rgba(167,139,250,0.25);
    }
    .content-card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #F5F3FF;
        margin-bottom: 0.75rem;
    }
    .content-card p {
        font-size: 0.93rem;
        color: #A1A1AA;
        line-height: 1.75;
        margin: 0;
    }

    /* Pill / chip */
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
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
    .pill.muted {
        background-color: #181C25;
        color: #A1A1AA;
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* Footer */
    .footer-note {
        font-size: 0.85rem;
        color: #A1A1AA;
        text-align: center;
        padding-top: 1.5rem;
        margin-top: 1rem;
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
# SECTION 1 — Hero Header
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Bayesian A/B Testing Analytics Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Experiment intelligence for data-driven decisions.</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-card">
        <p>
            Compare experiment variants, measure business impact, and identify winning
            experiences with confidence. This platform combines statistical testing,
            Bayesian inference, and early stopping to enable faster and more reliable
            decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# SECTION 2 — KPI Cards
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(label="Users", value="99992")
with kpi2:
    st.metric(label="P(B Wins)", value="100%")
with kpi3:
    st.metric(label="Revenue Lift", value="+₹8.3L")
with kpi4:
    st.metric(label="Recommendation", value="Deploy B")

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 3 — Dual Containers (Overview & Modules)
# ──────────────────────────────────────────────────────────────────────────
left, right = st.columns(2, gap="medium")

with left:
    st.markdown(
        """
        <div class="content-card">
            <div class="content-card-title">Overview</div>
            <p>
                This platform helps teams compare experiment variants, understand their
                impact on user behavior, and make confident deployment decisions using
                data-driven insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class="content-card">
            <div class="content-card-title">Modules</div>
            <div class="pill-row">
                <span class="pill">EDA</span>
                <span class="pill">Frequentist</span>
                <span class="pill">Bayesian</span>
                <span class="pill">Early Stopping</span>
                <span class="pill">Decision Engine</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 4 — Experiment Summary Card
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="content-card">
        <div class="content-card-title">Experiment Summary</div>
        <p>
            Variant B outperforms Variant A with strong statistical evidence and
            substantial business impact, making it the recommended choice for
            deployment.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 5 — Analysis Workflow
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Analysis Workflow</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="pill-row">
        <span class="pill muted">Raw Data</span>
        <span class="pill muted">EDA</span>
        <span class="pill muted">Frequentist</span>
        <span class="pill muted">Bayesian</span>
        <span class="pill muted">Early Stop</span>
        <span class="pill muted">Decision</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-note">Navigate using the sidebar to explore individual analysis modules.</div>',
    unsafe_allow_html=True,
)
