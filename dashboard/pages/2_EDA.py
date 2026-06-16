import streamlit as st
import matplotlib.pyplot as plt

from src.utils import load_data

from src.eda import (
    get_variant_distribution,
    conversion_rates,
    get_ctr_rates,
    get_revenue_metrics,
    get_device_conversion,
    get_segment_conversion,
)

st.set_page_config(
    page_title="EDA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# Global styling (matches app.py / 1_Overview.py)
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

    /* Insight / info cards (replace success/warning/error) */
    .info-card {
        background-color: #1B1830;
        border: 1px solid rgba(167,139,250,0.15);
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        font-size: 0.95rem;
        color: #D4D4D8;
        line-height: 1.85;
    }
    .info-card strong {
        color: #C4B5FD;
        font-weight: 600;
    }

    .insight-note {
        font-size: 0.88rem;
        color: #A1A1AA;
        margin-top: 0.85rem;
        padding-top: 0.85rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    .insight-note strong {
        color: #C4B5FD;
        font-weight: 600;
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
# Data
# ──────────────────────────────────────────────────────────────────────────
df = load_data()

traffic = get_variant_distribution(df)
conversion = conversion_rates(df)
ctr = get_ctr_rates(df)
revenue = get_revenue_metrics(df)

device = get_device_conversion(df)
segment = get_segment_conversion(df)

# ──────────────────────────────────────────────────────────────────────────
# Calculations
# ──────────────────────────────────────────────────────────────────────────
traffic_a = traffic.iloc[0, 1]
traffic_b = traffic.iloc[1, 1]

conv_a = conversion.loc[conversion["variant"] == "A", "converted"].iloc[0]
conv_b = conversion.loc[conversion["variant"] == "B", "converted"].iloc[0]
conversion_uplift = ((conv_b - conv_a) / conv_a) * 100

ctr_a = ctr.loc[ctr["variant"] == "A", "clicked"].iloc[0]
ctr_b = ctr.loc[ctr["variant"] == "B", "clicked"].iloc[0]
ctr_uplift = ((ctr_b - ctr_a) / ctr_a) * 100

rev_a = revenue.loc[revenue["variant"] == "A", "experiment_revenue"].iloc[0]
rev_b = revenue.loc[revenue["variant"] == "B", "experiment_revenue"].iloc[0]
revenue_uplift = ((rev_b - rev_a) / rev_a) * 100

# Dynamic device-level insight: which device shows the strongest B uplift
device_pivot = device.pivot(
    index="device_type", columns="variant", values="converted"
)
device_uplift = device_pivot["B"] - device_pivot["A"]
top_device = device_uplift.idxmax()

# Dynamic segment-level insight: which segment shows the strongest B uplift
segment_pivot = segment.pivot(
    index="customer_segment", columns="variant", values="converted"
)
segment_uplift = segment_pivot["B"] - segment_pivot["A"]
top_segment = segment_uplift.idxmax()

# ──────────────────────────────────────────────────────────────────────────
# SECTION 0 — Header
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Understand experiment traffic, user behavior, and business impact.</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# SECTION 1 — Executive Summary
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="content-card-title">Key Insights</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="info-card">
            &bull; Variant <strong>B</strong> improves conversion by <strong>{conversion_uplift:.1f}%</strong><br>
            &bull; Click-through rate increased by <strong>{ctr_uplift:.1f}%</strong><br>
            &bull; Revenue per user improved by <strong>{revenue_uplift:.1f}%</strong><br>
            &bull; Experiment results indicate strong business impact for Variant B
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 2 — KPI Row
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        label="Traffic Split",
        value=f"{traffic_a * 100:.0f}% / {traffic_b * 100:.0f}%",
    )

with k2:
    st.metric(label="Conversion Lift", value=f"+{conversion_uplift:.1f}%")

with k3:
    st.metric(label="CTR Lift", value=f"+{ctr_uplift:.1f}%")

with k4:
    st.metric(label="Revenue Lift", value=f"+{revenue_uplift:.1f}%")

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 3 — Traffic Allocation & Conversion
# ──────────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2, gap="medium")

with c1:
    with st.container(border=True):
        st.markdown('<div class="content-card-title">Traffic Allocation</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="content-card-sub">Share of total users assigned to each variant.</div>',
            unsafe_allow_html=True,
        )
        st.bar_chart(
            traffic.set_index("variant"),
            color="#A78BFA",
            use_container_width=True,
        )

with c2:
    with st.container(border=True):
        st.markdown('<div class="content-card-title">Conversion Rate</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="content-card-sub">Proportion of users completing the target action.</div>',
            unsafe_allow_html=True,
        )
        st.bar_chart(
            conversion.set_index("variant"),
            color="#A78BFA",
            use_container_width=True,
        )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 4 — CTR & Revenue
# ──────────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2, gap="medium")

with c1:
    with st.container(border=True):
        st.markdown('<div class="content-card-title">Click Through Rate</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="content-card-sub">Engagement rate per variant.</div>',
            unsafe_allow_html=True,
        )
        st.bar_chart(
            ctr.set_index("variant"),
            color="#A78BFA",
            use_container_width=True,
        )

with c2:
    with st.container(border=True):
        st.markdown('<div class="content-card-title">Revenue per User</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="content-card-sub">Average revenue generated per user.</div>',
            unsafe_allow_html=True,
        )
        st.bar_chart(
            revenue.set_index("variant"),
            color="#A78BFA",
            use_container_width=True,
        )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 5 — Device Analysis
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="content-card-title">Device Conversion Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="content-card-sub">Conversion rate by device type, split by variant.</div>',
        unsafe_allow_html=True,
    )

    st.bar_chart(
        device_pivot,
        color=["#3F3A52", "#A78BFA"],
        use_container_width=True,
    )

    st.markdown(
        f"""
        <div class="insight-note">
            <strong>{top_device}</strong> users show the strongest response to Variant B.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 6 — Customer Segment Analysis
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="content-card-title">Customer Segment Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="content-card-sub">Conversion rate by customer segment, split by variant.</div>',
        unsafe_allow_html=True,
    )

    st.bar_chart(
        segment_pivot,
        color=["#3F3A52", "#A78BFA"],
        use_container_width=True,
    )

    st.markdown(
        f"""
        <div class="insight-note">
            The <strong>{top_segment}</strong> segment exhibits the highest conversion uplift for Variant B.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 7 — Revenue Distribution
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="content-card-title">Revenue Distribution</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="content-card-sub">Distribution of per-user experiment revenue.</div>',
        unsafe_allow_html=True,
    )

    revenue_clip = df[
        df["experiment_revenue"] < df["experiment_revenue"].quantile(0.99)
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#181C25")
    ax.set_facecolor("#181C25")

    ax.hist(
        revenue_clip["experiment_revenue"],
        bins=40,
        color="#A78BFA",
        edgecolor="#0F1117",
    )

    ax.set_xlabel("Revenue", color="#A1A1AA")
    ax.set_ylabel("Frequency", color="#A1A1AA")

    ax.tick_params(colors="#A1A1AA")

    for spine in ax.spines.values():
        spine.set_color("#2A2E3A")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="y", color="#2A2E3A", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    st.pyplot(fig)

    st.caption("Revenue distribution after removing extreme outliers.")

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 8 — Experiment Takeaway
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="content-card-title">Experiment Takeaway</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="info-card">
            Variant B consistently outperforms Variant A across conversion
            (<strong>+{conversion_uplift:.1f}%</strong>), engagement
            (<strong>+{ctr_uplift:.1f}%</strong>), and revenue
            (<strong>+{revenue_uplift:.1f}%</strong>) metrics, suggesting strong
            evidence for deployment.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-note">Continue to Frequentist and Bayesian modules for statistical validation.</div>',
    unsafe_allow_html=True,
)