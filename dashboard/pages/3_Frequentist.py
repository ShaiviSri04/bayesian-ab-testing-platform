import numpy as np
import pandas as pd
import altair as alt
import streamlit as st


def render_html(html: str) -> None:
    """Render an HTML block, stripping per-line leading whitespace so
    markdown does not mistake indented HTML for a code block."""
    lines = [line.strip() for line in html.strip().splitlines()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)

from src.utils import load_data
from src.frequentist import (
    calculate_conversion_test,
    calculate_ctr_test,
    calculate_revenue_test,
)

st.set_page_config(
    page_title="Frequentist Testing",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# Global styling (matches app.py / Overview / EDA)
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
    /* Fix overflow / truncation on longer values like "Deploy Variant B" */
    div[data-testid="stMetricValue"] {
        font-size: 1.45rem;
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

    /* ── Decision pill row under metrics ── */
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
    .pill-muted {
        background-color: transparent;
        color: #71717A;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Custom test-summary table ── */
    .freq-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .freq-table thead th {
        text-align: left;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #71717A;
        padding: 0 0.75rem 0.7rem 0.75rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .freq-table tbody td {
        padding: 0.85rem 0.75rem;
        color: #D4D4D8;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-weight: 500;
    }
    .freq-table tbody tr:last-child td {
        border-bottom: none;
    }
    .freq-table tbody tr:hover {
        background-color: rgba(167,139,250,0.04);
    }
    .freq-table td.metric-name {
        font-weight: 600;
        color: #F5F3FF;
    }
    .freq-table td.mono {
        font-family: 'Inter', sans-serif;
        color: #A1A1AA;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    /* ── Interpretation blocks ── */
    .interp-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.85rem;
    }
    .interp-block {
        background-color: #14121C;
        border: 1px solid rgba(255,255,255,0.04);
        border-left: 3px solid #A78BFA;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        height: 100%;
    }
    .interp-title {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #C4B5FD;
        margin-bottom: 0.5rem;
    }
    .interp-text {
        font-size: 0.88rem;
        color: #A1A1AA;
        line-height: 1.7;
    }

    /* ── Info / takeaway cards ── */
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
# Data & tests
# ──────────────────────────────────────────────────────────────────────────
df = load_data()

z_conv, p_conv = calculate_conversion_test(df)
z_ctr, p_ctr = calculate_ctr_test(df)
t_rev, p_rev = calculate_revenue_test(df)

ALPHA = 0.05

decision = "Deploy Variant B" if p_conv < ALPHA else "Continue Testing"

# ──────────────────────────────────────────────────────────────────────────
# SECTION 0 — Header
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Frequentist Testing</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Evaluate whether experiment results are statistically significant.</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# SECTION 1 — Hero Information Card
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">What is Frequentist Testing?</div>', unsafe_allow_html=True)
    render_html(
        """
        <div class="content-card-body">
            Frequentist testing evaluates whether observed differences between
            experiment variants are likely due to real effects or random chance.
            Smaller p-values provide stronger evidence that the measured
            improvements are statistically meaningful rather than random
            fluctuations in the data.
        </div>
        """
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 2 — KPI Cards
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Test Results</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(label="Conversion P-Value", value=f"{p_conv:.2e}")
    sig_class = "pill-active" if p_conv < ALPHA else "pill-muted"
    sig_text = "Significant" if p_conv < ALPHA else "Not significant"
    st.markdown(
        f'<div class="metric-foot"><span class="pill {sig_class}">{sig_text}</span></div>',
        unsafe_allow_html=True,
    )

with k2:
    st.metric(label="CTR P-Value", value=f"{p_ctr:.2e}")
    sig_class = "pill-active" if p_ctr < ALPHA else "pill-muted"
    sig_text = "Significant" if p_ctr < ALPHA else "Not significant"
    st.markdown(
        f'<div class="metric-foot"><span class="pill {sig_class}">{sig_text}</span></div>',
        unsafe_allow_html=True,
    )

with k3:
    st.metric(label="Revenue P-Value", value=f"{p_rev:.2e}")
    sig_class = "pill-active" if p_rev < ALPHA else "pill-muted"
    sig_text = "Significant" if p_rev < ALPHA else "Not significant"
    st.markdown(
        f'<div class="metric-foot"><span class="pill {sig_class}">{sig_text}</span></div>',
        unsafe_allow_html=True,
    )

with k4:
    st.metric(label="Decision", value=decision)
    basis_class = "pill-active" if p_conv < ALPHA else "pill-muted"
    st.markdown(
        f'<div class="metric-foot"><span class="pill {basis_class}">Based on conversion test</span></div>',
        unsafe_allow_html=True,
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 3 — Test Summary
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Test Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="content-card-sub">Statistical results for each evaluated metric.</div>',
        unsafe_allow_html=True,
    )

    rows = [
        ("Conversion", z_conv, p_conv),
        ("CTR", z_ctr, p_ctr),
        ("Revenue", t_rev, p_rev),
    ]

    table_rows = ""
    for name, stat, pval in rows:
        is_sig = pval < ALPHA
        pill_class = "pill-active" if is_sig else "pill-muted"
        pill_text = "Yes" if is_sig else "No"
        table_rows += (
            "<tr>"
            f'<td class="metric-name">{name}</td>'
            f'<td class="mono">{stat:.3f}</td>'
            f'<td class="mono">{pval:.2e}</td>'
            f'<td><span class="pill {pill_class}">{pill_text}</span></td>'
            "</tr>"
        )

    render_html(
        f"""
        <table class="freq-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Statistic</th>
                    <th>P-Value</th>
                    <th>Significant</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 4 — Interpretation
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Interpretation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="content-card-sub">Plain-language explanations of each test result.</div>',
        unsafe_allow_html=True,
    )

    if p_conv < ALPHA:
        conv_text = (
            "The conversion test produces a p-value below 0.05, indicating "
            "that the observed improvement in Variant B is unlikely to have "
            "occurred by random chance."
        )
    else:
        conv_text = (
            "The conversion difference is not statistically significant, "
            "suggesting that additional data may be required before drawing "
            "conclusions."
        )

    if p_ctr < ALPHA:
        ctr_text = (
            "User engagement differs significantly between variants, "
            "providing evidence that the new experience meaningfully "
            "affects user behavior."
        )
    else:
        ctr_text = (
            "Differences in click-through rate may be explained by normal "
            "sampling variation rather than a true effect."
        )

    if p_rev < ALPHA:
        rev_text = (
            "Revenue differences between variants are statistically "
            "significant, indicating a measurable impact on business "
            "outcomes."
        )
    else:
        rev_text = (
            "Revenue improvements are not statistically significant at the "
            "chosen threshold, and further testing is recommended."
        )

    render_html(
        f"""
        <div class="interp-grid">
            <div class="interp-block">
                <div class="interp-title">Conversion Rate</div>
                <div class="interp-text">{conv_text}</div>
            </div>
            <div class="interp-block">
                <div class="interp-title">Click Through Rate</div>
                <div class="interp-text">{ctr_text}</div>
            </div>
            <div class="interp-block">
                <div class="interp-title">Revenue Impact</div>
                <div class="interp-text">{rev_text}</div>
            </div>
        </div>
        """
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 5 — P-Value Visualization
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Significance Threshold</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="content-card-sub">'
        "Larger bars indicate stronger statistical significance. "
        "The dashed line marks the &alpha; = 0.05 threshold."
        "</div>",
        unsafe_allow_html=True,
    )

    sig_df = pd.DataFrame(
        {
            "Metric": ["Conversion", "CTR", "Revenue"],
            "neg_log_p": [
                -np.log10(p_conv),
                -np.log10(p_ctr),
                -np.log10(p_rev),
            ],
        }
    )

    threshold = -np.log10(ALPHA)

    bars = (
        alt.Chart(sig_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=64)
        .encode(
            x=alt.X(
                "Metric:N",
                title=None,
                axis=alt.Axis(
                    labelColor="#A1A1AA",
                    labelFontSize=12,
                    domainColor="#2A2E3A",
                    tickColor="#2A2E3A",
                ),
            ),
            y=alt.Y(
                "neg_log_p:Q",
                title="-log10(p-value)",
                axis=alt.Axis(
                    labelColor="#A1A1AA",
                    titleColor="#A1A1AA",
                    domainColor="#2A2E3A",
                    tickColor="#2A2E3A",
                    gridColor="#1F2330",
                ),
            ),
            color=alt.condition(
                alt.datum.neg_log_p > threshold,
                alt.value("#A78BFA"),
                alt.value("#3F3A52"),
            ),
        )
    )

    rule = (
        alt.Chart(pd.DataFrame({"y": [threshold]}))
        .mark_rule(color="#C4B5FD", strokeDash=[5, 5], size=1.5)
        .encode(y="y:Q")
    )

    chart = (bars + rule).properties(height=300, background="transparent")

    st.altair_chart(chart, use_container_width=True)

    st.caption("All p-values below \u03b1 = 0.05 indicate statistical significance.")

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 6 — Business Takeaway
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Business Takeaway</div>', unsafe_allow_html=True)

    all_significant = (p_conv < ALPHA) and (p_ctr < ALPHA) and (p_rev < ALPHA)

    if all_significant:
        takeaway_text = (
            "Variant B demonstrates statistically significant improvements "
            "across conversion, engagement, and revenue metrics. The "
            "evidence strongly supports deployment of Variant B in "
            "production."
        )
    else:
        takeaway_text = (
            "The experiment does not yet provide sufficient statistical "
            "evidence across all metrics for a confident deployment "
            "decision. Additional data collection is recommended before "
            "proceeding."
        )

    st.markdown(f'<div class="info-card">{takeaway_text}</div>', unsafe_allow_html=True)

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 7 — Final Conclusion
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Conclusion</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Experiment Conclusion</div>', unsafe_allow_html=True)

    if all_significant:
        conclusion_tail = (
            "These results indicate strong evidence that Variant B "
            "outperforms Variant A and that the observed improvements are "
            "statistically reliable."
        )
    else:
        conclusion_tail = (
            "These results do not yet provide consistent evidence across "
            "all metrics, and continued experimentation is recommended "
            "before making a deployment decision."
        )

    render_html(
        f"""
        <div class="info-card">
            Using an alpha threshold of <strong>{ALPHA}</strong>, the
            experiment produced a conversion p-value of
            <strong>{p_conv:.2e}</strong>, a CTR p-value of
            <strong>{p_ctr:.2e}</strong>, and a revenue p-value of
            <strong>{p_rev:.2e}</strong>. {conclusion_tail}
        </div>
        """
    )

# ──────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-note">Proceed to Bayesian analysis for probability-based decision making.</div>',
    unsafe_allow_html=True,
)