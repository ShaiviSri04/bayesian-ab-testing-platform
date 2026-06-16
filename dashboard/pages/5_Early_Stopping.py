import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils import load_data
from src.bayesian import probability_b_wins

from src.early_stopping import (
    confidence_curve,
    find_early_stop,
    users_saved,
)

st.set_page_config(
    page_title="Early Stopping",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(html: str) -> None:
    """Render an HTML block, stripping per-line leading whitespace so
    markdown does not mistake indented HTML for a code block."""
    lines = [line.strip() for line in html.strip().splitlines()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Global styling (matches app.py / Overview / EDA / Frequentist / Bayesian)
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

    /* ── Dataframe container ── */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        overflow: hidden;
    }

    /* ── Info cards (no green/red banners) ── */
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
# Sidebar branding
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand-title">A/B Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-brand-sub">Experiment Analytics Engine</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# Data & calculations
# ──────────────────────────────────────────────────────────────────────────
df = load_data()

curve = confidence_curve(df)
early_stop = find_early_stop(df)
saved_users = users_saved(df)
final_prob = probability_b_wins(df)

if final_prob >= 0.95:
    decision = "Deploy Variant B"
elif final_prob >= 0.80:
    decision = "Continue Experiment"
else:
    decision = "Insufficient Evidence"

stop_reached = early_stop is not None

# ──────────────────────────────────────────────────────────────────────────
# SECTION 0 — Hero Header
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Early Stopping</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Determine when experiments can safely stop '
    'while maintaining statistical confidence.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<div class="card-kicker">Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Why Early Stopping?</div>', unsafe_allow_html=True)
    render_html(
        """
        <div class="content-card-body">
            Traditional experiments often continue until all traffic is
            exhausted. Bayesian early stopping allows teams to terminate
            experiments sooner when sufficient evidence has accumulated.
            This reduces experimentation costs, speeds up product
            decisions, and minimizes exposure to inferior experiences.
        </div>
        """
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 1 — KPI Cards
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Stopping Metrics</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    stop_value = f"{early_stop:.0f}%" if stop_reached else "Not Reached"
    st.metric(label="Early Stop Point", value=stop_value)
    pill_class = "pill-active" if stop_reached else "pill-muted"
    pill_text = "Threshold crossed" if stop_reached else "Still accumulating"
    render_html(f'<div class="metric-foot"><span class="pill {pill_class}">{pill_text}</span></div>')

with k2:
    st.metric(label="Users Saved", value=f"{saved_users:,}")
    render_html(
        '<div class="metric-foot"><span class="pill pill-soft">'
        'Reduced exposure</span></div>'
    )

with k3:
    st.metric(label="Final P(B Wins)", value=f"{final_prob:.2%}")
    if final_prob >= 0.95:
        prob_class, prob_text = "pill-active", "Strong evidence"
    elif final_prob >= 0.80:
        prob_class, prob_text = "pill-soft", "Promising"
    else:
        prob_class, prob_text = "pill-muted", "Inconclusive"
    render_html(f'<div class="metric-foot"><span class="pill {prob_class}">{prob_text}</span></div>')

with k4:
    st.metric(label="Decision", value=decision)
    if decision == "Deploy Variant B":
        dec_class = "pill-active"
    elif decision == "Continue Experiment":
        dec_class = "pill-soft"
    else:
        dec_class = "pill-muted"
    render_html(f'<div class="metric-foot"><span class="pill {dec_class}">Based on final P(B Wins)</span></div>')

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 2 — Confidence Curve
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Confidence Curve</div>', unsafe_allow_html=True)
    render_html(
        """
        <div class="content-card-sub">
            The chart below illustrates how confidence in Variant B evolves
            as additional traffic enters the experiment.
        </div>
        """
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=curve["traffic_pct"],
            y=curve["prob_b_better"] * 100,
            mode="lines",
            name="P(B Wins)",
            line=dict(color="#A78BFA", width=3),
            fill="tozeroy",
            fillcolor="rgba(167,139,250,0.08)",
        )
    )

    fig.add_hline(
        y=95,
        line_dash="dash",
        line_color="#C4B5FD",
        line_width=1.5,
        annotation_text="95% threshold",
        annotation_font_color="#C4B5FD",
        annotation_position="top left",
    )

    if stop_reached:
        fig.add_vline(
            x=early_stop,
            line_dash="dot",
            line_color="#71717A",
            line_width=1.5,
        )

    fig.update_layout(
        paper_bgcolor="#181C25",
        plot_bgcolor="#181C25",
        font=dict(family="Inter, sans-serif", color="#A1A1AA"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=380,
        xaxis=dict(
            title="Traffic (%)",
            color="#A1A1AA",
            gridcolor="#1F2330",
            zerolinecolor="#1F2330",
        ),
        yaxis=dict(
            title="P(B Wins) (%)",
            color="#A1A1AA",
            gridcolor="#1F2330",
            zerolinecolor="#1F2330",
            range=[0, 100],
        ),
        showlegend=False,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    if stop_reached:
        st.caption(f"Confidence threshold crossed at {early_stop:.0f}% of traffic.")
    else:
        st.caption("The confidence threshold was not reached.")

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 3 — Confidence Progression Table
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Confidence Progression</div>', unsafe_allow_html=True)
    render_html(
        """
        <div class="content-card-sub">
            Probability that Variant B wins at each traffic checkpoint.
        </div>
        """
    )

    display_df = pd.DataFrame(
        {
            "Traffic": curve["traffic_pct"].map(lambda x: f"{x:.0f}%"),
            "P(B Wins)": curve["prob_b_better"].map(lambda x: f"{x:.2%}"),
        }
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 4 — Interpretation
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Interpretation</div>', unsafe_allow_html=True)

    if stop_reached:
        interpretation_text = (
            f"The experiment reached the required confidence threshold "
            f"after observing approximately <strong>{early_stop:.0f}%</strong> "
            f"of total traffic. By stopping early, the organization could "
            f"save approximately <strong>{saved_users:,} user exposures</strong> "
            f"while accelerating deployment decisions."
        )
    else:
        interpretation_text = (
            "The experiment did not reach the required confidence "
            "threshold. Additional data collection may be necessary "
            "before making deployment decisions."
        )

    render_html(f'<div class="info-card">{interpretation_text}</div>')

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 5 — Business Takeaway
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Decision</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Business Takeaway</div>', unsafe_allow_html=True)

    if stop_reached:
        takeaway_text = (
            f"Bayesian monitoring suggests that the experiment could have "
            f"been safely stopped at <strong>{early_stop:.0f}%</strong> of "
            f"traffic while maintaining high confidence. Early stopping "
            f"enables faster product iteration, reduces experimentation "
            f"costs, and improves operational efficiency."
        )
    else:
        takeaway_text = (
            "Current evidence suggests continuing the experiment until "
            "uncertainty is further reduced. Bayesian monitoring enables "
            "organizations to reduce experimentation costs while "
            "preserving decision quality."
        )

    render_html(f'<div class="info-card">{takeaway_text}</div>')

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# SECTION 6 — Experiment Conclusion
# ──────────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="card-kicker">Conclusion</div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card-title">Experiment Conclusion</div>', unsafe_allow_html=True)

    if stop_reached:
        conclusion_text = (
            f"The experiment could have been safely stopped at "
            f"<strong>{early_stop:.0f}%</strong> of traffic while "
            f"maintaining high confidence in Variant B. The final "
            f"probability that Variant B outperforms Variant A is "
            f"<strong>{final_prob:.2%}</strong>. This supports deploying "
            f"Variant B with relatively low decision risk."
        )
    else:
        conclusion_text = (
            f"The experiment has not yet accumulated sufficient evidence "
            f"for early stopping. The final probability that Variant B "
            f"outperforms Variant A is <strong>{final_prob:.2%}</strong>. "
            f"Additional traffic is recommended before making deployment "
            f"decisions."
        )

    render_html(f'<div class="info-card">{conclusion_text}</div>')

# ──────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-note">Proceed to the Decision Engine for final '
    'deployment recommendations.</div>',
    unsafe_allow_html=True,
)