"""
SwipeIQ Streamlit Dashboard — Theme & Styling Utilities
Custom CSS injection and color constants matching the HTML dashboard tokens.
"""
import streamlit as st

# ── Color Constants (matching dashboard.html design tokens) ──
PURPLE = "#8b5cf6"
PURPLE_LIGHT = "#a78bfa"
PINK = "#ec4899"
TEAL = "#14b8a6"
AMBER = "#f59e0b"
GREEN = "#10b981"
RED = "#ef4444"
SKY = "#38bdf8"
INDIGO = "#6366f1"
BG_BASE = "#0e1117"
BG_CARD = "#161b22"
TEXT_PRIMARY = "#f1f5f9"
TEXT_SECONDARY = "#94a3b8"
TEXT_MUTED = "#4b5563"

# Plotly color sequence
PLOTLY_COLORS = [PURPLE, PINK, TEAL, AMBER, GREEN, SKY, RED, INDIGO, PURPLE_LIGHT, "#f472b6"]


def get_plotly_layout(title="", height=450):
    """Return a consistent dark-themed Plotly layout dict."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=12),
        title=dict(text=title, font=dict(color=TEXT_PRIMARY, size=14)),
        height=height,
        margin=dict(l=40, r=20, t=50 if title else 20, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=TEXT_SECONDARY),
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)", 
            zeroline=False,
            showline=False,
            tickfont=dict(color=TEXT_MUTED, size=10)
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)", 
            zeroline=False,
            showline=False,
            tickfont=dict(color=TEXT_MUTED, size=10)
        ),
    )


def inject_css():
    """Inject custom CSS for premium styling beyond Streamlit defaults."""
    st.markdown("""
    <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Global font override */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: #0a0f1c;
            border-right: 1px solid rgba(255,255,255,0.07);
        }

        [data-testid="stSidebar"] .stMarkdown h1 {
            background: linear-gradient(135deg, #8b5cf6, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }

        /* Metric card styling */
        [data-testid="stMetric"] {
            background: #161b22;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            padding: 16px 20px;
            transition: border-color 0.25s, transform 0.2s;
        }

        [data-testid="stMetric"]:hover {
            border-color: rgba(139,92,246,0.4);
            transform: translateY(-2px);
        }

        [data-testid="stMetricValue"] {
            font-size: 28px !important;
            font-weight: 800 !important;
        }

        /* ── KPI CARDS (From Dashboard) ── */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .kpi-card {
            background: #161b22;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            padding: 20px 22px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            position: relative;
            overflow: hidden;
            transition: border-color 0.25s, transform 0.2s;
        }
        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: var(--accent-gradient, #8b5cf6);
        }
        .kpi-card:hover {
            border-color: rgba(139,92,246,0.4);
            transform: translateY(-2px);
        }
        .kpi-card.purple { --accent-gradient: linear-gradient(90deg, #8b5cf6, #ec4899); }
        .kpi-card.teal   { --accent-gradient: linear-gradient(90deg, #14b8a6, #38bdf8); }
        .kpi-card.green  { --accent-gradient: linear-gradient(90deg, #10b981, #14b8a6); }
        .kpi-card.amber  { --accent-gradient: linear-gradient(90deg, #f59e0b, #ec4899); }
        
        .kpi-top { display: flex; align-items: flex-start; justify-content: space-between; }
        .kpi-label { font-size: 12px; font-weight: 500; color: #94a3b8; }
        .kpi-icon {
            width: 36px; height: 36px;
            border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            font-size: 17px;
        }
        .kpi-card.purple .kpi-icon { background: rgba(139,92,246,0.15); }
        .kpi-card.teal   .kpi-icon { background: rgba(20,184,166,0.15); }
        .kpi-card.green  .kpi-icon { background: rgba(16,185,129,0.15); }
        .kpi-card.amber  .kpi-icon { background: rgba(245,158,11,0.15); }
        
        .kpi-value { font-size: 30px; font-weight: 800; letter-spacing: -1px; line-height: 1; color: #f1f5f9; }
        .kpi-footer { display: flex; align-items: center; gap: 6px; font-size: 12px; margin-top: 4px;}
        .badge-up   { color: #10b981; }
        .badge-down { color: #ef4444; }
        .kpi-sub { color: #4b5563; }

        /* ── CARD HEADERS (From Dashboard) ── */
        .dash-card {
            background: #161b22;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            padding: 20px 22px;
            margin-bottom: 16px;
        }
        .card-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            margin-bottom: 16px;
        }
        .card-title {
            font-size: 14px;
            font-weight: 700;
            color: #f1f5f9;
        }
        .card-subtitle {
            font-size: 11.5px;
            color: #94a3b8;
            margin-top: 2px;
        }
        .card-tag {
            font-size: 11px;
            font-weight: 600;
            padding: 3px 9px;
            border-radius: 20px;
            white-space: nowrap;
        }
        .tag-purple { background: rgba(139,92,246,0.15); color: #a78bfa; }
        .tag-teal   { background: rgba(20,184,166,0.15);  color: #14b8a6; }
        .tag-pink   { background: rgba(236,72,153,0.15);  color: #ec4899; }
        .tag-amber  { background: rgba(245,158,11,0.15);  color: #f59e0b; }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: rgba(255,255,255,0.02);
            border-radius: 10px;
            padding: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(139,92,246,0.15) !important;
        }

        /* Expander styling */
        [data-testid="stExpander"] {
            background: #161b22;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
        }

        /* Dataframe styling */
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
        }

        /* Custom gradient divider */
        .gradient-divider {
            height: 2px;
            background: linear-gradient(90deg, #8b5cf6, #ec4899, #14b8a6);
            border-radius: 1px;
            margin: 24px 0;
        }

        /* Section header */
        .section-label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            color: #4b5563;
            margin-bottom: 8px;
        }

        /* Pipeline step card */
        .pipeline-step {
            background: #161b22;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 16px;
            border-left: 3px solid #8b5cf6;
        }

        .pipeline-step h4 {
            color: #a78bfa;
            margin-bottom: 8px;
        }

        /* Notebook plot container */
        .notebook-plot {
            background: #161b22;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            padding: 16px;
            text-align: center;
        }

        .notebook-plot img {
            border-radius: 8px;
            max-width: 100%;
        }

        /* KPI row */
        .kpi-container {
            display: flex;
            gap: 16px;
        }

        /* Info callout */
        .ml-callout {
            background: rgba(139,92,246,0.08);
            border: 1px solid rgba(139,92,246,0.25);
            border-radius: 10px;
            padding: 16px 20px;
            color: #a78bfa;
            font-size: 14px;
            line-height: 1.6;
        }

        /* Warning callout */
        .caveat-callout {
            background: rgba(245,158,11,0.08);
            border: 1px solid rgba(245,158,11,0.25);
            border-radius: 10px;
            padding: 16px 20px;
            color: #f59e0b;
            font-size: 14px;
            line-height: 1.6;
        }

        /* Sidebar navigation stage indicator */
        .stage-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .stage-active { background: #8b5cf6; box-shadow: 0 0 8px rgba(139,92,246,0.5); }
        .stage-done { background: #10b981; }
        .stage-pending { background: #4b5563; }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the common sidebar with project info."""
    with st.sidebar:
        st.markdown("# 💘 SwipeIQ")
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        st.caption("ML-Powered Dating App Behaviour Dashboard")
        st.markdown("---")
        st.markdown("**Dataset:** Extended (50k × 25)")
        st.markdown("**Models:** 6 classifiers")
        st.markdown("**Task:** Binary classification")
        st.markdown("---")
        st.markdown(
            '<p style="font-size:11px;color:#4b5563;">WIA1006/WID3006 · Universiti Malaya<br>Sem 2, 2025/2026</p>',
            unsafe_allow_html=True,
        )
