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

        /* Hide default Streamlit developer top bar elements but keep collapse arrow */
        header {
            background-color: transparent !important;
        }
        [data-testid="stHeader"] {
            background-color: transparent !important;
            backdrop-filter: none !important;
        }
        [data-testid="stHeader"] [data-testid="stHeaderActionElements"],
        .stAppDeployButton {
            display: none !important;
            visibility: hidden !important;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Replace default "app" sidebar link text with "🏠 Homepage" */
        [data-testid="stSidebarNavItems"] li:first-child p {
            font-size: 0 !important;
            margin: 0 !important;
        }
        [data-testid="stSidebarNavItems"] li:first-child p::after {
            content: "🏠 Homepage" !important;
            font-size: 14px !important;
            display: inline-block !important;
            visibility: visible !important;
            font-weight: 500 !important;
        }

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

        /* Advanced technique card */
        .advanced-card {
            background: #161b22;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 16px;
            position: relative;
            overflow: hidden;
            transition: border-color 0.3s, transform 0.2s;
        }
        .advanced-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #8b5cf6, #ec4899, #14b8a6);
        }
        .advanced-card:hover {
            border-color: rgba(139,92,246,0.4);
            transform: translateY(-2px);
        }

        /* Flex badge */
        .flex-badge {
            display: inline-block;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            padding: 3px 10px;
            border-radius: 20px;
            margin-right: 6px;
            margin-bottom: 6px;
        }
        .flex-badge.v4 { background: rgba(236,72,153,0.15); color: #ec4899; }
        .flex-badge.v5 { background: rgba(20,184,166,0.15); color: #14b8a6; }
        .flex-badge.new { background: rgba(245,158,11,0.15); color: #f59e0b; }

        /* Hero animated gradient */
        @keyframes heroGlow {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }
        .hero-glow {
            background: linear-gradient(135deg, rgba(139,92,246,0.1), rgba(236,72,153,0.1), rgba(20,184,166,0.1));
            border: 1px solid rgba(139,92,246,0.2);
            border-radius: 16px;
            padding: 30px;
            text-align: center;
            animation: heroGlow 4s ease-in-out infinite;
        }

        /* Plot container with label */
        .plot-container {
            background: #161b22;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 16px;
        }
        .plot-container .plot-label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: #4b5563;
            margin-bottom: 12px;
        }
        .plot-container img {
            border-radius: 8px;
            width: 100%;
        }

        /* Technique description card */
        .technique-card {
            background: rgba(139,92,246,0.04);
            border: 1px solid rgba(139,92,246,0.15);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 14px;
        }
        .technique-card h4 {
            color: #a78bfa;
            margin-bottom: 8px;
            font-size: 15px;
        }
        .technique-card p {
            color: #94a3b8;
            font-size: 13px;
            line-height: 1.6;
            margin: 0;
        }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the common sidebar with project info and inject global header."""
    import os
    import base64
    import inspect
    
    # ── Resolve page title by frame stack inspection ──
    try:
        frame = inspect.currentframe().f_back
        calling_file = frame.f_code.co_filename
        # Step back if the caller is inside the theme.py helper module
        if "theme.py" in os.path.basename(calling_file):
            frame = frame.f_back
            calling_file = frame.f_code.co_filename
        calling_base = os.path.basename(calling_file)
    except Exception:
        calling_base = "app.py"
        
    title_map = {
        "app.py": "Homepage",
        "1_📊_Overview.py": "Dataset Overview",
        "2_🔍_EDA.py": "Exploratory Data Analysis",
        "3_⚙️_Preprocessing.py": "Data Preprocessing",
        "4_🎯_Feature_Selection.py": "Feature Selection",
        "5_🤖_Model_Training.py": "Model Training",
        "6_🧠_Advanced_Models.py": "Advanced Deep Models",
        "7_🔧_Hyperparameter_Tuning.py": "Hyperparameter Tuning",
        "8_🔬_Feature_Importance.py": "Feature Importance & SHAP",
        "9_🛡️_Robustness.py": "Robustness & Uncertainty",
        "10_🧬_Causal_Uplift.py": "Causal Inference & Uplift",
        "11_🔄_Compression_Recourse.py": "Compression & Recourse",
        "12_💘_Love_Forecaster.py": "Love Forecaster Simulator",
        "13_👥_Team.py": "Research Team",
        "14_📄_Documentation.py": "Project Repository & Report"
    }
    page_title = title_map.get(calling_base, "Homepage")
    
    # Render global header on main page area first
    render_header(page_title=page_title)
    
    # Render sidebar elements
    with st.sidebar:
            
        st.caption("ML Pipeline Dashboard")
        st.markdown("---")
        st.markdown("**Dataset:** Extended (50k × 25)")
        st.markdown("**Models:** 16+ classifiers & deep nets")
        st.markdown("**Task:** Binary classification")
        st.markdown("---")
        st.markdown(
            '<p style="font-size:11px;color:#4b5563;">WIA1006 · University of Malaya<br>Sem 2, 2025/2026</p>',
            unsafe_allow_html=True,
        )


def render_header(page_title=None):
    """Render a premium global website-style header at the top of the page content."""
    import streamlit as st
    import os
    import base64
    import inspect
    
    # Resolve current page filename to apply active tab highlighting
    try:
        frame = inspect.currentframe().f_back
        calling_file = frame.f_code.co_filename
        if "theme.py" in os.path.basename(calling_file):
            frame = frame.f_back
            calling_file = frame.f_code.co_filename
        calling_base = os.path.basename(calling_file)
    except Exception:
        calling_base = "app.py"
        
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    crest_path = os.path.join(ROOT_DIR, "assets", "um_crest_logo.png")
    
    if os.path.exists(crest_path):
        with open(crest_path, "rb") as f:
            crest_base64 = base64.b64encode(f.read()).decode("utf-8")
        logo_src = f"data:image/png;base64,{crest_base64}"
    else:
        logo_src = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/University_of_Malaya_coat_of_arms.png/220px-University_of_Malaya_coat_of_arms.png"
        
    # Active states
    pipeline_pages = [
        "1_📊_Overview.py", 
        "2_🔍_EDA.py", 
        "3_⚙️_Preprocessing.py", 
        "4_🎯_Feature_Selection.py", 
        "5_🤖_Model_Training.py", 
        "7_🔧_Hyperparameter_Tuning.py", 
        "8_🔬_Feature_Importance.py", 
        "9_🛡️_Robustness.py"
    ]
    advanced_pages = [
        "6_🧠_Advanced_Models.py", 
        "10_🧬_Causal_Uplift.py", 
        "11_🔄_Compression_Recourse.py"
    ]
    docs_pages = [
        "13_👥_Team.py", 
        "14_📄_Documentation.py"
    ]
    
    act_home = "active" if calling_base == "app.py" else ""
    act_pipeline = "active" if calling_base in pipeline_pages else ""
    act_advanced = "active" if calling_base in advanced_pages else ""
    act_love = "active" if calling_base == "12_💘_Love_Forecaster.py" else ""
    act_docs = "active" if calling_base in docs_pages else ""
    
    st.markdown(f"""
    <style>
        .website-header {{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            height: 60px !important;
            z-index: 1000005 !important; /* Placed ABOVE stSidebar */
            display: flex !important;
            justify-content: space-between !important;
            align-items: center !important;
            background: rgba(15, 23, 42, 1.0) !important; /* Solid background so sidebar doesn't bleed through */
            border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
            padding: 0 24px 0 80px !important; /* Leaves 80px empty space on left for sidebar toggle */
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
            backdrop-filter: blur(12px) !important;
            margin: 0 !important;
            pointer-events: none !important; /* Allow clicks on empty space to pass through (e.g. to sidebar toggle) */
        }}
        /* Push the main app content down so it's not hidden behind the fixed header */
        .block-container {{
            padding-top: 90px !important;
            margin-top: 0px !important;
        }}
        /* Push the sidebar down so the header doesn't cover its top elements */
        section[data-testid="stSidebar"] {{
            top: 60px !important;
            height: calc(100vh - 60px) !important;
        }}
        /* Fix the Streamlit sidebar toggle and top right elements */
        header[data-testid="stHeader"] {{
            background: transparent !important;
            z-index: 1000001 !important;
            pointer-events: none !important;
        }}
        /* Only re-enable clicks for the sidebar collapse/expand button */
        header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"],
        header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] *,
        header[data-testid="stHeader"] [data-testid="stSidebarCollapseButton"],
        header[data-testid="stHeader"] [data-testid="stSidebarCollapseButton"] *,
        header[data-testid="stHeader"] [data-testid="stHeaderSidebarCollapsedControl"],
        header[data-testid="stHeader"] [data-testid="stHeaderSidebarCollapsedControl"] * {{
            pointer-events: auto !important;
        }}
        [data-testid="stHeaderActionElements"], .stAppDeployButton {{
            display: none !important;
        }}
        /* Move Streamlit's loading/running widget to the bottom left so it doesn't block the top right buttons */
        .stApp [data-testid="stStatusWidget"],
        .stAppHeader [data-testid="stStatusWidget"],
        [data-testid="stStatusWidget"],
        .stStatusWidget {{
            position: fixed !important;
            top: auto !important;
            bottom: 20px !important;
            right: auto !important;
            left: 20px !important;
            transform: none !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            z-index: 9999999 !important;
            background-color: rgba(15, 23, 42, 0.95) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 8px !important;
            padding: 6px 12px !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5) !important;
            pointer-events: auto !important;
        }}
        .web-logo-box {{
            display: flex !important;
            align-items: center !important;
            gap: 14px !important;
            margin-left: 0px !important; /* Managed by header padding-left now */
            pointer-events: auto !important; /* Re-enable pointer events for logo and title */
        }}
        .web-logo-img {{
            height: 38px !important;
            width: auto !important;
            background: white !important;
            padding: 5px !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
        }}
        .web-logo-title {{
            font-family: 'Outfit', sans-serif !important;
            font-weight: 800 !important;
            font-size: 15px !important;
            color: #ffffff !important;
            letter-spacing: -0.2px !important;
        }}
        .web-nav-menu {{
            display: flex !important;
            align-items: center !important;
            gap: 12px !important;
            background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%) !important;
            padding: 8px 16px !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.2) !important;
            pointer-events: auto !important; /* Re-enable pointer events for the menu */
        }}
        .web-nav-item {{
            font-family: 'Inter', sans-serif !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            color: rgba(255, 255, 255, 0.8) !important;
            text-decoration: none !important;
            padding: 6px 14px !important;
            border-radius: 10px !important;
            transition: all 0.2s ease !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
            cursor: pointer !important;
            position: relative !important;
            z-index: 1000003 !important;
            pointer-events: auto !important;
        }}
        .web-dropdown {{
            position: relative;
            display: flex;
            align-items: center;
        }}
        .web-dropdown::after {{
            content: "";
            position: absolute;
            top: 100%;
            left: 0;
            width: 100%;
            height: 12px; /* Invisible bridge to prevent hover loss */
        }}
        .web-dropdown-content {{
            display: none;
            position: absolute;
            background-color: #1e293b;
            min-width: 220px;
            box-shadow: 0px 10px 25px rgba(0,0,0,0.5);
            z-index: 1000004;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.15);
            top: 100%; /* Connects dropdown directly to eliminate hover loss tunnel */
            left: 50%;
            transform: translateX(-50%);
            pointer-events: auto !important;
        }}
        /* Checkbox Hack for Dropdowns on mobile/click */
        .web-dropdown-check {{
            display: none !important;
            visibility: hidden !important;
        }}
        .web-dropdown-check:checked ~ .web-dropdown-content {{
            display: block !important;
        }}
        .web-dropdown-content a {{
            color: rgba(255, 255, 255, 0.8) !important;
            padding: 12px 16px;
            text-decoration: none;
            display: block;
            font-family: 'Inter', sans-serif !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            transition: all 0.2s;
            text-transform: none !important;
            letter-spacing: 0 !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            pointer-events: auto !important;
        }}
        .web-dropdown-content a:last-child {{
            border-bottom: none;
        }}
        .web-dropdown:hover .web-dropdown-content,
        .web-dropdown:focus-within .web-dropdown-content {{
            display: block !important;
        }}
        .web-nav-item:hover,
        .web-nav-item:focus {{
            color: #ffffff !important;
            background: rgba(255, 255, 255, 0.15) !important;
            outline: none !important;
        }}
        .web-nav-item.active {{
            color: #ffffff !important;
            background: rgba(255, 255, 255, 0.25) !important;
            box-shadow: inset 0 1px 4px rgba(0,0,0,0.15) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }}
    </style>
<div class="website-header">
<div class="web-logo-box">
<img class="web-logo-img" src="{logo_src}" alt="UM Crest" />
<span class="web-logo-title">💘 Tying the (Data) Knot: Love, Life & Likes SwipeIQ App</span>
</div>
<div class="web-nav-menu">
<a class="web-nav-item {act_home}" href="/" target="_self">🏠 Home</a>
<div class="web-dropdown">
<input type="checkbox" id="drop-pipeline" class="web-dropdown-check">
<label class="web-nav-item {act_pipeline}" for="drop-pipeline" tabindex="0">⚙️ ML Pipeline ▾</label>
<div class="web-dropdown-content">
<a href="/Overview" target="_self">📊 Overview</a>
<a href="/EDA" target="_self">🔍 EDA</a>
<a href="/Preprocessing" target="_self">⚙️ Preprocessing</a>
<a href="/Feature_Selection" target="_self">🎯 Feature Selection</a>
<a href="/Model_Training" target="_self">🤖 Model Training</a>
<a href="/Hyperparameter_Tuning" target="_self">🔧 Hyperparameter Tuning</a>
<a href="/Feature_Importance" target="_self">🔬 Feature Importance</a>
<a href="/Robustness" target="_self">🛡️ Robustness</a>
</div>
</div>
<div class="web-dropdown">
<input type="checkbox" id="drop-advanced" class="web-dropdown-check">
<label class="web-nav-item {act_advanced}" for="drop-advanced" tabindex="0">🧠 Advanced Methods ▾</label>
<div class="web-dropdown-content">
<a href="/Advanced_Models" target="_self">🧠 Advanced Models</a>
<a href="/Causal_Uplift" target="_self">⚖️ Causal Uplift</a>
<a href="/Compression_Recourse" target="_self">🔄 Compression Recourse</a>
</div>
</div>
<a class="web-nav-item {act_love}" href="/Love_Forecaster" target="_self">💝 Love Forecaster</a>
<div class="web-dropdown">
<input type="checkbox" id="drop-docs" class="web-dropdown-check">
<label class="web-nav-item {act_docs}" for="drop-docs" tabindex="0">📄 Docs ▾</label>
<div class="web-dropdown-content">
<a href="/Documentation" target="_self">📄 Documentation</a>
<a href="/Team" target="_self">👥 Team</a>
</div>
</div>
</div>
</div>
    """, unsafe_allow_html=True)


def render_footer():
    """Render a premium global footer at the bottom of the page content."""
    import streamlit as st
    st.markdown("""
    <style>
        .global-footer-bar {
            margin-top: 60px;
            padding-top: 24px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
        }
        .footer-left {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .footer-text-primary {
            font-size: 12px;
            font-weight: 600;
            color: #94a3b8;
        }
        .footer-text-secondary {
            font-size: 11px;
            color: #4b5563;
        }
        .footer-right {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .footer-link {
            font-size: 11px;
            font-weight: 700;
            color: #a78bfa !important;
            text-decoration: none !important;
            transition: color 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .footer-link:hover {
            color: #f472b6 !important;
        }
    </style>
    <div class="global-footer-bar">
        <div class="footer-left">
            <div class="footer-text-primary">🏛️ Department of Artificial Intelligence · University of Malaya</div>
            <div class="footer-text-secondary">WIA1006 Machine Learning Assignment · FCSIT OCC 6 Group 3 · Session 2025/2026</div>
        </div>
        <div class="footer-right">
            <a class="footer-link" href="https://github.com/ChewWJ1/ML-dating-app-behaviour" target="_blank">🐙 GitHub Repository</a>
            <span style="color: rgba(255,255,255,0.15);">|</span>
            <a class="footer-link" href="/Documentation" target="_self">📄 Report & Jupyter Notes</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
