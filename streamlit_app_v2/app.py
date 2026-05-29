import streamlit as st
import os
import base64
from utils.theme import inject_css, render_sidebar

# ── Page Configuration ──
st.set_page_config(
    page_title="Homepage | SwipeIQ V2",
    page_icon="💘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom premium CSS & render sidebar
inject_css()
render_sidebar()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load and Base64-encode the UM FCSIT widescreen logo
fcsit_logo_path = os.path.join(ROOT_DIR, "assets", "um_fcsit_logo.png")
if os.path.exists(fcsit_logo_path):
    with open(fcsit_logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode("utf-8")
    logo_src = f"data:image/png;base64,{logo_base64}"
else:
    logo_src = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/University_of_Malaya_coat_of_arms.png/220px-University_of_Malaya_coat_of_arms.png"

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS EXTENSIONS FOR PREMIUM HOME-PAGE ELEMENTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
    /* Translucent glassmorphic header */
    .um-header-container {{
        display: flex;
        align-items: center;
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 16px 28px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
    }}
    .um-logo-container {{
        display: flex;
        align-items: center;
        background: #ffffff;
        padding: 8px 16px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }}
    .um-logo-img {{
        height: 60px;
        width: auto;
    }}
    .um-badge-column {{
        margin-left: auto;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 6px;
    }}
    .academic-capsule {{
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.05);
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .academic-course {{
        background: rgba(139, 92, 246, 0.15);
        color: #c084fc;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }}
    
    /* Hero Glow Banner */
    .hero-glow {{
        text-align: center;
        padding: 60px 24px 50px;
        position: relative;
        background: radial-gradient(circle at center, rgba(139,92,246,0.15) 0%, rgba(236,72,153,0.08) 50%, transparent 80%);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        background-color: rgba(15, 23, 42, 0.3);
        margin-bottom: 30px;
        backdrop-filter: blur(8px);
    }}
    .hero-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 60px;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #a78bfa 0%, #ec4899 50%, #14b8a6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        line-height: 1.1;
    }}
    .hero-subtitle {{
        font-family: 'Outfit', sans-serif;
        font-size: 24px;
        font-weight: 600;
        color: #f472b6;
        margin-bottom: 24px;
        letter-spacing: 0.5px;
    }}
    .hero-desc {{
        font-family: 'Inter', sans-serif;
        font-size: 17px;
        line-height: 1.8;
        color: #f1f5f9;
        max-width: 950px;
        margin: 0 auto;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.5);
        text-align: center;
        white-space: normal;
    }}
    .hero-desc b {{ color: #a78bfa; font-weight: 700; }}

    /* KPI Cards Grid */
    .metric-pill-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 20px;
        margin-bottom: 40px;
    }}
    .metric-pill-card {{
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        padding: 24px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(6px);
    }}
    .metric-pill-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }}
    .metric-pill-card.purple::before {{ background: linear-gradient(90deg, #8b5cf6, #a78bfa); }}
    .metric-pill-card.teal::before {{ background: linear-gradient(90deg, #14b8a6, #38bdf8); }}
    .metric-pill-card.pink::before {{ background: linear-gradient(90deg, #ec4899, #f472b6); }}
    .metric-pill-card.amber::before {{ background: linear-gradient(90deg, #f59e0b, #fbbf24); }}
    
    .metric-pill-card:hover {{
        transform: translateY(-4px);
        border-color: rgba(139, 92, 246, 0.3);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
        background: rgba(30, 41, 59, 0.45);
    }}
    .metric-pill-val {{
        font-family: 'Outfit', sans-serif;
        font-size: 38px;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 8px;
    }}
    .metric-pill-card.purple .metric-pill-val {{ color: #c084fc; }}
    .metric-pill-card.teal .metric-pill-val {{ color: #2dd4bf; }}
    .metric-pill-card.pink .metric-pill-val {{ color: #f472b6; }}
    .metric-pill-card.amber .metric-pill-val {{ color: #fbbf24; }}
    
    .metric-pill-lbl {{
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }}
    .metric-pill-sub {{
        font-size: 11px;
        color: #64748b;
    }}

    /* Academic Container Styles */
    .premium-section-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 10px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .premium-section-subtitle {{
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        color: #94a3b8;
        margin-top: 0;
        margin-bottom: 28px;
    }}
    .academic-box {{
        background: rgba(30, 41, 59, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        padding: 28px;
        height: 100%;
        line-height: 1.7;
        backdrop-filter: blur(4px);
    }}

    /* Playgrounds Launcher Grid */
    .launcher-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 20px;
        margin-bottom: 45px;
    }}
    .launcher-card {{
        background: rgba(30, 41, 59, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        padding: 26px;
        display: flex;
        flex-direction: column;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        height: 250px;
        backdrop-filter: blur(6px);
    }}
    .launcher-card::before {{
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0; width: 4px;
        background: transparent;
        transition: background 0.3s;
    }}
    .launcher-card:hover::before {{
        background: linear-gradient(180deg, #8b5cf6, #ec4899);
    }}
    .launcher-card:hover {{
        transform: translateY(-5px);
        border-color: rgba(139,92,246,0.35);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4);
        background: rgba(30, 41, 59, 0.55);
    }}
    .launcher-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
    }}
    .launcher-icon {{
        font-size: 26px;
    }}
    .launcher-badge {{
        font-size: 9px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .launcher-badge.core {{ background: rgba(14, 165, 233, 0.12); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.25); }}
    .launcher-badge.adv {{ background: rgba(168, 85, 247, 0.12); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.25); }}
    .launcher-badge.sota {{ background: rgba(236, 72, 153, 0.12); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.25); }}
    
    .launcher-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 19px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }}
    .launcher-desc {{
        font-family: 'Inter', sans-serif;
        font-size: 13.5px;
        line-height: 1.6;
        color: #94a3b8;
        flex-grow: 1;
        margin-bottom: 18px;
    }}
    .launcher-btn-wrap {{
        display: flex;
        justify-content: flex-end;
    }}
    .launcher-btn {{
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        color: #ffffff !important;
        font-size: 11px;
        font-weight: 700;
        text-align: center;
        padding: 8px 18px;
        border-radius: 8px;
        text-decoration: none !important;
        transition: all 0.25s ease;
        border: 1px solid rgba(255,255,255,0.1);
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}
    .launcher-btn:hover {{
        opacity: 0.95;
        box-shadow: 0 4px 15px rgba(139,92,246,0.35);
        transform: scale(1.02);
    }}
</style>
""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER — VISUAL GLOW BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-glow">
    <div class="hero-title">💘 Tying the (Data) Knot</div>
    <div class="hero-subtitle">Love, Life & Likes — SwipeIQ App</div>
    <p class="hero-desc">
        Welcome to the official <b>SwipeIQ V2 Interactive Analytics Hub</b>. 
        Developed as an academic capstone showcase, this platform hosts a 
        <b>state of the art 15-stage machine learning pipeline</b> exploring <b>16+ baseline and advanced models</b>. 
        By incorporating <b>Causal Discovery (DAGs)</b>, <b>Double Machine Learning (DML)</b>, 
        <b>Graph Attention Networks (GNN)</b>, <b>Self-Supervised SCARF pre-training</b>, 
        <b>TabPFN Zero-Shot Transformers</b>, and <b>Algorithmic Recourse (DiCE)</b>, 
        we stress-test and visualize the mathematical predictability boundaries of programmatic human behavioral distributions.
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KEY METRICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="metric-pill-grid">
    <div class="metric-pill-card purple">
        <div class="metric-pill-val">50,000</div>
        <div class="metric-pill-lbl">Data Cohort Volume</div>
        <div class="metric-pill-sub">Balanced Swipe Outcomes</div>
    </div>
    <div class="metric-pill-card teal">
        <div class="metric-pill-val">67</div>
        <div class="metric-pill-lbl">Feature Dimensions</div>
        <div class="metric-pill-sub">Nominal, Ordinal & Multi-hot Spaces</div>
    </div>
    <div class="metric-pill-card pink">
        <div class="metric-pill-val">16+</div>
        <div class="metric-pill-lbl">Classifiers Evaluated</div>
        <div class="metric-pill-sub">Traditional, Deep Tabular & GNNs</div>
    </div>
    <div class="metric-pill-card amber">
        <div class="metric-pill-val">60.30%</div>
        <div class="metric-pill-lbl">Empirical Signal Ceiling</div>
        <div class="metric-pill-sub">Validated Test Accuracy Baseline</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LIMITS OF PREDICTIVE MATCHMAKING RESEARCH IMAGE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #f472b6; font-family: Outfit, sans-serif;'>🎯 Limits of Predictive Matchmaking Research</h4>", unsafe_allow_html=True)
limits_pipeline_img = os.path.join(ROOT_DIR, "assets", "NotebookLM", "new", "Science_of_Digital_Romance_Infographic.png")
if os.path.exists(limits_pipeline_img):
    st.image(limits_pipeline_img, caption="Science of Digital Romance Infographic")
else:
    st.warning("Science of Digital Romance Infographic not found in assets.")

journey_img = os.path.join(ROOT_DIR, "assets", "NotebookLM", "new", "Dating_Success_Machine_Learning_Journey.png")
if os.path.exists(journey_img):
    st.image(journey_img, caption="Dating Success Machine Learning Journey")
else:
    st.warning("Dating Success Machine Learning Journey not found in assets.")
st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURED SEGMENT: EXECUTIVE SUMMARY & CORE FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="premium-section-title">
    <span>🔬</span> Executive Summary & Core Research Findings
</div>
<div class="premium-section-subtitle">
    Technical synthesis and empirical findings of the SwipeIQ V5.2 Machine Learning Pipeline
</div>
""", unsafe_allow_html=True)

col_exec1, col_exec2 = st.columns(2)

with col_exec1:
    st.markdown("""
    <div class="academic-box" style="border-left: 4px solid #8b5cf6;">
        <h4 style="color: #a78bfa; margin-top:0; font-family:'Outfit', sans-serif; font-size:18px;">🏗️ Project Scope & Pipeline Architecture</h4>
        <p style="font-size: 14px; color: #cbd5e1; margin-bottom: 12px;">
            This research presents a State-of-the-Art end-to-end Machine Learning classification pipeline designed to predict romantic compatibility outcomes on mobile dating applications. 
            Utilizing a 50,000-sample behavioral dataset, we binarize 10 multi-class swipe outcomes into a robust target compatibility variable (Mutual Match, Instant Match, Date Happened, and Relationship Formed).
        </p>
        <p style="font-size: 14px; color: #cbd5e1; margin-bottom: 12px;">
            To maintain strict engineering rigor, the data undergoes comprehensive preprocessing: ordinal scaling for educational and income levels, target interest tags multi-hot encoding, and numerical outlier mitigation using <i>RobustScaler</i> and <i>QuantileTransformer</i>. 
            Crucially, an unsupervised <i>Isolation Forest</i> is integrated at the tail-end of preprocessing as an Out-of-Distribution (OOD) rejection guardrail.
        </p>
        <p style="font-size: 14px; color: #cbd5e1; margin-bottom: 0;">
            Features are mathematically selected via a union of ANOVA F-scores, Mutual Information splits, and Boruta shadow feature filtering, compressing the input space into 67 highly optimized feature dimensions for cross-validated model tuning.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_exec2:
    st.markdown("""
    <div class="academic-box" style="border-left: 4px solid #ec4899;">
        <h4 style="color: #f472b6; margin-top:0; font-family:'Outfit', sans-serif; font-size:18px;">💡 Key Scientific Breakthroughs & Production Paths</h4>
        <p style="font-size: 14px; color: #cbd5e1; margin-bottom: 12px;">
            <b>The Predictability Limit:</b> While our 15-stage pipeline exhibits complete structural integrity, all 16 evaluated classifiers (including GAT GNNs, SCARF self-supervised contrastive nets, and TabPFN Zero-Shot Transformers) converge directly at the majority class baseline of <b>60.30% accuracy (ROC-AUC ≈ 0.50)</b>.
        </p>
        <p style="font-size: 14px; color: #cbd5e1; margin-bottom: 12px;">
            This convergence represents a crucial, mathematically honest research finding. It mathematically proves the <b>absence of predictive signal</b> within typical structured profile attributes (such as zodiac sign, swipe ratio, or demographic variables). 
            Furthermore, Double Machine Learning (DML) causal estimation confirms that the Average Treatment Effect (ATE) of profile photo counts on matching success is statistically zero (p > 0.60).
        </p>
        <p style="font-size: 14px; color: #cbd5e1; margin-bottom: 0;">
            <b>Next-Gen Production Recommendations:</b> We recommend that future dating matching architectures pivot from structured demographics to <b>unstructured NLP bio-profile embedding extraction (LLMs)</b> and <b>longitudinal behavioral telemetry</b> (active chat lengths, messaging latency, response entropy) to capture true compatibility signals.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# THE 9 MATHEMATICAL PLAYGROUNDS LAUNCHER GRID
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="premium-section-title">
    <span>🎛️</span> Interactive Playgrounds & Sandbox Suite
</div>
<div class="premium-section-subtitle">
    Explore the 9 integrated real-time mathematical sandboxes to stress-test preprocessing, training, causal inference, and network topologies
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="launcher-grid">
    <!-- OLS Sandbox -->
    <div class="launcher-card">
        <div class="launcher-top">
            <span class="launcher-icon">📈</span>
            <span class="launcher-badge core">Core</span>
        </div>
        <div class="launcher-title">Bivariate OLS Slope Sandbox</div>
        <div class="launcher-desc">Audits simple linear slopes, residuals, and coefficient changes dynamically. Inspects how traditional OLS fits adapt to noise.</div>
        <div class="launcher-btn-wrap">
            <a href="/Overview" target="_self" class="launcher-btn">Launch OLS</a>
        </div>
    </div>
    <!-- Scaling Sandbox -->
    <div class="launcher-card">
        <div class="launcher-top">
            <span class="launcher-icon">⚙️</span>
            <span class="launcher-badge core">Core</span>
        </div>
        <div class="launcher-title">Outlier Robust Scaling Sandbox</div>
        <div class="launcher-desc">Interactive comparison of RobustScaler, StandardScaler, and MinMax behaviors under heavy-tailed synthetic outliers.</div>
        <div class="launcher-btn-wrap">
            <a href="/Preprocessing" target="_self" class="launcher-btn">Launch Scaler</a>
        </div>
    </div>
    <!-- PCA Sandbox -->
    <div class="launcher-card">
        <div class="launcher-top">
            <span class="launcher-icon">🎯</span>
            <span class="launcher-badge core">Core</span>
        </div>
        <div class="launcher-title">3D PCA Variance Projection</div>
        <div class="launcher-desc">Rotates and explores PCA dimension clusters in a premium 3D Plotly canvas to observe class separation.</div>
        <div class="launcher-btn-wrap">
            <a href="/Feature_Selection" target="_self" class="launcher-btn">Launch PCA</a>
        </div>
    </div>
    <!-- Boundaries Sandbox -->
    <div class="launcher-card">
        <div class="launcher-top">
            <span class="launcher-icon">🤖</span>
            <span class="launcher-badge adv">Advanced</span>
        </div>
        <div class="launcher-title">15-Model Decision Boundaries</div>
        <div class="launcher-desc">Renders custom classification probability contours across 15 separate classifiers on synthetic binary sets.</div>
        <div class="launcher-btn-wrap">
            <a href="/Model_Training" target="_self" class="launcher-btn">Launch Boundary</a>
        </div>
    </div>
    <!-- Optuna Sandbox -->
    <div class="launcher-card">
        <div class="launcher-top">
            <span class="launcher-icon">🔧</span>
            <span class="launcher-badge adv">Advanced</span>
        </div>
        <div class="launcher-title">GPU Optuna Pareto Frontier</div>
        <div class="launcher-desc">Navigates multi-objective tradeoffs between precision and recall curves. Tracks candidate hyperparameter trials.</div>
        <div class="launcher-btn-wrap">
            <a href="/Hyperparameter_Tuning" target="_self" class="launcher-btn">Launch Optuna</a>
        </div>
    </div>
    <!-- Causal Sandbox -->
    <div class="launcher-card">
        <div class="launcher-top">
            <span class="launcher-icon">⚖️</span>
            <span class="launcher-badge sota">SOTA</span>
        </div>
        <div class="launcher-title">Targeted Causal Uplift (DML)</div>
        <div class="launcher-desc">Recalculates push notification campaign profit margins dynamically using double machine learning treatments.</div>
        <div class="launcher-btn-wrap">
            <a href="/Causal_Uplift" target="_self" class="launcher-btn">Launch Causal</a>
        </div>
    </div>
    <!-- Attention Sandbox -->
    <div class="launcher-card">
        <div class="launcher-top">
            <span class="launcher-icon">🧠</span>
            <span class="launcher-badge sota">SOTA</span>
        </div>
        <div class="launcher-title">FT-Transformer Attention Map</div>
        <div class="launcher-desc">Modulates temperature scales to view self-attention weight heatmaps across tabular inputs in PyTorch.</div>
        <div class="launcher-btn-wrap">
            <a href="/Advanced_Models" target="_self" class="launcher-btn">Launch Attention</a>
        </div>
    </div>
    <!-- GNN Sandbox -->
    <div class="launcher-card">
        <div class="launcher-top">
            <span class="launcher-icon">🧬</span>
            <span class="launcher-badge sota">SOTA</span>
        </div>
        <div class="launcher-title">GNN Neighbor Similarity Graph</div>
        <div class="launcher-desc">Simulates topological graph network neighbors, connection layers, and user clustering distances dynamically.</div>
        <div class="launcher-btn-wrap">
            <a href="/Advanced_Models" target="_self" class="launcher-btn">Launch GNN Graph</a>
        </div>
    </div>
    <!-- Concept Drift Sandbox -->
    <div class="launcher-card">
        <div class="launcher-top">
            <span class="launcher-icon">🗜️</span>
            <span class="launcher-badge sota">SOTA</span>
        </div>
        <div class="launcher-title">ADWIN Concept Drift Simulator</div>
        <div class="launcher-desc">Triggers abrupt or gradual drift spikes across a mock pipeline to watch ADWIN alarms and rolling PSI metrics.</div>
        <div class="launcher-btn-wrap">
            <a href="/Compression_Recourse" target="_self" class="launcher-btn">Launch Drift</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DETAILS & LIFECYCLE REF — TABBED SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="premium-section-title" style="margin-top:20px;">
    <span>🗺️</span> Technical Reference, Teams & Lifecycles
</div>
<div class="premium-section-subtitle">
    Deep dive into the 15-stage machine learning lifecycle stages, academic research roster, and system architecture mappings
</div>
""", unsafe_allow_html=True)

tab_journey, tab_team, tab_diagrams = st.tabs([
    "🧭 The 15-Stage ML Lifecycle Map", 
    "👥 Research Team & Contribution Matrix", 
    "🔮 Technical Architecture & Pipeline Diagrams"
])

# ── TAB 1: PIPELINE JOURNEY MAP ──
with tab_journey:
    st.markdown("""
    <div style="margin-top:16px; margin-bottom:24px;">
        <h3 style="font-family:'Outfit', sans-serif; font-weight:600; margin-bottom:4px; font-size:20px;">🧭 Chronological System Architecture Reference</h3>
        <p style="font-size:14px; color:#94a3b8; margin-top:0;">Navigate to individual pipeline audits by selecting a card below or matching the sidebar index.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        .pipeline-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .pipeline-card {
            background: rgba(30, 41, 59, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            height: 220px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .pipeline-card:hover {
            transform: translateY(-5px);
            border-color: rgba(139,92,246,0.3);
            box-shadow: 0 8px 30px rgba(139,92,246,0.08);
            background: rgba(30, 41, 59, 0.5);
        }
        .pipeline-card-num {
            font-family: 'Outfit', sans-serif;
            font-size: 12px;
            font-weight: 700;
            color: #818cf8;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .pipeline-card-title {
            font-family: 'Outfit', sans-serif;
            font-size: 18px;
            font-weight: 600;
            color: #f1f5f9;
            margin-bottom: 10px;
        }
        .pipeline-card-desc {
            font-family: 'Inter', sans-serif;
            font-size: 13.5px;
            line-height: 1.6;
            color: #94a3b8;
            flex-grow: 1;
        }
        .pipeline-card-tag {
            position: absolute;
            bottom: 20px;
            right: 20px;
            font-size: 10px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 6px;
            text-transform: uppercase;
        }
        .tag-core { background: rgba(14, 165, 233, 0.12); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.2); }
        .tag-adv { background: rgba(168, 85, 247, 0.12); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.2); }
        .tag-sota { background: rgba(236, 72, 153, 0.12); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.2); }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <a href="/Overview" target="_self" style="text-decoration:none; color:inherit;">
            <div class="pipeline-card">
                <div class="pipeline-card-num">Stage 01</div>
                <div class="pipeline-card-title">📊 Dataset & EDA</div>
                <div class="pipeline-card-desc">Explores the 50,000 dating profiles, distributions, correlation matrices, and demographic spreads.</div>
                <span class="pipeline-card-tag tag-core">Core</span>
            </div>
        </a>
        """, unsafe_allow_html=True)
        st.markdown("""
        <a href="/Advanced_Models" target="_self" style="text-decoration:none; color:inherit;">
            <div class="pipeline-card" style="margin-top:20px;">
                <div class="pipeline-card-num">Stage 05</div>
                <div class="pipeline-card-title">🧠 Deep Learning</div>
                <div class="pipeline-card-desc">Custom PyTorch neural models, SAINT self-attention, TabNet feature selectors, and GNN networks.</div>
                <span class="pipeline-card-tag tag-adv">Advanced</span>
            </div>
        </a>
        """, unsafe_allow_html=True)
        st.markdown("""
        <a href="/Causal_Uplift" target="_self" style="text-decoration:none; color:inherit;">
            <div class="pipeline-card" style="margin-top:20px;">
                <div class="pipeline-card-num">Stage 09</div>
                <div class="pipeline-card-title">⚖️ Causal AI</div>
                <div class="pipeline-card-desc">Confounder backdoor adjustment, Double ML Average Treatment Effects, and T-Learner uplift matrices.</div>
                <span class="pipeline-card-tag tag-sota">SOTA</span>
            </div>
        </a>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <a href="/Preprocessing" target="_self" style="text-decoration:none; color:inherit;">
            <div class="pipeline-card">
                <div class="pipeline-card-num">Stage 02</div>
                <div class="pipeline-card-title">⚙️ Preprocessing</div>
                <div class="pipeline-card-desc">Demographic encodings, Ordinal Consolidations, RobustScaler layers, and OOD Isolation Forest pre-filters.</div>
                <span class="pipeline-card-tag tag-core">Core</span>
            </div>
        </a>
        """, unsafe_allow_html=True)
        st.markdown("""
        <a href="/Hyperparameter_Tuning" target="_self" style="text-decoration:none; color:inherit;">
            <div class="pipeline-card" style="margin-top:20px;">
                <div class="pipeline-card-num">Stage 06</div>
                <div class="pipeline-card-title">🔧 Optuna Tuning</div>
                <div class="pipeline-card-desc">1,000-trial GPU-accelerated random search and Multi-Objective Pareto optimization.</div>
                <span class="pipeline-card-tag tag-adv">Advanced</span>
            </div>
        </a>
        """, unsafe_allow_html=True)
        st.markdown("""
        <a href="/Compression_Recourse" target="_self" style="text-decoration:none; color:inherit;">
            <div class="pipeline-card" style="margin-top:20px;">
                <div class="pipeline-card-num">Stage 10</div>
                <div class="pipeline-card-title">🗜️ Distillation & DiCE</div>
                <div class="pipeline-card-desc">350x Teacher-Student surrogate model compression and Microsoft DiCE recourse pathways.</div>
                <span class="pipeline-card-tag tag-sota">SOTA</span>
            </div>
        </a>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <a href="/Feature_Selection" target="_self" style="text-decoration:none; color:inherit;">
            <div class="pipeline-card">
                <div class="pipeline-card-num">Stage 03</div>
                <div class="pipeline-card-title">🎯 Feature Selection</div>
                <div class="pipeline-card-desc">ANOVA, Mutual Information splits, Boruta Shadow-feature filtering, and 3D PCA clusters.</div>
                <span class="pipeline-card-tag tag-core">Core</span>
            </div>
        </a>
        """, unsafe_allow_html=True)
        st.markdown("""
        <a href="/Feature_Importance" target="_self" style="text-decoration:none; color:inherit;">
            <div class="pipeline-card" style="margin-top:20px;">
                <div class="pipeline-card-num">Stage 07</div>
                <div class="pipeline-card-title">🔬 SHAP Interactions</div>
                <div class="pipeline-card-desc">Friedman's H-statistics, global attributions, and TreeSHAP joint interaction curves.</div>
                <span class="pipeline-card-tag tag-adv">Advanced</span>
            </div>
        </a>
        """, unsafe_allow_html=True)
        st.markdown("""
        <a href="/Love_Forecaster" target="_self" style="text-decoration:none; color:inherit;">
            <div class="pipeline-card" style="margin-top:20px;">
                <div class="pipeline-card-num">Stage 11</div>
                <div class="pipeline-card-title">💘 Love Forecaster</div>
                <div class="pipeline-card-desc">Premium real-time matching inference engine utilizing the actual trained XGBoost champion.</div>
                <span class="pipeline-card-tag tag-sota">SOTA</span>
            </div>
        </a>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <a href="/Model_Training" target="_self" style="text-decoration:none; color:inherit;">
            <div class="pipeline-card">
                <div class="pipeline-card-num">Stage 04</div>
                <div class="pipeline-card-title">🤖 Model Training</div>
                <div class="pipeline-card-desc">AutoML base-line validations, 16 distinct classifiers, and interactive decision boundary contours.</div>
                <span class="pipeline-card-tag tag-core">Core</span>
            </div>
        </a>
        """, unsafe_allow_html=True)
        st.markdown("""
        <a href="/Robustness" target="_self" style="text-decoration:none; color:inherit;">
            <div class="pipeline-card" style="margin-top:20px;">
                <div class="pipeline-card-num">Stage 08</div>
                <div class="pipeline-card-title">🛡️ Robust & Privacy</div>
                <div class="pipeline-card-desc">Conformal MAPIE set coverage, Bayesian MC Dropout, Opacus DP-SGD, and FGSM adversarial.</div>
                <span class="pipeline-card-tag tag-adv">Advanced</span>
            </div>
        </a>
        """, unsafe_allow_html=True)
        st.markdown("""
        <a href="/Documentation" target="_self" style="text-decoration:none; color:inherit;">
            <div class="pipeline-card" style="margin-top:20px;">
                <div class="pipeline-card-num">Stage 12</div>
                <div class="pipeline-card-title">📄 Project Repository</div>
                <div class="pipeline-card-desc">Access the official 54-page PDF SOTA report, Jupyter source notebooks, and project metadata.</div>
                <span class="pipeline-card-tag tag-sota">SOTA</span>
            </div>
        </a>
        """, unsafe_allow_html=True)

# ── TAB 2: RESEARCH TEAM & CONTRIBUTIONS ──
with tab_team:
    st.markdown("""
    <div style="margin-top:16px; margin-bottom:24px;">
        <h3 style="font-family:'Outfit', sans-serif; font-weight:600; margin-bottom:4px; font-size:20px;">👥 FCSIT Research Group OCC 3</h3>
        <p style="font-size:14px; color:#94a3b8; margin-top:0;">University of Malaya · OCC 6 Group 3 · Allocation Matrix and Technical Ownership Breakdown</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        .team-card-home {
            background: rgba(30, 41, 59, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s ease;
            position: relative;
        }
        .team-card-home:hover {
            border-color: rgba(236, 72, 153, 0.3);
            box-shadow: 0 8px 30px rgba(236, 72, 153, 0.05);
        }
        .team-card-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        .team-card-avatar {
            font-size: 32px;
            margin-right: 16px;
        }
        .team-card-name {
            font-family: 'Outfit', sans-serif;
            font-size: 18px;
            font-weight: 700;
            color: #ffffff;
        }
        .team-card-matrix {
            font-size: 13px;
            color: #94a3b8;
        }
        .team-card-role {
            font-size: 12px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 6px;
            background: rgba(139,92,246,0.12);
            color: #a78bfa;
            border: 1px solid rgba(139,92,246,0.2);
            width: fit-content;
            margin-bottom: 12px;
        }
        .team-card-list {
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            line-height: 1.6;
            color: #cbd5e1;
            padding-left: 18px;
            margin: 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    c_team1, c_team2 = st.columns(2)
    with c_team1:
        st.markdown("""
        <div class="team-card-home">
            <div class="team-card-header">
                <div class="team-card-avatar">👑</div>
                <div>
                    <div class="team-card-name">Chew Wei Jian</div>
                    <div class="team-card-matrix">Matrix No: 23118568/2</div>
                </div>
            </div>
            <div class="team-card-role">Project Leader & ML Pipeline Lead</div>
            <ul class="team-card-list">
                <li>Coordinates task delegation, project timeline tracking, and repository management.</li>
                <li>Programmed the core pipeline execution script and automatic Google Colab/local path configs.</li>
                <li>Implemented parallel computing, multi-threading SVM, and cross-validation thread isolation.</li>
            </ul>
        </div>
        
        <div class="team-card-home" style="margin-top:20px;">
            <div class="team-card-header">
                <div class="team-card-avatar">📊</div>
                <div>
                    <div class="team-card-name">Ng Jin Ru</div>
                    <div class="team-card-matrix">Matrix No: 23116192/2</div>
                </div>
            </div>
            <div class="team-card-role" style="color:#38bdf8; background:rgba(14,165,233,0.12); border-color:rgba(14,165,233,0.2);">Exploratory Data Analysis (EDA) Analyst</div>
            <ul class="team-card-list">
                <li>Performed initial univariate and bivariate visualizations (histograms, count plots, box plots).</li>
                <li>Analyzed target class balance and examined missing values and duplicate records.</li>
                <li>Visualized correlation matrices (Pearson) and feature-versus-target relationships (likes, swipe ratio).</li>
            </ul>
        </div>
        
        <div class="team-card-home" style="margin-top:20px;">
            <div class="team-card-header">
                <div class="team-card-avatar">💻</div>
                <div>
                    <div class="team-card-name">Chaang Wai Chiu</div>
                    <div class="team-card-matrix">Matrix No: 23104771/2</div>
                </div>
            </div>
            <div class="team-card-role" style="color:#fbbf24; background:rgba(245,158,11,0.12); border-color:rgba(245,158,11,0.2);">Explainability, Ethics & UI Developer</div>
            <ul class="team-card-list">
                <li>Implemented SHAP (Shapley Additive exPlanations) values and generated beeswarm interpretability plots.</li>
                <li>Evaluated fairness through demographic parity checks across user gender identities.</li>
                <li>Constructed the premium, interactive HTML/CSS dashboard with an embedded prediction simulator.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with c_team2:
        st.markdown("""
        <div class="team-card-home">
            <div class="team-card-header">
                <div class="team-card-avatar">🔧</div>
                <div>
                    <div class="team-card-name">Ku Jian Cheng</div>
                    <div class="team-card-matrix">Matrix No: 23079373/2</div>
                </div>
            </div>
            <div class="team-card-role" style="color:#2dd4bf; background:rgba(20,184,166,0.12); border-color:rgba(20,184,166,0.2);">Data Preprocessing & Feature Engineer</div>
            <ul class="team-card-list">
                <li>Handled data extraction and cleaned redundant variables from the 50,000 dating dataset records.</li>
                <li>Designed ordinal mappings for education and income, using regex/keyword matching to fix unicode character issues.</li>
                <li>Built categorical nominal one-hot encoders and interest tag multi-hot encoders.</li>
            </ul>
        </div>
        
        <div class="team-card-home" style="margin-top:20px;">
            <div class="team-card-header">
                <div class="team-card-avatar">🤖</div>
                <div>
                    <div class="team-card-name">Ang Ying En</div>
                    <div class="team-card-matrix">Matrix No: 23116738/2</div>
                </div>
            </div>
            <div class="team-card-role" style="color:#f472b6; background:rgba(236,72,153,0.12); border-color:rgba(236,72,153,0.2);">Model Optimization & Tuning Engineer</div>
            <ul class="team-card-list">
                <li>Configured and trained 6 baseline ML models: Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, and SVM.</li>
                <li>Programmed cross-validation performance loops to evaluate accuracy, precision, recall, F1, and ROC-AUC.</li>
                <li>Setup RandomizedSearchCV tuning grids and executed 150 fits per candidate estimator.</li>
            </ul>
        </div>
        
        <div class="team-card-home" style="margin-top:20px; padding: 22px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, rgba(139,92,246,0.05), rgba(236,72,153,0.05)); height: 160px;">
            <div style="text-align: center;">
                <div style="font-size: 32px; margin-bottom: 4px;">🎓</div>
                <div style="font-family:'Outfit', sans-serif; font-size:16px; font-weight:700; color:#f1f5f9;">University of Malaya OCC 6</div>
                <div style="font-size:13px; color:#94a3b8; font-weight:500;">Department of Artificial Intelligence</div>
                <div style="font-size:11px; color:#64748b;">Faculty of Computer Science and Information Technology</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 3: CONCEPTUAL FLOW DIAGRAMS ──
with tab_diagrams:
    st.markdown("""
    <div style="margin-top:16px; margin-bottom:24px;">
        <h3 style="font-family:'Outfit', sans-serif; font-weight:600; margin-bottom:4px; font-size:20px;">🔮 Structural Architecture & Romance Mappings</h3>
        <p style="font-size:14px; color:#94a3b8; margin-top:0;">Visual processes demonstrating representation learning flows, target transformations, and causal-backdoor adjustments.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_diag1, col_diag2 = st.columns(2)
    
    with col_diag1:
        st.markdown("#### 🧭 Modern Romance System Pipeline Flow")
        romance_pipeline_img = os.path.join(ROOT_DIR, "assets", "NotebookLM", "Modern_Romance_Machine_Learning_Pipeline.png")
        if os.path.exists(romance_pipeline_img):
            st.image(romance_pipeline_img, caption="Process Flow: Modern Romance Machine Learning Pipeline Map", use_container_width=True)
        else:
            st.warning("Romance pipeline flow diagram not found in assets.")

    with col_diag2:
        st.info("Additional architectural diagrams and methodology charts will be populated here.")
# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:13px; padding:30px 0; border-top:1px solid rgba(255,255,255,0.06); margin-top: 40px;">
    University of Malaya · FCSIT OCC 6 Group 3· SwipeIQ V2 Research Hub · Semester 2, Session 2025/2026
    <br>
    <p style="margin-top:10px; font-size:14.5px; color:#a78bfa;">👈 Select a stage from the sidebar or click any button above to begin auditing.</p>
</div>
""", unsafe_allow_html=True)


# ── Render Global Footer ──
from utils.theme import render_footer
render_footer()






