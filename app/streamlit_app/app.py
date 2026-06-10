import streamlit as st
from utils.theme import inject_css, render_sidebar

# Configure page settings
st.set_page_config(
    page_title="SwipeIQ Dashboard",
    page_icon="💘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom premium CSS
inject_css()

# Render sidebar
render_sidebar()

# ── Main Content ──
st.markdown('<h1 style="text-align: center; margin-bottom: 0;">💘 SwipeIQ</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #a78bfa; font-size: 20px; font-weight: 600; margin-bottom: 30px;">Tying the Data Knot: Predicting Meaningful Connections</p>', unsafe_allow_html=True)

# Hero section
st.markdown("""
<div style="background: rgba(139,92,246,0.05); border: 1px solid rgba(139,92,246,0.2); border-radius: 14px; padding: 24px; text-align: center; margin-bottom: 40px;">
    <p style="font-size: 16px; line-height: 1.6; color: #f1f5f9; max-width: 800px; margin: 0 auto;">
        Welcome to the <b>SwipeIQ Machine Learning Dashboard</b>. This interactive application walks you through an end-to-end machine learning pipeline designed to predict whether a dating app user will achieve a <b>meaningful connection</b> based on their demographics and in-app behaviour patterns.
    </p>
</div>
""", unsafe_allow_html=True)

# Pipeline Flow Diagram
st.markdown("### 🗺️ Machine Learning Pipeline Journey")
st.markdown("Navigate through the sidebar pages to explore each stage of the data science lifecycle:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="pipeline-step" style="height: 180px;">
        <h4>1. 📊 Overview & 🔍 EDA</h4>
        <p style="font-size: 13px; color: #94a3b8;">Explore the 50,000 user dataset, distributions, and initial correlations.</p>
    </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.markdown("""
    <div class="pipeline-step" style="height: 180px;">
        <h4>2. ⚙️ Preprocessing</h4>
        <p style="font-size: 13px; color: #94a3b8;">Clean, encode (Ordinal/One-Hot/Multi-Hot), and scale the features.</p>
    </div>
    """, unsafe_allow_html=True)
    
with col3:
    st.markdown("""
    <div class="pipeline-step" style="height: 180px;">
        <h4>3. 🎯 Feature Selection</h4>
        <p style="font-size: 13px; color: #94a3b8;">Filter noise using ANOVA F-score, Mutual Information, and PCA.</p>
    </div>
    """, unsafe_allow_html=True)
    
with col4:
    st.markdown("""
    <div class="pipeline-step" style="height: 180px;">
        <h4>4. 🤖 Model Training</h4>
        <p style="font-size: 13px; color: #94a3b8;">Train 6 baseline models and evaluate performance metrics.</p>
    </div>
    """, unsafe_allow_html=True)

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.markdown("""
    <div class="pipeline-step" style="height: 180px;">
        <h4>5. 🔧 Tuning</h4>
        <p style="font-size: 13px; color: #94a3b8;">Optimize hyperparameters using RandomizedSearchCV.</p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class="pipeline-step" style="height: 180px;">
        <h4>6. 🔬 Explainability</h4>
        <p style="font-size: 13px; color: #94a3b8;">Extract feature importances to understand model decisions.</p>
    </div>
    """, unsafe_allow_html=True)

with col7:
    st.markdown("""
    <div class="pipeline-step" style="height: 180px; border-left: 3px solid #ec4899;">
        <h4 style="color: #f472b6;">7. 💘 Love Forecaster</h4>
        <p style="font-size: 13px; color: #94a3b8;">Real-time interactive prediction using the actual trained models.</p>
    </div>
    """, unsafe_allow_html=True)
    
with col8:
    st.markdown("""
    <div class="pipeline-step" style="height: 180px; border-left: 3px solid #14b8a6;">
        <h4 style="color: #2dd4bf;">8. 📁 Reports</h4>
        <p style="font-size: 13px; color: #94a3b8;">Download project assets, CSV data, and documentation.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #4b5563; font-size: 14px;'>👈 Select a stage from the sidebar to begin.</p>", unsafe_allow_html=True)
