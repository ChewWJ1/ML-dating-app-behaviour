import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from utils.data_loader import load_raw_data, load_eda_stats
from utils.theme import inject_css, render_sidebar, get_plotly_layout, PLOTLY_COLORS

st.set_page_config(page_title="EDA | SwipeIQ", page_icon="🔍", layout="wide")
inject_css()
render_sidebar()

st.title("🔍 Exploratory Data Analysis")

st.markdown("""
<div style="background:rgba(59,130,246,0.06); border:1px dashed rgba(59,130,246,0.3); border-radius:8px; padding:16px; font-size:13px; color:#93c5fd; line-height:1.5; margin-bottom: 24px;">
    <strong>📊 Scientific Objective & EDA Findings:</strong><br>
    The core project objective is to identify behavioral patterns that correlate with positive connection outcomes, allowing dating applications to optimize matching pools and reduce ghosting rates. However, our Exploratory Data Analysis confirms the synthetic programmatic nature of the dataset: <strong>all 25 features are uniformly distributed</strong>, no extreme outliers exist, and Pearson correlation matrices show an absence of any strong correlation (values near 0).
</div>
""", unsafe_allow_html=True)

df = load_raw_data()
eda_stats = load_eda_stats()

# Helper to get plot path
def get_plot_path(filename):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "plots", filename)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🎯 Feature vs Target", "🔗 Correlations", "🏷️ Interest Tags"])

with tab1:
    st.markdown("### Categorical Distributions")
    st.markdown("All categorical variables exhibit near-perfect uniform distribution.")
    plot_path = get_plot_path("02_3_4_categorical_feature_distributions.png")
    if os.path.exists(plot_path):
        st.image(plot_path, use_container_width=True)
        
    st.markdown("#### Interactive View (Sample)")
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(df, x='gender', color_discrete_sequence=[PLOTLY_COLORS[0]])
            fig1.update_layout(**get_plotly_layout("Gender Distribution", height=300))
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.histogram(df, x='location_type', color_discrete_sequence=[PLOTLY_COLORS[1]])
            fig2.update_layout(**get_plotly_layout("Location Distribution", height=300))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### Numerical Distributions")
    st.markdown("Numerical variables also follow uniform distributions with no distinct skewness.")
    plot_path_num = get_plot_path("03_3_5_numerical_feature_distributions.png")
    if os.path.exists(plot_path_num):
        st.image(plot_path_num, use_container_width=True)

    st.markdown("---")
    st.markdown("### Outlier Detection")
    st.markdown("Boxplots show that no extreme outliers exist in the dataset due to its controlled synthetic generation.")
    plot_path_out = get_plot_path("04_3_6_numerical_features_outlier_detection_boxplots.png")
    if os.path.exists(plot_path_out):
        st.image(plot_path_out, use_container_width=True)

with tab2:
    st.markdown("### Numerical Features by Target")
    st.markdown("When split by outcome, the distributions remain largely identical, showing no obvious separability.")
    plot_path_num_tgt = get_plot_path("05_3_7_feature_vs_target_numerical_features_by_outcom.png")
    if os.path.exists(plot_path_num_tgt):
        st.image(plot_path_num_tgt, use_container_width=True)
        
    st.markdown("---")
    st.markdown("### Categorical Features by Target")
    st.markdown("The proportion of outcomes is evenly distributed across all categories.")
    plot_path_cat_tgt = get_plot_path("06_3_8_feature_vs_target_categorical_features_by_outc.png")
    if os.path.exists(plot_path_cat_tgt):
        st.image(plot_path_cat_tgt, use_container_width=True)

with tab3:
    st.markdown("### Correlation Heatmap")
    st.markdown("Pearson correlation coefficients between all numerical features. Notice the absence of any strong correlation (values near 0).")
    plot_path_corr = get_plot_path("07_3_9_correlation_heatmap_numerical_features.png")
    if os.path.exists(plot_path_corr):
        st.image(plot_path_corr, use_container_width=True)
        
    st.markdown("#### Interactive Correlation Heatmap")
    if not df.empty:
        numeric_cols = df.select_dtypes(include=['number']).columns
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Purples")
        fig_corr.update_layout(**get_plotly_layout("Pearson Correlation", height=600))
        st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    st.markdown("### Interest Tags Analysis")
    st.markdown("Frequency of the various interest tags chosen by users.")
    plot_path_tags = get_plot_path("08_3_10_interest_tags_analysis.png")
    if os.path.exists(plot_path_tags):
        st.image(plot_path_tags, use_container_width=True)
        
    st.markdown("#### Interactive View (Top 10)")
    if 'top_interest_tags' in eda_stats:
        top_tags = eda_stats['top_interest_tags']
        fig_tags = px.bar(x=list(top_tags.values()), y=list(top_tags.keys()), orientation='h', color_discrete_sequence=[PLOTLY_COLORS[2]])
        fig_tags.update_layout(**get_plotly_layout("Top Interest Tags", height=400))
        fig_tags.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_tags, use_container_width=True)
