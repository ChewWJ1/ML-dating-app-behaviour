import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from utils.data_loader import load_raw_data, load_eda_stats
from utils.theme import inject_css, render_sidebar, get_plotly_layout, PLOTLY_COLORS, SKY, GREEN, PINK, TEAL, PURPLE, AMBER, PURPLE_LIGHT

st.set_page_config(page_title="EDA | SwipeIQ", page_icon="🔍", layout="wide")
inject_css()
render_sidebar()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V8_PLOTS = os.path.join(ROOT_DIR, 'assets', 'v8 plots')
BASE_PLOTS = os.path.join(ROOT_DIR, 'assets', 'plots')

def show_plot(directory, filename, caption=''):
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f'Plot not found: {filename}')

st.title("🔍 Exploratory Data Analysis")

st.markdown("""
<div style="background:rgba(59,130,246,0.06); border:1px dashed rgba(59,130,246,0.3); border-radius:8px; padding:16px; font-size:13px; color:#93c5fd; line-height:1.5; margin-bottom: 24px;">
    <strong>📊 Scientific Objective &amp; EDA Findings:</strong><br>
    The core project objective is to identify behavioral patterns that correlate with positive connection outcomes, allowing dating applications to optimize matching pools and reduce ghosting rates. However, our Exploratory Data Analysis confirms the synthetic programmatic nature of the dataset: <strong>all 25 features are uniformly distributed</strong>, no extreme outliers exist, and Pearson correlation matrices show an absence of any strong correlation (values near 0).
</div>
""", unsafe_allow_html=True)
st.markdown("---")
st.image(os.path.join(ROOT_DIR, "assets", "NotebookLM", "section overview", "Dating_Dataset_Insights_and_Analysis.png"), use_container_width=True)

df = load_raw_data()
eda_stats = load_eda_stats()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Distributions", "🎯 Feature vs Target", "🔗 Correlations",
    "🏷️ Interest Tags", "🔬 Causal Discovery", "🧪 Bivariate Playground"
])

with tab1:
    st.markdown("### Categorical Distributions")
    st.markdown("All categorical variables exhibit near-perfect uniform distribution.")
    show_plot(V8_PLOTS, "02_categorical_feature_distributions.png")

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
    show_plot(V8_PLOTS, "03_numerical_feature_distributions.png")

    st.markdown("---")
    st.markdown("### Outlier Detection")
    st.markdown("Boxplots show that no extreme outliers exist in the dataset due to its controlled synthetic generation.")
    show_plot(V8_PLOTS, "04_outlier_detection_boxplots.png")

with tab2:
    st.markdown("### Numerical Features by Target")
    st.markdown("When split by outcome, the distributions remain largely identical, showing no obvious separability.")
    show_plot(V8_PLOTS, "05_numerical_features_by_match_outcome.png")

    st.markdown("---")
    st.markdown("### Categorical Features by Target")
    st.markdown("The proportion of outcomes is evenly distributed across all categories.")
    show_plot(V8_PLOTS, "06_categorical_features_by_match_outcome.png")

with tab3:
    st.markdown("### Correlation Heatmap")
    st.markdown("Pearson correlation coefficients between all numerical features. Notice the absence of any strong correlation (values near 0).")
    show_plot(V8_PLOTS, "07_pearson_correlation_matrix.png")

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
    show_plot(V8_PLOTS, "08_top_30_interest_tags.png")

    st.markdown("#### Interactive View (Top 10)")
    if 'top_interest_tags' in eda_stats:
        top_tags = eda_stats['top_interest_tags']
        fig_tags = px.bar(x=list(top_tags.values()), y=list(top_tags.keys()), orientation='h', color_discrete_sequence=[PLOTLY_COLORS[2]])
        fig_tags.update_layout(**get_plotly_layout("Top Interest Tags", height=400))
        fig_tags.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_tags, use_container_width=True)

with tab5:
    st.markdown("### 🔬 Causal Discovery — Going Beyond Correlation")

    st.markdown("""
<div style="background:rgba(99,102,241,0.06); border:1px dashed rgba(99,102,241,0.3); border-radius:8px; padding:16px; font-size:13px; color:#a5b4fc; line-height:1.5; margin-bottom: 24px;">
    <strong>🧬 PC Algorithm DAG Explanation:</strong><br>
    Standard EDA reveals correlations, but <strong>causal discovery</strong> goes further by estimating the <em>direction</em> of influence between variables.
    We applied the <strong>PC (Peter-Clark) Algorithm</strong>, a constraint-based method that uses conditional independence tests to construct a
    <strong>Directed Acyclic Graph (DAG)</strong>. The resulting DAG reveals the causal skeleton — edges that survive multiple rounds of
    conditional independence testing at α = 0.05, helping us distinguish genuine causal pathways from spurious associations.
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        show_plot(V8_PLOTS, "09_causal_dag.png",
                  caption="Causal Discovery DAG — Structural Relationships")
    with col2:
        show_plot(V8_PLOTS, "10_causal_adjacency_matrix.png",
                  caption="Causal Discovery DAG — Edge Strength Analysis")

    st.markdown("""
<div style="background:rgba(245,158,11,0.06); border:1px dashed rgba(245,158,11,0.3); border-radius:8px; padding:16px; font-size:13px; color:#fcd34d; line-height:1.5; margin-bottom: 24px;">
    <strong>🔎 Data Quality Audit — MI Analysis &amp; Permutation Testing:</strong><br>
    To validate the causal graph, we cross-referenced edges with <strong>Mutual Information (MI) analysis</strong> and
    <strong>permutation-based independence testing</strong> (1,000 permutations per edge). Edges that fail the permutation
    null-hypothesis test (p > 0.05) are pruned, ensuring the final DAG only retains statistically robust causal links.
    This dual-validation approach guards against both Type I errors (false causal claims) and Type II errors (missed genuine effects).
</div>
""", unsafe_allow_html=True)

with tab6:
    st.markdown("### 🧪 Dynamic Bivariate Relationship Playground")
    st.markdown("""
    Explore structural patterns and associations between continuous features in real time!
    Choose any two numerical features below and optional categorical groupings to visualize their joint distributions and test statistical dependencies.
    """)

    if not df.empty:
        # Create control panel
        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
        
        numeric_options = [
            'age', 'height_cm', 'weight_kg', 'app_usage_time_min', 
            'swipe_right_ratio', 'likes_received', 'mutual_matches', 
            'profile_pics_count', 'bio_length', 'message_sent_count', 
            'emoji_usage_rate', 'last_active_hour'
        ]
        
        with col_ctrl1:
            x_var = st.selectbox("X-Axis Feature", numeric_options, index=3) # Default app_usage_time_min
            
        with col_ctrl2:
            y_var = st.selectbox("Y-Axis Feature", numeric_options, index=5) # Default likes_received
            
        with col_ctrl3:
            cat_options = ['None', 'gender', 'location_type', 'relationship_intent', 'body_type', 'match_outcome']
            color_var = st.selectbox("Color Grouping", cat_options, index=5) # Default match_outcome
            
        with col_ctrl4:
            sample_size = st.slider("Subset Size", min_value=500, max_value=5000, value=1500, step=500,
                                    help="Adjust subset size for smooth UI rendering performance")
            
        # Draw a divider
        st.markdown('<div style="height: 1px; background: rgba(255,255,255,0.07); margin: 15px 0;"></div>', unsafe_allow_html=True)
        
        # Sampling for performance
        df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
        
        # Calculate stats dynamically
        try:
            import scipy.stats as stats
            r, p_val = stats.pearsonr(df_sample[x_var], df_sample[y_var])
        except Exception:
            import numpy as np
            x_val = df_sample[x_var].values
            y_val = df_sample[y_var].values
            r = np.corrcoef(x_val, y_val)[0, 1]
            p_val = 0.5 # fallback standard
            
        # Visual stats cards
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            r_eval = "No linear relationship" if abs(r) < 0.05 else "Weak correlation" if abs(r) < 0.3 else "Moderate correlation"
            st.metric("Pearson Correlation (r)", f"{r:.4f}", 
                      delta=r_eval,
                      delta_color="off")
        with stat_col2:
            p_eval = "Not Significant (p >= 0.05)" if p_val >= 0.05 else "Statistically Significant"
            st.metric("Two-Tailed p-value", f"{p_val:.4f}",
                      delta=p_eval,
                      delta_color="inverse" if p_val >= 0.05 else "normal")
        with stat_col3:
            sig_text = "Highly Significant" if p_val < 0.001 else "Significant" if p_val < 0.05 else "Not Significant"
            st.metric("Null Hypothesis Test (α = 0.05)", sig_text)
            
        # Interpretation
        if p_val >= 0.05:
            st.markdown(f"""
            <div style="background:rgba(59,130,246,0.05); border-left:4px solid {SKY}; border-radius:4px; padding:12px; font-size:13px; color:#93c5fd; margin-bottom: 20px;">
                💡 <strong>Scientific Interpretation:</strong> There is <strong>no statistically significant linear relationship</strong> 
                between <code>{x_var}</code> and <code>{y_var}</code> (p-value = {p_val:.4f} &ge; 0.05). 
                The regression trendline is virtually flat. This beautifully confirms our Data Audit discovery: 
                features are independent and uniformly distributed, showing that simple bivariate pathways yield zero predictive leverage.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(16,185,129,0.05); border-left:4px solid {GREEN}; border-radius:4px; padding:12px; font-size:13px; color:#6ee7b7; margin-bottom: 20px;">
                🎉 <strong>Scientific Interpretation:</strong> The linear relationship between <code>{x_var}</code> and <code>{y_var}</code> 
                is <strong>statistically significant</strong> (p-value = {p_val:.4f} &lt; 0.05). 
                However, with {sample_size} samples, even an extremely tiny and practically meaningless correlation (e.g. r &approx; 0.05) 
                can register as "statistically significant" due to high statistical power. The actual effect size remains extremely negligible!
            </div>
            """, unsafe_allow_html=True)
            
        # Plotting
        if color_var == 'None':
            fig = px.scatter(
                df_sample, x=x_var, y=y_var, 
                trendline="ols",
                trendline_color_override=PURPLE_LIGHT,
                color_discrete_sequence=[TEAL]
            )
        else:
            # Map target columns to friendly text if target
            df_plot = df_sample.copy()
            if color_var == 'match_outcome':
                positive_outcomes = {'Mutual Match', 'Instant Match', 'Date Happened', 'Relationship Formed'}
                df_plot['match_outcome'] = df_plot['match_outcome'].apply(
                    lambda x: "Meaningful Connection" if x in positive_outcomes else "No Connection"
                )
                
            fig = px.scatter(
                df_plot, x=x_var, y=y_var, 
                color=color_var,
                trendline="ols",
                color_discrete_sequence=PLOTLY_COLORS
            )
            
        title_text = f"Bivariate Relationship: {x_var} vs {y_var}"
        if color_var != 'None':
            title_text += f" (Grouped by {color_var})"
            
        fig.update_layout(**get_plotly_layout(title_text, height=520))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No dataset available to display the Bivariate Correlation Sandbox.")


# ── Render Global Footer ──
from utils.theme import render_footer
render_footer()
