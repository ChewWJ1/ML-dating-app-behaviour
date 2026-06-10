import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from utils import theme
from utils import data_loader
from utils import model_loader

st.set_page_config(page_title="Feature Importance | SwipeIQ", page_icon="🔬", layout="wide")
theme.inject_css()
theme.render_sidebar()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
V8_PLOTS = os.path.join(ROOT_DIR, 'assets', 'v8 plots')
BASE_PLOTS = os.path.join(ROOT_DIR, 'assets', 'plots')

def show_plot(directory, filename, caption=''):
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f'Plot not found: {filename}')

st.title("🔬 Feature Importance & Explainability")
st.markdown("---")
st.image(os.path.join(ROOT_DIR, "assets", "New NotebookLM", "Section overview", "SHAP_Explainability_Model_Analysis.png"), use_container_width=True)

st.markdown("""
<div style="background:rgba(239,68,68,0.06); border:1px dashed rgba(239,68,68,0.3); border-radius:8px; padding:16px; font-size:13px; color:#f87171; line-height:1.5; margin-bottom: 24px;">
    <strong>🔎 SHAP Game-Theoretic Explainability:</strong><br>
    Standard impurity feature importances only measure split magnitude, lacking directionality. To deeply inspect feature attributions, we deployed a SHAP (Shapley Additive exPlanations) TreeExplainer on the best model. Our SHAP analysis revealed that the absolute feature attribution margins for <em>all</em> predictors are under 0.02. This confirms that the model's split decisions are forced to rely on low-level statistical noise rather than genuine predictive signals, proving the lack of any single dominant predictor in the dataset.
</div>
""", unsafe_allow_html=True)

st.markdown("""
### What drives the models' decisions?
Feature importance helps us understand which characteristics the model relies on most when predicting a match outcome. 
For tree-based models like **Random Forest** and **XGBoost**, importance is calculated based on how much each feature decreases impurity (or increases information gain) across all trees.
""")

st.info("💡 **Methodology Note:** With ROC-AUC ≈ 0.50, feature importances reflect which features the model *uses most for splitting*, not necessarily which features are truly predictive. In synthetic data, all importances are expected to be roughly equal.")

# ── Feature Importance Analysis Plot ────────────────────────────────────────
st.markdown('<div class="section-label">IMPURITY-BASED IMPORTANCE</div>', unsafe_allow_html=True)
show_plot(V8_PLOTS, "28_feature_importance.png",
          caption="Feature Importance Analysis — Tree-Based Models")

st.markdown("---")

# Load importances for interactive charts
fi_data = data_loader.load_feature_importances()
if fi_data:
    rf_importances = fi_data.get("Random Forest", [])
    xgb_importances = fi_data.get("XGBoost", [])

    # We need feature names
    feature_names = []
    try:
        baseline_models = model_loader.load_baseline_models()
        if 'Random Forest' in baseline_models and 'model' in baseline_models['Random Forest']:
            model = baseline_models['Random Forest']['model']
            if hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
    except Exception:
        pass

    # Fallback: generic names
    if not feature_names or len(feature_names) != len(rf_importances):
        feature_names = [f"Feature {i+1}" for i in range(len(rf_importances))]

    # Create DataFrame
    df_fi = pd.DataFrame({
        "Feature": feature_names,
        "Random Forest Importance": rf_importances,
        "XGBoost Importance": xgb_importances
    })

    # Top 20 RF
    df_rf_top = df_fi.sort_values(by="Random Forest Importance", ascending=False).head(20)

    st.subheader("Interactive: Top 20 Features (Random Forest)")
    fig1 = px.bar(
        df_rf_top,
        x="Random Forest Importance",
        y="Feature",
        orientation="h",
        color="Random Forest Importance",
        color_continuous_scale="Purples"
    )
    layout_dict1 = theme.get_plotly_layout("Random Forest Top 20 Features")
    layout_dict1['yaxis']['categoryorder'] = 'total ascending'
    fig1.update_layout(**layout_dict1)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")
    st.subheader("Interactive: Random Forest vs XGBoost Comparison")

    df_fi_sorted = df_fi.sort_values(by="Random Forest Importance", ascending=False).head(20)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        y=df_fi_sorted["Feature"],
        x=df_fi_sorted["Random Forest Importance"],
        name="Random Forest",
        orientation='h',
        marker=dict(color=theme.PURPLE)
    ))
    fig2.add_trace(go.Bar(
        y=df_fi_sorted["Feature"],
        x=df_fi_sorted["XGBoost Importance"],
        name="XGBoost",
        orientation='h',
        marker=dict(color=theme.TEAL)
    ))

    layout_dict2 = theme.get_plotly_layout("RF vs XGBoost Importance Comparison (Top 20)")
    layout_dict2['yaxis']['categoryorder'] = 'total ascending'
    fig2.update_layout(barmode='group', **layout_dict2)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Top Features Explanation")

    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        <div class="ml-callout">
        <strong>1. High Activity Metrics</strong><br>
        Metrics like <code>app_usage_time_min</code> and <code>message_sent_count</code> consistently appear near the top, indicating that overall app engagement is heavily utilized by the models for splitting.
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown("""
        <div class="ml-callout">
        <strong>2. Swipe Ratios</strong><br>
        <code>swipe_right_ratio</code> provides insight into user selectivity. Models may leverage this to distinguish between passive and active searchers.
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown("""
        <div class="ml-callout">
        <strong>3. Numeric Over Categorical</strong><br>
        Continuous numeric features tend to have higher importances in tree models due to the larger number of potential split points compared to binary one-hot encoded features.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Full Feature Table")
    st.markdown(f"Below is the complete dataset of feature importances across all {len(feature_names)} features.")
    st.dataframe(df_fi.sort_values(by="Random Forest Importance", ascending=False), use_container_width=True)

else:
    st.warning("Feature importances not found. Please ensure `feature_importances.json` is available. Showing static plot instead.")

# ── Permutation Feature Interaction (H-Statistic) [V4] ─────────────────────
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">FEATURE INTERACTIONS [V4]</div>', unsafe_allow_html=True)
st.subheader("Permutation Feature Interaction (H-Statistic)")

st.markdown("""
<div style="background:rgba(245,158,11,0.06); border:1px dashed rgba(245,158,11,0.3); border-radius:8px; padding:16px; font-size:13px; color:#fcd34d; line-height:1.5; margin-bottom: 24px;">
    <strong>📊 Friedman's H-Statistic — Second-Order Interactions:</strong><br>
    Standard feature importance methods measure <em>marginal</em> contributions of individual features. But what about
    <strong>synergistic effects</strong> — pairs of features whose combined effect is greater (or lesser) than the sum of their
    individual contributions?<br><br>
    <strong>Friedman's H-statistic</strong> quantifies the strength of second-order interactions between feature pairs:
    <ul style="margin-top:8px;">
        <li><strong>H = 0</strong>: no interaction — the joint effect equals the sum of individual partial dependence functions.</li>
        <li><strong>H → 1</strong>: strong interaction — the features' combined effect deviates significantly from additivity.</li>
    </ul>
    We compute H-statistics for the top-20 feature pairs using permutation-based partial dependence, identifying which behavioral
    dimensions interact non-additively in predicting match outcomes.
</div>
""", unsafe_allow_html=True)

show_plot(V8_PLOTS, "feature_interactions.png",
          "Permutation Feature Interaction Detection — Top Interacting Pairs")

# ── SHAP Interaction Values [V5] ───────────────────────────────────────────
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">SHAP INTERACTIONS [V5]</div>', unsafe_allow_html=True)
st.subheader("SHAP Interaction Values")

st.markdown("""
<div style="background:rgba(99,102,241,0.06); border:1px dashed rgba(99,102,241,0.3); border-radius:8px; padding:16px; font-size:13px; color:#a5b4fc; line-height:1.5; margin-bottom: 24px;">
    <strong>🧬 Shapley Interaction Index — Joint Pair Attributions:</strong><br>
    While Friedman's H-statistic detects the <em>presence</em> of interactions, <strong>SHAP Interaction Values</strong> go further
    by decomposing each prediction into <strong>main effects + pairwise interaction effects</strong> using the Shapley Interaction Index.<br><br>
    For each sample, the SHAP interaction matrix Φᵢⱼ satisfies:
    <code>f(x) = E[f(X)] + Σᵢ Φᵢᵢ + Σᵢ≠ⱼ Φᵢⱼ</code><br><br>
    • <strong>Diagonal elements Φᵢᵢ</strong>: main effects (standard SHAP values minus interaction terms).<br>
    • <strong>Off-diagonal elements Φᵢⱼ</strong>: the synergistic attribution of feature pair (i, j) — how much the <em>combination</em>
    of features i and j contributes beyond their individual effects.<br><br>
    This produces a full N×N interaction matrix, revealing which feature pairs create emergent predictive power.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    show_plot(V8_PLOTS, "31_shap_dependence_interaction_plot.png",
              caption="SHAP Interaction Values — Synergistic Attribution (1)")
with col2:
    show_plot(V8_PLOTS, "32_shap_interaction_matrix_heatmap.png",
              caption="SHAP Interaction Values — Synergistic Attribution (2)")

st.markdown("""
<div style="background:rgba(16,185,129,0.06); border:1px dashed rgba(16,185,129,0.3); border-radius:8px; padding:16px; font-size:13px; color:#6ee7b7; line-height:1.5; margin-bottom: 24px;">
    <strong>✅ Key Insight:</strong> The SHAP interaction matrices confirm that pairwise interaction effects (Φᵢⱼ) are negligible
    across all feature pairs — consistent with the dataset's synthetic, independently generated feature structure. In real-world
    dating data, we would expect meaningful interactions (e.g., age × preferred_age_range synergy).
</div>
""", unsafe_allow_html=True)


# ── Render Global Footer ──
from utils.theme import render_footer
render_footer()
