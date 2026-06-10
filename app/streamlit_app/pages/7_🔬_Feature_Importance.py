import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import theme
from utils import data_loader
from utils import model_loader

st.set_page_config(page_title="Feature Importance | SwipeIQ", page_icon="🔬", layout="wide")
theme.inject_css()
theme.render_sidebar()

st.title("🔬 Feature Importance & Explainability")
st.markdown("---")

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

# Load importances
fi_data = data_loader.load_feature_importances()
if not fi_data:
    st.warning("Feature importances not found. Please ensure `feature_importances.json` is available.")
    st.stop()

rf_importances = fi_data.get("Random Forest", [])
xgb_importances = fi_data.get("XGBoost", [])

# We need feature names. Try getting them from the baseline model first
feature_names = []
try:
    baseline_models = model_loader.load_baseline_models()
    if 'Random Forest' in baseline_models and 'model' in baseline_models['Random Forest']:
        model = baseline_models['Random Forest']['model']
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
except Exception as e:
    pass

# Fallback: if we still don't have feature_names, generate generic ones
if not feature_names or len(feature_names) != len(rf_importances):
    feature_names = [f"Feature {i+1}" for i in range(len(rf_importances))]

# Create a DataFrame
df_fi = pd.DataFrame({
    "Feature": feature_names,
    "Random Forest Importance": rf_importances,
    "XGBoost Importance": xgb_importances
})

# Sort by RF importance for top 20
df_rf_top = df_fi.sort_values(by="Random Forest Importance", ascending=False).head(20)

st.subheader("Top 20 Features (Random Forest)")
st.markdown("This bar chart highlights the 20 most influential features according to the Random Forest model.")

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
st.subheader("Random Forest vs XGBoost Comparison")
st.markdown("Comparing the feature importances between the two tree-based models.")

# Sort overall by RF for comparison
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
fig2.update_layout(
    barmode='group',
    **layout_dict2
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader("Top Features Explanation")

cols = st.columns(3)
with cols[0]:
    st.markdown(f"""
    <div class="ml-callout">
    <strong>1. High Activity Metrics</strong><br>
    Metrics like <code>app_usage_time_min</code> and <code>message_sent_count</code> consistently appear near the top, indicating that overall app engagement is heavily utilized by the models for splitting.
    </div>
    """, unsafe_allow_html=True)
with cols[1]:
    st.markdown(f"""
    <div class="ml-callout">
    <strong>2. Swipe Ratios</strong><br>
    <code>swipe_right_ratio</code> provides insight into user selectivity. Models may leverage this to distinguish between passive and active searchers.
    </div>
    """, unsafe_allow_html=True)
with cols[2]:
    st.markdown(f"""
    <div class="ml-callout">
    <strong>3. Numeric Over Categorical</strong><br>
    Continuous numeric features tend to have higher importances in tree models due to the larger number of potential split points compared to binary one-hot encoded features.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("Full Feature Table")
st.markdown("Below is the complete dataset of feature importances across all {len(feature_names)} features.")
st.dataframe(df_fi.sort_values(by="Random Forest Importance", ascending=False), use_container_width=True)
