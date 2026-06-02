import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from utils.data_loader import load_raw_data, get_preprocessed_data
from utils.theme import inject_css, render_sidebar, get_plotly_layout, PURPLE, PINK, TEAL, AMBER, GREEN, RED, SKY, BG_CARD

st.set_page_config(page_title="Preprocessing | SwipeIQ", page_icon="⚙️", layout="wide")
inject_css()
render_sidebar()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V5_PLOTS = os.path.join(ROOT_DIR, 'assets', 'v5_plots')
BASE_PLOTS = os.path.join(ROOT_DIR, 'assets', 'plots')

def show_plot(directory, filename, caption=''):
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f'Plot not found: {filename}')

st.title("⚙️ Preprocessing Pipeline")
st.markdown("Transforming the raw dataset into a machine-readable format. Our goal is to handle categorical variables, normalize numerical features, and prepare a clean binary target for the model to predict.")

# Load raw and preprocessed data to show shapes
df_raw = load_raw_data()
df_prep, y_prep, feature_names, scaler = get_preprocessed_data()

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.metric("Raw Dataset Shape", f"{df_raw.shape[0]:,} × {df_raw.shape[1]}")
with col2:
    if df_prep is not None:
        st.metric("Preprocessed Features", f"{df_prep.shape[0]:,} × {len(feature_names)}")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── Step 1: Drop Redundant ──────────────────────────────────────────────────
st.markdown("""
<div class="pipeline-step">
    <h4>Step 1: Drop Redundant Columns</h4>
    <p>Columns like <code>app_usage_time_label</code> (a string version of <code>app_usage_time_min</code>) and <code>swipe_right_label</code> are redundant and provide no extra information. We remove them.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("Show Python Snippet: Drop Columns"):
    st.code("""
cols_to_drop = ['app_usage_time_label', 'swipe_right_label']
df.drop(columns=cols_to_drop, inplace=True)
    """, language="python")

# ── Step 2: Binary Target ───────────────────────────────────────────────────
st.markdown("""
<div class="pipeline-step">
    <h4>Step 2: Binary Target Conversion</h4>
    <p>The original target <code>match_outcome</code> has 10 classes. We map them into a binary classification problem (1 = Meaningful Connection, 0 = No Connection).</p>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.success("**Positive Outcomes (1)**\n- Mutual Match\n- Instant Match\n- Date Happened\n- Relationship Formed")
with c2:
    st.error("**Negative Outcomes (0)**\n- No Match\n- Ghosted\n- Unmatched\n- Blocked\n- One-sided Like\n- Ignored")

# ── Step 2.5: Advanced Feature Engineering [V8] ────────────────────────────
st.markdown("""
<div class="pipeline-step">
    <h4>Step 2.5: Advanced Feature Engineering <span style="color:#8b5cf6;">[V8]</span></h4>
    <p>Beyond basic encoding, V8 introduces domain-driven <strong>engineered features</strong> that capture higher-order behavioral signals not present in the raw columns, along with crucial adjustments to prevent target leakage.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(139,92,246,0.06); border:1px dashed rgba(139,92,246,0.3); border-radius:8px; padding:16px; font-size:13px; color:#c4b5fd; line-height:1.5; margin-bottom: 24px;">
    <strong>🧪 Engineered Features Added:</strong><br>
    • <code>engagement_score</code> — composite metric combining likes received, swipe right ratio, and messages sent into a single proxy.<br>
    • <code>profile_completeness</code> — product of profile pics count and bio length.<br>
    • <code>activity_intensity</code> — combination of app usage time and emoji usage rate.<br>
    • <code>selectivity_ratio</code> — ratio of messages sent to likes received (adjusted to prevent target leakage from `mutual_matches`).<br>
    • <code>late_night_user</code> — binary indicator for users active between 10 PM and 4 AM.<br>
    • <strong>Log-transforms</strong> — applied to right-skewed distributions (e.g., <code>log1p(message_sent_count)</code>) to reduce the influence of extreme values.
</div>
""", unsafe_allow_html=True)

with st.expander("Show Python Snippet: Feature Engineering"):
    st.code("""
# Engagement score (composite)
df['engagement_score'] = df['likes_received'] * df['swipe_right_ratio'] * df['message_sent_count']

# Profile completeness & Activity Intensity
df['profile_completeness'] = df['profile_pics_count'] * df['bio_length']
df['activity_intensity'] = df['app_usage_time_min'] * df['emoji_usage_rate']

# Selectivity ratio
df['selectivity_ratio'] = df['message_sent_count'] / (df['likes_received'] + 1)

# Late night user flag
df['late_night_user'] = ((df['last_active_hour'] >= 22) | (df['last_active_hour'] <= 4)).astype(int)

# Log-transform skewed features
for col in ['likes_received', 'message_sent_count', 'bio_length', 'app_usage_time_min']:
    df[f'{col}_log'] = np.log1p(df[col])
    """, language="python")

# ── Step 3: Ordinal Encoding ────────────────────────────────────────────────
st.markdown("""
<div class="pipeline-step">
    <h4>Step 3: Ordinal Encoding</h4>
    <p>Variables with a natural order (like Income and Education) are encoded as integers. This preserves their ordinal relationship.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("**Income Encoding:** `Very Low/Low` → 0, `Lower-Middle/Middle/Upper-Middle` → 1, `High/Very High` → 2")
st.markdown("**Education Encoding:** `High School/Diploma` → 0, `Bachelor/Associate` → 1, `Master/PhD` → 2")

with st.expander("Show Python Snippet: Ordinal Encoding"):
    st.code("""
income_map = {'Very Low': 0, 'Low': 0, 'Middle': 1, 'High': 2, ...}
df['income_bracket'] = df['income_bracket'].map(income_map)
    """, language="python")

# ── Step 4: One-Hot Encoding ────────────────────────────────────────────────
st.markdown("""
<div class="pipeline-step">
    <h4>Step 4: One-Hot Encoding</h4>
    <p>We apply one-hot encoding to 7 nominal columns (e.g., gender, orientation, zodiac) that have no inherent order. This creates binary columns for each category.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("Show Example Output (Zodiac Sign)"):
    st.dataframe(pd.DataFrame({
        'zodiac_sign_Aries': [1, 0, 0],
        'zodiac_sign_Taurus': [0, 1, 0],
        'zodiac_sign_Gemini': [0, 0, 1]
    }), use_container_width=True)

# ── Step 5: Multi-Hot Encoding ──────────────────────────────────────────────
st.markdown("""
<div class="pipeline-step">
    <h4>Step 5: Multi-Hot Encoding (Interest Tags)</h4>
    <p>The <code>interest_tags</code> column contains comma-separated lists (e.g., "Fitness, Gaming"). We create 49 separate binary columns representing presence (1) or absence (0) of each specific tag.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("Show Example Output (Interests)"):
    st.dataframe(pd.DataFrame({
        'interest_Fitness': [1, 1, 0],
        'interest_Gaming': [1, 0, 1],
        'interest_Cooking': [0, 0, 0]
    }), use_container_width=True)

# ── Step 5.5: RobustScaler [V8] ────────────────────────────────────────────
st.markdown("""
<div class="pipeline-step">
    <h4>Step 5.5: RobustScaler <span style="color:#8b5cf6;">[V8]</span></h4>
    <p>V8 replaces the original <code>StandardScaler</code> with <code>RobustScaler</code> for all numerical features. Unlike StandardScaler (which uses mean/std), RobustScaler uses the <strong>median and interquartile range (IQR)</strong>, making it inherently resistant to outliers.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(20,184,166,0.06); border:1px dashed rgba(20,184,166,0.3); border-radius:8px; padding:16px; font-size:13px; color:#5eead4; line-height:1.5; margin-bottom: 24px;">
    <strong>🛡️ Why RobustScaler over StandardScaler?</strong><br>
    • <strong>StandardScaler</strong>: centers on mean, scales by std — a single extreme value can shift all scaled values.<br>
    • <strong>RobustScaler</strong>: centers on median, scales by IQR — outliers have minimal effect on the scaling parameters.<br>
    • This is especially important for features like <code>app_usage_time_min</code> and <code>message_sent_count</code> that may exhibit heavy tails after feature engineering.
</div>
""", unsafe_allow_html=True)

with st.expander("Show Python Snippet: RobustScaler"):
    st.code("""
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    """, language="python")

# ── Step 6: Feature Scaling (Legacy) ────────────────────────────────────────
st.markdown("""
<div class="pipeline-step">
    <h4>Step 6: Feature Scaling (StandardScaler — V1-V3 Legacy)</h4>
    <p>In V1-V3, we applied <code>StandardScaler</code> to the 12 numerical features so they have a mean of 0 and a standard deviation of 1. This is crucial for distance-based models like KNN and SVM. <em>Superseded by RobustScaler in V4+.</em></p>
</div>
""", unsafe_allow_html=True)

with st.expander("Show Python Snippet: StandardScaler (Legacy)"):
    st.code("""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    """, language="python")

# ── Step 7: OOD Rejection Guardrail [V5] ───────────────────────────────────
st.markdown("""
<div class="pipeline-step">
    <h4>Step 7: OOD Rejection Guardrail <span style="color:#ec4899;">[V5]</span></h4>
    <p>V5 introduces an <strong>Out-of-Distribution (OOD) Rejection</strong> layer using an <strong>Isolation Forest</strong> anomaly detector trained on the preprocessed feature matrix. Samples flagged as anomalies (contamination = 5%) are rejected before reaching the classifier, preventing the model from making unreliable predictions on data it has never seen.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(236,72,153,0.06); border:1px dashed rgba(236,72,153,0.3); border-radius:8px; padding:16px; font-size:13px; color:#f472b6; line-height:1.5; margin-bottom: 24px;">
    <strong>🚧 How Isolation Forest Works:</strong><br>
    • Isolation Forest isolates anomalies by randomly selecting a feature and then randomly selecting a split value between the max and min of the selected feature.<br>
    • Anomalies require <strong>fewer splits</strong> to be isolated (shorter path length), making detection efficient and scalable.<br>
    • With <code>contamination=0.05</code>, the top 5% most anomalous samples are filtered out, ensuring the downstream classifier trains and predicts only on in-distribution data.
</div>
""", unsafe_allow_html=True)

show_plot(V5_PLOTS, "11_flex_11_out_of_distribution_ood_rejection_guardrai.png",
          caption="OOD Rejection Guardrail — Isolation Forest Anomaly Scores")

with st.expander("Show Python Snippet: Isolation Forest OOD"):
    st.code("""
from sklearn.ensemble import IsolationForest
 
iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
ood_labels = iso_forest.fit_predict(X_train)

# Keep only inliers (label == 1)
X_train_clean = X_train[ood_labels == 1]
y_train_clean = y_train[ood_labels == 1]
    """, language="python")


# ── Step 8: Scaling & Robustness Playground [V5.1+] ─────────────────────────
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.header("🧪 Interactive Scaling & Outlier Robustness Playground")
st.markdown("""
Select a continuous numerical feature and a scaling strategy to visualize how outlier values shift distributions!
Toggling the **"Inject Outliers"** checkbox shows why our V4 pipeline upgraded from **StandardScaler** to **RobustScaler** to ensure model stability under real-world data noise.
""")

if not df_raw.empty:
    # Playground Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    
    numeric_features = [
        'likes_received', 'app_usage_time_min', 'message_sent_count',
        'age', 'height_cm', 'weight_kg', 'swipe_right_ratio', 'activity_intensity',
        'profile_pics_count', 'bio_length', 'emoji_usage_rate', 'last_active_hour',
        'engagement_score', 'selectivity_ratio', 'profile_completeness'
    ]
    
    with col_ctrl1:
        selected_feature = st.selectbox("Numerical Feature", numeric_features, index=0) # Default likes_received
        
    with col_ctrl2:
        selected_scaler_name = st.selectbox("Scaling Strategy", ["RobustScaler [V8]", "StandardScaler [Legacy]", "MinMaxScaler"], index=0)
        
    with col_ctrl3:
        inject_outliers = st.checkbox("Inject Extreme Outliers", value=False, help="Adds synthetic anomalous outliers to simulate real-world logging noise")

    # Extra slider if outliers are active
    outlier_pct = 2
    outlier_magnitude = 10
    if inject_outliers:
        col_out1, col_out2 = st.columns(2)
        with col_out1:
            outlier_pct = st.slider("Outlier Contamination Ratio (%)", min_value=1, max_value=10, value=3, step=1)
        with col_out2:
            outlier_magnitude = st.slider("Outlier Magnitude (Multiplier)", min_value=2, max_value=50, value=15, step=1,
                                          help="Multiplies maximum values to generate extreme scale skewness")
            
    # Prepare Data
    raw_series = df_raw[selected_feature].dropna().copy()
    data_df = pd.DataFrame({"Original": raw_series})
    
    # Inject outliers if toggled
    if inject_outliers:
        n_outliers = int(len(data_df) * (outlier_pct / 100))
        random_indices = np.random.default_rng(42).choice(data_df.index, size=n_outliers, replace=False)
        max_val = data_df["Original"].max()
        # Set extreme outliers
        data_df.loc[random_indices, "Original"] = max_val * outlier_magnitude

    # Fit scaler
    raw_values = data_df["Original"].values.reshape(-1, 1)
    
    if selected_scaler_name == "RobustScaler [V8]":
        scaler_obj = RobustScaler()
        scaled_values = scaler_obj.fit_transform(raw_values).flatten()
    elif selected_scaler_name == "StandardScaler [Legacy]":
        scaler_obj = StandardScaler()
        scaled_values = scaler_obj.fit_transform(raw_values).flatten()
    else:
        scaler_obj = MinMaxScaler()
        scaled_values = scaler_obj.fit_transform(raw_values).flatten()
        
    data_df["Scaled"] = scaled_values

    # Render Side-by-Side Plots
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        # Original histogram
        fig_raw = px.histogram(data_df, x="Original", nbins=50, color_discrete_sequence=[PINK])
        fig_raw.update_layout(**get_plotly_layout(f"Raw Input: {selected_feature} Distribution", height=320))
        st.plotly_chart(fig_raw, use_container_width=True)
        
        # Original stats
        orig_mean = data_df["Original"].mean()
        orig_median = data_df["Original"].median()
        orig_std = data_df["Original"].std()
        orig_iqr = data_df["Original"].quantile(0.75) - data_df["Original"].quantile(0.25)
        st.markdown(f"""
        <div style="background:{BG_CARD}; padding:14px; border:1px solid rgba(255,255,255,0.05); border-radius:8px; font-size:12px;">
            📝 <strong>Original Stats:</strong><br>
            • Mean: <code>{orig_mean:.2f}</code> | Median: <code>{orig_median:.2f}</code><br>
            • Std Dev: <code>{orig_std:.2f}</code> | IQR (Interquartile): <code>{orig_iqr:.2f}</code>
        </div>
        """, unsafe_allow_html=True)
        
    with plot_col2:
        # Scaled histogram
        fig_scaled = px.histogram(data_df, x="Scaled", nbins=50, color_discrete_sequence=[TEAL])
        fig_scaled.update_layout(**get_plotly_layout(f"Scaled output ({selected_scaler_name})", height=320))
        st.plotly_chart(fig_scaled, use_container_width=True)
        
        # Scaled stats
        scaled_mean = data_df["Scaled"].mean()
        scaled_median = data_df["Scaled"].median()
        scaled_std = data_df["Scaled"].std()
        scaled_iqr = data_df["Scaled"].quantile(0.75) - data_df["Scaled"].quantile(0.25)
        st.markdown(f"""
        <div style="background:{BG_CARD}; padding:14px; border:1px solid rgba(255,255,255,0.05); border-radius:8px; font-size:12px;">
            📝 <strong>Scaled Output Stats:</strong><br>
            • Mean: <code>{scaled_mean:.4f}</code> | Median: <code>{scaled_median:.4f}</code><br>
            • Std Dev: <code>{scaled_std:.4f}</code> | IQR (Interquartile): <code>{scaled_iqr:.4f}</code>
        </div>
        """, unsafe_allow_html=True)

    # Scientific Insights Callout
    st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
    if inject_outliers:
        if "StandardScaler" in selected_scaler_name:
            st.markdown(f"""
            <div style="background:rgba(239,68,68,0.05); border-left:4px solid {RED}; border-radius:4px; padding:12px; font-size:13px; color:#f87171;">
                ⚠️ <strong>StandardScaler Failure Analysis:</strong> Notice how the extreme outlier values ({outlier_magnitude}x) 
                have inflated the standard deviation parameter. Consequently, the scaled bounds of the <em>normal data points</em> 
                have been heavily compressed into a tight, uninformative spike between <code>-{3/outlier_magnitude:.3f}</code> and 
                <code>{3/outlier_magnitude:.3f}</code> near 0! The model is forced to split on extremely squeezed ranges, drastically hurting convergence.
            </div>
            """, unsafe_allow_html=True)
        elif "RobustScaler" in selected_scaler_name:
            st.markdown(f"""
            <div style="background:rgba(16,185,129,0.05); border-left:4px solid {GREEN}; border-radius:4px; padding:12px; font-size:13px; color:#6ee7b7;">
                ✅ <strong>RobustScaler Success Analysis:</strong> Excellent! Even under high outlier contamination ({outlier_pct}% at {outlier_magnitude}x), 
                RobustScaler relies solely on median and IQR. The bulk of normal features remain beautifully spread across a wide, clean range 
                from <code>-2.0</code> to <code>+2.0</code>. The anomalous outliers are cleanly isolated in the tail without compressing the valid signal. 
                This directly translates to higher validation stability and more robust gradient updates!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(245,158,11,0.05); border-left:4px solid {AMBER}; border-radius:4px; padding:12px; font-size:13px; color:#fcd34d;">
                ⚡ <strong>MinMaxScaler Failure Analysis:</strong> Because MinMaxScaler is bounded explicitly between <code>[0.0, 1.0]</code>, 
                forcing the outliers to equal exactly <code>1.0</code> compresses all valid, normal data points into a narrow range 
                between <code>0.0</code> and <code>{1/outlier_magnitude:.3f}</code>. Any distance-based models (like SVM or KNN) will be severely 
                desensitized to standard feature variations!
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.05); border-left:4px solid {PURPLE}; border-radius:4px; padding:12px; font-size:13px; color:#c4b5fd;">
            💡 <strong>Playground Tip:</strong> Check the **"Inject Extreme Outliers"** box and swap between **RobustScaler** and **StandardScaler** 
            to dynamically witness how standard deviations expand under anomalies, causing StandardScaler to compress non-outlier data, 
            while RobustScaler preserves normal features seamlessly.
        </div>
        """, unsafe_allow_html=True)

else:
    st.warning("Raw dataset not loaded. Feature Scaling Sandbox is unavailable.")


# ── Final Preprocessed Data ─────────────────────────────────────────────────
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown("### 🏆 Final Preprocessed Data")
st.markdown("Here's a peek at the final preprocessed feature matrix fed into our models:")
if df_prep is not None:
    st.dataframe(df_prep.head(100), use_container_width=True)
else:
    st.error("Failed to load preprocessed data.")


# ── Render Global Footer ──
from utils.theme import render_footer
render_footer()
