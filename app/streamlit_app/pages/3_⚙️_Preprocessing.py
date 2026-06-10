import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_raw_data, get_preprocessed_data
from utils.theme import inject_css, render_sidebar, get_plotly_layout, PURPLE, PINK, TEAL, AMBER

st.set_page_config(page_title="Preprocessing | SwipeIQ", page_icon="⚙️", layout="wide")
inject_css()
render_sidebar()

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

# Step 1: Drop Redundant
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

# Step 2: Binary Target
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

# Step 3: Ordinal Encoding
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

# Step 4: One-Hot Encoding
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

# Step 5: Multi-Hot Encoding
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

# Step 6: StandardScaler
st.markdown("""
<div class="pipeline-step">
    <h4>Step 6: Feature Scaling</h4>
    <p>Finally, we apply <code>StandardScaler</code> to the 12 numerical features so they have a mean of 0 and a standard deviation of 1. This is crucial for distance-based models like KNN and SVM.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("Show Python Snippet: Scaling"):
    st.code("""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    """, language="python")

st.markdown("### 🏆 Final Preprocessed Data")
st.markdown("Here's a peek at the final preprocessed feature matrix fed into our models:")
if df_prep is not None:
    st.dataframe(df_prep.head(100), use_container_width=True)
else:
    st.error("Failed to load preprocessed data.")
