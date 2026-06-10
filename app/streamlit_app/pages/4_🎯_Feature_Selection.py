import os
import streamlit as st
from PIL import Image
from utils.theme import inject_css, render_sidebar

st.set_page_config(page_title="Feature Selection | SwipeIQ", page_icon="🎯", layout="wide")
inject_css()
render_sidebar()

# Base path for plots
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOTS_DIR = os.path.join(ROOT_DIR, 'assets', 'plots')

def load_image(filename):
    path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None

st.title("🎯 Feature Selection & PCA")

st.markdown("""
<div style="background:rgba(245,158,11,0.06); border:1px dashed rgba(245,158,11,0.3); border-radius:8px; padding:16px; font-size:13px; color:#fcd34d; line-height:1.5; margin-bottom: 24px;">
    <strong>🎯 Methodology Insights:</strong><br>
    To reduce compute time and eliminate noisy columns, we executed two feature ranking algorithms: ANOVA F-Score and Mutual Information. By taking the union of the top 40 features from each, we retained a streamlined subset of 67 features from the original 113. Additionally, Principal Component Analysis (PCA) demonstrated that 55 principal components are required to explain 95.2% of the dataset variance, confirming the dataset's high-dimensionality and lack of simple dominating factors.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# 1. ANOVA F-Score
st.header("1. ANOVA F-Score")
st.markdown("ANOVA F-score tests the linear relationship between each feature and the target variable. Higher F-scores mean the feature's distribution differs more significantly between 'Match' and 'No Match' classes.")
img_anova = load_image("09_5_2_anova_f_score_feature_selection_selectkbest.png")
if img_anova:
    st.image(img_anova, use_container_width=True)
else:
    st.error("Plot not found: 09_5_2_anova_f_score_feature_selection_selectkbest.png")

# 2. Mutual Information
st.header("2. Mutual Information (MI)")
st.markdown("Mutual Information measures both linear and non-linear dependencies. It captures complex relationships that ANOVA might miss.")
img_mi = load_image("10_5_3_mutual_information_feature_selection.png")
if img_mi:
    st.image(img_mi, use_container_width=True)
else:
    st.error("Plot not found: 10_5_3_mutual_information_feature_selection.png")

# 3. Union Strategy
st.header("3. The Union Strategy")
st.markdown("""
<div class="ml-callout">
    <strong>Selection Approach:</strong> Instead of relying on a single metric, we selected the top 40 features from ANOVA and the top 40 from Mutual Information. 
    By taking the union of both sets, we obtained <strong>67 unique features</strong> out of the original 113.
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Starting Features", "113")
with col2:
    st.metric("Features Retained", "67", "-46")
with col3:
    st.metric("Reduction", "40.7%")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# 4. PCA
st.header("4. Principal Component Analysis (PCA)")
st.markdown("We also explored PCA for dimensionality reduction. PCA transforms the features into a new set of orthogonal components that explain the variance in the data.")

col_pca1, col_pca2 = st.columns(2)
with col_pca1:
    st.subheader("Explained Variance")
    st.markdown("We found that **55 principal components** are required to explain 95% of the variance.")
    img_pca_var = load_image("11_6_1_explained_variance_analysis.png")
    if img_pca_var:
        st.image(img_pca_var, use_container_width=True)

with col_pca2:
    st.subheader("PCA Biplot")
    st.markdown("The first two components don't show clear class separation, confirming that our target variable has complex, non-linear relationships with the features.")
    img_pca_biplot = load_image("12_6_3_pca_biplot_first_two_principal_components.png")
    if img_pca_biplot:
        st.image(img_pca_biplot, use_container_width=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# 5. Train/Test Split
st.header("5. Train / Test Split")
st.markdown("Before modeling, we split the data 80/20 with stratification to preserve the target class balance.")
col_split1, col_split2 = st.columns([1, 2])
with col_split1:
    st.metric("Training Set (80%)", "40,000")
    st.metric("Testing Set (20%)", "10,000")
with col_split2:
    img_split = load_image("13_section_7_train_test_split.png")
    if img_split:
        st.image(img_split, use_container_width=True)
