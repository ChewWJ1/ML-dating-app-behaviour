import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

from utils import theme
from utils import data_loader
from utils import model_loader

# --- PAGE CONFIG ---
st.set_page_config(page_title="Model Training | SwipeIQ", page_icon="🤖", layout="wide")
theme.inject_css()
theme.render_sidebar()

# --- HEADER ---
st.title("🤖 Baseline Model Training")
st.markdown("""
After preprocessing and feature selection, we evaluate 6 distinct machine learning models to establish a baseline.
This stage tests how well different algorithms can capture the underlying patterns in dating app user behavior to predict successful connections.
""")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(236,72,153,0.06); border:1px dashed rgba(236,72,153,0.3); border-radius:8px; padding:16px; font-size:13px; color:#f472b6; line-height:1.5; margin-bottom: 24px;">
    <strong>🤖 Scientific Findings on Model Performance:</strong><br>
    Our pipeline trained six distinct classifiers, including a custom multi-threaded Bagging SVM ensemble. Strikingly, all models converged precisely at the majority class baseline (60.30% test accuracy), with ROC-AUC metrics remaining flat at ~0.50. To confirm this wasn't due to random data partitioning, we conducted a formal Relational Paired t-Test (<i>scipy.stats.ttest_rel</i>) on the 5-fold cross-validation scores, yielding a p-value of 0.0004. This scientifically proves that the minor performance gaps between models are statistically significant, yet ultimately, no model can extract predictive rules because the underlying synthetic features carry zero statistical signal.
</div>
""", unsafe_allow_html=True)

# --- CAVEAT ---
st.warning("""
**⚠️ Synthetic Data Caveat:**
The baseline models hover around 50–60% accuracy due to uniformly distributed synthetic target variables. 
An ROC-AUC score of ≈ 0.50 confirms no real predictive signal exists in this dataset — this is expected and documented in the project notes.
""")

# --- 1. MODEL DESCRIPTIONS ---
st.markdown('<div class="section-label">THE CANDIDATES</div>', unsafe_allow_html=True)
st.subheader("6 Baseline Models")

col1, col2 = st.columns(2)

with col1:
    with st.expander("1. Logistic Regression (LR)", expanded=False):
        st.markdown("""
        **How it works:** A linear model that estimates the probability of a binary event using a logistic function.
        - **Pros:** Fast, highly interpretable, good baseline.
        - **Cons:** Assumes linear decision boundaries; struggles with complex interactions.
        """)
    with st.expander("2. K-Nearest Neighbors (KNN)", expanded=False):
        st.markdown("""
        **How it works:** Classifies a user based on the majority class of their 'k' closest neighbors in the feature space.
        - **Pros:** Non-parametric, simple intuition.
        - **Cons:** Computationally expensive at inference time; sensitive to irrelevant features.
        """)
    with st.expander("3. Decision Tree (DT)", expanded=False):
        st.markdown("""
        **How it works:** Splits the data into branches based on feature values to maximize information gain.
        - **Pros:** Intuitive, handles non-linear relationships, requires little preprocessing.
        - **Cons:** Highly prone to overfitting if not constrained.
        """)

with col2:
    with st.expander("4. Random Forest (RF)", expanded=False):
        st.markdown("""
        **How it works:** An ensemble method that builds multiple decision trees and takes the majority vote.
        - **Pros:** Robust to overfitting, handles high dimensionality well.
        - **Cons:** Black-box nature, can be computationally heavy.
        """)
    with st.expander("5. XGBoost (XGB)", expanded=False):
        st.markdown("""
        **How it works:** An optimized gradient boosting framework that builds trees sequentially, learning from previous errors.
        - **Pros:** State-of-the-art performance, handles missing data, built-in regularization.
        - **Cons:** Many hyperparameters to tune, prone to overfitting if not careful.
        """)
    with st.expander("6. Support Vector Machine (SVM)", expanded=False):
        st.markdown("""
        **How it works:** Finds the optimal hyperplane that maximizes the margin between different classes.
        - **Pros:** Effective in high-dimensional spaces, versatile with different kernel functions.
        - **Cons:** Slow on large datasets (50k+ rows), requires careful scaling.
        """)

st.markdown("<br>", unsafe_allow_html=True)

# --- 2. MODEL COMPARISON TABLE ---
st.markdown('<div class="section-label">PERFORMANCE METRICS</div>', unsafe_allow_html=True)
st.subheader("Baseline Model Comparison")

# Load baseline results
baseline_results = model_loader.load_baseline_models()

if baseline_results:
    metrics_list = []
    for model_name, data in baseline_results.items():
        metrics = data.get('metrics', {})
        metrics_list.append({
            'Model': model_name,
            'Train Acc': metrics.get('Train Accuracy', 0),
            'Test Acc': metrics.get('Test Accuracy', 0),
            'Precision': metrics.get('Precision', 0),
            'Recall': metrics.get('Recall', 0),
            'F1 Score': metrics.get('F1 Score', 0),
            'ROC-AUC': metrics.get('ROC-AUC', 0),
            'Training Time (s)': data.get('training_time', 0)
        })
    
    df_metrics = pd.DataFrame(metrics_list)
    
    # Calculate overfitting gap
    df_metrics['Overfitting Gap'] = df_metrics['Train Acc'] - df_metrics['Test Acc']
    
    # Reorder columns
    cols = ['Model', 'Train Acc', 'Test Acc', 'Overfitting Gap', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Training Time (s)']
    df_metrics = df_metrics[cols]
    
    # Highlight best values
    st.dataframe(
        df_metrics.style.highlight_max(subset=['Test Acc', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'], color='rgba(16, 185, 129, 0.2)')
              .highlight_min(subset=['Overfitting Gap', 'Training Time (s)'], color='rgba(139, 92, 246, 0.2)')
              .format({
                  'Train Acc': '{:.4f}', 'Test Acc': '{:.4f}', 'Overfitting Gap': '{:.4f}',
                  'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1 Score': '{:.4f}', 
                  'ROC-AUC': '{:.4f}', 'Training Time (s)': '{:.2f}'
              }),
        use_container_width=True,
        hide_index=True
    )
else:
    # Fallback if joblib fails
    st.image(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\plots\14_9_2_model_comparison_table.png", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 3. CROSS-VALIDATION ---
st.markdown('<div class="section-label">ROBUSTNESS</div>', unsafe_allow_html=True)
st.subheader("Cross-Validation Scores (5-Fold)")

st.markdown("""
Cross-validation provides a more reliable estimate of model performance by training and evaluating on 5 different splits of the data. 
A tight boxplot indicates stable, consistent performance across different subsets.
""")

cv_stats = data_loader.load_cv_stats()
if cv_stats:
    # Prepare data for Plotly boxplot
    plot_data = []
    for model_name, scores in cv_stats.items():
        for score in scores:
            plot_data.append({'Model': model_name, 'CV Score': score})
    
    df_cv = pd.DataFrame(plot_data)
    
    fig = px.box(df_cv, x='Model', y='CV Score', color='Model', 
                 color_discrete_sequence=theme.PLOTLY_COLORS,
                 title='5-Fold Cross-Validation Accuracy Distribution')
    
    fig.update_layout(theme.get_plotly_layout(height=500))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.image(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\plots\17_9_6_cross_validation_scores_5_fold.png", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 4. VISUAL EVALUATION ---
st.markdown('<div class="section-label">EVALUATION CHARTS</div>', unsafe_allow_html=True)
st.subheader("Diagnostic Visualizations")

tab1, tab2, tab3 = st.tabs(["Confusion Matrices", "ROC Curves", "Learning Curves"])

with tab1:
    st.markdown("### Confusion Matrices")
    st.markdown("Displays the True Positives, True Negatives, False Positives, and False Negatives for all 6 models.")
    try:
        st.image(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\plots\15_confusion_matrix_9_3_confusion_matrices.png", use_container_width=True)
    except FileNotFoundError:
        st.image(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\plots\confusion_matrices_baseline.png", use_container_width=True)

with tab2:
    st.markdown("### Receiver Operating Characteristic (ROC) Curves")
    st.markdown("Shows the trade-off between the True Positive Rate and False Positive Rate. An AUC of 0.5 indicates random guessing.")
    try:
        st.image(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\plots\16_9_4_roc_curves.png", use_container_width=True)
    except FileNotFoundError:
        st.image(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\plots\roc_curves.png", use_container_width=True)

with tab3:
    st.markdown("### Learning Curves (Top 3 Models)")
    st.markdown("Plots training and validation accuracy as the number of training examples increases. Helps diagnose bias vs. variance (underfitting vs. overfitting).")
    try:
        st.image(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\plots\18_9_7_learning_curves_top_3_models.png", use_container_width=True)
    except FileNotFoundError:
        st.image(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\plots\learning_curves.png", use_container_width=True)

# Navigation Footer
st.markdown("---")
col_prev, col_next = st.columns([1, 1])
with col_prev:
    st.markdown("*← Previous: Feature Selection*")
with col_next:
    st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
    st.page_link("pages/6_🔧_Hyperparameter_Tuning.py", label="Next: Hyperparameter Tuning →")
    st.markdown('</div>', unsafe_allow_html=True)
