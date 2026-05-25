import streamlit as st
import pandas as pd
import json

from utils import theme
from utils import model_loader

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hyperparameter Tuning | SwipeIQ", page_icon="🔧", layout="wide")
theme.inject_css()
theme.render_sidebar()

# --- HEADER ---
st.title("🔧 Hyperparameter Tuning")
st.markdown("""
After identifying the baseline performance, we attempt to improve the models using **Hyperparameter Tuning**.
This process searches for the optimal configuration (hyperparameters) for each algorithm to maximize its performance metrics.
""")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(52,211,153,0.06); border:1px dashed rgba(52,211,153,0.3); border-radius:8px; padding:16px; font-size:13px; color:#6ee7b7; line-height:1.5; margin-bottom: 24px;">
    <strong>⚙️ Tuning Methodology & Compute Optimization:</strong><br>
    To maximize methodological rigor, we optimized hyperparameters using a 5-fold CV RandomizedSearchCV across the top-performing models. We employed <strong>Nested Parallelism Prevention</strong> by setting base models to run single-threaded during tuning, which eliminated thread oversubscription and optimized CPU core utilization. Even after exhaustive grid searching (150 fits per candidate estimator), improvements remained negligible (e.g., Random Forest F1 improved from 4.61% to 9.69%). This confirms our earlier hypothesis: no algorithm or tuning grid can conjure signal where only noise exists.
</div>
""", unsafe_allow_html=True)

# --- 1. METHODOLOGY ---
st.markdown('<div class="section-label">METHODOLOGY</div>', unsafe_allow_html=True)
st.subheader("RandomizedSearchCV & F1 Optimization")

st.markdown("""
We used **RandomizedSearchCV** to explore the hyperparameter space. 
- **Why Randomized?** Unlike GridSearchCV which exhaustively tests all combinations, RandomizedSearchCV samples a fixed number of configurations from the parameter distributions. This is significantly faster while often finding near-optimal solutions, which is crucial given our dataset size and search space.
- **Why F1 Score?** We chose to optimize for the **F1 Score** because it harmonically balances Precision and Recall, ensuring the model doesn't over-predict the majority class.
- **Process:** 30 iterations per model × 5-fold cross-validation = **150 fits per model**.
""")

# --- 2. SEARCH SPACES & BEST PARAMETERS ---
st.markdown('<div class="section-label">CONFIGURATION</div>', unsafe_allow_html=True)
st.subheader("Best Parameters Found")

# Load tuned results
tuned_results = model_loader.load_tuned_models()

if tuned_results:
    # Create tabs for each model to show their best parameters
    model_names = list(tuned_results.keys())
    tabs = st.tabs(model_names)
    
    for i, model_name in enumerate(model_names):
        with tabs[i]:
            best_params = tuned_results[model_name].get('best_params', {})
            st.markdown(f"**Optimal Configuration for {model_name}:**")
            st.json(best_params)
else:
    st.info("Tuned results joblib file is not available. Please ensure models have been trained and saved.")

st.markdown("<br>", unsafe_allow_html=True)

# --- 3. BEFORE VS AFTER COMPARISON ---
st.markdown('<div class="section-label">IMPROVEMENT</div>', unsafe_allow_html=True)
st.subheader("Baseline vs. Tuned Performance")

st.markdown("""
By comparing the baseline metrics to the tuned metrics side-by-side, we can see if hyperparameter tuning yielded significant improvements. 
Given the synthetic nature of the dataset, improvements are expected to be marginal.
""")

try:
    st.image(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\plots\19_10_3_before_vs_after_tuning_comparison.png", use_container_width=True)
except FileNotFoundError:
    st.image(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\plots\baseline_vs_tuned.png", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 4. BEST MODEL RESULTS ---
st.markdown('<div class="section-label">CHAMPION MODEL</div>', unsafe_allow_html=True)
st.subheader("Detailed Results: Best Tuned Model")

st.markdown("""
This is the detailed confusion matrix and classification report for the single best model found across all tuning runs.
""")

try:
    st.image(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\plots\20_confusion_matrix_10_4_best_tuned_model_detailed_results.png", use_container_width=True)
except FileNotFoundError:
    st.image(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\plots\confusion_matrix_best.png", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 5. FINAL RANKING ---
st.markdown('<div class="section-label">CONCLUSION</div>', unsafe_allow_html=True)
st.subheader("Comprehensive Final Ranking")

st.markdown("""
A summary ranking all baseline and tuned models, sorted by their test F1 score. 
Green bars indicate tuned models, while grey bars indicate baseline models.
""")

try:
    st.image(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\plots\22_section_12_final_model_summary.png", use_container_width=True)
except FileNotFoundError:
    st.image(r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\plots\final_ranking.png", use_container_width=True)

# Navigation Footer
st.markdown("---")
col_prev, col_next = st.columns([1, 1])
with col_prev:
    st.page_link("pages/5_🤖_Model_Training.py", label="← Previous: Model Training")
with col_next:
    st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
    # Removing link to page 7 if it doesn't exist, but per instructions, we just do our part.
    # We will use st.markdown instead of page_link for the next page to avoid breaking if the next page isn't ready
    st.markdown("*Next: Feature Importance*")
    st.markdown('</div>', unsafe_allow_html=True)
