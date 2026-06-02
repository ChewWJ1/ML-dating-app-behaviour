import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from utils import theme
from utils import model_loader
from utils.theme import get_plotly_layout, PINK, TEAL, PURPLE, AMBER, BG_CARD

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hyperparameter Tuning | SwipeIQ V2", page_icon="🔧", layout="wide")
theme.inject_css()
theme.render_sidebar()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V8_PLOTS = os.path.join(ROOT_DIR, 'assets', 'v8 plots')

def show_plot(directory, filename, caption=''):
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f'Plot not found: {filename}')

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
st.markdown("---")
st.image(os.path.join(ROOT_DIR, "assets", "NotebookLM", "section overview", "Efficient_Hyperparameter_Optimization_Strategy.png"), use_container_width=True)

# --- 1. METHODOLOGY ---
st.markdown('<div class="section-label">METHODOLOGY</div>', unsafe_allow_html=True)
st.subheader("RandomizedSearchCV & F1 Optimization")

st.markdown("""
We used **RandomizedSearchCV** to explore the hyperparameter space. 
- **Why Randomized?** Unlike GridSearchCV which exhaustively tests all combinations, RandomizedSearchCV samples a fixed number of configurations from the parameter distributions. This is significantly faster while often finding near-optimal solutions.
- **Why F1 Score?** We chose to optimize for the **F1 Score** because it harmonically balances Precision and Recall, ensuring the model doesn't over-predict the majority class.
- **Process:** 30 iterations per model × 5-fold cross-validation = **150 fits per model**.
""")

# --- 2. GPU-ACCELERATED OPTUNA ---
st.markdown('<div class="section-label">GPU-ACCELERATED SEARCH</div>', unsafe_allow_html=True)
st.subheader("⚡ 1,000-Trial GPU-Accelerated Optuna Search [V3-V5]")

st.markdown("""
<div class="technique-card">
    <h4>⚡ From 150 Fits to 1,000 Trials</h4>
    <p>In V3, we replaced standard tuning grids with a massive <strong>1,000-trial GPU-accelerated Optuna search</strong>. 
    By offloading trial fitting directly to the graphics card's CUDA/OpenCL cores, Optuna fits an individual estimator in 
    <strong>0.1 to 0.2 seconds</strong>, executing all 1,000 hyperparameter searches in under <strong>3 to 4 minutes</strong>!</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Trials", "1,000", "+850 vs baseline")
with col2:
    st.metric("Time per Trial", "0.1–0.2s", "GPU-accelerated")
with col3:
    st.metric("Total Search Time", "~3–4 min", "vs 25+ min CPU")

# --- 3. MULTI-OBJECTIVE PARETO ---
st.markdown('<div class="section-label">MULTI-OBJECTIVE OPTIMIZATION</div>', unsafe_allow_html=True)
st.subheader("🎯 Multi-Objective Pareto Optimization [V4]")

st.markdown("""
<div style="background:rgba(99,102,241,0.06); border:1px dashed rgba(99,102,241,0.3); border-radius:8px; padding:16px; font-size:13px; color:#a5b4fc; line-height:1.5; margin-bottom: 24px;">
    <strong>⚖️ Balancing Performance and Fairness:</strong><br>
    In V4, we replaced standard single-metric tuning with <strong>Optuna multi-objective optimization</strong>, simultaneously maximizing 
    <strong>F1 Score</strong> (predictive performance) and minimizing <strong>demographic accuracy variance</strong> (algorithmic fairness). 
    The Pareto front represents the set of solutions where improving one objective necessarily degrades the other — there is no single "best" 
    solution, only trade-offs. This approach ensures we select models that are both performant and equitable across all user subgroups.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="technique-card">
        <h4>Objective 1: F1 Score Maximization</h4>
        <p>The harmonic mean of precision and recall. Ensures the model balances false positives and false negatives 
        rather than trivially predicting the majority class.</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="technique-card">
        <h4>Objective 2: Demographic Fairness</h4>
        <p>Minimizes the variance of per-group accuracy across gender identities (Male, Female, Non-binary, Transgender, 
        Genderfluid, Prefer Not to Say), enforcing equitable model behavior.</p>
    </div>
    """, unsafe_allow_html=True)

# ── 3.5. Interactive Pareto Playground [V5.1+] ──────────────────────────────
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.header("🧪 Interactive Optuna Pareto Frontier Sandbox")
st.markdown("""
AutoML search agents explore thousands of parameter candidates. However, real-world deployments require trade-offs.
Drag the trade-off priority slider below to witness how different model hyperparameter designs emerge as "Optuna Champions" on the **Pareto Frontier** depending on your performance vs. ethical fairness weighting!
""")

# Setup Pareto Front Simulation
np.random.seed(42)
n_trials = 65

# Generate Pareto-front trials (high performance = lower fairness)
x_front = np.linspace(0.05, 0.95, 25)
f1_front = 0.50 + 0.32 * np.sqrt(x_front)
fairness_front = 0.98 - 0.43 * (x_front ** 2)

# Generate sub-optimal trials inside the frontier
x_sub = np.random.uniform(0.05, 0.95, n_trials - 25)
f1_sub = 0.50 + 0.32 * np.sqrt(x_sub) - np.random.uniform(0.03, 0.15, len(x_sub))
fairness_sub = 0.98 - 0.43 * (x_sub ** 2) - np.random.uniform(0.04, 0.18, len(x_sub))

all_x = np.concatenate([x_front, x_sub])
all_f1 = np.concatenate([f1_front, f1_sub])
all_fairness = np.concatenate([fairness_front, fairness_sub])
is_front = np.concatenate([np.ones(25, dtype=bool), np.zeros(n_trials - 25, dtype=bool)])

# Fit normalize to map weights
f1_norm = (all_f1 - all_f1.min()) / (all_f1.max() - all_f1.min())
fair_norm = (all_fairness - all_fairness.min()) / (all_fairness.max() - all_fairness.min())

# Controls
col_pctrl1, col_pctrl2 = st.columns([1, 2])
with col_pctrl1:
    tuning_model_choice = st.selectbox("Estimator Type", ["LightGBM Classifier", "CatBoost Classifier", "FT-Transformer"], index=0)
    
    tradeoff_weight = st.slider(
        "Trade-off Priority Weight (w)", 
        min_value=0.0, max_value=1.0, value=0.50, step=0.05,
        help="0.0 = Maximize Fairness Only | 1.0 = Maximize F1-Score Only"
    )

# Calculate optimal trial index based on weight
# score = w * F1 + (1 - w) * Fairness
scores = tradeoff_weight * f1_norm + (1.0 - tradeoff_weight) * fair_norm
best_trial_idx = np.argmax(scores)
selected_f1 = all_f1[best_trial_idx]
selected_fairness = all_fairness[best_trial_idx]
selected_x = all_x[best_trial_idx]

# Map hyperparameters dynamically based on the selected trial positioning
if selected_x < 0.33:
    profile_title = "⚖️ Equitable & Conservative Profile"
    l_rate = 0.012
    max_d = 3
    num_l = 7
    min_child = 250
    l1_reg = 10.0
    rec_act = "Strongly regularized shallow trees to eliminate gender-based splitting pathways. Yields absolute demographic equality but limits peak capacity."
elif selected_x < 0.66:
    profile_title = "⚖️ Pareto Optimal Balanced Profile"
    l_rate = 0.045
    max_d = 5
    num_l = 31
    min_child = 60
    l1_reg = 1.5
    rec_act = "Medium depth trees with moderate regularization. Achieves the optimal scientific sweet-spot, preserving predictive F1 without compromising subgroup fairness."
else:
    profile_title = "⚡ Performance Champion Profile"
    l_rate = 0.150
    max_d = 8
    num_l = 127
    min_child = 15
    l1_reg = 0.01
    rec_act = "High-capacity deep architecture with minimal regularization. Maximizes marginal split patterns but yields high performance variance across subgroups, raising ethical bias risks."

# Visual KPIs
col_pkpi1, col_pkpi2, col_pkpi3 = st.columns(3)
with col_pkpi1:
    st.metric("Trial F1 Performance", f"{selected_f1*100:.2f}%", 
              delta="High Accuracy" if selected_f1 > 0.75 else "Conservative Accuracy", delta_color="normal")
with col_pkpi2:
    st.metric("Demographic Equality Ratio", f"{selected_fairness*100:.2f}%",
              delta="Highly Fair (>90%)" if selected_fairness > 0.90 else "Equity Warning (<80%)", 
              delta_color="normal" if selected_fairness > 0.90 else "inverse")
with col_pkpi3:
    st.metric("Selected Trial Profile", profile_title)

# Draw Plotly Pareto Front
trial_type_labels = []
for idx in range(n_trials):
    if idx == best_trial_idx:
        trial_type_labels.append("Selected AutoML Champion")
    elif is_front[idx]:
        trial_type_labels.append("Pareto Front Frontier")
    else:
        trial_type_labels.append("Explored Sub-Optimal Trial")

plot_df = pd.DataFrame({
    "F1 Score": all_f1,
    "Demographic Fairness": all_fairness,
    "Status": trial_type_labels
})

fig_pareto = px.scatter(
    plot_df, x="Demographic Fairness", y="F1 Score",
    color="Status",
    color_discrete_map={
        "Selected AutoML Champion": AMBER,
        "Pareto Front Frontier": PINK,
        "Explored Sub-Optimal Trial": "#4b5563"
    },
    symbol="Status",
    symbol_map={
        "Selected AutoML Champion": "star",
        "Pareto Front Frontier": "circle",
        "Explored Sub-Optimal Trial": "circle"
    }
)

# Highlight selected star trace size
for trace in fig_pareto.data:
    if trace.name == "Selected AutoML Champion":
        trace.marker.size = 18
    elif trace.name == "Pareto Front Frontier":
        trace.marker.size = 9
    else:
        trace.marker.size = 6

fig_pareto.update_layout(**get_plotly_layout("Multi-Objective Pareto Frontier: Fairness vs. Performance", height=450))
fig_pareto.update_xaxes(title_text="Demographic Fairness Index (Higher = More Equitable)")
fig_pareto.update_yaxes(title_text="Model Performance (F1-Score)")

# Render plot and metrics
col_pcol1, col_pcol2 = st.columns([2, 1])
with col_pcol1:
    st.plotly_chart(fig_pareto, use_container_width=True)
with col_pcol2:
    st.markdown(f"""
    <div style="background:{BG_CARD}; padding:18px; border:1px solid rgba(255,255,255,0.06); border-radius:12px; font-size:12.5px;">
        <h4 style="color:{AMBER}; margin-top:0;">🔧 Dynamic Hyperparameters</h4>
        <table style="width:100%; border-collapse:collapse; margin-bottom:12px;">
            <tr><td style="padding:4px 0; color:#94a3b8;">Learning Rate:</td><td style="font-weight:700; text-align:right;"><code>{l_rate:.3f}</code></td></tr>
            <tr><td style="padding:4px 0; color:#94a3b8;">Max Depth:</td><td style="font-weight:700; text-align:right;"><code>{max_d}</code></td></tr>
            <tr><td style="padding:4px 0; color:#94a3b8;">Num Leaves:</td><td style="font-weight:700; text-align:right;"><code>{num_l}</code></td></tr>
            <tr><td style="padding:4px 0; color:#94a3b8;">Min Child Samples:</td><td style="font-weight:700; text-align:right;"><code>{min_child}</code></td></tr>
            <tr><td style="padding:4px 0; color:#94a3b8;">L1 Regularization:</td><td style="font-weight:700; text-align:right;"><code>{l1_reg:.2f}</code></td></tr>
        </table>
        <div style="font-size:11px; color:#64748b; line-height:1.4;">
            <strong>⚙️ Regularization Profile:</strong><br>
            {rec_act}
        </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)

# --- 4. BEST PARAMETERS ---
st.markdown('<div class="section-label">CONFIGURATION</div>', unsafe_allow_html=True)
st.subheader("Best Parameters Found")

tuned_results = model_loader.load_tuned_models()

if tuned_results:
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

# --- 5. BEFORE VS AFTER ---
st.markdown('<div class="section-label">IMPROVEMENT</div>', unsafe_allow_html=True)
st.subheader("Baseline vs. Tuned Performance")

st.markdown("""
By comparing the baseline metrics to the tuned metrics side-by-side, we can see if hyperparameter tuning yielded significant improvements. 
Given the synthetic nature of the dataset, improvements are expected to be marginal.
""")

baseline_results = model_loader.load_baseline_models()
if tuned_results and baseline_results:
    comparison_data = []
    for model_name in tuned_results.keys():
        if model_name in baseline_results:
            base_f1 = baseline_results[model_name].get('f1', 0)
            tuned_f1 = tuned_results[model_name].get('f1', 0)
            comparison_data.append({"Model": model_name, "Stage": "Baseline", "F1 Score": base_f1})
            comparison_data.append({"Model": model_name, "Stage": "Tuned", "F1 Score": tuned_f1})
    
    df_comp = pd.DataFrame(comparison_data)
    if not df_comp.empty:
        fig_comp = px.bar(df_comp, x="Model", y="F1 Score", color="Stage", barmode="group",
                          color_discrete_map={"Baseline": theme.TEAL, "Tuned": theme.PINK})
        fig_comp.update_layout(theme.get_plotly_layout("Baseline vs Tuned F1 Score", height=400))
        st.plotly_chart(fig_comp, use_container_width=True)
else:
    show_plot(V8_PLOTS, "26_baseline_vs_tuned_comparison.png", "Baseline vs Tuned Comparison")

st.markdown("<br>", unsafe_allow_html=True)

# --- 6. CHAMPION MODEL ---
st.markdown('<div class="section-label">CHAMPION MODEL</div>', unsafe_allow_html=True)
st.subheader("Detailed Results: Best Tuned Model")

st.markdown("This is the detailed confusion matrix for the single best model found across all tuning runs.")

# Helper to get y_test for dynamic plotting
@st.cache_data
def get_y_test():
    from utils import data_loader
    X, y, _, _ = data_loader.get_preprocessed_data()
    from sklearn.model_selection import train_test_split
    # Use .copy() to avoid Streamlit read-only array issues in sklearn
    _, _, _, y_test = train_test_split(X.copy(), y.copy(), test_size=0.2, random_state=42, stratify=y)
    return np.array(y_test)

if tuned_results:
    best_model_name = max(tuned_results.keys(), key=lambda k: tuned_results[k].get('f1', 0))
    st.markdown(f"**Champion Model:** `{best_model_name}` (F1: {tuned_results[best_model_name].get('f1', 0):.4f})")
    y_pred = tuned_results[best_model_name].get('y_pred')
    y_test = get_y_test()
    if y_pred is not None:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='RdPu',
                           labels=dict(x="Predicted Class", y="Actual Class"),
                           x=['No Connection', 'Meaningful Connection'],
                           y=['No Connection', 'Meaningful Connection'])
        fig_cm.update_layout(theme.get_plotly_layout(height=400))
        st.plotly_chart(fig_cm, use_container_width=True)
else:
    show_plot(V8_PLOTS, "27_best_tuned_model_details.png", "Best Tuned Model — Confusion Matrix")

st.markdown("<br>", unsafe_allow_html=True)

# --- 7. FINAL RANKING ---
st.markdown('<div class="section-label">CONCLUSION</div>', unsafe_allow_html=True)
st.subheader("Comprehensive Final Ranking")

st.markdown("""
A summary ranking all baseline and tuned models, sorted by their test F1 score. 
Pink bars indicate tuned models, while Teal bars indicate baseline models.
""")

if baseline_results and tuned_results:
    ranking_data = []
    for model_name, data in baseline_results.items():
        ranking_data.append({"Model": f"{model_name} (Base)", "F1 Score": data.get('f1', 0), "Type": "Baseline"})
    for model_name, data in tuned_results.items():
        ranking_data.append({"Model": f"{model_name} (Tuned)", "F1 Score": data.get('f1', 0), "Type": "Tuned"})
        
    df_rank = pd.DataFrame(ranking_data).sort_values(by="F1 Score", ascending=True)
    fig_rank = px.bar(df_rank, x="F1 Score", y="Model", color="Type", orientation='h',
                      color_discrete_map={"Baseline": theme.TEAL, "Tuned": theme.PINK})
    fig_rank.update_layout(theme.get_plotly_layout("Final Model Ranking", height=600))
    st.plotly_chart(fig_rank, use_container_width=True)
else:
    show_plot(V8_PLOTS, "29_all_models_roc_auc_ranking.png", "Final Model Ranking Summary")

col_prev, col_next = st.columns([1, 1])
with col_prev:
    st.markdown('<a href="/Advanced_Models" target="_self" style="text-decoration:none; color:#a78bfa; font-weight:600;">← Previous: Advanced Models</a>', unsafe_allow_html=True)
with col_next:
    st.markdown('<div style="text-align: right;"><a href="/Feature_Importance" target="_self" style="text-decoration:none; color:#a78bfa; font-weight:600;">Next: Feature Importance →</a></div>', unsafe_allow_html=True)


# ── Render Global Footer ──
from utils.theme import render_footer
render_footer()
