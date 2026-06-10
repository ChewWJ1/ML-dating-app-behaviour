import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from sklearn.datasets import make_moons

from utils import theme
from utils import data_loader
from utils import model_loader
from utils.theme import get_plotly_layout, PINK, TEAL, PURPLE, AMBER, GREEN, RED, SKY, BG_CARD, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED

# --- PAGE CONFIG ---
st.set_page_config(page_title="Model Training | SwipeIQ V2", page_icon="🤖", layout="wide")
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
st.title("🤖 Model Training & Evaluation")
st.markdown("""
After preprocessing and feature selection, we evaluate **13 distinct machine learning models** — spanning classical sklearn classifiers, 
gradient boosting frameworks, and custom PyTorch deep learning architectures — to establish comprehensive baselines and push performance boundaries.
""")
st.markdown("---")
st.image(os.path.join(ROOT_DIR, "assets", "New NotebookLM", "Section overview", "Model_Training_and_Statistical_Evaluation.png"), use_container_width=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(236,72,153,0.06); border:1px dashed rgba(236,72,153,0.3); border-radius:8px; padding:16px; font-size:13px; color:#f472b6; line-height:1.5; margin-bottom: 24px;">
    <strong>🤖 Scientific Findings on Model Performance (V8):</strong><br>
    Our pipeline trained distinct classifiers including gradient boosters (LightGBM, CatBoost, XGBoost), ensemble methods, and custom PyTorch architectures (FT-Transformer, SAINT, NODE). 
    With <strong>ImbPipeline</strong> isolating SMOTE exclusively within Cross-Validation folds, we scientifically prevented target leakage.
    Strikingly, all models converge precisely at the baseline accuracy (~60%), with ROC-AUC metrics remaining flat at ~0.50. This definitively proves that no algorithm can extract meaningful predictive rules from uniformly distributed synthetic behavioural features.
</div>
""", unsafe_allow_html=True)

st.warning("""
**⚠️ Synthetic Data Caveat:**
The baseline models hover around 50–60% accuracy due to uniformly distributed synthetic target variables. 
An ROC-AUC score of ≈ 0.50 confirms no real predictive signal exists in this dataset — this is expected and documented in the project notes.
""")

# --- 1. MODEL DESCRIPTIONS ---
st.markdown('<div class="section-label">THE CANDIDATES</div>', unsafe_allow_html=True)
st.subheader("16 Baseline & Advanced Models")

tab_classic, tab_ensemble, tab_boost, tab_other = st.tabs(["📐 Linear & Instance", "🌲 Ensemble Methods", "⚡ Gradient Boosting", "🧪 Other"])

with tab_classic:
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
    with col2:
        with st.expander("3. Support Vector Machine (SVM)", expanded=False):
            st.markdown("""
            **How it works:** Finds the optimal hyperplane that maximizes the margin between different classes.
            - **Pros:** Effective in high-dimensional spaces, versatile with kernel functions.
            - **Cons:** Slow on large datasets (50k+ rows), requires careful scaling.
            """)
        with st.expander("4. Naive Bayes (NB)", expanded=False):
            st.markdown("""
            **How it works:** Applies Bayes' theorem with the "naive" assumption of feature independence.
            - **Pros:** Extremely fast, works well with high-dimensional data, strong baseline.
            - **Cons:** Independence assumption rarely holds; poor probability calibration.
            """)

with tab_ensemble:
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("5. Decision Tree (DT)", expanded=False):
            st.markdown("""
            **How it works:** Splits the data into branches based on feature values to maximize information gain.
            - **Pros:** Intuitive, handles non-linear relationships, requires little preprocessing.
            - **Cons:** Highly prone to overfitting if not constrained.
            """)
        with st.expander("6. Random Forest (RF)", expanded=False):
            st.markdown("""
            **How it works:** An ensemble of decision trees using bagging (bootstrap aggregation) with majority vote.
            - **Pros:** Robust to overfitting, handles high dimensionality, parallel training.
            - **Cons:** Black-box nature, can be memory-heavy.
            """)
        with st.expander("7. Extra Trees (ET)", expanded=False):
            st.markdown("""
            **How it works:** Similar to Random Forest but uses random split thresholds instead of optimal splits.
            - **Pros:** Faster than RF, lower variance, better bias-variance tradeoff.
            - **Cons:** Can produce larger trees, less interpretable.
            """)
    with col2:
        with st.expander("8. AdaBoost", expanded=False):
            st.markdown("""
            **How it works:** Iteratively trains weak learners, focusing more on misclassified samples each round.
            - **Pros:** Simple, adaptive, less prone to overfitting than individual trees.
            - **Cons:** Sensitive to noisy data and outliers.
            """)
        with st.expander("9. Bagging SVM", expanded=False):
            st.markdown("""
            **How it works:** Custom multi-threaded ensemble wrapping SVM classifiers in a bagging meta-estimator.
            - **Pros:** Reduces SVM training time via parallelism, smooths decision boundaries.
            - **Cons:** Still inherits SVM's O(n²) memory complexity.
            """)
        with st.expander("10. Stacking Ensemble", expanded=False):
            st.markdown("""
            **How it works:** Combines predictions from multiple base learners using a meta-learner (e.g., Logistic Regression).
            - **Pros:** Can capture diverse model strengths, often achieves best ensemble performance.
            - **Cons:** Complex, computationally expensive, risk of overfitting the meta-layer.
            """)

with tab_boost:
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("11. Gradient Boosting (GBDT)", expanded=False):
            st.markdown("""
            **How it works:** Sequentially builds trees that correct the residual errors of previous trees.
            - **Pros:** High accuracy, handles heterogeneous features well.
            - **Cons:** Slow training (sequential), prone to overfitting without regularization.
            """)
        with st.expander("12. XGBoost (XGB)", expanded=False):
            st.markdown("""
            **How it works:** Optimized gradient boosting with L1/L2 regularization, tree pruning, and parallel processing.
            - **Pros:** State-of-the-art tabular performance, handles missing data, GPU support.
            - **Cons:** Many hyperparameters to tune, prone to overfitting if not careful.
            """)
    with col2:
        with st.expander("13. LightGBM (LGBM)", expanded=False):
            st.markdown("""
            **How it works:** Gradient boosting using histogram-based splits and leaf-wise growth for extreme speed.
            - **Pros:** Fastest gradient booster, native categorical support, GPU-accelerated.
            - **Cons:** Can overfit on small datasets, leaf-wise growth requires careful tuning.
            """)
        with st.expander("14. CatBoost", expanded=False):
            st.markdown("""
            **How it works:** Gradient boosting with ordered boosting to prevent target leakage and native categorical handling.
            - **Pros:** Best out-of-the-box performance, minimal preprocessing needed, robust.
            - **Cons:** Slower training than LightGBM, larger model files.
            """)

with tab_other:
    st.markdown("*See the **🧠 Advanced Models** page for deep learning architectures (MLP, FT-Transformer, SAINT, NODE, TabPFN, GNN, SCARF).*")

st.markdown("<br>", unsafe_allow_html=True)

# --- 2. MODEL COMPARISON TABLE ---
st.markdown('<div class="section-label">PERFORMANCE METRICS</div>', unsafe_allow_html=True)
st.subheader("Model Comparison Table")

baseline_results = model_loader.load_baseline_models()

if baseline_results:
    metrics_list = []
    for model_name, data in baseline_results.items():
        metrics_list.append({
            'Model': model_name,
            'Train Acc': data.get('train_acc', 0),
            'Test Acc': data.get('test_acc', 0),
            'Precision': data.get('precision', 0),
            'Recall': data.get('recall', 0),
            'F1 Score': data.get('f1', 0),
            'ROC-AUC': data.get('roc_auc', 0),
            'Training Time (s)': data.get('train_time', 0)
        })
    
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics['Overfitting Gap'] = df_metrics['Train Acc'] - df_metrics['Test Acc']
    cols = ['Model', 'Train Acc', 'Test Acc', 'Overfitting Gap', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Training Time (s)']
    df_metrics = df_metrics[cols]
    
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
    show_plot(V8_PLOTS, "18_model_performance_comparison.png", "V5 Model Comparison Table")

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
    show_plot(V8_PLOTS, "21_cross_validation_accuracy.png", "5-Fold CV Scores")

st.markdown("<br>", unsafe_allow_html=True)

# --- 4. VISUAL EVALUATION ---
st.markdown('<div class="section-label">EVALUATION CHARTS</div>', unsafe_allow_html=True)
st.subheader("Diagnostic Visualizations")

tab1, tab2, tab3 = st.tabs(["Confusion Matrices", "ROC Curves", "Learning Curves"])

# Helper to get y_test for dynamic plotting
@st.cache_data
def get_y_test():
    X, y, _, _ = data_loader.get_preprocessed_data()
    from sklearn.model_selection import train_test_split
    # Use .copy() to avoid Streamlit read-only array issues in sklearn
    _, _, _, y_test = train_test_split(X.copy(), y.copy(), test_size=0.2, random_state=42, stratify=y)
    return np.array(y_test)

with tab1:
    st.markdown("### Dynamic Confusion Matrices")
    st.markdown("Displays the True Positives, True Negatives, False Positives, and False Negatives.")
    if baseline_results:
        selected_model_cm = st.selectbox("Select Model for Confusion Matrix:", list(baseline_results.keys()))
        y_test = get_y_test()
        y_pred = baseline_results[selected_model_cm].get('y_pred')
        
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
            st.warning("Predictions not found in cache.")
    else:
        show_plot(V8_PLOTS, "19_confusion_matrices.png", "Confusion Matrices — All Models")

with tab2:
    st.markdown("### Receiver Operating Characteristic (ROC) Curves")
    st.markdown("Shows the trade-off between the True Positive Rate and False Positive Rate. An AUC of 0.5 indicates random guessing.")
    if baseline_results:
        from sklearn.metrics import roc_curve
        y_test = get_y_test()
        fig_roc = go.Figure()
        
        for model_name, data in baseline_results.items():
            y_prob = data.get('y_prob')
            if y_prob is not None:
                # Some models might return multi-class probas, others single array
                if len(np.array(y_prob).shape) > 1 and np.array(y_prob).shape[1] > 1:
                    probs = np.array(y_prob)[:, 1]
                else:
                    probs = np.array(y_prob)
                    
                fpr, tpr, _ = roc_curve(y_test, probs)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{model_name} (AUC: {data.get('roc_auc', 0):.2f})"))
                
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), showlegend=False))
        fig_roc.update_layout(theme.get_plotly_layout(title="ROC Curves", height=600))
        fig_roc.update_xaxes(title="False Positive Rate")
        fig_roc.update_yaxes(title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        show_plot(V8_PLOTS, "20_roc_curves.png", "ROC Curves — All Models Overlaid")

with tab3:
    st.markdown("### Learning Curves (Top 3 Models)")
    st.markdown("Plots training and validation accuracy as the number of training examples increases. Helps diagnose bias vs. variance.")
    
    lc_data = model_loader.load_learning_curve_data()
    if lc_data:
        selected_model_lc = st.selectbox("Select Model for Learning Curve:", list(lc_data.keys()))
        data = lc_data[selected_model_lc]
        train_sizes = data['train_sizes']
        train_scores_mean = np.mean(data['train_scores'], axis=1)
        val_scores_mean = np.mean(data['val_scores'], axis=1)
        
        fig_lc = go.Figure()
        fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean, mode='lines+markers', name='Training Score', line=dict(color=theme.PINK)))
        fig_lc.add_trace(go.Scatter(x=train_sizes, y=val_scores_mean, mode='lines+markers', name='Cross-Validation Score', line=dict(color=theme.TEAL)))
        fig_lc.update_layout(theme.get_plotly_layout(title=f"Learning Curve ({selected_model_lc})", height=500))
        fig_lc.update_xaxes(title="Training Examples")
        fig_lc.update_yaxes(title="Score")
        st.plotly_chart(fig_lc, use_container_width=True)
    else:
        show_plot(V8_PLOTS, "22_learning_curves.png", "Learning Curves — Top 3 Models")

st.markdown("<br>", unsafe_allow_html=True)

# --- 5. INTERACTIVE PLAYGROUND ---
st.markdown('<div class="section-label">🎮 PLAYGROUND</div>', unsafe_allow_html=True)
st.subheader("Interactive Decision Boundary & Model Architecture Playground")
st.markdown("""
Explore how **every single algorithm** forms decision boundaries and organizes its internal parameters!
Adjust the algorithm-specific hyperparameters on the left and witness the decision boundary and internal model visualizations update **instantly**!
""")

# Helper functions for geometric datasets
def generate_geometric_data(shape_name, noise_val, n_samples=300):
    if shape_name == "Two Moons":
        from sklearn.datasets import make_moons
        return make_moons(n_samples=n_samples, noise=noise_val, random_state=42)
    elif shape_name == "Concentric Circles":
        from sklearn.datasets import make_circles
        return make_circles(n_samples=n_samples, noise=noise_val, factor=0.5, random_state=42)
    elif shape_name == "Dual Spirals":
        # Generate mathematical spirals
        n = np.sqrt(np.random.default_rng(42).random((n_samples // 2, 1))) * 780 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.default_rng(42).standard_normal((n_samples // 2, 1)) * (noise_val * 4)
        d1y = np.sin(n) * n + np.random.default_rng(42).standard_normal((n_samples // 2, 1)) * (noise_val * 4)
        X0 = np.hstack((d1x, d1y))
        y0 = np.zeros(n_samples // 2)
        X1 = np.hstack((-d1x, -d1y))
        y1 = np.ones(n_samples // 2)
        X = np.vstack((X0, X1))
        y = np.concatenate((y0, y1))
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
        return X, y
    elif shape_name == "XOR Clusters":
        X = np.random.default_rng(42).uniform(-2.0, 2.0, (n_samples, 2))
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
        X += np.random.default_rng(42).standard_normal((n_samples, 2)) * (noise_val * 0.7)
        return X, y
    else: # Linear
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
            random_state=42, n_clusters_per_class=1, class_sep=1.5
        )
        return X, y

st.markdown("""
<div style="background:rgba(139,92,246,0.06); border:1px dashed rgba(139,92,246,0.3); border-radius:8px; padding:16px; font-size:13px; color:#c4b5fd; line-height:1.5; margin-bottom: 24px;">
    <strong>🎨 Complex Geometries Sandbox:</strong> Select different classification shapes below (e.g. Concentric Circles or Dual Spirals) and watch how linear models fail while non-linear kernels and neural architectures learn the boundaries!
</div>
""", unsafe_allow_html=True)

col_shape1, col_shape2 = st.columns(2)
with col_shape1:
    dataset_shape = st.selectbox("Dataset Geometry Pattern", ["Two Moons", "Concentric Circles", "Dual Spirals", "XOR Clusters", "Linear Separation"], index=0)
with col_shape2:
    dataset_noise = st.slider("Boundary Noise & Overlap", 0.05, 0.50, 0.20, 0.05)

X, y = generate_geometric_data(dataset_shape, dataset_noise, n_samples=250)

col_play_ctrl, col_play_viz = st.columns([1, 1.8])

with col_play_ctrl:
    st.markdown("""
    <div style="background:#161b22; border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:16px; margin-bottom:14px;">
        <h4 style="color:#a78bfa; margin-top:0; margin-bottom:10px;">Select Algorithm</h4>
    </div>
    """, unsafe_allow_html=True)
    
    algo = st.selectbox(
        "Choose Estimator",
        [
            "Logistic Regression", 
            "K-Nearest Neighbors (KNN)", 
            "Support Vector Machine (SVM)", 
            "Naive Bayes (Gaussian NB)",
            "Decision Tree", 
            "Random Forest", 
            "Extra Trees",
            "AdaBoost",
            "Gradient Boosting (GBDT)",
            "XGBoost Classifier",
            "LightGBM Classifier",
            "CatBoost Classifier",
            "Bagging SVM Ensemble",
            "Stacking Ensemble",
            "Multi-Layer Perceptron (MLP)"
        ],
        index=1
    )
    
    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    st.markdown(f"**Fine-tune {algo} Parameters:**")
    
    # Render hyperparams based on selected model
    if algo == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        c_val = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.05, help="Smaller values specify stronger regularization.")
        penalty = st.selectbox("Penalty Type", ["l2", "none"])
        model = LogisticRegression(C=c_val, penalty=penalty, solver="lbfgs", random_state=42)
        
    elif algo == "K-Nearest Neighbors (KNN)":
        from sklearn.neighbors import KNeighborsClassifier
        k_val = st.slider("Number of Neighbors (K)", 1, 25, 5, 2, help="Number of neighbors to vote on the class.")
        weights = st.selectbox("Weight Function", ["uniform", "distance"])
        metric = st.selectbox("Distance Metric", ["euclidean", "manhattan"])
        model = KNeighborsClassifier(n_neighbors=k_val, weights=weights, metric=metric)
        
    elif algo == "Support Vector Machine (SVM)":
        from sklearn.svm import SVC
        c_val = st.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)
        kernel = st.selectbox("Kernel Type", ["rbf", "linear", "poly"])
        gamma = st.selectbox("Kernel Gamma", ["scale", "auto"])
        model = SVC(C=c_val, kernel=kernel, gamma=gamma, probability=True, random_state=42)
        
    elif algo == "Naive Bayes (Gaussian NB)":
        from sklearn.naive_bayes import GaussianNB
        var_smoothing = st.slider("Var Smoothing (Variance Stabilizer)", 1e-11, 1e-1, 1e-9, format="%.1e")
        model = GaussianNB(var_smoothing=var_smoothing)
        
    elif algo == "Decision Tree":
        from sklearn.tree import DecisionTreeClassifier
        depth = st.slider("Max Depth", 1, 12, 4, 1, help="Maximum depth of the tree splits.")
        criterion = st.selectbox("Splitting Criterion", ["gini", "entropy"])
        min_samples_split = st.slider("Min Samples to Split", 2, 20, 2)
        model = DecisionTreeClassifier(max_depth=depth, criterion=criterion, min_samples_split=min_samples_split, random_state=42)
        
    elif algo == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        n_est = st.slider("Number of Trees", 5, 150, 50, 5, help="Number of trees in the forest ensemble.")
        depth = st.slider("Max Tree Depth", 1, 10, 4, 1)
        max_features = st.selectbox("Max Features per Split", ["sqrt", "log2", None])
        model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, max_features=max_features, random_state=42, n_jobs=-1)
        
    elif algo == "Extra Trees":
        from sklearn.ensemble import ExtraTreesClassifier
        n_est = st.slider("Number of Trees", 5, 150, 50, 5)
        depth = st.slider("Max Tree Depth", 1, 10, 4, 1)
        min_samples_leaf = st.slider("Min Samples in Leaf", 1, 10, 1)
        model = ExtraTreesClassifier(n_estimators=n_est, max_depth=depth, min_samples_leaf=min_samples_leaf, random_state=42, n_jobs=-1)
        
    elif algo == "AdaBoost":
        from sklearn.ensemble import AdaBoostClassifier
        n_est = st.slider("Number of Estimators", 5, 80, 20, 5)
        lr = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.05)
        model = AdaBoostClassifier(n_estimators=n_est, learning_rate=lr, random_state=42)
        
    elif algo == "Gradient Boosting (GBDT)":
        from sklearn.ensemble import GradientBoostingClassifier
        n_est = st.slider("Number of Trees", 5, 80, 20, 5)
        lr = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.05)
        subsample = st.slider("Subsample Ratio", 0.5, 1.0, 1.0, 0.05)
        model = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, subsample=subsample, random_state=42)
        
    elif algo == "XGBoost Classifier":
        from xgboost import XGBClassifier
        depth = st.slider("Max Depth", 1, 10, 4, 1)
        lr = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.05)
        reg_alpha = st.slider("L1 Regularization (Alpha)", 0.0, 5.0, 0.0, 0.1)
        reg_lambda = st.slider("L2 Regularization (Lambda)", 0.0, 5.0, 1.0, 0.1)
        model = XGBClassifier(max_depth=depth, learning_rate=lr, reg_alpha=reg_alpha, reg_lambda=reg_lambda, verbosity=0, random_state=42, n_jobs=-1)
        
    elif algo == "LightGBM Classifier":
        from lightgbm import LGBMClassifier
        num_leaves = st.slider("Num Leaves", 4, 127, 31, 2)
        lr = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.05)
        min_child = st.slider("Min Child Samples", 5, 50, 20)
        model = LGBMClassifier(num_leaves=num_leaves, learning_rate=lr, min_child_samples=min_child, verbosity=-1, random_state=42, n_jobs=-1)
        
    elif algo == "CatBoost Classifier":
        from catboost import CatBoostClassifier
        iterations = st.slider("Boosting Iterations", 5, 50, 20, 5, help="Kept low to ensure instantaneous UI response")
        depth = st.slider("Tree Depth", 1, 10, 4, 1)
        lr = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.05)
        model = CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=lr, verbose=False, random_state=42)
        
    elif algo == "Bagging SVM Ensemble":
        from sklearn.ensemble import BaggingClassifier
        from sklearn.svm import SVC
        n_est = st.slider("SVM Estimators in Bag", 2, 20, 5)
        max_samples = st.slider("Max Samples per Bag (%)", 10, 100, 80, 5)
        model = BaggingClassifier(estimator=SVC(C=1.0, kernel="rbf", random_state=42), n_estimators=n_est, max_samples=max_samples/100.0, random_state=42, n_jobs=-1)
        
    elif algo == "Stacking Ensemble":
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        meta_est_name = st.selectbox("Meta Learner", ["Logistic Regression", "Decision Tree"])
        meta_est = LogisticRegression() if meta_est_name == "Logistic Regression" else DecisionTreeClassifier(max_depth=3, random_state=42)
        estimators = [
            ('lr', LogisticRegression(C=1.0, random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('rf', RandomForestClassifier(max_depth=3, n_estimators=20, random_state=42))
        ]
        model = StackingClassifier(estimators=estimators, final_estimator=meta_est, n_jobs=-1)
        
    elif algo == "Multi-Layer Perceptron (MLP)":
        from sklearn.neural_network import MLPClassifier
        hidden_layout = st.selectbox("Hidden Layer Layout", ["8", "16", "8, 8", "16, 8", "16, 16"])
        activation = st.selectbox("Activation Function", ["relu", "tanh", "logistic"])
        solver = st.selectbox("Optimization Solver", ["adam", "sgd"])
        
        hidden_units = [int(x.strip()) for x in hidden_layout.split(",")]
        model = MLPClassifier(hidden_layer_sizes=hidden_units, activation=activation, solver=solver, max_iter=400, random_state=42)

with col_play_viz:
    # Train model on moons dataset
    model.fit(X, y)
    
    # Predict grid to plot contour
    h = 0.06
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Calculate model accuracy on this toy set
    accuracy = model.score(X, y) * 100
    
    # Build Plotly boundary chart
    fig = go.Figure()
    
    # Add colored decision regions
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        showscale=False,
        opacity=0.28,
        colorscale=[[0, '#ec4899'], [1, '#14b8a6']], # Pink vs Teal
        hoverinfo='skip'
    ))
    
    # Add support vectors highlight if SVM is chosen
    if algo == "Support Vector Machine (SVM)":
        svs = model.support_vectors_
        fig.add_trace(go.Scatter(
            x=svs[:, 0], y=svs[:, 1],
            mode='markers',
            name='Support Vectors (Margins)',
            marker=dict(size=12, color='rgba(0,0,0,0)', line=dict(width=1.5, color='#ffffff'))
        ))

    # Add Gaussian likelihood ellipses if Naive Bayes is chosen
    elif algo == "Naive Bayes (Gaussian NB)":
        for class_idx, class_color in enumerate([PINK, TEAL]):
            mean = model.theta_[class_idx]
            var = model.var_[class_idx]
            t = np.linspace(0, 2*np.pi, 100)
            ell_x = mean[0] + np.sqrt(var[0]) * np.cos(t)
            ell_y = mean[1] + np.sqrt(var[1]) * np.sin(t)
            fig.add_trace(go.Scatter(
                x=ell_x, y=ell_y,
                mode='lines',
                line=dict(color=class_color, width=1.5, dash='dot'),
                name=f'Class {class_idx} Normal Variance'
            ))
            
    # Add data scatter points
    fig.add_trace(go.Scatter(
        x=X[y==0, 0], y=X[y==0, 1],
        mode='markers',
        name='No Connection (y=0)',
        marker=dict(size=8, color='#ec4899', line=dict(width=1, color='#0f172a'))
    ))
    
    fig.add_trace(go.Scatter(
        x=X[y==1, 0], y=X[y==1, 1],
        mode='markers',
        name='Matched (y=1)',
        marker=dict(size=8, color='#14b8a6', line=dict(width=1, color='#0f172a'))
    ))
    
    # Add KNN Query point exploration if KNN selected
    if algo == "K-Nearest Neighbors (KNN)":
        st.markdown("##### 🔍 KNN Neighborhood query:")
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            q_x1 = st.slider("Query Feature 1 (X)", -2.0, 2.0, 0.30, 0.10)
        with col_q2:
            q_x2 = st.slider("Query Feature 2 (Y)", -2.0, 2.0, 0.35, 0.10)
            
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(X, np.array([[q_x1, q_x2]]), metric=metric).flatten()
        nearest_indices = np.argsort(distances)[:k_val]
        
        # Plot query point
        fig.add_trace(go.Scatter(
            x=[q_x1], y=[q_x2],
            mode='markers',
            name='Query Point (Active User)',
            marker=dict(size=14, color=AMBER, symbol='star', line=dict(width=2, color='#ffffff'))
        ))
        
        # Connect query point to nearest neighbors
        for idx in nearest_indices:
            fig.add_trace(go.Scatter(
                x=[q_x1, X[idx, 0]], y=[q_x2, X[idx, 1]],
                mode='lines',
                line=dict(color=AMBER, width=1.5, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
    layout = theme.get_plotly_layout(height=420)
    layout['margin'] = dict(l=10, r=10, t=30, b=10)
    layout['xaxis']['title'] = 'Feature 1 (e.g., App Usage Time)'
    layout['yaxis']['title'] = 'Feature 2 (e.g., Likes Received)'
    layout['legend'] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    fig.update_layout(**layout)
    
    st.plotly_chart(fig, use_container_width=True, key="playground_fig")
    
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; background:#161b22; border:1px solid rgba(255,255,255,0.07); border-radius:10px; padding:10px 18px; margin-top:8px;">
        <span style="font-size:13px; color:#94a3b8;">Decision Boundary Accuracy on Toy Set:</span>
        <span style="font-size:16px; font-weight:700; color:#14b8a6;">{accuracy:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)

# ── Model Specific Extra Interactive Visualizations ────────────────────────
st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)

if algo == "Logistic Regression":
    st.markdown("### 📊 Logistic Regression Coefficient Analysis")
    st.markdown("Logistic Regression assigns linear weights (coefficients) to input dimensions. Here are the learned weights for Feature 1 and Feature 2:")
    
    coef_df = pd.DataFrame({
        "Feature": ["Feature 1 (X)", "Feature 2 (Y)"],
        "Coefficient": model.coef_[0]
    })
    fig_lr = px.bar(coef_df, x="Feature", y="Coefficient", color="Coefficient", color_continuous_scale="RdBu", color_continuous_midpoint=0)
    fig_lr.update_layout(**get_plotly_layout("Learned Coefficients (Weights)", height=280))
    
    col_l1, col_l2 = st.columns([2, 1])
    with col_l1:
        st.plotly_chart(fig_lr, use_container_width=True)
    with col_l2:
        st.markdown(f"""
        <div style="background:{BG_CARD}; padding:16px; border:1px solid rgba(255,255,255,0.05); border-radius:8px; font-size:12.5px; height:100%;">
            <strong>📐 Linear Equation:</strong><br>
            The model computes probability as:<br>
            <code>P(y=1) = sigmoid(w₁x₁ + w₂x₂ + b)</code><br><br>
            • w₁ (Feature 1 Weight): <code>{model.coef_[0][0]:.4f}</code><br>
            • w₂ (Feature 2 Weight): <code>{model.coef_[0][1]:.4f}</code><br>
            • b (Intercept Bias): <code>{model.intercept_[0]:.4f}</code><br><br>
            This linear relationship results in a completely straight decision boundary corridor.
        </div>
        """, unsafe_allow_html=True)

elif algo == "K-Nearest Neighbors (KNN)":
    st.markdown("### 🔍 KNN Distance Metric Comparison")
    st.markdown(f"KNN makes predictions on the fly by voting on neighbor identities. The nearest neighbors were calculated using **{metric}** distance.")
    
    neighbor_outcomes = y[nearest_indices]
    n_class_1 = np.sum(neighbor_outcomes)
    n_class_0 = len(neighbor_outcomes) - n_class_1
    pred_outcome = "Meaningful Connection" if n_class_1 > n_class_0 else "No Connection"
    
    st.markdown(f"""
    <div style="background:{BG_CARD}; padding:18px; border:1px solid rgba(255,255,255,0.06); border-radius:12px; font-size:13px;">
        📌 <strong>Live Voting Diagnostics:</strong><br>
        • Neighbors Voting for <strong>Meaningful Connection (y=1)</strong>: <code>{n_class_1}</code> / <code>{k_val}</code><br>
        • Neighbors Voting for <strong>No Connection (y=0)</strong>: <code>{n_class_0}</code> / <code>{k_val}</code><br>
        • Final Predicted Outcome: <strong style="color:{TEAL if pred_outcome == 'Meaningful Connection' else PINK};">{pred_outcome}</strong>
    </div>
    """, unsafe_allow_html=True)

elif algo == "Support Vector Machine (SVM)":
    st.markdown("### 🛡️ SVM Kernel Transformation")
    st.markdown("Support Vector Machines use the **kernel trick** to project data into a higher-dimensional space where classes are linearly separable.")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Total Support Vectors", f"{len(model.support_)}")
    with col_s2:
        st.metric("Support Vectors per Class", f"Class 0: {model.n_support_[0]} | Class 1: {model.n_support_[1]}")
        
    st.markdown(f"""
    <div style="background:{BG_CARD}; padding:16px; border:1px solid rgba(255,255,255,0.05); border-radius:8px; font-size:12.5px;">
        💡 <strong>SVM Mechanism:</strong><br>
        The decision boundary is defined entirely by the **support vectors** (highlighted on the plot with white rings). 
        Data points outside these margins do not affect the boundary position at all. Adjusting the regularization parameter <code>C</code> 
        controls the margin size vs. classification errors: higher <code>C</code> enforces narrower margins and fewer misclassifications on training points.
    </div>
    """, unsafe_allow_html=True)

elif algo == "Naive Bayes (Gaussian NB)":
    st.markdown("### 🔔 Gaussian Probability Density Distributions")
    st.markdown("Gaussian Naive Bayes models the continuous features as a normal (bell-curve) distribution for each class independently.")
    
    # Plot probability distributions along Feature 1
    f1_range = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200)
    pdf_c0 = (1 / np.sqrt(2 * np.pi * model.var_[0][0])) * np.exp(-((f1_range - model.theta_[0][0])**2) / (2 * model.var_[0][0]))
    pdf_c1 = (1 / np.sqrt(2 * np.pi * model.var_[1][0])) * np.exp(-((f1_range - model.theta_[1][0])**2) / (2 * model.var_[1][0]))
    
    fig_pdf = go.Figure()
    fig_pdf.add_trace(go.Scatter(x=f1_range, y=pdf_c0, name="Class 0 (No Connection) PDF", line=dict(color=PINK, width=2)))
    fig_pdf.add_trace(go.Scatter(x=f1_range, y=pdf_c1, name="Class 1 (Matched) PDF", line=dict(color=TEAL, width=2)))
    fig_pdf.update_layout(**get_plotly_layout("Feature 1 Probabilistic Likelihood Density (PDF)", height=280))
    st.plotly_chart(fig_pdf, use_container_width=True)

elif algo == "Decision Tree":
    st.markdown("### 🌲 Decision Tree Feature Importance")
    
    tree_fi = pd.DataFrame({
        "Feature": ["Feature 1 (X)", "Feature 2 (Y)"],
        "Importance": model.feature_importances_
    })
    fig_dt = px.bar(tree_fi, x="Feature", y="Importance", color="Importance", color_continuous_scale="Purples")
    fig_dt.update_layout(**get_plotly_layout("Gini Feature Importance", height=280))
    
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.plotly_chart(fig_dt, use_container_width=True)
    with col_t2:
        st.markdown(f"""
        <div style="background:{BG_CARD}; padding:16px; border:1px solid rgba(255,255,255,0.05); border-radius:8px; font-size:12.5px; height:100%;">
            <strong>🌲 Tree Splits Analysis:</strong><br>
            • Feature 1 splits account for <code>{model.feature_importances_[0]*100:.1f}%</code> of total impurity reduction.<br>
            • Feature 2 splits account for <code>{model.feature_importances_[1]*100:.1f}%</code> of total impurity reduction.<br><br>
            Adjusting the **Max Depth** slider reveals how deep splits create highly complex rectangular boundaries that quickly overfit to local noise.
        </div>
        """, unsafe_allow_html=True)

elif algo in ["Random Forest", "Extra Trees"]:
    st.markdown("### 🌲 Forest Variance Reduction: Individual Tree Boundaries")
    st.markdown("Both Random Forest and Extra Trees are ensembles. We can visualize the decision boundaries of individual tree estimators within the forest:")
    
    col_tr1, col_tr2, col_tr3 = st.columns(3)
    
    for idx, col in enumerate([col_tr1, col_tr2, col_tr3]):
        with col:
            tree_estimator = model.estimators_[idx]
            Z_tree = tree_estimator.predict(grid_points).reshape(xx.shape)
            
            fig_tr = go.Figure()
            fig_tr.add_trace(go.Contour(
                x=np.arange(x_min, x_max, h), y=np.arange(y_min, y_max, h), z=Z_tree,
                showscale=False, opacity=0.35, colorscale=[[0, '#ec4899'], [1, '#14b8a6']], hoverinfo='skip'
            ))
            fig_tr.add_trace(go.Scatter(
                x=X[y==0, 0], y=X[y==0, 1], mode='markers', showlegend=False,
                marker=dict(size=4, color='#ec4899', line=dict(width=0.5, color='#0f172a'))
            ))
            fig_tr.add_trace(go.Scatter(
                x=X[y==1, 0], y=X[y==1, 1], mode='markers', showlegend=False,
                marker=dict(size=4, color='#14b8a6', line=dict(width=0.5, color='#0f172a'))
            ))
            fig_tr.update_layout(**get_plotly_layout(f"Tree {idx+1} Split Boundary", height=240))
            fig_tr.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_tr, use_container_width=True)

elif algo in ["AdaBoost", "Gradient Boosting (GBDT)", "XGBoost Classifier", "LightGBM Classifier", "CatBoost Classifier"]:
    st.markdown("### ⚡ Boosting Ensemble Iterative Learning")
    st.markdown("Boosting models iteratively correct errors of previous estimators. This plot illustrates the relative feature importances calculated by the ensembled booster:")
    
    # Extract importances safely
    try:
        importances = model.feature_importances_
    except Exception:
        importances = [0.5, 0.5]
        
    booster_fi = pd.DataFrame({
        "Feature": ["Feature 1 (X)", "Feature 2 (Y)"],
        "Importance": importances
    })
    
    fig_bst = px.bar(booster_fi, x="Feature", y="Importance", color="Importance", color_continuous_scale="Viridis")
    fig_bst.update_layout(**get_plotly_layout("Booster Attributed Feature Importance", height=280))
    st.plotly_chart(fig_bst, use_container_width=True)

elif algo == "Bagging SVM Ensemble":
    st.markdown("### 👜 Bootstrap Aggregation (Bagging)")
    st.markdown("Bagging trains multiple estimators on bootstrap samples (random sub-samples with replacement). Here we show the estimators in the bag:")
    
    col_bg1, col_bg2 = st.columns(2)
    with col_bg1:
        st.metric("Estimators Trained", f"{len(model.estimators_)}")
    with col_bg2:
        st.metric("Bootstrap Sampling Ratio", f"{max_samples}%")
        
    st.markdown(f"""
    <div style="background:{BG_CARD}; padding:16px; border:1px solid rgba(255,255,255,0.05); border-radius:8px; font-size:12.5px;">
        💡 <strong>Bagging Strategy:</strong><br>
        By training <code>{len(model.estimators_)}</code> separate Support Vector Machines on random subsets 
        containing only <code>{max_samples}%</code> of the training samples, the ensemble stabilizes predictions. 
        The final boundary is an average vote of all estimators, smoothing the decision boundaries and reducing variance!
    </div>
    """, unsafe_allow_html=True)

elif algo == "Stacking Ensemble":
    st.markdown("### 📚 Stacking Meta-Learner Weightings")
    st.markdown("Stacking fits a second-stage meta-classifier to combine the predicted probabilities of the base estimators. Let's inspect how the meta-learner weighs base model outputs:")
    
    try:
        if hasattr(model.final_estimator_, "coef_"):
            meta_coefs = model.final_estimator_.coef_[0]
            base_names = ["Logistic Regression", "KNN", "Random Forest"]
            
            fig_stk = px.bar(x=base_names, y=meta_coefs, color=meta_coefs, color_continuous_scale="Teals")
            fig_stk.update_layout(**get_plotly_layout("Meta-Learner Coefficient Weights", height=280))
            
            col_sk1, col_sk2 = st.columns([2, 1])
            with col_sk1:
                st.plotly_chart(fig_stk, use_container_width=True)
            with col_sk2:
                st.markdown(f"""
                <div style="background:{BG_CARD}; padding:16px; border:1px solid rgba(255,255,255,0.05); border-radius:8px; font-size:12.5px; height:100%;">
                    📚 <strong>Stacking Meta-Weights:</strong><br>
                    • LR Weight: <code>{meta_coefs[0]:.4f}</code><br>
                    • KNN Weight: <code>{meta_coefs[1]:.4f}</code><br>
                    • RF Weight: <code>{meta_coefs[2]:.4f}</code><br><br>
                    This reflects how much trust the meta-learner places in each base model's predictions when negotiating conflict regions.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Meta-Learner weights are unavailable for tree final estimators. Tree final estimators utilize splitting hierarchies rather than coefficient weights.")
    except Exception as e:
        st.warning(f"Unable to extract meta-weights: {e}")

elif algo == "Multi-Layer Perceptron (MLP)":
    st.markdown("### 🧠 Neural Network Graph Visualization")
    st.markdown("This feedforward node-link diagram visualizes the active neural layers and connections of the Multi-Layer Perceptron model:")
    
    # Draw feedforward MLP architecture
    layers = [2] + list(hidden_units) + [1]
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    edge_x = []
    edge_y = []
    
    dx = 1.0 / (len(layers) - 1)
    
    for l_idx, num_neurons in enumerate(layers):
        x = l_idx * dx
        dy = 1.0 / (num_neurons + 1)
        for n_idx in range(num_neurons):
            y = (n_idx + 1) * dy
            node_x.append(x)
            node_y.append(y)
            
            if l_idx == 0:
                node_text.append(f"Input {n_idx+1}")
                node_color.append(TEAL)
            elif l_idx == len(layers) - 1:
                node_text.append("Output")
                node_color.append(PINK)
            else:
                node_text.append(f"Hidden L{l_idx} N{n_idx+1}")
                node_color.append(PURPLE)
                
    for l_idx in range(len(layers) - 1):
        num_src = layers[l_idx]
        num_dst = layers[l_idx + 1]
        
        dy_src = 1.0 / (num_src + 1)
        dy_dst = 1.0 / (num_dst + 1)
        
        x_src = l_idx * dx
        x_dst = (l_idx + 1) * dx
        
        for s in range(num_src):
            y_src = (s + 1) * dy_src
            for d in range(num_dst):
                y_dst = (d + 1) * dy_dst
                edge_x.extend([x_src, x_dst, None])
                edge_y.extend([y_src, y_dst, None])
                
    fig_mlp = go.Figure()
    fig_mlp.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='rgba(255,255,255,0.06)', width=1),
        hoverinfo='skip'
    ))
    fig_mlp.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        marker=dict(size=14, color=node_color, line=dict(color='#0e1117', width=2)),
        hoverinfo='text'
    ))
    fig_mlp.update_layout(
        **theme.get_plotly_layout(f"Feedforward Architecture (Activation: {activation})", height=320),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig_mlp, use_container_width=True)

# Navigation Footer
st.markdown("---")
col_prev, col_next = st.columns([1, 1])
with col_prev:
    st.markdown('<a href="/Feature_Selection" target="_self" style="text-decoration:none; color:#a78bfa; font-weight:600;">← Previous: Feature Selection</a>', unsafe_allow_html=True)
with col_next:
    st.markdown('<div style="text-align: right;"><a href="/Advanced_Models" target="_self" style="text-decoration:none; color:#a78bfa; font-weight:600;">Next: Advanced Models →</a></div>', unsafe_allow_html=True)


# ── Render Global Footer ──
from utils.theme import render_footer
render_footer()
