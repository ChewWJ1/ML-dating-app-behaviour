import os

notes_path = 'PROJECT_NOTES.md'
with open(notes_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update Step 2: EDA & Causal Discovery
old_step2 = """### Step 2: Exploratory Data Analysis (EDA)
- Checked missing values and duplicates (zero found).
- Analyzed target class distributions (~40/60 binary split).
- Generated boxplots for outlier detection and correlation heatmaps.
- **[V4] Data Quality Audit**: Performed Mutual Information analysis and Permutation Testing to quantify inherent dataset learnability.
- **[V4] Causal Structure Discovery**: Applied the PC Algorithm to infer a Directed Acyclic Graph (DAG) distinguishing causal relationships from mere correlations."""

new_step2 = """### Step 2: Exploratory Data Analysis (EDA)
- Checked missing values and duplicates (zero found).
- Analyzed target class distributions (~40/60 binary split).
- Generated boxplots for outlier detection and correlation heatmaps.
- **[V4] Data Quality Audit**: Performed Mutual Information analysis and Permutation Testing to quantify inherent dataset learnability.
- **[V4] Causal Structure Discovery**: Applied the PC Algorithm to infer a Directed Acyclic Graph (DAG) distinguishing causal relationships from mere correlations.
- **[V5.1] Double Machine Learning Causal Estimation**: Programmed a two-stage residual DML regression engine to isolate the true **Average Treatment Effect (ATE)** of user profile presentation quality on matchmaking outcomes, complete with bootstrap 95% confidence intervals and causal significance p-values."""

if old_step2 in content:
    content = content.replace(old_step2, new_step2)
    print("Updated Step 2 successfully.")
else:
    print("Error updating Step 2.")

# 2. Update Step 9: Advanced Model Training
old_step9 = """### Step 9: Advanced Model Training (Section 10 in notebook)
We train 15 distinct baseline models, similarity recommenders, PyTorch deep learning architectures, and zero-shot transformers.
- **[V4] Graph Neural Network (GNN):** Treats users as nodes in a similarity graph, applying a Graph Attention Network (GAT) for semi-supervised node classification.
- **[V4] Self-Supervised Contrastive Pre-Training (SCARF):** Extracts latent structure without labels via random feature corruption.
- **[V4] Differential Privacy Training:** Trained a PyTorch deep network with Opacus, achieving strict (ε=8.0, δ=1e-5)-differential privacy guarantees.
- **[V5] Zero-Shot Tabular Transformers (TabPFN):** Deployed a zero-shot prior-data fitted network pre-trained on millions of synthetic datasets, approximating the true Bayesian posterior in a single forward pass.
- **[V5] Label Smoothing & Mixup Regularization:** Integrated label smoothing (0.1/0.9 mapping) and Mixup input interpolation into our PyTorch wrapper's training loop to regularize deep neural models against overconfidence and noisy labels."""

new_step9 = """### Step 9: Advanced Model Training (Section 10 in notebook)
We train 16 distinct baseline models, similarity recommenders, PyTorch deep learning architectures, and zero-shot transformers.
- **[V4] Graph Neural Network (GNN):** Treats users as nodes in a similarity graph, applying a Graph Attention Network (GAT) for semi-supervised node classification.
- **[V4] Self-Supervised Contrastive Pre-Training (SCARF):** Extracts latent structure without labels via random feature corruption.
- **[V4] Differential Privacy Training:** Trained a PyTorch deep network with Opacus, achieving strict (ε=8.0, δ=1e-5)-differential privacy guarantees.
- **[V5] Zero-Shot Tabular Transformers (TabPFN):** Deployed a zero-shot prior-data fitted network pre-trained on millions of synthetic datasets, approximating the true Bayesian posterior in a single forward pass.
- **[V5] Label Smoothing & Mixup Regularization:** Integrated label smoothing (0.1/0.9 mapping) and Mixup input interpolation into our PyTorch wrapper's training loop to regularize deep neural models against overconfidence and noisy labels.
- **[V5.1] TabNet-style Attentive Neural Network**: Implemented a PyTorch Attentive Tabular Network that outputs dynamic, instance-wise feature selection masks, visualizing individual column targeting choices in an explainable selection heatmap."""

if old_step9 in content:
    content = content.replace(old_step9, new_step9)
    print("Updated Step 9 successfully.")
else:
    print("Error updating Step 9.")

# 3. Update Step 14: Ethical Considerations & Demographic Parity
old_step14 = """### Step 14: Ethical Considerations & Demographic Parity (Section 15 in notebook)
Analyzes model accuracy and bias across sensitive demographic attributes."""

new_step14 = """### Step 14: Ethical Considerations, Demographic Parity & Uplift Modeling (Section 15 & 17 in notebook)
Analyzes model accuracy and bias across sensitive demographic attributes.
- **[V5.1] Causal Uplift Modeling (T-Learner Meta-Classifier)**: Programmed Treatment ($M_1$) and Control ($M_0$) meta-learners to estimate the Individual Treatment Effect (ITE) of profile interventions, segmenting dating app users into *Persuadables*, *Sure Things*, *Lost Causes*, and *Sleeping Dogs* to enable targeted prescriptive premium feature recommendations."""

if old_step14 in content:
    content = content.replace(old_step14, new_step14)
    print("Updated Step 14 successfully.")
else:
    print("Error updating Step 14.")

# 4. Update Pipeline Diagram ASCII
old_diagram = """                     [Train 15 Baseline & Advanced Models]
                     + Custom PyTorch (FT-Transformer, SAINT, Deep MLP)
                     + [V4] Graph Neural Network (GNN Node Classification)
                     + [V4] SCARF Contrastive Pre-Training
                     + [V4] Opacus Differential Privacy Training
                     + [V5] TabPFN Zero-Shot Tabular Transformer
                     + [V5] Label Smoothing & Mixup regularization"""

new_diagram = """                     [Train 16 Baseline & Advanced Models]
                     + Custom PyTorch (FT-Transformer, SAINT, Deep MLP)
                     + [V4] Graph Neural Network (GNN Node Classification)
                     + [V4] SCARF Contrastive Pre-Training
                     + [V4] Opacus Differential Privacy Training
                     + [V5] TabPFN Zero-Shot Tabular Transformer
                     + [V5] Label Smoothing & Mixup regularization
                     + [V5.1] Custom Attentive Tabular Network (TabNet-style selection)"""

if old_diagram in content:
    content = content.replace(old_diagram, new_diagram)
    print("Updated diagram training models successfully.")
else:
    print("Error updating diagram training models.")

old_diagram2 = """  [Causal Discovery]  ->  PC Algorithm DAG (Causality vs Correlation)"""
new_diagram2 = """  [Causal Discovery]  ->  PC Algorithm DAG (Causality vs Correlation)
        |
        v
  [V5.1 Double Machine Learning]  ->  Average Treatment Effect (ATE) causal estimation"""

if old_diagram2 in content:
    content = content.replace(old_diagram2, new_diagram2)
    print("Updated diagram causal estimation successfully.")
else:
    print("Error updating diagram causal estimation.")

old_diagram3 = """                     [Model Compression, Recourse & Deployment]
                     + [V4] Knowledge Distillation (Complex Ensemble -> Logistic Student)
                     + [V5] Algorithmic Recourse Counterfactuals (Microsoft DiCE)"""

new_diagram3 = """                     [Model Compression, Recourse & Deployment]
                     + [V4] Knowledge Distillation (Complex Ensemble -> Logistic Student)
                     + [V5] Algorithmic Recourse Counterfactuals (Microsoft DiCE)
                     + [V5.1] Causal Uplift T-Learner (Persuadable Targeting Segmentor)"""

if old_diagram3 in content:
    content = content.replace(old_diagram3, new_diagram3)
    print("Updated diagram recourse & uplift successfully.")
else:
    print("Error updating diagram recourse & uplift.")

# 5. Update SOTA section
old_sota = """## 🌌 V5 "PhD-Level" Methodologies (The State-of-the-Art Edition)

For the V5 iteration, we integrated 6 cutting-edge methodologies focused on **Safe Deployment, Uncertainty Alignment, and Ethical Actionability**, elevating the engineering complexity of this pipeline to the standards of a research-grade ML system:"""

new_sota = """## 🌌 V5 "PhD-Level" Methodologies (The State-of-the-Art Edition)

For the V5 and V5.1 iterations, we integrated **9 cutting-edge methodologies** focused on **Causal Estimation, Attentive Deep Networks, Safe Deployment, Uncertainty Alignment, and Ethical Actionability**, elevating the engineering complexity of this pipeline to the standards of a research-grade ML system:

1. **Causal Inference via Double Machine Learning (DML):** We programmed a custom two-stage residual regression engine to calculate the Average Treatment Effect (ATE) of profile effort (photos count) on match outcomes. By regressing propensity-adjusted outcome residuals on propensity-adjusted treatment residuals, DML successfully controls for high-dimensional demographic and locational confounders, computing bootstrap 95% confidence intervals and causal significance p-values.
2. **Uplift Modeling (T-Learner Meta-Classifier):** Fitted Treatment ($M_1$) and Control ($M_0$) champion estimators to predict the Individual Treatment Effect (ITE) of profile interventions. Segmented users into *Persuadables (high target uplift)*, *Sure Things*, *Lost Causes*, and *Sleeping Dogs* to enable targeted prescriptive targeting algorithms.
3. **TabNet-style Attentive Tabular Network:** Coded a custom PyTorch tabular neural network featuring an `AttentiveTransformer` layer that outputs dynamic feature selection masks $M(X)$ via Softmax constraints, visualizing user-level column-wise neural attention in a heatmap.
4. **Out-of-Distribution (OOD) Rejection System (Isolation Forest):** We implemented an unsupervised Isolation Forest at the end of preprocessing. It isolates anomalies by random feature selection and binary splits. By monitoring path lengths, the system flags and rejects out-of-distribution profile configurations at inference time to prevent downstream predictive failure.
5. **Zero-Shot Tabular Transformers (TabPFN):** TabPFN is a prior-data fitted network pre-trained on millions of synthetic tabular datasets. It approximates the true Bayesian posterior in a single forward pass without requiring gradient descent or hyperparameter tuning. We deployed it using a subsampled prior support context of 1,000 training instances.
6. **Advanced Regularization (Label Smoothing & Mixup):** We modified our PyTorch Sklearn-compatible wrapper's `fit` loop. It applies label smoothing (mapping binary labels 0/1 to 0.1/0.9) to prevent model overconfidence, and Mixup input interpolation (convex combinations of feature pairs and smooth labels) to regularize decision boundaries on noisy dating app data.
7. **SHAP Joint Interaction Values:** Using the TreeExplainer framework, we computed the Shapley Interaction Index matrix for the champion tree ensemble. This splits feature contributions into main effects and joint pair attributions, mapping the precise mathematical synergy between top interacting features (e.g. `swipe_right_ratio` and `mutual_matches`).
8. **Probability Calibration & Reliability Diagrams:** We wrapped the champion model in Isotonic Regression to map raw confidence scores to empirical frequencies. We validated predictive uncertainty by plotting Calibration reliability curves and calculating Brier Score reductions.
9. **Algorithmic Recourse (Microsoft DiCE):** To enforce algorithmic agency, we deployed the DiCE framework. For a user predicted to be "Ghosted", DiCE uses randomized optimization to find the minimal actionable alterations (e.g., target changes in bio length or engagement metrics) required to flip the prediction to "Matched"."""

if old_sota in content:
    content = content.replace(old_sota, new_sota)
    print("Updated SOTA methodologies section successfully.")
else:
    print("Error updating SOTA section.")

# 6. Update Notebook Section Index
old_index = """| 4 — Preprocessing | Drop, encode, V4 features, robust scaling, **[V5] Isolation Forest OOD Rejection Guardrail** |"""
new_index = """| 4 — Preprocessing | Drop, encode, V4 features, robust scaling, **[V5] Isolation Forest OOD Rejection Guardrail** |
| 4.1 — [V5.1] Causal Inference | **[V5.1] Double Machine Learning Causal Estimation & Causal significance bootstrapping** |"""

if old_index in content:
    content = content.replace(old_index, new_index)
    print("Updated index preprocessing successfully.")
else:
    print("Error updating index preprocessing.")

old_index2 = """| 10 — Advanced Model Training | 15 baseline/PyTorch/zero-shot models, GNN node classification, SCARF contrastive learning, Differential Privacy, **[V5] Zero-Shot TabPFN**, **[V5] Label Smoothing & Mixup** |"""
new_index2 = """| 10 — Advanced Model Training | 16 baseline/PyTorch/zero-shot models, GNN node classification, SCARF contrastive learning, Differential Privacy, **[V5] Zero-Shot TabPFN**, **[V5] Label Smoothing & Mixup**, **[V5.1] Label Smoothing Loss Visualizer**, **[V5.1] TabNet-style Attentive Tabular Selection Network** |"""

if old_index2 in content:
    content = content.replace(old_index2, new_index2)
    print("Updated index training successfully.")
else:
    print("Error updating index training.")

old_index3 = """| 14 — Model Compression & Recourse | Knowledge Distillation, **[V5] Microsoft DiCE Actionable Recourse counterfactuals** |"""
new_index3 = """| 14 — Model Compression & Recourse | Knowledge Distillation, **[V5] Microsoft DiCE Actionable Recourse counterfactuals** |
| 17 — [V5.1] Causal Uplift | **[V5.1] Causal Uplift Modeling (T-Learner Meta-Classifier) & Causal segmentation targeting** |"""

if old_index3 in content:
    content = content.replace(old_index3, new_index3)
    print("Updated index recourse successfully.")
else:
    print("Error updating index recourse.")

with open(notes_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("PROJECT_NOTES.md updated successfully with V5.1 Extension!")
