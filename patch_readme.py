import re

with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update Title
content = content.replace('# 💘 Tying the Data Knot: Predicting Meaningful Connections',
                          '# 💘 Tying the Data Knot: Predicting Meaningful Connections (V8 Pipeline)')

# 2. Update Pipeline - Step by Step
new_pipeline = """## ⚙️ Pipeline — Step by Step

### Section 1: Environment Setup & Library Installation
- Configures the runtime environment, installs dependencies, and initiates the **Dynamic Hardware Auto-Detection Engine** (routing to CUDA or DirectML).

---

### Section 2: Data Loading & Schema Verification
- Loads the extended dataset (50,000 × 25 features) and verifies the schema.

---

### Section 3: Exploratory Data Analysis (EDA)
![Dataset Insights and Analysis](assets/NotebookLM/section%20overview/Dating_Dataset_Insights_and_Analysis.png)
- Extensive 10-part EDA including distributions, boxplot outlier detection, correlation heatmaps, and target analysis.

---

### Section 4: Data Preprocessing & Feature Engineering
- **Causal Structure Discovery:** Applies the PC Algorithm (using `kci` conditional independence test) to map the causal Directed Acyclic Graph (DAG).
- **Double Machine Learning (DML):** Calculates the Average Treatment Effect (ATE) of profile effort on matches, eliminating selection bias.
- **Feature Engineering:** Creates interaction terms (e.g., selectivity ratios) and performs log transformations.
- **Normalization:** Applies `RobustScaler` to numerical features *after* train-test splitting to prevent pre-split leakage.
- **OOD Rejection Guardrail:** Implements an unsupervised Isolation Forest to detect and reject anomalous inputs at inference time.

---

### Section 5: Feature Selection
![Feature Selection Infographic](assets/NotebookLM/section%20overview/Dating_Success_Feature_Selection_Infographic.png)
- **ANOVA F-Score & Mutual Information:** Selects the top robust features based on linear and non-linear dependencies.
- **Boruta Selection:** Applies Boruta all-relevant feature selection via a Random Forest backbone to extract a robust feature subset.

---

### Section 6: Dimensionality Reduction — PCA
- Retains 95% explained variance, strictly used for visualization and explicitly benchmarked to prove inferiority against raw feature trees.

---

### Section 7: Train / Test Split & Class Resampling
- Stratified 80/20 train/test split.
- Training set balanced using Synthetic Minority Over-sampling Technique (SMOTE).

---

### Section 8: Pre-Training Checklist
- Verifies shapes and variables before launching the training pipelines to guarantee mathematical stability.

---

### Section 9: Model Training & Baseline Benchmarking
![Model Training Pipeline Overview](assets/NotebookLM/section%20overview/Model_Training_Pipeline_Overview.png)
- Establishes FLAML and PyCaret AutoML baselines.
- Trains **16 models** including Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, SVM (16-thread bagging), LightGBM, CatBoost, Balanced RF, and Cosine KNN CF.
- Also integrates PyTorch architectures: MLP, FT-Transformer, SAINT, and NODE.

---

### Section 10: Model Evaluation & Performance Comparisons
- **Label Smoothing & Mixup Regularization:** Empirically proves regularization smoothing using real live PyTorch training loss curves.
- **Cross-Validation & Significance:** Repeated cross-validation evaluated on strictly un-SMOTEd data, analyzing statistical stability using the Friedman Test and Nemenyi post-hoc analysis.

---

### Section 11: Privacy, Representation & Advanced Architectures
- **Opacus Differential Privacy:** Trains a PyTorch network with a strict industry-standard (ε=2.0) privacy budget.
- **Graph Neural Network (GNN):** Applies a Graph Attention Network (GAT) for semi-supervised transductive user matchmaking.
- **Attentive Tabular Network:** Instance-wise feature selection using a custom soft-mask attention sequential head.
- **Self-Supervised SCARF:** Contrastive pre-training embeddings.
- **TabPFN:** Zero-shot prior-data fitted network evaluation, computed strictly without metric dilution.

---

### Section 12: Hyperparameter Optimization
![Efficient Hyperparameter Optimization Strategy](assets/NotebookLM/section%20overview/Efficient_Hyperparameter_Optimization_Strategy.png)
- GPU-accelerated Optuna multi-objective tuning balancing F1 and Demographic Parity, fully cached.

---

### Section 13: Feature Importance & Ethical Considerations
- Evaluates Demographic Parity, Privacy Implications, and Homogeneity Risk.

---

### Section 14: Feature Importance & Interaction Analysis
- Extracts global attribution scores and computes **Friedman's H-Statistic** for pairwise interactions.
- Computes **SHAP Interaction Values** to map local synergy attributions.

---

### Section 15: Advanced Model Robustness & Uncertainty
![Trustworthy AI Robustness Framework](assets/NotebookLM/section%20overview/Trustworthy_AI_Dating_Robustness_Framework.png)
- **Conformal Prediction:** Strict calibration without test-set leakage, establishing 95% coverage uncertainty sets.
- **Bayesian Uncertainty:** Monte Carlo Dropout.
- **Adversarial Robustness:** FGSM attacks structurally masked to mutate only logical, continuous features.
- **Isotonic Calibration:** Calibrates confidence scores and plots Reliability Diagrams.

---

### Section 16: Model Compression, Recourse & Deployment
![AI Efficiency and Agency](assets/NotebookLM/section%20overview/Dating_AI_Efficiency_and_Agency.png)
- **Knowledge Distillation:** Student training optimized using mini-batching.
- **Algorithmic Recourse (DiCE):** Actionable counterfactuals constrained strictly to mutable user features.
- **Causal Uplift Modeling (T-Learner):** Propensity score matching with Inverse Probability Weighting (IPW) applied to extract purely causal persuadable segments.

---

### Section 17: Final Pipeline Summary & Hardware Optimisations
- Consolidates the **Dynamic Champion Model** inheriting weights to all downstream components.
- Outlines the `models_v8/` dynamic checkpoint caching layer."""

content = re.sub(r'## ⚙️ Pipeline — Step by Step.*?## 📊 Full Pipeline Diagram', new_pipeline + '\n\n## 📊 Full Pipeline Diagram', content, flags=re.DOTALL)

# 3. Update Notebook Section Index table at the bottom
new_index = """## 📓 Notebook Section Index

| Section | Description |
|---|---|
| 1 — Install & Import | Libraries, plot style, DirectML setups, and dynamic hardware engine |
| 2 — Data Loading | Load CSV, column overview, schema verification |
| 3 — EDA | 10 subsections of exploration and visualisation |
| 4 — Preprocessing & Engineering | Causal Discovery, DML, robust scaling (post-split), OOD Rejection Guardrail |
| 5 — Feature Selection | ANOVA, Boruta, Mutual Information |
| 6 — PCA | Variance analysis, biplot benchmarking |
| 7 — Train/Test Split | 80/20 stratified, SMOTE |
| 8 — Pre-Training Checklist | Shape verification & pipeline preparation |
| 9 — Model Training | 16 baseline and advanced deep architectures |
| 10 — Evaluation | Label Smoothing, Mixup, CV, Friedman stats |
| 11 — Privacy & Advanced Architectures | Opacus DP, GNN, Attentive Tabular Network, SCARF, TabPFN |
| 12 — Hyperparameter Optimization | GPU Optuna Multi-Objective Pareto search |
| 13 — Ethics & AutoML | Demographic Parity Check, FLAML & PyCaret |
| 14 — Feature Importance | H-Statistic pairwise interactions, SHAP Joint Interaction Curves |
| 15 — Robustness & Uncertainty | Conformal Prediction, MC Dropout, FGSM, Isotonic Calibration |
| 16 — Compression & Deployment | Knowledge Distillation, Microsoft DiCE, T-Learner Causal Uplift |
| 17 — Summary & Hardware | Final Dynamic Champion selection & caching strategy |"""

content = re.sub(r'## 📓 Notebook Section Index.*?\n---', new_index + '\n\n---', content, flags=re.DOTALL)

# 4. Update the Technical Notes section to replace models_v5 and models_v4_cache with models_v8 logic
content = content.replace('models_advanced/', 'models_v8/')
content = content.replace('models_v4_cache/', 'models_v8/')
content = content.replace('models_v5/', 'models_v8/')
content = content.replace('V3 Baseline Caching (`models_advanced/`)', 'V8 Baseline Caching (`models_v8/`)')
content = content.replace('V4 Deep Computation Caching (`models_v8_cache/`)', 'Deep Computation Caching (`models_v8/`)')
content = content.replace('V5 Caching and SCARF Flow Optimizations (`models_v8/`)', 'V8 Caching and Logic Flow Optimizations (`models_v8/`)')

# Add 16-thread SVM to Technical Notes under hardware
hw_opt = """### ⚡ Hardware Acceleration & Speed Optimisations:

* **16-Thread SVM Bagging Ensemble:** Upgraded standard single-threaded SVM to a parallelized **16-estimator Bagging Classifier**. This leverages 16GB of system RAM cache in parallel, slashing baseline training times from 40 minutes down to less than 20 seconds!
* **Dynamic GPU Auto-Detection:** Routes PyTorch to CUDA/DirectML/MPS automatically.
* **Max-RAM Tree Scaling:** Baseline and grid search parameters for Random Forest and XGBoost scaled up to 1000 trees and depth 12.
"""

content = content.replace('### AMD Radeon GPU Setup (For Teammate\'s Laptop)', hw_opt + '\n### AMD Radeon GPU Setup (For Teammate\'s Laptop)')

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(content)
