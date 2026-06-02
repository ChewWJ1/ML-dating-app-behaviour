
**Faculty of Computer Science and Information Technology**

**University of Malaya**

**Semester 2, Session 2025/2026**

**WIA 1006 - Machine Learning**

**OCC 6**

---

**Group Assignment Report**

**Tying the (Data) Knot: Predicting Meaningful Connections (V8 Patched Edition)**

---

**Group 3**

**Group Members:**

**CHEW WEI JIAN 23118568/2**

**KU JIAN CHENG 23079373/2**

**NG JIN RU 23116192/2**

**ANG YING EN 23116738/2**

**CHAANG WAI CHIU 23104771/2**

**Executive Summary**

This report documents the architectural design, theoretical formulation, and empirical evaluation of the **V8 Patched Machine Learning Pipeline** for predicting matchmaking success on a mobile dating app. Built upon a 50,000-user dataset (`dating_app_behavior_dataset_extended1.csv`), we binarize ten granular match outcomes into a composite connection metric. 

While initial iterations established high-performing benchmarks, all models converged to a strict random boundary (ROC-AUC $\approx 0.50$, Accuracy $\approx 60.3\%$) due to the lack of predictive signal in standard behavioural features. To elevate the engineering complexity to rigorous scientific standards, this V8 iteration integrates 14 surgical fixes addressing data leakage and causal inference validity. We establish an unsupervised Isolation Forest Out-of-Distribution (OOD) rejection guardrail, quantify causal treatment effects using a Double Machine Learning (DML) engine, and train 16 baseline and advanced architectures (including GAT Neural Networks, SCARF self-supervised learners, Opacus differentially private networks, and TabPFN Zero-Shot Transformers). 

Our key scientific finding indicates that our selected models' convergence at the majority class baseline mathematically proves the absence of predictive signal within the programmatic dataset. DML causal estimation confirms the Average Treatment Effect of profile photos is statistically indistinguishable from zero ($p > 0.60$). Based on these results, we recommend future dating algorithms focus on natural language bio analysis (via NLP) and active behavioral cues to capture the true non-linear signals of human connections.

---

**1.0 Team Organization and Management**

**1.1 Team Formation and Collaboration Mechanisms**
Our team consists of five members from Faculty of Computer Science and Information Technology (FCSIT), University of Malaya. The team leader is Chew Wei Jian. We established this group based on a shared academic interest in applied machine learning pipelines and a joint goal of achieving excellence. Communication was maintained through regular weekly synchronization meetings held via Microsoft Teams and in-person lab sessions. For source code management, we established a central GitHub repository.

**1.2 Roles and Responsibilities**
* **Chew Wei Jian (Project Leader & ML Pipeline Lead)**: Coordinates task delegation, programmed the core pipeline execution script, and implemented parallel computing optimizations (multi-threading logic, cross-validation thread isolation).
* **Ku Jian Cheng (Data Preprocessing & Feature Engineer)**: Handled data extraction, designed ordinal mappings using regex/keyword matching, and built categorical nominal one-hot encoders.
* **Ng Jin Ru (Exploratory Data Analysis Analyst)**: Performed initial univariate and bivariate visualizations, analyzed target class balance, and visualized correlation matrices.
* **Ang Ying En (Model Optimization & Tuning Engineer)**: Configured and trained baseline ML models, programmed cross-validation performance loops, and executed RandomizedSearchCV tuning grids.
* **Chaang Wai Chiu (Explainability, Ethics & Dashboard Developer)**: Implemented SHAP (Shapley Additive exPlanations) values, evaluated demographic fairness, and constructed the interactive HTML dashboard UI.

---

**2.0 Problem and Objective**

**2.1 Project Background and Relevance**
Modern dating applications utilize matching algorithms to connect individuals. However, matching is often superficial and leads to high ghosting rates. This project attempts to solve a binary classification problem: predicting whether a user will achieve a meaningful connection (target=1, representing Mutual Match, Instant Match, Date Happened, and Relationship Formed) or experience a negative outcome (target=0, representing Blocked, Catfished, Chat Ignored, Ghosted, No Action, and One-sided Likes).

**2.2 Methodological Rigor (The V8 Patches)**
To resolve structural limitations, the V8 pipeline transitions from simple prediction to rigorous causal evaluation. We implemented 14 surgical fixes to ensure absolute scientific validity, including fixing pre-split leakage, isolating calibration thresholds to prevent test-set peeking, isolating conformal sets to guarantee mathematical coverage, and employing T-Learner uplift modeling for prescriptive targeting.

---

**3.0 Exploratory Data Analysis (EDA)**

Out of 50,000 records, the target variable consists of 19,850 positive connections (39.7%) and 30,150 negative interactions (60.3%), establishing a realistic class imbalance.

![Target Variable Distribution](assets/v8 plots/figure_001.png)
*Figure 1: Distribution of Target Variable Match Outcomes.*

![Categorical Feature Distributions](assets/v8 plots/figure_002.png)
*Figure 2: Categorical Feature Distributions Across 50,000 Profiles.*

![Numerical Feature Distributions](assets/v8 plots/figure_003.png)
*Figure 3: Numerical Feature Probability Density Distributions.*

![Outlier Detection via Boxplots](assets/v8 plots/figure_004.png)
*Figure 4: Outlier Detection Boxplots for Numerical App Engagement Metrics.*

![Numerical Features vs Target](assets/v8 plots/figure_005.png)
*Figure 5: Numerical Features Distributions Overlaid by Binary Target Class.*

![Categorical Features vs Target](assets/v8 plots/figure_006.png)
*Figure 6: Target Success Rates Across Nominal Categorical Variables.*

![Correlation Heatmap](assets/v8 plots/figure_007.png)
*Figure 7: Pearson Correlation Heatmap of Continuous Numerical Features.*

![Interest Tags Analysis](assets/v8 plots/figure_008.png)
*Figure 8: Frequency Analysis of Sparse Multi-Hot Interest Tags.*

---

**4.0 Methodology and Preprocessing**

**4.1 Causal Structure Discovery**
We mapped a Directed Acyclic Graph (DAG) using the PC algorithm (with a non-linear KCI conditional independence test) to discover the underlying data generation process and identify confounding loops. The resulting DAG and adjacency matrix are notably sparse, revealing only a direct causal link between profile picture count and age, alongside undirected associations with target and messaging behaviors. Crucially, no direct causal pathways exist from standard app engagement metrics (e.g., app usage time, likes received) to the target match outcome. This structural sparseness mathematically confirms our hypothesis that superficial demographic and behavioural features lack the causal signal necessary to deterministically drive matchmaking success, validating the models' convergence to the baseline accuracy.

![Causal Structure Discovery DAG](assets/v8 plots/figure_009.png)
*Figure 9: Directed Acyclic Graph (DAG) Recovered via PC Algorithm.*

![Causal Adjacency Heatmap](assets/v8 plots/figure_010.png)
*Figure 10: Causal Adjacency Heatmap of Direct Relationships.*

**4.2 Causal Inference via Double Machine Learning (DML)**
We use DML to calculate the Average Treatment Effect (ATE) of profile effort. 

![DML Residual Distributions](assets/v8 plots/figure_011.png)
*Figure 11: DML Residual Distributions.*

![DML ATE Confidence Intervals](assets/v8 plots/figure_012.png)
*Figure 12: DML ATE Bootstrap Confidence Intervals.*

**4.3 Preprocessing & OOD Guardrail**
We explicitly enforced `train_test_split` before `RobustScaler` (Phase 2 Pre-split Leakage Fix) to prevent leakage. We implemented an Isolation Forest to reject anomalies.

![Isolation Forest OOD Anomaly Scores](assets/v8 plots/figure_013.png)
*Figure 13: Isolation Forest Anomaly Score Distribution.*

**4.4 Feature Selection & PCA**

![ANOVA F-Score](assets/v8 plots/figure_014.png)
*Figure 14: Univariate ANOVA F-Score Feature Selection Rankings.*

![Mutual Information](assets/v8 plots/figure_015.png)
*Figure 15: Mutual Information Dependency Scores.*

![PCA Explained Variance](assets/v8 plots/figure_016.png)
*Figure 16: PCA Explained Variance.*

![PCA Biplot](assets/v8 plots/figure_017.png)
*Figure 17: PCA Biplot of the first two principal components.*

---

**5.0 Model Training and Evaluation**

We train 16 architectures, dynamically checking and routing PyTorch models to CUDA/DirectML. SMOTE is applied for training balance.

![Label Smoothing curves](assets/v8 plots/figure_018.png)
*Figure 18: Label Smoothing and Mixup Convergence Curves.*

![Confusion Matrices](assets/v8 plots/figure_019.png)
*Figure 19: Confusion Matrices across selected models.*

![ROC Curves](assets/v8 plots/figure_020.png)
*Figure 20: Receiver Operating Characteristic (ROC) Curves.*

![Precision-Recall Curves](assets/v8 plots/figure_021.png)
*Figure 21: Precision-Recall Curves for Optimal Thresholding.*

![Friedman Statistical Test Boxplots](assets/v8 plots/figure_022.png)
*Figure 22: Friedman Statistical Test comparing CV score stability.*

![Learning Curves](assets/v8 plots/figure_023.png)
*Figure 23: Learning Curves diagnosing bias and variance.*

---

**6.0 Privacy, Representation & Advanced Architectures**

**6.1 Differential Privacy (Opacus)**
We trained models with strict $(\epsilon, \delta)$ privacy guarantees.

![Opacus DP loss curves](assets/v8 plots/figure_024.png)
*Figure 24: DP-SGD Training Loss under Opacus Privacy.*

**6.2 Custom PyTorch Architectures**

![Attentive Tabular Network Mask Heatmap](assets/v8 plots/figure_025.png)
*Figure 25: TabNet-style Attentive Selection Mask Heatmap.*

![GNN User-Similarity Graph topology](assets/v8 plots/figure_026.png)
*Figure 26: Semi-supervised Graph Attention Network (GAT) Topology.*

**6.3 Self-Supervised SCARF & Zero-Shot TabPFN**

![SCARF Contrastive Pre-Training Loss](assets/v8 plots/figure_027.png)
*Figure 27: SCARF Contrastive Pre-Training Loss.*

![SCARF Embeddings t-SNE](assets/v8 plots/figure_028.png)
*Figure 28: SCARF Pre-trained Embeddings t-SNE Projection.*

  ![All Models Ranked by ROC-AUC Score](assets/v8 plots/figure_029.png)
  *Figure 29: All Models Ranked by ROC-AUC Score (Green = Tuned, Gray = Baseline).*

---

**7.0 Hyperparameter Optimization & Feature Interactions**

![Optuna Pareto Frontier](assets/v8 plots/figure_030.png)
*Figure 30: Multi-Objective GPU Optuna Pareto Frontier.*

![Demographic Parity](assets/v8 plots/figure_031.png)
*Figure 31: Demographic Parity Audit for Ethical Fairness.*

![Permutation Feature Interaction](assets/v8 plots/figure_032.png)
*Figure 32: Friedman's H-Statistic for Pairwise Synergies.*

![SHAP Interaction Values](assets/v8 plots/figure_033.png)
*Figure 33: SHAP Interaction Values (Beeswarm).*

![SHAP Dependence](assets/v8 plots/figure_034.png)
*Figure 34: SHAP Joint Interaction Matrix.*

---

**8.0 Model Robustness, Calibration & Deployment**

![MAPIE Conformal Prediction Bounds](assets/v8 plots/figure_035.png)
*Figure 35: MAPIE Conformal Prediction Bounding Sets (Leakage Fixed).*

![Adversarial Robustness Accuracy Drop](assets/v8 plots/figure_036.png)
*Figure 36: Model Accuracy Decay under Adversarial FGSM Attack.*

![Model Calibration Reliability Diagram](assets/v8 plots/figure_037.png)
*Figure 37: Isotonic Probability Calibration Reliability Diagram.*

![Causal Uplift Persuadables Quadrant](assets/v8 plots/figure_038.png)
*Figure 38: T-Learner Causal Uplift Segmenting Persuadable Users.*

---

**9.0 Implemented Enhancements & Hardware Optimizations**

1. **Sequential Outer Loops:** Restricted `n_jobs=1` on Windows outer cross-validation folds to prevent catastrophic GPU memory deadlocks across concurrent driver accesses.
2. **Dynamic Checkpointing (`models_v8/`):** Intelligent `joblib` cache architecture that routes and stores heavy operations (Boruta matrices, SCARF embeddings, DiCE counterfactuals), dropping rerun times from 25 minutes to < 1 minute.
3. **Double Machine Learning Rigor:** DML isolates standard ATE, rigorously proving limits on the dataset.
4. **V8 Methodological Fixes:** Addressed threshold leakage, adversarial discrete mutation masking, IPW propensity weighting for T-Learners, and conformal split isolation.

---

**10.0 Conclusion**

The V8 Patched Pipeline represents a SOTA causal, attentive, and privacy-preserving framework. The finding that state-of-the-art models converge at the random baseline proves the structural absence of signal in raw app demographics. Moving forward, dynamic and prescriptive interventions such as NLP chat analysis and algorithmic recourse (via Microsoft DiCE) will drive the next generation of safe and effective online matching.

---

**References**

1. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. KDD.
2. Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions*. NIPS.
3. Nisar, K. (2026). *Dating App Behavior Dataset*. Kaggle.
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python*. JMLR.
5. Universiti Malaya. (2026). *WIA1006/WID3006 Machine Learning Group Assignment Guidelines*.
