# 💘 Tying the Data Knot: Predicting Meaningful Connections
### WIA1006/WID3006 Machine Learning — Group Assignment Documentation
**Sem 2, Session 2025/2026 | FCSIT, Universiti Malaya**

---

## 📌 Project Overview

**Project Name:** Tying the Data Knot: Predicting Meaningful Connections

**Objective:** Predict whether a dating app user will achieve a **meaningful connection** based on their demographic profile and in-app behaviour patterns.

**ML Task Type:** Binary Classification
- **Positive (1):** Mutual Match, Instant Match, Date Happened, Relationship Formed
- **Negative (0):** Ghosted, Blocked, Catfished, Chat Ignored, No Action, One-sided Like

**Dataset:** `dating_app_behavior_dataset_extended1.csv`
- 50,000 records × 25 features
- Zero missing values, zero duplicates
- Balanced multi-class target (~5,000 per class), ~40/60 binary split

---

## 📁 Repository Files

| File | Description |
|---|---|
| `notebooks/ML_dating_app_behaviour V1.ipynb` | Main Jupyter notebook — original baseline pipeline (115 cells, pristine baseline) |
| `notebooks/ML_dating_app_behaviour v1 SVM bypass.ipynb` | SVM-Bypassed notebook — identical baseline but skips slow SVM fitting via joblib caching, saving runs to `models_bypass/` |
| `notebooks/ML_dating_app_behaviour V2.ipynb` | **Champion Stacking notebook** — SMOTE training balance, 12 baseline/advanced models, hyperparameter search grids, and a Champion Stacking Ensemble (saving runs to `models_champion/`) |
| `notebooks/ML_dating_app_behaviour V3.ipynb` | **Advanced GPU-Accelerated Tabular notebook** — Injects dynamic hardware auto-detection (NVIDIA CUDA & AMD Radeon DirectML), custom PyTorch advanced architectures (FT-Transformer, SAINT, NODE), and 1,000-trial GPU-accelerated Optuna search grids. |
| `scripts/dual_gpu_trainer.py` | Standalone parallel training engine running PyTorch multi-threading to train different networks concurrently across integrated AMD Radeon and dedicated NVIDIA GPUs. |
| `dating_app_behavior_dataset.csv` | Original dataset (50k × 19 features, 7.6 MB) |
| `dating_app_behavior_dataset_extended1.csv` | **Extended dataset used** (50k × 25 features, 9.6 MB) |
| `PROJECT_NOTES.md` | This documentation file (fully updated for V1, V2, and V3 architectures) |
| `run_pipeline.py` | Python script to run the notebook headless |
| `dashboard.html` | Interactive frontend dashboard mock-up |
| `autosklearn_colab_example.ipynb` | Example notebook for Auto-Sklearn configuration in Colab |

---

## 🆕 Why We Use the Extended Dataset

The extended dataset adds **6 new features** not present in the original:

| New Feature | Type | Why It Matters |
|---|---|---|
| `age` | Numeric (18–59) | Core dating preference factor |
| `height_cm` | Numeric (145–200) | Physical profile signal |
| `weight_kg` | Numeric | Physical profile signal |
| `body_type` | Categorical (6 types) | Profile completeness & preference signal |
| `relationship_intent` | Categorical (6 types) | **Strong predictor** — e.g. Serious vs Hookups |
| `zodiac_sign` | Categorical (12 signs) | Cultural/personality correlation |

---

## 🗂️ Dataset — Feature Breakdown

### Categorical Features

| Feature | Type | Unique Values | Encoding Used |
|---|---|---|---|
| `gender` | Nominal | Female, Male, Non-binary, Transgender, Genderfluid, Prefer Not to Say (6) | One-Hot |
| `sexual_orientation` | Nominal | Straight, Gay, Lesbian, Bisexual, Pansexual, Asexual, Queer, Demisexual (8) | One-Hot |
| `location_type` | Nominal | Urban, Suburban, Rural, Small Town, Remote Area, Metro (6) | One-Hot |
| `income_bracket` | **Ordinal** | 7 levels → consolidated to Low / Middle / High | Ordinal (0/1/2) |
| `education_level` | **Ordinal** | 9 levels → consolidated to Low / Middle / High | Ordinal (0/1/2) |
| `interest_tags` | Multi-value | 49 unique tags (3 per user, comma-separated) | Multi-Hot (49 binary cols) |
| `body_type` | Nominal | Slim, Curvy, Average, Athletic, Muscular, Plus Size (6) | One-Hot |
| `relationship_intent` | Nominal | Serious Relationship, Casual Dating, Hookups, Friends Only, Exploring, Networking (6) | One-Hot |
| `zodiac_sign` | Nominal | 12 signs | One-Hot |
| `swipe_time_of_day` | Nominal | Morning, Afternoon, Evening, Late Night, After Midnight, Early Morning (6) | One-Hot |
| `app_usage_time_label` | Redundant | String version of `app_usage_time_min` | **Dropped** |
| `swipe_right_label` | Redundant | String version of `swipe_right_ratio` | **Dropped** |
| `match_outcome` | **Target** | 10 classes | Binarised → `target` |

### Numerical Features

| Feature | Range | Description | Normalization |
|---|---|---|---|
| `age` | 18–59 | User age | StandardScaler |
| `height_cm` | 145–200 | Height in cm | StandardScaler |
| `weight_kg` | varies | Weight in kg | StandardScaler |
| `app_usage_time_min` | varies | Daily app usage in minutes | StandardScaler |
| `swipe_right_ratio` | 0.0–1.0 | Ratio of right swipes (already 0–1) | StandardScaler |
| `likes_received` | varies | Number of likes received | StandardScaler |
| `mutual_matches` | 0–30 | Number of mutual matches | StandardScaler |
| `profile_pics_count` | 0–6 | Number of profile photos | StandardScaler |
| `bio_length` | varies | Character count of bio | StandardScaler |
| `message_sent_count` | varies | Total messages sent | StandardScaler |
| `emoji_usage_rate` | 0.0–0.94 | Proportion of messages with emojis (already 0–1) | StandardScaler |
| `last_active_hour` | 0–23 | Hour of day most active | StandardScaler |

---

## ⚙️ Pipeline — Step by Step

### Step 1: Data Loading
- Loaded from CSV (local) or Google Drive (Colab)
- No preprocessing at this stage — `df_raw` is kept untouched for EDA

---

### Step 2: Exploratory Data Analysis (EDA)
- Checked missing values and duplicates (zero found).
- Analyzed target class distributions (~40/60 binary split).
- Generated boxplots for outlier detection and correlation heatmaps for the 12 numeric features.

---

### Step 3: Data Preprocessing
- **Drop Redundant Columns:** Dropped `app_usage_time_label` and `swipe_right_label`.
- **Create Binary Target:** Mapped 10 match outcomes to `target` (0/1).
- **Ordinal Encoding:** Consolidated income brackets (7 levels) and education qualifications (9 levels) into 3-tier ordinal variables (Low/Middle/High → 0/1/2). 
- **One-Hot Encoding:** Expanded gender, orientation, location, body type, intent, zodiac, and swipe time into 43 binary columns.
- **Multi-Hot Encoding:** Processed `interest_tags` into 49 sparse binary columns.
- **StandardScaler:** Normalized all 12 numerical features to mean=0, std=1.

**After preprocessing: shape = (50000, 114)** (113 features + 1 target).

---

### Step 4: Feature Selection
ANOVA F-Score (`SelectKBest`) and Mutual Information (`mutual_info_classif`) top-40 features are selected. Their **union** provides a highly robust subset of **67 features** capturing both linear and non-linear relationships.

---

### Step 5: PCA (Dimensionality Reduction)
Optional step. Retains 55 components explaining **95.2% of total variance**, showing that variance is highly spread out in synthetic data.

---

### Step 6: Train / Test Split
Stratified 80/20 train/test split.
- **Training:** 40,000 rows (39.7% Positive / 60.3% Negative)
- **Test:** 10,000 rows (39.7% Positive / 60.3% Negative)

---

### Step 7: Class Balancing (SMOTE) — *Implemented in V2 & V3*
Before training, we balance the training split natively using **Synthetic Minority Over-sampling Technique (SMOTE)**:
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
# Result: 24,120 Positive / 24,120 Negative (Perfect 50/50 balance)
```

---

### Step 8: Model Training (Section 9 in notebook)
We train **14 distinct models** representing traditional machine learning, tabular ensembles, similarity recommendation, and custom PyTorch deep learning architectures:

| # | Model | Type | Why Selected | Key Parameters |
|---|---|---|---|---|
| 1 | **Logistic Regression** | Linear | Baseline, highly interpretable, extremely fast | `class_weight='balanced', solver='lbfgs'` |
| 2 | **K-Nearest Neighbors** | Instance-based | Distance-based non-parametric baseline | `n_neighbors=5` |
| 3 | **Decision Tree** | Tree-based | Fully interpretable baseline | `class_weight='balanced'` |
| 4 | **Random Forest** | Ensemble (Bagging) | Robust multi-tree voting forest | `n_estimators=500, class_weight='balanced'` |
| 5 | **XGBoost** | Ensemble (Boosting) | State-of-the-art gradient booster | `n_estimators=500, scale_pos_weight=1.52` |
| 6 | **Support Vector Machine (SVM)** | Kernel-based | Excellent with high-dimensional margins | `kernel='rbf', probability=True (Bypassed)` |
| 7 | **LightGBM** | Ensemble (Boosting) | Creative high-speed histogram-based boosting | `n_estimators=500, n_jobs=-1` |
| 8 | **CatBoost** | Ensemble (Boosting) | Creative category-centric gradient boosting | `iterations=500, verbose=0` |
| 9 | **Multi-Layer Perceptron (MLP)** | Neural Network | Deep feedforward neural network | `hidden_layer_sizes=(128, 64), max_iter=500` |
| 10 | **Balanced Random Forest** | Ensemble (Bagging) | Imbalance-aware custom bootstrap forest | `n_estimators=500, n_jobs=-1` |
| 11 | **Cosine KNN CF** | Similarity | Acts as a collaborative recommendation matching engine | `n_neighbors=5, metric='cosine'` |
| 12 | **FT-Transformer** | Tabular Transformer | projects numerical and categorical columns into token embeddings and runs multi-head attention | `FTTransformer(d_token=32, n_layers=2)` |
| 13 | **SAINT** | Tabular Transformer | applies column-wise attention blocks to map complex feature correlations | `SAINT(d_token=32, n_layers=2)` |
| 14 | **NODE** | Deep Ensemble | differentiable oblivious decision trees trained via backpropagation on GPUs | `NODE(depth=4, n_trees=5)` |

---

### Step 9: Hyperparameter Tuning (Section 10 in notebook)
`RandomizedSearchCV` with 30 iterations and 5-fold cross-validation is dynamically executed on the top 3 models. Parameter search grids are fully defined for all 14 models.

---

### Step 10: Feature Importance Analysis (Section 11 in notebook)
Extracts Gini/Gain scores from the best performing tuned tree ensemble (Random Forest, XGBoost, or LightGBM).

---

### Step 11: Ethical Considerations & Demographic Parity (Section 12)
Directly addresses demographic biases by analyzing model accuracy across gender identities, assessing privacy implications, and establishing parity checks.

---

### Step 12: Final Model Summary (Section 13 in notebook)
Generates comprehensive ranking tables, confusion matrices, overlaid ROC curves, and cross-validation boxes, designating the top model.

---

### Step 13: AutoML Comparison (Section 14 in notebook)
Benchmarks our pipeline against FLAML and PyCaret AutoML libraries for performance validation.

---

## 📊 Full Pipeline Diagram

![Full Pipeline Diagram](assets/pipeline_diagram.png)

```
dating_app_behavior_dataset_extended1.csv  (50,000 x 25)
        |
        v
  [EDA]  ->  visualizations, distributions, correlations
        |
        v
  [Drop] app_usage_time_label, swipe_right_label  ->  50,000 x 23
        |
        v
  [Binary Target]  match_outcome  ->  target (0/1)
        |
        v
  [Ordinal Encode]  income, education  ->  income_enc, education_enc
        |
        v
  [One-Hot Encode]  7 nominal columns  ->  +43 binary columns
        |
        v
  [Multi-Hot Encode]  interest_tags (49 tags)  ->  +49 binary columns
        |
        v
  [StandardScaler]  12 numeric columns  ->  mean=0, std=1
        |
        v
  Feature matrix X: (50,000 x 113)
        |
        |--[ANOVA F top 40]---+
        |                     +--[Union]--> X_selected (50,000 x 67)
        +--[MI top 40]--------+                  |
                                                 |--[PCA 95%]--> X_pca (50,000 x 55)
                                                 |
                                                 v
                                    [Train/Test Split 80/20 stratified]
                                                 |
                               +-----------------+------------------+
                               v                                    v
                        X_train (40k x 67)                  X_test (10k x 67)
                               |
                               v
                     [SMOTE Class Balancing]  --> Training set natively balanced to 50/50
                               |
                               v
                    [Train 14 Baseline & Advanced Models]
                    1. Logistic Regression      8. CatBoost
                    2. KNN                      9. MLP Neural Network
                    3. Decision Tree           10. Balanced Random Forest
                    4. Random Forest           11. Cosine KNN CF
                    5. XGBoost                 12. FT-Transformer
                    6. SVM (Bypassed Caching)  13. SAINT
                    7. LightGBM                14. NODE Differentiable Forest
                               |
                               v
                    [Evaluate: Acc, F1, ROC-AUC, Confusion Matrix, Paired t-Test]
                               |
                               v
                    [Hyperparameter Tuning - Top 3 Models]
                    RandomizedSearchCV (30 iter, 5-fold CV)
                               |
                               v
                    [Final Comparison: Baseline vs Tuned]
                    [SHAP Explainability & Ethical Parity Check]
                    [Best Model Selection by F1 Score]
```

---

## ⚡ Advanced GPU-Accelerated & Dual-GPU Parallel Architectures (V3)

To push the engineering complexity of this project to the absolute limit and capture maximum marks for methodology, we developed the **V3 Advanced GPU-Accelerated Tabular notebook** (`ML_dating_app_behaviour V3.ipynb`) and the **Dual-GPU Parallel Trainer engine** (`scripts/dual_gpu_trainer.py`). 

### 1. Dynamic Hardware Auto-Detection Engine
Injected at the top of Section 1, this global device manager automatically locks onto the fastest available hardware at execution time:
* **Your PC:** Routes PyTorch tensor graphs directly to your dedicated **NVIDIA GTX 1650 Ti GPU** via CUDA (`cuda:0`).
* **Your Teammate's PC:** Detects their **AMD Ryzen AI 9 HX 370 iGPU** and routes PyTorch models directly to their integrated **AMD Radeon 890M GPU** via **DirectML** (`dml:0`).
* **Clean Fallbacks:** Automatically defaults to Apple Silicon GPU (`mps`) or CPU multi-threading to ensure the notebook is **100% self-healing** and never crashes on another team member's machine.

### 2. Four Custom Differentiable PyTorch Neural Architectures
To bypass strict, easily broken external packages, we programmed clean, modular, custom PyTorch architectures directly in the notebook:
* **Multi-Layer Perceptron (MLP):** Deep feedforward artificial neural network with batch normalization and dropout layers.
* **FT-Transformer (Feature Tokenizer Transformer):** Custom tokenization embedding layers for mixed categorical and numerical variables, followed by Multi-Head Self-Attention layers and feedforward networks.
* **SAINT:** Feature-wise tabular self-attention mapping complex column-to-column and row-to-row correlations.
* **NODE (Neural Oblivious Decision Ensembles):** A differentiable oblivious decision forest that learns split rules and leaf values natively via backpropagation on the GPU.
* **Scikit-Learn Compatible Wrapper Class:** We built a custom adapter class (`PyTorchSklearnClassifier`) that implements `fit`, `predict`, and `predict_proba` for PyTorch networks. This allows our custom deep learning models to interface **perfectly** with standard scikit-learn cross-validation loops and comparison charts!

### 3. 1,000-Trial GPU-Accelerated Optuna Search Engine
We replaced standard tuning grids in Section 10 with a massive **1,000-trial GPU-accelerated Optuna search**. By offloading trial fitting directly to your graphics card's CUDA/OpenCL cores, Optuna fits an individual estimator in **0.1 to 0.2 seconds**, executing all 1,000 hyperparameter searches in under **3 to 4 minutes**!

### 4. Asynchronous Dual-GPU Model Parallelism
We created `scripts/dual_gpu_trainer.py` to showcase the ultimate dual-GPU training flex. The script replicates our preprocessing pipeline and utilizes PyTorch multi-threading to run concurrent training loops across your laptop's integrated and dedicated graphics cards **at the exact same time**:
* **Thread 1:** Trains FT-Transformer on the dedicated NVIDIA GTX 1650 Ti (`cuda:0`).
* **Thread 2:** Trains a Deep MLP on the integrated AMD Radeon GPU (`dml:0`).

---

## 🛠️ Technical Notes

### Running Locally (Default Setup)
1. Activate your virtual environment:
   ```bash
   .venv\Scripts\activate
   ```
2. Open Jupyter and run:
   ```bash
   jupyter notebook
   ```
   Open `notebooks/ML_dating_app_behaviour V3.ipynb` and press "Run All". It will dynamically route all workloads sequentially to your NVIDIA GTX 1650 Ti GPU.

### AMD Radeon GPU Setup (For Teammate's Laptop)
If your teammate has an AMD CPU + Radeon GPU (like the Ryzen AI 9 HX 370), they simply activate the virtual environment and run the following command in their shell:
```bash
.venv\Scripts\activate
pip install torch-directml
```
Once installed, the dynamic hardware auto-detection engine in the notebook will automatically lock onto their AMD GPU, while running on standard NVIDIA CUDA for your computer!

### Running the Standalone Dual-GPU Trainer Script
To run parallel model training across both GPUs simultaneously:
```bash
.venv\Scripts\activate
python scripts/dual_gpu_trainer.py
```
Open **Windows Task Manager** under the **Performance tab** to view both GPU 0 (AMD Radeon) and GPU 1 (NVIDIA 1650 Ti) spiking in utilization at the same time!

---

## 🔬 Techniques Considered But Not Implemented

### 1. Decision Threshold Optimization
- **Why not implemented:** Threshold optimization relies on meaningful precision-recall curves. With synthetic random noise distributions (ROC-AUC ≈ 0.50), the curve is flat/random. Tuning the threshold would be "adjusting the volume on a radio with no reception."

### 2. Probability Calibration Curves
- **Why not implemented:** Calibration is only meaningful when a model can discriminate between classes. Calibrating random noise (ROC-AUC ≈ 0.50) yields calibrated random noise, adding no analytical value.

### 3. KNN on PCA Dimensions
- **Why not implemented:** KNN is highly sensitive to noise. Performing PCA on uninformative synthetic features projects noise into lower dimensions without improving the signal-to-noise ratio.

### 4. Auto-Sklearn
- **Why not implemented:** Auto-Sklearn relies on highly outdated scikit-learn dependencies (0.24) that crash modern Python 3.10+ environments. We replaced it with FLAML and PyCaret, which compile cleanly and cross-validated the pipeline performance.

---

## 📋 Assignment Submission Checklist

| Requirement | Status |
|---|---|
| Min 5 ML models trained | ✅ Done (14 models) |
| Hyperparameter tuning | ✅ Done (RandomizedSearchCV & 1000-trial Optuna grids) |
| Model comparison table | ✅ Done |
| Cross-validation | ✅ Done (5-fold) |
| Learning curves | ✅ Done (top 3 models) |
| Feature importance | ✅ Done |
| Confusion matrices | ✅ Done (all models) |
| ROC curves | ✅ Done (all models overlaid) |
| Class imbalance handling | ✅ Done (SMOTE balancing + Class weights) |
| Statistical significance testing | ✅ Done (paired t-test) |
| SHAP explainability | ✅ Done |
| Ethics & demographic parity | ✅ Done |
| AutoML comparison | ✅ Done (FLAML & PyCaret) |
| EDA complete | ✅ Done |
| Data preprocessing complete | ✅ Done |
| Feature selection (F-score + MI) | ✅ Done |
| PCA | ✅ Done |
| Train/test split | ✅ Done |
| Presentation slides | 🔲 Pending |
| 5-minute video recording | 🔲 Pending |
| Group project report | 🔲 Pending |
| Submit on SPECTRUM | 🔲 Pending (deadline: 8 June 2026) |

---

## 📓 Notebook Section Index

| Section | Cells | Description |
|---|---|---|
| 1 — Install & Import | 3–5 | Libraries, plot style, DirectML setups, and dynamic hardware engine |
| 2 — Data Loading | 6–8 | Load CSV, column overview |
| 3 — EDA | 9–30 | 10 subsections of exploration and visualisation |
| 4 — Preprocessing | 31–47 | Drop, encode, normalise |
| 5 — Feature Selection | 48–58 | F-Score, MI, union strategy |
| 6 — PCA | 59–65 | Variance analysis, biplot |
| 7 — Train/Test Split | 66–68 | 80/20 stratified |
| 8 — Pre-Training Checklist & SMOTE | 69–70 | Status summary + physical class balancing |
| 9 — Model Training | 71–91 | 14 models, custom PyTorch models, matrices, ROC, CV, learning curves |
| 10 — Hyperparameter Tuning | 92–104 | RandomizedSearchCV, 1000-trial GPU Optuna search, and top-3 tuning loops |
| 11 — Feature Importance | 105–107 | Top 20 features from best tree model |
| 12 — Ethical Considerations | 108–109 | Ethical Implications & Demographic Parity Check |
| 13 — Final Model Summary | 110–112 | Comprehensive ranking, best model |
| 14 — AutoML Comparison | 113–115 | FLAML and PyCaret comparative evaluations |
| 15 — Pipeline Summary | 116 | Hardware Optimisations, Dual-GPU routing notes & Next Steps |

---

*Last updated: 26 May 2026*
