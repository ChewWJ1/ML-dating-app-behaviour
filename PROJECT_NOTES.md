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
| `ML_dating_app_behaviour.ipynb` | Main Jupyter notebook — full pipeline |
| `dating_app_behavior_dataset.csv` | Original dataset (50k × 19 features, 7.6 MB) |
| `dating_app_behavior_dataset_extended1.csv` | **Extended dataset used** (50k × 25 features, 9.6 MB) |
| `PROJECT_NOTES.md` | This documentation file |

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
```python
df_raw = pd.read_csv('dating_app_behavior_dataset_extended1.csv')
# Shape: (50000, 25)
```
- Loaded from CSV (local) or Google Drive (Colab)
- No preprocessing at this stage — `df_raw` is kept untouched for EDA

---

### Step 2: Exploratory Data Analysis (EDA)

**What was explored:**

1. **Basic Info** — `df.info()`, `df.describe()` for all 25 columns
2. **Missing values** — None found
3. **Duplicates** — None found
4. **Target distribution** — All 10 `match_outcome` classes are balanced (~5,000 each); binary split is 39.7% Positive / 60.3% Negative
5. **Categorical distributions** — Bar charts for all 9 categorical columns
6. **Numerical distributions** — Histograms for all 12 numeric columns
7. **Outlier detection** — Boxplots for all 12 numeric columns; no extreme outliers found
8. **Feature vs Target** — Overlaid histograms (numeric) and stacked % bars (categorical) split by Positive/Negative outcome
9. **Correlation heatmap** — Pearson correlation among all 12 numeric features
10. **Interest tag analysis** — Frequency chart of all 49 unique interest tags

**Key EDA findings:**
- All features are uniformly distributed — the dataset is synthetically generated and well-balanced
- No strong linear correlations between numeric features (expected for synthetic data)
- `relationship_intent` and some interest tags show slight variation in positive match rates
- No outlier removal needed — all numeric ranges are plausible

---

### Step 3: Data Preprocessing

#### 3.1 Drop Redundant Columns
```python
df.drop(columns=['app_usage_time_label', 'swipe_right_label'], inplace=True)
```
- `app_usage_time_label` is just a string category of `app_usage_time_min` → redundant
- `swipe_right_label` is just a string category of `swipe_right_ratio` → redundant

#### 3.2 Create Binary Target
```python
positive_outcomes = {'Mutual Match', 'Instant Match', 'Date Happened', 'Relationship Formed'}
df['target'] = df['match_outcome'].apply(lambda x: 1 if x in positive_outcomes else 0)
df.drop(columns=['match_outcome'], inplace=True)
```
- **Why binary?** Higher accuracy, clearer metrics (ROC-AUC, F1), easier to present
- **Positive class:** 4 outcomes representing actual human connection
- **Negative class:** 6 outcomes representing failed/one-sided interactions
- **Result:** 19,850 Positive (39.7%) / 30,150 Negative (60.3%)

#### 3.3 Ordinal Encoding — income_bracket
```python
income_map = {
    'Very Low': 'Low',   'Low': 'Low',
    'Lower-Middle': 'Middle', 'Middle': 'Middle', 'Upper-Middle': 'Middle',
    'High': 'High',      'Very High': 'High'
}
# Then: OrdinalEncoder(categories=[['Low', 'Middle', 'High']])
```
- **Why ordinal?** Income has a natural order (Low < Middle < High)
- Consolidates 7 granular levels → 3 meaningful tiers → encoded as 0, 1, 2

#### 3.4 Ordinal Encoding — education_level
```python
def map_education(val):
    if any(k in val for k in ['No Formal', 'High School', 'Diploma']): return 'Low'
    elif any(k in val for k in ['Associate', 'Bachelor']): return 'Middle'
    elif any(k in val for k in ['Master', 'MBA', 'PhD', 'Postdoc']): return 'High'
```
- **Why keyword matching?** The CSV uses curly apostrophes (`Bachelor's` → `Bachelor\u2019s`) which break exact string matching
- Consolidates 9 qualification levels → 3 academic tiers → encoded as 0, 1, 2

#### 3.5 One-Hot Encoding — Nominal Categoricals
```python
nominal_cols = ['gender', 'sexual_orientation', 'location_type',
                'swipe_time_of_day', 'body_type', 'relationship_intent', 'zodiac_sign']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=False, dtype=int)
```
- **Why one-hot?** These features have no natural order — all values are equally valid
- `drop_first=False` keeps all categories (avoids dummy variable trap only an issue for linear regression, which is not our primary model)
- Results in ~43 new binary columns

#### 3.6 Multi-Hot Encoding — interest_tags
```python
mlb = MultiLabelBinarizer()
interests_split = df['interest_tags'].str.split(', ')
interest_dummies = pd.DataFrame(mlb.fit_transform(interests_split),
                                columns=['interest_' + c for c in mlb.classes_])
```
- Each user has exactly 3 interests stored as a comma-separated string
- `MultiLabelBinarizer` creates 1 binary column per unique tag
- Results in **49 new binary columns** (one per unique interest)

#### 3.7 StandardScaler Normalization
```python
numeric_cols = ['age', 'height_cm', 'weight_kg', 'app_usage_time_min',
                'swipe_right_ratio', 'likes_received', 'mutual_matches',
                'profile_pics_count', 'bio_length', 'message_sent_count',
                'emoji_usage_rate', 'last_active_hour']
df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
```
- **Why StandardScaler?** Transforms each column to mean=0, std=1
- Essential for distance-based models (KNN, SVM) and gradient descent models (Logistic Regression)
- Tree-based models (Random Forest, XGBoost) don't require normalization but it doesn't hurt them

**After all preprocessing: shape = (50000, 114)** — 113 feature columns + 1 target column

---

### Step 4: Feature Selection

**Goal:** Reduce from 113 features to a smaller set of the most informative features, reducing noise and training time.

#### Method 1: ANOVA F-Score (SelectKBest)
```python
selector_f = SelectKBest(score_func=f_classif, k='all')
selector_f.fit(X, y)
```
- Tests whether the mean of each feature differs significantly between Positive and Negative classes
- Higher F-score = more statistically significant difference between classes
- All features scored; we rank them and take the **top 40**

#### Method 2: Mutual Information
```python
mi_scores = mutual_info_classif(X, y, random_state=42)
```
- Measures how much information each feature provides about the target
- Non-linear — captures relationships that F-score misses
- More robust for binary-encoded features (interest tags, one-hot columns)
- All features scored; take the **top 40**

#### Final Selection
```python
selected_features = sorted(set(f_scores.head(40)['feature']).union(set(mi_df.head(40)['feature'])))
# Result: 67 features selected
```
- **Union strategy:** Keep any feature that ranks highly in either method
- **Result: 67 features** selected from 113 total

> **Note:** Top-scoring features include numeric columns (`mutual_matches`, `likes_received`, `message_sent_count`) and relationship_intent one-hot columns, suggesting these are the strongest predictors.

---

### Step 5: PCA (Dimensionality Reduction)

**Goal:** Optionally reduce 67 features further by projecting into principal component space. Used as an alternative feature set to compare model performance.

```python
pca_full = PCA(random_state=42)
pca_full.fit(X_selected)
cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
n_components_95 = int(np.argmax(cumvar >= 95) + 1)  # = 55 components
```

**What PCA does:**
1. Computes the directions of maximum variance in the feature space (principal components)
2. Projects the data onto these directions
3. The first component captures the most variance, each subsequent component less

**Results:**
- **55 components** retain **95.2% of total variance**
- Reduces from 67 → 55 dimensions
- Components are linear combinations of original features — **interpretability is lost**

**Two feature sets maintained for modeling:**
| Variable | Shape | Description |
|---|---|---|
| `X_selected` | (50000, 67) | Original 67 selected features |
| `X_pca` | (50000, 55) | PCA-reduced to 55 components |

---

### Step 6: Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # ensures same class ratio in both splits
)
```

**Why stratify?**  
Without `stratify=y`, a random split might put more Positive examples in train than test, making evaluation unreliable. Stratification ensures both splits have the same ~39.7% / 60.3% ratio.

**Result:**

| Split | Rows | Positive | Negative |
|---|---|---|---|
| Training | 40,000 | ~15,880 (39.7%) | ~24,120 (60.3%) |
| Test | 10,000 | ~3,970 (39.7%) | ~6,030 (60.3%) |

**Available objects for model training:**
```python
X_train, X_test, y_train, y_test      # original 67 features
X_train_pca, X_test_pca               # PCA-reduced 55 components
RANDOM_STATE = 42                       # use in all models
```

---

## 🔜 What's Next — Model Training

The following models are planned (minimum 5 required by assignment):

| # | Model | Why Selected |
|---|---|---|
| 1 | **Logistic Regression** | Baseline linear model, interpretable, fast |
| 2 | **K-Nearest Neighbors (KNN)** | Non-parametric, distance-based |
| 3 | **Decision Tree** | Fully interpretable, no scaling needed |
| 4 | **Random Forest** | Ensemble of trees, robust, handles high dims |
| 5 | **XGBoost / Gradient Boosting** | Usually best performer on tabular data |
| 6 | **Support Vector Machine (SVM)** | Good for medium-sized datasets |

**Evaluation metrics to report:**
- Accuracy, Precision, Recall, F1-score (macro & weighted)
- ROC-AUC score
- Confusion matrix
- Learning curves (to detect overfitting)

**Hyperparameter tuning:**
- `GridSearchCV` or `RandomizedSearchCV` for top 2–3 models
- 5-fold cross-validation

**Auto-sklearn comparison:**
- Must be run in Google Colab (Linux-only dependency)
- 120-second budget for autoML search

---

## 🛠️ Technical Notes

### Running in Google Colab
1. Upload both CSV files to Google Drive under `MyDrive/Dataset/`
2. In Section 2 (Data Loading), uncomment the Drive mount lines and comment out the local path
3. Run all cells top to bottom — Section 1 installs all required packages

### Running Locally
1. Ensure CSV files are in the same directory as the notebook
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn`
3. The local path `DATA_PATH = 'dating_app_behavior_dataset_extended1.csv'` is already set

### Important: Education Level Encoding
The CSV stores values like `Bachelor's` with a curly apostrophe (`\u2019`), not a straight apostrophe (`'`). Direct dictionary mapping with `str.map()` would leave these as `NaN`. We use keyword-based matching instead:
```python
def map_education(val):
    val = str(val)
    if any(k in val for k in ['No Formal', 'High School', 'Diploma']): return 'Low'
    elif any(k in val for k in ['Associate', 'Bachelor']): return 'Middle'
    elif any(k in val for k in ['Master', 'MBA', 'PhD', 'Postdoc']): return 'High'
    return 'Low'
```

### RANDOM_STATE = 42
Used in all stochastic operations to ensure full reproducibility:
- `train_test_split`
- `PCA`
- `mutual_info_classif`
- All ML models

---

## 📊 Pipeline Summary Diagram

```
dating_app_behavior_dataset_extended1.csv  (50,000 × 25)
        │
        ▼
  [EDA]  →  visualizations, distributions, correlations
        │
        ▼
  [Drop] app_usage_time_label, swipe_right_label  →  50,000 × 23
        │
        ▼
  [Binary Target]  match_outcome  →  target (0/1)
        │
        ▼
  [Ordinal Encode]  income_bracket, education_level  →  income_enc, education_enc
        │
        ▼
  [One-Hot Encode]  7 nominal columns  →  +43 binary columns
        │
        ▼
  [Multi-Hot Encode]  interest_tags (49 tags)  →  +49 binary columns
        │
        ▼
  [StandardScaler]  12 numeric columns  →  mean=0, std=1
        │
        ▼
  Feature matrix X: (50,000 × 113)
        │
        ├──[ANOVA F-Score top 40]──┐
        │                          ├──[Union]──▶  X_selected (50,000 × 67)
        └──[Mutual Info top 40]────┘                    │
                                                        ├──[PCA 95% var]──▶  X_pca (50,000 × 55)
                                                        │
                                                        ▼
                                              [Train/Test Split 80/20 stratified]
                                                        │
                                    ┌───────────────────┴─────────────────────┐
                                    ▼                                         ▼
                             X_train (40,000 × 67)                  X_test (10,000 × 67)
                             y_train (40,000)                        y_test (10,000)
                             X_train_pca (40,000 × 55)               X_test_pca (10,000 × 55)

                                              ↓  [NEXT STEP]
                                          Model Training
```

---

## 📋 Assignment Submission Checklist

| Requirement | Status |
|---|---|
| Min 5 ML models trained | 🔲 Pending |
| Hyperparameter tuning | 🔲 Pending |
| Model comparison table | 🔲 Pending |
| Auto-sklearn comparison | 🔲 Pending (Colab, Linux) |
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

*Last updated: 23 May 2026*
