### Paragraph 19 (Style: Normal) [KW: Executive Summary]

Executive Summary

---

### Paragraph 21 (Style: Normal) [KW: majority class]

Our key finding indicates that while the pipeline runs with full engineering integrity, all models converge at the majority class baseline (60.30% test accuracy, ROC-AUC ≈ 0.50). This result is a valuable scientific finding, mathematically demonstrating the absence of predictive signal within the programmatic dataset. Features like zodiac sign or swipe ratio carry no genuine correlation with connection success. Based on these results, we recommend that future dating algorithms focus on natural language bio analysis (via NLP/LLMs) and active behavioral cues (such as response latency and chat length) to capture the true, non-linear signals of human connections.

---

### Paragraph 73 (Style: List Paragraph) [KW: 1. Column Filtering]

1. Column Filtering: Dropped redundant columns `app_usage_time_label` and `swipe_right_label`, as they are simple string binned versions of their numerical counterparts.

---

### Paragraph 78 (Style: List Paragraph) [KW: StandardScaler]

6. Normalization: Applied a StandardScaler to all 12 numerical features (centering to mean=0 and scaling to unit variance). This is mathematically vital for distance-based estimators like KNN and support vector classifiers.

---

### Paragraph 113 (Style: Normal) [KW: 14 baseline]

The models were trained on 80% of the dataset (40,000 samples), which was natively balanced using SMOTE to a perfect 50/50 split (24,120 positive / 24,120 negative), and evaluated on a stratified 20% test split (10,000 samples). Performance metrics across all 14 baseline and advanced models are tabulated below:

---

### Paragraph 151 (Style: Normal) [KW: majority class]

A critical finding of our modeling pipeline is that no model beats the majority class baseline of 60.30% accuracy, and all ROC-AUC metrics hover around 0.50. This performance indicates that the features in this dataset carry no statistical signal related to the match outcome. Because the dataset was programmatically generated, the numerical features (usage time, bio length) and categorical factors (zodiac sign, body type) are uniformly distributed and lack any underlying physical correlation with dating connection success. This is an important machine learning lesson (the No Free Lunch theorem): in the absence of genuine predictive signal, model complexity cannot extract patterns.

---

### Paragraph 210 (Style: Normal) [KW: 14 baseline]

In this group project, we successfully implemented a robust, end-to-end Machine Learning pipeline to predict dating app connections using a 50,000-sample dataset. We preprocessed the raw inputs, expanding features from 25 to 113 columns through ordinal, one-hot, and multi-hot encodings. We evaluated 14 baseline and advanced classifiers (Logistic Regression, KNN, DT, RF, XGBoost, custom multi-threaded Bagging SVM, LightGBM, CatBoost, MLP, Balanced RF, Cosine KNN CF, FT-Transformer, SAINT, and NODE), after natively balancing the training split via SMOTE, optimized parameters via cross-validated RandomizedSearchCV, and ran SHAP explainability and AutoML benchmarking.

---

### Paragraph 211 (Style: Normal) [KW: majority class]

Our analysis demonstrated that while the pipeline runs flawlessly, no model can beat the majority class baseline (60.30% accuracy, ROC-AUC ≈ 0.50). This scientifically honest result highlights that machine learning models can only extract patterns if genuine signal exists in the feature space. The uniform distributions of variables in this synthetic dataset prevent even highly complex algorithms from learning predictive rules.

---

### Paragraph 215 (Style: Normal) [KW: References]

References

---
