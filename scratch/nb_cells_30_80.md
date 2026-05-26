Cell 30 (markdown): --- ## 🧹 Section 4: Data Preprocessing
Cell 31 (markdown): ### 2.11 Causal Structure Discovery Understanding true relationships beyond correlation. 
Cell 32 (markdown): ## Flex 2: 🔍 Causal Discovery — Going Beyond Correlation While conventional ML pipelines focus on predictive association
Cell 33 (code): # pip install causal-learn from causallearn.search.ConstraintBased.PC import pc from causallearn.utils.GraphUtils import
Cell 34 (markdown): > [!TIP] > **Report flex:** Discuss how the causal DAG reveals that `mutual_matches` may be a **collider** variable (cau
Cell 35 (markdown): ### 2.12 Create Working Copy & Drop Redundant Columns
Cell 36 (code): df = df_raw.copy()  # Drop label/string versions of numeric columns (they add no new information) 
Cell 37 (markdown): ### 2.13 Create Binary Target Variable
Cell 38 (code): # Define positive outcome = any form of meaningful connection positive_outcomes = {'Mutual Match', 'Instant Match', 'Dat
Cell 39 (markdown): ### 2.14 Encode Ordinal Feature — income_bracket (7 levels → 3 tiers)
Cell 40 (code): print('income_bracket unique values:', df['income_bracket'].unique())  # Consolidate 7 granular levels into 3 interpreta
Cell 41 (markdown): ### 2.15 Encode Ordinal Feature — education_level (9 levels → 3 tiers)
Cell 42 (code): print('education_level unique values:', df['education_level'].unique())  # Note: CSV contains curly apostrophes (e.g. Ba
Cell 43 (markdown): ### 2.16 One-Hot Encode Nominal Categorical Features
Cell 44 (code): # These features have no natural order — use one-hot encoding nominal_cols = [     'gender', 
Cell 45 (markdown): ### 2.17 Multi-Hot Encode Interest Tags
Cell 46 (code): # Each user has 3 interests (comma-separated) — create binary columns per unique tag mlb = MultiLabelBinarizer() interes
Cell 47 (markdown): ### 4.6.1 V4 Advanced Feature Engineering Creating interaction terms, log transforms, and frequency encoding. 
Cell 48 (code): df['engagement_score'] = df['likes_received'] * df['swipe_right_ratio'] * df['message_sent_count'] df['profile_completen
Cell 49 (markdown): ### 2.18 Normalize Numerical Features with RobustScaler
Cell 50 (code): from sklearn.preprocessing import RobustScaler numeric_cols = [     'age', 'height_cm', 'weight_kg',     'app_usage_time
Cell 51 (markdown): ### 2.19 Final Preprocessed Dataset Overview
Cell 52 (code): print(f'Final dataset shape: {df.shape}') print(f'Total features: {df.shape[1] - 1}  |  Target column: target') print(f'
Cell 53 (markdown): --- ## 🎯 Section 5: Feature Selection
Cell 54 (markdown): ### 2.20 Prepare Feature Matrix & Target Vector
Cell 55 (code): X = df.drop(columns=['target']) y = df['target']  
Cell 56 (markdown): ### 2.21 ANOVA F-Score Feature Selection (SelectKBest)
Cell 57 (code): selector_f = SelectKBest(score_func=f_classif, k='all') selector_f.fit(X, y)  
Cell 58 (code): top25_f = f_scores.head(25)  plt.figure(figsize=(12, 8)) 
Cell 59 (markdown): ### 5.1.1 Boruta Feature Selection All-relevant feature selection. 
Cell 60 (code): import os, joblib os.makedirs('models_v4_cache', exist_ok=True) cache_file_boruta = 'models_v4_cache/boruta_support.jobl
Cell 61 (markdown): ### 2.22 Mutual Information Feature Selection
Cell 62 (code): mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)  mi_df = pd.DataFrame({ 
Cell 63 (code): top25_mi = mi_df.head(25)  plt.figure(figsize=(12, 8)) 
Cell 64 (markdown): ### 2.23 Select Final Feature Set
Cell 65 (code): # Keep union of top-40 features from both F-score and Mutual Information rankings top_f_features  = set(f_scores.head(40
Cell 66 (markdown): --- ## 📐 Section 6: Dimensionality Reduction — PCA
Cell 67 (markdown): ### 2.24 Explained Variance Analysis
Cell 68 (code): pca_full = PCA(random_state=RANDOM_STATE) pca_full.fit(X_selected)  
Cell 69 (markdown): ### 2.25 Apply PCA (retain 95% explained variance)
Cell 70 (code): # We keep BOTH feature sets to compare models with and without PCA N_COMPONENTS = n_components_95  
Cell 71 (markdown): ### 2.26 PCA Biplot — First Two Principal Components
Cell 72 (code): plt.figure(figsize=(9, 6)) sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(X_pca), size=3000, replace=False)
Cell 73 (markdown): --- ## ✂️ Section 7: Train / Test Split
Cell 74 (code): # --- Split on ORIGINAL selected features (primary — used for most models) --- X_train, X_test, y_train, y_test = train_
Cell 75 (code): # Visualise class balance in train and test sets fig, axes = plt.subplots(1, 2, figsize=(10, 4))  
Cell 76 (markdown): --- ## ✅ Section 8: Pre-Training Checklist  Confirm all preprocessing steps completed before model training:  | Step | D
Cell 77 (code): # Apply SMOTE to perfectly balance training set (50/50 split) natively in the pipeline from imblearn.over_sampling impor
Cell 78 (markdown): --- ## 🤖 Section 9: Model Training  We train **14 models** on the balanced selected features, then compare performance. 
Cell 79 (code): from sklearn.linear_model import LogisticRegression from sklearn.neighbors import KNeighborsClassifier from sklearn.tree