"""
Patch ML_dating_app_behaviour V6.ipynb → V7_Strict.ipynb

Applies the following structural fixes:
1. Moves `RobustScaler` from Sec 4.10 to Sec 5.1 (after train/test split).
2. Updates `SelectKBest`, `Boruta`, and `mutual_info_classif` to use X_train.
3. Updates PCA to use X_train_selected.
4. Caches `X_train_raw` before SMOTE application.
5. Updates `cross_val_score` to use `ImbPipeline` on `X_train_raw`.
6. Updates `RandomizedSearchCV` to use `ImbPipeline` on `X_train_raw`.
7. Adds dynamic missing-model computation to learning curve cell (KeyError XGBoost fix).
"""

import json
import shutil
import os
import re

# --- Paths ---
NOTEBOOK_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')
SRC = os.path.join(NOTEBOOK_DIR, 'ML_dating_app_behaviour V6.ipynb')
DST = os.path.join(NOTEBOOK_DIR, 'ML_dating_app_behaviour V7_Strict.ipynb')

def load_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  ✅ Saved: {path}")

def get_source_str(cell):
    return ''.join(cell.get('source', []))

def set_source(cell, new_source_str):
    lines = new_source_str.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            if line:
                result.append(line)
    cell['source'] = result

def find_cell_by_content(cells, search_str):
    for i, cell in enumerate(cells):
        src = get_source_str(cell)
        if search_str in src:
            return i
    return None

def replace_in_source(cell, old_str, new_str):
    src = get_source_str(cell)
    if old_str not in src:
        return False
    new_src = src.replace(old_str, new_str)
    set_source(cell, new_src)
    return True

def main():
    print("=" * 60)
    print("  Patching V6 → V7_Strict")
    print("=" * 60)

    shutil.copy2(SRC, DST)
    nb = load_notebook(DST)
    cells = nb['cells']
    fixes_applied = 0

    # 1. Remove Scaler from Sec 4.10
    idx = find_cell_by_content(cells, "scaler.fit_transform(df[numeric_cols])")
    if idx:
        old_src = get_source_str(cells[idx])
        new_src = old_src.replace("df[numeric_cols] = scaler.fit_transform(df[numeric_cols])", 
                                  "# df[numeric_cols] = scaler.fit_transform(df[numeric_cols])\nprint('Scaling deferred to Section 5.1 to prevent data leakage.')")
        set_source(cells[idx], new_src)
        fixes_applied += 1
        print(f"✅ 1. Removed global scaling from Cell {idx}")

    # 2. Insert Split & Scaler in Sec 5.1
    idx = find_cell_by_content(cells, "X = df.drop(columns=['target'])")
    if idx:
        new_src = """X = df.drop(columns=['target'])
y = df['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print(f'Feature matrix X: {X.shape}')
print(f'Target vector  y: {y.shape}')
print(f'\\nClass balance:')
print(y.value_counts().rename({0: 'Negative', 1: 'Positive'}))
print(f"\\n✅ Split data into Train ({X_train.shape}) and Test ({X_test.shape})")
print("✅ Scaled numeric columns strictly after splitting")"""
        set_source(cells[idx], new_src)
        fixes_applied += 1
        print(f"✅ 2. Injected Split and Strict Scaler to Cell {idx}")

    # 3. Update Feature Selection
    idx_f = find_cell_by_content(cells, "selector_f.fit(X, y)")
    if idx_f:
        replace_in_source(cells[idx_f], "selector_f.fit(X, y)", "selector_f.fit(X_train, y_train)")
        fixes_applied += 1
    idx_b = find_cell_by_content(cells, "feat_selector.fit(X.values, y.values)")
    if idx_b:
        replace_in_source(cells[idx_b], "feat_selector.fit(X.values, y.values)", "feat_selector.fit(X_train.values, y_train.values)")
        fixes_applied += 1
    idx_mi = find_cell_by_content(cells, "mi_scores = mutual_info_classif(X, y")
    if idx_mi:
        replace_in_source(cells[idx_mi], "mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)", "mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)")
        fixes_applied += 1
        
    idx_sel = find_cell_by_content(cells, "X_selected = X[selected_features]")
    if idx_sel:
        replace_in_source(cells[idx_sel], 
                          "X_selected = X[selected_features]\nprint(f'\\nX_selected shape: {X_selected.shape}')",
                          "X_train_selected = X_train[selected_features]\nX_test_selected = X_test[selected_features]\nprint(f'\\nX_train_selected shape: {X_train_selected.shape}')")
        fixes_applied += 1
        print("✅ 3. Updated Feature Selection to use X_train")

    # 4. Update PCA
    idx_pca1 = find_cell_by_content(cells, "pca_full.fit(X_selected)")
    if idx_pca1:
        replace_in_source(cells[idx_pca1], "pca_full.fit(X_selected)", "pca_full.fit(X_train_selected)")
        replace_in_source(cells[idx_pca1], "X_selected.shape", "X_train_selected.shape")
        fixes_applied += 1
    idx_pca2 = find_cell_by_content(cells, "X_pca = pca.fit_transform(X_selected)")
    if idx_pca2:
        replace_in_source(cells[idx_pca2], 
                          "X_pca = pca.fit_transform(X_selected)", 
                          "X_train_pca = pca.fit_transform(X_train_selected)\nX_test_pca = pca.transform(X_test_selected)")
        replace_in_source(cells[idx_pca2], "X_pca.shape", "X_train_pca.shape")
        fixes_applied += 1
        print("✅ 4. Updated PCA to use X_train_selected")

    # 5. Remove old Train/Test Split
    idx_split = find_cell_by_content(cells, "X_train, X_test, y_train, y_test = train_test_split(")
    if idx_split:
        # Check if it's the Cell 78
        if "X_selected, y" in get_source_str(cells[idx_split]):
            set_source(cells[idx_split], "print('Data already split in Section 5.1. PCA applied to X_train and X_test.')")
            fixes_applied += 1
            print(f"✅ 5. Removed duplicate global split from Cell {idx_split}")

    # 6. Cache X_train_raw
    idx_smote = find_cell_by_content(cells, "smote = SMOTE(random_state=RANDOM_STATE)")
    if idx_smote:
        replace_in_source(cells[idx_smote], 
                          "smote = SMOTE(random_state=RANDOM_STATE)",
                          "# Cache raw imbalanced data for strict Cross Validation\nX_train_raw = X_train.copy()\ny_train_raw = y_train.copy()\n\nsmote = SMOTE(random_state=RANDOM_STATE)")
        replace_in_source(cells[idx_smote],
                          "X_train, y_train = smote.fit_resample(X_train, y_train)",
                          "X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)")
        fixes_applied += 1
        print(f"✅ 6. Added X_train_raw caching before SMOTE (Cell {idx_smote})")

    # 7. Update cross_val_score
    idx_cv = find_cell_by_content(cells, "cv_scores = cross_val_score(model, X_train, y_train")
    if idx_cv:
        replace_in_source(cells[idx_cv],
                          "cv_scores = cross_val_score(model, X_train, y_train, cv=5,",
                          "from imblearn.pipeline import Pipeline as ImbPipeline\n    from imblearn.over_sampling import SMOTE\n    cv_pipeline = ImbPipeline([('smote', SMOTE(random_state=RANDOM_STATE)), ('clf', model)])\n    cv_scores = cross_val_score(cv_pipeline, X_train_raw, y_train_raw, cv=5,")
        fixes_applied += 1
        print("✅ 7. Updated cross_val_score to use ImbPipeline on X_train_raw")

    # 8. XGBoost KeyError
    idx_lc = find_cell_by_content(cells, "if os.path.exists(lc_checkpoint_path):")
    if idx_lc:
        old_lc_code = """if os.path.exists(lc_checkpoint_path):
    print(f'⚡ Loading computed learning curve data from {lc_checkpoint_path}  instantly!...')
    lc_data = joblib.load(lc_checkpoint_path)
else:"""
        new_lc_code = """if os.path.exists(lc_checkpoint_path):
    print(f'⚡ Loading computed learning curve data from {lc_checkpoint_path}  instantly!...')
    lc_data = joblib.load(lc_checkpoint_path)
    
    # DYNAMIC FALLBACK: Compute any missing curves for new models (like XGBoost)
    missing = [name for name in top3 if name not in lc_data]
    if missing:
        print(f'🔄 Computing missing learning curves for: {missing}')
        for name in missing:
            model = results[name]['model']
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train,
                train_sizes=np.linspace(0.1, 1.0, 8),
                cv=5, scoring='accuracy', n_jobs=1
            )
            lc_data[name] = {'train_sizes': train_sizes, 'train_scores': train_scores, 'val_scores': val_scores}
        joblib.dump(lc_data, lc_checkpoint_path)
else:"""
        replace_in_source(cells[idx_lc], old_lc_code, new_lc_code)
        fixes_applied += 1
        print("✅ 8. Added missing-model computation to learning curve cell (KeyError fix)")

    # 9. Update RandomizedSearchCV
    idx_rs = find_cell_by_content(cells, "search = RandomizedSearchCV(")
    if idx_rs:
        old_rs_code = """        search = RandomizedSearchCV(
            estimator=base_models[name],
            param_distributions=param_grids[name],
            n_iter=30,
            cv=5,
            scoring='f1',
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=1
        )

        start = time.time()
        search.fit(X_train, y_train)
        tune_time = time.time() - start

        best_model = search.best_estimator_"""
        new_rs_code = """        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        cv_pipeline = ImbPipeline([('smote', SMOTE(random_state=RANDOM_STATE)), ('clf', base_models[name])])
        grid = {'clf__' + k: v for k, v in param_grids[name].items()}

        search = RandomizedSearchCV(
            estimator=cv_pipeline,
            param_distributions=grid,
            n_iter=30,
            cv=5,
            scoring='f1',
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=1
        )

        start = time.time()
        search.fit(X_train_raw, y_train_raw)
        tune_time = time.time() - start

        best_pipeline = search.best_estimator_
        best_model = best_pipeline.named_steps['clf']"""
        replace_in_source(cells[idx_rs], old_rs_code, new_rs_code)
        fixes_applied += 1
        print("✅ 9. Updated RandomizedSearchCV to use ImbPipeline on X_train_raw")

    print("\n" + "=" * 60)
    print(f"  Results: {fixes_applied} patches applied")
    print("=" * 60)
    save_notebook(nb, DST)

if __name__ == '__main__':
    main()
